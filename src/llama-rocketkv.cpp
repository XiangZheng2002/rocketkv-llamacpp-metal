// RocketKV research implementation. See docs/rocketkv/NOTICE.
#include "llama-rocketkv.h"

#include "llama-context.h"
#include "llama-graph.h"
#include "llama-impl.h"
#include "llama-kv-cache.h"
#include "llama-model.h"
#include "ggml-rocketkv.h"

#include <algorithm>
#include <cmath>
#include <stdexcept>
#include <string>

static ggml_tensor * rocket_name(ggml_tensor * t, const char * component, int il) {
    ggml_format_name(t, "rocketkv_%s_%d", component, il);
    return t;
}

struct rocket_input : llm_graph_input_i {
    const llama_rocketkv & state;
    const int graph_phase;
    const uint32_t n_tokens;
    ggml_tensor * observe = nullptr;
    ggml_tensor * order = nullptr;
    ggml_tensor * mask = nullptr;
    ggml_tensor * live = nullptr;
    ggml_tensor * write = nullptr;

    rocket_input(const llama_rocketkv & state, const llama_ubatch & ubatch) :
        state(state), graph_phase(state.phase(ubatch)), n_tokens(ubatch.n_tokens) {}

    void set_input(const llama_ubatch * ubatch) override {
        if (observe) {
            const int count = observe->ne[0];
            std::vector<int64_t> ids(count);
            for (int i = 0; i < count; ++i) {
                ids[i] = ubatch->pos[ubatch->n_tokens - count + i] % state.params.observation_window;
            }
            ggml_backend_tensor_set(observe, ids.data(), 0, ggml_nbytes(observe));
        }
        if (order) {
            const int w = state.params.observation_window, n = state.params.prompt_tokens;
            std::vector<int32_t> ids(w);
            std::vector<ggml_fp16_t> causal(size_t(n)*w);
            for (int i = 0; i < w; ++i) {
                ids[i] = (n - w + i) % w;
                for (int t = 0; t < n; ++t) {
                    causal[i*n + t] = ggml_fp32_to_fp16(t <= n - w + i ? 0 : -INFINITY);
                }
            }
            ggml_backend_tensor_set(order, ids.data(), 0, ggml_nbytes(order));
            ggml_backend_tensor_set(mask, causal.data(), 0, ggml_nbytes(mask));
        }
        if (live) {
            const int32_t n = graph_phase == 2 ? state.info.prompt_kept + ubatch->pos[0] - state.params.prompt_tokens + 1 : state.info.prompt_kept;
            ggml_backend_tensor_set(live, &n, 0, sizeof(n));
            if (write) {
                const int64_t index = n - 1;
                ggml_backend_tensor_set(write, &index, 0, sizeof(index));
            }
        }
    }

    bool can_reuse(const llm_graph_params & next) override {
        return next.ubatch.n_tokens == n_tokens && state.phase(next.ubatch) == graph_phase;
    }
};

static ggml_tensor * rocket_input_tensor(ggml_context * ctx, ggml_type type, int n) {
    auto * t = ggml_new_tensor_1d(ctx, type, n);
    ggml_set_input(t);
    return t;
}

llama_rocketkv::llama_rocketkv(const llama_model & model, const llama_cparams & cparams, llama_kv_cache & cache, const llama_rocketkv_params & params) :
    params(params), cache(cache), dim(model.hparams.n_embd_head_k()), heads(model.hparams.n_head()), kv_heads(model.hparams.n_head_kv()) {
    if (model.arch != LLM_ARCH_LLAMA || !cparams.causal_attn || !cparams.flash_attn ||
        cparams.n_seq_max != 1 || cparams.training || cparams.ctx_type != LLAMA_CONTEXT_TYPE_DEFAULT ||
        cache.type_k() != GGML_TYPE_F16 || cache.type_v() != GGML_TYPE_F16 ||
        model.hparams.swa_type != LLAMA_SWA_TYPE_NONE || dim != (int) model.hparams.n_embd_head_v() ||
        model.hparams.use_alibi || model.hparams.attn_soft_cap || model.hparams.no_alloc) {
        throw std::invalid_argument("RocketKV MVP requires a Llama decoder, one sequence, causal flash attention, and F16 K/V");
    }
    if (cache.get_cells(0).get_used() != 0) {
        throw std::invalid_argument("enable RocketKV on an empty context");
    }
    if (params.prompt_tokens < 1 || params.prompt_tokens > 32768 || params.decode_tokens < 1 ||
        params.decode_tokens > 8192 || params.prompt_tokens + params.decode_tokens > (int64_t) cparams.n_ctx ||
        params.token_budget < 2 || params.token_budget > 8192 ||
        params.observation_window < 1 || params.observation_window > 256 ||
        params.pooling_kernel < 1 || params.pooling_kernel > 255 || params.pooling_kernel % 2 == 0) {
        throw std::invalid_argument("invalid RocketKV prompt/decode length, budget, observation window, or odd pooling kernel");
    }
    const int total = params.prompt_tokens + params.decode_tokens;
    info.active = total > params.token_budget;
    info.prompt_kept = params.prompt_tokens;
    info.capacity = total;
    info.page_size = 1;
    info.query_dims = dim;
    info.attention_tokens = total;
    const auto layer_ids = cache.get_layer_ids();
    for (int il : layer_ids) {
        if ((int) model.hparams.n_head(il) != heads || (int) model.hparams.n_head_kv(il) != kv_heads ||
            (int) model.hparams.n_embd_head_k(il) != dim || (int) model.hparams.n_embd_head_v(il) != dim) {
            throw std::invalid_argument("RocketKV requires uniform attention dimensions across layers");
        }
        auto * device = ggml_backend_buft_get_device(ggml_backend_buffer_get_type(cache.get_k_storage(il)->buffer));
        if (device && ggml_backend_dev_type(device) != GGML_BACKEND_DEVICE_TYPE_CPU &&
            std::string(ggml_backend_reg_name(ggml_backend_dev_backend_reg(device))) != "MTL") {
            throw std::invalid_argument("RocketKV currently supports only CPU and Metal caches");
        }
        info.allocated_kv_bytes += ggml_nbytes(cache.get_k_storage(il)) + ggml_nbytes(cache.get_v_storage(il));
    }
    if (!info.active) {
        info.active_kv_bytes = uint64_t(total)*dim*kv_heads*4*layer_ids.size();
        return;
    }
    const double ratio = double(total)/params.token_budget;
    const double alpha = std::min(0.2 + std::log2(ratio)*0.06, 0.8);
    info.capacity = int(total/std::pow(ratio, alpha));
    info.prompt_kept = info.capacity - params.decode_tokens;
    if (info.prompt_kept <= params.observation_window || info.prompt_kept > params.prompt_tokens || info.capacity > 8192) {
        throw std::invalid_argument("RocketKV capacity must leave room for the observation window and generated tokens");
    }
    const double hsa_ratio = std::max(1.0, double(info.capacity)/params.token_budget);
    info.page_size = std::min(int(std::floor(hsa_ratio)), int(std::ceil(std::sqrt(hsa_ratio))));
    info.query_dims = std::min(dim, std::max(1, int(std::nearbyint(dim*info.page_size/hsa_ratio))));
    info.attention_tokens = std::min(info.prompt_kept, int(std::nearbyint(params.token_budget/2.0)));
    info.active_kv_bytes = uint64_t(info.capacity)*dim*kv_heads*4*layer_ids.size();
    layers.resize(model.hparams.n_layer());
    for (int il : layer_ids) {
        ggml_context_ptr ctx(ggml_init({8*ggml_tensor_overhead(), nullptr, true}));
        layer l;
        l.q = ggml_new_tensor_2d(ctx.get(), GGML_TYPE_F32, dim*heads, params.observation_window);
        l.metadata = ggml_new_tensor_3d(ctx.get(), GGML_TYPE_F16, (info.capacity + info.page_size - 1)/info.page_size, 2*dim, kv_heads);
        l.retained = ggml_new_tensor_2d(ctx.get(), GGML_TYPE_I32, info.prompt_kept, kv_heads);
        ggml_format_name(l.q, "rocketkv_q_observation_%d", il);
        ggml_format_name(l.metadata, "rocketkv_page_metadata_%d", il);
        ggml_format_name(l.retained, "rocketkv_original_positions_%d", il);
        auto buft = ggml_backend_buffer_get_type(cache.get_k_storage(il)->buffer);
        ggml_backend_buffer_ptr buffer(ggml_backend_alloc_ctx_tensors_from_buft(ctx.get(), buft));
        if (!buffer) {
            throw std::runtime_error("cannot allocate RocketKV auxiliary tensors");
        }
        info.auxiliary_bytes += ggml_backend_buffer_get_size(buffer.get());
        layers[il] = l;
        storage.emplace_back(std::move(ctx), std::move(buffer));
    }
    LLAMA_LOG_INFO("RocketKV: prompt %d -> %d, capacity %d, page %d, r %d, top-k %d; full KV allocation retained\n",
                   params.prompt_tokens, info.prompt_kept, info.capacity, info.page_size, info.query_dims, info.attention_tokens);
}

bool llama_rocketkv::validate(const llama_batch & batch) const {
    if (cache.get_has_shift() || cache.get_cells(0).get_used() != (uint32_t) accepted || cache.seq_pos_max(0) != accepted - 1) {
        LLAMA_LOG_ERROR("RocketKV: cache mutation/reuse is unsupported; create a fresh context for each request\n");
        return false;
    }
    if (accepted + batch.n_tokens > params.prompt_tokens + params.decode_tokens ||
        (accepted < params.prompt_tokens && accepted + batch.n_tokens > params.prompt_tokens) ||
        (accepted >= params.prompt_tokens && batch.n_tokens != 1)) {
        LLAMA_LOG_ERROR("RocketKV: batch crosses the prefill boundary, exceeds the declared length, or is not single-token decode\n");
        return false;
    }
    for (int i = 0; i < batch.n_tokens; ++i) {
        if (batch.pos[i] != accepted + i || batch.n_seq_id[i] != 1 || batch.seq_id[i][0] != 0) {
            LLAMA_LOG_ERROR("RocketKV: only contiguous absolute positions in sequence zero are supported\n");
            return false;
        }
    }
    return true;
}

int llama_rocketkv::phase(const llama_ubatch & ubatch) const {
    if (ubatch.pos[0] >= params.prompt_tokens) {
        return 2;
    }
    return ubatch.pos[ubatch.n_tokens - 1] == params.prompt_tokens - 1 ? 1 : 0;
}

std::map<ggml_backend_buffer_type_t, size_t> llama_rocketkv::memory_breakdown() const {
    std::map<ggml_backend_buffer_type_t, size_t> result;
    for (const auto & entry : storage) {
        result[ggml_backend_buffer_get_type(entry.second.get())] += ggml_backend_buffer_get_size(entry.second.get());
    }
    return result;
}

ggml_tensor * llama_rocketkv::cache_view(ggml_context * ctx, ggml_tensor * data, int n) const {
    return ggml_view_3d(ctx, data, dim, kv_heads, n, dim*sizeof(ggml_fp16_t), dim*kv_heads*sizeof(ggml_fp16_t), 0);
}

void llama_rocketkv::prefill(ggml_context * ctx, llm_graph_result * res, const llama_ubatch & ubatch, ggml_tensor * q, float scale, int il) const {
    auto input = std::make_unique<rocket_input>(*this, ubatch);
    auto * gf = res->get_gf();
    const auto & layer = layers.at(il);
    const int w = params.observation_window, n = params.prompt_tokens, keep = info.prompt_kept;
    const int count = std::min<int>(w, ubatch.n_tokens);
    input->observe = rocket_input_tensor(ctx, GGML_TYPE_I64, count);
    auto * tail = ggml_view_3d(ctx, q, dim, heads, count, q->nb[1], q->nb[2], (ubatch.n_tokens - count)*q->nb[2]);
    auto * flat = rocket_name(ggml_cont_2d(ctx, tail, dim*heads, count), "s1_observe", il);
    auto * observed = rocket_name(ggml_set_rows(ctx, layer.q, flat, input->observe), "s1_observe", il);
    ggml_build_forward_expand(gf, observed);
    if (phase(ubatch) == 1) {
        input->order = rocket_input_tensor(ctx, GGML_TYPE_I32, w);
        input->live = rocket_input_tensor(ctx, GGML_TYPE_I32, 1);
        input->mask = ggml_new_tensor_2d(ctx, GGML_TYPE_F16, n, w);
        ggml_set_input(input->mask);
        auto * qobs = rocket_name(ggml_get_rows(ctx, observed, input->order), "s1_observe", il);
        qobs = ggml_permute(ctx, ggml_reshape_3d(ctx, qobs, dim, heads, w), 0, 2, 1, 3);
        auto * keys = ggml_permute(ctx, cache_view(ctx, cache.get_k_storage(il), n), 0, 2, 1, 3);
        auto * values = ggml_permute(ctx, cache_view(ctx, cache.get_v_storage(il), n), 0, 2, 1, 3);
        auto * raw = rocket_name(ggml_mul_mat(ctx, keys, qobs), "s1_score", il);
        ggml_prec_set_acc(raw, GGML_PREC_F32);
        auto * prob = rocket_name(ggml_soft_max_ext(ctx, raw, input->mask, scale, 0), "s1_score", il);
        auto * sums = rocket_name(ggml_rocketkv_reduce(ctx, prob, kv_heads, n - w), "s1_score", il);
        auto * pool = rocket_name(ggml_pool_1d(ctx, sums, GGML_OP_POOL_MAX, params.pooling_kernel, 1, params.pooling_kernel/2), "s1_pool", il);
        auto * top = rocket_name(ggml_rocketkv_top_k(ctx, pool, keep - w), "s1_select", il);
        auto * ids = rocket_name(ggml_rocketkv_indices(ctx, top, w, n), "s1_index", il);
        ids = rocket_name(ggml_cpy(ctx, ids, layer.retained), "s1_index", il);
        auto * kc = rocket_name(ggml_get_rows(ctx, keys, ids), "s1_compact", il);
        auto * vc = rocket_name(ggml_get_rows(ctx, values, ids), "s1_compact", il);
        ggml_build_forward_expand(gf, kc);
        ggml_build_forward_expand(gf, vc);
        auto * kd = ggml_permute(ctx, cache_view(ctx, cache.get_k_storage(il), keep), 0, 2, 1, 3);
        auto * vd = ggml_permute(ctx, cache_view(ctx, cache.get_v_storage(il), keep), 0, 2, 1, 3);
        auto * kw = rocket_name(ggml_cpy(ctx, kc, kd), "s1_compact", il);
        auto * vw = rocket_name(ggml_cpy(ctx, vc, vd), "s1_compact", il);
        ggml_build_forward_expand(gf, kw);
        ggml_build_forward_expand(gf, vw);
        auto * meta = ggml_rocketkv_metadata(ctx, ggml_permute(ctx, kw, 0, 2, 1, 3), layer.metadata, input->live, info.page_size, true);
        ggml_build_forward_expand(gf, rocket_name(meta, "s1_metadata", il));
    }
    res->add_input(std::move(input));
}

ggml_tensor * llama_rocketkv::decode(ggml_context * ctx, llm_graph_result * res, const llama_ubatch & ubatch,
                                    ggml_tensor * q, ggml_tensor * k, ggml_tensor * v, float scale, int il) const {
    auto input = std::make_unique<rocket_input>(*this, ubatch);
    input->live = rocket_input_tensor(ctx, GGML_TYPE_I32, 1);
    input->write = rocket_input_tensor(ctx, GGML_TYPE_I64, 1);
    auto * gf = res->get_gf();
    auto * kflat = rocket_name(ggml_is_contiguous(k) ?
        ggml_reshape_2d(ctx, k, dim*kv_heads, 1) : ggml_cont_2d(ctx, k, dim*kv_heads, 1), "s2_index", il);
    auto * vflat = rocket_name(ggml_is_contiguous(v) ?
        ggml_reshape_2d(ctx, v, dim*kv_heads, 1) : ggml_cont_2d(ctx, v, dim*kv_heads, 1), "s2_index", il);
    auto * kw = rocket_name(ggml_set_rows(ctx, cache.get_k_storage(il), kflat, input->write), "s2_index", il);
    auto * vw = rocket_name(ggml_set_rows(ctx, cache.get_v_storage(il), vflat, input->write), "s2_index", il);
    ggml_build_forward_expand(gf, kw);
    ggml_build_forward_expand(gf, vw);
    auto * keys = cache_view(ctx, kw, info.capacity);
    auto * values = cache_view(ctx, vw, info.capacity);
    auto * meta = rocket_name(ggml_rocketkv_metadata(ctx, keys, layers.at(il).metadata, input->live, info.page_size, false), "s2_metadata", il);
    auto * sums = rocket_name(ggml_rocketkv_query(ctx, q, kv_heads), "s2_query", il);
    auto * magnitude = ggml_view_2d(ctx, sums, dim, kv_heads, sums->nb[2], 0);
    auto * dims = rocket_name(ggml_rocketkv_top_k(ctx, magnitude, info.query_dims), "s2_query", il);
    auto * scores = rocket_name(ggml_rocketkv_scores(ctx, q, meta, dims, sums, input->live, info.page_size, info.capacity), "s2_score", il);
    auto * weights = rocket_name(ggml_soft_max(ctx, scores), "s2_score", il);
    auto * group_weights = rocket_name(ggml_rocketkv_reduce(ctx, weights, kv_heads, info.capacity), "s2_score", il);
    auto * top = rocket_name(ggml_rocketkv_page_top_k(ctx, group_weights, input->live, info.page_size, info.attention_tokens), "s2_select", il);
    auto * gathered = rocket_name(ggml_rocketkv_gather(ctx, ggml_permute(ctx, keys, 0, 2, 1, 3),
        ggml_permute(ctx, values, 0, 2, 1, 3), top), "s2_gather", il);
    auto * kg = ggml_view_3d(ctx, gathered, dim, info.attention_tokens, kv_heads, gathered->nb[1], gathered->nb[2], 0);
    auto * vg = ggml_view_3d(ctx, gathered, dim, info.attention_tokens, kv_heads, gathered->nb[1], gathered->nb[2], gathered->nb[3]);
    auto * out = ggml_flash_attn_ext(ctx, ggml_permute(ctx, q, 0, 2, 1, 3), kg, vg, nullptr, scale, 0, 0);
    ggml_prec_set_acc(out, GGML_PREC_F32);
    rocket_name(out, "s2_attention", il);
    ggml_build_forward_expand(gf, out);
    res->add_input(std::move(input));
    return ggml_reshape_2d(ctx, out, dim*heads, 1);
}

bool llama_rocketkv_init(llama_context * ctx, const llama_rocketkv_params & params) {
    if (!ctx) {
        LLAMA_LOG_ERROR("RocketKV: null context\n");
        return false;
    }
    return ctx->init_rocketkv(params);
}

llama_rocketkv_info llama_rocketkv_get_info(const llama_context * ctx) {
    if (!ctx) {
        LLAMA_LOG_ERROR("RocketKV: null context in statistics query\n");
        return {};
    }
    return ctx->get_rocketkv_info();
}
