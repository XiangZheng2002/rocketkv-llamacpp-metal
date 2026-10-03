// RocketKV research implementation. See docs/rocketkv/NOTICE.
#include "rocketkv-reference.h"
#include "json.h"
#include "ggml-rocketkv.h"
#include "ggml-backend.h"
#include "ggml-cpp.h"

#include <cstdio>
#include <fstream>
#include <string>

using json = common_json;
using namespace rocketkv_ref;

static void check(const std::string & name, const std::vector<float> & actual, const json & expected, float tolerance = 2e-5f) {
    const auto ref = expected.get<std::vector<float>>();
    if (actual.size() != ref.size()) {
        throw std::runtime_error(name + ": shape mismatch");
    }
    for (size_t i = 0; i < ref.size(); ++i) {
        if (actual[i] == ref[i]) {
            continue;
        }
        if (!std::isfinite(actual[i]) || std::abs(actual[i] - ref[i]) > tolerance*(1 + std::abs(ref[i]))) {
            throw std::runtime_error(name + ": mismatch at " + std::to_string(i) + ": " + std::to_string(actual[i]) + " != " + std::to_string(ref[i]));
        }
    }
}

static void check_ids(const std::string & name, const std::vector<int> & actual, const json & expected) {
    if (actual != expected.get<std::vector<int>>()) {
        throw std::runtime_error(name + ": indices differ");
    }
}

static tensor load(const json & c, const char * key, int h, int n, int d) {
    tensor t(h, n, d);
    t.data = c.at(key).get<std::vector<float>>();
    if (t.data.size() != size_t(h*n*d)) {
        throw std::runtime_error("invalid fixture shape");
    }
    return t;
}

static std::vector<float> read_tensor(ggml_tensor * t) {
    std::vector<float> values(ggml_nelements(t));
    if (t->type == GGML_TYPE_F16) {
        std::vector<ggml_fp16_t> half(values.size());
        ggml_backend_tensor_get(t, half.data(), 0, ggml_nbytes(t));
        ggml_fp16_to_fp32_row(half.data(), values.data(), values.size());
    } else {
        ggml_backend_tensor_get(t, values.data(), 0, ggml_nbytes(t));
    }
    return values;
}

static void test_graph(const json & c, ggml_backend_t backend) {
    const int h = c.at("heads").get<int>(), hk = c.at("kv_heads").get<int>(), n = c.at("n").get<int>();
    const int d = c.at("d").get<int>(), w = c.at("window").get<int>(), cap = c.at("capacity").get<int>();
    const int page = c.at("page").get<int>(), r = c.at("r").get<int>(), count = c.at("k").get<int>();
    ggml_context_ptr ctx(ggml_init({8*1024*1024, nullptr, true}));
    auto * gf = ggml_new_graph(ctx.get());
    auto * keys = ggml_new_tensor_3d(ctx.get(), GGML_TYPE_F16, d, n, hk);
    auto * values = ggml_new_tensor_3d(ctx.get(), GGML_TYPE_F16, d, n, hk);
    auto * qobs = ggml_new_tensor_3d(ctx.get(), GGML_TYPE_F32, d, w, h);
    auto * query = ggml_new_tensor_3d(ctx.get(), GGML_TYPE_F32, d, h, 1);
    auto * mask = ggml_new_tensor_2d(ctx.get(), GGML_TYPE_F16, n, w);
    auto * live = ggml_new_tensor_1d(ctx.get(), GGML_TYPE_I32, 1);
    auto * meta = ggml_new_tensor_3d(ctx.get(), GGML_TYPE_F16, (cap + page - 1)/page, 2*d, hk);
    auto * raw = ggml_mul_mat(ctx.get(), keys, qobs);
    auto * prob = ggml_soft_max_ext(ctx.get(), raw, mask, 1/std::sqrt(float(d)), 0);
    auto * sums = ggml_rocketkv_reduce(ctx.get(), prob, hk, n - w);
    const int kernel = c.at("kernel").get<int>();
    auto * pooled = ggml_pool_1d(ctx.get(), sums, GGML_OP_POOL_MAX, kernel, 1, kernel/2);
    auto * selected = ggml_rocketkv_top_k(ctx.get(), pooled, cap - w);
    auto * retained = ggml_rocketkv_indices(ctx.get(), selected, w, n);
    auto * kc = ggml_cast(ctx.get(), ggml_get_rows(ctx.get(), keys, retained), GGML_TYPE_F16);
    auto * vc = ggml_cast(ctx.get(), ggml_get_rows(ctx.get(), values, retained), GGML_TYPE_F16);
    auto * metadata = ggml_rocketkv_metadata(ctx.get(), ggml_permute(ctx.get(), kc, 0, 2, 1, 3), meta, live, page, true);
    auto * qsums = ggml_rocketkv_query(ctx.get(), query, hk);
    auto * magnitudes = ggml_view_2d(ctx.get(), qsums, d, hk, qsums->nb[2], 0);
    auto * dims = ggml_rocketkv_top_k(ctx.get(), magnitudes, r);
    auto * approx = ggml_rocketkv_scores(ctx.get(), query, metadata, dims, qsums, live, page, cap);
    auto * weights = ggml_soft_max(ctx.get(), approx);
    auto * group_weights = ggml_rocketkv_reduce(ctx.get(), weights, hk, cap);
    auto * top = ggml_rocketkv_top_k(ctx.get(), group_weights, count);
    auto * kg = ggml_get_rows(ctx.get(), kc, top);
    auto * vg = ggml_get_rows(ctx.get(), vc, top);
    auto * q = ggml_permute(ctx.get(), query, 0, 2, 1, 3);
    auto * exact = ggml_soft_max_ext(ctx.get(), ggml_mul_mat(ctx.get(), kg, q), nullptr, 1/std::sqrt(float(d)), 0);
    auto * output = ggml_mul_mat(ctx.get(), ggml_cont(ctx.get(), ggml_transpose(ctx.get(), vg)), exact);
    ggml_build_forward_expand(gf, output);
    for (int i = 0; i < ggml_graph_n_nodes(gf); ++i) {
        auto * node = ggml_graph_node(gf, i);
        if (!ggml_backend_supports_op(backend, node)) {
            throw std::runtime_error(std::string(ggml_backend_name(backend)) + " does not support " + ggml_op_name(node->op));
        }
    }
    ggml_backend_buffer_ptr buffer(ggml_backend_alloc_ctx_tensors(ctx.get(), backend));
    if (!buffer) {
        throw std::runtime_error("cannot allocate fixture graph");
    }
    auto set = [&](ggml_tensor * t, const json & j) {
        const auto data = j.get<std::vector<float>>();
        if (t->type == GGML_TYPE_F16) {
            std::vector<ggml_fp16_t> half(data.size());
            ggml_fp32_to_fp16_row(data.data(), half.data(), data.size());
            ggml_backend_tensor_set(t, half.data(), 0, ggml_nbytes(t));
        } else {
            ggml_backend_tensor_set(t, data.data(), 0, ggml_nbytes(t));
        }
    };
    set(keys, c.at("keys"));
    set(values, c.at("values"));
    set(qobs, c.at("q_obs"));
    set(query, c.at("query"));
    std::vector<ggml_fp16_t> mask_data(n*w);
    for (int i = 0; i < w; ++i) {
        for (int t = 0; t < n; ++t) {
            mask_data[i*n + t] = ggml_fp32_to_fp16(t <= n - w + i ? 0 : -INFINITY);
        }
    }
    ggml_backend_tensor_set(mask, mask_data.data(), 0, ggml_nbytes(mask));
    ggml_backend_tensor_set(live, &cap, 0, sizeof(cap));
    if (ggml_backend_graph_compute(backend, gf) != GGML_STATUS_SUCCESS) {
        throw std::runtime_error("fixture graph computation failed");
    }
    auto ids = [&](const char * name, ggml_tensor * t, const json & expected) {
        std::vector<int> data(ggml_nelements(t));
        ggml_backend_tensor_get(t, data.data(), 0, ggml_nbytes(t));
        check_ids(name, data, expected);
    };
    check("graph SnapKV scores", read_tensor(sums), c.at("snap_scores"), 2e-4f);
    check("graph pooling", read_tensor(pooled), c.at("pooled"), 2e-4f);
    ids("graph retained", retained, c.at("retained"));
    check("graph compaction K", read_tensor(kc), c.at("compact_k"), 0);
    check("graph compaction V", read_tensor(vc), c.at("compact_v"), 0);
    const auto metadata_values = read_tensor(metadata);
    std::vector<float> lo, hi;
    const int pages = meta->ne[0];
    for (int head = 0; head < hk; ++head) {
        for (int p = 0; p < pages; ++p) {
            for (int dim = 0; dim < d; ++dim) {
                lo.push_back(metadata_values[p + pages*(dim + 2*d*head)]);
                hi.push_back(metadata_values[p + pages*(dim + d + 2*d*head)]);
            }
        }
    }
    check("graph Kmin", lo, c.at("kmin"), 0);
    check("graph Kmax", hi, c.at("kmax"), 0);
    ids("graph top-r", dims, c.at("dims"));
    const auto approx_values = read_tensor(approx);
    std::vector<float> pages_actual;
    for (int head = 0; head < h; ++head) {
        for (int p = 0; p < pages; ++p) {
            pages_actual.push_back(approx_values[head*cap + p*page]);
        }
    }
    check("graph approximate pages", pages_actual, c.at("page_scores"), 2e-4f);
    check("graph approximate tokens", read_tensor(group_weights), c.at("token_scores"), 2e-4f);
    ids("graph top-k", top, c.at("selected"));
    check("graph gather K", read_tensor(kg), c.at("gather_k"), 0);
    check("graph gather V", read_tensor(vg), c.at("gather_v"), 0);
    check("graph attention", read_tensor(output), c.at("output"), 5e-4f);
    std::vector<float> zeros(d*h);
    ggml_backend_tensor_set(query, zeros.data(), 0, ggml_nbytes(query));
    if (ggml_backend_graph_compute(backend, gf) != GGML_STATUS_SUCCESS) {
        throw std::runtime_error("zero-query graph failed");
    }
    tensor zero(h, 1, d);
    const auto ck = load(c, "compact_k", hk, cap, d), cv = load(c, "compact_v", hk, cap, d);
    rocketkv_ref::metadata zero_meta(hk, cap, d, page);
    update(zero_meta, ck, 0);
    const auto zero_ref = hsa(zero, ck, cv, zero_meta, r, count);
    ids("graph zero-query ties", top, zero_ref.indices);
    check("graph zero-query attention", read_tensor(output), zero_ref.output.data, 5e-4f);
    std::printf("PASS %s graph %s\n", ggml_backend_name(backend), c.at("name").get<std::string>().c_str());
}

static void test_incremental(ggml_backend_t backend, ggml_type type, int page) {
    const int dim = 16, heads = 4, kv_heads = 2, capacity = 13;
    ggml_context_ptr ctx(ggml_init({2*1024*1024, nullptr, true}));
    auto * k = ggml_new_tensor_3d(ctx.get(), type, dim, kv_heads, capacity);
    auto * q = ggml_new_tensor_3d(ctx.get(), GGML_TYPE_F32, dim, heads, 1);
    auto * meta = ggml_new_tensor_3d(ctx.get(), type, (capacity + page - 1)/page, 2*dim, kv_heads);
    auto * live = ggml_new_tensor_1d(ctx.get(), GGML_TYPE_I32, 1);
    auto * init = ggml_rocketkv_metadata(ctx.get(), k, meta, live, page, true);
    auto * inc = ggml_rocketkv_metadata(ctx.get(), k, meta, live, page, false);
    auto * sums = ggml_rocketkv_query(ctx.get(), q, kv_heads);
    auto * dims = ggml_rocketkv_top_k(ctx.get(), ggml_view_2d(ctx.get(), sums, dim, kv_heads, sums->nb[2], 0), 5);
    auto * scores = ggml_rocketkv_scores(ctx.get(), q, meta, dims, sums, live, page, capacity);
    auto * weights = ggml_soft_max(ctx.get(), scores);
    auto * g0 = ggml_new_graph(ctx.get()), * g1 = ggml_new_graph(ctx.get()), * g2 = ggml_new_graph(ctx.get());
    ggml_build_forward_expand(g0, init);
    ggml_build_forward_expand(g1, inc);
    ggml_build_forward_expand(g2, weights);
    ggml_backend_buffer_ptr buffer(ggml_backend_alloc_ctx_tensors(ctx.get(), backend));
    if (!buffer) {
        throw std::runtime_error("incremental fixture allocation failed");
    }
    tensor source(kv_heads, capacity, dim);
    std::vector<float> packed(dim*kv_heads*capacity), queries(dim*heads);
    for (int t = 0; t < capacity; ++t) {
        for (int h = 0; h < kv_heads; ++h) {
            for (int d = 0; d < dim; ++d) {
                const float value = ((h*31 + t*7 + d) % 53 - 26)/16.0f;
                source.at(h, t, d) = packed[(t*kv_heads + h)*dim + d] = value;
            }
        }
    }
    if (type == GGML_TYPE_F16) {
        std::vector<ggml_fp16_t> half(packed.size());
        ggml_fp32_to_fp16_row(packed.data(), half.data(), packed.size());
        ggml_backend_tensor_set(k, half.data(), 0, ggml_nbytes(k));
    } else {
        ggml_backend_tensor_set(k, packed.data(), 0, ggml_nbytes(k));
    }
    for (int i = 0; i < (int) queries.size(); ++i) {
        queries[i] = (i % 7 - 3)/4.0f;
    }
    ggml_backend_tensor_set(q, queries.data(), 0, ggml_nbytes(q));
    for (int n = 1; n <= capacity; ++n) {
        ggml_backend_tensor_set(live, &n, 0, sizeof(n));
        if (ggml_backend_graph_compute(backend, n == 1 ? g0 : g1) != GGML_STATUS_SUCCESS ||
            ggml_backend_graph_compute(backend, g2) != GGML_STATUS_SUCCESS) {
            throw std::runtime_error("incremental graph failed");
        }
        tensor prefix(kv_heads, n, dim);
        for (int h = 0; h < kv_heads; ++h) {
            for (int t = 0; t < n; ++t) {
                for (int d = 0; d < dim; ++d) {
                    prefix.at(h, t, d) = source.at(h, t, d);
                }
            }
        }
        metadata expected(kv_heads, capacity, dim, page);
        update(expected, prefix, 0);
        const auto data = read_tensor(meta);
        const int pages = meta->ne[0];
        std::vector<float> lo, hi;
        for (int h = 0; h < kv_heads; ++h) {
            for (int p = 0; p < pages; ++p) {
                for (int d = 0; d < dim; ++d) {
                    lo.push_back(data[p + pages*(d + 2*dim*h)]);
                    hi.push_back(data[p + pages*(d + dim + 2*dim*h)]);
                }
            }
        }
        check("incremental GPU/CPU min", lo, expected.min.data, 0);
        check("incremental GPU/CPU max", hi, expected.max.data, 0);
        const auto prob = read_tensor(weights);
        for (int h = 0; h < heads; ++h) {
            float sum = 0;
            for (int t = 0; t < capacity; ++t) {
                const float x = prob[h*capacity + t];
                if (!std::isfinite(x) || (t >= n && x != 0)) {
                    throw std::runtime_error("future/padding token received attention");
                }
                sum += x;
            }
            if (std::abs(sum - 1) > 1e-5f) {
                throw std::runtime_error("masked probabilities do not sum to one");
            }
        }
    }
    std::printf("PASS %s incremental %s page=%d, partial pages and padding\n", ggml_backend_name(backend), ggml_type_name(type), page);
}

static void test_case(const json & c) {
    const int h = c.at("heads").get<int>(), hk = c.at("kv_heads").get<int>(), n = c.at("n").get<int>();
    const int d = c.at("d").get<int>(), w = c.at("window").get<int>();
    auto qobs = load(c, "q_obs", h, w, d);
    auto k = load(c, "keys", hk, n, d);
    auto v = load(c, "values", hk, n, d);
    auto q = load(c, "query", h, 1, d);
    const auto s = snap(qobs, k, c.at("capacity").get<int>(), c.at("kernel").get<int>());
    check("snap scores", s.scores, c.at("snap_scores"));
    check("pooling", s.pooled, c.at("pooled"));
    check_ids("retained", s.indices, c.at("retained"));
    k = gather(k, s.indices);
    v = gather(v, s.indices);
    check("compact K", k.data, c.at("compact_k"));
    check("compact V", v.data, c.at("compact_v"));
    metadata meta(hk, k.n, d, c.at("page").get<int>());
    update(meta, k, 0);
    check("Kmin", meta.min.data, c.at("kmin"));
    check("Kmax", meta.max.data, c.at("kmax"));
    const auto result = hsa(q, k, v, meta, c.at("r").get<int>(), c.at("k").get<int>());
    check_ids("dimensions", result.dims, c.at("dims"));
    check("page scores", result.page_scores, c.at("page_scores"));
    check("token scores", result.token_scores, c.at("token_scores"));
    check_ids("top-k", result.indices, c.at("selected"));
    check("gather K", result.keys.data, c.at("gather_k"));
    check("gather V", result.values.data, c.at("gather_v"));
    check("attention", result.output.data, c.at("output"));
    const auto exact_ids = exact_topk(q, k, c.at("k").get<int>());
    check_ids("Exact-TopK indices", exact_ids, c.at("exact_selected"));
    check("Exact-TopK", attention(q, gather(k, exact_ids), gather(v, exact_ids)).data, c.at("exact_output"));

    metadata incremental(hk, k.n, d, c.at("page").get<int>());
    for (int t = 0; t < k.n; ++t) {
        tensor prefix(hk, t + 1, d);
        for (int head = 0; head < hk; ++head) {
            for (int i = 0; i <= t; ++i) {
                for (int dim = 0; dim < d; ++dim) {
                    prefix.at(head, i, dim) = k.at(head, i, dim);
                }
            }
        }
        update(incremental, prefix, t);
    }
    check("incremental min", incremental.min.data, c.at("kmin"), 0);
    check("incremental max", incremental.max.data, c.at("kmax"), 0);
    std::fill(q.data.begin(), q.data.end(), 0);
    const auto zero = hsa(q, k, v, meta, d, k.n);
    check("zero query/full budget", zero.output.data, attention(q, k, v).data);
    std::printf("PASS %s\n", c.at("name").get<std::string>().c_str());
}

int main(int argc, char ** argv) {
    try {
        if (argc < 2 || argc > 3) {
            throw std::invalid_argument("usage: test-rocketkv fixture.json [CPU|MTL0]");
        }
        std::ifstream file(argv[1]);
        if (!file) {
            throw std::runtime_error("cannot open fixture");
        }
        const json fixtures = json::parse(std::string(std::istreambuf_iterator<char>(file), {}));
        for (const auto & c : fixtures.at("cases")) {
            test_case(c);
        }
        ggml_backend_load_all();
        ggml_backend_ptr backend(ggml_backend_init_by_name(argc == 3 ? argv[2] : "CPU", nullptr));
        if (!backend) {
            throw std::runtime_error("requested backend is unavailable");
        }
        for (const auto & c : fixtures.at("cases")) {
            test_graph(c, backend.get());
        }
        for (auto type : {GGML_TYPE_F16, GGML_TYPE_F32}) {
            for (int page : {1, 2, 3, 5}) {
                test_incremental(backend.get(), type, page);
            }
        }
        check_ids("stable ties", topk({1, 1, 1, 0}, 2), json::array({0, 1}));
        bool rejected = false;
        try {
            topk({1}, 2);
        } catch (const std::invalid_argument &) {
            rejected = true;
        }
        if (!rejected) {
            throw std::runtime_error("invalid top-k was accepted");
        }
        return 0;
    } catch (const std::exception & e) {
        std::fprintf(stderr, "FAIL: %s\n", e.what());
        return 1;
    }
}
