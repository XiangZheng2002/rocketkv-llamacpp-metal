// RocketKV research implementation. See docs/rocketkv/NOTICE.
#include "ggml-rocketkv.h"
#include "ggml-impl.h"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <numeric>
#include <vector>

static ggml_tensor * rocket_op(ggml_tensor * out, ggml_rocketkv_op kind, ggml_tensor * a, int p1 = 0, int p2 = 0) {
    out->op = GGML_OP_ROCKETKV;
    out->op_params[0] = kind;
    out->op_params[1] = p1;
    out->op_params[2] = p2;
    out->src[0] = a;
    return out;
}

ggml_tensor * ggml_rocketkv_reduce(ggml_context * ctx, ggml_tensor * scores, int32_t kv_heads, int32_t tokens_out) {
    GGML_ASSERT(scores->type == GGML_TYPE_F32 && scores->ne[3] == 1);
    GGML_ASSERT(kv_heads > 0 && scores->ne[2] % kv_heads == 0);
    GGML_ASSERT(tokens_out > 0 && tokens_out <= scores->ne[0]);
    return rocket_op(ggml_new_tensor_2d(ctx, GGML_TYPE_F32, tokens_out, kv_heads), GGML_ROCKETKV_REDUCE, scores);
}

ggml_tensor * ggml_rocketkv_top_k(ggml_context * ctx, ggml_tensor * scores, int32_t k) {
    GGML_ASSERT(scores->type == GGML_TYPE_F32 && scores->ne[2] == 1 && scores->ne[3] == 1);
    GGML_ASSERT(scores->ne[0] <= 8192 && k > 0 && k <= scores->ne[0]);
    return rocket_op(ggml_new_tensor_2d(ctx, GGML_TYPE_I32, k, scores->ne[1]), GGML_ROCKETKV_TOP_K, scores);
}

ggml_tensor * ggml_rocketkv_indices(ggml_context * ctx, ggml_tensor * selected, int32_t window, int32_t prompt) {
    GGML_ASSERT(selected->type == GGML_TYPE_I32 && ggml_is_contiguous(selected));
    GGML_ASSERT(selected->ne[2] == 1 && selected->ne[3] == 1);
    GGML_ASSERT(window > 0 && prompt > window && selected->ne[0] <= prompt - window);
    return rocket_op(ggml_new_tensor_2d(ctx, GGML_TYPE_I32, selected->ne[0] + window, selected->ne[1]),
                     GGML_ROCKETKV_INDICES, selected, window, prompt);
}

ggml_tensor * ggml_rocketkv_metadata(ggml_context * ctx, ggml_tensor * keys, ggml_tensor * metadata, ggml_tensor * live, int32_t page, bool initialize) {
    GGML_ASSERT(keys->type == GGML_TYPE_F16 || keys->type == GGML_TYPE_F32);
    GGML_ASSERT(metadata->type == keys->type && ggml_is_contiguous(metadata));
    GGML_ASSERT(keys->ne[3] == 1 && metadata->ne[3] == 1 && page > 0);
    GGML_ASSERT(metadata->ne[0] >= (keys->ne[2] + page - 1)/page);
    GGML_ASSERT(metadata->ne[1] == 2*keys->ne[0] && metadata->ne[2] == keys->ne[1]);
    GGML_ASSERT(live->type == GGML_TYPE_I32 && ggml_nelements(live) == 1);
    auto * out = rocket_op(ggml_view_tensor(ctx, metadata), GGML_ROCKETKV_METADATA, keys, page, initialize);
    out->src[1] = live;
    out->src[2] = metadata;
    return out;
}

ggml_tensor * ggml_rocketkv_query(ggml_context * ctx, ggml_tensor * queries, int32_t kv_heads) {
    GGML_ASSERT(queries->type == GGML_TYPE_F32 && queries->ne[2] == 1 && queries->ne[3] == 1);
    GGML_ASSERT(kv_heads > 0 && queries->ne[1] % kv_heads == 0);
    return rocket_op(ggml_new_tensor_3d(ctx, GGML_TYPE_F32, queries->ne[0], 2, kv_heads), GGML_ROCKETKV_QUERY, queries);
}

ggml_tensor * ggml_rocketkv_scores(ggml_context * ctx, ggml_tensor * queries, ggml_tensor * metadata, ggml_tensor * dims,
                                  ggml_tensor * sums, ggml_tensor * live, int32_t page, int32_t capacity) {
    GGML_ASSERT(queries->type == GGML_TYPE_F32 && queries->ne[2] == 1 && queries->ne[3] == 1);
    GGML_ASSERT(metadata->type == GGML_TYPE_F16 || metadata->type == GGML_TYPE_F32);
    GGML_ASSERT(ggml_is_contiguous(metadata) && metadata->ne[3] == 1);
    GGML_ASSERT(metadata->ne[1] == 2*queries->ne[0] && queries->ne[1] % metadata->ne[2] == 0);
    GGML_ASSERT(dims->type == GGML_TYPE_I32 && ggml_is_contiguous(dims) && dims->ne[1] == metadata->ne[2]);
    GGML_ASSERT(dims->ne[0] > 0 && dims->ne[0] <= queries->ne[0] && dims->ne[2] == 1 && dims->ne[3] == 1);
    GGML_ASSERT(sums->type == GGML_TYPE_F32 && ggml_is_contiguous(sums));
    GGML_ASSERT(sums->ne[0] == queries->ne[0] && sums->ne[1] == 2 && sums->ne[2] == metadata->ne[2] && sums->ne[3] == 1);
    GGML_ASSERT(live->type == GGML_TYPE_I32 && ggml_nelements(live) == 1);
    GGML_ASSERT(page > 0 && capacity > 0 && capacity <= 8192 && (capacity + page - 1)/page <= metadata->ne[0]);
    auto * out = rocket_op(ggml_new_tensor_3d(ctx, GGML_TYPE_F32, capacity, 1, queries->ne[1]),
                           GGML_ROCKETKV_SCORES, queries, page);
    out->src[1] = metadata;
    out->src[2] = dims;
    out->src[3] = sums;
    out->src[4] = live;
    return out;
}

bool ggml_rocketkv_supported(const ggml_tensor * op) {
    if (op->op != GGML_OP_ROCKETKV || !op->src[0] || !ggml_is_contiguous(op)) {
        return false;
    }
    switch (op->op_params[0]) {
        case GGML_ROCKETKV_REDUCE:
        case GGML_ROCKETKV_QUERY:
            return op->type == GGML_TYPE_F32 && op->src[0]->type == GGML_TYPE_F32;
        case GGML_ROCKETKV_TOP_K:
            return op->type == GGML_TYPE_I32 && op->src[0]->type == GGML_TYPE_F32 && op->src[0]->ne[0] <= 8192;
        case GGML_ROCKETKV_INDICES:
            return op->type == GGML_TYPE_I32 && op->src[0]->type == GGML_TYPE_I32;
        case GGML_ROCKETKV_METADATA:
            return op->src[1] && op->src[1]->type == GGML_TYPE_I32 &&
                (op->type == GGML_TYPE_F16 || op->type == GGML_TYPE_F32) && op->type == op->src[0]->type;
        case GGML_ROCKETKV_SCORES:
            return op->type == GGML_TYPE_F32 && op->src[0]->type == GGML_TYPE_F32 && op->src[1] &&
                (op->src[1]->type == GGML_TYPE_F16 || op->src[1]->type == GGML_TYPE_F32) &&
                op->src[2] && op->src[2]->type == GGML_TYPE_I32 &&
                op->src[3] && op->src[3]->type == GGML_TYPE_F32 &&
                op->src[4] && op->src[4]->type == GGML_TYPE_I32;
        default:
            return false;
    }
}

static float rocket_read(const ggml_tensor * t, int a, int b = 0, int c = 0) {
    const char * p = (const char *) t->data + a*t->nb[0] + b*t->nb[1] + c*t->nb[2];
    if (t->type == GGML_TYPE_F16) {
        return ggml_fp16_to_fp32(*(const ggml_fp16_t *) p);
    }
    return *(const float *) p;
}

static void rocket_write(ggml_tensor * t, size_t i, float value) {
    if (t->type == GGML_TYPE_F16) {
        ((ggml_fp16_t *) t->data)[i] = ggml_fp32_to_fp16(value);
    } else {
        ((float *) t->data)[i] = value;
    }
}

void ggml_rocketkv_compute_forward(ggml_tensor * out, int ith, int nth) {
    GGML_ASSERT(ggml_rocketkv_supported(out));
    const auto * a = out->src[0];
    switch (out->op_params[0]) {
        case GGML_ROCKETKV_REDUCE: {
            const int group = a->ne[2]/out->ne[1];
            for (int64_t i = ith; i < ggml_nelements(out); i += nth) {
                const int t = i % out->ne[0], h = i/out->ne[0];
                float sum = 0;
                for (int g = 0; g < group; ++g) {
                    for (int w = 0; w < a->ne[1]; ++w) {
                        sum += rocket_read(a, t, w, h*group + g);
                    }
                }
                ((float *) out->data)[i] = sum;
            }
        } break;
        case GGML_ROCKETKV_TOP_K: {
            for (int h = ith; h < out->ne[1]; h += nth) {
                std::vector<int32_t> ids(a->ne[0]);
                std::iota(ids.begin(), ids.end(), 0);
                std::partial_sort(ids.begin(), ids.begin() + out->ne[0], ids.end(), [&](int x, int y) {
                    const float vx = rocket_read(a, x, h), vy = rocket_read(a, y, h);
                    return vx > vy || (vx == vy && x < y);
                });
                std::memcpy((int32_t *) out->data + h*out->ne[0], ids.data(), out->ne[0]*sizeof(int32_t));
            }
        } break;
        case GGML_ROCKETKV_INDICES: {
            for (int h = ith; h < out->ne[1]; h += nth) {
                int32_t * dst = (int32_t *) out->data + h*out->ne[0];
                const int32_t * src = (const int32_t *) a->data + h*a->ne[0];
                std::copy(src, src + a->ne[0], dst);
                std::sort(dst, dst + a->ne[0]);
                for (int i = a->ne[0]; i < out->ne[0]; ++i) {
                    dst[i] = out->op_params[2] - out->op_params[1] + i - a->ne[0];
                }
            }
        } break;
        case GGML_ROCKETKV_METADATA: {
            const int live = *(const int32_t *) out->src[1]->data;
            const int page = out->op_params[1], dim = a->ne[0], pages = out->ne[0];
            GGML_ASSERT(live > 0 && live <= a->ne[2]);
            const bool init = out->op_params[2];
            const int count = init ? pages*dim*a->ne[1] : dim*a->ne[1];
            for (int i = ith; i < count; i += nth) {
                const int p = init ? i % pages : (live - 1)/page;
                const int d = init ? (i/pages) % dim : i % dim;
                const int h = init ? i/(pages*dim) : i/dim;
                const size_t imin = p + pages*(d + 2*dim*h), imax = imin + pages*dim;
                float lo = INFINITY, hi = -INFINITY;
                int first = p*page;
                if (!init && (live - 1) % page != 0) {
                    lo = rocket_read(out, p, d, h);
                    hi = rocket_read(out, p, d + dim, h);
                    first = live - 1;
                }
                for (int t = first; t < std::min(live, (p + 1)*page); ++t) {
                    const float x = rocket_read(a, d, h, t);
                    lo = std::min(lo, x);
                    hi = std::max(hi, x);
                }
                rocket_write(out, imin, lo);
                rocket_write(out, imax, hi);
            }
        } break;
        case GGML_ROCKETKV_QUERY: {
            const int dim = a->ne[0], group = a->ne[1]/out->ne[2];
            for (int i = ith; i < dim*out->ne[2]; i += nth) {
                const int d = i % dim, h = i/dim;
                float magnitude = 0, sum = 0;
                for (int g = 0; g < group; ++g) {
                    const float q = rocket_read(a, d, h*group + g);
                    magnitude += std::abs(q);
                    sum += q;
                }
                ((float *) out->data)[d + dim*2*h] = magnitude;
                ((float *) out->data)[d + dim*(2*h + 1)] = sum;
            }
        } break;
        case GGML_ROCKETKV_SCORES: {
            const auto * meta = out->src[1];
            const auto * dims = out->src[2];
            const auto * sums = out->src[3];
            const int live = *(const int32_t *) out->src[4]->data;
            const int dim = a->ne[0], group = a->ne[1]/meta->ne[2], page = out->op_params[1];
            GGML_ASSERT(live > 0 && live <= out->ne[0]);
            for (int h = ith; h < a->ne[1]; h += nth) {
                const int kh = h/group;
                const auto * ids = (const int32_t *) dims->data + kh*dims->ne[0];
                float total = 0, selected = 0;
                for (int d = 0; d < dim; ++d) {
                    total += std::abs(rocket_read(a, d, h));
                }
                for (int r = 0; r < dims->ne[0]; ++r) {
                    GGML_ASSERT(ids[r] >= 0 && ids[r] < dim);
                    selected += std::abs(rocket_read(a, ids[r], h));
                }
                const float scale = selected > 0 ? std::sqrt(dim*selected/total) : 1.0f;
                for (int p = 0; p < meta->ne[0]; ++p) {
                    float score = 0;
                    if (p*page < live) {
                        for (int r = 0; r < dims->ne[0]; ++r) {
                            const int d = ids[r];
                            const int signed_d = d + (rocket_read(sums, d, 1, kh) > 0 ? dim : 0);
                            score += rocket_read(a, d, h)*rocket_read(meta, p, signed_d, kh);
                        }
                    }
                    for (int t = p*page; t < std::min<int>(out->ne[0], (p + 1)*page); ++t) {
                        ((float *) out->data)[h*out->ne[0] + t] = t < live ? score/scale : -INFINITY;
                    }
                }
            }
        } break;
        default:
            GGML_ABORT("invalid RocketKV operation");
    }
}
