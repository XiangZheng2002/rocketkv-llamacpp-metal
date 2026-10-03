// Experimental RocketKV research operations. See docs/rocketkv/NOTICE.
#pragma once

#include "ggml.h"

#ifdef __cplusplus
extern "C" {
#endif

enum ggml_rocketkv_op {
    GGML_ROCKETKV_REDUCE,
    GGML_ROCKETKV_TOP_K,
    GGML_ROCKETKV_INDICES,
    GGML_ROCKETKV_METADATA,
    GGML_ROCKETKV_QUERY,
    GGML_ROCKETKV_SCORES,
};

// scores [tokens, queries, heads] -> [tokens_out, kv_heads].
GGML_API struct ggml_tensor * ggml_rocketkv_reduce(struct ggml_context * ctx, struct ggml_tensor * scores, int32_t kv_heads, int32_t tokens_out);
// Descending scores, ascending index on ties. Supports rows of at most 8192 entries.
GGML_API struct ggml_tensor * ggml_rocketkv_top_k(struct ggml_context * ctx, struct ggml_tensor * scores, int32_t k);
// Sort retained prompt indices chronologically, then append the observation window.
GGML_API struct ggml_tensor * ggml_rocketkv_indices(struct ggml_context * ctx, struct ggml_tensor * selected, int32_t window, int32_t prompt);
// keys [dim, kv_heads, capacity], metadata [pages, 2*dim, kv_heads], live [1] I32.
GGML_API struct ggml_tensor * ggml_rocketkv_metadata(struct ggml_context * ctx, struct ggml_tensor * keys, struct ggml_tensor * metadata, struct ggml_tensor * live, int32_t page, bool initialize);
// queries [dim, heads, 1] -> group sums [dim, 2, kv_heads] (absolute, signed).
GGML_API struct ggml_tensor * ggml_rocketkv_query(struct ggml_context * ctx, struct ggml_tensor * queries, int32_t kv_heads);
GGML_API struct ggml_tensor * ggml_rocketkv_scores(struct ggml_context * ctx, struct ggml_tensor * queries, struct ggml_tensor * metadata, struct ggml_tensor * dims, struct ggml_tensor * sums, struct ggml_tensor * live, int32_t page, int32_t capacity);

GGML_API bool ggml_rocketkv_supported(const struct ggml_tensor * op);
GGML_API void ggml_rocketkv_compute_forward(struct ggml_tensor * op, int ith, int nth);

#ifdef __cplusplus
}
#endif
