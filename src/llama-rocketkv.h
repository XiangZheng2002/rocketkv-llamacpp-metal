// RocketKV research implementation. See docs/rocketkv/NOTICE.
#pragma once

#include "llama-ext.h"
#include "ggml-cpp.h"

#include <vector>

struct llama_model;
struct llama_cparams;
struct llama_ubatch;
class llm_graph_result;
class llama_kv_cache;

struct llama_rocketkv {
    llama_rocketkv(const llama_model & model, const llama_cparams & cparams, llama_kv_cache & cache, const llama_rocketkv_params & params);

    bool validate(const llama_batch & batch) const;
    int phase(const llama_ubatch & ubatch) const;
    std::map<ggml_backend_buffer_type_t, size_t> memory_breakdown() const;
    void prefill(ggml_context * ctx, llm_graph_result * res, const llama_ubatch & ubatch, ggml_tensor * q, float scale, int il) const;
    ggml_tensor * decode(ggml_context * ctx, llm_graph_result * res, const llama_ubatch & ubatch, ggml_tensor * q, ggml_tensor * k, ggml_tensor * v, float scale, int il) const;

    llama_rocketkv_params params;
    llama_rocketkv_info info;
    int32_t accepted = 0;

private:
    llama_kv_cache & cache;
    int32_t dim, heads, kv_heads;
    struct layer {
        ggml_tensor * q;
        ggml_tensor * metadata;
        ggml_tensor * retained;
    };
    std::vector<layer> layers;
    std::vector<std::pair<ggml_context_ptr, ggml_backend_buffer_ptr>> storage;

    ggml_tensor * cache_view(ggml_context * ctx, ggml_tensor * data, int n) const;
};
