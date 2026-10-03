# RocketKV on llama.cpp and Metal

Local, experimental research implementation for a single Llama request. This is not a claim of upstream readiness. Read `NOTICE` and `LICENSE.NVIDIA` before redistribution. The original `Material/RocketKV` tree is an external, read-only reference.

Visual summaries with animated diagrams and interactive charts:
[Chinese](../../../../Docs/rocketkv-metal-mvp-report.html) |
[English](../../../../Docs/rocketkv-metal-mvp-report-en.html) |
[English implementation process](../../../../Docs/rocketkv-implementation-process-en.html).

## Reproducible build

The baseline is official `ggml-org/llama.cpp` revision `99b95488cac0f00ce3f05af113a8c1e287753f87`. Commands below run from `Code/llama.cpp`, with the local source changes applied.

```sh
cmake -S . -B build \
  -DCMAKE_BUILD_TYPE=Release -DGGML_METAL=ON \
  -DLLAMA_BUILD_SERVER=OFF -DLLAMA_BUILD_APP=OFF \
  -DLLAMA_BUILD_TESTS=ON -DLLAMA_OPENSSL=OFF
cmake --build build --target llama-rocketkv llama-completion llama-bench \
  test-rocketkv test-backend-ops -j 6
```

At this pinned revision, `llama-cli` depends on the server build. `llama-completion` is the standalone completion CLI; the experimental `llama-rocketkv` tool does not need the server.

The only model downloaded for this experiment is the ready-made `Llama-3.2-3B-Instruct-Q4_K_M.gguf`. No original FP16/BF16 weights, conversions, 8B weights or benchmark datasets are required.

```sh
curl --fail --location --retry 3 \
  -o models/Llama-3.2-3B-Instruct-Q4_K_M.gguf.part \
  https://huggingface.co/bartowski/Llama-3.2-3B-Instruct-GGUF/resolve/5ab33fa94d1d04e903623ae72c95d1696f09f9e8/Llama-3.2-3B-Instruct-Q4_K_M.gguf
mv models/Llama-3.2-3B-Instruct-Q4_K_M.gguf.part \
  models/Llama-3.2-3B-Instruct-Q4_K_M.gguf
shasum -a 256 models/Llama-3.2-3B-Instruct-Q4_K_M.gguf
```

Expected SHA-256: `6c1a2b41161032677be168d354123594c0e6e67d2b9227c84f296ad037c728ff`.

## Run

The same Q4 model file is used in every mode. Model-weight quantization is not a RocketKV saving.

```sh
build/bin/llama-rocketkv \
  -m models/Llama-3.2-3B-Instruct-Q4_K_M.gguf \
  --mode full --kv f16 --prompt-tokens 2048 --generate 32 \
  --warmup 1 --repetitions 3 --output full.json

build/bin/llama-rocketkv \
  -m models/Llama-3.2-3B-Instruct-Q4_K_M.gguf \
  --mode rocket --budget 256 --window 32 --pool 63 \
  --prompt-tokens 2048 --generate 32 \
  --warmup 1 --repetitions 3 --output rocket.json

# Diagnostic run only: encoder-boundary GPU timestamps change scheduling.
build/bin/llama-rocketkv \
  -m models/Llama-3.2-3B-Instruct-Q4_K_M.gguf \
  --mode rocket --budget 256 --prompt-tokens 2048 \
  --warmup 1 --repetitions 3 --profile --output profile.json

# A single quantized-KV sanity comparison, not a quantization matrix.
build/bin/llama-rocketkv \
  -m models/Llama-3.2-3B-Instruct-Q4_K_M.gguf \
  --mode full --kv q8_0 --prompt-tokens 2048 \
  --warmup 1 --repetitions 1 --output q8-kv.json
```

`--prompt` and `--prompt-file` use the Llama-3 instruction template. Otherwise the tool constructs an exact-token-count synthetic archive. `--passkey 37461 --depth 0.25 --prompt-tokens 4096` inserts a five-digit code into that archive. The depth is the fraction of filler tokens before the needle, not a percentage of characters.

Generation is greedy, batch size one, and fixed at the requested emitted length. The first token comes from prefill; 32 emitted tokens therefore require 31 decode calls. Generation continues after an EOG token to keep the performance work fixed. Accuracy uses only text before the first EOG. Raw output, EOG index and all token IDs are retained.

The library API is staged in `src/llama-ext.h`:

```cpp
llama_rocketkv_params params;
params.prompt_tokens = prompt.size();
params.decode_tokens = 32;
params.token_budget = 256;
if (!llama_rocketkv_init(ctx, params)) {
    // The library logs the reason. Do not continue as if compression succeeded.
}
```

Call once on an empty context configured with causal flash attention and F16 K/V. Supply sequential positions and sequence ID zero. Prefill chunks may be smaller than the observation window, but a call must not cross the declared prefill/decode boundary. Declare the maximum decode token count before prefill. Create a fresh context for each request.

Ordinary `llama-completion`, `llama-bench` and library contexts do not enable RocketKV. `--mode full` also leaves the normal attention/cache graph intact. If the entire declared request fits the RocketKV token budget, the RocketKV configuration deliberately uses the normal dense path.

## Architecture and invariants

`src/llama-rocketkv.{h,cpp}` owns auxiliary tensors in the same backend buffer type as the real cache. `llama-context` owns its lifetime, validates the single-turn contract, reserves both final-prefill and decode graphs, and rejects unsupported state serialization. `llama-graph` routes only the standard Llama KV attention path through RocketKV when enabled.

During prefill, ordinary dense causal attention remains unchanged. A small GPU ring keeps the last observation queries for each layer. At the declared final prompt token, chronological observation queries score all prompt keys, with the correct observation-row causal mask and attention scale. Probabilities are summed over observation rows and query heads sharing a KV head. Max pooling is applied along the earlier-token sequence. Selected earlier indices are sorted chronologically, and the complete observation window is appended. Gathered K/V are materialized before either cache is overwritten. K values already contain RoPE from their original positions; they are copied, never re-rotated. Per-layer/per-KV-head retained original indices are also stored.

Compression uses the official reference's adaptive budget decomposition:

```text
N = prompt_tokens + decode_tokens
C = max(1, N / token_budget)
alpha = min(0.2 + 0.06 * log2(C), 0.8)
capacity = floor(N / C**alpha)
prompt_kept = capacity - decode_tokens
C_hsa = max(1, capacity / token_budget)
page = min(floor(C_hsa), ceil(sqrt(C_hsa)))
r = clamp(round_to_even(head_dim * page / C_hsa), 1, head_dim)
k = min(prompt_kept, round_to_even(token_budget / 2))
```

The token budget is a bandwidth proxy, not the number of stored KV tokens. Stored capacity, retained prompt length, page size, `r` and exact-attention `k` are all reported separately.

Decode writes each new K/V at `prompt_kept + original_position - prompt_tokens`. The logical llama.cpp cell positions continue to represent the original sequence. Per-page minima and maxima are constructed once after compaction, then only the page receiving the new key is updated. The first token in a new page replaces the neutral metadata; partial pages never include unused values.

Query-dimension importance is the group sum of absolute query values. The signed group sum chooses Kmax or Kmin. Approximate dot products use the original signed query. The scale is `sqrt(D * selected_L1 / full_L1)` per query head, followed by softmax and group aggregation. Exactly zero queries produce finite uniform probabilities. Future and padding positions are masked before approximate softmax. Stable top-k uses descending score and ascending index on exact ties. All heads in a KV group share the selected positions.

Gather uses the original F16 K/V, and exact sparse attention uses the existing GGML flash-attention implementation. Selection, metadata, gathered values and attention output remain on the GPU. The CPU supplies positions/live counts as input scalars but never reads queries, metadata or selection indices during decode. Normal model logits are synchronized for greedy sampling, as in the dense baseline.

Existing GGML matmul, softmax, pooling, row gather/scatter, copy and flash attention are reused. `GGML_OP_ROCKETKV`, declared in `ggml-rocketkv.h`, supplies only group reductions, deterministic top-k, chronological index assembly, query group statistics, metadata updates and signed approximate page scores. CPU implementations and Metal implementations have matching primitive and full-graph tests.

Stable sorting is deliberately bounded to 8192 entries. The existing general Metal top-k does not specify the tie membership/order needed for page scores and max-pooling plateaus, so the experimental path uses an explicit stable comparator. There is no speculative kernel fusion in this MVP.

## Correctness and profiling

```sh
ctest --test-dir build -R '^test-rocketkv' --output-on-failure
build/bin/test-backend-ops -b MTL0 -o ROCKETKV
python3 -m unittest discover -s tools/rocketkv -p 'test_*.py' -v
```

The checked-in fixtures are generated by actually importing the official Python reference at revision `63637c8eb4b06b5eb83cd687723b0eeb19f48f65`, without modifying it:

```sh
python3 tests/rocketkv/generate_fixture.py --reference ../../Material/RocketKV
```

This optional regeneration needs an existing PyTorch installation; normal C++ tests do not. The generator disables bytecode writes in the reference tree. Its only algorithm hooks are stable top-k tie breaking and an explicitly documented sentinel around the official `LAST_PREFILL` exclusive end slice. It records chronological compaction before running official `rocket_attn`.

Fixtures cover MHA, MQA and GQA (including three-query-head groups), dimensions 8 and 128, page sizes 1/2/3/4, pooling, original retained indices, Kmin/Kmax, top-r, page/token scores, top-k, gather and output. An independent C++ reference also verifies Exact-TopK, incremental updates and full-budget/zero-query behavior. Additional CPU/Metal tests cover incremental partial pages, padding masks and stable ties up to 8192 entries.

Tolerance is `2e-5 * (1 + abs(reference))` for C++ floating-point reference values, `2e-4` for backend scoring and `5e-4` for backend output. Indices, F16 gather/compaction and metadata must match exactly. Invalid scalar parameters fail explicitly. Real-model smoke checks exercise observation windows spanning uneven prefill chunks and reject serialization, nonconsecutive positions and cache mutation.

The profiler instruments 14 component categories:

```text
s1_observe s1_score s1_pool s1_select s1_index s1_compact s1_metadata
s2_index   s2_metadata s2_query s2_score s2_select s2_gather s2_attention
```

The M2 supports timestamp sampling at compute-pass boundaries, not at individual dispatch boundaries. Profiling therefore creates a separate serial encoder for each named operation and disables GGML fusion for that diagnostic request. Timestamps and category IDs are retained in bounded buffers and resolved once after the request; there is no per-token profiling readback or extra wait. GPU-to-CPU clock calibration is sampled at request boundaries. An allocation, capacity or timestamp error fails profiling instead of reporting zero as success.

`stage1_gpu_ms` includes observation-query maintenance, scoring, pooling, index selection, compaction and initial metadata. `stage2_selection_metadata_gather_gpu_ms` includes query selection, signed scoring, top-k, K/V writes, metadata updates, index handling and gather. `sparse_attention_gpu_ms` is the existing exact flash-attention operation over selected values. All per-component call counts are saved; the harness requires one metadata update and one sparse-attention call per layer and decode step.

Normal `tpot_ms` is the sum of `llama_decode` plus greedy sampling/logit synchronization divided by decode steps. It excludes detokenization and memory monitoring. Engine TTFT begins immediately before prefill and ends when the first sampled token is available; request TTFT also includes context setup. One-time model loading, tokenization, Python orchestration and process lifetime are recorded separately or explicitly excluded. Profiling-run wall times must not be mixed with uninstrumented performance results.

## Bounded evaluation and storage

```sh
python3 tools/rocketkv/run_mvp.py --dry-run
python3 tools/rocketkv/run_mvp.py --output results/rocketkv-mvp-recheck
# Resume only unchanged source/model runs:
python3 tools/rocketkv/run_mvp.py --output results/rocketkv-mvp-recheck --resume
```

The harness runs six uninstrumented performance configurations, four separate RocketKV profile configurations, one Q8-KV sanity comparison, three paired short prompts and three paired 4K passkeys. Performance and profile configurations each have one warm-up and three measured repetitions. Only the 3B model, 2K/4K prompt sizes, budgets 256/512 and 32 emitted tokens are used.

The completed study lives under `results/rocketkv-mvp/`: raw JSON and logs per configuration, exact prompts/tokens, commands, model/source hashes, machine state before/after each process, a CSV performance table and JSON summaries. The commands above use a new output directory to preserve those measurements. The report is `REPORT.md`.

On macOS, `task_info` samples resident size and physical footprint at prefill-chunk/decode-token boundaries. These are explicitly sampled peaks, not an exact GPU high-water mark. Tensor allocations are also reported. Swap usage, OS memory pressure and thermal state are recorded without changing system settings. Existing swap occupancy is not assumed to have been caused by the experiment.

This MVP retains the full preallocated `n_ctx` K/V buffer, the logical cell table and the original-size scheduler reservation. It compacts the active values into the front of that buffer; it does not free the tail or claim a larger end-to-end maximum context. Auxiliary observation queries, retained positions and metadata add storage. Prefill and decode sampled memory are separate. Compressed-capacity allocation and releasing/re-reserving compute buffers are separate future milestones.

## Deferred evaluation, not executed

```sh
python3 tools/rocketkv/prepare_full.py \
  --model models/Meta-Llama-3.1-8B-Instruct-Q4_K_M.gguf \
  --output future-full-evaluation.json
```

This prints/prepares an 8B configuration only. It never downloads a model or executes a run. Entries exceeding the current 8192-token bound are marked `requires_implementation_extension`; they are not represented as working 16K/32K commands. Memory budgeting and explicit authorization are prerequisites for those runs.

LongBench/RULER prediction adapters accept locally prepared JSONL. They default to a three-example dry run, preserve original scoring fields, and refuse to silently drop LongBench context:

```sh
python3 tools/rocketkv/eval_jsonl.py \
  --kind longbench --input prepared-longbench.jsonl --output predictions.jsonl \
  --model models/Meta-Llama-3.1-8B-Instruct-Q4_K_M.gguf --mode rocket --budget 1024

python3 tools/rocketkv/eval_jsonl.py \
  --kind ruler --input prepared-ruler.jsonl --output predictions.jsonl \
  --model models/Meta-Llama-3.1-8B-Instruct-Q4_K_M.gguf --mode rocket --budget 1024
```

Use the official task-specific prompt in a `prompt` field, or pass a task template through `--template`. Only an explicit `--execute` runs inference; `--limit 0` requests all examples. Official dataset acquisition and official benchmark scoring are external. No LongBench/RULER score is claimed here.

Unsupported in this MVP: concurrent/multiple sequences, prompt cache reuse, state serialization, context shifting, beam/speculative decoding, multi-turn compression, non-Llama architectures, quantized RocketKV K/V, non-CPU/non-Metal caches, training, and context bounds above 8192. Invalid configurations are rejected. No commits, pushes or upstream submissions are performed by the experiment.
