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

The score kernel retains the original per-page computation. Separate-prepass, threadgroup-sharing and packed-page scoring experiments were rejected because they did not demonstrate clear, repeatable end-to-end gains.

The grouped-query-head scoring experiment was also rejected: it increased the measured score-kernel time without a clear end-to-end gain.

The current KV-write candidate flattens already contiguous decode K/V tensors with reshape views before the existing cache row writes. Noncontiguous tensors keep the original materializing copy. The cache layout, stored values and attention selection are unchanged; retention requires measured end-to-end benefit.

The page-aware top-k experiment uses identical grouped token probabilities within each live page. It stably selects `ceil(k/page)+1` pages (bounded by the page count), then expands them in score order while skipping the unused tail of the one partial live page. This must produce exactly the original descending-score/ascending-token selection; probabilities themselves are unchanged. Page-size-one requests use the original token top-k. Page indices, live count, and expansion stay on the GPU.

Compact page logits with a multiplicity-weighted softmax passed the correctness checks but were rejected after two valid 8K rounds: pooled throughput was 0.969x the retained version despite lower instrumented score time. Both rounds remain in the experiment log. The retained path still normalizes token-level scores.

### Mandatory two-stage algorithm

The active research scope permits only implementation-equivalent changes to the original RocketKV algorithm. Stage 1 compression and Stage 2 hybrid sparse attention both remain enabled for active RocketKV requests. Stage-removing variants, alternate algorithms, new automatic bypass policies, and changing budgets or selection semantics to obtain speedups are out of scope. The existing full-budget fast path is unchanged.

The earlier Stage-1-only experiment was stopped and withdrawn on October 6, 2026. Its API, command-line option, dense-decode branch and live-mask implementation have been removed. Historical evidence is retained as cancelled/out-of-scope data and is not eligible to replace the best two-stage implementation. The RULER runner rejects non-hybrid experiment variants rather than resuming them.

The research round may stop when no further worthwhile, measurement-supported implementation optimization is identified. Preserve the best verified implementation and limitations; do not claim global optimality or keep running unsuccessful experiments merely to continue.

Decode gathers original F16 K/V together into a single native-type output, without F16-to-F32 intermediates or casts back to F16. Exact sparse attention uses the existing GGML flash-attention implementation. Selection, metadata, gathered values and attention output remain on the GPU. The CPU supplies positions/live counts as input scalars but never reads selection indices during decode. Normal model logits are synchronized for greedy sampling, as in the dense baseline. A mask-indexed attention experiment was rejected because mask/index processing and scattered attention did not consistently outperform the paired gather.

The Metal gather uses four-element copies only when the dimension, row strides, and input/output buffer offsets are suitably aligned. Odd dimensions and unaligned views retain the scalar kernel. Both paths preserve the native F16/F32 values and selection order exactly.

Existing GGML matmul, softmax, pooling, row gather/scatter, copy and flash attention are reused. `GGML_OP_ROCKETKV`, declared in `ggml-rocketkv.h`, supplies group reductions, deterministic top-k, chronological index assembly, query group statistics, metadata updates, signed approximate page scores, and paired native-type K/V gather. CPU implementations and Metal implementations have matching primitive and full-graph tests.

Stable sorting supports up to 32768 entries. Metal retains the original single-block path through 8192 entries and uses sorted blocks plus GPU merge passes above that size. Merge scratch storage is part of the backend allocation, not a per-token allocation or CPU readback. The existing general Metal top-k does not specify the tie membership/order needed for page scores and max-pooling plateaus, so the experimental path uses an explicit stable comparator.

For rows above 512 entries with at most 256 selected indices, Metal sorts 256-entry blocks and merges them on the GPU. Other shapes use the original single-block path through 8192 entries or the larger-block merge path. All paths preserve descending scores and ascending original indices on exact ties. The score-caching experiment was rejected after the three-context comparison showed no consistent throughput gain.

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

For a separate scheduling diagnostic, `GGML_METAL_GRAPH_DEBUG=1` enables existing Metal node/concurrency logs in `llama-rocketkv`. These runs are marked instrumented and rejected by the performance harness. Do not compare their wall times with normal inference.

`--mode full --profile` starts timestamps only after prefill and the first sampled token, and records every decode `FLASH_ATTN_EXT` operation. Its `dense_attention_gpu_ms` is separate from RocketKV's sparse-attention field; `gpu_profile_scope` states which phase is captured. The existing result structure and 14 component slots are unchanged; dense capture uses the attention slot and requires exactly `layers * decode_steps` calls. It is a diagnostic, not a source of benchmark throughput or a direct subtraction from uninstrumented wall time.

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

This prints/prepares an 8B configuration only. It never downloads a model or executes a run. RocketKV entries whose adaptive compressed capacity exceeds 8192 are marked `requires_implementation_extension`, even when their prompt fits the 32768-token limit. Memory budgeting and explicit authorization are prerequisites for those 8B runs.

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

## RULER autoresearch

Prepared prompts now use `--raw-prompt`: the dataset's template and answer prefix are passed unchanged, without a second chat wrapper. The JSONL adapter honors each record's `tokens_to_generate`. Ordinary `--prompt` and `--prompt-file` calls retain their existing chat-template behavior.

```sh
python3 tools/rocketkv/run_ruler.py \
  --experiment candidate-8k --output results/rocketkv-autoresearch/candidate-8k \
  --samples 1 --warmup 1 --repetitions 3 --budget 512
```

This runs a fixed screening cohort: the first existing record from each of `niah_single_1`, `niah_multikey_1`, `vt`, and `qa_1` at 2K/4K/8K. The user deferred 16K/32K on October 5, 2026 because of machine resource contention; longer lengths remain available only by explicit `--contexts` selection after reauthorization. Historical five-length baseline results remain unchanged. This is not a full RULER quality result. Use `--samples 0` for all locally prepared records. The existing dataset is never regenerated, shortened, or retargeted. Its length is prompt tokens plus the task-specific generation allowance. Token IDs, model/data/runtime hashes, source patch, raw predictions/timings, machine state, failures, and `results.tsv` are retained. Resume requires identical source, model, executable, libraries, and configuration.

New experiment directories also preserve the exact `run_ruler.py` driver, which is not included in a tracked-file Git diff while untracked. Earlier experiments still retain executable/library hashes, raw inputs, and per-process commands for replay.

For independent small-sample quality confirmation, use `--samples 5 --sample-offset 1 --warmup 0 --repetitions 1` in a new experiment directory. This selects the next five existing records per task and length, without reusing the screening inputs or choosing examples based on model outputs. It is a holdout check, not a full-suite accuracy claim or a repeated timing comparison.

The runner compares realtime and monotonic process durations. A discrepancy above five seconds or two percent of active duration, whichever is larger, invalidates timing because system sleep or a clock discontinuity may have contaminated the engine timer. It preserves the invalid artifacts and retries the identical case once; a second invalid attempt remains a reported failure and is excluded from throughput. Sleep settings are not changed. Small interruptions below this detection threshold remain a limitation.

Use `--reference-binary path/to/archived/llama-rocketkv` to compare the previous RocketKV implementation in the same run, with its original shared libraries alongside the executable. The runner fingerprints those files, sets the library search path for reference processes, and rotates Full KV/candidate/reference order between inputs. All three receive identical model, prompts, budget, generation, warm-up, and repetition settings. Reference throughput, quality, and candidate/reference speedup are reported separately; incomplete or invalid reference measurements cannot establish a successful comparison.

`--contexts 8192 --tasks niah_single_1` is a bounded first screen on the same predetermined input used in earlier candidate screens. Task subsets are recorded in provenance and the experiment ledger; a one-task result cannot replace the four-task 2K/4K/8K decision matrix or the independent quality holdout.

Full and Rocket modes alternate order between cases and share F16 KV, all GPU layers, greedy fixed-length decoding, 512-token prefill chunks, and four CPU threads. Throughput is total decode steps divided by their measured total time. Prefill throughput includes the first sample and is labeled accordingly. Request latency excludes model loading; per-process wall latency includes it. KV allocation, total context allocation, and sampled prefill/decode RSS are distinct metrics, not claims of an exact GPU high-water mark. Scores call the existing official RULER metric functions and equally average the four task scores.

The frozen original RocketKV engine rejects prompts above 8192; those baseline entries are explicit failures, not zero-throughput measurements. The subsequent long-context extension supports prompts through 32768 with unchanged budget decomposition and a compressed-capacity limit of 8192. The transport repair alone changed no inference kernel.

### Current measured checkpoint

This research round is closed under the user's meaningful-optimization stopping condition. No further sufficiently justified implementation-equivalent experiment is proposed in the tested scope; this is not proof of a global optimum. The complete two-stage algorithm is retained. The stage-removing branch, initializer, option and supporting mask code are removed, and the experiment runner rejects non-hybrid variants. Historical withdrawn experiments are excluded from best-version decisions.

As of October 6, 2026, the retained best is `kv-write-views`; `results/rocketkv-autoresearch/best.json` points to its exact runtime, source patch and matrix. Its measured Full KV/RocketKV decode rates are 34.229/30.459 tok/s at 2K, 26.117/27.684 at 4K, and 25.355/27.061 at 8K. These are small-cohort results on a shared machine, not an optimum or a stable win in every future run. The 4K aggregate crossover is driven mainly by one task; three other 4K tasks still trail Full KV. The prior tiled version's independent five-record-per-task quality check has equal Full/Rocket macro scores of 84/83/80 at 2K/4K/8K; one 4K variable-tracking loss and one gain cancel, so this is not a lossless-quality claim. The screening cohort still scores 75/70 at 8K.

`results/rocketkv-autoresearch/restore-two-stage/final-validation.json` records the restored build's actual Stage 1 and Stage 2 execution at 2K/4K/8K, unchanged selection parameters, and 128-token equality against the retained measured version on the fixed NIAH inputs. It also records active CPU compression/lifecycle parity and rejection of removed interfaces. `algorithm-source-parity.json` independently reconstructs the retained source snapshot and verifies matching nonblank source lines in eight core algorithm files. These final runs validate restoration; their instrumented timings are not new performance claims.

The page-aware top-k matrix completed after nine sleep-invalid 8K processes were rerun with a temporary `caffeinate -is` assertion while on AC power. The assertion lasted only for the benchmark child process and was verified released afterward; no persistent power settings or display wake were changed. All 12 screening outputs match the previous best token-for-token. Original valid measurements and both invalid attempts remain archived, with wrapper provenance in `page-topk-matrix/resume-awake-policy.json`. Because the 8K aggregate combines original and resumed sessions, it is not a noise-free effect estimate. Sleep-invalid timings are never repaired by subtracting estimated sleep time.

The pure-operation scheduling-allowlist experiment was rejected: existing fixture graphs and a real 8K model trace showed the same encoded-node and concurrency counts as the retained version. The original graph-reordering rules are restored; no scheduling speedup is claimed.

The original frozen 32K Full KV timings also include two sleep-interrupted processes, identified in `results/rocketkv-autoresearch/timing-clock-audit.json`. Their raw evidence remains frozen, but those throughput values must not support a 32K speedup claim. Long-context reruns remain deferred by the user.

Unsupported: concurrent/multiple sequences, prompt cache reuse, state serialization, context shifting, beam/speculative decoding, multi-turn compression, non-Llama architectures, quantized RocketKV K/V, non-CPU/non-Metal caches, training, prompts above 32768, and compressed capacities above 8192. Invalid configurations are rejected. No commits, pushes or upstream submissions are performed by the experiment.
