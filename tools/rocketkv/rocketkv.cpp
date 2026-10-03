// RocketKV research benchmark. See docs/rocketkv/NOTICE.
#include "llama.h"
#include "llama-ext.h"
#include "llama-cpp.h"
#include "ggml-backend.h"
#include "json.h"
#ifdef ROCKETKV_METAL
#include "ggml-metal.h"
#include "llama-context.h"
#endif

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <ctime>
#include <fstream>
#include <iostream>
#include <numeric>
#include <stdexcept>
#include <string>
#include <vector>

#ifdef __APPLE__
#include <mach/mach.h>
#endif

using clock_type = std::chrono::steady_clock;
using json = common_json;

struct options {
    std::string model, output, prompt, prompt_file, passkey;
    std::string mode = "full";
    std::string kv = "f16";
    int prompt_tokens = 2048, generate = 32, budget = 512, window = 32, pool = 63;
    int repetitions = 3, warmup = 1, ubatch = 512, gpu_layers = 99;
    double depth = 0.5;
    bool lifecycle = false;
    bool profile = false;
};

static double ms(clock_type::time_point start) {
    return std::chrono::duration<double, std::milli>(clock_type::now() - start).count();
}

static options parse(int argc, char ** argv) {
    options opt;
    for (int i = 1; i < argc; ++i) {
        const std::string key = argv[i];
        if (key == "--help" || key == "-h") {
            std::puts("llama-rocketkv -m model.gguf [--mode full|rocket] [--kv f16|q8_0]\n"
                      "  --prompt-tokens 2048 --generate 32 --budget 256 --window 32 --pool 63\n"
                      "  --repetitions 3 --warmup 1 --ubatch 512 --gpu-layers 99\n"
                      "  --prompt TEXT | --prompt-file FILE | --passkey 74219 --depth 0.5\n"
                      "  --output result.json --validate-lifecycle --profile\n"
                      "Generation is greedy and fixed-length, including tokens after the first EOG.\n"
                      "RocketKV is an experimental single-turn Llama/F16-KV path, not a server mode.");
            std::exit(0);
        }
        if (key == "--validate-lifecycle") {
            opt.lifecycle = true;
            continue;
        }
        if (key == "--profile") {
            opt.profile = true;
            continue;
        }
        if (++i == argc) {
            throw std::invalid_argument("missing value for " + key);
        }
        const std::string value = argv[i];
        auto integer = [&] {
            size_t end = 0;
            const int v = std::stoi(value, &end);
            if (end != value.size()) {
                throw std::invalid_argument("invalid integer for " + key);
            }
            return v;
        };
        if (key == "-m" || key == "--model") { opt.model = value; }
        else if (key == "--output") { opt.output = value; }
        else if (key == "--mode") { opt.mode = value; }
        else if (key == "--kv") { opt.kv = value; }
        else if (key == "--prompt") { opt.prompt = value; }
        else if (key == "--prompt-file") { opt.prompt_file = value; }
        else if (key == "--passkey") { opt.passkey = value; }
        else if (key == "--prompt-tokens") { opt.prompt_tokens = integer(); }
        else if (key == "--generate") { opt.generate = integer(); }
        else if (key == "--budget") { opt.budget = integer(); }
        else if (key == "--window") { opt.window = integer(); }
        else if (key == "--pool") { opt.pool = integer(); }
        else if (key == "--repetitions") { opt.repetitions = integer(); }
        else if (key == "--warmup") { opt.warmup = integer(); }
        else if (key == "--ubatch") { opt.ubatch = integer(); }
        else if (key == "--gpu-layers") { opt.gpu_layers = integer(); }
        else if (key == "--depth") {
            size_t end = 0;
            opt.depth = std::stod(value, &end);
            if (end != value.size() || !std::isfinite(opt.depth) || opt.depth < 0 || opt.depth > 1) {
                throw std::invalid_argument("depth must be between zero and one");
            }
        } else {
            throw std::invalid_argument("unknown option " + key);
        }
    }
    if (opt.model.empty() || (opt.mode != "full" && opt.mode != "rocket") || (opt.kv != "f16" && opt.kv != "q8_0") ||
        opt.prompt_tokens < 64 || opt.prompt_tokens > 8192 || opt.generate < 2 || opt.generate > 512 ||
        opt.repetitions < 1 || opt.repetitions > 100 || opt.warmup < 0 || opt.warmup > 10 ||
        opt.ubatch < 1 || opt.ubatch > 2048 || opt.gpu_layers < 0 || opt.gpu_layers > 999 ||
        (!opt.prompt.empty() && !opt.prompt_file.empty()) || (opt.mode == "rocket" && opt.kv != "f16") ||
        (opt.profile && opt.gpu_layers == 0)) {
        throw std::invalid_argument("invalid configuration; use --help");
    }
    return opt;
}

static std::vector<llama_token> tokenize(const llama_vocab * vocab, const std::string & text) {
    int n = llama_tokenize(vocab, text.data(), text.size(), nullptr, 0, false, true);
    if (n >= 0) {
        throw std::runtime_error("expected a nonempty prompt");
    }
    std::vector<llama_token> result(-n);
    n = llama_tokenize(vocab, text.data(), text.size(), result.data(), result.size(), false, true);
    if (n < 0) {
        throw std::runtime_error("tokenization failed");
    }
    result.resize(n);
    return result;
}

static std::string piece(const llama_vocab * vocab, llama_token token) {
    char local[256];
    const int n = llama_token_to_piece(vocab, token, local, sizeof(local), 0, true);
    if (n >= 0) {
        return std::string(local, n);
    }
    std::string large(-n, '\0');
    const int count = llama_token_to_piece(vocab, token, &large[0], large.size(), 0, true);
    if (count < 0) {
        throw std::runtime_error("token detokenization failed");
    }
    large.resize(count);
    return large;
}

static std::vector<llama_token> make_prompt(const options & opt, const llama_vocab * vocab) {
    const std::string header = "<|begin_of_text|><|start_header_id|>user<|end_header_id|>\n\n";
    const std::string footer = "<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n";
    if (!opt.prompt.empty() || !opt.prompt_file.empty()) {
        std::string text = opt.prompt;
        if (!opt.prompt_file.empty()) {
            std::ifstream file(opt.prompt_file, std::ios::ate);
            if (!file) {
                throw std::runtime_error("cannot open prompt file");
            }
            if (file.tellg() < 0 || file.tellg() > 4*1024*1024) {
                throw std::runtime_error("prompt file exceeds the 4 MiB input limit");
            }
            file.seekg(0);
            text.assign(std::istreambuf_iterator<char>(file), {});
        }
        return tokenize(vocab, header + text + footer);
    }
    auto prefix = tokenize(vocab, header + "Read this archive carefully.\n");
    auto suffix = tokenize(vocab, opt.passkey.empty() ?
        "\nSummarize the archive in one sentence." + footer :
        "\nWhat is the secret access code? Reply with only the code." + footer);
    auto filler = tokenize(vocab, "The archive records an ordinary day. The weather was mild and the town was quiet. No important event was recorded.\n");
    std::vector<llama_token> needle;
    if (!opt.passkey.empty()) {
        if (opt.passkey.size() != 5 || opt.passkey.find_first_not_of("0123456789") != std::string::npos) {
            throw std::invalid_argument("passkey must contain exactly five digits");
        }
        needle = tokenize(vocab, "\nIMPORTANT: The secret access code is " + opt.passkey + ". Remember this code.\n");
    }
    const int padding = opt.prompt_tokens - prefix.size() - suffix.size() - needle.size();
    if (padding < 0) {
        throw std::invalid_argument("prompt is too short for the template and needle");
    }
    const int needle_at = int(padding*opt.depth);
    for (int i = 0; i <= padding; ++i) {
        if (i == needle_at) {
            prefix.insert(prefix.end(), needle.begin(), needle.end());
        }
        if (i < padding) {
            prefix.push_back(filler[i % filler.size()]);
        }
    }
    prefix.insert(prefix.end(), suffix.begin(), suffix.end());
    return prefix;
}

struct memory_sample {
    uint64_t rss = 0;
    uint64_t footprint = 0;
};

static memory_sample sample_memory() {
#ifdef __APPLE__
    task_vm_info_data_t info{};
    mach_msg_type_number_t count = TASK_VM_INFO_COUNT;
    if (task_info(mach_task_self(), TASK_VM_INFO, (task_info_t) &info, &count) != KERN_SUCCESS) {
        throw std::runtime_error("cannot sample task VM information");
    }
    return {info.resident_size, info.phys_footprint};
#else
    return {};
#endif
}

static void update_peak(memory_sample & peak) {
    const auto current = sample_memory();
    peak.rss = std::max(peak.rss, current.rss);
    peak.footprint = std::max(peak.footprint, current.footprint);
}

struct batch_owner {
    llama_batch batch;
    explicit batch_owner(int n) : batch(llama_batch_init(n, 0, 1)) {}
    ~batch_owner() { llama_batch_free(batch); }
};

static json run(const options & opt, llama_model * model, const std::vector<llama_token> & prompt, int repetition) {
    const auto request_start = clock_type::now();
    auto cp = llama_context_default_params();
    cp.n_ctx = prompt.size() + opt.generate;
    cp.n_batch = opt.ubatch;
    cp.n_ubatch = opt.ubatch;
    cp.n_seq_max = 1;
    cp.n_threads = cp.n_threads_batch = 4;
    cp.flash_attn_type = LLAMA_FLASH_ATTN_TYPE_ENABLED;
    cp.type_k = cp.type_v = opt.kv == "f16" ? GGML_TYPE_F16 : GGML_TYPE_Q8_0;
    cp.no_perf = false;
    llama_context_ptr ctx(llama_init_from_model(model, cp));
    if (!ctx) {
        throw std::runtime_error("cannot create inference context");
    }
    if (opt.mode == "rocket") {
        llama_rocketkv_params rp;
        rp.prompt_tokens = prompt.size();
        rp.decode_tokens = opt.generate;
        rp.token_budget = opt.budget;
        rp.observation_window = opt.window;
        rp.pooling_kernel = opt.pool;
        if (!llama_rocketkv_init(ctx.get(), rp)) {
            throw std::runtime_error("RocketKV initialization failed");
        }
    }
    const double setup_ms = ms(request_start);
    ggml_backend_t profile_backend = nullptr;
    if (opt.profile) {
#ifdef ROCKETKV_METAL
        auto * sched = ctx->get_sched();
        for (int i = 0; i < ggml_backend_sched_get_n_backends(sched); ++i) {
            auto * backend = ggml_backend_sched_get_backend(sched, i);
            if (ggml_backend_is_metal(backend)) {
                profile_backend = backend;
                break;
            }
        }
        if (!profile_backend || !ggml_backend_metal_rocketkv_profile_begin(profile_backend)) {
            throw std::runtime_error("Metal timestamp profiling is unavailable");
        }
#else
        throw std::runtime_error("this build has no Metal profiler");
#endif
    }
    llama_sampler_ptr sampler(llama_sampler_init_greedy());
    batch_owner owner(opt.ubatch);
    auto & batch = owner.batch;
    memory_sample prefill_peak{}, decode_peak{};
    const auto engine_start = clock_type::now();
    for (size_t start = 0; start < prompt.size(); start += opt.ubatch) {
        batch.n_tokens = std::min<size_t>(opt.ubatch, prompt.size() - start);
        for (int i = 0; i < batch.n_tokens; ++i) {
            batch.token[i] = prompt[start + i];
            batch.pos[i] = start + i;
            batch.n_seq_id[i] = 1;
            batch.seq_id[i][0] = 0;
            batch.logits[i] = i == batch.n_tokens - 1;
        }
        if (llama_decode(ctx.get(), batch) != 0) {
            throw std::runtime_error("prefill failed");
        }
        update_peak(prefill_peak);
    }
    llama_token next = llama_sampler_sample(sampler.get(), ctx.get(), -1);
    const double ttft = ms(engine_start);
    const double request_ttft = ms(request_start);
    update_peak(prefill_peak);
    update_peak(decode_peak);
    std::vector<int> generated{next};
    std::vector<float> decode_times;
    const llama_vocab * vocab = llama_model_get_vocab(model);
    std::string output = piece(vocab, next);
    std::string answer = llama_vocab_is_eog(vocab, next) ? "" : output;
    int first_eog = llama_vocab_is_eog(vocab, next) ? 0 : -1;
    const auto decode_start = clock_type::now();
    for (int i = 1; i < opt.generate; ++i) {
        batch.n_tokens = 1;
        batch.token[0] = next;
        batch.pos[0] = prompt.size() + i - 1;
        batch.n_seq_id[0] = 1;
        batch.seq_id[0][0] = 0;
        batch.logits[0] = 1;
        const auto step = clock_type::now();
        if (llama_decode(ctx.get(), batch) != 0) {
            throw std::runtime_error("decode failed");
        }
        next = llama_sampler_sample(sampler.get(), ctx.get(), -1);
        decode_times.push_back(ms(step));
        generated.push_back(next);
        const auto text = piece(vocab, next);
        output += text;
        if (first_eog < 0) {
            if (llama_vocab_is_eog(vocab, next)) {
                first_eog = i;
            } else {
                answer += text;
            }
        }
        update_peak(decode_peak);
    }
    const double decode_wall_ms = ms(decode_start), request_ms = ms(request_start);
    const double decode_ms = std::accumulate(decode_times.begin(), decode_times.end(), 0.0);
    const auto ri = llama_rocketkv_get_info(ctx.get());
    json component_ms = json::object(), component_calls = json::object();
    double stage1_ms = 0, selection_ms = 0, attention_ms = 0;
    if (opt.profile) {
#ifdef ROCKETKV_METAL
        ggml_metal_rocketkv_profile profile{};
        if (!ggml_backend_metal_rocketkv_profile_end(profile_backend, &profile)) {
            throw std::runtime_error("Metal timestamp profiling failed");
        }
        for (const auto & entry : profile.entries) {
            component_ms[entry.name] = entry.gpu_ms;
            component_calls[entry.name] = entry.calls;
            const std::string name = entry.name;
            if (name.find("s1_") == 0) {
                stage1_ms += entry.gpu_ms;
            } else if (name == "s2_attention") {
                attention_ms += entry.gpu_ms;
            } else {
                selection_ms += entry.gpu_ms;
            }
            if (ri.active && (profile.entries[8].calls != uint64_t(llama_model_n_layer(model))*(opt.generate - 1) ||
                              profile.entries[13].calls != profile.entries[8].calls)) {
                throw std::runtime_error("not all RocketKV layers were profiled on Metal");
            }
        }
#endif
    }
    uint64_t model_bytes = 0, context_bytes = 0, compute_bytes = 0;
    for (const auto & entry : llama_get_memory_breakdown(ctx.get())) {
        model_bytes += entry.second.model;
        context_bytes += entry.second.context;
        compute_bytes += entry.second.compute;
    }
    if (opt.lifecycle && opt.mode == "rocket") {
        if (llama_state_get_size(ctx.get()) != 0) {
            throw std::runtime_error("unsupported state serialization was accepted");
        }
        batch.pos[0] += 7;
        if (llama_decode(ctx.get(), batch) >= 0) {
            throw std::runtime_error("non-contiguous decode was accepted");
        }
        llama_memory_seq_rm(llama_get_memory(ctx.get()), 0, 0, 1);
        batch.pos[0] = prompt.size() + opt.generate - 1;
        if (llama_decode(ctx.get(), batch) >= 0) {
            throw std::runtime_error("mutated cache was accepted");
        }
    }
    json result = json::object();
    result["repetition"] = repetition;
    result["warmup"] = repetition < 0;
    result["mode"] = opt.mode;
    result["kv_type"] = opt.kv;
    result["prompt_tokens"] = prompt.size();
    result["n_ctx_allocated"] = llama_n_ctx(ctx.get());
    result["generated_tokens"] = opt.generate;
    result["layers"] = llama_model_n_layer(model);
    result["decode_steps"] = opt.generate - 1;
    result["token_budget"] = opt.mode == "rocket" ? opt.budget : 0;
    result["rocket_active"] = ri.active;
    result["prompt_kept"] = ri.prompt_kept;
    result["compressed_capacity"] = ri.capacity;
    result["page_size"] = ri.page_size;
    result["query_dims"] = ri.query_dims;
    result["attention_tokens"] = ri.attention_tokens;
    result["engine_ttft_ms"] = ttft;
    result["instrumented"] = opt.profile;
    result["component_gpu_ms"] = std::move(component_ms);
    result["component_calls"] = std::move(component_calls);
    result["stage1_gpu_ms"] = opt.profile ? json(stage1_ms) : json(nullptr);
    result["stage2_selection_metadata_gather_gpu_ms"] = opt.profile ? json(selection_ms) : json(nullptr);
    result["sparse_attention_gpu_ms"] = opt.profile ? json(attention_ms) : json(nullptr);
    result["request_ttft_ms"] = request_ttft;
    result["context_setup_ms"] = setup_ms;
    result["decode_total_ms"] = decode_ms;
    result["decode_wall_ms"] = decode_wall_ms;
    result["tpot_ms"] = decode_ms/(opt.generate - 1);
    result["engine_request_ms"] = ttft + decode_ms;
    result["request_ms"] = request_ms;
    result["decode_step_ms"] = decode_times;
    result["prefill_peak_rss_sampled_bytes"] = prefill_peak.rss;
    result["prefill_peak_footprint_sampled_bytes"] = prefill_peak.footprint;
    result["decode_peak_rss_sampled_bytes"] = decode_peak.rss;
    result["decode_peak_footprint_sampled_bytes"] = decode_peak.footprint;
    result["decode_allocated_tensor_bytes"] = model_bytes + context_bytes + compute_bytes;
    result["model_tensor_bytes"] = model_bytes;
    result["context_tensor_bytes"] = context_bytes;
    result["compute_tensor_bytes"] = compute_bytes;
    result["rocket_full_kv_allocation_bytes"] = ri.allocated_kv_bytes;
    result["rocket_active_kv_capacity_bytes"] = ri.active_kv_bytes;
    result["rocket_auxiliary_bytes"] = ri.auxiliary_bytes;
    result["allocated_kv_bytes"] = opt.mode == "rocket" ? ri.allocated_kv_bytes : context_bytes;
    result["tokens"] = generated;
    result["output"] = output;
    result["answer_before_eog"] = answer;
    result["first_eog_index"] = first_eog;
    result["passkey"] = opt.passkey;
    result["passkey_depth"] = opt.depth;
    result["passkey_pass"] = opt.passkey.empty() ? json(nullptr) : json(answer.find(opt.passkey) != std::string::npos);
    result["lifecycle_checks"] = opt.lifecycle && opt.mode == "rocket";
    return result;
}

int main(int argc, char ** argv) {
    try {
        const auto opt = parse(argc, argv);
        const auto logger = [](ggml_log_level level, const char * text, void *) {
            if (level != GGML_LOG_LEVEL_DEBUG) {
                std::fputs(text, stderr);
            }
        };
        llama_log_set(logger, nullptr);
        ggml_log_set(logger, nullptr);
        ggml_backend_load_all();
        llama_backend_init();
        auto mp = llama_model_default_params();
        mp.n_gpu_layers = opt.gpu_layers;
        const auto load_start = clock_type::now();
        llama_model_ptr model(llama_model_load_from_file(opt.model.c_str(), mp));
        if (!model) {
            throw std::runtime_error("cannot load model");
        }
        const double load_ms = ms(load_start);
        const auto prompt = make_prompt(opt, llama_model_get_vocab(model.get()));
        if (prompt.size() > 8192) {
            throw std::invalid_argument("this MVP limits prompts to 8192 tokens");
        }
        json records = json::array();
        for (int i = -opt.warmup; i < opt.repetitions; ++i) {
            auto result = run(opt, model.get(), prompt, i);
            std::fprintf(stderr, "RocketKV benchmark: %s repetition %d: TTFT %.2f ms, TPOT %.2f ms\n",
                         opt.mode.c_str(), i, result["engine_ttft_ms"].get<double>(), result["tpot_ms"].get<double>());
            records.push_back(std::move(result));
        }
        json document = json::object();
        document["schema_version"] = 1;
        document["preliminary"] = true;
        document["model"] = opt.model;
        document["model_load_ms"] = load_ms;
        document["prompt_token_ids"] = prompt;
        document["ubatch"] = opt.ubatch;
        document["gpu_layers"] = opt.gpu_layers;
        document["observation_window"] = opt.window;
        document["pooling_kernel"] = opt.pool;
        document["records"] = std::move(records);
        const std::string serialized = document.dump(2) + "\n";
        if (opt.output.empty()) {
            std::cout << serialized;
        } else {
            std::ofstream file(opt.output);
            if (!file || !(file << serialized)) {
                throw std::runtime_error("cannot write result file");
            }
        }
        return 0;
    } catch (const std::exception & e) {
        std::fprintf(stderr, "llama-rocketkv: %s\n", e.what());
        return 1;
    }
}
