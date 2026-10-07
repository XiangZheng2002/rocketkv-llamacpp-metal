#!/usr/bin/env python3
"""Paired local RULER screening with immutable inputs and resumable raw results."""
import argparse
import csv
from datetime import datetime, timezone
import importlib.util
import json
import os
from pathlib import Path
import re
import statistics
import subprocess
import sys
import time

from run_mvp import ROOT, command_text, file_hash, machine_state, save_json, source_hash


CONTEXTS = (2048, 4096, 8192, 16384, 32768)
DEFAULT_CONTEXTS = (2048, 4096, 8192)
TASKS = ("niah_single_1", "niah_multikey_1", "vt", "qa_1")
FIELDS = ("commit", "experiment", "context", "full_kv_tps", "rocketkv_tps",
          "speedup", "ruler_score", "status", "notes")


def load_cases(dataset, samples, contexts, offset=0, tasks=TASKS):
    if samples < 0 or offset < 0:
        raise ValueError("sample count and offset must be nonnegative")
    manifest = json.loads((dataset / "manifest.json").read_text())
    cases = []
    for context in contexts:
        for task in tasks:
            relative = f"{context}/{task}/validation.jsonl"
            entry = next(item for item in manifest["files"] if item["path"] == relative)
            path = dataset / relative
            if file_hash(path) != entry["sha256"]:
                raise ValueError(f"dataset checksum mismatch: {path}")
            rows = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
            if len(rows) != entry["samples"] or offset >= len(rows) or (samples and len(rows) < offset + samples):
                raise ValueError(f"dataset sample count mismatch: {path}")
            for row in rows[offset:offset + samples if samples else len(rows)]:
                if (row["prompt"] != row["input"] + row["answer_prefix"] or
                        row["prompt_tokens"] + row["tokens_to_generate"] != context or
                        row["target_length"] != context):
                    raise ValueError(f"invalid prepared prompt: {path}, index {row['index']}")
                cases.append({"context": context, "task": task, "record": row})
    return manifest, cases


def measured(document, case, mode, repetitions, warmup, budget):
    rows = document["records"]
    records = [row for row in rows if not row["warmup"]]
    source = case["record"]
    if len(records) != repetitions or len(rows) != repetitions + warmup or not document["raw_prompt"]:
        raise ValueError("invalid repetition counts or benchmark prompt was re-templated")
    if len(document["prompt_token_ids"]) != source["prompt_tokens"]:
        raise ValueError("tokenized prompt does not match the frozen dataset")
    for row in rows:
        if (row["prompt_tokens"] != source["prompt_tokens"] or
                row["generated_tokens"] != source["tokens_to_generate"] or
                row["decode_steps"] != source["tokens_to_generate"] - 1 or
                len(row["tokens"]) != source["tokens_to_generate"] or
                row["mode"] != mode or row["kv_type"] != "f16" or row["instrumented"] or
                row["token_budget"] != (budget if mode == "rocket" else 0) or
                row["tpot_ms"] <= 0 or row["engine_ttft_ms"] <= 0):
            raise ValueError("run does not match the frozen configuration")
        actual_variant = row.get("rocket_variant", "full_kv" if mode == "full" else "rocketkv_hybrid")
        if actual_variant != ("full_kv" if mode == "full" else "rocketkv_hybrid"):
            raise ValueError("algorithm variant does not match the frozen configuration")
    if any(row["tokens"] != records[0]["tokens"] for row in records):
        raise ValueError("greedy generation changed across repetitions")
    return records


def timing_error(state):
    elapsed_ms = state["process_realtime_ms"]
    active_ms = state["process_wall_ms"]
    if abs(elapsed_ms - active_ms) > max(5000, active_ms * 0.02):
        return f"clock discontinuity: realtime {elapsed_ms:.0f} ms, monotonic {active_ms:.0f} ms; possible system sleep"
    return None


def load_metrics(dataset):
    path = dataset.parents[1] / "scripts/eval/synthetic/constants.py"
    spec = importlib.util.spec_from_file_location("ruler_metrics", path)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot load official RULER metrics: {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module, path


def score(metrics, task, predictions, references):
    predictions = [re.sub(r"[\x00-\x1f]", "\n", pred.strip()).strip() for pred in predictions]
    metric = metrics.string_match_part if task == "qa_1" else metrics.string_match_all
    return metric(predictions, references)


def summarize(output, cases, provenance, metrics):
    summary = []
    rows_tsv = []
    variant = provenance.get("candidate_variant", "rocketkv_hybrid")
    if variant != "rocketkv_hybrid":
        raise ValueError("only complete two-stage RocketKV experiments are supported")
    modes = ("full", "rocket", "reference") if provenance.get("reference_runtime") else ("full", "rocket")
    for context in provenance["contexts"]:
        entry = {"context": context, "modes": {}, "speedup": None}
        paired = {}
        for mode in modes:
            measurements, task_predictions, failures = [], {}, []
            for case in [case for case in cases if case["context"] == context]:
                name = f"{context}_{case['task']}_{case['record']['index']}_{mode}"
                raw = output / "raw" / f"{name}.json"
                meta = output / "raw" / f"{name}.meta.json"
                if not meta.exists():
                    continue
                status = json.loads(meta.read_text())
                if status["returncode"] != 0 or status.get("timing_error"):
                    failures.append({"id": name, "log": str(raw.with_suffix(".log")),
                                     "returncode": status["returncode"], "timing_error": status.get("timing_error")})
                    continue
                doc = json.loads(raw.read_text())
                records = measured(doc, case, "rocket" if mode == "reference" else mode,
                                   provenance["repetitions"], provenance["warmup"], provenance["budget"])
                pair_key = (case["task"], case["record"]["index"])
                if mode == "full":
                    paired[pair_key] = doc
                elif pair_key in paired:
                    full = paired[pair_key]
                    if doc["prompt_token_ids"] != full["prompt_token_ids"]:
                        raise ValueError("paired modes received different prompt tokens")
                    if records[0]["tokens"][0] != full["records"][-1]["tokens"][0]:
                        raise ValueError("RocketKV changed the prefill-produced first token")
                measurements.extend(records)
                task_predictions.setdefault(case["task"], []).append(
                    (records[0]["answer_before_eog"], case["record"]["outputs"]))
            info = {"failures": failures, "measured_requests": len(measurements),
                    "variant": "full_kv" if mode == "full" else variant if mode == "rocket" else "rocketkv_hybrid"}
            expected = sum(case["context"] == context for case in cases) * provenance["repetitions"]
            info["complete"] = len(measurements) == expected and not failures
            if measurements:
                info["decode_tps"] = 1000 * sum(row["decode_steps"] for row in measurements) / sum(
                    row["decode_total_ms"] for row in measurements)
                info["ms_per_token"] = 1000 / info["decode_tps"]
                info["prefill_tps_including_first_sample"] = 1000 * sum(row["prompt_tokens"] for row in measurements) / sum(
                    row["engine_ttft_ms"] for row in measurements)
                info["request_ms_mean"] = statistics.mean(row["request_ms"] for row in measurements)
                info["peak_allocated_kv_bytes"] = max(row["allocated_kv_bytes"] for row in measurements)
                info["peak_context_tensor_bytes"] = max(row["context_tensor_bytes"] for row in measurements)
                info["peak_decode_rss_sampled_bytes"] = max(row["decode_peak_rss_sampled_bytes"] for row in measurements)
                info["peak_prefill_rss_sampled_bytes"] = max(row["prefill_peak_rss_sampled_bytes"] for row in measurements)
                info["task_scores"] = {
                    task: score(metrics, task, [p for p, _ in pairs], [r for _, r in pairs])
                    for task, pairs in task_predictions.items()
                }
                info["ruler_score"] = statistics.mean(info["task_scores"].values())
            entry["modes"][mode] = info
        full, rocket = entry["modes"]["full"], entry["modes"]["rocket"]
        reference = entry["modes"].get("reference")
        complete = all(info["complete"] for info in entry["modes"].values())
        if complete:
            entry["speedup"] = rocket["decode_tps"] / full["decode_tps"]
            if reference:
                entry["speedup_vs_reference"] = rocket["decode_tps"] / reference["decode_tps"]
        entry["status"] = "measured" if complete else "failed" if any(
            info["failures"] for info in entry["modes"].values()) else "pending"
        summary.append(entry)
        rows_tsv.append({
            "commit": provenance["commit"], "experiment": provenance["experiment"], "context": context,
            "full_kv_tps": full.get("decode_tps", ""), "rocketkv_tps": rocket.get("decode_tps", ""),
            "speedup": entry["speedup"] if complete else "", "ruler_score": rocket.get("ruler_score", ""),
            "status": entry["status"],
            "notes": json.dumps({"full_ruler_score": full.get("ruler_score"), "samples_per_task": provenance["samples"],
                                 "tasks": provenance.get("tasks", list(TASKS)),
                                 "sample_offset": provenance.get("sample_offset", 0),
                                 "repetitions": provenance["repetitions"], "budget": provenance["budget"],
                                 "candidate_variant": variant,
                                 "reference_tps": reference.get("decode_tps") if reference else None,
                                 "reference_ruler_score": reference.get("ruler_score") if reference else None,
                                 "speedup_vs_reference": entry.get("speedup_vs_reference"),
                                 "summary": str(output / "summary.json")}, separators=(",", ":")),
        })
    save_json(output / "summary.json", {
        "screening_only": provenance["samples"] != 0 or provenance.get("sample_offset", 0) != 0,
        "candidate_variant": variant, "contexts": summary})
    with (output / "results.tsv").open("w") as file:
        writer = csv.DictWriter(file, FIELDS, delimiter="\t")
        writer.writeheader()
        writer.writerows(rows_tsv)
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, default=ROOT.parent / "RULER/datasets/rocketkv-ruler")
    parser.add_argument("--model", type=Path, default=ROOT / "models/Llama-3.2-3B-Instruct-Q4_K_M.gguf")
    parser.add_argument("--binary", type=Path, default=ROOT / "build/bin/llama-rocketkv")
    parser.add_argument("--reference-binary", type=Path, help="archived previous RocketKV executable with its own shared libraries")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--experiment", required=True)
    parser.add_argument("--samples", type=int, default=1, help="first N records per task; 0 uses all 50")
    parser.add_argument("--sample-offset", type=int, default=0, help="skip fixed records to use an independent holdout")
    parser.add_argument("--tasks", nargs="+", choices=TASKS, default=list(TASKS),
                        help="fixed task subset for targeted screening; defaults to all four tasks")
    parser.add_argument("--repetitions", type=int, default=3)
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--budget", type=int, default=512)
    parser.add_argument("--contexts", type=int, nargs="+", default=list(DEFAULT_CONTEXTS), choices=CONTEXTS,
                        help="defaults to 2K/4K/8K; deferred longer contexts require explicit selection")
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    if args.samples < 0 or args.sample_offset < 0 or not 1 <= args.repetitions <= 100 or not 0 <= args.warmup <= 10 or args.budget < 2:
        parser.error("invalid sample, repetition, warmup, or budget setting")
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    (output / "raw").mkdir(exist_ok=True)
    dataset = args.dataset.resolve()
    manifest, cases = load_cases(dataset, args.samples, args.contexts, args.sample_offset, args.tasks)
    metrics, metric_path = load_metrics(dataset)
    model_hash = file_hash(args.model)
    if model_hash != manifest["model_sha256"]:
        raise ValueError("model does not match the dataset tokenizer")
    provenance = {
        "experiment": args.experiment, "commit": command_text(["git", "-C", str(ROOT), "rev-parse", "HEAD"]),
        "source_sha256": source_hash(), "model_sha256": model_hash, "binary_sha256": file_hash(args.binary),
        "runtime_sha256": {p.name: file_hash(p) for p in sorted(args.binary.parent.glob("*.dylib")) if not p.is_symlink()},
        "dataset_manifest_sha256": file_hash(dataset / "manifest.json"), "metric_sha256": file_hash(metric_path),
        "contexts": args.contexts, "tasks": args.tasks, "samples": args.samples, "repetitions": args.repetitions,
        "warmup": args.warmup, "budget": args.budget, "ubatch": 512, "gpu_layers": 99,
        "candidate_variant": "rocketkv_hybrid",
        "sample_offset": args.sample_offset,
        "window": 32, "pool": 63, "kv": "f16", "generation": manifest["tokens_to_generate"],
        "protocol": "greedy fixed-length including post-EOG tokens; score only answer_before_eog",
        "cohort": [(case["context"], case["task"], case["record"]["index"]) for case in cases],
    }
    if args.reference_binary:
        reference = args.reference_binary.resolve()
        provenance["reference_runtime"] = {
            "binary": str(reference), "binary_sha256": file_hash(reference),
            "libraries": {p.name: file_hash(p) for p in sorted(reference.parent.glob("*.dylib")) if not p.is_symlink()},
        }
    provenance = json.loads(json.dumps(provenance))
    path = output / "provenance.json"
    if path.exists():
        if not args.resume or json.loads(path.read_text()) != provenance:
            raise ValueError("existing experiment differs or --resume was not provided")
    else:
        for artifact in (output / "source.patch", output / "run_ruler.py"):
            if artifact.exists():
                raise FileExistsError(f"refusing to overwrite source snapshot: {artifact}")
        patch = subprocess.check_output(["git", "-C", str(ROOT), "diff", "--binary", "HEAD"])
        (output / "source.patch").write_bytes(patch)
        (output / "run_ruler.py").write_bytes(Path(__file__).read_bytes())
        save_json(path, provenance)
    summarize(output, cases, provenance, metrics)
    for index, case in enumerate(cases):
        prompt = output / "raw" / f"{case['context']}_{case['task']}_{case['record']['index']}.txt"
        prompt.write_text(case["record"]["prompt"])
        modes = ["full", "rocket", "reference"] if args.reference_binary else ["full", "rocket"]
        rotation = index % len(modes)
        modes = modes[rotation:] + modes[:rotation]
        for mode in modes:
            name = f"{case['context']}_{case['task']}_{case['record']['index']}_{mode}"
            raw = output / "raw" / f"{name}.json"
            meta = raw.with_suffix(".meta.json")
            if meta.exists():
                continue
            binary = args.reference_binary if mode == "reference" else args.binary
            environment = {}
            if mode == "reference":
                variable = "DYLD_LIBRARY_PATH" if sys.platform == "darwin" else "LD_LIBRARY_PATH"
                environment[variable] = str(binary.resolve().parent)
            command = [
                str(binary.resolve()), "-m", str(args.model.resolve()), "--mode", "rocket" if mode == "reference" else mode, "--kv", "f16",
                "--prompt-file", str(prompt), "--raw-prompt", "--generate", str(case["record"]["tokens_to_generate"]),
                "--ubatch", "512", "--gpu-layers", "99", "--warmup", str(args.warmup),
                "--repetitions", str(args.repetitions), "--budget", str(args.budget), "--window", "32", "--pool", "63",
                "--output", str(raw),
            ]
            for attempt in range(2):
                state = {"command": command, "environment": environment, "before": machine_state(),
                         "started_at": datetime.now(timezone.utc).isoformat()}
                print(f"{index + 1}/{len(cases)} {name}, attempt {attempt + 1}", flush=True)
                start = time.monotonic()
                start_realtime = time.time()
                with raw.with_suffix(".log").open("w") as log:
                    result = subprocess.run(command, stdout=log, stderr=subprocess.STDOUT, env=os.environ | environment)
                state.update({"process_realtime_ms": (time.time() - start_realtime) * 1000,
                              "process_wall_ms": (time.monotonic() - start) * 1000,
                              "returncode": result.returncode, "after": machine_state()})
                state["timing_error"] = timing_error(state)
                save_json(meta, state)
                if result.returncode or not state["timing_error"]:
                    break
                print(f"INVALID TIMING {name}: {state['timing_error']}", flush=True)
                if attempt == 0:
                    invalid = output / "invalid-timing"
                    invalid.mkdir(exist_ok=True)
                    for artifact in (raw, meta, raw.with_suffix(".log")):
                        destination = invalid / artifact.name
                        if destination.exists():
                            raise FileExistsError(f"refusing to overwrite invalid timing evidence: {destination}")
                        artifact.rename(destination)
                    print(f"Retrying the identical input and configuration once: {name}", flush=True)
            if result.returncode:
                print(f"FAILED {name}: exit {result.returncode}; see {raw.with_suffix('.log')}", flush=True)
            summarize(output, cases, provenance, metrics)
    summary = summarize(output, cases, provenance, metrics)
    print(json.dumps(summary, indent=2), flush=True)
    if any(row["status"] != "measured" for row in summary):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
