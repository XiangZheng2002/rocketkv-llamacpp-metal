#!/usr/bin/env python3
"""Run only the bounded RocketKV MVP. No model or dataset downloads."""
import argparse
import csv
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import platform
import random
import shutil
import statistics
import subprocess
import sys
import time


ROOT = Path(__file__).resolve().parents[2]


def command_text(command):
    return subprocess.run(command, check=True, capture_output=True, text=True).stdout.strip()


def source_hash():
    paths = ["ggml", "src", "include", "tests", "tools/rocketkv", "tools/CMakeLists.txt"]
    digest = hashlib.sha256(subprocess.check_output(["git", "diff", "--binary", "HEAD", "--", *paths], cwd=ROOT))
    names = subprocess.check_output(
        ["git", "ls-files", "--others", "--exclude-standard", "-z", "--", *paths], cwd=ROOT
    ).decode().split("\0")
    for name in sorted(filter(None, names)):
        path = ROOT / name
        if path.is_file():
            digest.update(name.encode())
            digest.update(path.read_bytes())
    return digest.hexdigest()


def file_hash(path):
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for block in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def machine_state():
    state = {"utc": datetime.now(timezone.utc).isoformat(), "disk_free_bytes": shutil.disk_usage(ROOT).free}
    if sys.platform == "darwin":
        state["swap"] = command_text(["sysctl", "vm.swapusage"])
        state["memory_pressure"] = command_text(["memory_pressure"])
        state["thermal"] = command_text(["pmset", "-g", "therm"])
    return state


def make_jobs(repetitions):
    jobs = []
    for context in (2048, 4096):
        for mode, budget in (("full", 0), ("rocket", 256), ("rocket", 512)):
            jobs.append({
                "id": f"perf_{context}_{mode}_{budget}",
                "kind": "performance", "mode": mode, "budget": budget, "context": context,
                "warmup": 1, "repetitions": repetitions,
            })
    random.Random(6222).shuffle(jobs)
    for context in (2048, 4096):
        for budget in (256, 512):
            jobs.append({
                "id": f"profile_{context}_rocket_{budget}",
                "kind": "profile", "mode": "rocket", "budget": budget, "context": context,
                "warmup": 1, "repetitions": repetitions, "profile": True,
            })
    jobs.append({
        "id": "quantized_kv_sanity", "kind": "quantized_sanity", "mode": "full", "budget": 0,
        "context": 2048, "kv": "q8_0", "warmup": 1, "repetitions": 1,
    })
    short_prompts = [
        "What is the capital of France? Answer in one word.",
        "What is 2 + 2? Reply with only the number.",
        "Write exactly the word hello.",
    ]
    for i, prompt in enumerate(short_prompts):
        for mode, budget in (("full", 0), ("rocket", 256), ("rocket", 512)):
            jobs.append({
                "id": f"short_{i}_{mode}_{budget}", "kind": "short", "mode": mode, "budget": budget,
                "context": 2048, "prompt": prompt, "warmup": 0, "repetitions": 1,
                "short_index": i,
            })
    for i, (depth, passkey) in enumerate(((0.25, "37461"), (0.5, "58293"), (0.75, "81627"))):
        for mode, budget in (("full", 0), ("rocket", 256), ("rocket", 512)):
            jobs.append({
                "id": f"niah_{i}_{mode}_{budget}", "kind": "niah", "mode": mode, "budget": budget,
                "context": 4096, "passkey": passkey, "depth": depth, "warmup": 0, "repetitions": 1,
            })
    return jobs


def make_command(binary, model, output, job):
    command = [
        str(binary), "-m", str(model), "--mode", job["mode"], "--kv", job.get("kv", "f16"),
        "--prompt-tokens", str(job["context"]), "--generate", "32", "--ubatch", "512",
        "--warmup", str(job["warmup"]), "--repetitions", str(job["repetitions"]),
        "--output", str(output),
    ]
    if job["mode"] == "rocket":
        command += ["--budget", str(job["budget"]), "--window", "32", "--pool", "63"]
    if job.get("profile"):
        command.append("--profile")
    if "prompt" in job:
        command += ["--prompt", job["prompt"]]
    if "passkey" in job:
        command += ["--passkey", job["passkey"], "--depth", str(job["depth"])]
    return command


def validate_document(document, job):
    records = document["records"]
    measured = [row for row in records if not row["warmup"]]
    if len(measured) != job["repetitions"] or len(records) - len(measured) != job["warmup"]:
        raise ValueError("incorrect warm-up/repetition counts")
    for row in records:
        if row["generated_tokens"] != 32 or row["decode_steps"] != 31 or len(row["tokens"]) != 32:
            raise ValueError("benchmark did not generate exactly 32 tokens")
        if job["kind"] != "short" and row["prompt_tokens"] != job["context"]:
            raise ValueError("benchmark prompt length is not the requested token count")
        if row["mode"] != job["mode"] or row["token_budget"] != job["budget"]:
            raise ValueError("mode/budget mismatch")
        if row["kv_type"] != job.get("kv", "f16") or row["instrumented"] != bool(job.get("profile")):
            raise ValueError("KV type/profiling mismatch")
        if row["tpot_ms"] <= 0 or row["engine_ttft_ms"] <= 0:
            raise ValueError("invalid timing")
        if job.get("profile"):
            calls = row["component_calls"]
            if not all(calls.values()) or calls["s1_metadata"] * 31 != calls["s2_metadata"]:
                raise ValueError("incomplete GPU profiling coverage")
            if calls["s2_attention"] != calls["s2_metadata"] or calls["s2_metadata"] != 31 * row["layers"]:
                raise ValueError("sparse attention was not profiled for every layer/token")
    if any(row["tokens"] != measured[0]["tokens"] for row in measured):
        raise ValueError("greedy output changed between measured repetitions")
    return measured


def summarize(documents):
    summary = {"performance": [], "profiles": [], "accuracy": [], "quantized_sanity": []}
    short = {}
    dense_prefill = {}
    for job, document in documents:
        if job["mode"] == "full" and job["kind"] in ("performance", "niah"):
            key = (job["kind"], job["context"], job.get("passkey", ""))
            dense_prefill[key] = document
    for job, document in documents:
        records = validate_document(document, job)
        if job["mode"] == "rocket" and job["kind"] in ("performance", "niah"):
            key = (job["kind"], job["context"], job.get("passkey", ""))
            baseline = dense_prefill[key]
            if document["prompt_token_ids"] != baseline["prompt_token_ids"]:
                raise ValueError("Full-KV and RocketKV received different prompts")
            if records[0]["tokens"][0] != baseline["records"][-1]["tokens"][0]:
                raise ValueError("compression changed the prefill-produced first token")
        entry = {
            "id": job["id"], "mode": job["mode"], "budget": job["budget"],
            "prompt_tokens": records[0]["prompt_tokens"], "measured_repetitions": len(records),
        }
        if job["kind"] in ("performance", "quantized_sanity"):
            for metric in (
                "engine_ttft_ms", "request_ttft_ms", "tpot_ms", "decode_total_ms",
                "decode_peak_footprint_sampled_bytes", "decode_peak_rss_sampled_bytes",
                "prefill_peak_footprint_sampled_bytes", "decode_allocated_tensor_bytes",
                "context_tensor_bytes", "compute_tensor_bytes", "rocket_full_kv_allocation_bytes",
                "rocket_active_kv_capacity_bytes", "rocket_auxiliary_bytes", "allocated_kv_bytes",
            ):
                values = [row[metric] for row in records]
                entry[metric + "_mean"] = statistics.mean(values)
                entry[metric + "_stdev"] = statistics.stdev(values) if len(values) > 1 else 0
            entry["first_eog_index"] = records[0]["first_eog_index"]
            summary["performance" if job["kind"] == "performance" else "quantized_sanity"].append(entry)
        elif job["kind"] == "profile":
            entry["instrumented_separate_run"] = True
            for key in ("stage1_gpu_ms", "stage2_selection_metadata_gather_gpu_ms", "sparse_attention_gpu_ms"):
                entry[key] = statistics.mean(row[key] for row in records)
            entry["component_gpu_ms"] = {
                key: statistics.mean(row["component_gpu_ms"][key] for row in records)
                for key in records[0]["component_gpu_ms"]
            }
            entry["component_calls_per_request"] = records[0]["component_calls"]
            entry["selection_gpu_ms_per_decode_step"] = entry["stage2_selection_metadata_gather_gpu_ms"] / 31
            entry["attention_gpu_ms_per_decode_step"] = entry["sparse_attention_gpu_ms"] / 31
            summary["profiles"].append(entry)
        elif job["kind"] == "short":
            short[(job["short_index"], job["mode"], job["budget"])] = records[0]
        elif job["kind"] == "niah":
            entry.update({
                "task": "synthetic_passkey", "depth": job["depth"], "passkey": job["passkey"],
                "output": records[0]["answer_before_eog"], "pass": records[0]["passkey_pass"],
            })
            summary["accuracy"].append(entry)
    for i in range(3):
        baseline = short[(i, "full", 0)]
        for budget in (256, 512):
            candidate = short[(i, "rocket", budget)]
            identical = baseline["tokens"] == candidate["tokens"]
            if not identical or candidate["rocket_active"]:
                raise ValueError("short prompt Full-KV equivalence check failed")
            summary["accuracy"].append({
                "task": "short_full_kv_equivalence", "index": i, "budget": budget,
                "pass": identical, "output": candidate["answer_before_eog"],
            })
    summary["performance"].sort(key=lambda row: (row["prompt_tokens"], row["mode"], row["budget"]))
    return {"preliminary": True, **summary}


def save_json(path, data):
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(data, indent=2) + "\n")
    temp.replace(path)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, default=ROOT / "models/Llama-3.2-3B-Instruct-Q4_K_M.gguf")
    parser.add_argument("--binary", type=Path, default=ROOT / "build/bin/llama-rocketkv")
    parser.add_argument("--output", type=Path, default=ROOT / "results/rocketkv-mvp")
    parser.add_argument("--repetitions", type=int, default=3)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    if args.repetitions < 3:
        parser.error("MVP requires at least three measured repetitions")
    jobs = make_jobs(args.repetitions)
    if args.dry_run:
        print(json.dumps(jobs, indent=2))
        return
    if not args.model.is_file() or not args.binary.is_file():
        parser.error("model or binary is missing; no automatic downloads are performed")
    if shutil.disk_usage(ROOT).free < 6 * 1024**3:
        parser.error("less than 6 GiB disk headroom remains")
    args.output.mkdir(parents=True, exist_ok=True)
    raw = args.output / "raw"
    raw.mkdir(exist_ok=True)
    model_sha256 = file_hash(args.model)
    known_primary_model = model_sha256 == "6c1a2b41161032677be168d354123594c0e6e67d2b9227c84f296ad037c728ff"
    provenance = {
        "base_commit": command_text(["git", "-C", str(ROOT), "rev-parse", "HEAD"]),
        "implementation_sha256": source_hash(), "model_sha256": model_sha256,
        "system": platform.platform(),
        "cpu": command_text(["sysctl", "-n", "machdep.cpu.brand_string"]) if sys.platform == "darwin" else platform.processor(),
        "ram_bytes": int(command_text(["sysctl", "-n", "hw.memsize"])) if sys.platform == "darwin" else None,
        "model_repository": "bartowski/Llama-3.2-3B-Instruct-GGUF" if known_primary_model else None,
        "model_revision": "5ab33fa94d1d04e903623ae72c95d1696f09f9e8" if known_primary_model else None,
        "start_state": machine_state(),
    }
    save_json(args.output / "provenance.json", provenance)
    documents = []
    for index, job in enumerate(jobs):
        path = raw / (job["id"] + ".json")
        command = make_command(args.binary.resolve(), args.model.resolve(), path.resolve(), job)
        if path.exists():
            if not args.resume:
                raise FileExistsError(f"{path} already exists; use --resume with the same source/model")
            document = json.loads(path.read_text())
            if document["runner"]["implementation_sha256"] != provenance["implementation_sha256"] or document["runner"]["model_sha256"] != provenance["model_sha256"]:
                raise ValueError(f"stale source/model in {path}")
        else:
            before = machine_state()
            started = time.perf_counter()
            print(f"[{index+1}/{len(jobs)}] {job['id']}", flush=True)
            with path.with_suffix(".log").open("w") as log:
                subprocess.run(command, check=True, stdout=log, stderr=subprocess.STDOUT, timeout=600)
            document = json.loads(path.read_text())
            document["runner"] = {
                "implementation_sha256": provenance["implementation_sha256"],
                "model_sha256": provenance["model_sha256"], "job": job,
                "command": command, "process_wall_seconds": time.perf_counter() - started,
                "before": before, "after": machine_state(),
            }
            save_json(path, document)
        validate_document(document, job)
        documents.append((job, document))
        save_json(args.output / "progress.json", {"completed": [entry[0]["id"] for entry in documents], "total": len(jobs)})
    summary = summarize(documents)
    summary["provenance"] = provenance
    summary["end_state"] = machine_state()
    summary["limitations"] = [
        "Preliminary single-machine synthetic MVP, not a final break-even result.",
        "Full n_ctx KV allocation is retained; logical compaction does not increase maximum context.",
        "GPU component timings use separate instrumented runs with encoder boundaries and fusion disabled.",
        "OS memory peaks are samples at token/chunk boundaries, not hardware high-water counters.",
        "32 emitted tokens require 31 decode steps after the prefill-produced first token.",
        "TPOT sums llama_decode plus greedy sampling/logit synchronization, excluding detokenization and memory monitoring.",
        "Fixed-length generation continues after EOG; accuracy uses only the answer before the first EOG.",
        "Request TTFT includes context setup, but excludes one-time model load and prompt preparation.",
    ]
    save_json(args.output / "summary.json", summary)
    with (args.output / "performance.csv").open("w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=summary["performance"][0].keys())
        writer.writeheader()
        writer.writerows(summary["performance"])
    save_json(args.output / "accuracy.json", summary["accuracy"])
    save_json(args.output / "profiles.json", summary["profiles"])
    print(f"Completed {len(jobs)} bounded jobs. Results: {args.output / 'summary.json'}", flush=True)


if __name__ == "__main__":
    main()
