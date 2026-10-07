#!/usr/bin/env python3
"""Print future evaluation configuration; never download models or execute runs."""
import argparse
import json
import math
from pathlib import Path


def configuration(model, binary):
    runs = []
    for context in (4096, 8192, 16384, 32768):
        for generated in (128, 512):
            for mode, budget in (("full", 0), ("rocket", 1024), ("rocket", 2048)):
                command = [
                    str(binary), "-m", str(model), "--mode", mode, "--prompt-tokens", str(context),
                    "--generate", str(generated), "--warmup", "1", "--repetitions", "3",
                ]
                if budget:
                    command += ["--budget", str(budget)]
                total = context + generated
                ratio = total / budget if budget else 1
                capacity = int(total / ratio ** min(0.2 + math.log2(ratio) * 0.06, 0.8))
                runs.append({
                    "context": context, "generated": generated, "mode": mode, "budget": budget,
                    "command": command,
                    "requires_implementation_extension": context > 32768 or (mode == "rocket" and capacity > 8192),
                })
    return {
        "status": "prepared_not_executed",
        "model": str(model),
        "model_weights": "Llama-3.1-8B-Instruct, ready-made Q4 GGUF; do not convert source weights",
        "warning": "Prompts through 32768 are supported; RocketKV compressed capacity is limited to 8192. Reassess unified memory before any 8B run.",
        "runs": runs,
        "dataset_adapter": "tools/rocketkv/eval_jsonl.py (prepared LongBench/RULER JSONL; official scoring remains external)",
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, default=Path("models/Meta-Llama-3.1-8B-Instruct-Q4_K_M.gguf"))
    parser.add_argument("--binary", type=Path, default=Path("build/bin/llama-rocketkv"))
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    text = json.dumps(configuration(args.model, args.binary), indent=2) + "\n"
    if args.output:
        args.output.write_text(text)
    else:
        print(text, end="")


if __name__ == "__main__":
    main()
