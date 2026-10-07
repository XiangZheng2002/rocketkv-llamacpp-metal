#!/usr/bin/env python3
"""Prepared-prompt LongBench/RULER prediction adapter. Dry-run unless --execute."""
import argparse
import json
from pathlib import Path
import subprocess
import tempfile


def normalize(record, kind, template=None):
    if template:
        prompt = template.format_map(record)
    elif "prompt" in record:
        prompt = record["prompt"]
    elif kind == "ruler" and "input" in record:
        prompt = record["input"] + record.get("answer_prefix", "")
    else:
        raise ValueError("LongBench needs a prepared 'prompt' or --template with {context}/{input}; raw context is never silently dropped")
    if not isinstance(prompt, str) or not prompt:
        raise ValueError("prompt must be a nonempty string")
    return prompt


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--binary", type=Path, default=Path("build/bin/llama-rocketkv"))
    parser.add_argument("--kind", choices=("longbench", "ruler"), required=True)
    parser.add_argument("--mode", choices=("full", "rocket"), default="rocket")
    parser.add_argument("--budget", type=int, default=512)
    parser.add_argument("--generate", type=int, default=32)
    parser.add_argument("--template", type=Path, help="task-specific UTF-8 format string; use the official benchmark template")
    parser.add_argument("--limit", type=int, default=3, help="0 means all rows; only use for explicitly authorized full evaluation")
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args()
    if args.limit < 0:
        parser.error("limit must be nonnegative")
    template = args.template.read_text() if args.template else None
    records = []
    with args.input.open() as file:
        for line in file:
            if line.strip():
                record = json.loads(line)
                records.append((record, normalize(record, args.kind, template)))
                if args.limit and len(records) == args.limit:
                    break
    if not args.execute:
        print(json.dumps({"status": "dry_run", "examples": len(records), "kind": args.kind,
                          "mode": args.mode, "budget": args.budget, "prompt_characters": [len(p) for _, p in records],
                          "scoring": "use the official benchmark scorer on the saved 'pred' records"}, indent=2))
        return
    if args.output.exists():
        raise FileExistsError(f"refusing to overwrite {args.output}")
    with tempfile.TemporaryDirectory(prefix="rocketkv-eval-") as temp, args.output.open("w") as output:
        temp = Path(temp)
        for index, (record, prompt) in enumerate(records):
            prompt_file, result_file = temp / "prompt.txt", temp / "result.json"
            prompt_file.write_text(prompt)
            command = [
                str(args.binary.resolve()), "-m", str(args.model.resolve()), "--mode", args.mode,
                "--prompt-file", str(prompt_file), "--raw-prompt",
                "--generate", str(record.get("tokens_to_generate", args.generate)),
                "--warmup", "0", "--repetitions", "1", "--output", str(result_file),
            ]
            if args.mode == "rocket":
                command += ["--budget", str(args.budget)]
            subprocess.run(command, check=True)
            result = json.loads(result_file.read_text())["records"][0]
            prediction = dict(record)
            prediction["pred"] = result["answer_before_eog"]
            prediction["rocketkv"] = {"mode": args.mode, "budget": args.budget, "prompt_tokens": result["prompt_tokens"]}
            output.write(json.dumps(prediction, ensure_ascii=False) + "\n")
            output.flush()
            print(f"predicted {index+1}/{len(records)}")


if __name__ == "__main__":
    main()
