import unittest
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace

from eval_jsonl import normalize
from prepare_full import configuration
from run_mvp import file_hash, make_command, make_jobs, validate_document
from run_ruler import DEFAULT_CONTEXTS, TASKS, load_cases, measured, summarize, timing_error


class HarnessTests(unittest.TestCase):
    def test_ruler_default_scope_is_at_most_8k(self):
        self.assertEqual(DEFAULT_CONTEXTS, (2048, 4096, 8192))

    def test_ruler_rejects_sleep_contaminated_timing(self):
        self.assertIsNone(timing_error({"process_realtime_ms": 150001, "process_wall_ms": 150000}))
        self.assertIsNotNone(timing_error({"process_realtime_ms": 1003000, "process_wall_ms": 48000}))
        self.assertIsNotNone(timing_error({"process_realtime_ms": 30000, "process_wall_ms": 48000}))

    def test_ruler_reference_comparison_and_invalid_timing(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "raw").mkdir()
            case = {"context": 2048, "task": "qa_1", "record": {
                "index": 0, "prompt_tokens": 2016, "tokens_to_generate": 32, "outputs": ["answer"],
            }}
            provenance = {"contexts": [2048], "reference_runtime": {"binary": "reference"},
                          "repetitions": 1, "warmup": 0, "budget": 512, "samples": 1,
                          "commit": "test", "experiment": "test"}
            metrics = SimpleNamespace(string_match_part=lambda predictions, references: 100)
            for mode, decode_ms in (("full", 620), ("rocket", 310), ("reference", 465)):
                row = {
                    "warmup": False, "prompt_tokens": 2016, "generated_tokens": 32, "decode_steps": 31,
                    "tokens": [1] * 32, "mode": "full" if mode == "full" else "rocket",
                    "kv_type": "f16", "instrumented": False, "token_budget": 0 if mode == "full" else 512,
                    "tpot_ms": decode_ms / 31, "decode_total_ms": decode_ms, "engine_ttft_ms": 100,
                    "request_ms": 100 + decode_ms, "allocated_kv_bytes": 1024, "context_tensor_bytes": 1024,
                    "decode_peak_rss_sampled_bytes": 2048, "prefill_peak_rss_sampled_bytes": 2048,
                    "answer_before_eog": "answer",
                }
                path = root / "raw" / f"2048_qa_1_0_{mode}.json"
                path.write_text(json.dumps({"raw_prompt": True, "prompt_token_ids": [1] * 2016, "records": [row]}))
                path.with_suffix(".meta.json").write_text(json.dumps({"returncode": 0}))
            result = summarize(root, [case], provenance, metrics)[0]
            self.assertEqual(result["status"], "measured")
            self.assertAlmostEqual(result["speedup"], 2)
            self.assertAlmostEqual(result["speedup_vs_reference"], 1.5)
            candidate = root / "raw/2048_qa_1_0_rocket.json"
            doc = json.loads(candidate.read_text())
            doc["records"][0]["rocket_variant"] = "stage_removed"
            candidate.write_text(json.dumps(doc))
            with self.assertRaisesRegex(ValueError, "algorithm variant"):
                summarize(root, [case], provenance, metrics)
            provenance["candidate_variant"] = "stage_removed"
            with self.assertRaisesRegex(ValueError, "two-stage RocketKV"):
                summarize(root, [case], provenance, metrics)
            del provenance["candidate_variant"]
            doc["records"][0]["rocket_variant"] = "rocketkv_hybrid"
            candidate.write_text(json.dumps(doc))
            (root / "raw/2048_qa_1_0_reference.meta.json").write_text(
                json.dumps({"returncode": 0, "timing_error": "system sleep"}))
            result = summarize(root, [case], provenance, metrics)[0]
            self.assertEqual(result["status"], "failed")
            self.assertEqual(result["modes"]["reference"]["measured_requests"], 0)
            self.assertIsNone(result["speedup"])

    def test_mvp_is_bounded(self):
        jobs = make_jobs(3)
        perf = [job for job in jobs if job["kind"] == "performance"]
        self.assertEqual(len(perf), 6)
        self.assertEqual({job["context"] for job in perf}, {2048, 4096})
        self.assertEqual({job["budget"] for job in perf}, {0, 256, 512})
        self.assertTrue(all(job["warmup"] == 1 and job["repetitions"] == 3 for job in perf))
        self.assertEqual(sum(job["kind"] == "quantized_sanity" for job in jobs), 1)
        self.assertEqual(sum(job["kind"] == "niah" for job in jobs), 9)
        for job in jobs:
            command = make_command(Path("binary with spaces"), Path("model.gguf"), Path("out.json"), job)
            self.assertEqual(command[0], "binary with spaces")
            self.assertEqual(command[command.index("--generate") + 1], "32")
            self.assertEqual("--profile" in command, job["kind"] == "profile")

    def test_full_config_does_not_hide_limits(self):
        config = configuration("8b.gguf", "binary")
        self.assertEqual(config["status"], "prepared_not_executed")
        self.assertTrue(all(row["requires_implementation_extension"] ==
                            (row["context"] == 32768 and row["budget"] == 2048) for row in config["runs"]))

    def test_dataset_context_is_not_dropped(self):
        with self.assertRaises(ValueError):
            normalize({"context": "long context", "input": "question"}, "longbench")
        self.assertEqual(normalize({"context": "text", "input": "question"}, "longbench", "{context}\n{input}"), "text\nquestion")
        self.assertEqual(normalize({"input": "ready prompt"}, "ruler"), "ready prompt")
        self.assertEqual(normalize({"input": "ready prompt", "answer_prefix": " Answer:"}, "ruler"), "ready prompt Answer:")
        self.assertEqual(normalize({"prompt": "exact", "answer_prefix": "not twice"}, "ruler"), "exact")
        with self.assertRaises(ValueError):
            normalize({"prompt": ""}, "ruler")

    def test_incomplete_runs_are_rejected(self):
        job = make_jobs(3)[0]
        with self.assertRaises(ValueError):
            validate_document({"records": []}, job)

    def test_ruler_rejects_retemplated_input(self):
        with self.assertRaises(ValueError):
            measured({"records": [], "raw_prompt": False}, {"record": {}}, "full", 1, 0, 512)

    def test_ruler_fixed_holdout_and_integrity(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            files = []
            for task in TASKS:
                path = root / "2048" / task / "validation.jsonl"
                path.parent.mkdir(parents=True)
                rows = [{
                    "index": i, "input": f"input {i}", "answer_prefix": " Answer:",
                    "prompt": f"input {i} Answer:", "prompt_tokens": 1920,
                    "tokens_to_generate": 128, "target_length": 2048,
                } for i in range(6)]
                path.write_text("".join(json.dumps(row) + "\n" for row in rows))
                files.append({"path": str(path.relative_to(root)), "samples": 6, "sha256": file_hash(path)})
            (root / "manifest.json").write_text(json.dumps({"files": files}))
            _, screening = load_cases(root, 1, [2048])
            _, holdout = load_cases(root, 5, [2048], 1)
            _, focused = load_cases(root, 1, [2048], tasks=("qa_1",))
            self.assertEqual(len(focused), 1)
            self.assertEqual(focused[0]["task"], "qa_1")
            self.assertEqual(len(screening), 4)
            self.assertEqual(len(holdout), 20)
            self.assertEqual({c["record"]["index"] for c in screening}, {0})
            self.assertEqual({c["record"]["index"] for c in holdout}, {1, 2, 3, 4, 5})
            with self.assertRaises(ValueError):
                load_cases(root, 6, [2048], 1)
            path.write_text(path.read_text() + "\n")
            with self.assertRaises(ValueError):
                load_cases(root, 1, [2048])


if __name__ == "__main__":
    unittest.main()
