import unittest
from pathlib import Path

from eval_jsonl import normalize
from prepare_full import configuration
from run_mvp import make_command, make_jobs, validate_document


class HarnessTests(unittest.TestCase):
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
        self.assertTrue(all(row["requires_implementation_extension"] == (row["context"] > 8192) for row in config["runs"]))

    def test_dataset_context_is_not_dropped(self):
        with self.assertRaises(ValueError):
            normalize({"context": "long context", "input": "question"}, "longbench")
        self.assertEqual(normalize({"context": "text", "input": "question"}, "longbench", "{context}\n{input}"), "text\nquestion")
        self.assertEqual(normalize({"input": "ready prompt"}, "ruler"), "ready prompt")
        with self.assertRaises(ValueError):
            normalize({"prompt": ""}, "ruler")

    def test_incomplete_runs_are_rejected(self):
        job = make_jobs(3)[0]
        with self.assertRaises(ValueError):
            validate_document({"records": []}, job)


if __name__ == "__main__":
    unittest.main()
