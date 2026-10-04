"""evals/long.py report: accuracy counts a refusal (label None) apart from a
wrong label, and wall time is summarised per run. No model, no man pages."""

import contextlib
import importlib.machinery
import importlib.util
import io
import json
import os
import tempfile
import unittest
from pathlib import Path

SCRIPT = Path(__file__).resolve().parents[1] / "evals" / "long.py"


def load():
    loader = importlib.machinery.SourceFileLoader("long_eval", str(SCRIPT))
    mod = importlib.util.module_from_spec(importlib.util.spec_from_loader("long_eval", loader))
    loader.exec_module(mod)
    return mod


class ReportTests(unittest.TestCase):
    e = load()

    def test_right_wrong_and_no_answer_are_counted_apart(self):
        rows = [{"want": "yes", "got": "yes", "wall": 1.0}, {"want": "no", "got": "yes", "wall": 3.0},
                {"want": "no", "got": None, "wall": 9.0}, {"want": "yes", "got": "unknown", "wall": 2.0}]
        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, "run.jsonl")
            with open(path, "w") as fh:
                fh.writelines(json.dumps(r) + "\n" for r in rows)
            out = io.StringIO()
            with contextlib.redirect_stdout(out):
                self.e.report([path])
        self.assertEqual(out.getvalue().strip(),
                         "run.jsonl: 1/4 right, 2 wrong, 1 no answer; wall p50 3.0s, max 9.0s, total 15s")

    def test_every_case_names_a_page_and_a_yes_or_no(self):
        for name, want, claim in self.e.CASES:
            self.assertIn(name, ("bash", "rsync", "ledger", "book", "novel"))
            self.assertIn(want, ("yes", "no"))
            self.assertTrue(claim.endswith((".", "?")))


if __name__ == "__main__":
    unittest.main()
