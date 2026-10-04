"""evals/calibrate.py fits the temperature classif applies. These tests feed it
rows with a known answer; no model, no network."""

import importlib.machinery
import importlib.util
import json
import math
import tempfile
import unittest
from pathlib import Path

SCRIPT = Path(__file__).resolve().parents[1] / "evals" / "calibrate.py"


def load():
    loader = importlib.machinery.SourceFileLoader("calibrate", str(SCRIPT))
    mod = importlib.util.module_from_spec(importlib.util.spec_from_loader("calibrate", loader))
    loader.exec_module(mod)
    return mod


def overconfident_rows(t_true):
    """For log-odds z, true yes happens sigmoid(z) of the time; the model
    reports logp z * t_true, too sure by a factor of t_true."""
    rows = []
    for z in (-3, -2, -1, -0.5, 0.5, 1, 2, 3):
        yes = round(100 / (1 + math.exp(-z)))
        logp = {"yes": z * t_true, "no": 0.0}
        rows += [{"logp": logp, "y": "yes"}] * yes + [{"logp": logp, "y": "no"}] * (100 - yes)
    return rows


class CalibrateTests(unittest.TestCase):
    cal = load()

    def test_fit_recovers_how_overconfident_the_model_is(self):
        self.assertAlmostEqual(self.cal.fit(overconfident_rows(2.5)), 2.5, delta=0.05)

    def test_ece_is_0_when_each_confidence_matches_its_hit_rate(self):
        p = {"yes": 0.8, "no": 0.2}
        pairs = [(p, "yes")] * 8 + [(p, "no")] * 2
        self.assertAlmostEqual(self.cal.metrics(pairs)["ece"], 0.0)
        self.assertAlmostEqual(self.cal.metrics([(p, "no")] * 10)["ece"], 0.8)

    def test_write_sets_one_model_and_keeps_the_others(self):
        path = Path(self.enterContext(tempfile.TemporaryDirectory())) / "calibration.json"
        path.write_text(json.dumps({"other:1b": {"T": 1.7, "n": 9, "fitted": "2026-01-01"}}))
        self.cal.write(str(path), "gemma4:12b", 3.14159, 180)
        table = json.loads(path.read_text())
        self.assertEqual(table["other:1b"], {"T": 1.7, "n": 9, "fitted": "2026-01-01"})
        self.assertEqual((table["gemma4:12b"]["T"], table["gemma4:12b"]["n"]), (3.142, 180))

    def test_write_for_enum_questions_keeps_the_models_other_temperature(self):
        path = Path(self.enterContext(tempfile.TemporaryDirectory())) / "calibration.json"
        self.cal.write(str(path), "gemma4:12b", 4.36, 200)
        self.cal.write(str(path), "gemma4:12b", 2.68, 144, enum=True)
        entry = json.loads(path.read_text())["gemma4:12b"]
        self.assertEqual((entry["T"], entry["n"], entry["T_enum"], entry["n_enum"]), (4.36, 200, 2.68, 144))

    def test_each_kind_of_question_gets_its_own_fit(self):
        rows = overconfident_rows(3.0) + [{**r, "enum": True} for r in overconfident_rows(1.5)]
        got = {key: (round(t, 1), len(kind)) for key, kind, t in self.cal.fits(rows)}
        self.assertEqual(got, {"T": (3.0, 800), "T_enum": (1.5, 800)})

    def test_a_kind_with_too_few_rows_is_not_fitted(self):
        few = [{"logp": {"a": 0.0, "b": -1.0}, "y": "a", "enum": True}] * 9
        self.assertEqual([key for key, *_ in self.cal.fits(overconfident_rows(3.0) + few)], ["T"])
        self.assertEqual([key for key, *_ in self.cal.fits(few + few[:1])], ["T_enum"])
