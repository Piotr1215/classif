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


def jev_rows(t_true, kind="score"):
    """Rows whose Jev distribution is classif's raw one tempered by t_true,
    as the eval writes them; a choice row also carries -e's none."""
    rows = []
    for a, b in ((-0.1, -3.0), (-0.5, -1.2), (-2.0, -0.2), (-0.05, -6.0), (-1.0, -1.1)):
        logp = {"0": a, "1": b} if kind == "score" else {"a": a, "b": b, "none": -9.0}
        keys = [k for k in logp if k != "none"]
        z = sum(math.exp(logp[k] / t_true) for k in keys)
        rows.append({"kind": kind, "logp": logp, "jev": {k: math.exp(logp[k] / t_true) / z for k in keys}})
    return rows * 4


class JevCalibrateTests(unittest.TestCase):
    cal = load()

    def test_fit_to_jev_finds_the_temperature_that_reproduces_jevs_distributions(self):
        self.assertAlmostEqual(self.cal.fit_jev(jev_rows(1.7)), 1.7, delta=0.02)

    def test_a_choice_rows_none_is_left_out_before_tempering(self):
        row = jev_rows(1.0, kind="choice")[0]
        self.assertEqual(set(self.cal.at(row, 1.0)), {"a", "b"})
        self.assertAlmostEqual(self.cal.tv(self.cal.at(row, 1.0), row["jev"]), 0.0)

    def test_scores_e_questions_and_l_questions_are_fitted_apart(self):
        rows = jev_rows(1.5) + jev_rows(3.0, kind="choice") + [{**r, "kind": "noul", "logp": {
            "yes": r["logp"]["a"], "no": r["logp"]["b"]}, "jev": {"yes": r["jev"]["a"], "no": r["jev"]["b"]}}
            for r in jev_rows(0.8, kind="choice")]
        got = {key: (round(t, 1), len(kind)) for key, kind, t in self.cal.jev_fits(rows)}
        self.assertEqual(got, {"T_score": (1.5, 20), "T_enum": (3.0, 20), "T": (0.8, 20)})

    def test_write_for_scores_keeps_the_models_other_temperatures(self):
        path = Path(self.enterContext(tempfile.TemporaryDirectory())) / "calibration.json"
        self.cal.write(str(path), "winnow", 2.79, 200)
        self.cal.write(str(path), "winnow", 1.21, 52, kind="_score")
        entry = json.loads(path.read_text())["winnow"]
        self.assertEqual((entry["T"], entry["T_score"], entry["n_score"]), (2.79, 1.21, 52))


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
