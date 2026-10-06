#!/usr/bin/env python3
"""evals/tag_eval.py builds two-question cases, asks them through tag in both
orders and alone through -e, and reports per position."""
import importlib.machinery
import importlib.util
import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

SCRIPT = Path(__file__).resolve().parent.parent / "evals" / "tag_eval.py"


def load():
    loader = importlib.machinery.SourceFileLoader("tag_eval_script", str(SCRIPT))
    mod = importlib.util.module_from_spec(importlib.util.spec_from_loader("tag_eval_script", loader))
    loader.exec_module(mod)
    return mod


def row(q, state, ids, label):
    return {"question": q, "state": state, "options": [{"id": i, "description": i} for i in ids], "label": label}


class CaseTests(unittest.TestCase):
    def write(self, text, suffix):
        with tempfile.NamedTemporaryFile("w", suffix=suffix, delete=False) as fh:
            fh.write(text)
        self.addCleanup(Path(fh.name).unlink)
        return fh.name

    def test_only_states_asked_two_questions_pair_and_truth_follows_each_rows_option_order(self):
        rows = [row("fitted?", "S", ["a", "b"], 1), row("ordered?", "S", ["b", "a"], 1), row("alone?", "T", ["a", "b"], 0)]
        path = self.write("".join(json.dumps(r) + "\n" for r in rows), ".jsonl")
        got = list(load().pair_cases(path))
        self.assertEqual(got, [("pairs", "S", [("fitted?", ["a", "b"], "b"), ("ordered?", ["b", "a"], "a")])])

    def test_each_mail_is_asked_whole_and_short_and_flagged_mail_is_left_out(self):
        mails = [{"src": "newsletter", "y": "garbage", "x": "SALE"}, {"src": "flagged", "y": "important", "x": "F"}]
        got = list(load().email_cases(self.write(json.dumps(mails), ".json")))
        self.assertEqual([(s, x, [(q, t) for q, _, t in qs]) for s, x, qs in got], [
            ("email-whole", "SALE", [("Is this email a newsletter?", "yes"),
                                     ("Does this email need a reply or action from me?", "no")]),
            ("email-short", "SALE", [("newsletter", "yes"), ("needs reply or action", "no")])])


class RunTests(unittest.TestCase):
    def test_tag_runs_in_both_orders_and_each_question_runs_alone_once(self):
        ev, calls = load(), []

        def fake(cmd, input=None, **kw):
            calls.append(cmd)
            if cmd[1] == "tag":
                tags = {"a?": {"label": "x", "p": {"x": 0.9, "y": 0.1}}, "b?": {"label": "y", "p": {"x": 0.2, "y": 0.8}}}
                out = {"tags": tags, "ms": 300}
            else:
                out = {"label": "x", "p": {"x": 0.7, "y": 0.3}, "ms": 200}
            return type("R", (), {"stdout": json.dumps(out), "stderr": ""})

        qs = [("a?", ["x", "y"], "x"), ("b?", ["x", "y"], "y")]
        with mock.patch.object(ev.subprocess, "run", fake):
            rows = ev.rows_for("pairs", "text", qs, {})
        self.assertEqual([c[1:] for c in calls], [["-j", "-e", "x,y", "a?"], ["-j", "-e", "x,y", "b?"],
                                                  ["tag", "-j", "a?=x,y", "b?=x,y"], ["tag", "-j", "b?=x,y", "a?=x,y"]])
        self.assertEqual([(r["pos"], r["q"], r["tag"]["label"], r["single"]["label"]) for r in rows],
                         [(1, "a?", "x", "x"), (2, "b?", "y", "x"), (1, "b?", "y", "x"), (2, "a?", "x", "x")])
        self.assertEqual((rows[0]["tag_ms"], rows[0]["single_ms"]), (300, 400))


class MeasureTests(unittest.TestCase):
    def test_brier_scores_the_true_option_and_unscored_rows_count_apart(self):
        rows = [{"truth": "x", "tag": {"label": "x", "p": {"x": 0.9, "y": 0.1}}},
                {"truth": "y", "tag": {"label": "x", "p": {"x": 0.6, "y": 0.4}}},
                {"truth": "x", "tag": {"label": None, "p": None}}]
        m = load().measure(rows, "tag")
        self.assertEqual((m["n"], m["unscored"], m["right"]), (3, 1, 1))
        self.assertAlmostEqual(m["brier"], ((1 - 0.9) ** 2 + (1 - 0.4) ** 2) / 2)

    def test_ece_weighs_each_bin_by_its_share(self):
        # Bin 0.9: p 0.95 twice, one right: |0.5 - 0.95|. Bin 0.6: p 0.6, right: |1 - 0.6|.
        got = load().ece([(0.95, True), (0.95, False), (0.6, True)])
        self.assertAlmostEqual(got, (2 * 0.45 + 1 * 0.4) / 3)


if __name__ == "__main__":
    unittest.main()
