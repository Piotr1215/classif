#!/usr/bin/env python3
"""evals/eval.py scores SemIf choice cases and reports per task."""
import contextlib
import importlib.machinery
import importlib.util
import io
import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

SCRIPT = Path(__file__).resolve().parent.parent / "evals" / "eval.py"


def load():
    loader = importlib.machinery.SourceFileLoader("eval_script", str(SCRIPT))
    mod = importlib.util.module_from_spec(importlib.util.spec_from_loader("eval_script", loader))
    loader.exec_module(mod)
    return mod


CHOICE_ROW = {"question": "Assess the claim: the lenses have been fitted.",
              "state": "The workshop confirms they have not yet been fitted.",
              "options": [{"id": "supported", "description": "The evidence establishes the claim"},
                          {"id": "insufficient", "description": "The evidence does not establish either"},
                          {"id": "contradicted", "description": "The evidence establishes the opposite, clearly"}],
              "label": 2}


class ChoiceCaseTests(unittest.TestCase):
    def cases(self, *rows):
        with tempfile.NamedTemporaryFile("w", suffix=".jsonl", delete=False) as fh:
            fh.write("".join(json.dumps(r) + "\n" for r in rows))
        self.addCleanup(Path(fh.name).unlink)
        return list(load().choice(fh.name))

    def test_the_label_indexes_the_rows_own_option_order(self):
        swapped = {**CHOICE_ROW, "options": CHOICE_ROW["options"][::-1], "label": 0}
        got = self.cases(CHOICE_ROW, swapped)
        self.assertEqual([(c[0], c[3]) for c in got], [("choice", "contradicted"), ("choice", "contradicted")])

    def test_options_go_to_class_as_a_newline_list_with_descriptions_and_none_is_a_label(self):
        task, q, x, y, labels, enum = self.cases(CHOICE_ROW)[0]
        self.assertEqual((q, x), (CHOICE_ROW["question"], CHOICE_ROW["state"]))
        self.assertEqual(labels, ["supported", "insufficient", "contradicted", "none"])
        self.assertEqual(enum.split("\n")[2], "contradicted=The evidence establishes the opposite, clearly")

    def test_score_asks_with_e_for_an_enum_case_and_with_l_otherwise(self):
        ev, calls = load(), []

        class Done:
            stdout, stderr = json.dumps({"label": "no", "p": {"yes": 0.1, "no": 0.9}}), ""

        with mock.patch.object(ev.subprocess, "run", lambda cmd, **kw: calls.append(cmd) or Done):
            ev.score("choice", "q?", "t", "a", ["a", "b", "none"], "a=first\nb=second", {})
            row = ev.score("rag", "q?", "t", "no", ["yes", "no"], None, {})
        self.assertEqual([c[2:4] for c in calls], [["-e", "a=first\nb=second"], ["-l", "yes,no"]])
        self.assertEqual((row["task"], row["y"], row["label"]), ("rag", "no", "no"))

    def test_a_scored_row_says_whether_it_was_asked_with_e(self):
        ev = load()

        class Done:
            stdout, stderr = json.dumps({"label": "a", "p": {"a": 0.9, "b": 0.1}}), ""

        with mock.patch.object(ev.subprocess, "run", lambda cmd, **kw: Done):
            rows = [ev.score("choice", "q?", "t", "a", ["a", "b", "none"], "a=first\nb=second", {}),
                    ev.score("rag", "q?", "t", "no", ["yes", "no"], None, {})]
        self.assertEqual([r["enum"] for r in rows], [True, False])

    def test_the_tracked_cases_load_and_every_answer_is_one_of_its_options(self):
        got = [c for c in load().items() if c[0] == "choice"]
        self.assertEqual(len(got), 144)
        self.assertTrue(all(c[3] in c[4][:-1] and c[4][-1] == "none" for c in got))


if __name__ == "__main__":
    unittest.main()
