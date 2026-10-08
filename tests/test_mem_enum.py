"""Coverage checks for enum decisions over external-memory passages."""
import math
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

from test_mem import load
from test_judge import FakeOllama, chat


class EnumAsk:
    def __init__(self, leaves, final="1", scores=None, fail=None):
        self.leaves, self.final = leaves, final
        self.scores, self.fail = scores or {}, fail
        self.calls = []

    def __call__(self, question, text, labels, options=None):
        self.calls.append((question, text, tuple(labels), tuple(options or [])))
        if self.fail and self.fail in text:
            return {"label": None, "unscored": "injected failure"}
        label = self.leaves.get(text, self.final) if question.startswith("Going only by this passage:") else self.final
        p = self.scores.get(text, {k: float(k == label) for k in labels})
        return {"label": label, "p": p}


class EnumTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.mem = load()

    def run_case(self, text, leaves, final="1", **kwargs):
        doc = self.mem.Doc(text)
        ask = EnumAsk(leaves, final)
        result = self.mem.run_enum(doc, "What is the invoice status?", ["paid", "reversed"], ask,
                                   block=1, **kwargs)
        return doc, ask, result

    def test_complete_answer_preserves_flagged_sources(self):
        text = "\u2615 Invoice INV-42 paid.\r\nUnrelated weather report.\r\nThat payment was reversed."
        doc = self.mem.Doc(text)
        leaves = {doc.span(0): "1", doc.span(1): "0", doc.span(2): "2"}
        _, ask, result = self.run_case(text, leaves, final="2")
        self.assertEqual(result["verdict"], "answered")
        self.assertEqual(result["label"], "reversed")
        self.assertEqual(result["read"]["checked"], result["read"]["passages"])
        self.assertEqual(result["read"]["basis"], "complete")
        final_text = ask.calls[-1][1]
        self.assertIn(doc.span(0), final_text)
        self.assertIn(doc.span(2), final_text)
        self.assertNotIn(doc.span(1), final_text)
        for source in result["read"]["sources"]:
            self.assertIn(text[source["start"]:source["end"]], final_text)
        # The answer rests on the passage that voted for it, not the one voting paid.
        self.assertEqual(result["read"]["evidence"], [{"line": 3, "end": 3, "text": "That payment was reversed."}])

    STATUS = "Invoice INV-42 was issued.\nIt was sent to the customer.\nThat payment was reversed.\nThe customer was notified."

    def test_lines_narrow_a_voting_passage_to_the_line_that_gives_the_answer(self):
        # Without lines the whole passage is cited; with them only the line that says reversed.
        def ask(question, text, labels, options=None):
            label = "2" if not question.startswith("Going only by") or "reversed" in text else "0"
            return {"label": label, "p": {k: float(k == label) for k in labels}}
        doc = self.mem.Doc(self.STATUS)
        plain = self.mem.run_enum(doc, "What is the status?", ["paid", "reversed"], ask)
        self.assertEqual([(e["line"], e["end"]) for e in plain["read"]["evidence"]], [(1, 4)])
        result = self.mem.run_enum(doc, "What is the status?", ["paid", "reversed"], ask, lines=True)
        self.assertEqual(result["label"], "reversed")
        self.assertEqual(result["read"]["evidence"], [{"line": 3, "end": 3, "text": "That payment was reversed."}])
        # Halves 1-2 and 3-4, then line 3, which gives it: three calls past the passage read and the judge.
        self.assertEqual(result["read"]["lines"]["calls"], 3)
        self.assertEqual(result["read"]["calls"], plain["read"]["calls"] + 3)

    def test_lines_stop_at_the_first_run_when_every_line_gives_the_answer(self):
        # Every line of a source file says python; narrowing them all would cost two calls a line.
        def ask(question, text, labels, options=None):
            return {"label": "1", "p": {k: float(k == "1") for k in labels}}
        doc = self.mem.Doc("\n".join(f"import mod{i}" for i in range(8)))
        result = self.mem.run_enum(doc, "Which language?", ["python", "go"], ask, lines=True)
        self.assertEqual(result["read"]["evidence"], [{"line": 1, "end": 1, "text": "import mod0"}])
        self.assertEqual(result["read"]["lines"]["calls"], 3)

    def test_a_unanimous_vote_too_big_for_the_judge_answers_from_the_surest_passages(self):
        # No passage voted otherwise, so none can be outvoted by the ones left out.
        text = "Service maintenance entry.\nService maintenance update.\nService maintenance log."
        ask = EnumAsk({line: "1" for line in text.splitlines()}, final="1")
        result = self.mem.run_enum(self.mem.Doc(text), "What predominates?", ["maintenance", "delivery"], ask,
                                   block=1, budget=len(text.splitlines()[0]) + 1)
        self.assertEqual((result["verdict"], result["label"], result["read"]["basis"]),
                         ("answered", "maintenance", "vote"))
        self.assertEqual(len(result["read"]["sources"]), 1)

    def test_lines_cite_a_passage_whole_when_no_part_gives_the_answer_alone(self):
        # Reversed needs the invoice and the reversal together, so no half answers it.
        def ask(question, text, labels, options=None):
            both = "INV-42" in text and "reversed" in text
            label = "2" if not question.startswith("Going only by") or both else "0"
            return {"label": label, "p": {k: float(k == label) for k in labels}}
        doc = self.mem.Doc(self.STATUS)
        result = self.mem.run_enum(doc, "What is the status?", ["paid", "reversed"], ask, lines=True)
        self.assertEqual([(e["line"], e["end"]) for e in result["read"]["evidence"]], [(1, 4)])
        self.assertEqual(result["read"]["lines"]["calls"], 2)

    def test_none_requires_complete_passage_read(self):
        text = "Weather report.\nMaintenance schedule."
        _, ask, result = self.run_case(text, {s: "0" for s in text.splitlines()})
        self.assertEqual(result["verdict"], "none")
        self.assertEqual(result["label"], "none")
        self.assertEqual(len(ask.calls), 2)
        self.assertEqual(result["read"]["sources"], [])
        self.assertEqual(result["read"]["evidence"], [])

    def test_incomplete_stream_cannot_return_none(self):
        doc = self.mem.Doc("Weather report.", complete=False)
        ask = EnumAsk({doc.span(0): "0"})
        result = self.mem.run_enum(doc, "What is the invoice status?", ["paid", "reversed"], ask)
        self.assertEqual(result["verdict"], "insufficient")

    def test_a_read_reports_how_far_it_is_in_passages(self):
        doc = self.mem.Doc("Invoice INV-42 paid.\nWeather report.\nAnother note.")
        seen = []
        self.mem.run_enum(doc, "What is the status?", ["paid", "reversed"], EnumAsk({doc.span(0): "1"}), block=1,
                          progress=lambda done, total, unit: seen.append((done, total, unit)))
        self.assertEqual(seen, [(1, 3, "passages"), (2, 3, "passages"), (3, 3, "passages")])

    def test_required_leaf_failure_is_unscored(self):
        doc = self.mem.Doc("Invoice INV-42 paid.\nReader fails here.")
        ask = EnumAsk({doc.span(0): "1"}, fail="fails")
        result = self.mem.run_enum(doc, "What is the status?", ["paid", "reversed"], ask, block=1)
        self.assertEqual(result["verdict"], "unscored")
        self.assertEqual(result["read"]["failed"], 1)

    def test_expired_deadline_after_leaf_does_not_start_judge(self):
        doc = self.mem.Doc("Invoice INV-42 paid.")
        ask = EnumAsk({doc.span(0): "1"})
        clock = [0.0]

        def timed(*args):
            r = ask(*args)
            clock[0] += 1.0
            return r

        with patch.object(self.mem.time, "monotonic", side_effect=lambda: clock[0]):
            result = self.mem.run_enum(doc, "What is the status?", ["paid", "reversed"], timed,
                                       deadline=0.5)
        self.assertEqual(result["verdict"], "insufficient")
        self.assertEqual(len(ask.calls), 1)

    def test_majority_agreement_cannot_authorize_dropped_latest_fact(self):
        old = [f"2026-01-0{i} Invoice INV-42 paid." for i in range(1, 4)]
        latest = "2026-10-02 Invoice INV-42 payment reversed."
        doc = self.mem.Doc("\n".join(old + [latest]))
        leaves = {s: "1" for s in old} | {latest: "2"}
        scores = {latest: {"1": 0.2, "2": 0.7, "0": 0.1}}
        ask = EnumAsk(leaves, final="1", scores=scores)
        result = self.mem.run_enum(doc, "What is the latest status of INV-42?", ["paid", "reversed"],
                                   ask, block=1, budget=len(old[0]) + 1)
        self.assertEqual(result["verdict"], "insufficient")
        self.assertNotEqual(result["read"]["basis"], "complete")

    def test_support_that_cannot_fit_does_not_call_judge(self):
        _, ask, result = self.run_case("Invoice INV-42 paid.", {"Invoice INV-42 paid.": "1"}, budget=1)
        self.assertEqual(result["verdict"], "insufficient")
        self.assertEqual(len(ask.calls), 1)

    def test_final_judge_failure_is_unscored(self):
        doc = self.mem.Doc("Invoice INV-42 paid.")
        calls = []

        def ask(question, text, labels, options):
            calls.append(text)
            if len(calls) == 1:
                return {"label": "1", "p": {"1": 1.0, "2": 0.0, "0": 0.0}}
            return {"label": None, "unscored": "judge failed"}

        result = self.mem.run_enum(doc, "What is the status?", ["paid", "reversed"], ask)
        self.assertEqual(result["verdict"], "unscored")
        self.assertEqual(result["read"]["calls"], len(calls))

    def test_option_count_is_bounded(self):
        for names in ([], [str(i) for i in range(10)]):
            with self.subTest(names=names), self.assertRaises(ValueError):
                self.mem.run_enum(self.mem.Doc("Text."), "Choose.", names, EnumAsk({}))

    def test_transport_deadline_marker_is_insufficient(self):
        calls = []

        def ask(*args):
            calls.append(args)
            return {"label": None, "unscored": "timed out", "deadline": True}

        result = self.mem.run_enum(self.mem.Doc("Invoice INV-42 paid."), "What is the status?",
                                   ["paid", "unpaid"], ask)
        self.assertEqual(result["verdict"], "insufficient")
        self.assertEqual(len(calls), 1)

    def test_explicit_share_reducer_reports_vote_basis(self):
        text = "Service maintenance entry.\nService maintenance update.\nCustomer delivery entry."
        leaves = {text.splitlines()[0]: "1", text.splitlines()[1]: "1", text.splitlines()[2]: "2"}
        ask = EnumAsk(leaves, final="1")
        result = self.mem.run_enum(self.mem.Doc(text), "What predominates?", ["maintenance", "delivery"], ask,
                                   block=1, budget=len(text.splitlines()[0]) + 1, share=True)
        self.assertEqual(result["verdict"], "answered")
        self.assertEqual(result["label"], "maintenance")
        self.assertEqual(result["read"]["basis"], "vote")
        self.assertEqual(result["read"]["checked"], 3)
        self.assertEqual(len(result["read"]["sources"]), 1)

    def test_share_judge_disagreement_is_insufficient(self):
        text = "Service maintenance entry.\nService maintenance update.\nCustomer delivery entry."
        leaves = {text.splitlines()[0]: "1", text.splitlines()[1]: "1", text.splitlines()[2]: "2"}
        ask = EnumAsk(leaves, final="2")
        result = self.mem.run_enum(self.mem.Doc(text), "What predominates?", ["maintenance", "delivery"], ask,
                                   block=1, budget=len(text.splitlines()[0]) + 1, share=True)
        self.assertEqual(result["verdict"], "insufficient")
        self.assertIsNone(result["label"])


class EnumCliTests(unittest.TestCase):
    def cli(self, digit):
        fake = FakeOllama(chat([(digit, math.log(0.95))]))
        self.addCleanup(fake.close)
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        command = [sys.executable, str(Path(__file__).resolve().parents[1] / "mem.py"), "-j", "-e",
                   "paid=invoice settled,reversed=payment undone", "What is the invoice status?"]
        env = dict(os.environ, CLASSIF_HOSTS=fake.host, CLASSIF_CALIBRATION="/nonexistent",
                   XDG_CACHE_HOME=tmp.name)
        result = subprocess.run(command, input="Invoice INV-42 paid.\n", text=True, capture_output=True,
                                env=env, timeout=10)
        return fake, result

    def test_first_named_option_exits_zero_without_printing_description(self):
        import json
        fake, result = self.cli("1")
        self.assertEqual(result.returncode, 0, result.stderr)
        data = json.loads(result.stdout)
        self.assertEqual(data["label"], "paid")
        self.assertEqual(set(data["p"]), {"paid", "reversed", "none"})
        self.assertEqual(len(fake.requests), 2)
        self.assertIn("invoice settled", fake.requests[0]["messages"][-1]["content"])

    def test_second_named_option_exits_one(self):
        import json
        _, result = self.cli("2")
        self.assertEqual(result.returncode, 1, result.stderr)
        self.assertEqual(json.loads(result.stdout)["label"], "reversed")

    def test_none_exits_one_after_reading_the_input(self):
        import json
        fake, result = self.cli("0")
        self.assertEqual(result.returncode, 1, result.stderr)
        data = json.loads(result.stdout)
        self.assertEqual(data["verdict"], "none")
        self.assertEqual(data["label"], "none")
        self.assertEqual(data["read"]["checked"], data["read"]["passages"])
        self.assertEqual(len(fake.requests), 1)

    def test_incompatible_graph_and_enum_flags_are_rejected(self):
        command = [sys.executable, str(Path(__file__).resolve().parents[1] / "mem.py"),
                   "-g", "-e", "paid,unpaid", "What is the status?"]
        result = subprocess.run(command, input="Invoice INV-42 paid.", text=True, capture_output=True,
                                env=dict(os.environ, CLASSIF_HOSTS="127.0.0.1:1"), timeout=10)
        self.assertEqual(result.returncode, 2)
        self.assertIn("-g", result.stderr)
        self.assertIn("-e", result.stderr)

    def test_share_requires_an_enum_question(self):
        command = [sys.executable, str(Path(__file__).resolve().parents[1] / "mem.py"),
                   "--share", "The invoice is paid."]
        result = subprocess.run(command, input="Invoice INV-42 paid.", text=True, capture_output=True,
                                env=dict(os.environ, CLASSIF_HOSTS="127.0.0.1:1"), timeout=10)
        self.assertEqual(result.returncode, 2)
        self.assertIn("--share", result.stderr)
        self.assertIn("-e", result.stderr)


if __name__ == "__main__":
    unittest.main()
