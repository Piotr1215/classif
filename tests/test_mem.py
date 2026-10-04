"""Independent checks for the external-memory executor's public contract."""
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import tempfile
import types
import unittest
from unittest.mock import patch


ROOT = Path(__file__).resolve().parents[1]
MODULE = ROOT / "mem.py"
FIXTURES = Path(__file__).with_name("fixtures") / "external-memory.json"


def load():
    spec = importlib.util.spec_from_file_location("classif_memory_test", MODULE)
    mod = importlib.util.module_from_spec(spec)
    # Dataclasses need the module available during class creation.
    import sys
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


class ScriptedAsk:
    """Supplies known leaf labels and a final answer, while recording calls.

    These tests check the executor, not the model's semantic accuracy.
    """
    def __init__(self, leaf, final="yes", fail=None, links=None):
        self.leaf = leaf
        self.final = final
        self.fail = fail
        self.links = links or {}
        self.calls = []

    def __call__(self, question, text, labels):
        self.calls.append((question, text, tuple(labels)))
        if self.fail and self.fail in text:
            return {"label": None, "unscored": "injected reader failure"}
        if question.startswith("The last line uses a word like"):
            label = self.links.get(text.split("Last line: ")[-1], "0")
        elif set(labels) == {"yes", "no", "unknown"}:
            label = self.final
            if question.startswith("Does any of these lines break this claim:"):
                label = {"yes": "no", "no": "yes", "unknown": "unknown"}[label]
        else:
            label = self.leaf[text.strip()]
        return {"label": label, "p": {k: float(k == label) for k in labels}}


@unittest.skipUnless(MODULE.exists(), "external-memory implementation is not present yet")
class DocumentTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.mem = load()

    def test_document_key_tracks_exact_bytes_and_edits(self):
        text = "\u2615 Invoice INV-42 paid.\n"
        self.assertEqual(self.mem.Doc(text).key, hashlib.sha256(text.encode()).hexdigest())
        self.assertNotEqual(self.mem.Doc(text).key, self.mem.Doc(text.rstrip()).key)
        self.assertNotEqual(self.mem.Doc("caf\u00e9").key, self.mem.Doc("cafe\u0301").key)

    def test_declared_character_spans_reproduce_unicode_source(self):
        text = "\u2615 Invoice INV-42 paid.\n\U0001f642 Customer C9 approved.\ncafe\u0301."
        doc = self.mem.Doc(text)
        self.assertEqual(doc.unit, "char")
        for i, (start, end) in enumerate(doc.spans):
            self.assertEqual(doc.span(i), text[start:end])

    def test_invalid_span_ids_are_rejected(self):
        doc = self.mem.Doc("first\nlast")
        for i in (-1, len(doc.spans)):
            with self.subTest(i=i), self.assertRaises((IndexError, ValueError)):
                doc.span(i)

    def test_find_reports_total_before_limit(self):
        doc = self.mem.Doc("INV-42 paid.\nINV-42 cancelled.\nINV-42 refunded.")
        ids, total = doc.find(["INV-42"], 1)
        self.assertEqual(total, 3)
        self.assertEqual(len(ids), 1)

    def test_identifier_lookup_casefolds_without_changing_source(self):
        doc = self.mem.Doc("Invoice inv-42 paid.")
        ids, total = doc.find(["INV-42"], 10)
        self.assertEqual(total, 1)
        self.assertEqual(doc.span(ids[0]), "Invoice inv-42 paid.")

    def test_unicode_identifier_is_searchable(self):
        doc = self.mem.Doc("Zam\u00f3wienie \u017b\u00d3\u0141\u0106-42 op\u0142acone.")
        ids, total = doc.find(["\u017b\u00d3\u0141\u0106-42"], 10)
        self.assertEqual(total, 1)
        self.assertTrue(ids)

    def test_the_identifier_chain_does_not_follow_a_name_most_lines_share(self):
        # Every invoice names SUP-1: followed, it would chain one invoice to the whole ledger.
        doc = self.mem.Doc("".join(f"Invoice INV-{n} issued by SUP-1.\nInvoice INV-{n} paid.\n" for n in range(30)))
        own = [i for i in range(len(doc.spans)) if "INV-5 " in doc.span(i)]
        self.assertEqual(sorted(self.mem.closure(doc, "Invoice INV-5 is paid.")[0]), own)
        # The question's own name is followed however many lines carry it.
        self.assertEqual(len(self.mem.closure(doc, "Every invoice from SUP-1 is paid.")[0]), 60)


@unittest.skipUnless(MODULE.exists(), "external-memory implementation is not present yet")
class ExecutorTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.mem = load()
        cls.corpus = json.loads(FIXTURES.read_text())

    def leaf_for(self, case):
        plan = case["operation"]
        lines = case["state"].splitlines()
        if plan in ("join", "semantic_join"):
            return {s: "related" for s in lines}
        if plan in ("count_distinct", "semantic_check"):
            return {s: "fits" for s in lines}
        want_unpaid = plan == "exists"
        return {s: "fits" if ("unpaid" in s) == want_unpaid else "breaks" for s in lines}

    def run_case(self, case, only=None):
        doc = self.mem.Doc(case["state"], complete=case.get("stream_complete", True))
        final = {"supported": "yes", "contradicted": "no", "insufficient": "yes"}[case["expected"]]
        links = {doc.span(1).strip(): "1"} if case["id"] == "pronoun-reversal" else {}
        ask = ScriptedAsk(self.leaf_for(case), final=final, links=links)
        result = self.mem.run(doc, case["question"], ask, plan=case["operation"], only=only)
        return doc, ask, result

    def test_operator_goldens_with_complete_and_partial_reads(self):
        for case in self.corpus["cases"]:
            with self.subTest(case=case["id"], coverage="full"):
                _, _, result = self.run_case(case)
                self.assertEqual(result["verdict"], case["expected"])
            with self.subTest(case=case["id"], coverage="partial"):
                _, _, result = self.run_case(case, only=case["partial_span_ids"])
                self.assertEqual(result["verdict"], case["partial_expected"])

    def test_default_program_cannot_stop_on_a_confident_partial_yes(self):
        doc = self.mem.Doc("Invoice INV-42 was paid.\nThat payment was reversed.")
        ask = ScriptedAsk({doc.span(0): "fits", doc.span(1): "breaks"}, final="yes")
        result = self.mem.run(doc, "Is INV-42 currently paid?", ask, only=[0])
        self.assertEqual(result["verdict"], "insufficient")
        self.assertFalse(result["read"]["scan_complete"])

    def test_auto_meets_every_golden_and_answers_from_part_only_where_the_claim_allows(self):
        for case in self.corpus["cases"]:
            doc = self.mem.Doc(case["state"], complete=case.get("stream_complete", True))
            final = {"supported": "yes", "contradicted": "no", "insufficient": "yes"}[case["expected"]]
            links = {doc.span(1).strip(): "1"} if case["id"] == "pronoun-reversal" else {}
            # A part settles a claim only under the early stop the root call picked.
            stop_at = case["operation"] if case["operation"] in ("exists", "all") else None
            part = case["partial_expected"] if stop_at else "insufficient"
            for only, want in ((None, case["expected"]), (case["partial_span_ids"], part)):
                with self.subTest(case=case["id"], only=only):
                    ask = ScriptedAsk(self.leaf_for(case), final=final, links=links)
                    self.assertEqual(self.mem.run(doc, case["question"], ask, only=only, stop_at=stop_at)["verdict"], want)

    def test_the_root_call_reads_the_question_alone_and_an_unsure_pick_reads_every_line(self):
        cases = [(label, 0.9, name) for label, (name, _) in self.mem.KINDS.items()] + [("1", 0.6, None)]
        for label, p, want in cases:
            seen = []

            def ask(question, text, labels, options=None):
                seen.append((question, text, labels, options))
                return {"label": label, "p": {l: (p if l == label else (1 - p) / 3) for l in labels}}
            with self.subTest(label=label, p=p):
                self.assertEqual(self.mem.kind("Is this about Georgiana?", ask)[0], want)
                self.assertEqual(seen[0][1], "Is this about Georgiana?")
                self.assertEqual(len(seen[0][3]), 4)

    @staticmethod
    def failure_reader(asked):
        def ask(question, text, labels):
            asked.append(question)
            if set(labels) == {"yes", "no", "unknown"}:
                label = "yes"
            elif question.startswith("The last line uses"):
                label = labels[-2]      # the line just before
            elif "reversed" in text:
                label = "breaks"
            else:
                label = "fits" if "failure" in text else "unrelated"
            return {"label": label, "p": {k: float(k == label) for k in labels}}
        return ask

    def test_auto_stops_at_a_witness_once_the_lines_after_it_are_read(self):
        doc = self.mem.Doc("Deployment failure: access denied.\n" + "".join(f"Note {i}: check passed.\n" for i in range(50)))
        asked = []
        result = self.mem.run(doc, "An entry reports a deployment failure.", self.failure_reader(asked), stop_at="exists")
        self.assertEqual((result["verdict"], result["read"]["basis"]), ("supported", "witness"))
        self.assertLess(len(asked), 10)

    def test_auto_reads_on_when_a_line_after_the_witness_breaks_it(self):
        doc = self.mem.Doc("Deployment failure: access denied.\nThat failure was reversed.\n"
                           + "".join(f"Note {i}: check passed.\n" for i in range(50)))
        asked = []
        result = self.mem.run(doc, "An entry reports a deployment failure.", self.failure_reader(asked), stop_at="exists")
        self.assertEqual(result["read"]["basis"], "complete")
        self.assertTrue(result["read"]["scan_complete"])

    def test_the_lines_an_answer_rests_on_come_back_as_line_numbers(self):
        doc = self.mem.Doc("Header line.\n\nDeployment failure: access denied.\nThat failure was reversed.\n"
                           + "".join(f"Note {i}: check passed.\n" for i in range(50)))
        result = self.mem.run(doc, "An entry reports a deployment failure.", self.failure_reader([]))
        self.assertEqual(result["read"]["evidence"],
                         [{"line": 3, "end": 4, "text": "Deployment failure: access denied.\nThat failure was reversed."}])

    def test_evidence_keeps_the_lines_indentation_lined_up(self):
        doc = self.mem.Doc('{\n  "T": 4.3,\n  "U": 2.7\n}\n')
        ids = [i for i, (s, e) in enumerate(doc.spans) if '"' in doc.text[s:e]]
        self.assertEqual(self.mem.evidence(doc, ids), [{"line": 2, "end": 3, "text": '"T": 4.3,\n"U": 2.7'}])
        doc = self.mem.Doc("x\n  a:\n    b\n")
        self.assertEqual(self.mem.evidence(doc, [1, 2]), [{"line": 2, "end": 3, "text": "a:\n  b"}])

    def test_a_read_reports_how_far_it_is_in_lines(self):
        doc = self.mem.Doc("".join(f"Note {i}: check passed.\n" for i in range(50)))
        seen = []
        self.mem.run(doc, "An entry reports a deployment failure.", self.failure_reader([]),
                     progress=lambda done, total, unit: seen.append((done, total, unit)))
        self.assertEqual(seen[-1], (50, 50, "lines"))
        self.assertEqual([d for d, _, _ in seen], sorted(d for d, _, _ in seen))

    def test_an_answer_no_line_settles_points_at_no_line(self):
        doc = self.mem.Doc("".join(f"Note {i}: check passed.\n" for i in range(50)))
        result = self.mem.run(doc, "An entry reports a deployment failure.", self.failure_reader([]))
        self.assertEqual(result["read"]["evidence"], [])

    def test_a_kind_without_an_early_stop_reads_every_line(self):
        doc = self.mem.Doc("Deployment failure: access denied.\n" + "".join(f"Note {i}: check passed.\n" for i in range(50)))
        for kind in ("state", "whole"):
            with self.subTest(kind=kind):
                result = self.mem.run(doc, "An entry reports a deployment failure.", self.failure_reader([]),
                                      stop_at=kind)
                self.assertEqual(result["read"]["basis"], "complete")
                self.assertTrue(result["read"]["scan_complete"])

    def test_code_notes_count_names_dates_levels_and_line_forms(self):
        log = "".join(f"2026-10-0{1 + i % 3} {'ERROR' if i % 50 == 0 else 'INFO'} Mr Darcy served req-{i}\n"
                      for i in range(200))
        facts = " ".join(self.mem.notes(self.mem.Doc(log)))
        self.assertIn("200 lines", facts)
        self.assertIn("Most named: Mr Darcy (200).", facts)
        self.assertIn("2026-10-01 to 2026-10-03", facts)
        self.assertIn("INFO 196, ERROR 4", facts)
        self.assertIn("(196)", facts)
        self.assertNotIn("req-0", facts)        # seen once: not among the most named

    def test_a_whole_text_question_is_one_call_over_facts_and_an_even_sample(self):
        doc = self.mem.Doc("".join(f"Chapter line {i}: Elizabeth walks with Jane.\n" for i in range(3000)))
        seen = []

        def ask(question, text, labels, options=None):
            seen.append((question, text))
            return {"label": "no", "p": {"yes": 0.05, "no": 0.9, "unknown": 0.05}}
        result = self.mem.whole(doc, "Is this about Georgiana?", ask)
        self.assertEqual((result["verdict"], result["label"], result["read"]["basis"]), ("contradicted", "no", "sample"))
        self.assertEqual(len(seen), 1)
        self.assertEqual(seen[0][0], "Is this about Georgiana?")
        self.assertIn("Most named: Elizabeth (3000), Jane (3000)", seen[0][1])
        judged = result["read"]["judged"]
        self.assertLessEqual(sum(len(doc.span(i)) for i in judged), self.mem.WHOLE_BUDGET)
        self.assertGreater(judged[-1], len(doc.spans) * 3 // 4)     # spread to the end, not only the start

    def test_passages_with_the_best_matches_are_read_first(self):
        # Ten matching lines open the text, so the first eight fill the head.
        # The witness shares no word with the claim; only the line before it
        # matches, and that passage sits at the end of the text.
        doc = self.mem.Doc("".join(f"Deployment {i} is scheduled.\n" for i in range(10))
                           + "".join(f"Note {i}: routine check passed without incident today.\n" for i in range(400))
                           + "Deployment 99 started.\nThe access check was denied and the job broke.\n")
        passages = []

        def ask(question, text, labels, options=None):
            if "Does this passage" in question:
                passages.append(text)
            label = ("yes" if set(labels) == {"yes", "no", "unknown"} else
                     "fits" if "broke" in text else "unrelated")
            return {"label": label, "p": {k: float(k == label) for k in labels}}
        result = self.mem.run(doc, "An entry reports a deployment failure.", ask, block=3000, stop_at="exists")
        self.assertEqual(result["read"]["basis"], "witness")
        first = next(k for k, t in enumerate(passages) if "broke" in t)
        self.assertLessEqual(first, 1)      # in document order it is the last of about eight

    @staticmethod
    def unsure(leaf, inverted):
        """Leaf labels by line; the judge says unknown to the claim and
        inverted to whether any line breaks it."""
        asked = []

        def ask(question, text, labels):
            asked.append(question)
            if set(labels) != {"yes", "no", "unknown"}:
                label = leaf[text.strip()]
            else:
                label = inverted if question.startswith("Does any of these lines") else "unknown"
            return {"label": label, "p": {k: float(k == label) for k in labels}}
        return ask, asked

    def test_auto_confirms_a_full_read_no_line_breaks_by_the_inverted_question(self):
        doc = self.mem.Doc("Invoice INV-A is paid.\nInvoice INV-B is paid.")
        ask, asked = self.unsure({doc.span(0).strip(): "fits", doc.span(1).strip(): "fits"}, "no")
        result = self.mem.run(doc, "Every invoice is paid.", ask)
        self.assertEqual((result["verdict"], result["label"]), ("supported", "yes"))
        self.assertTrue(asked[-1].startswith("Does any of these lines break"))

    def test_auto_does_not_invert_when_a_line_breaks_the_claim(self):
        doc = self.mem.Doc("Invoice INV-A is paid.\nInvoice INV-B is unpaid.")
        ask, asked = self.unsure({doc.span(0).strip(): "fits", doc.span(1).strip(): "breaks"}, "no")
        result = self.mem.run(doc, "Every invoice is paid.", ask)
        self.assertEqual(result["verdict"], "insufficient")
        self.assertFalse(any(q.startswith("Does any of these lines") for q in asked))

    def test_failed_required_reader_call_is_unscored(self):
        doc = self.mem.Doc("Invoice INV-A paid.\nInvoice INV-B unpaid.")
        ask = ScriptedAsk({doc.span(0): "fits"}, fail="INV-B")
        result = self.mem.run(doc, "Every invoice is paid.", ask, plan="all")
        self.assertEqual(result["verdict"], "unscored")
        self.assertGreater(result["read"]["failed"], 0)

    def test_join_keeps_related_rows_for_the_joint_read(self):
        case = next(c for c in self.corpus["cases"] if c["id"] == "join-four-facts")
        doc, ask, result = self.run_case(case)
        final = [c for c in ask.calls if set(c[2]) == {"yes", "no", "unknown"}]
        self.assertEqual(len(final), 1)
        for line in case["state"].splitlines():
            self.assertIn(line, final[0][1])
        self.assertEqual(set(result["read"]["judged"]), set(range(len(doc.spans))))

    def test_middle_amount_and_date_reach_the_judge(self):
        for state, needed in (
            ("Invoice INV-42 total 500 EUR.\nInvoice INV-42 total 1500 EUR.\nInvoice INV-42 total 700 EUR.", "1500"),
            ("Invoice INV-42 paid on 2026-01-01.\nInvoice INV-42 paid on 2026-10-02.\nInvoice INV-42 paid on 2026-02-01.", "2026-10-02"),
        ):
            with self.subTest(needed=needed):
                doc = self.mem.Doc(state)
                ask = ScriptedAsk({doc.span(i): "related" for i in range(len(doc.spans))})
                self.mem.run(doc, "Assess this invoice.", ask)
                final = [c for c in ask.calls if set(c[2]) == {"yes", "no", "unknown"}]
                self.assertEqual(len(final), 1)
                self.assertIn(needed, final[0][1])

    def test_support_over_budget_does_not_become_a_complete_verdict(self):
        doc = self.mem.Doc("Invoice INV-A paid.\nInvoice INV-B unpaid.")
        ask = ScriptedAsk({doc.span(i): "related" for i in range(len(doc.spans))})
        result = self.mem.run(doc, "Assess all invoices.", ask, budget=5)
        self.assertEqual(result["verdict"], "insufficient")
        final = [c for c in ask.calls if set(c[2]) == {"yes", "no", "unknown"}]
        if final:
            self.assertLessEqual(len(final[0][1]), 5)

    def test_report_counts_actual_calls_and_judged_sources(self):
        case = next(c for c in self.corpus["cases"] if c["id"] == "all-complete-success")
        doc, ask, result = self.run_case(case)
        report = result["read"]
        self.assertEqual(report["calls"], len(ask.calls))
        self.assertTrue(report["doc_complete"])
        self.assertTrue(report["scan_complete"])
        self.assertEqual(report["checked"], len(doc.spans))
        final = [c for c in ask.calls if set(c[2]) == {"yes", "no", "unknown"}][0]
        for i in report["judged"]:
            self.assertIn(doc.span(i), final[1])

    def test_expired_deadline_does_not_start_a_final_judge(self):
        doc = self.mem.Doc("Invoice INV-A paid.")
        ask = ScriptedAsk({doc.span(0): "fits"})
        clock = [0.0]

        def timed_ask(question, text, labels):
            result = ask(question, text, labels)
            clock[0] += 1.0
            return result

        with patch.object(self.mem.time, "monotonic", side_effect=lambda: clock[0]):
            result = self.mem.run(doc, "Every invoice is paid.", timed_ask, deadline=0.5)
        self.assertEqual(result["verdict"], "insufficient")
        self.assertFalse(any(set(c[2]) == {"yes", "no", "unknown"} for c in ask.calls))

    def test_transport_deadline_marker_is_insufficient_without_clock_roundoff(self):
        calls = []

        def ask(*args):
            calls.append(args)
            return {"label": None, "unscored": "timed out", "deadline": True}

        result = self.mem.run(self.mem.Doc("Invoice INV-42 paid."), "Is INV-42 paid?", ask)
        self.assertEqual(result["verdict"], "insufficient")
        self.assertEqual(len(calls), 1)

    def test_json_source_offsets_identify_exact_unicode_and_crlf_spans(self):
        text = "\u2615 Invoice INV-42 paid.\r\n\U0001f642 That payment was reversed.\r\n"
        doc = self.mem.Doc(text)
        ask = ScriptedAsk({doc.span(i).strip(): "related" for i in range(len(doc.spans))},
                          links={doc.span(1).strip(): "1"})
        result = self.mem.run(doc, "Assess invoice INV-42.", ask)
        read = result["read"]
        self.assertEqual(read["doc_key"], hashlib.sha256(text.encode()).hexdigest())
        self.assertEqual(read["unit"], "char")
        self.assertEqual({s["span"] for s in read["sources"]}, set(read["judged"]))
        for source in read["sources"]:
            self.assertEqual(text[source["start"]:source["end"]], doc.span(source["span"]))

    def test_full_scan_pairs_a_reversal_dismissed_by_the_reader(self):
        doc = self.mem.Doc("Invoice INV-42 was paid.\nThat payment was reversed.")
        ask = ScriptedAsk({doc.span(0): "fits", doc.span(1): "unrelated"}, final="no",
                          links={doc.span(1): "1"})
        result = self.mem.run(doc, "Invoice INV-42 is currently paid.", ask)
        self.assertEqual(result["verdict"], "contradicted")
        self.assertEqual(result["read"]["basis"], "complete")
        self.assertEqual(result["read"]["by_link"], 1)
        self.assertEqual(set(result["read"]["judged"]), {0, 1})
        self.assertIn(doc.span(1), ask.calls[-1][1])

    def test_full_scan_link_failure_cannot_become_a_verdict(self):
        doc = self.mem.Doc("Invoice INV-42 was paid.\nThat payment was reversed.")
        ask = ScriptedAsk({doc.span(0): "fits", doc.span(1): "unrelated"})

        def failed_link(question, text, labels):
            if question == self.mem.LINK_QUESTION:
                return {"label": None, "unscored": "link backend unavailable"}
            return ask(question, text, labels)

        result = self.mem.run(doc, "Invoice INV-42 is currently paid.", failed_link)
        self.assertEqual(result["verdict"], "unscored")
        self.assertIn("link", result["read"]["why"])
        self.assertFalse(any(set(c[2]) == set(self.mem.JUDGE) for c in ask.calls))

    def test_full_scan_link_deadline_cannot_become_a_verdict(self):
        doc = self.mem.Doc("Invoice INV-42 was paid.\nThat payment was reversed.")
        ask = ScriptedAsk({doc.span(0): "fits", doc.span(1): "unrelated"})

        def cut_link(question, text, labels):
            if question == self.mem.LINK_QUESTION:
                return {"label": None, "unscored": "timed out", "deadline": True}
            return ask(question, text, labels)

        result = self.mem.run(doc, "Invoice INV-42 is currently paid.", cut_link, deadline=10)
        self.assertEqual(result["verdict"], "insufficient")
        self.assertFalse(any(set(c[2]) == set(self.mem.JUDGE) for c in ask.calls))

    def passage_run(self, block_label):
        text = "\n".join(f"Routine maintenance entry {i}." for i in range(20))
        doc = self.mem.Doc(text)
        ask = ScriptedAsk({doc.span(i): "unrelated" for i in range(len(doc.spans))}, final="unknown")

        def passage_ask(question, text, labels):
            if "Does this passage hold a line" in question:
                ask.calls.append((question, text, tuple(labels)))
                if block_label is None:
                    return {"label": None, "unscored": "injected passage failure"}
                return {"label": block_label, "p": {k: float(k == block_label) for k in labels}}
            return ask(question, text, labels)

        result = self.mem.run(doc, "The transaction completed.", passage_ask, block=5000)
        return doc, ask, result

    def test_passage_dismissal_is_reported_separately_from_line_checks(self):
        doc, ask, result = self.passage_run("unrelated")
        read = result["read"]
        self.assertGreater(read["blocks"], 0)
        self.assertGreater(read["by_block"], 0)
        self.assertEqual(read["checked"], len(doc.spans))
        self.assertTrue(read["scan_complete"])
        self.assertEqual(read["calls"], len(ask.calls))
        self.assertEqual(result["verdict"], "insufficient")

    def test_related_passage_stays_one_passage_and_is_not_judge_material(self):
        """A passage that decides nothing by itself is counted, not read line
        by line: that fan-out is where a loose claim turned a scan into
        thousands of calls. Its lines are not packed for the judge."""
        doc, ask, result = self.passage_run("related")
        read = result["read"]
        self.assertGreater(read["blocks"], 0)
        self.assertEqual(read["by_block"], 0)
        self.assertGreater(read["related_blocks"], 0)
        self.assertEqual(read["checked"], len(doc.spans))
        self.assertEqual(read["calls"], len(ask.calls))
        alone = sum("Taken alone" in c[0] for c in ask.calls)
        self.assertLessEqual(alone, self.mem.HEAD)             # only the ranked head is read line by line
        self.assertEqual(read["counts"]["related"], len(doc.spans) - alone)
        self.assertEqual(read["support"], [])
        self.assertEqual(result["verdict"], "insufficient")

    def test_fitting_passage_is_narrowed_by_halves_not_read_line_by_line(self):
        text = "\n".join(f"Routine maintenance entry {i}." for i in range(400))
        doc = self.mem.Doc(text)
        hit = doc.span(333)
        ask = ScriptedAsk({doc.span(i): ("fits" if i == 333 else "unrelated") for i in range(len(doc.spans))})

        def passage_ask(question, text, labels):
            if "Does this passage hold a line" in question:
                ask.calls.append((question, text, tuple(labels)))
                label = "fits" if hit in text else "unrelated"
                return {"label": label, "p": {k: float(k == label) for k in labels}}
            return ask(question, text, labels)

        result = self.mem.run(doc, "Entry 333 completed.", passage_ask, block=5000)
        read = result["read"]
        self.assertEqual(result["verdict"], "supported")
        self.assertEqual(read["checked"], len(doc.spans))
        self.assertIn(333, read["support"])
        line_reads = sum("Taken alone" in c[0] for c in ask.calls)
        self.assertLess(line_reads, len(doc.spans) // 8)     # the head and a few lines near the hit, not the file
        self.assertGreater(read["blocks"], 2)                 # the passage, then halves of the flagged side

    def test_a_sentence_wrapped_across_lines_is_kept_where_no_line_fits_alone(self):
        # man bash wraps set -e over lines; alone, each only reads as related.
        top, bottom = "-e  Exit immediately if a pipeline (which may consist", "of one command) exits non-zero."
        text = "\n".join([f"Routine maintenance entry {i}." for i in range(333)] + [top, bottom]
                         + [f"Routine maintenance entry {i}." for i in range(335, 400)])
        doc = self.mem.Doc(text)
        asked = []

        def ask(question, text, labels, options=None):
            asked.append(question)
            whole = top in text and bottom in text
            if set(labels) == {"yes", "no", "unknown"}:
                label = "yes" if whole else "no"
            elif "Does this passage hold a line" in question:
                label = "fits" if whole else "related" if top in text else "unrelated"
            else:
                label = "related" if top in text else "unrelated"     # the tail alone says nothing of the shell
            return {"label": label, "p": {k: float(k == label) for k in labels}}
        for stop_at, basis in ((None, "complete"), ("exists", "witness")):
            with self.subTest(stop_at=stop_at):
                result = self.mem.run(doc, "The shell can be told to quit on a failure.", ask, block=5000,
                                      stop_at=stop_at)
                self.assertEqual((result["verdict"], result["read"]["basis"]), ("supported", basis))
                self.assertTrue({333, 334} <= set(result["read"]["judged"]))

    def test_failed_passage_cannot_become_unrelated_rows(self):
        _, ask, result = self.passage_run(None)
        self.assertEqual(result["verdict"], "unscored")
        self.assertGreater(result["read"]["failed"], 0)
        self.assertFalse(result["read"]["scan_complete"])
        self.assertFalse(any(set(c[2]) == {"yes", "no", "unknown"} for c in ask.calls))

class SearchTests(unittest.TestCase):
    """The search past the window: passages indexed once, the closest judged in one call."""

    @classmethod
    def setUpClass(cls):
        cls.mem = load()

    def test_chunks_cut_at_size_and_at_a_paragraph_end_once_half_full(self):
        doc = self.mem.Doc("a one\nb one\n\nc two\nd two\n\ne three\n")
        self.assertEqual(self.mem.chunks(doc, 12), [[0, 1], [2, 3], [4]])
        self.assertEqual(self.mem.chunks(doc, 1000), [[0, 1, 2, 3, 4]])
        self.assertEqual([i for c in self.mem.chunks(doc, 8) for i in c], [0, 1, 2, 3, 4])

    def test_stem_meets_word_forms_and_leaves_identifiers(self):
        stem = self.mem.stem
        self.assertEqual({stem("pay"), stem("pays"), stem("paying"), stem("invoices")}, {"pay", "invoice"})
        self.assertEqual((stem("INV-0042"), stem("is")), ("INV-0042", "is"))

    def test_the_index_ranks_the_closest_passage_by_vector_then_by_words(self):
        mem = self.mem
        text = "\n\n".join([f"Note {i}: routine check passed." for i in range(5)]
                            + ["Deployment failure: access denied.", "The build turned red."])
        idx = mem.Index(mem.Doc(text), size=10)
        self.assertEqual(len(idx.blks), 7)
        idx.vectors = [[0.0, 1.0]] * 6 + [[1.0, 0.0]]
        ranked = idx.rank("An entry reports a deployment failure.", qvec=[1.0, 0.0])
        self.assertEqual(ranked[:2], [5, 6])    # by words first, then by vector
        self.assertEqual(sorted(ranked), list(range(7)))
        self.assertEqual(idx.rank("An entry reports a deployment failure.")[0], 5)

    def test_search_answers_exists_yes_or_no_and_state_may_be_unknown(self):
        mem = self.mem
        text = "\n\n".join([f"Note {i}: routine check passed." for i in range(30)] + ["Deployment failure: access denied."])
        doc = mem.Doc(text)
        idx, asked = mem.Index(doc, size=10), []

        def ask(question, text, labels, options=None):
            asked.append((question, text, tuple(labels)))
            label = "yes" if "access denied" in text else ("no" if len(labels) == 2 else "unknown")
            return {"label": label, "p": {l: float(l == label) for l in labels}}
        r = mem.search(doc, idx, "An entry reports a deployment failure.", ask, mode="exists", budget=200)
        self.assertEqual((r["verdict"], r["label"], r["read"]["basis"], r["read"]["plan"]),
                         ("supported", "yes", "search", "search"))
        self.assertEqual(asked[-1][2], ("yes", "no"))
        self.assertIn("is this shown", asked[-1][0])
        self.assertEqual(r["read"]["by"], "words")
        self.assertIn(30, r["read"]["judged"])
        self.assertLess(r["read"]["checked"], r["read"]["passages"])
        r = mem.search(doc, idx, "Every check passed.", ask, mode="exists", budget=200)
        self.assertEqual((r["verdict"], r["label"]), ("contradicted", "no"))
        r = mem.search(doc, idx, "Every check passed.", ask, mode="state", budget=200)
        self.assertEqual((r["verdict"], r["label"], r["read"]["basis"]), ("insufficient", None, "none"))
        self.assertEqual(asked[-1][2], ("yes", "no", "unknown"))
        self.assertIn("closest passages do not settle", r["read"]["why"])

    def test_search_puts_the_lines_chained_to_the_questions_identifiers_before_the_passages(self):
        mem = self.mem
        lines = [f"Invoice INV-{i:04d} issued by SUP-{i % 9}." for i in range(400)]
        lines[20] = "Invoice INV-0020 issued by SUP-777."
        lines[390] = "Supplier SUP-777 is based in Berlin."
        doc = mem.Doc("\n".join(lines) + "\n")
        idx, asked = mem.Index(doc, size=200), []

        def ask(question, text, labels, options=None):
            asked.append(text)
            return {"label": "yes", "p": {"yes": 1.0, "no": 0.0}}
        r = mem.search(doc, idx, "Invoice INV-0020 came from a supplier based in Berlin.", ask, budget=1200)
        head = asked[0].split("\n\n")[0]
        self.assertIn("Invoice INV-0020 issued by SUP-777.", head)
        self.assertIn("Supplier SUP-777 is based in Berlin.", head)
        self.assertLessEqual({20, 390}, set(r["read"]["judged"]))
        self.assertLess(r["read"]["checked"], r["read"]["passages"])

    def test_saved_vectors_round_trip(self):
        with tempfile.TemporaryDirectory() as tmp:
            saved = self.mem.Saved(os.path.join(tmp, "mem.sqlite"), None)
            self.assertIsNone(saved.vectors("k"))
            saved.keep("k", [[0.5, 0.25], [1.0, 0.0]])
            self.assertEqual(saved.vectors("k"), [[0.5, 0.25], [1.0, 0.0]])

    def test_answer_searches_an_exists_claim_first_and_scans_only_when_a_no_rests_on_words(self):
        mem = self.mem
        long = "".join(f"Note {i}: routine check passed.\n" for i in range(300)) + "Deployment failure: access denied.\n"

        def judge(question, text, labels, options=None, model=None, host=None, num_ctx=None, timeout=None,
                  context=None):
            if question == mem.KIND_QUESTION:
                label = "4"
            elif "Taken alone" in question or "Does this passage" in question:
                label = "fits" if "failure" in text else "unrelated"
            else:
                label = "yes" if "failure" in question and "access denied" in text else "no"
            return {"label": label, "p": {l: (0.9 if l == label else 0.1 / max(1, len(labels) - 1)) for l in labels}}

        def embed(texts, model=None, host=None, timeout=None, query=False):
            return {"vectors": [[1.0, 0.0] if "failure" in t else [0.0, 1.0] for t in texts], "tokens": len(texts)}
        klass = types.SimpleNamespace(judge=judge, embed=embed, EMBED_MODEL="e", NUM_CTX=32768)
        r = mem.answer(klass, long, "An entry reports a deployment failure.", "h", "m", cache=False)
        self.assertEqual((r["verdict"], r["read"]["basis"], r["read"]["kind"]["kind"]), ("supported", "search", "exists"))
        self.assertEqual(r["read"]["calls"], 2)     # the root call and the one judge call
        self.assertEqual(r["read"]["index"]["passages"], r["read"]["passages"])
        r = mem.answer(klass, long, "An entry reports a deployment failure.", "h", "m", cache=False, evidence=True)
        self.assertEqual(r["read"]["evidence"], [{"line": 301, "end": 301, "text": "Deployment failure: access denied."}])
        self.assertEqual(r["read"]["lines"]["verdict"], "supported")
        # Not shown among the closest passages by meaning: no, with no full read.
        r = mem.answer(klass, long, "An entry reports a disk on fire.", "h", "m", cache=False)
        self.assertEqual((r["verdict"], r["label"], r["read"]["calls"]), ("contradicted", "no", 2))
        # By words alone a no is not trusted: the scan runs.
        gone = types.SimpleNamespace(judge=judge, EMBED_MODEL="e", NUM_CTX=32768,
                                     embed=lambda *a, **k: {"vectors": None, "unscored": "HTTP 404", "missing": True})
        r = mem.answer(gone, long, "An entry reports a disk on fire.", "h", "m", cache=False)
        self.assertTrue(r["read"]["scan_complete"])
        self.assertEqual((r["read"]["search"]["verdict"], r["read"]["search"]["index"]["missing"]), ("contradicted", True))
        self.assertGreater(r["read"]["calls"], 2)
        # The passages embedded but the question did not: by words alone again.
        half = types.SimpleNamespace(judge=judge, EMBED_MODEL="e", NUM_CTX=32768,
                                     embed=lambda texts, **k: {"vectors": None, "unscored": "cut"} if k.get("query")
                                     else embed(texts))
        r = mem.answer(half, long, "An entry reports a disk on fire.", "h", "m", cache=False)
        self.assertTrue(r["read"]["scan_complete"])
        self.assertNotIn("missing", r["read"]["search"]["index"])


if __name__ == "__main__":
    unittest.main()
