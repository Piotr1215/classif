"""Checks for question-independent links, reuse and component boundaries."""
import math
from pathlib import Path
import tempfile
import unittest

from test_mem import ScriptedAsk, load


class LinkAsk:
    def __init__(self, label="1", fail=False):
        self.label, self.fail, self.calls = label, fail, []

    def __call__(self, question, text, labels, options=None):
        self.calls.append((question, text, tuple(labels), tuple(options or [])))
        if self.fail:
            return {"label": None, "unscored": "injected link failure"}
        logp = {l: math.log(0.99 if l == self.label else 0.01 / (len(labels) - 1)) for l in labels}
        p = {l: math.exp(v) for l, v in logp.items()}
        return {"label": self.label, "p": p, "logp": logp, "mass": 1.0}


class GraphTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.mem = load()

    def test_anaphor_with_its_own_identifier_still_links(self):
        doc = self.mem.Doc("Supplier SUP-1 operates.\nThat supplier issued INV-42.")
        graph, ask = self.mem.Graph(doc), LinkAsk()
        ids, read = graph.component("Is INV-42 approved?", ask)
        self.assertEqual(ids, [0, 1])
        self.assertEqual(read["resolved"], 1)
        self.assertEqual(len(ask.calls), 1)

    def test_second_question_reuses_existing_edge(self):
        doc = self.mem.Doc("Supplier SUP-1 operates.\nThat supplier issued INV-42.")
        graph, ask = self.mem.Graph(doc), LinkAsk()
        first = graph.component("Is INV-42 approved?", ask)
        second = graph.component("Is SUP-1 active?", ask)
        self.assertEqual(first[0], second[0])
        self.assertEqual(len(ask.calls), 1)
        self.assertNotIn("approved", ask.calls[0][0])
        self.assertNotIn("active", ask.calls[0][0])

    def test_proper_name_anchor_survives_capitalized_question_prefix(self):
        doc = self.mem.Doc("Acme Ltd supplies parts.\nThat supplier issued INV-42.")
        ids, _ = self.mem.Graph(doc).component("Is Acme Ltd approved?", LinkAsk())
        self.assertEqual(ids, [0, 1])

    def test_unresolved_reference_is_visible(self):
        doc = self.mem.Doc("Supplier SUP-1 operates.\nThat supplier issued INV-42.")
        ids, read = self.mem.Graph(doc).component("Is INV-42 paid?", LinkAsk(label="0"))
        self.assertEqual(ids, [1])
        self.assertEqual(read["unresolved"], 1)

    def test_window_does_not_claim_distant_reference_discovery(self):
        doc = self.mem.Doc("Supplier SUP-1 operates.\nRoutine weather report.\nThat supplier issued INV-42.")
        ask = LinkAsk(label="0")
        ids, read = self.mem.Graph(doc, window=1).component("Is INV-42 paid?", ask)
        self.assertEqual(ids, [2])
        self.assertEqual(read["window"], 1)
        self.assertEqual(read["unresolved"], 1)
        self.assertNotIn(doc.span(0), ask.calls[0][1])

    def test_link_failure_cannot_be_reported_as_a_supported_component(self):
        doc = self.mem.Doc("Supplier SUP-1 operates.\nThat supplier issued INV-42.")
        graph, link = self.mem.Graph(doc), LinkAsk(fail=True)
        ask = ScriptedAsk({doc.span(1): "fits"})

        def leaf(question, text, labels):
            return link(question, text, labels) if question == self.mem.LINK_QUESTION else ask(question, text, labels)

        result = self.mem.run(doc, "Is INV-42 paid?", ask, graph=graph, leaf=leaf)
        self.assertEqual(result["read"]["graph"]["failed"], 1)
        self.assertEqual(result["verdict"], "unscored")

    def test_component_answer_carries_component_basis_and_exact_sources(self):
        text = "\u2615 Invoice INV-42 paid.\r\nRoutine weather report.\r\n"
        doc = self.mem.Doc(text)
        ask = ScriptedAsk({doc.span(0).strip(): "fits"})
        result = self.mem.run(doc, "Is INV-42 paid?", ask, graph=self.mem.Graph(doc))
        self.assertEqual(result["verdict"], "supported")
        self.assertEqual(result["read"]["basis"], "component")
        self.assertFalse(result["read"]["scan_complete"])
        self.assertEqual(result["read"]["checked"], 1)
        for source in result["read"]["sources"]:
            self.assertEqual(text[source["start"]:source["end"]], doc.span(source["span"]))

    def test_global_plans_reject_component_scope(self):
        doc = self.mem.Doc("Invoice INV-42 paid.\nInvoice INV-43 unpaid.")
        graph = self.mem.Graph(doc)
        for plan in ("all", "count_distinct"):
            with self.subTest(plan=plan), self.assertRaises(ValueError):
                self.mem.run(doc, "Assess all invoices.", ScriptedAsk({}), plan=plan, graph=graph)

    def test_missing_anchor_returns_insufficient_without_a_judge(self):
        doc = self.mem.Doc("Routine weather report.")
        ask = ScriptedAsk({})
        result = self.mem.run(doc, "Is INV-42 paid?", ask, graph=self.mem.Graph(doc))
        self.assertEqual(result["verdict"], "insufficient")
        self.assertEqual(ask.calls, [])

    def saved_probe(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        saved = self.mem.Saved(str(Path(tmp.name) / "links.sqlite"), self.mem.load_class())
        self.addCleanup(saved.db.close)
        source = LinkAsk()
        ident = {"model": "fixture-model", "digest": "v1", "num_ctx": 2048}
        return saved, source, saved.wrap(source, "fixture-model", ident)

    def test_rebuilt_graph_reuses_disk_link_for_a_different_question(self):
        saved, source, ask = self.saved_probe()
        text = "Supplier SUP-1 operates.\nThat supplier issued INV-42."
        first = self.mem.Graph(self.mem.Doc(text)).component("Is INV-42 paid?", ask)
        second = self.mem.Graph(self.mem.Doc(text)).component("Is SUP-1 active?", ask)
        self.assertEqual(first[0], second[0])
        self.assertEqual(len(source.calls), 1)
        self.assertEqual(saved.hits, 1)

    def test_source_edit_invalidates_link(self):
        saved, source, ask = self.saved_probe()
        for ref in ("That supplier issued INV-42.", "That supplier withdrew INV-42."):
            self.mem.Graph(self.mem.Doc("Supplier SUP-1 operates.\n" + ref)).component("INV-42", ask)
        self.assertEqual(len(source.calls), 2)
        self.assertEqual(saved.hits, 0)

    def test_candidate_tail_edit_invalidates_link(self):
        saved, source, ask = self.saved_probe()
        prefix = "Supplier SUP-1 " + "routine details " * 12
        self.assertGreater(len(prefix), 160)
        for tail in ("owns the disputed shipment.", "does not own the disputed shipment."):
            doc = self.mem.Doc(prefix + tail + "\nThat supplier issued INV-42.")
            self.mem.Graph(doc).component("INV-42", ask)
        self.assertEqual(len(source.calls), 2)
        self.assertEqual(saved.hits, 0)

    def test_linking_obeys_deadline_before_another_call_starts(self):
        from unittest.mock import patch
        doc = self.mem.Doc("Supplier SUP-1 operates.\nThat supplier issued INV-42.\nIt withdrew INV-42.")
        ask, clock = LinkAsk(), [0.0]

        def timed(question, text, labels):
            result = ask(question, text, labels)
            clock[0] += 1.0
            return result

        with patch.object(self.mem.time, "monotonic", side_effect=lambda: clock[0]):
            result = self.mem.run(doc, "Is INV-42 paid?", timed, graph=self.mem.Graph(doc), deadline=0.5)
        self.assertEqual(result["verdict"], "insufficient")
        self.assertEqual(len(ask.calls), 1)
        self.assertEqual(result["read"]["calls"], 1)
        self.assertTrue(result["read"]["graph"]["cut"])

    def test_second_question_reports_no_new_link_calls(self):
        graph = self.mem.Graph(self.mem.Doc("Supplier SUP-1 operates.\nThat supplier issued INV-42."))
        ask = LinkAsk()
        graph.component("INV-42", ask)
        _, read = graph.component("SUP-1", ask)
        self.assertEqual(read["link_calls"], 0)

    def test_graph_cannot_be_used_with_a_different_source(self):
        original = self.mem.Doc("Invoice INV-42 paid.\nWeather report.")
        edited = self.mem.Doc("Weather report.\nInvoice INV-42 payment reversed.")
        with self.assertRaises(ValueError):
            self.mem.run(edited, "Is INV-42 paid?", ScriptedAsk({}), graph=self.mem.Graph(original))

    def test_transport_deadline_marker_stops_linking_and_leaf_calls(self):
        doc = self.mem.Doc("Supplier SUP-1 operates.\nThat supplier issued INV-42.\nIt withdrew INV-42.")
        graph, calls = self.mem.Graph(doc), []

        def ask(*args):
            calls.append(args)
            return {"label": None, "unscored": "timed out", "deadline": True}

        result = self.mem.run(doc, "Is INV-42 paid?", ask, graph=graph)
        self.assertEqual(result["verdict"], "insufficient")
        self.assertEqual(len(calls), 1)
        self.assertEqual(graph.links, {})

    def test_common_name_hub_is_reported_when_not_followed(self):
        text = "Invoice INV-42 belongs to Acme Ltd.\n" + "\n".join(
            f"Account ACC-{i} belongs to Acme Ltd." for i in range(self.mem.HUB + 1))
        graph = self.mem.Graph(self.mem.Doc(text))
        ids, read = graph.component("Is INV-42 paid?", LinkAsk(label="0"))
        self.assertEqual(ids, [0])
        self.assertIn("acme ltd", read["hubs"])

    def test_question_anchor_is_followed_even_when_it_is_a_hub(self):
        text = "\n".join(f"Account ACC-{i} belongs to Acme Ltd." for i in range(self.mem.HUB + 1))
        graph = self.mem.Graph(self.mem.Doc(text))
        ids, _ = graph.component("Is Acme Ltd active?", LinkAsk(label="0"))
        self.assertEqual(len(ids), self.mem.HUB + 1)

    def test_three_hops_reach_four_fact_policy_chain(self):
        text = "INV-42 belongs to C9.\nC9 has category A7.\nA7 follows P7.\nP7 requires approval."
        graph = self.mem.Graph(self.mem.Doc(text))
        ids, read = graph.component("Does INV-42 require approval?", LinkAsk())
        self.assertEqual(ids, [0, 1, 2, 3])
        self.assertEqual(read["pending"], 0)

    def test_hop_limit_reports_known_pending_spans(self):
        text = "INV-42 belongs to C9.\nC9 has category A7.\nA7 follows P7.\nP7 requires approval."
        graph = self.mem.Graph(self.mem.Doc(text))
        ids, read = graph.component("Does INV-42 require approval?", LinkAsk(), hops=1)
        self.assertEqual(ids, [0, 1])
        self.assertEqual(read["pending"], 1)


if __name__ == "__main__":
    unittest.main()
