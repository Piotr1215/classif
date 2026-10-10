"""evals/jev.py asks Jev and classif the same cases and measures how far
apart their answers are. These tests cover the parts that decide the
numbers: the request each primitive sends, how both answers become one
distribution, the distances, and the cache that keeps reruns off Jev."""
import importlib.machinery
import importlib.util
import tempfile
import unittest
from pathlib import Path

SCRIPT = Path(__file__).resolve().parent.parent / "evals" / "jev.py"


def load():
    loader = importlib.machinery.SourceFileLoader("jev_eval", str(SCRIPT))
    mod = importlib.util.module_from_spec(importlib.util.spec_from_loader("jev_eval", loader))
    loader.exec_module(mod)
    return mod


J = load()
CHOICE = {"task": "choice", "kind": "choice", "question": "Assess the claim.", "text": "It was not fitted.",
          "options": ["supported", "contradicted"],
          "criteria": {"supported": "establishes it", "contradicted": "establishes the opposite"}}
SCORE = {"task": "license", "kind": "score", "question": "How restrictive?", "path": "/x/GPL-3",
         "levels": ["permissive", "notice", "copyleft"]}


class RequestTests(unittest.TestCase):
    def test_each_kind_asks_jev_its_own_primitive(self):
        noul = {"task": "synthetic", "kind": "noul", "question": "Is it k8s?", "text": "pod", "options": ["yes", "no"]}
        self.assertEqual(J.jev_request(noul, "pod")["questions"]["q"], {"type": "noul", "instructions": "Is it k8s?"})
        self.assertEqual(J.jev_request(CHOICE, "x")["questions"]["q"]["criteria"], CHOICE["criteria"])
        req = J.jev_request(SCORE, "license text")
        self.assertEqual((req["state"], req["model"], req["questions"]["q"]["criteria"]),
                         ("license text", "jev-latest", ["permissive", "notice", "copyleft"]))

    def test_classif_gets_l_for_a_noul_e_for_a_choice_and_s_with_i_for_a_score(self):
        noul = {"kind": "noul", "question": "Is it k8s?", "text": "pod", "options": ["yes", "no"]}
        self.assertEqual(J.classif_args(noul), ["-l", "yes,no", "Is it k8s?", "pod"])
        self.assertEqual(J.classif_args(CHOICE), ["-e", "supported=establishes it\ncontradicted=establishes the opposite",
                                                  "Assess the claim.", "It was not fitted."])
        self.assertEqual(J.classif_args(SCORE), ["-s", "permissive", "-s", "notice", "-s", "copyleft",
                                                 "How restrictive?", "-i", "/x/GPL-3"])


class DistributionTests(unittest.TestCase):
    def test_a_noul_becomes_yes_and_no(self):
        self.assertEqual(J.jev_dist({"type": "noul", "noul": 0.75}, {"kind": "noul"}), {"yes": 0.75, "no": 0.25})

    def test_classifs_none_is_dropped_and_the_options_renormalized(self):
        got = J.classif_dist({"p": {"supported": 0.6, "contradicted": 0.2, "none": 0.2}}, CHOICE)
        self.assertEqual({k: round(v, 9) for k, v in got.items()}, {"supported": 0.75, "contradicted": 0.25})

    def test_total_variation_is_half_the_summed_gap(self):
        self.assertAlmostEqual(J.tv({"a": 0.5, "b": 0.5}, {"a": 1.0, "b": 0.0}), 0.5)
        self.assertAlmostEqual(J.tv({"a": 0.2, "b": 0.8}, {"b": 0.8, "a": 0.2}), 0.0)

    def test_the_spread_of_a_level_distribution_is_its_standard_deviation(self):
        self.assertAlmostEqual(J.level_sd({"0": 0.0, "1": 0.5, "2": 0.5}), 0.5)
        self.assertAlmostEqual(J.level_sd({"0": 0.0, "1": 1.0, "2": 0.0}), 0.0)


class CompareTests(unittest.TestCase):
    def test_a_score_is_within_when_its_gap_is_inside_jevs_own_spread(self):
        jev = {"score": 1.5, "probabilities": {"0": 0.0, "1": 0.5, "2": 0.5}}
        near = J.compare(SCORE, jev, {"score": 1.9, "probabilities": {"0": 0.0, "1": 0.1, "2": 0.9}})
        far = J.compare(SCORE, jev, {"score": 0.4, "probabilities": {"0": 0.6, "1": 0.4, "2": 0.0}})
        self.assertEqual((round(near["gap"], 3), near["sd"], near["within"]), (0.4, 0.5, True))
        self.assertEqual((far["within"], far["agree"]), (False, False))

    def test_a_sharp_jev_answer_still_allows_the_floor(self):
        jev = {"score": 2.0, "probabilities": {"0": 0.0, "1": 0.0, "2": 1.0}}
        row = J.compare(SCORE, jev, {"score": 1.95, "probabilities": {"0": 0.0, "1": 0.05, "2": 0.95}})
        self.assertTrue(row["within"])

    def test_a_choice_reports_the_distance_and_whether_the_top_agrees(self):
        row = J.compare(CHOICE, {"type": "choice", "probabilities": {"supported": 0.1, "contradicted": 0.9}},
                        {"p": {"supported": 0.3, "contradicted": 0.6, "none": 0.1}})
        self.assertEqual((round(row["tv"], 3), row["agree"]), (0.233, True))
        self.assertNotIn("within", row)


class CacheTests(unittest.TestCase):
    def test_a_repeated_request_is_answered_from_the_cache(self):
        calls = []

        def post(req):
            calls.append(req)
            return {"answers": {"q": {"type": "noul", "noul": 0.9}}, "model": "jev-1.13.0"}
        with tempfile.TemporaryDirectory() as d:
            req = {"state": "pod", "model": "jev-latest", "questions": {"q": {"type": "noul", "instructions": "k8s?"}}}
            first, again = J.jev_answer(req, d, post), J.jev_answer(dict(req), d, post)
            other = J.jev_answer({**req, "state": "bread"}, d, post)
        self.assertEqual((first, again), (first, first))
        self.assertEqual(len(calls), 2)
        self.assertEqual(other["noul"], 0.9)


if __name__ == "__main__":
    unittest.main()
