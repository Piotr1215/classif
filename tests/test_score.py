"""-s asks a Score question, modeled on Jev's Score: ordered levels, low to
high, answered in one call as digits 0 to n-1. The score is each level's
number times its probability, summed; the confidence is Jev's Score formula,
which counts how far probability sits from the likeliest level. These tests
run the real command against a fake Ollama whose level probabilities are set
by words in the text, so every number below is computed by hand."""
import json
import math
import tempfile
import unittest
from pathlib import Path

from test_judge import FakeOllama, chat, load, run

LEVELS = ["-s", "Cosmetic; no impact", "-s", "Broken, but a workaround exists", "-s", "Blocking; no workaround"]


def model(rules, default=(0.2, 0.6, 0.2)):
    """A fake model: the first rule whose word is in the text sets the level
    probabilities, else default. A text holding UNSCORED has no label mass."""
    def answer(req):
        user = req["messages"][-1]["content"]
        text = user.rsplit("\n\n", 1)[0]
        if "UNSCORED" in text:
            return chat([("the", 0.0)])
        p = next((p for word, p in rules if word in text), default)
        return chat([(str(i), math.log(v)) for i, v in enumerate(p) if v > 0])
    return answer


class ScoreMathTests(unittest.TestCase):
    cls = load()

    def test_the_score_is_each_levels_number_times_its_probability_summed(self):
        self.assertAlmostEqual(self.cls.level_score([0.0, 0.57, 0.43]), 1.43)
        self.assertAlmostEqual(self.cls.level_score([0.0, 0.14, 0.86, 0.0, 0.0]), 1.86)

    def test_confidence_follows_jevs_score_formula(self):
        # The worked examples on docs.typesafe.ai/confidence and /primitives/score.
        for p, want in [([0.0, 0.5, 0.5], 0.25), ([0.5, 0.0, 0.5], 0.0), ([0.0, 0.0, 0.48, 0.52], 0.52),
                        ([0.0, 0.74, 0.26], 0.61), ([0.0, 0.0, 0.0, 1.0], 1.0), ([1 / 3] * 3, 0.0)]:
            self.assertAlmostEqual(self.cls.score_confidence(p), want, places=2, msg=p)

    def test_mass_on_a_neighbour_costs_less_than_mass_at_the_far_end(self):
        self.assertGreater(self.cls.score_confidence([0.6, 0.4, 0.0, 0.0]),
                           self.cls.score_confidence([0.6, 0.0, 0.0, 0.4]))


class ScoreTests(unittest.TestCase):
    def serve(self, *rules, default=(0.2, 0.6, 0.2)):
        fake = FakeOllama(model(rules, default))
        self.addCleanup(fake.close)
        return fake

    def test_prints_the_score_and_its_confidence_from_one_call(self):
        fake = self.serve(("Safari", (0.0, 0.6, 0.4)))
        r = run(fake.host, "How severe is it?", "The export crashes in Safari", *LEVELS)
        self.assertEqual((r.returncode, r.stdout), (0, "1.40 0.40\n"))
        self.assertEqual(len(fake.requests), 1)

    def test_the_levels_reach_the_model_as_digits_low_to_high(self):
        fake = self.serve()
        run(fake.host, "How severe is it?", "a report", *LEVELS)
        req = fake.requests[0]
        self.assertIn("How severe is it? Options: 0=Cosmetic; no impact, 1=Broken, but a workaround exists, "
                      "2=Blocking; no workaround. Answer 0, 1 or 2.", req["messages"][-1]["content"])
        self.assertEqual(req["options"]["num_predict"], 1)

    def test_json_carries_jevs_answer_fields(self):
        fake = self.serve(("Safari", (0.0, 0.6, 0.4)))
        r = run(fake.host, "How severe is it?", "The export crashes in Safari", *LEVELS, "-j")
        out = json.loads(r.stdout)
        self.assertEqual({k: out[k] for k in ("type", "score", "confidence", "legend", "probabilities")}, {
            "type": "score", "score": 1.4, "confidence": 0.4,
            "legend": {"0": "Cosmetic; no impact", "1": "Broken, but a workaround exists", "2": "Blocking; no workaround"},
            "probabilities": {"0": 0.0, "1": 0.6, "2": 0.4}})

    def test_why_shows_where_the_probability_went_and_the_lines_that_moved_the_score(self):
        fake = self.serve(("Safari", (0.0, 0.6, 0.4)), default=(0.2, 0.6, 0.2))
        r = run(fake.host, "How severe is it?", "The export crashes\nonly in Safari", *LEVELS, "-w")
        self.assertEqual(r.stdout, "1.40 0.40\n"
                                   "where the probability went\n"
                                   "  0  0.00  Cosmetic; no impact\n"
                                   "  1  0.60  Broken, but a workaround exists\n"
                                   "  2  0.40  Blocking; no workaround\n"
                                   "why: the lines whose removal moves the score most\n"
                                   "  +0.400  only in Safari\n")

    def test_under_t_the_confidence_is_unsure_and_exits_3(self):
        fake = self.serve(("Safari", (0.0, 0.6, 0.4)))
        r = run(fake.host, "How severe is it?", "The export crashes in Safari", *LEVELS, "-t", "0.5")
        self.assertEqual((r.returncode, r.stdout), (3, "1.40 0.40 unsure\n"))

    def test_s_needs_2_to_10_levels(self):
        fake = self.serve()
        for levels in (["-s", "only"], [x for i in range(11) for x in ("-s", f"level {i}")]):
            r = run(fake.host, "How severe?", "text", *levels)
            self.assertEqual(r.returncode, 2)
            self.assertIn("-s needs 2 to 10 levels", r.stderr)
        self.assertEqual(fake.requests, [])

    def test_s_refuses_e_and_p(self):
        fake = self.serve()
        self.assertEqual(run(fake.host, "How severe?", "text", *LEVELS, "-e", "a,b").returncode, 2)
        r = run(fake.host, "How severe?", "text", *LEVELS, "-p")
        self.assertEqual(r.returncode, 2)
        self.assertIn("-p", r.stderr)

    def test_no_label_mass_is_unscored(self):
        fake = self.serve()
        r = run(fake.host, "How severe?", "UNSCORED", *LEVELS)
        self.assertEqual(r.returncode, 2)


class ScoreCalibrationTests(unittest.TestCase):
    def calibration(self, table):
        d = self.enterContext(tempfile.TemporaryDirectory())
        path = Path(d) / "calibration.json"
        path.write_text(json.dumps({load().DEFAULT_MODEL: table}))
        return str(path)

    def serve(self):
        fake = FakeOllama(model([], default=(0.2, 0.6, 0.2)))
        self.addCleanup(fake.close)
        return fake

    def test_a_score_uses_the_temperature_fitted_on_scores(self):
        fake = self.serve()
        cal = self.calibration({"T": 2.0, "T_enum": 4.0, "T_score": 1.5})
        out = json.loads(run(fake.host, "How severe?", "text", *LEVELS, "-j", calibration=cal).stdout)
        e = [v ** (1 / 1.5) for v in (0.2, 0.6, 0.2)]
        self.assertEqual(out["T"], 1.5)
        self.assertAlmostEqual(out["probabilities"]["1"], e[1] / sum(e), places=3)

    def test_without_a_score_temperature_a_score_keeps_the_e_one(self):
        fake = self.serve()
        out = json.loads(run(fake.host, "How severe?", "text", *LEVELS, "-j",
                             calibration=self.calibration({"T": 2.0, "T_enum": 4.0})).stdout)
        self.assertEqual(out["T"], 4.0)


class EachScoreTests(unittest.TestCase):
    def serve(self, *rules):
        fake = FakeOllama(model(rules))
        self.addCleanup(fake.close)
        return fake

    def test_items_sort_by_score_with_the_confidence_of_each(self):
        fake = self.serve(("GPL", (0.0, 0.1, 0.9)), ("MIT", (0.9, 0.1, 0.0)))
        r = run(fake.host, "each", "How restrictive?", "MIT terms", "GPL terms", "other", *LEVELS)
        self.assertEqual(r.returncode, 0)
        self.assertEqual(r.stdout, "score  conf  item\n"
                                   "1.90   0.85  GPL terms\n"
                                   "1.00   0.40  other\n"
                                   "0.10   0.85  MIT terms\n")

    def test_json_items_carry_score_confidence_and_probabilities(self):
        fake = self.serve(("GPL", (0.0, 0.1, 0.9)))
        r = run(fake.host, "each", "How restrictive?", "GPL terms", *LEVELS, "-j")
        out = json.loads(r.stdout)
        self.assertEqual(out["legend"], {"0": "Cosmetic; no impact", "1": "Broken, but a workaround exists",
                                         "2": "Blocking; no workaround"})
        self.assertEqual(out["items"][0], {"name": "GPL terms", "score": 1.9, "confidence": 0.85,
                                           "probabilities": {"0": 0.0, "1": 0.1, "2": 0.9}})

    def test_why_shows_where_the_top_picks_probability_went(self):
        fake = self.serve(("GPL", (0.0, 0.1, 0.9)))
        r = run(fake.host, "each", "How restrictive?", "GPL terms", "other", *LEVELS, "-w")
        self.assertIn("where the probability went for GPL terms\n"
                      "  0  0.00  Cosmetic; no impact\n"
                      "  1  0.10  Broken, but a workaround exists\n"
                      "  2  0.90  Blocking; no workaround\n", r.stdout)


if __name__ == "__main__":
    unittest.main()
