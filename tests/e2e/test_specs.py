"""Specs and the subcommands that read them, against a live model: specs,
classify, smoke, histogram and unload. Each spec is
written into the test's own spec dir and names the routed model, so any model
that answers plain questions passes."""
import json
import re
import unittest
import urllib.request

from cli import E2E, HOST, MODEL, judge, judgement

GREETINGS = [
    {"message": "Hello there!", "expect": "yes"},
    {"message": "Good morning, everyone.", "expect": "yes"},
    {"message": "Hi Anna, nice to see you.", "expect": "yes"},
    {"message": "Hey, how are you doing?", "expect": "yes"},
    {"message": "The invoice is overdue by ten days.", "expect": "no"},
    {"message": "Restart the server after the patch.", "expect": "no"},
    {"message": "The quarterly report is attached.", "expect": "no"},
    {"message": "Water boils at 100 degrees Celsius.", "expect": "no"},
]


class SpecBase(E2E):
    def spec(self, name, **over):
        d = {"question": "Is this message a greeting?", "text": "{message}", "fields": {"message": 200},
             "collapse": ["message"], "labels": ["yes", "no"], "model": MODEL, "hosts": [HOST],
             "num_ctx": judge.NUM_CTX, "threshold": {}, "smoke": GREETINGS, **over}
        (self.tmp / "specs" / f"{name}.json").write_text(json.dumps(d))

    def classify(self, name, *fields, hook="e2e", env=None):
        r = self.cli("classify", name, "--hook", hook, *fields, env=env)
        return r, (json.loads(r.out) if r.out.strip() else {})


class Specs(SpecBase):
    def test_specs_lists_the_spec_dir_and_nothing_else(self):
        self.spec("greeting")
        self.spec("other")
        listed = {line.split()[0] for line in self.cli("specs").out.splitlines() if line.rstrip().endswith(".json")}
        self.assertEqual(listed, {"greeting", "other"})

    def test_classify_prints_json_exits_by_label_and_logs_under_the_hook(self):
        self.spec("greeting")
        yes, d_yes = self.classify("greeting", "message=Hello there!")
        no, d_no = self.classify("greeting", "message=The invoice is overdue.")
        self.assertEqual((yes.rc, d_yes["label"], d_yes["decision"]), (0, "yes", "shadow"))
        self.assertEqual((no.rc, d_no["label"]), (1, "no"))
        rows = [json.loads(line) for line in (self.tmp / "state" / "log.jsonl").read_text().splitlines()]
        self.assertEqual([(r["spec"], r["hook"]) for r in rows], [("greeting", "e2e")] * 2)

    def test_smoke_passes_a_well_posed_spec(self):
        self.spec("greeting")
        r = self.cli("smoke", "greeting")
        self.assertEqual(r.rc, 0, r.out)
        self.assertTrue(r.out.rstrip().endswith(": pass"), r.out)

    def test_smoke_fails_when_rows_land_on_the_wrong_label(self):
        flipped = [{**row, "expect": "no" if row["expect"] == "yes" else "yes"} for row in GREETINGS]
        self.spec("flipped", smoke=flipped)
        r = self.cli("smoke", "flipped")
        right, low = map(int, re.search(r"^(\d+)/8 expected labels, (\d+) under mass", r.out, re.M).groups())
        self.assertEqual((r.rc, low), (1, 0), r.out)
        self.assertLess(right, 7, r.out)

    def test_a_field_past_its_cap_never_reaches_the_model(self):
        self.spec("password", question="Does this text mention a password?", fields={"message": 20}, smoke=[])
        _, cut = self.classify("password", "message=The weather is nice. The password is swordfish.")
        _, kept = self.classify("password", "message=The password is swordfish. The weather is nice.")
        self.assertEqual((cut["label"], kept["label"]), ("no", "yes"))

    def test_a_missing_field_is_unscored(self):
        self.spec("greeting")
        r, d = self.classify("greeting", "other=x")
        self.assertEqual((r.rc, d["label"], d["unscored"]), (2, None, "empty field: message"))

    def test_a_template_naming_an_unknown_field_is_unscored(self):
        self.spec("typo", text="{mesage}")
        r, d = self.classify("typo", "message=x")
        self.assertEqual((r.rc, d["unscored"]), (2, "spec template: unknown field 'mesage'"))

    def test_an_unknown_spec_names_the_specs_there_are(self):
        self.spec("greeting")
        r = self.cli("classify", "nosuch", "message=x")
        self.assertEqual(r.rc, 2)
        self.assertIn("no spec named nosuch; specs: greeting", r.err)

    @unittest.expectedFailure
    def test_a_malformed_spec_exits_2_without_a_traceback(self):
        """KNOWN: it raises JSONDecodeError and exits 1, which the hook
        contract reads as the second label, a verdict, instead of unscored."""
        (self.tmp / "specs" / "broken.json").write_text('{"question": "oops", "labels": [')
        r = self.cli("classify", "broken", "message=x")
        self.assertEqual(r.rc, 2)
        self.assertNotIn("Traceback", r.err)

    def test_a_threshold_on_no_drops_a_confident_no(self):
        self.spec("drop", threshold={"no": 0.9})
        r, d = self.classify("drop", "message=The invoice is overdue by ten days.")
        self.assertEqual((r.rc, d["label"], d["decision"]), (1, "no", "drop"))

    def test_a_winner_under_min_p_is_unsure_and_exits_3(self):
        # Above any p a model can reach, so the decision cannot hang on the model.
        self.spec("picky", min_p=1.0001)
        r, d = self.classify("picky", "message=Hello there!")
        self.assertEqual((r.rc, d["decision"]), (3, "unsure"))

    def test_histogram_counts_the_rows_a_hook_logged(self):
        self.spec("greeting")
        self.classify("greeting", "message=Hello there!")
        self.classify("greeting", "message=The invoice is overdue.")
        self.classify("greeting", "message=Good morning.", hook="other")
        listed = self.cli("histogram").out
        self.assertRegex(listed, r"(?m)^greeting\s+e2e\s+2$")
        self.assertRegex(listed, r"(?m)^greeting\s+other\s+1$")
        self.assertEqual(self.cli("histogram", "greeting", "--hook", "e2e").out.splitlines()[0],
                         "greeting p(yes) n=2 hook=e2e")

    def test_classify_after_unload_answers_without_a_resume(self):
        self.spec("greeting")
        # A dead host, so unload frees nothing on the live server.
        self.cli("unload", env={"CLASSIF_HOSTS": "localhost:1"})
        r, d = self.classify("greeting", "message=Hello there!")
        self.assertEqual((r.rc, d["label"]), (0, "yes"))


@judgement
class SpecJudgement(SpecBase):
    def test_smoke_fails_a_label_whose_first_piece_is_one_character(self):
        """The default model's tokenizer writes "sincere" as "s" + "incere",
        and a one-character piece is credited to no label, so those rows score
        no label mass at all. This is the case smoke's mass check catches."""
        rows = [{"message": "You single-handedly fought your way into this hopeless mess.", "expect": "backhanded"},
                {"message": "You are confused; but this is your normal state.", "expect": "backhanded"},
                {"message": "Good news will come to you by mail.", "expect": "sincere"},
                {"message": "Your present plans will be successful.", "expect": "sincere"}]
        self.spec("tone", question="Is this fortune cookie message backhanded or sincere?",
                  labels=["backhanded", "sincere"], smoke=rows)
        r = self.cli("smoke", "tone")
        self.assertEqual(r.rc, 1)
        self.assertRegex(r.out, r"(?m)^2/4 expected labels, 2 under mass")
        self.assertEqual(len(re.findall(r"(?m)^BAD want sincere got None .* mass 0\.00", r.out)), 2, r.out)


if __name__ == "__main__":
    unittest.main()
