"""Independent checks for the persistent reader-score cache."""
import math
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from test_mem import load


class SavedTests(unittest.TestCase):
    def setUp(self):
        self.mem = load()
        self.klass = self.mem.load_class()
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.path = str(Path(self.tmp.name) / "scores.sqlite")
        self.saved = self.mem.Saved(self.path, self.klass)
        self.addCleanup(self.saved.db.close)
        self.calls = []

    def judge(self, question, text, labels, options=None):
        self.calls.append((question, text, list(labels), options))
        logp = {k: math.log(0.9 if k == labels[0] else 0.1 / (len(labels) - 1)) for k in labels}
        p = self.klass.calibrate(logp, 1.0)
        return {"label": labels[0], "p": p, "logp": logp, "mass": 1.0}

    def wrap(self, model="fixture-model"):
        ident = dict(model=model, digest="digest-v1", num_ctx=2048)
        return self.saved.wrap(self.judge, model, ident)

    def test_identical_read_uses_disk_score(self):
        ask = self.wrap()
        first = ask("Is this paid?", "Invoice INV-42 paid.", ["yes", "no"])
        second = ask("Is this paid?", "Invoice INV-42 paid.", ["yes", "no"])
        self.assertEqual(len(self.calls), 1)
        self.assertEqual(self.saved.hits, 1)
        self.assertEqual(first["logp"], second["logp"])

    def test_changed_question_text_labels_or_model_causes_a_new_read(self):
        self.wrap()("Paid?", "INV-42 paid.", ["yes", "no"])
        for model, question, text, labels in (
            ("fixture-model", "Reversed?", "INV-42 paid.", ["yes", "no"]),
            ("fixture-model", "Paid?", "INV-42 unpaid.", ["yes", "no"]),
            ("fixture-model", "Paid?", "INV-42 paid.", ["yes", "no", "unknown"]),
            ("other-model", "Paid?", "INV-42 paid.", ["yes", "no"]),
        ):
            self.wrap(model)(question, text, labels)
        self.assertEqual(len(self.calls), 5)
        self.assertEqual(self.saved.hits, 0)

    def test_enum_description_change_invalidates_its_read(self):
        ask = self.wrap()
        ask("Choose.", "Deployment message.", ["1", "2", "0"], ["prod", "staging", "none"])
        ask("Choose.", "Deployment message.", ["1", "2", "0"], ["prod (live traffic)", "staging", "none"])
        self.assertEqual(len(self.calls), 2)

    def test_lookup_recalibrates_saved_log_masses(self):
        ask = self.wrap()
        with patch.object(self.klass, "temperature", return_value=1.0):
            first = ask("Paid?", "INV-42 paid.", ["yes", "no"])
        with patch.object(self.klass, "temperature", return_value=4.0):
            second = ask("Paid?", "INV-42 paid.", ["yes", "no"])
        self.assertEqual(len(self.calls), 1)
        self.assertLess(second["p"]["yes"], first["p"]["yes"])
        self.assertEqual(first["logp"], second["logp"])

    def test_failed_read_is_retried_instead_of_saved(self):
        def fail(*args):
            self.calls.append(args)
            return {"label": None, "unscored": "temporary error"}

        ask = self.saved.wrap(fail, "fixture-model", {"model": "fixture-model", "digest": "digest-v1", "num_ctx": 2048})
        ask("Paid?", "INV-42 paid.", ["yes", "no"])
        ask("Paid?", "INV-42 paid.", ["yes", "no"])
        self.assertEqual(len(self.calls), 2)
        self.assertEqual(self.saved.hits, 0)

    def test_reopening_cache_reuses_scores(self):
        self.wrap()("Paid?", "INV-42 paid.", ["yes", "no"])
        reopened = self.mem.Saved(self.path, self.klass)
        self.addCleanup(reopened.db.close)
        r = reopened.wrap(self.judge, "fixture-model", {"model": "fixture-model", "digest": "digest-v1", "num_ctx": 2048})(
            "Paid?", "INV-42 paid.", ["yes", "no"])
        self.assertEqual(len(self.calls), 1)
        self.assertEqual(reopened.hits, 1)
        self.assertEqual(r["label"], "yes")

    def test_unchanged_record_survives_another_record_edit(self):
        ask = self.wrap()
        for text in ("INV-1 paid.", "INV-2 paid.", "INV-1 paid.", "INV-2 reversed."):
            ask("Paid?", text, ["yes", "no"])
        self.assertEqual(len(self.calls), 3)
        self.assertEqual(self.saved.hits, 1)

    def test_model_digest_and_context_changes_invalidate_readings(self):
        for digest, num_ctx in (("digest-v1", 2048), ("digest-v2", 2048), ("digest-v1", 8192)):
            ident = {"model": "fixture-model", "digest": digest, "num_ctx": num_ctx}
            ask = self.saved.wrap(self.judge, "fixture-model", ident)
            ask("Paid?", "INV-42 paid.", ["yes", "no"])
        self.assertEqual(len(self.calls), 3)
        self.assertEqual(self.saved.hits, 0)

    def test_prompt_template_change_invalidates_readings(self):
        ask = self.wrap()
        ask("Paid?", "INV-42 paid.", ["yes", "no"])
        original = self.klass.prompt

        def changed(*args):
            system, user = original(*args)
            return system + " New template version.", user

        with patch.object(self.klass, "prompt", side_effect=changed):
            ask("Paid?", "INV-42 paid.", ["yes", "no"])
        self.assertEqual(len(self.calls), 2)
        self.assertEqual(self.saved.hits, 0)


if __name__ == "__main__":
    unittest.main()
