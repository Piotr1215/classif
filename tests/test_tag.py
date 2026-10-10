"""classif tag asks several questions about one text in one call: the grammar
writes each question's name as a JSON key, the model the digit after it, and
each digit's top_logprobs score that question. A positional starts a
question; the -e and -s after it are its own. These tests run the real
command against a fake Ollama; no model, no network."""
import json
import math
import tempfile
import unittest
from pathlib import Path

import subprocess

from test_judge import FakeOllama, run

MAIL = "Prod login is down since the 3pm deploy. Can you roll back now?"
URGENCY = ["urgency", "-e", "today,this week,no deadline"]
KIND = ["kind", "-e", "asks me,fyi,newsletter"]


def keyed(*steps):
    """A tag reply. A string step is a token the grammar forced; a (token,
    {token: p}) step is an answer and the distribution it was picked from."""
    lps = []
    for s in steps:
        tok, top = (s, {s: 1.0}) if isinstance(s, str) else s
        lps.append({"token": tok, "logprob": math.log(top[tok]),
                    "top_logprobs": [{"token": t, "logprob": math.log(p)} for t, p in top.items()]})
    return 200, {"done": True, "message": {"content": "".join(l["token"] for l in lps)}, "logprobs": lps}


URGENT_ASK = keyed('{"', "urg", "ency", '":', ("1", {"1": 0.9, "2": 0.1}),
                   ',"', "kind", '":', ("1", {"1": 0.8, "3": 0.2}), "}")


class TagTests(unittest.TestCase):
    def serve(self, reply):
        fake = FakeOllama(reply)
        self.addCleanup(fake.close)
        return fake

    def test_each_question_prints_its_answer_and_p(self):
        fake = self.serve(URGENT_ASK)
        r = run(fake.host, "tag", *URGENCY, *KIND, stdin=MAIL)
        self.assertEqual((r.returncode, r.stdout), (0, "urgency  today    0.90\nkind     asks me  0.80\n"))

    def test_each_e_adds_options_to_the_question_before_it_as_in_classif(self):
        fake = self.serve(URGENT_ASK)
        r = run(fake.host, "tag", "urgency", "-e", "today", "-e", "this week", "-e", "no deadline",
                "kind", "-e", "asks me=wants something from me", "-e", "fyi,newsletter", stdin=MAIL)
        self.assertEqual((r.returncode, r.stdout), (0, "urgency  today    0.90\nkind     asks me  0.80\n"))
        user = fake.requests[0]["messages"][-1]["content"]
        self.assertIn("urgency: Options: 1=today, 2=this week, 3=no deadline, 0=none of these.", user)
        self.assertIn("kind: Options: 1=asks me (wants something from me), 2=fyi, 3=newsletter, 0=none of these.",
                      user)

    def test_a_question_with_no_e_is_yes_no_or_unknown(self):
        fake = self.serve(keyed('{"', "is it spam?", '":', ("2", {"2": 0.9, "1": 0.1}), "}"))
        r = run(fake.host, "tag", "is it spam?", stdin=MAIL)
        self.assertEqual((r.returncode, r.stdout), (0, "is it spam?  no  0.90\n"))
        req = fake.requests[0]
        self.assertIn("is it spam?: Options: 1=yes, 2=no, 0=unknown.", req["messages"][-1]["content"])
        self.assertEqual(req["format"]["properties"]["is it spam?"]["enum"], [1, 2, 0])

    def test_s_grades_a_question_on_its_e_levels_and_prints_level_and_grade(self):
        # Score 2 * 0.6 + 1 * 0.4 = 1.6 over a top level of 2: grade 0.80, nearest today.
        fake = self.serve(keyed('{"', "how urgent?", '":', ("2", {"2": 0.6, "1": 0.4}),
                                ',"', "spam?", '":', ("2", {"2": 0.9, "1": 0.1}), "}"))
        r = run(fake.host, "tag", "how urgent?", "-e", "later,this week,today", "-s", "spam?", stdin=MAIL)
        self.assertEqual((r.returncode, r.stdout), (0, "how urgent?  today  0.80\nspam?        no     0.90\n"))
        req = fake.requests[0]
        self.assertIn("how urgent?: Options: 0=later, 1=this week, 2=today.", req["messages"][-1]["content"])
        self.assertEqual(req["format"]["properties"]["how urgent?"]["enum"], [0, 1, 2])

    def test_s_before_its_e_and_s_alone_grade_as_in_classif(self):
        fake = self.serve(keyed('{"', "how urgent?", '":', ("0", {"0": 1.0}),
                                ',"', "how polite?", '":', ("4", {"4": 1.0}), "}"))
        r = run(fake.host, "tag", "how urgent?", "-s", "-e", "later,today", "how polite?", "-s", stdin=MAIL)
        self.assertEqual(r.stdout, "how urgent?  later      0.00\nhow polite?  extremely  1.00\n")
        self.assertIn("how polite?: Options: 0=not at all, 1=slightly, 2=moderately, 3=very, 4=extremely.",
                      fake.requests[0]["messages"][-1]["content"])

    def test_json_carries_a_graded_questions_score_fields(self):
        fake = self.serve(keyed('{"', "how urgent?", '":', ("2", {"2": 0.6, "1": 0.4}), "}"))
        d = json.loads(run(fake.host, "tag", "-j", "how urgent?", "-e", "later,this week,today", "-s",
                           stdin=MAIL).stdout)
        t = d["tags"]["how urgent?"]
        self.assertEqual((t["label"], t["level"], t["legend"]), ("today", "today",
                                                                   {"0": "later", "1": "this week", "2": "today"}))
        self.assertAlmostEqual(t["grade"], 0.8, places=9)
        self.assertAlmostEqual(t["probabilities"]["1"], 0.4, places=9)

    def test_min_p_with_a_graded_question_is_a_usage_error(self):
        fake = self.serve(URGENT_ASK)
        r = run(fake.host, "tag", "-t", "0.8", "how urgent?", "-s", stdin=MAIL)
        self.assertEqual(r.returncode, 2)
        self.assertIn("-t", r.stderr)
        self.assertEqual(fake.requests, [])

    def test_one_request_reads_the_text_once_and_forces_each_question_as_a_key(self):
        fake = self.serve(URGENT_ASK)
        run(fake.host, "tag", *URGENCY, *KIND, stdin=MAIL)
        self.assertEqual(len(fake.requests), 1)
        req = fake.requests[0]
        user = req["messages"][-1]["content"]
        self.assertEqual(user.count(MAIL), 1)
        self.assertIn("urgency: Options: 1=today, 2=this week, 3=no deadline, 0=none of these.", user)
        self.assertEqual(req["format"]["required"], ["urgency", "kind"])
        self.assertEqual(req["format"]["properties"]["kind"], {"type": "integer", "enum": [1, 2, 3, 0]})
        self.assertEqual((req["logprobs"], req["options"]["temperature"]), (True, 0))

    def test_answers_follow_their_key_whatever_order_the_keys_come_in(self):
        fake = self.serve(keyed('{"', "kind", '":', ("3", {"3": 0.7, "1": 0.3}),
                                ',"', "urgency", '":', ("2", {"2": 0.6, "1": 0.4}), "}"))
        r = run(fake.host, "tag", *URGENCY, *KIND, stdin=MAIL)
        self.assertEqual(r.stdout, "urgency  this week   0.60\nkind     newsletter  0.70\n")

    def test_a_digit_inside_a_key_is_not_an_answer(self):
        fake = self.serve(keyed('{"', "q", "1", '":', ("2", {"2": 0.9, "1": 0.1}),
                                ',"', "q", "2", '":', ("1", {"1": 0.8, "2": 0.2}), "}"))
        r = run(fake.host, "tag", "q1", "q2", stdin=MAIL)
        self.assertEqual(r.stdout, "q1  no   0.90\nq2  yes  0.80\n")

    def test_zero_is_none_of_the_options(self):
        fake = self.serve(keyed('{"', "kind", '":', ("0", {"0": 0.9, "2": 0.1}), "}"))
        r = run(fake.host, "tag", *KIND, stdin=MAIL)
        self.assertEqual((r.returncode, r.stdout), (0, "kind  none  0.90\n"))

    def test_a_question_without_enough_label_mass_is_unscored_and_the_run_exits_2(self):
        fake = self.serve(keyed('{"', "urgency", '":', ("1", {"1": 0.9, "2": 0.1}),
                                ',"', "kind", '":', ("1", {"1": 0.3, '"': 0.7}), "}"))
        r = run(fake.host, "tag", *URGENCY, *KIND, stdin=MAIL)
        self.assertEqual((r.returncode, r.stdout), (2, "urgency  today  0.90\nkind     unscored\n"))
        self.assertIn("kind: unscored: label mass 0.30 < 0.5", r.stderr)

    def test_a_question_the_response_never_answers_is_unscored(self):
        fake = self.serve(keyed('{"', "urgency", '":', ("1", {"1": 0.9, "2": 0.1}), ","))
        r = run(fake.host, "tag", *URGENCY, *KIND, stdin=MAIL)
        self.assertEqual(r.returncode, 2)
        self.assertIn("kind: unscored: no answer for it in the response", r.stderr)

    def test_an_answer_under_min_p_is_unsure_and_exits_3(self):
        fake = self.serve(URGENT_ASK)
        r = run(fake.host, "tag", "-t", "0.85", *URGENCY, *KIND, stdin=MAIL)
        self.assertEqual((r.returncode, r.stdout), (3, "urgency  today    0.90\nkind     asks me  0.80 unsure\n"))

    def test_min_p_compares_the_unrounded_p(self):
        # 0.9996 and 0.7996 print as 1.00 and 0.80 but sit under -t 1 and -t 0.8.
        for p, floor in ((0.9996, "1"), (0.7996, "0.8")):
            fake = self.serve(keyed('{"', "kind", '":', ("1", {"1": p, "2": 1 - p}), "}"))
            r = run(fake.host, "tag", "-t", floor, *KIND, stdin=MAIL)
            self.assertEqual(r.returncode, 3, (p, floor))
            self.assertTrue(r.stdout.endswith(" unsure\n"), (p, floor))

    def test_an_answer_at_min_p_or_past_it_passes(self):
        # 0.80045 rounds to 0.800, under -t 0.8004; the unrounded p is past it.
        for top, floor in (({"1": 1.0}, "1"), ({"1": 0.80045, "2": 0.19955}, "0.8004")):
            fake = self.serve(keyed('{"', "kind", '":', ("1", top), "}"))
            r = run(fake.host, "tag", "-t", floor, *KIND, stdin=MAIL)
            self.assertEqual(r.returncode, 0, floor)
            self.assertNotIn("unsure", r.stdout)

    def test_json_keeps_p_unrounded(self):
        fake = self.serve(keyed('{"', "kind", '":', ("1", {"1": 0.9996, "2": 0.0004}), "}"))
        kind = json.loads(run(fake.host, "tag", "-j", "-t", "1", *KIND, stdin=MAIL).stdout)["tags"]["kind"]
        self.assertAlmostEqual(kind["p"]["asks me"], 0.9996, places=9)
        self.assertIs(kind["unsure"], True)

    def test_a_response_cut_by_its_token_budget_says_so(self):
        fake = self.serve(keyed('{"', "urgency", '":', ("1", {"1": 0.9, "2": 0.1}), ',"', "ki"))
        fake.reply[1].update(done_reason="length", eval_count=6)
        r = run(fake.host, "tag", *URGENCY, *KIND, stdin=MAIL)
        self.assertEqual(r.returncode, 2)
        self.assertIn("kind: unscored: the response stopped at its 6-token budget before this answer", r.stderr)
        fake.reply = keyed('{"', "urgency", '":', ("1", {"1": 0.9, "2": 0.1}), ',"', "ki")
        fake.reply[1].update(done_reason="length", eval_count=6)
        d = json.loads(run(fake.host, "tag", "-j", *URGENCY, *KIND, stdin=MAIL).stdout)
        self.assertEqual((d["done_reason"], d["tokens"]), ("length", 6))

    def test_a_name_outside_the_vocabulary_is_answered(self):
        # Ollama 0.34.3 returned each U+20000 of the key as one token string.
        name = "\U00020000" * 12
        fake = self.serve(keyed('{"', *["\U00020000"] * 12, '":', ("1", {"1": 0.9, "2": 0.1}),
                                ',"', "kind", '":', ("2", {"2": 0.8, "1": 0.2}), "}"))
        r = run(fake.host, "tag", name, "-e", "a,b", *KIND, stdin=MAIL)
        self.assertEqual((r.returncode, r.stdout), (0, f"{name}  a    0.90\nkind          fyi  0.80\n"))

    def test_the_token_budget_covers_names_written_a_byte_per_token(self):
        # Twelve U+20000 and kind took Winnow 57 generated tokens; a budget of
        # characters gave it 40 and cut the first key.
        fake = self.serve(URGENT_ASK)
        run(fake.host, "tag", "\U00020000" * 12, "-e", "a,b", *KIND, stdin=MAIL)
        self.assertGreaterEqual(fake.requests[0]["options"]["num_predict"], 57)

    def test_json_keys_each_question_by_its_name(self):
        fake = self.serve(URGENT_ASK)
        d = json.loads(run(fake.host, "tag", "-j", *URGENCY, *KIND, stdin=MAIL).stdout)
        self.assertEqual(list(d["tags"]), ["urgency", "kind"])
        kind = d["tags"]["kind"]
        self.assertEqual((kind["label"], list(kind["p"])), ("asks me", ["asks me", "fyi", "newsletter", "none"]))
        for o, want in (("asks me", 0.8), ("fyi", 0.0), ("newsletter", 0.2), ("none", 0.0)):
            self.assertAlmostEqual(kind["p"][o], want, places=9)
        self.assertAlmostEqual(kind["confidence"], (4 * 0.8 - 1) / 3, places=9)
        self.assertEqual(d["host"], fake.host)

    def test_input_and_context_come_from_files(self):
        with tempfile.TemporaryDirectory() as d:
            mail, policy = Path(d, "mail.txt"), Path(d, "policy.txt")
            mail.write_text(MAIL)
            policy.write_text("Rollbacks need a ticket.")
            fake = self.serve(URGENT_ASK)
            r = run(fake.host, "tag", *URGENCY, *KIND, "-i", str(mail), "-c", str(policy))
        self.assertEqual(r.returncode, 0)
        self.assertTrue(fake.requests[0]["messages"][-1]["content"].startswith(
            f"Context:\nRollbacks need a ticket.\n\nText:\n{MAIL}\n\n"))

    def test_with_no_input_c_is_the_text_tagged(self):
        fake = self.serve(URGENT_ASK)
        r = run(fake.host, "tag", *URGENCY, *KIND, "-c", MAIL, stdin=subprocess.DEVNULL)
        self.assertEqual(r.returncode, 0, r.stderr)
        self.assertTrue(fake.requests[0]["messages"][-1]["content"].startswith(f"Text:\n{MAIL.strip()}\n\n"))

    def test_an_input_past_the_window_is_unscored(self):
        fake = self.serve((400, {"error": "request (40000 tokens) exceeds the available context size (32768 tokens)"}))
        r = run(fake.host, "tag", *URGENCY, stdin=MAIL)
        self.assertEqual((r.returncode, r.stdout), (2, ""))
        self.assertIn("exceeds the available context size", r.stderr)
        self.assertIn("tag reads the whole text in one call", r.stderr)

    def test_malformed_questions_are_usage_errors_and_call_no_model(self):
        fake = self.serve(URGENT_ASK)
        for bad in ([], ["-e", "a,b", "u"], ["-s", "u"], ["u", "-e", ",".join("abcdefghij")], ["u", "-e", "a,A"],
                    ["u", "-e", "a,none"], ["u", "U"], ["u", "-e", "only", "-s"],
                    ["u", "-e", ",".join("abcdefghijk"), "-s"]):
            r = run(fake.host, "tag", *bad, stdin=MAIL)
            self.assertEqual(r.returncode, 2, bad)
            self.assertIn("usage:", r.stderr)
        self.assertEqual(fake.requests, [])

    def test_the_old_name_equals_options_form_names_the_new_one(self):
        fake = self.serve(URGENT_ASK)
        r = run(fake.host, "tag", "urgency=today,this week", stdin=MAIL)
        self.assertEqual(r.returncode, 2)
        self.assertIn("urgency -e today,this week", r.stderr)
        self.assertEqual(fake.requests, [])

    def test_no_text_is_a_usage_error(self):
        fake = self.serve(URGENT_ASK)
        r = run(fake.host, "tag", *URGENCY, stdin="")
        self.assertEqual(r.returncode, 2)
        self.assertIn("no input", r.stderr)
        self.assertEqual(fake.requests, [])


if __name__ == "__main__":
    unittest.main()
