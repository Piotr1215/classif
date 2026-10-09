"""classif rank asks one question of each candidate and ranks them, best
first. These tests run the real command against a fake Ollama whose p(yes),
or p of the first -e option, starts at 0.5 and moves by fixed amounts when a
word appears in the text or the context, so every score and mark below is
computed by hand. No model, no network."""
import json
import math
import re
import subprocess
import tempfile
import unittest
from pathlib import Path

from test_judge import FakeOllama, chat, run


def model(*rules):
    """A fake model: each rule (word, d) moves p by d when the word is in the
    text, (word, d, "ctx") when it is in the context. The rest of p goes to
    no, or to the last -e option. A text holding UNSCORED has no label mass."""
    def answer(req):
        user = req["messages"][-1]["content"]
        body = user.rsplit("\n\n", 1)[0]
        ctx, _, text = body.rpartition("Text:\n")
        if "UNSCORED" in text:
            return chat([("the", 0.0)])
        y = 0.5 + sum(r[1] for r in rules if r[0] in (ctx if r[2:] == ("ctx",) else text))
        y = min(max(y, 0.01), 0.99)
        digits = [d for d in re.findall(r"(\d)=", user) if d != "0"]
        first, other = ("1", max(digits)) if digits else ("yes", "no")
        return chat([(first, math.log(y)), (other, math.log(1 - y))])
    return answer


class RankTests(unittest.TestCase):
    def serve(self, *rules):
        fake = FakeOllama(model(*rules))
        self.addCleanup(fake.close)
        return fake

    def file(self, name, text):
        d = getattr(self, "dir", None) or tempfile.mkdtemp()
        self.dir = d
        path = Path(d) / name
        path.write_text(text)
        return str(path)

    def test_candidates_rank_by_p_yes_best_first_each_with_the_answer_it_got(self):
        fake = self.serve(("alpha", 0.4), ("beta", -0.3))
        r = run(fake.host, "rank", "Is it good?", "gamma", "alpha", "beta")
        self.assertEqual(r.returncode, 0)
        self.assertEqual(r.stdout, "p(yes)  answer     candidate\n"
                                   "0.900   yes 0.900  alpha\n"
                                   "0.500   yes 0.500  gamma\n"
                                   "0.200   no 0.800   beta\n")
        self.assertEqual(len(fake.requests), 3)

    def test_with_e_the_score_is_p_of_the_first_option(self):
        fake = self.serve(("running", 0.45), ("Go", -0.4))
        r = run(fake.host, "rank", "What should happen to this task?", "Rewrite the CLI in Go",
                "Sign up for a running plan", "-e", "prioritize=do it this week", "-e", "defer,drop")
        self.assertEqual(r.stdout, "p(prioritize)  answer            candidate\n"
                                   "0.950          prioritize 0.950  Sign up for a running plan\n"
                                   "0.100          drop 0.900        Rewrite the CLI in Go\n")
        self.assertIn("What should happen to this task? Options: 1=prioritize (do it this week), 2=defer, "
                      "3=drop, 0=none of these.", fake.requests[0]["messages"][-1]["content"])

    def test_the_context_is_read_before_each_candidate(self):
        fake = self.serve(("Goal: run", 0.3, "ctx"))
        goals = self.file("goals.md", "Goal: run a half marathon\n")
        r = run(fake.host, "rank", "Is this the right next step?", "-c", goals, "buy shoes", "sort books")
        self.assertEqual([row.split()[0] for row in r.stdout.splitlines()[1:]], ["0.800", "0.800"])
        for req in fake.requests:
            self.assertTrue(req["messages"][-1]["content"].startswith("Context:\nGoal: run a half marathon\n\nText:\n"))

    def test_each_file_is_a_candidate_named_by_its_file_name(self):
        fake = self.serve(("above market", 0.4))
        a = self.file("a.md", "Pays 90k, above market")
        b = self.file("b.md", "Pays 60k")
        r = run(fake.host, "rank", "Does this offer pay well?", "-i", b, a)
        self.assertEqual([row.split()[-1] for row in r.stdout.splitlines()[1:]], ["a.md", "b.md"])

    def test_stdin_lines_are_candidates_plain_or_json_with_a_name(self):
        fake = self.serve(("tests", 0.3))
        lines = 'reorganize the bookshelf\n\n{"name": "ci", "text": "add tests to CI"}\n'
        r = run(fake.host, "rank", "Does this help ship?", stdin=lines)
        self.assertEqual([row.split(None, 3)[-1] for row in r.stdout.splitlines()[1:]],
                         ["ci", "reorganize the bookshelf"])

    def test_terminal_escapes_are_dropped_from_names_and_texts(self):
        fake = self.serve()
        line = "\x1b[1;33mr\x1b[31mw\x1b[0m notes.md \x1b]8;;file:///notes.md\x1b\\link\x1b]8;;\x1b\\"
        r = run(fake.host, "rank", "Is this a file?", stdin=line + "\n")
        self.assertEqual(r.stdout.splitlines()[1], "0.500   yes 0.500  rw notes.md link")
        self.assertNotIn("\x1b", fake.requests[0]["messages"][-1]["content"])

    def test_with_no_candidates_the_context_is_the_one_judged(self):
        fake = self.serve(("tired", 0.4))
        plans = self.file("plans.md", "I am tired\nAlarm at 6")
        r = run(fake.host, "rank", "Should I go to sleep?", "-c", plans, stdin=subprocess.DEVNULL)
        self.assertEqual(r.stdout.splitlines()[1], "0.900   yes 0.900  plans.md")
        self.assertNotIn("Context:", fake.requests[0]["messages"][-1]["content"])

    def test_an_empty_pipe_is_no_candidates_not_the_context(self):
        fake = self.serve()
        r = run(fake.host, "rank", "Should I?", "-c", "my goals", stdin="")
        self.assertEqual(r.returncode, 2)
        self.assertIn("no candidates", r.stderr)
        self.assertEqual(fake.requests, [])

    def test_k_shows_the_top_n(self):
        fake = self.serve(("a", 0.1))
        r = run(fake.host, "rank", "Good?", "a", "b", "c", "-k", "1")
        self.assertEqual(r.stdout.splitlines()[1:], ["0.600   yes 0.600  a"])

    def test_why_marks_the_lines_of_the_top_pick_and_the_context_that_move_its_score(self):
        fake = self.serve(("key fact", 0.3), ("but late", -0.1), ("Goal", 0.1, "ctx"))
        offer = self.file("offer.md", "key fact\nfiller\nbut late")
        goals = self.file("goals.md", "Goal: ship\nnoise")
        r = run(fake.host, "rank", "Take it?", "-i", offer, "-c", goals, "other", "-w")
        self.assertEqual(r.stdout.split("\n\n", 1)[1], "why offer.md: the lines whose removal moves p(yes) most\n"
                                                     "  +0.300  key fact\n"
                                                     "  -0.100  but late\n"
                                                     "  +0.100  context: Goal: ship\n")
        # Two scores, then one call per line: three of the text, two of the context.
        self.assertEqual(len(fake.requests), 2 + 5)

    def test_why_says_when_no_line_moves_it_and_when_nothing_can_be_marked(self):
        fake = self.serve()
        r = run(fake.host, "rank", "Good?", "one\ntwo", "-w")
        self.assertTrue(r.stdout.endswith("why one two: the lines whose removal moves p(yes) most\n"
                                          "  (no single line moves it)\n"))
        r = run(fake.host, "rank", "Good?", "one line", "-w")
        self.assertIn("why one line: not marked", r.stdout)

    def test_an_unscored_candidate_is_left_out_with_a_note_and_nothing_scored_exits_2(self):
        fake = self.serve()
        r = run(fake.host, "rank", "Good?", "fine", "UNSCORED here")
        self.assertEqual((r.returncode, r.stdout.splitlines()[1:]), (0, ["0.500   yes 0.500  fine"]))
        self.assertIn("UNSCORED here: unscored: label mass", r.stderr)
        r = run(fake.host, "rank", "Good?", "UNSCORED")
        self.assertEqual((r.returncode, r.stdout), (2, ""))

    def test_json_carries_each_candidate_its_p_and_the_marks(self):
        fake = self.serve(("alpha", 0.4))
        r = run(fake.host, "rank", "Good?", "beta", "alpha\nfiller", "UNSCORED", "-j", "-w")
        out = json.loads(r.stdout)
        self.assertEqual(out["question"], "Good?")
        self.assertEqual([(row["name"], row["score"], row["label"]) for row in out["ranking"]],
                         [("alpha\nfiller", 0.9, "yes"), ("beta", 0.5, "yes")])
        self.assertEqual(out["ranking"][0]["p"], {"yes": 0.9, "no": 0.1, "unknown": 0.0})
        self.assertEqual(out["ranking"][0]["marks"], [{"from": "text", "line": "alpha", "delta": 0.4}])
        self.assertEqual([m["name"] for m in out["unscored"]], ["UNSCORED"])

    def test_past_nine_options_each_candidate_is_screened_and_ranks_by_its_first_options_p_yes(self):
        def answer(req):
            user = req["messages"][-1]["content"]
            if "Options:" in user:
                return chat([("1", math.log(0.9)), ("2", math.log(0.1))])
            asked = re.search(r"Is the answer (\w+)\?", user)[1]
            y = {"o1": 0.8 if "running" in user else 0.3, "o2": 0.6}.get(asked, 0.01)
            return chat([("yes", math.log(y)), ("no", math.log(1 - y))])
        fake = FakeOllama(answer)
        self.addCleanup(fake.close)
        opts = ",".join(f"o{i}" for i in range(1, 13))
        r = run(fake.host, "rank", "Which?", "sort books", "go running", "-e", opts)
        self.assertEqual(r.returncode, 0, r.stderr)
        self.assertEqual(r.stdout, "p(o1)  answer    candidate\n"
                                   "0.800  o1 0.900  go running\n"
                                   "0.300  o1 0.900  sort books\n")
        # Per candidate, twelve screens and one pick.
        self.assertEqual(len(fake.requests), 2 * (12 + 1))

    def test_a_question_is_required(self):
        r = run("127.0.0.1:1", "rank", stdin=subprocess.DEVNULL)
        self.assertEqual(r.returncode, 2)
        self.assertIn("QUESTION", r.stderr)


if __name__ == "__main__":
    unittest.main()
