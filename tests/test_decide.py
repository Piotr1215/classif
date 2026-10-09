"""examples/decide.py ranks candidates on yes/no criteria and a verdict that
classif answers, marks the lines the top answers rest on and records the
decision. These tests run the script against a stub classif on PATH whose
p(yes) starts at 0.5 and moves by fixed amounts when a word appears in the text
or the context, so every score and mark below is computed by hand."""
import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

SCRIPT = Path(__file__).resolve().parents[1] / "examples" / "decide.py"

# Each rule: when the question holds "q" and the text (or context) holds
# "has", p(yes) moves by "d". A text holding UNSCORED exits 2.
STUB = r'''#!/usr/bin/env python3
import json, os, sys
args, ctx, opts, q, i = sys.argv[1:], "", [], None, 0
while i < len(args):
    a = args[i]
    if a == "-c":
        ctx = open(args[i + 1]).read(); i += 2
    elif a == "-e":
        opts.append(args[i + 1]); i += 2
    elif a == "-j":
        i += 1
    else:
        q = a; i += 1
text = sys.stdin.read()
with open(os.environ["STUB_LOG"], "a") as f:
    f.write(json.dumps({"q": q, "ctx": ctx, "text": text, "opts": opts}) + "\n")
if "UNSCORED" in text:
    sys.exit(2)
y = 0.5
for r in json.loads(os.environ.get("STUB_RULES", "[]")):
    if r["q"] in q and r["has"] in (ctx if r.get("in") == "ctx" else text):
        y += r["d"]
y = round(min(max(y, 0.01), 0.99), 4)
if opts:
    names = [o.split("=")[0] for o in opts]
    p = {n: 0.0 for n in names}
    p[names[0]] = y
    p[names[-1]] = round((1 - y) * 0.7, 4)
    for n in names[1:-1]:
        p[n] = round((1 - y) * 0.3 / (len(names) - 2), 4)
    p["none"] = 0.0
else:
    p = {"yes": y, "no": round(1 - y, 4), "unknown": 0.0}
label = max(p, key=p.get)
print(json.dumps({"label": label, "p": p}))
sys.exit(0 if label == next(iter(p)) else 1)
'''



def scores(out):
    """The table's rows below the header, each split into cells."""
    lines = out.splitlines()
    start = next(i for i, l in enumerate(lines) if l.startswith("score"))
    rows = []
    for l in lines[start + 1:]:
        if not l.strip():
            break
        rows.append(l.split())
    return rows


class DecideTest(unittest.TestCase):
    def setUp(self):
        self.tmp = Path(self.enterContext(tempfile.TemporaryDirectory()))
        bin_dir = self.tmp / "bin"
        bin_dir.mkdir()
        stub = bin_dir / "classif"
        stub.write_text(STUB)
        stub.chmod(0o755)
        self.env = dict(os.environ, PATH=f"{bin_dir}:{os.environ['PATH']}", STUB_LOG=str(self.tmp / "log"),
                        XDG_STATE_HOME=str(self.tmp / "state"), DECIDE_DIR=str(self.tmp / "decide"))

    def decide(self, *args, stdin="", rules=()):
        return subprocess.run([sys.executable, str(SCRIPT), *args], input=stdin,
                              env=dict(self.env, STUB_RULES=json.dumps(list(rules))),
                              capture_output=True, text=True, check=False)

    def record(self, name="adhoc"):
        (path,) = (Path(self.env["XDG_STATE_HOME"]) / "decide" / name).glob("*.json")
        return json.loads(path.read_text())

    def decision_file(self, name, body):
        Path(self.env["DECIDE_DIR"]).mkdir(exist_ok=True)
        (Path(self.env["DECIDE_DIR"]) / f"{name}.json").write_text(json.dumps(body))

    def test_ranks_by_the_product_of_the_p_each_criterion_wants(self):
        r = self.decide("Is it fun?", "-n", "Is it far?", stdin="fun near\nfun far\ndull near\n",
                        rules=[{"q": "fun", "has": "fun", "d": 0.4}, {"q": "far", "has": "far", "d": 0.3}])
        self.assertEqual(r.returncode, 0, r.stderr)
        self.assertIn("  q1  want yes  Is it fun?", r.stdout)
        self.assertIn("  q2  want no   Is it far?", r.stdout)
        # fun near 0.9 * 0.5, dull near 0.5 * 0.5, fun far 0.9 * 0.2
        self.assertEqual(scores(r.stdout), [["0.450", "0.900", "0.500", "fun", "near"],
                                            ["0.250", "0.500", "0.500", "dull", "near"],
                                            ["0.180", "0.900", "0.200", "fun", "far"]])

    def test_files_are_candidates_and_marks_name_the_line_that_moved_p(self):
        (self.tmp / "a.txt").write_text("fun thing\nboring line\n")
        (self.tmp / "b.txt").write_text("dull\n")
        r = self.decide("Is it fun?", "-i", str(self.tmp / "a.txt"), str(self.tmp / "b.txt"),
                        rules=[{"q": "fun", "has": "fun", "d": 0.4}])
        self.assertEqual(r.returncode, 0, r.stderr)
        self.assertEqual([row[-1] for row in scores(r.stdout)], ["a.txt", "b.txt"])
        self.assertIn("why a.txt: lines whose removal moves the answer most\n  q1\n    +0.400  fun thing\n", r.stdout)
        self.assertNotIn("boring line", r.stdout)
        self.assertNotIn("why b.txt", r.stdout)

    def test_a_verdict_labels_each_candidate_and_marks_the_context_line_behind_it(self):
        r = self.decide("What should happen to this task?", "-c", "Goal: race in April\nNote: tired",
                        "-e", "prioritize=do it now", "-e", "defer=later", "-e", "drop=serves no goal",
                        stdin="run plan\nshelf\n",
                        rules=[{"q": "happen", "has": "run", "d": 0.3},
                               {"q": "happen", "has": "race", "d": 0.15, "in": "ctx"},
                               {"q": "happen", "has": "shelf", "d": -0.4}])
        self.assertEqual(r.returncode, 0, r.stderr)
        self.assertIn("  verdict  one of    What should happen to this task? (prioritize, defer, drop)", r.stdout)
        # run plan: 0.5 + 0.3 + 0.15 = 0.95; shelf: 0.5 + 0.15 - 0.4 = 0.25, drop at 0.75 * 0.7
        self.assertEqual(scores(r.stdout), [["0.950", "prioritize", "0.950", "run", "plan"],
                                            ["0.250", "drop", "0.525", "shelf"]])
        self.assertIn("why run plan: lines whose removal moves the answer most\n  verdict (prioritize)\n"
                      "    +0.150  context: Goal: race in April\n", r.stdout)
        self.assertNotIn("Note: tired", r.stdout.split("why")[1])

    def test_the_question_comes_before_y_and_n_and_a_verdict_takes_it_instead(self):
        r = self.decide("Is it fun?", "-y", "Is it near?", "-x", "0", stdin="fun\n")
        self.assertIn("  q1  want yes  Is it fun?\n  q2  want yes  Is it near?\n", r.stdout)
        r = self.decide("Keep it?", "-y", "Is it near?", "-e", "keep", "-e", "toss", "-x", "0", stdin="fun\n")
        self.assertIn("  q1       want yes  Is it near?\n  verdict  one of    Keep it? (keep, toss)\n", r.stdout)

    def test_candidates_can_be_arguments_after_the_question_among_options(self):
        r = self.decide("-c", "Goal: fun", "Is it fun?", "fun walk", "-x", "0", "dull chore",
                        stdin="ignored when arguments are given\n", rules=[{"q": "fun", "has": "walk", "d": 0.3}])
        self.assertEqual(r.returncode, 0, r.stderr)
        self.assertEqual(scores(r.stdout), [["0.800", "0.800", "fun", "walk"], ["0.500", "0.500", "dull", "chore"]])
        self.assertEqual(self.record()["criteria"][0]["context"], "Goal: fun")

    def test_k_narrows_the_report_and_the_record_keeps_every_candidate(self):
        r = self.decide("-y", "Is it fun?", "-k", "1", "-x", "0", stdin="fun\ndull\nflat\n",
                        rules=[{"q": "fun", "has": "fun", "d": 0.4}])
        self.assertEqual(r.returncode, 0, r.stderr)
        self.assertEqual([row[-1] for row in scores(r.stdout)], ["fun"])
        self.assertEqual(len(self.record()["ranking"]), 3)

    def test_a_decision_file_runs_its_commands_and_skips_a_criterion_whose_when_fails(self):
        self.decision_file("lunch", {
            "question": "Where to eat?",
            "candidates": "printf '%s\\n' '{\"name\": \"cafe\", \"text\": \"cheap cafe\\nnear\"}' 'diner'",
            "criteria": [
                {"name": "cheap", "want": "yes", "question": "Is it cheap?", "context_cmd": "echo Budget: low"},
                {"name": "never", "want": "yes", "question": "Is it open?", "when": "false"},
            ],
        })
        r = self.decide("-d", "lunch", rules=[{"q": "cheap", "has": "cheap", "d": 0.3}])
        self.assertEqual(r.returncode, 0, r.stderr)
        self.assertTrue(r.stdout.startswith("Where to eat?\n  cheap  want yes  Is it cheap?\n\n"))
        self.assertEqual([row[-1] for row in scores(r.stdout)], ["cafe", "diner"])
        rec = self.record("lunch")
        self.assertEqual((rec["decision"], rec["question"]), ("lunch", "Where to eat?"))
        self.assertEqual(rec["criteria"], [{"kind": "label", "name": "cheap", "want": "yes",
                                            "question": "Is it cheap?", "context": "Budget: low"}])
        self.assertEqual(rec["ranking"][0]["text"], "cheap cafe\nnear")
        self.assertNotIn("never", r.stdout)

    def test_stdin_candidates_win_over_the_decision_files_command(self):
        self.decision_file("d", {"candidates": "echo from-command",
                                 "criteria": [{"name": "ok", "want": "yes", "question": "Is it ok?"}]})
        r = self.decide("-d", "d", stdin='{"name": "piped", "text": "from stdin"}\n')
        self.assertEqual([row[-1] for row in scores(r.stdout)], ["piped"])

    def test_the_record_holds_criteria_context_every_p_and_marks(self):
        r = self.decide("-y", "Is it fun?", "-c", "Rule: fun wins", "-j",
                        stdin='{"name": "a", "text": "fun one\\nplain"}\n',
                        rules=[{"q": "fun", "has": "fun", "d": 0.4}])
        self.assertEqual(r.returncode, 0, r.stderr)
        printed = json.loads(r.stdout)
        self.assertEqual(printed, self.record())
        self.assertEqual(printed["criteria"], [{"kind": "label", "name": "q1", "want": "yes",
                                                "question": "Is it fun?", "context": "Rule: fun wins"}])
        (row,) = printed["ranking"]
        self.assertEqual((row["name"], row["score"], row["p"], row["text"]), ("a", 0.9, {"q1": 0.9}, "fun one\nplain"))
        self.assertEqual(row["marks"], {"q1": [{"from": "text", "line": "fun one", "delta": 0.4}]})
        self.assertIn(f"decide: record in {Path(self.env['XDG_STATE_HOME'])}/decide/adhoc/", r.stderr)

    def test_an_unscored_candidate_is_left_out_and_nothing_scored_exits_1(self):
        r = self.decide("-y", "Is it ok?", stdin="fine\nUNSCORED thing\n")
        self.assertEqual(r.returncode, 0)
        self.assertIn("decide: classif could not score UNSCORED thing", r.stderr)
        self.assertEqual([row[-1] for row in scores(r.stdout)], ["fine"])
        r = self.decide("-y", "Is it ok?", stdin="UNSCORED\n")
        self.assertEqual(r.returncode, 1)
        self.assertIn("nothing could be scored", r.stderr)

    def test_no_question_is_a_usage_error_and_no_candidates_exits_1(self):
        r = self.decide(stdin="a\n")
        self.assertEqual(r.returncode, 2)
        self.assertIn("no question", r.stderr)
        r = self.decide("-y", "Is it ok?", stdin="")
        self.assertEqual(r.returncode, 1)
        self.assertIn("no candidates", r.stderr)

    def test_one_line_text_without_context_is_not_marked(self):
        r = self.decide("-y", "Is it ok?", stdin="only line\n")
        self.assertIn("why only line: not marked, the text has 1 line and no context has 1 to 60", r.stdout)


if __name__ == "__main__":
    unittest.main()
