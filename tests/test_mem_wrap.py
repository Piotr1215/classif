"""Wrap-width invariance of the reader, pinned by a cassette of real model
answers. One authored paragraph about wait -n, wrapped at 60, 80, 100 and 200
columns, is read the way --why reads after a search. The lines carrying the
-n sentence must read fits at every width and be cited exactly: read alone, a
line cut mid-sentence read breaks, and a passage kept whole was cited as one.

Replay needs no model. Record again after a prompt change, with Ollama up:
    python3 tests/test_mem_wrap.py record
"""
import hashlib
import importlib.util
import json
from pathlib import Path
import sys
import textwrap
import unittest


ROOT = Path(__file__).resolve().parents[1]
CASSETTE = Path(__file__).with_name("fixtures") / "wrapped-wait.json"
WIDTHS = (60, 80, 100, 200)
CLAIM = "Can Bash wait for whichever child process terminates next rather than waiting for a specific process?"
HEADER = "wait [-fn] [-p varname] [id ...]"
PARAGRAPH = ("Wait for every child process named by id and return the status it exited with. An id is either a "
             "process ID or a job specification, and naming a job waits for every process in that job's pipeline. "
             "When no id is given, wait waits for all running background jobs and for the most recent process "
             "substitution, and returns zero. If the -n option is given, wait waits for just one job, any of the "
             "listed ids or, with none listed, any job at all, to finish, and returns that job's exit status. If no "
             "id names a child of this shell, the return status is 127. With -p, the id of the job whose status is "
             "returned is stored in the variable varname, which only makes sense together with -n. With -f and job "
             "control enabled, wait waits for each id to terminate instead of returning when its status changes.")
TARGET = ("If the -n option is given, wait waits for just one job, any of the listed ids or, with none listed, any "
          "job at all, to finish, and returns that job's exit status.")
FILLER = [
    ("umask [-p] [-S] [mode]",
     "The user file-creation mask is set to mode. If mode begins with a digit, it is read as an octal number; "
     "otherwise it is a symbolic mode mask like the one chmod accepts. With no mode, the current mask is printed."),
    ("unalias [-a] [name ...]",
     "Remove each name from the list of defined aliases. With -a, every alias definition is removed. The return "
     "value is true unless a name given is not a defined alias."),
    ("unset [-fv] [-n] [name ...]",
     "For each name, remove the corresponding variable or function. With -v, each name refers to a shell variable, "
     "and that variable is removed. Read-only variables may not be unset."),
]


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


def page(width):
    """The entries wrapped like man output at width, and the 1-based numbers
    of the lines holding any word of TARGET."""
    def wrap(body):
        return textwrap.wrap(body, width, initial_indent=" " * 14, subsequent_indent=" " * 14,
                             break_on_hyphens=False)
    out = []
    for name, body in FILLER:
        out += ["       " + name] + wrap(body) + [""]
    first = len(out) + 2
    lines = wrap(PARAGRAPH)
    out += ["       " + HEADER] + lines + ["", "       times", " " * 14 + "Print the accumulated user and system times."]
    t0, t1, at, carry = PARAGRAPH.index(TARGET), PARAGRAPH.index(TARGET) + len(TARGET), 0, []
    for k, line in enumerate(lines):
        a = PARAGRAPH.index(line.strip(), at)
        at = a + len(line.strip())
        if min(at, t1) > max(a, t0):
            carry.append(first + k)
    return "\n".join(out) + "\n", carry


def key(question, text, labels, options):
    return hashlib.sha1(json.dumps([question, text, list(labels), options]).encode()).hexdigest()


def read(mem, width, ask):
    """The limited read --why runs after a search: every line but the last
    allowed, in passages of CHUNK."""
    doc = mem.Doc(page(width)[0])
    return mem.run(doc, CLAIM, ask, only=list(range(len(doc.spans) - 1)), block=mem.CHUNK)


class WrapWidthTests(unittest.TestCase):
    mem = load("classif_memory_wrap", ROOT / "mem.py")

    @classmethod
    def setUpClass(cls):
        cls.answers = json.loads(CASSETTE.read_text())["answers"]

    def replay(self, question, text, labels, options=None):
        k = key(question, text, labels, options)
        if k not in self.answers:
            self.fail(f"call not in {CASSETTE.name}; record again: python3 tests/test_mem_wrap.py record")
        return self.answers[k]

    def test_the_lines_of_the_wait_n_sentence_are_cited_fits_and_alone_at_every_width(self):
        for width in WIDTHS:
            with self.subTest(width=width):
                carry = page(width)[1]
                ev = read(self.mem, width, self.replay)["read"]["evidence"]
                lines = {n: e["label"] for e in ev for n in range(e["line"], e["end"] + 1)}
                self.assertEqual({n: lines.get(n) for n in carry}, {n: "fits" for n in carry})
                # the run holding the sentence ends where the sentence does
                run = next(e for e in ev if e["line"] <= carry[0] <= e["end"])
                self.assertEqual((run["line"], run["end"]), (carry[0], carry[-1]))
                self.assertIn("p", run)

    def test_the_carrying_lines_are_counted_from_the_words_they_hold(self):
        # the first word of the sentence ends a line at 80 columns, so that line carries it
        self.assertEqual(page(80)[1], [22, 23, 24, 25])
        self.assertEqual(len(page(200)[1]), 2)


def record():
    """Read at every width with the default model and save each answer."""
    judge = load("judge_wrap", ROOT / "judge.py")
    mem = load("classif_memory_wrap", ROOT / "mem.py")
    answers = {}

    def ask(question, text, labels, options=None):
        r = judge.judge(question, text, labels, options, timeout=120)
        if r.get("label") is None:
            raise SystemExit(f"unscored: {r.get('unscored')}")
        answers[key(question, text, labels, options)] = {"label": r["label"], "p": r["p"]}
        return r

    for width in WIDTHS:
        ev = read(mem, width, ask)["read"]["evidence"]
        print(width, page(width)[1], [(e["line"], e["end"], e["label"], e.get("p")) for e in ev], flush=True)
    CASSETTE.write_text(json.dumps({"model": judge.DEFAULT_MODEL, "claim": CLAIM, "answers": answers},
                                   indent=1, sort_keys=True) + "\n")
    print(f"{len(answers)} answers -> {CASSETTE}")


if __name__ == "__main__":
    if sys.argv[1:] == ["record"]:
        record()
    else:
        unittest.main()
