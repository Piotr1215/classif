"""-c FILE: a policy or reference read before the text on every reader and
judge call, direct or piecewise, never on a link call. A stub Ollama records
what each call carried; nothing here measures a model."""
import importlib.machinery
import importlib.util
import json
import os
import subprocess
import tempfile
import threading
import unittest
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

PUBLIC = Path(__file__).resolve().parent.parent / "classif"
OVERFLOW = (400, {"error": "request (40000 tokens) exceeds the available context size (32768 tokens)"})
POLICY = "Comments address ideas, not people. Advertising is removed."
SOURCE = "Record 1: service maintenance was completed.\nComment: buy followers at my site!\nThat offer was repeated twice.\n"


def default_model():
    """What classif runs when nothing names a model. The stub lists it so the
    reader cache's digest lookup finds it."""
    loader = importlib.machinery.SourceFileLoader("class_cli", str(PUBLIC.parent / "judge.py"))
    mod = importlib.util.module_from_spec(importlib.util.spec_from_loader("class_cli", loader))
    loader.exec_module(mod)
    return mod.DEFAULT_MODEL


DEFAULT_MODEL = default_model()


def scored(label):
    return 200, {"done": True, "logprobs": [{"top_logprobs": [{"token": label, "logprob": -0.01}]}]}


class Stub:
    def __init__(self, chat):
        self.calls, lock, outer = [], threading.Lock(), self

        class Handler(BaseHTTPRequestHandler):
            def reply(self, status, body):
                data = json.dumps(body).encode()
                self.send_response(status)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(data)))
                self.end_headers()
                self.wfile.write(data)

            def do_GET(self):
                self.reply(200, {"models": [{"name": DEFAULT_MODEL, "digest": "stub"}]})

            def do_POST(self):
                body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
                user = body["messages"][1]["content"]
                with lock:
                    outer.calls.append(user)
                    n = len(outer.calls)
                self.reply(*chat(n, user))

            def log_message(self, *_):
                pass

        self.server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        self.server.daemon_threads = True
        self.host = f"127.0.0.1:{self.server.server_port}"
        threading.Thread(target=self.server.serve_forever, kwargs={"poll_interval": 0.01}, daemon=True).start()

    def close(self):
        self.server.shutdown()
        self.server.server_close()


def reader(n, user):
    """Overflow the whole read, then flag the comment lines, judge yes."""
    if n == 1:
        return OVERFLOW
    if "Last line:" in user:
        # the candidate numbered as the comment line
        return scored(next(l.split(":")[0] for l in user.splitlines() if "buy followers" in l))
    if "Taken alone" in user or "Does this passage" in user:
        return scored("fits" if "followers" in user or "offer" in user else "unrelated")
    return scored("yes")


class ContextTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.policy = Path(self.tmp.name) / "policy.txt"
        self.policy.write_text(POLICY + "\n")
        self.source = Path(self.tmp.name) / "comments.txt"
        self.source.write_text(SOURCE)

    def call(self, stub, *args, text=None):
        self.addCleanup(stub.close)
        env = {k: v for k, v in os.environ.items() if not k.startswith("CLASSIF")}
        env.update(CLASSIF_HOSTS=stub.host, CLASSIF_CALIBRATION="/nonexistent", XDG_CACHE_HOME=self.tmp.name)
        return subprocess.run([str(PUBLIC), *args], input=text, text=True, capture_output=True, env=env, timeout=20)

    def test_direct_call_reads_the_context_before_the_text(self):
        stub = Stub(lambda n, user: scored("2"))    # -e options are numbered; 2 is stop
        r = self.call(stub, "-j", "-c", str(self.policy), "-e", "pass,stop", "Should this comment pass or stop?",
                      text="buy followers at my site!")
        self.assertEqual(r.returncode, 1, r.stderr)
        data = json.loads(r.stdout)
        self.assertEqual((data["label"], data["mode"], data["context"]), ("stop", "direct", str(self.policy)))
        user = stub.calls[0]
        self.assertLess(user.index("Context:\n" + POLICY), user.index("Text:\nbuy followers"))

    def test_every_reader_and_judge_call_carries_the_context_but_no_link_call_does(self):
        stub = Stub(reader)
        # "latest" needs every line read, so the link step runs.
        r = self.call(stub, "-j", "-c", str(self.policy), "-i", str(self.source),
                      "The latest comment breaks the context policy.")
        self.assertEqual(r.returncode, 0, r.stderr)
        data = json.loads(r.stdout)
        self.assertEqual((data["mode"], data["label"], data["context"]), ("memory", "yes", str(self.policy)))
        links = [u for u in stub.calls[1:] if "Last line:" in u]
        # The root call reads the question alone, so it carries no context either.
        roots = [u for u in stub.calls[1:] if "Which kind is it?" in u]
        others = [u for u in stub.calls[1:] if "Last line:" not in u and u not in roots]
        self.assertTrue(roots and all("Context:" not in u for u in roots))
        self.assertTrue(links and others)
        self.assertTrue(all(u.startswith("Context:\n" + POLICY) for u in others))
        self.assertTrue(all("Context:" not in u for u in links))
        # offsets index the -f file, not the context
        for s in data["read"]["sources"]:
            self.assertIn(SOURCE[s["start"]:s["end"]].strip(), ("Comment: buy followers at my site!",
                                                                "That offer was repeated twice."))
        self.assertEqual(data["read"]["file"], str(self.source))

    def test_a_changed_policy_reads_again_and_a_link_is_kept(self):
        stub = Stub(reader)
        first = self.call(stub, "--cache", "-j", "-c", str(self.policy), "-i", str(self.source),
                          "A comment breaks the context policy.")
        self.assertEqual(first.returncode, 0, first.stderr)
        reads = len(stub.calls)
        self.policy.write_text("Anything goes. Advertising is welcome.\n")
        stub2 = Stub(reader)
        second = self.call(stub2, "--cache", "-j", "-c", str(self.policy), "-i", str(self.source),
                           "A comment breaks the context policy.")
        self.assertEqual(second.returncode, 0, second.stderr)
        again = json.loads(second.stdout)["read"]
        links_first = sum("Last line:" in u for u in stub.calls)
        links_second = sum("Last line:" in u for u in stub2.calls)
        self.assertEqual(links_second, 0)             # the link was saved under the first policy and reused
        self.assertEqual(again["saved"], links_first)  # nothing else was reused
        self.assertEqual(len(stub2.calls), reads - links_first)

    def test_a_context_the_server_refuses_is_unscored_with_its_reason(self):
        stub = Stub(lambda n, user: OVERFLOW)
        r = self.call(stub, "-j", "-c", str(self.policy), "-i", str(self.source), "A comment breaks the policy.")
        self.assertEqual(r.returncode, 2, r.stderr)
        self.assertIn("exceeds the available context size", r.stderr)

    def test_missing_or_empty_context_file_is_a_usage_error(self):
        stub = Stub(lambda n, user: scored("yes"))
        empty = Path(self.tmp.name) / "empty.txt"
        empty.write_text(" \n")
        for path in (str(Path(self.tmp.name) / "absent.txt"), str(empty)):
            r = self.call(stub, "-c", path, "Is this fine?", text="fine")
            self.assertEqual(r.returncode, 2, path)
            self.assertIn("-c", r.stderr)
        self.assertEqual(stub.calls, [])

    def test_several_files_are_one_text_with_a_range_each_and_the_gate_passes_them_whole(self):
        rules = Path(self.tmp.name) / "rules.txt"
        rules.write_text("Rule: advertising is removed")      # no final newline
        stub = Stub(reader)
        r = self.call(stub, "-j", "-i", f"{rules},{self.source}", "A comment breaks the rules.")
        self.assertEqual(r.returncode, 0, r.stderr)
        data = json.loads(r.stdout)
        joined = rules.read_text() + "\n" + SOURCE
        ranges = data["read"]["files"]
        self.assertEqual([f["file"] for f in ranges], [str(rules), str(self.source)])
        self.assertEqual(joined[ranges[1]["start"]:ranges[1]["end"]], SOURCE)
        self.assertEqual(data["read"]["file"], f"{rules},{self.source}")
        for src in data["read"]["sources"]:
            self.assertIn(joined[src["start"]:src["end"]].strip(), ("Comment: buy followers at my site!",
                                                                    "That offer was repeated twice."))
        gate = self.call(Stub(lambda n, user: scored("yes")), "-p", "-i", str(rules), "-i", str(self.source),
                         "Is this fine?")
        self.assertEqual((gate.returncode, gate.stdout), (0, joined))

    def test_rules_given_as_input_are_read_as_passages_and_can_be_dismissed(self):
        """What -c is for: a rule inside -f is one passage among others on a
        long input, and a passage the reader calls unrelated is gone."""
        rules = Path(self.tmp.name) / "rules.txt"
        rules.write_text(POLICY + "\n")
        stub = Stub(reader)
        r = self.call(stub, "-j", "-i", f"{rules},{self.source}", "The latest comment breaks the rules.")
        self.assertEqual(r.returncode, 0, r.stderr)
        seen = [u for u in stub.calls[1:] if POLICY in u]
        self.assertTrue(seen)                                        # read once, as a passage or line
        self.assertFalse(all(POLICY in u for u in stub.calls[1:]))   # not in front of every call

    def test_enum_over_a_long_input_carries_the_context_on_every_passage_and_the_judge(self):
        def enum(n, user):
            if n == 1:
                return OVERFLOW
            return scored("2" if "followers" in user else "0")   # options: 1 pass, 2 stop, 0 none
        stub = Stub(enum)
        r = self.call(stub, "-j", "-c", str(self.policy), "-i", str(self.source), "-e", "pass,stop",
                      "Under the context policy, should the comment pass or stop?")
        self.assertEqual(r.returncode, 1, r.stderr)
        data = json.loads(r.stdout)
        self.assertEqual((data["mode"], data["label"], data["verdict"]), ("memory", "stop", "answered"))
        self.assertGreaterEqual(len(stub.calls), 3)      # overflow, passages, judge
        self.assertTrue(all(u.startswith("Context:\n" + POLICY) for u in stub.calls[1:]))


if __name__ == "__main__":
    unittest.main()
