"""-d bounds everything classif does for one question: the whole read, the
model lookup the reader cache needs, and the piecewise reads after an
overflow. A stub Ollama answers slowly on purpose; nothing here measures a
model."""
import importlib.machinery
import importlib.util
import json
import os
import subprocess
import tempfile
import threading
import time
import unittest
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

PUBLIC = Path(__file__).resolve().parent.parent / "classif"
OVERFLOW = (400, {"error": "request (40000 tokens) exceeds the available context size (32768 tokens)"})
SOURCE = "".join(f"Record {i}: service maintenance was completed.\n" for i in range(60))


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
    """chat(n) answers the n-th /api/chat request as (status, body), after
    sleeping if it wants to; tags_delay holds /api/tags back."""

    def __init__(self, chat, tags_delay=0.0):
        self.chats, lock, outer = 0, threading.Lock(), self

        class Handler(BaseHTTPRequestHandler):
            def reply(self, status, body):
                data = json.dumps(body).encode()
                try:
                    self.send_response(status)
                    self.send_header("Content-Type", "application/json")
                    self.send_header("Content-Length", str(len(data)))
                    self.end_headers()
                    self.wfile.write(data)
                except OSError:
                    pass    # the client gave up at its deadline

            def do_GET(self):
                time.sleep(tags_delay)
                self.reply(200, {"models": [{"name": DEFAULT_MODEL, "digest": "stub"}]})

            def do_POST(self):
                self.rfile.read(int(self.headers["Content-Length"]))
                with lock:
                    outer.chats += 1
                    n = outer.chats
                self.reply(*chat(n))

            def log_message(self, *_):
                pass

        self.server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        self.server.daemon_threads = True
        self.host = f"127.0.0.1:{self.server.server_port}"
        threading.Thread(target=self.server.serve_forever, kwargs={"poll_interval": 0.01}, daemon=True).start()

    def close(self):
        self.server.shutdown()
        self.server.server_close()


class DeadlineTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.source = Path(self.tmp.name) / "source.txt"
        self.source.write_text(SOURCE)

    def call(self, stub, *args, text=None):
        self.addCleanup(stub.close)
        env = {k: v for k, v in os.environ.items() if not k.startswith("CLASSIF")}
        env.update(CLASSIF_HOSTS=stub.host, CLASSIF_CALIBRATION="/nonexistent", XDG_CACHE_HOME=self.tmp.name)
        t0 = time.monotonic()
        r = subprocess.run([str(PUBLIC), *args], input=text, text=True, capture_output=True, env=env, timeout=20)
        return r, time.monotonic() - t0

    def test_slow_model_lookup_spends_the_deadline_and_no_read_starts(self):
        # The lookup runs only under --cache. The whole read is refused as past the window, so the reader runs.
        stub = Stub(lambda n: OVERFLOW if n == 1 else scored("fits"), tags_delay=3.0)
        r, wall = self.call(stub, "--cache", "-j", "-d", "1", "-i", str(self.source), "A record reports a fault.")
        data = json.loads(r.stdout)
        self.assertEqual((r.returncode, data["verdict"], data["label"]), (3, "insufficient", None), r.stderr)
        self.assertEqual(stub.chats, 1)
        self.assertLess(wall, 2.5)

    def test_reads_after_an_overflow_get_only_the_time_the_whole_read_left(self):
        def chat(n):
            time.sleep(0.6 if n == 1 else 0.3)
            return OVERFLOW if n == 1 else scored("unrelated")
        stub = Stub(chat)
        r, wall = self.call(stub, "-j", "-d", "1.2", "-i", str(self.source), "A record reports a fault.")
        data = json.loads(r.stdout)
        self.assertEqual((r.returncode, data["mode"], data["verdict"], data["label"]),
                         (3, "memory", "insufficient", None), r.stderr)
        self.assertIn("deadline", data["read"]["why"])
        # 0.6 s of 1.2 remain: the root call and a read of 0.3 s fit, another may be cut in flight.
        self.assertLessEqual(stub.chats, 4)
        self.assertLess(wall, 2.7)

    def test_whole_read_cut_by_the_deadline_is_insufficient_and_nothing_follows(self):
        def chat(n):
            time.sleep(2.0)
            return scored("yes")
        stub = Stub(chat)
        r, wall = self.call(stub, "-j", "-p", "-d", "0.5", "Is this about maintenance?", text=SOURCE)
        self.assertEqual(r.returncode, 3, r.stderr)
        self.assertEqual(r.stdout, "")      # the gate passes nothing
        data = json.loads(r.stderr.splitlines()[0])
        self.assertEqual((data["mode"], data["verdict"], data["label"]), ("direct", "insufficient", None))
        self.assertEqual(stub.chats, 1)
        self.assertLess(wall, 1.8)

    def test_piped_text_without_a_question_is_a_usage_error(self):
        stub = Stub(lambda n: scored("yes"))
        r, _ = self.call(stub, text="Installation finished.\n")
        self.assertEqual((r.returncode, r.stdout, stub.chats), (2, "", 0))
        self.assertIn("question", r.stderr)

    def test_flag_combinations_without_a_meaning_are_refused_before_any_call(self):
        stub = Stub(lambda n: scored("yes"))
        for args in (("-d", "0", "claim"), ("-d", "-3", "claim")):
            r, _ = self.call(stub, *args, text="Record 1 is paid.")
            self.assertEqual((r.returncode, r.stdout), (2, ""), args)
            self.assertIn("usage:", r.stderr)
        self.assertEqual(stub.chats, 0)


if __name__ == "__main__":
    unittest.main()
