"""Shared by the e2e tests: the real classif command, in a sandbox per test,
talking to Ollama through a local proxy that records or replays its traffic.

CLASSIF_E2E picks the mode:
  replay (default)  answer from cassette.json; no model, no network
  record            forward to the host classif routes to and rewrite cassette.json
  live              forward to that host and keep nothing

A replayed request with no recorded twin fails its test: the prompt, the
options or the text changed, so the recording no longer covers it."""
import atexit
import datetime
import hashlib
import importlib.machinery
import importlib.util
import json
import os
import shutil
import subprocess
import tempfile
import threading
import unittest
import urllib.error
import urllib.request
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[2]
CLI = ROOT / "classif"
CASSETTE = Path(__file__).with_name("cassette.json")
MODE = os.environ.get("CLASSIF_E2E", "replay")


def _judge():
    loader = importlib.machinery.SourceFileLoader("e2e_judge", str(ROOT / "judge.py"))
    mod = importlib.util.module_from_spec(importlib.util.spec_from_loader("e2e_judge", loader))
    loader.exec_module(mod)
    return mod


judge = _judge()


def key(method, path, body):
    """One request's identity. keep_alive is left out: it follows the
    machine's classif-autounload toggle, not anything a test asks."""
    try:
        doc = json.loads(body) if body else None
    except ValueError:
        doc = body.decode(errors="replace")
    if isinstance(doc, dict):
        doc = {k: v for k, v in doc.items() if k != "keep_alive"}
    canon = json.dumps([method, path, doc], sort_keys=True)
    return hashlib.sha1(canon.encode()).hexdigest()[:16], doc


VOLATILE = {"created_at", "expires_at", "total_duration", "load_duration", "prompt_eval_duration", "eval_duration"}


def steady(doc):
    """A response without the clock. Timestamps and durations change on every
    run and classif reads none of them, so a re-record diffs only on substance."""
    if isinstance(doc, dict):
        return {k: steady(v) for k, v in doc.items() if k not in VOLATILE}
    if isinstance(doc, list):
        return [steady(v) for v in doc]
    return doc


class Proxy:
    """Ollama as the CLI sees it. Forwards to upstream and keeps what it hears
    when upstream is set; otherwise answers from the recorded interactions."""

    def __init__(self, upstream, interactions):
        self.upstream, self.interactions, self.misses = upstream, interactions, []
        self.lock = threading.Lock()
        proxy = self

        class Handler(BaseHTTPRequestHandler):
            def handle_one(self):
                body = self.rfile.read(int(self.headers.get("Content-Length") or 0))
                status, payload = proxy.answer(self.command, self.path, body)
                try:
                    self.send_response(status)
                    self.send_header("Content-Type", "application/json")
                    self.send_header("Content-Length", str(len(payload)))
                    self.end_headers()
                    self.wfile.write(payload)
                except (BrokenPipeError, ConnectionResetError):
                    pass  # the client's deadline passed first; it has nobody left to read the answer

            do_GET = do_POST = do_DELETE = handle_one

            def log_message(self, *a):
                pass

        self.server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        self.address = f"127.0.0.1:{self.server.server_address[1]}"
        threading.Thread(target=self.server.serve_forever, daemon=True).start()

    def answer(self, method, path, body):
        k, doc = key(method, path, body)
        if self.upstream is None:
            with self.lock:
                hit = self.interactions.get(k)
                if hit is None:
                    self.misses.append(f"{method} {path} {json.dumps(doc)[:200]}")
            if hit is None:
                return 500, b'{"error": "e2e cassette miss; re-record with CLASSIF_E2E=record"}'
            return hit["status"], json.dumps(hit["response"]).encode()
        req = urllib.request.Request(f"http://{self.upstream}{path}", data=body or None, method=method,
                                     headers={"Content-Type": "application/json"})
        try:
            with urllib.request.urlopen(req, timeout=300) as r:
                status, raw = r.status, r.read()
        except urllib.error.HTTPError as e:
            status, raw = e.code, e.read()
        except OSError as e:
            return 502, json.dumps({"error": f"upstream {self.upstream}: {e}"}).encode()
        try:
            response = json.loads(raw)
        except ValueError:
            response = raw.decode(errors="replace")
        with self.lock:
            self.interactions[k] = {"request": {"method": method, "path": path, "body": doc},
                                    "status": status, "response": steady(response)}
        return status, raw


def _upstream():
    """(host, model, why) of the live model classif routes to, or why none can answer."""
    host, model = judge.route()
    if not host:
        return None, None, "no Ollama host answered; set CLASSIF_HOSTS=host:port=model"
    try:
        with urllib.request.urlopen(f"http://{host}/api/tags", timeout=5) as r:
            pulled = {m.get("model", m.get("name")) for m in json.load(r).get("models") or []}
    except (OSError, ValueError) as e:
        return None, None, f"{host}: {e}"
    if model not in pulled and f"{model}:latest" not in pulled:
        return None, None, f"{model} is not pulled on {host}"
    return host, model, ""


def _save(proxy, upstream, model):
    try:
        with urllib.request.urlopen(f"http://{upstream}/api/version", timeout=5) as r:
            version = json.load(r).get("version")
    except (OSError, ValueError):
        version = None
    doc = {"model": model, "ollama": version, "recorded": datetime.date.today().isoformat(),
           "interactions": dict(sorted(proxy.interactions.items()))}
    CASSETTE.write_text(json.dumps(doc, indent=1, sort_keys=True) + "\n")


def _start():
    """(proxy, model, why): the proxy every test talks to and the model it
    answers as, or why the suite cannot run in this mode."""
    if MODE == "replay":
        if not CASSETTE.exists():
            return None, None, f"no {CASSETTE.name}; record one with CLASSIF_E2E=record"
        doc = json.loads(CASSETTE.read_text())
        return Proxy(None, doc["interactions"]), doc["model"], ""
    if MODE not in ("record", "live"):
        return None, None, f"CLASSIF_E2E={MODE}: want replay, record or live"
    upstream, model, why = _upstream()
    if not upstream:
        return None, None, why
    proxy = Proxy(upstream, {})
    if MODE == "record":
        atexit.register(_save, proxy, upstream, model)
    return proxy, model, ""


PROXY, MODEL, WHY = _start()
HOST = PROXY.address if PROXY else None

# Answers measured on the default model. Another model may judge these
# differently without anything being broken, so they skip there.
judgement = unittest.skipUnless(MODEL == judge.DEFAULT_MODEL, f"judgement checks are pinned to {judge.DEFAULT_MODEL}")


class E2E(unittest.TestCase):
    """A spec dir and log of its own per test, so nothing reads or writes the
    live log, samples or the specs a consumer loads."""

    def setUp(self):
        if not PROXY:
            self.skipTest(WHY)
        self.tmp = Path(tempfile.mkdtemp())
        self.addCleanup(shutil.rmtree, self.tmp)
        (self.tmp / "specs").mkdir()
        (self.tmp / "state").mkdir()
        self.env = {**os.environ, "CLASSIF": "1", "CLASSIF_HOSTS": f"{HOST}={MODEL}",
                    "CLASSIF_DIR": str(self.tmp / "specs"), "CLASSIF_LOG": str(self.tmp / "state" / "log.jsonl"),
                    "XDG_CACHE_HOME": str(self.tmp / "cache")}
        self.seen = len(PROXY.misses)

    def tearDown(self):
        missed = PROXY.misses[self.seen:] if PROXY else []
        self.assertEqual(missed, [], "requests the cassette does not hold; re-record with CLASSIF_E2E=record")

    def cli(self, *args, stdin=b"", env=None):
        """Run classif in the sandbox. stdin is bytes so -p can be checked byte
        for byte; out and err come back decoded, raw is stdout as bytes."""
        p = subprocess.run([str(CLI), *args], input=stdin, capture_output=True, cwd=self.tmp,
                           env={**self.env, **(env or {})}, timeout=300)
        return SimpleNamespace(rc=p.returncode, raw=p.stdout, out=p.stdout.decode(), err=p.stderr.decode())

    def label(self, *args, **kw):
        """Exit code and winning label of one plain call."""
        r = self.cli(*args, **kw)
        return r.rc, (r.out.split() or [None])[0]

    def write(self, name, text):
        (self.tmp / name).write_text(text)
        return self.tmp / name
