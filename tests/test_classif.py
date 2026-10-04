"""classif against a fake Ollama: spool parsing, shadow scoring, cursor, host
order, histogram valley, smoke. No model and no network."""
import importlib.machinery
import email.utils
import importlib.util
import io
import json
import os
import shutil
import subprocess
import sys
import tempfile
import time
import unittest
from concurrent.futures import ThreadPoolExecutor
from contextlib import redirect_stderr, redirect_stdout
from pathlib import Path
from unittest import mock

import threading
from http.server import BaseHTTPRequestHandler, HTTPServer

from tests.test_judge import DEAD, FakeOllama, chat

ROOT = Path(__file__).resolve().parents[1]


def load():
    loader = importlib.machinery.SourceFileLoader("classif_cli", str(ROOT / "classif"))
    spec = importlib.util.spec_from_loader("classif_cli", loader)
    mod = importlib.util.module_from_spec(spec)
    loader.exec_module(mod)
    return mod


SPEC = {
    "question": "Would this chunk help answer the query: {query}",
    "text": "{chunk}",
    "fields": {"query": 40, "chunk": 30},
    "collapse": ["query"],
    "labels": ["yes", "no"],
    "model": "llama3.2:3b",
    "threshold": {},
    "smoke": [{"query": "q one", "chunk": "c one", "expect": "yes"},
              {"query": "q two", "chunk": "c two", "expect": "yes"}],
}


class Base(unittest.TestCase):
    def setUp(self):
        self.sf = load()
        self.tmp = Path(tempfile.mkdtemp())
        self.addCleanup(shutil.rmtree, self.tmp)
        (self.tmp / "specs").mkdir()
        (self.tmp / "specs" / "rag-relevance.json").write_text(json.dumps(SPEC))
        self.log = self.tmp / "log.jsonl"
        self.env = mock.patch.dict(os.environ, {
            "CLASSIF_DIR": str(self.tmp / "specs"), "CLASSIF_LOG": str(self.log),
            "CLASSIF_CALIBRATION": "/nonexistent", "CLASSIF": "1"})
        self.env.start()
        self.addCleanup(self.env.stop)

    def serve(self, reply):
        fake = FakeOllama(reply)
        self.addCleanup(fake.close)
        os.environ["CLASSIF_HOSTS"] = fake.host
        return fake

    def rows(self):
        return [json.loads(l) for l in self.log.read_text().splitlines()] if self.log.exists() else []


class SpecDirTests(Base):
    def lookup(self, user_spec=None):
        cfg = self.tmp / "config"
        (cfg / "classif" / "specs").mkdir(parents=True)
        if user_spec:
            (cfg / "classif" / "specs" / "rag-relevance.json").write_text(json.dumps(user_spec))
        with mock.patch.dict(os.environ, {"XDG_CONFIG_HOME": str(cfg)}):
            os.environ.pop("CLASSIF_DIR")
            return self.sf.load_spec("rag-relevance")

    def test_a_spec_comes_from_the_user_config_dir_when_classif_dir_is_unset(self):
        self.assertEqual(self.lookup(dict(SPEC, threshold={"no": 0.5}))["threshold"], {"no": 0.5})

    def test_a_spec_the_user_lacks_is_not_found_beside_the_script(self):
        self.assertFalse((ROOT / "specs").exists())
        with self.assertRaises(OSError):
            self.lookup()

    def test_classif_dir_is_the_only_place_looked_when_set(self):
        self.assertEqual(self.sf.spec_dirs(), [str(self.tmp / "specs")])
        with self.assertRaises(OSError):
            self.sf.load_spec("email-bulk")

    def test_the_user_config_dir_is_the_only_place_looked_when_it_is_not(self):
        cfg = self.tmp / "config"
        with mock.patch.dict(os.environ, {"XDG_CONFIG_HOME": str(cfg)}):
            os.environ.pop("CLASSIF_DIR")
            self.assertEqual(self.sf.spec_dirs(), [str(cfg / "classif" / "specs")])


    def test_an_unknown_spec_name_lists_the_specs_instead_of_a_traceback(self):
        r = subprocess.run([str(ROOT / "classif"), "histogram", "spec"], capture_output=True, text=True,
                           env=dict(os.environ), timeout=10)
        self.assertEqual(r.returncode, 2)
        self.assertNotIn("Traceback", r.stderr)
        self.assertIn("no spec named spec; specs: rag-relevance", r.stderr)

    def test_help_lists_the_commands_a_person_runs_and_names_the_rest(self):
        r = subprocess.run([str(ROOT / "classif"), "-h"], capture_output=True, text=True,
                           env=dict(os.environ), timeout=10)
        listed = [l.split()[0] for l in r.stdout.split("Commands:")[1].splitlines() if l.startswith("  ")]
        self.assertEqual(listed, ["specs", "histogram", "pause", "resume"])
        self.assertIn("For hooks and spec authoring: classify, smoke", r.stdout)
        self.assertTrue(r.stdout.rstrip().endswith("Full documentation <https://github.com/Piotr1215/classif>"))

    def test_specs_lists_each_spec_with_the_fields_classify_wants(self):
        (self.tmp / "specs" / "broken.json").write_text("{not json")
        r = subprocess.run([str(ROOT / "classif"), "specs"], capture_output=True, text=True,
                           env=dict(os.environ), timeout=10)
        self.assertEqual(r.returncode, 0, r.stderr)
        rows = {l.split()[0]: l for l in r.stdout.splitlines() if l.strip()}
        self.assertIn("query= chunk=", rows["rag-relevance"])
        self.assertIn("yes,no", rows["rag-relevance"])
        self.assertIn("unreadable", rows["broken"])
        self.assertIn("classif classify SPEC field=value", r.stdout)

    def test_a_spec_in_both_places_is_listed_once_from_where_it_is_read(self):
        cfg = self.tmp / "config"
        (cfg / "classif" / "specs").mkdir(parents=True)
        (cfg / "classif" / "specs" / "rag-relevance.json").write_text(json.dumps(dict(SPEC, labels=["keep", "drop"])))
        env = {k: v for k, v in os.environ.items() if k != "CLASSIF_DIR"}
        r = subprocess.run([str(ROOT / "classif"), "specs"], capture_output=True, text=True,
                           env=dict(env, XDG_CONFIG_HOME=str(cfg)), timeout=10)
        rows = [l for l in r.stdout.splitlines() if l.startswith("rag-relevance")]
        self.assertEqual(len(rows), 1)
        self.assertIn("keep,drop", rows[0])
        self.assertIn(str(cfg / "classif" / "specs"), rows[0])

class RenderTests(Base):
    def test_collapse_fields_become_one_line_and_every_field_is_cut_to_its_cap(self):
        q, t = self.sf.render(SPEC, {"query": "why  does\n\nthe trim " + "x" * 100, "chunk": "\nline one\nline two " + "y" * 50})
        self.assertEqual(q, "Would this chunk help answer the query: " + ("why does the trim " + "x" * 100)[:40])
        self.assertEqual(t, ("line one\nline two " + "y" * 50)[:30])

    def test_a_template_field_the_spec_does_not_declare_is_unscored_not_raised(self):
        bad = dict(SPEC, text="{chunk} {title}")
        r = self.sf.judge_item(bad, {"query": "q", "chunk": "c"}, DEAD)
        self.assertIsNone(r["label"])
        self.assertIn("title", r["unscored"])


    def test_an_empty_field_is_unscored_not_judged(self):
        r = self.sf.judge_item(SPEC, {"query": "  ", "chunk": "c"}, DEAD)
        self.assertEqual((r["label"], r["unscored"]), (None, "empty field: query"))


class HostTests(Base):
    def test_hosts_already_serving_the_spec_model_come_first(self):
        seen = []
        hosts = [("local:1", self.sf.cls.DEFAULT_MODEL), ("mac:1", SPEC["model"]), ("pop:1", SPEC["model"])]
        with mock.patch.object(self.sf.cls, "hosts", lambda: hosts), \
                mock.patch.object(self.sf.cls, "pick_host", lambda c: seen.append(c) or c[0]):
            self.assertEqual(self.sf.pick(SPEC), "mac:1")
        self.assertEqual(seen, [["mac:1", "pop:1", "local:1"]])

    def test_a_spec_that_names_hosts_is_served_by_them_in_order(self):
        seen = []
        os.environ.pop("CLASSIF_HOSTS", None)
        with mock.patch.object(self.sf.cls, "pick_host", lambda c: seen.append(c) or c[0]):
            self.assertEqual(self.sf.pick(dict(SPEC, hosts=["localhost:11434", "mac:1"])), "localhost:11434")
        self.assertEqual(seen, [["localhost:11434", "mac:1"]])


class HistogramTests(Base):
    def write_log(self, ps, hook="memory_enrich"):
        with open(self.log, "w") as fh:
            for p in ps:
                fh.write(json.dumps({"ts": "2026-09-27T10:00:00Z", "spec": "rag-relevance", "hook": hook,
                                     "decision": "shadow", "p": {"yes": p, "no": 1 - p}}) + "\n")

    def hist(self, **kw):
        return self.sf.histogram("rag-relevance", **kw)

    def test_bimodal_log_reports_the_valley_on_the_confident_no_side(self):
        self.write_log([0.02] * 50 + [0.12] * 5 + [0.97] * 30 + [0.62] * 3)
        self.assertIn("valley at 0.05-0.10 (0 rows)", self.hist())

    def test_flat_log_reports_no_valley(self):
        self.write_log([i / 100 for i in range(100)])
        self.assertIn("no valley", self.hist())

    def test_hook_filter_counts_only_that_hook(self):
        self.write_log([0.02] * 5)
        with open(self.log, "a") as fh:
            fh.write(json.dumps({"ts": "2026-09-27T10:00:00Z", "spec": "rag-relevance", "hook": "repo_vector_nudge",
                                 "decision": "shadow", "p": {"yes": 0.9, "no": 0.1}}) + "\n")
        self.assertIn("n=1", self.hist(hook="repo_vector_nudge"))
        self.assertIn("n=6", self.hist())


    def test_without_a_spec_it_lists_the_spec_and_hook_pairs_with_rows(self):
        self.write_log([0.02] * 5)
        with open(self.log, "a") as fh:
            fh.write(json.dumps({"ts": "2026-09-27T10:00:00Z", "spec": "rag-relevance", "hook": "repo_vector_nudge",
                                 "decision": "shadow", "p": {"yes": 0.9, "no": 0.1}}) + "\n")
            fh.write(json.dumps({"ts": "2026-09-27T10:00:00Z", "spec": "rag-relevance", "hook": "x"}) + "\n")
        lines = self.sf.histograms().splitlines()
        self.assertEqual([l.split() for l in lines[1:3]],
                         [["rag-relevance", "memory_enrich", "5"], ["rag-relevance", "repo_vector_nudge", "1"]])
        r = subprocess.run([str(ROOT / "classif"), "histogram"], capture_output=True, text=True,
                           env=dict(os.environ), timeout=10)
        self.assertEqual((r.returncode, r.stdout.splitlines()[1].split()), (0, ["rag-relevance", "memory_enrich", "5"]))

class SmokeTests(Base):
    def smoke(self):
        out = io.StringIO()
        with redirect_stdout(out), redirect_stderr(io.StringIO()):
            code = self.sf.smoke("rag-relevance")
        return code, out.getvalue()

    def test_passes_when_every_row_answers_its_label_with_full_mass(self):
        self.serve(chat([("Yes", -0.01), ("No", -5.0)]))
        self.assertEqual(self.smoke()[0], 0)

    def test_fails_and_names_the_row_when_the_label_word_is_not_what_the_model_wants(self):
        self.serve(chat([("Yes", -0.4), ("Maybe", -1.2)]))
        code, out = self.smoke()
        self.assertEqual(code, 1)
        self.assertIn("mass 0.67", out)
        self.assertIn("q one", out)

    def test_fails_when_full_mass_answers_land_on_the_wrong_label(self):
        self.serve(chat([("No", -0.01), ("Yes", -5.0)]))
        code, out = self.smoke()
        self.assertEqual(code, 1)
        self.assertIn("0/2 expected labels, 0 under mass", out)

    def test_no_host_is_a_skip_not_a_failure(self):
        os.environ["CLASSIF_HOSTS"] = DEAD
        self.assertEqual(self.smoke()[0], 2)


class ResidentOllama:
    """Answers /api/ps with the loaded models, records /api/generate unloads,
    and answers /api/chat by the text it is asked about."""

    def __init__(self, replies, loaded=({"name": "llama3.2:3b", "model": "llama3.2:3b", "context_length": 2048},),
                 delay=0.0):
        self.replies, self.loaded, self.delay, self.requests, self.unloads = replies, list(loaded), delay, [], []
        fake = self

        class H(BaseHTTPRequestHandler):
            def reply(self, body):
                data = json.dumps(body).encode()
                self.send_response(200)
                self.send_header("Content-Length", str(len(data)))
                self.end_headers()
                self.wfile.write(data)

            def do_GET(self):
                self.reply({"models": fake.loaded})

            def do_POST(self):
                req = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
                if self.path == "/api/generate":
                    fake.unloads.append(req)
                    return self.reply({"done": True})
                fake.requests.append(req)
                time.sleep(fake.delay)
                user = req["messages"][1]["content"]
                top = (fake.replies(user) if callable(fake.replies)
                       else next(t for k, t in fake.replies.items() if k in user))
                self.reply({"done": True, "logprobs": [{"top_logprobs": [
                    {"token": t, "logprob": lp} for t, lp in top]}]})

            def log_message(self, *a):
                pass

        self.srv = HTTPServer(("127.0.0.1", 0), H)
        self.srv.daemon_threads = True
        self.host = f"127.0.0.1:{self.srv.server_port}"
        threading.Thread(target=self.srv.serve_forever, kwargs={"poll_interval": 0.05}, daemon=True).start()

    def close(self):
        self.srv.shutdown()
        self.srv.server_close()


SURE_NO = [("No", -0.01), ("Yes", -5.0)]
LEAN_NO = [("No", -0.36), ("Yes", -1.2)]


class DecideTests(Base):
    def test_a_label_at_its_threshold_drops_and_beats_unsure(self):
        # LEAN_NO is p(no) 0.70: past a 0.6 drop threshold and under min_p at once.
        res = {"label": "no", "p": {"yes": 0.3, "no": 0.7}}
        self.assertEqual(self.sf.decide(dict(SPEC, threshold={"no": 0.6}, min_p=0.8), res), "drop")
        self.assertEqual(self.sf.decide(dict(SPEC, threshold={"no": 0.9}), res), "keep")
        self.assertEqual(self.sf.decide(dict(SPEC, threshold={"no": 0.9}, min_p=0.8), res), "unsure")
        self.assertEqual(self.sf.decide(dict(SPEC, threshold={}), res), "shadow")

    def test_an_unscored_answer_is_unscored(self):
        self.assertEqual(self.sf.decide(SPEC, {"label": None, "unscored": "x"}), "unscored")


class LogTests(Base):
    def test_a_log_past_its_cap_is_trimmed_to_its_last_rows(self):
        self.log.write_text("".join(json.dumps({"old": i}) + "\n" for i in range(50)))
        with mock.patch.object(self.sf, "LOG_MAX", 100), mock.patch.object(self.sf, "LOG_KEEP", 2):
            self.sf.append([{"new": 1}])
        self.assertEqual(self.rows(), [{"old": 49}, {"new": 1}])

    def test_concurrent_appends_lose_no_row_to_a_trim(self):
        with mock.patch.object(self.sf, "LOG_MAX", 2000), mock.patch.object(self.sf, "LOG_KEEP", 10 ** 6):
            with ThreadPoolExecutor(8) as pool:
                list(pool.map(lambda t: [self.sf.append([{"t": t, "i": i}]) for i in range(50)], range(8)))
        self.assertEqual(len(self.log.read_text().splitlines()), 400)


class PauseTests(Base):
    def test_pause_unloads_the_spec_model_and_classify_stays_off_until_resume(self):
        fake = ResidentOllama(lambda user: SURE_NO,
                              loaded=[{"model": "llama3.2:3b", "context_length": 2048},
                                      {"model": "qwen2.5:7b", "context_length": 4096}])
        self.addCleanup(fake.close)
        os.environ["CLASSIF_HOSTS"] = fake.host
        with redirect_stdout(io.StringIO()):
            self.assertEqual(self.sf.pause(), 0)
        self.assertEqual(fake.unloads, [{"model": "llama3.2:3b", "keep_alive": 0}])
        with redirect_stdout(io.StringIO()), redirect_stderr(io.StringIO()):
            self.assertEqual(self.sf.classify("rag-relevance", ["query=q", "chunk=c"]), 2)
        self.assertEqual(fake.requests, [])
        self.assertEqual(self.sf.resume(), 0)
        with redirect_stdout(io.StringIO()):
            self.assertEqual(self.sf.classify("rag-relevance", ["query=q", "chunk=c"]), 1)
        self.assertEqual(len(fake.requests), 1)


class ClassifyTests(Base):
    """One item from the command line: the verdict on stdout, the first
    label as exit 0, and a log row so a consumer's histogram can be read."""

    def run_classify(self, *pairs, hook="classify"):
        out = io.StringIO()
        with redirect_stdout(out), redirect_stderr(io.StringIO()):
            code = self.sf.classify("rag-relevance", list(pairs), hook=hook)
        return code, out.getvalue()

    def test_the_verdict_prints_exits_by_label_and_logs_a_row(self):
        self.serve(chat([("No", -0.05), ("Yes", -3.0)]))
        code, out = self.run_classify("query=why does the trim drop rows", "chunk=a cat photo", hook="triage")
        self.assertEqual((code, json.loads(out)["label"]), (1, "no"))
        self.assertEqual([(r["hook"], r["key"], r["label"]) for r in self.rows()],
                         [("triage", "why does the trim drop rows", "no")])

    def test_a_winner_under_the_spec_min_p_exits_3_and_logs_unsure(self):
        (self.tmp / "specs" / "rag-relevance.json").write_text(json.dumps({**SPEC, "min_p": 0.8}))
        self.serve(chat(LEAN_NO))
        code, out = self.run_classify("query=q", "chunk=c")
        self.assertEqual((code, json.loads(out)["label"], json.loads(out)["decision"]), (3, "no", "unsure"))
        self.assertEqual([r["decision"] for r in self.rows()], ["unsure"])

    def test_a_winner_exactly_at_min_p_is_not_unsure(self):
        (self.tmp / "specs" / "rag-relevance.json").write_text(json.dumps({**SPEC, "min_p": 1.0}))
        self.serve(chat([("No", -0.05)]))
        code, out = self.run_classify("query=q", "chunk=c")
        self.assertEqual((code, json.loads(out)["decision"]), (1, "shadow"))

    def test_a_pause_exits_2_without_asking(self):
        fake = self.serve(chat([("No", -0.05), ("Yes", -3.0)]))
        with redirect_stdout(io.StringIO()):
            self.sf.pause()
        self.assertEqual((self.run_classify("query=q", "chunk=c")[0], fake.requests, self.rows()), (2, [], []))

    def test_no_host_exits_2_and_logs_nothing(self):
        os.environ["CLASSIF_HOSTS"] = DEAD
        self.assertEqual((self.run_classify("query=q", "chunk=c")[0], self.rows()), (2, []))

if __name__ == "__main__":
    unittest.main()
