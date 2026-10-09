"""judge.py reads a label distribution from one /api/chat token and turns it
into `<label> <p>` plus an exit code a shell `if` can use. These tests run the
real script against a fake Ollama in a thread; no model, no network."""

import importlib.machinery
import importlib.util
import io
import json
import math
import subprocess
import sys
import tempfile
import os
import pty
import re
import shutil
import threading
import time
import unittest
from unittest import mock
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path

SCRIPT = Path(__file__).resolve().parents[1] / "classif"
MODULE = SCRIPT.with_name("judge.py")
DEAD = "127.0.0.1:1"


def load():
    loader = importlib.machinery.SourceFileLoader("class_cli", str(MODULE))
    spec = importlib.util.spec_from_loader("class_cli", loader)
    mod = importlib.util.module_from_spec(spec)
    loader.exec_module(mod)
    return mod


# What classif runs when nothing names a model, here and on the fallback hosts.
_cls = load()
DEFAULT_MODEL = _cls.DEFAULT_MODEL
# A second model, for hosts that name their own.
OTHER_MODEL = "tiny:1b"


def chat(top, done=True):
    return 200, {"done": done, "logprobs": [{"top_logprobs": [
        {"token": t, "logprob": lp} for t, lp in top]}]}


class FakeOllama:
    """Serves one canned (status, body), or what a function of the request
    returns, and records each request body."""

    def __init__(self, reply):
        self.reply, self.requests = reply, []
        fake = self

        class H(BaseHTTPRequestHandler):
            def do_POST(self):
                fake.requests.append(json.loads(self.rfile.read(int(self.headers["Content-Length"]))))
                status, body = fake.reply(fake.requests[-1]) if callable(fake.reply) else fake.reply
                data = json.dumps(body).encode()
                self.send_response(status)
                self.send_header("Content-Length", str(len(data)))
                self.end_headers()
                self.wfile.write(data)

            def log_message(self, *a):
                pass

        self.srv = HTTPServer(("127.0.0.1", 0), H)
        self.host = f"127.0.0.1:{self.srv.server_port}"
        threading.Thread(target=self.srv.serve_forever, kwargs={"poll_interval": 0.05}, daemon=True).start()

    def close(self):
        self.srv.shutdown()
        self.srv.server_close()


def env(hosts, calibration="/nonexistent"):
    # The repo's calibration.json would temper the default model's p; tests
    # that are not about calibration read none.
    return {"CLASSIF_HOSTS": hosts, "PATH": "/usr/bin:/bin", "CLASSIF_CALIBRATION": calibration}


def run(hosts, *args, stdin="", calibration="/nonexistent"):
    # stdin="" by default: a CLI given no input reads stdin, and an inherited
    # one can stay open and hang the call until its timeout.
    # stdin=subprocess.DEVNULL gives the CLI no input at all, as at a terminal.
    feed = {"stdin": stdin} if stdin is subprocess.DEVNULL else {"input": stdin}
    return subprocess.run([str(SCRIPT), *args], capture_output=True, text=True, env=env(hosts, calibration),
                          timeout=10, **feed)


class ScoreTests(unittest.TestCase):
    cls = load()

    def test_case_and_space_variants_sum_into_one_family(self):
        p, mass = self.cls.score([{"token": "Yes", "logprob": -0.54}, {"token": "No", "logprob": -0.94},
                                  {"token": " yes", "logprob": -4.0}, {"token": "no", "logprob": -4.7}],
                                 ["yes", "no"])
        self.assertAlmostEqual(p["yes"], 0.6006, places=3)
        self.assertAlmostEqual(mass, 1.0008, places=3)

    def test_first_piece_of_a_multi_token_label_is_credited(self):
        self.assertEqual(self.cls.credit("gar", ["important", "garbage", "other"]), "garbage")
        self.assertEqual(self.cls.credit(" Ch", ["bug", "feature", "chore"]), "chore")

    def test_prefix_shared_by_two_labels_is_not_credited(self):
        self.assertIsNone(self.cls.credit("ca", ["cat", "car"]))

    def test_single_char_prefix_is_not_credited_but_a_single_char_label_is(self):
        self.assertIsNone(self.cls.credit("g", ["garbage", "other"]))
        self.assertEqual(self.cls.credit(" B", ["A", "B"]), "B")

    def default_hosts(self, hosts_file=None, env=None):
        with tempfile.TemporaryDirectory() as cfg:
            if hosts_file is not None:
                os.makedirs(os.path.join(cfg, "classif"))
                with open(os.path.join(cfg, "classif", "hosts"), "w") as fh:
                    fh.write(hosts_file)
            with mock.patch.dict(os.environ, {"XDG_CONFIG_HOME": cfg, **(env or {})}, clear=True):
                return self.cls.hosts()

    def test_without_config_only_the_local_server_on_the_default_model(self):
        self.assertEqual(self.default_hosts(), [("localhost:11434", DEFAULT_MODEL)])

    def test_the_hosts_file_lists_hosts_in_order_each_on_its_own_model(self):
        got = self.default_hosts(f"# comment\nlaptop:11434\n\nmac:11434={OTHER_MODEL}  # the Mac\nlaptop:11434={OTHER_MODEL}\n")
        self.assertEqual(got, [("laptop:11434", DEFAULT_MODEL), ("mac:11434", OTHER_MODEL)])

    def test_an_empty_hosts_file_falls_back_to_the_local_server(self):
        self.assertEqual(self.default_hosts("# nothing yet\n"), [("localhost:11434", DEFAULT_MODEL)])

    def test_classif_hosts_wins_over_the_file_with_the_same_entries(self):
        got = self.default_hosts("laptop:11434\n", env={"CLASSIF_HOSTS": f"a:1, b:2={OTHER_MODEL}"})
        self.assertEqual(got, [("a:1", DEFAULT_MODEL), ("b:2", OTHER_MODEL)])

    def route(self, forced):
        def connect(addr, timeout):
            if addr[0] == "laptop":
                raise OSError("refused")
            return mock.Mock()
        cands = [("laptop:1", DEFAULT_MODEL), ("mac:1", OTHER_MODEL)]
        with mock.patch.object(self.cls, "hosts", lambda: cands), \
                mock.patch.object(self.cls.socket, "create_connection", connect):
            return self.cls.route(forced)

    def test_a_fallback_host_runs_its_own_model(self):
        self.assertEqual(self.route(None), ("mac:1", OTHER_MODEL))

    def test_dash_m_overrides_the_fallback_hosts_model(self):
        self.assertEqual(self.route(DEFAULT_MODEL), ("mac:1", DEFAULT_MODEL))

    def test_an_earlier_live_host_beats_a_faster_later_one(self):
        def connect(addr, timeout):
            if addr[0] == "slow":
                time.sleep(0.1)
            return mock.Mock()
        with mock.patch.object(self.cls.socket, "create_connection", connect):
            got = self.cls.pick_host(["slow:1", "fast:1"])
        self.assertEqual(got, "slow:1")

    def test_a_dead_earlier_host_yields_to_a_later_live_one(self):
        def connect(addr, timeout):
            if addr[0] == "dead":
                raise OSError("refused")
            return mock.Mock()
        with mock.patch.object(self.cls.socket, "create_connection", connect):
            t0 = time.monotonic()
            got = self.cls.pick_host(["dead:1", "live:1"])
        self.assertEqual(got, "live:1")
        self.assertLess(time.monotonic() - t0, self.cls.GRACE / 2)

    def test_a_hanging_probe_does_not_delay_a_live_host(self):
        def connect(addr, timeout):
            if addr[0] == "hangs":
                time.sleep(3)
            return mock.Mock()
        with mock.patch.object(self.cls.socket, "create_connection", connect):
            t0 = time.monotonic()
            got = self.cls.pick_host(["hangs:1", "live:1"])
        self.assertEqual(got, "live:1")
        self.assertLess(time.monotonic() - t0, 0.5)

    def test_tokens_outside_every_family_add_no_mass(self):
        p, mass = self.cls.score([{"token": "Maybe", "logprob": -0.1}], ["yes", "no"])
        self.assertEqual(mass, 0.0)


class CliTests(unittest.TestCase):
    def serve(self, reply):
        fake = FakeOllama(reply)
        self.addCleanup(fake.close)
        return fake

    def test_first_label_wins_exits_0_and_prints_label_and_p(self):
        fake = self.serve(chat([("Yes", -0.54), ("No", -0.94), ("yes", -4.0), ("no", -4.7)]))
        r = run(fake.host, "is it k8s?", "pod crashloop")
        self.assertEqual((r.returncode, r.stdout), (0, "yes 0.60\n"))

    def test_other_label_wins_exits_1(self):
        fake = self.serve(chat([("No", -0.05), ("Yes", -3.0)]))
        r = run(fake.host, "is it k8s?", "sourdough")
        self.assertEqual((r.returncode, r.stdout), (1, "no 0.95\n"))

    def test_a_winner_under_min_p_is_unsure_and_exits_3_whichever_label_leads(self):
        fake = self.serve(chat([("Yes", math.log(0.6)), ("No", math.log(0.3)), ("unknown", math.log(0.1))]))
        r = run(fake.host, "-t", "0.7", "q?", "t")
        self.assertEqual((r.returncode, r.stdout), (3, "yes 0.60 unsure\n"))
        fake.reply = chat([("No", math.log(0.6)), ("Yes", math.log(0.4))])
        r = run(fake.host, "--min-p", "0.7", "q?", "t")
        self.assertEqual((r.returncode, r.stdout), (3, "no 0.60 unsure\n"))

    def test_a_winner_at_or_past_min_p_keeps_its_usual_exit_code(self):
        fake = self.serve(chat([("Yes", math.log(0.6)), ("No", math.log(0.3)), ("unknown", math.log(0.1))]))
        r = run(fake.host, "-t", "0.5", "q?", "t")
        self.assertEqual((r.returncode, r.stdout), (0, "yes 0.60\n"))
        fake.reply = chat([("No", -0.05), ("Yes", -3.0)])
        r = run(fake.host, "-t", "0.9", "q?", "t")
        self.assertEqual((r.returncode, r.stdout), (1, "no 0.95\n"))
        fake.reply = chat([("Yes", -0.05)])
        r = run(fake.host, "-t", "1", "q?", "t")
        self.assertEqual((r.returncode, r.stdout), (0, "yes 1.00\n"))

    def test_json_says_unsure_only_when_min_p_is_asked_for(self):
        fake = self.serve(chat([("Yes", math.log(0.6)), ("No", math.log(0.4))]))
        out = json.loads(run(fake.host, "-j", "-t", "0.7", "q?", "t").stdout)
        self.assertEqual((out["label"], out["unsure"]), ("yes", True))
        self.assertIs(json.loads(run(fake.host, "-j", "-t", "0.5", "q?", "t").stdout)["unsure"], False)
        self.assertNotIn("unsure", json.loads(run(fake.host, "-j", "q?", "t").stdout))

    def test_an_unsure_gate_passes_nothing(self):
        fake = self.serve(chat([("Yes", math.log(0.6)), ("No", math.log(0.4))]))
        r = run(fake.host, "-p", "-t", "0.7", "is this an error?", stdin="boom\n")
        self.assertEqual((r.returncode, r.stdout, r.stderr), (3, "", ""))

    def test_min_p_outside_0_to_1_is_refused_before_any_call(self):
        fake = self.serve(chat([("Yes", -0.05)]))
        for bad in ("0", "1.5", "-0.2"):
            r = run(fake.host, f"--min-p={bad}", "q?", "t")
            self.assertEqual(r.returncode, 2, bad)
            self.assertIn("--min-p", r.stderr)
        self.assertEqual(fake.requests, [])

    def test_request_asks_for_one_token_with_logprobs_and_carries_the_prompt(self):
        fake = self.serve(chat([("yes", -0.1)]))
        run(f"{fake.host}=tiny:1b", "is it k8s?", stdin="pod crashloop from stdin")
        req = fake.requests[0]
        self.assertEqual((req["model"], req["options"], req["logprobs"], req["top_logprobs"], req["think"]),
                         ("tiny:1b", {"temperature": 0, "num_predict": 1, "num_ctx": 32768}, True, 20, False))
        self.assertEqual(req["messages"][0]["content"], "Answer with exactly one word: yes, no or unknown.")
        self.assertEqual(req["messages"][1]["content"],
                         "Text:\npod crashloop from stdin\n\nis it k8s? Answer yes, no or unknown.")

    def test_a_fact_the_text_never_states_can_come_back_unknown(self):
        fake = self.serve(chat([("unknown", -0.05), ("No", -3.0), ("Yes", -5.0)]))
        r = run(fake.host, "is his friend older than me?", "His friend Mark plays guitar.")
        self.assertEqual((r.returncode, r.stdout), (1, "unknown 0.94\n"))

    def test_the_routed_model_goes_into_the_request_and_the_json(self):
        fake = self.serve(chat([("Yes", -0.1), ("No", -2.5)]))
        cls, out = load(), io.StringIO()
        with mock.patch.object(cls, "route", lambda forced: (fake.host, "llama3.2:3b")), \
                mock.patch.object(sys, "argv", ["classif", "-j", "q?", "t"]), \
                mock.patch.object(sys, "stdout", out):
            cls.main()
        self.assertEqual((fake.requests[0]["model"], json.loads(out.getvalue())["model"]),
                         ("llama3.2:3b", "llama3.2:3b"))

    def test_three_labels_render_as_a_list_and_json_carries_every_p(self):
        fake = self.serve(chat([("gar", -0.52), ("other", -1.81), ("Gar", -2.57), ("important", -2.9)]))
        r = run(fake.host, "-j", "-l", "important,garbage,other", "triage?", "50% off sunglasses")
        out = json.loads(r.stdout)
        self.assertEqual((r.returncode, out["label"], out["host"]), (1, "garbage", fake.host))
        self.assertAlmostEqual(sum(out["p"].values()), 1.0, places=2)
        self.assertIn("Answer important, garbage or other.", fake.requests[0]["messages"][1]["content"])

    def test_json_confidence_is_pmax_rescaled_so_uniform_is_0_and_certain_is_1(self):
        # Jev's measure: (n*pmax - 1) / (n - 1) over the n labels asked.
        fake = self.serve(chat([("Yes", math.log(0.6)), ("No", math.log(0.3)), ("unknown", math.log(0.1))]))
        out = json.loads(run(fake.host, "-j", "q?", "t").stdout)
        self.assertEqual((out["label"], out["confidence"]), ("yes", 0.4))
        fake.reply = chat([("No", math.log(0.95)), ("Yes", math.log(0.05))])
        out = json.loads(run(fake.host, "-j", "-l", "yes,no", "q?", "t").stdout)
        self.assertEqual((out["label"], out["confidence"]), ("no", 0.9))

    def test_json_logp_is_each_family_log_mass_and_an_unseen_label_gets_the_lowest_row(self):
        # The lowest top_logprob bounds any token outside the list from above.
        fake = self.serve(chat([("Yes", -0.2), ("yes", -2.0), ("Maybe", -9.0)]))
        out = json.loads(run(fake.host, "-j", "-l", "yes,no", "q?", "t").stdout)
        self.assertAlmostEqual(out["logp"]["yes"], math.log(math.exp(-0.2) + math.exp(-2.0)), places=3)
        self.assertEqual(out["logp"]["no"], -9.0)

    def calibration(self, table):
        path = Path(self.enterContext(tempfile.TemporaryDirectory())) / "calibration.json"
        path.write_text(json.dumps(table))
        return str(path)

    def test_a_fitted_temperature_softens_p_and_keeps_the_winner_and_the_raw_p(self):
        fake = self.serve(chat([("No", -0.01), ("Yes", -4.6)]))
        cal = self.calibration({DEFAULT_MODEL: {"T": 2.0}})
        r = run(fake.host, "-j", "-l", "yes,no", "q?", "t", calibration=cal)
        out = json.loads(r.stdout)
        want = 1 / (1 + math.exp((-4.6 + 0.01) / 2.0))
        self.assertEqual((r.returncode, out["label"], out["T"]), (1, "no", 2.0))
        self.assertAlmostEqual(out["p"]["no"], want, places=3)
        self.assertEqual(out["p_raw"]["no"], 0.99)
        self.assertEqual(run(fake.host, "-l", "yes,no", "q?", "t", calibration=cal).stdout, f"no {want:.2f}\n")

    def test_an_e_question_uses_the_temperature_fitted_on_e_questions(self):
        fake = self.serve(chat([("1", -0.01), ("2", -4.6)]))
        cal = self.calibration({DEFAULT_MODEL: {"T": 2.0, "T_enum": 4.0}})
        out = json.loads(run(fake.host, "-j", "-e", "red,blue", "which?", "t", calibration=cal).stdout)
        self.assertEqual((out["label"], out["T"]), ("red", 4.0))
        self.assertAlmostEqual(out["p"]["red"], 1 / (1 + 2 * math.exp((-4.6 + 0.01) / 4.0)), places=3)

    def test_a_yes_no_temperature_is_never_applied_to_an_e_question(self):
        fake = self.serve(chat([("1", -0.01), ("2", -4.6)]))
        cal = self.calibration({DEFAULT_MODEL: {"T": 2.0}})
        out = json.loads(run(fake.host, "-j", "-e", "red,blue", "which?", "t", calibration=cal).stdout)
        self.assertEqual((out["T"], "p_raw" in out, out["p"]["red"]), (None, False, 0.99))

    def test_a_model_without_a_fitted_temperature_keeps_its_raw_p(self):
        fake = self.serve(chat([("No", -0.01), ("Yes", -4.6)]))
        cal = self.calibration({DEFAULT_MODEL: {"T": 2.0}})
        out = json.loads(run(f"{fake.host}=llama3.2:3b", "-j", "-l", "yes,no", "q?", "t", calibration=cal).stdout)
        self.assertEqual((out["p"]["no"], out["T"], "p_raw" in out), (0.99, None, False))

    def test_an_unreadable_calibration_file_leaves_p_raw(self):
        fake = self.serve(chat([("No", -0.01), ("Yes", -4.6)]))
        for text in ("{not json", json.dumps({DEFAULT_MODEL: {"T": 0}}), json.dumps({DEFAULT_MODEL: 2})):
            path = Path(self.enterContext(tempfile.TemporaryDirectory())) / "c.json"
            path.write_text(text)
            r = run(fake.host, "-j", "-l", "yes,no", "q?", "t", calibration=str(path))
            self.assertEqual((json.loads(r.stdout)["T"], r.returncode), (None, 1), text)

    def test_labels_alone_default_the_question(self):
        fake = self.serve(chat([("bin", -0.1)]))
        r = run(fake.host, "-l", "alias,binary", stdin="/usr/local/bin/claude")
        self.assertEqual((r.returncode, r.stdout), (1, "binary 1.00\n"))
        self.assertEqual(fake.requests[0]["messages"][1]["content"],
                         "Text:\n/usr/local/bin/claude\n\nWhich one fits this text? Answer alias or binary.")

    def test_gate_passes_the_input_unchanged_and_prints_nothing_else(self):
        fake = self.serve(chat([("Yes", -0.05), ("No", -3.0)]))
        r = run(fake.host, "-p", "is this an error?", stdin="  make: *** [all] Error 2\n\n")
        self.assertEqual((r.returncode, r.stdout, r.stderr), (0, "  make: *** [all] Error 2\n\n", ""))
        r = run(fake.host, "-p", "-j", "is this an error?", stdin="boom\n")
        self.assertEqual((r.stdout, json.loads(r.stderr)["label"]), ("boom\n", "yes"))

    def test_gate_prints_nothing_when_another_label_wins(self):
        fake = self.serve(chat([("No", -0.05), ("Yes", -3.0)]))
        r = run(fake.host, "-p", "is this an error?", stdin="all green\n")
        self.assertEqual((r.returncode, r.stdout, r.stderr), (1, "", ""))

    def test_gate_still_reports_why_it_could_not_decide(self):
        r = run(DEAD, "-p", "is this an error?", stdin="boom\n")
        self.assertEqual((r.returncode, r.stdout), (2, ""))
        self.assertIn("classif: unscored: no Ollama host answered", r.stderr)

    def test_gate_at_a_terminal_shows_its_verdict_on_stderr(self):
        import pty
        for top, label, out in ((("No", -0.05), ("Yes", -3.0)), "no 0.95", ""), \
                               ((("Yes", -0.05), ("No", -3.0)), "yes 0.95", "boom\n"):
            fake = self.serve(chat(list(top)))
            master, slave = pty.openpty()
            r = subprocess.run([str(SCRIPT), "-p", "is this an error?"], input="boom\n", stdout=subprocess.PIPE,
                               stderr=slave, text=True, timeout=10, env=env(fake.host))
            os.close(slave)
            err = os.read(master, 4096).decode()
            os.close(master)
            self.assertEqual(r.stdout, out)
            self.assertEqual(err, f"\x1b[2mclassif: {label}\x1b[0m\r\n")

    def test_long_input_goes_in_whole_with_a_32k_window_and_no_silent_cut(self):
        fake = self.serve(chat([("yes", -0.1)]))
        r = run(fake.host, "q?", "x" * 90000)
        req = fake.requests[0]
        self.assertEqual((req["options"]["num_ctx"], req["truncate"]), (32768, False))
        self.assertIn("Text:\n" + "x" * 90000 + "\n\nq?", req["messages"][1]["content"])
        self.assertEqual((r.returncode, r.stdout, r.stderr), (0, "yes 1.00\n", ""))

    def test_overflow_in_a_bounded_reader_stays_unscored_with_the_server_reason(self):
        inner = {"error": {"code": 400, "message": "request (40000 tokens) exceeds the available "
                           "context size (32768 tokens), try increasing it"}}
        fake = self.serve((400, {"error": json.dumps(inner)}))
        r = run(fake.host, "q?", "x")
        self.assertEqual((r.returncode, r.stdout), (2, ""))
        self.assertEqual(len(fake.requests), 3)     # the whole read, the root call, one leaf
        self.assertEqual(r.stderr, f"classif: unscored: 1 leaf calls failed: {fake.host} HTTP 400: request (40000 tokens) "
                                   "exceeds the available context size (32768 tokens), try increasing it\n")

    def test_context_past_the_window_with_no_input_is_unscored_not_none(self):
        inner = {"error": {"code": 400, "message": "request (40000 tokens) exceeds the available "
                           "context size (32768 tokens), try increasing it"}}
        fake = self.serve((400, {"error": json.dumps(inner)}))
        with tempfile.NamedTemporaryFile("w", suffix=".txt") as ctx:
            ctx.write("x" * 1000)
            ctx.flush()
            r = run(fake.host, "-e", "good,bad", "is it good?", "-c", ctx.name)
        self.assertEqual((r.returncode, r.stdout, len(fake.requests)), (2, "", 1))
        self.assertIn("-c is context added to every call and must fit the window", r.stderr)

    def test_low_label_mass_is_unscored_and_names_what_the_model_wanted(self):
        fake = self.serve(chat([("Maybe", -0.2), ("yes", -2.5)]))
        r = run(fake.host, "q?", "x")
        self.assertEqual((r.returncode, r.stdout), (2, ""))
        self.assertIn("label mass 0.08 < 0.5; model wanted: Maybe,yes", r.stderr)

    def test_response_without_logprobs_is_unscored(self):
        fake = self.serve((200, {"done": True, "message": {"content": "yes"}}))
        r = run(fake.host, "q?", "x")
        self.assertEqual(r.returncode, 2)
        self.assertIn("response has no logprobs", r.stderr)

    def test_unfinished_response_is_unscored(self):
        fake = self.serve(chat([("yes", -0.1)], done=False))
        self.assertEqual(run(fake.host, "q?", "x").returncode, 2)

    def test_missing_model_is_unscored_with_the_server_error(self):
        fake = self.serve((404, {"error": "model 'tiny:1b' not found"}))
        r = run(f"{fake.host}=tiny:1b", "q?", "x")
        self.assertEqual(r.returncode, 2)
        self.assertIn("HTTP 404", r.stderr)
        self.assertIn("model 'tiny:1b' not found", r.stderr)

    def test_dead_first_host_falls_through_to_a_live_one(self):
        fake = self.serve(chat([("yes", -0.1)]))
        r = run(f"{DEAD},{fake.host}", "-j", "q?", "x")
        self.assertEqual((r.returncode, json.loads(r.stdout)["host"]), (0, fake.host))

    def test_no_live_host_is_unscored_and_lists_the_hosts_tried(self):
        r = run(DEAD, "q?", "x")
        self.assertEqual(r.returncode, 2)
        self.assertIn(f"no Ollama host answered: {DEAD}", r.stderr)

    def test_text_split_by_its_own_quotes_points_to_stdin(self):
        r = run(DEAD, 'I thought "comfort', "in", 'between" mattered', "is this good")
        self.assertEqual(r.returncode, 2)
        self.assertIn("got 4 arguments, want at most 2", r.stderr)
        self.assertIn('xsel -ob | classif "question"', r.stderr)

    def test_enum_asks_for_a_digit_and_prints_the_name(self):
        fake = self.serve(chat([("2", -0.05), ("1", -3.0)]))
        r = run(fake.host, "-e", "Gamgee,Baggins", "Frodo's surname?", "frodo bollgkins")
        self.assertEqual((r.returncode, r.stdout), (1, "Baggins 0.95\n"))
        msgs = fake.requests[0]["messages"]
        self.assertEqual(msgs[0]["content"], "Answer with exactly one word: 1, 2 or 0.")
        self.assertEqual(msgs[1]["content"], "Text:\nfrodo bollgkins\n\nFrodo's surname? "
                                             "Options: 1=Gamgee, 2=Baggins, 0=none of these. Answer 1, 2 or 0.")

    def test_text_after_an_option_is_the_input(self):
        fake = self.serve(chat([("2", -0.05), ("1", -3.0)]))
        r = run(fake.host, "Frodo's surname?", "-e", "Gamgee,Baggins", "frodo bollgkins")
        self.assertEqual((r.returncode, r.stdout), (1, "Baggins 0.95\n"))
        self.assertTrue(fake.requests[0]["messages"][1]["content"].startswith("Text:\nfrodo bollgkins\n"))

    def test_enum_first_option_exits_0_and_json_is_keyed_by_name(self):
        fake = self.serve(chat([("1", -0.05), ("2", -3.0)]))
        r = run(fake.host, "-j", "-e", "Baggins,Gamgee", "q?", "x")
        out = json.loads(r.stdout)
        self.assertEqual((r.returncode, out["label"], sorted(out["p"])), (0, "Baggins", ["Baggins", "Gamgee", "none"]))

    def test_json_is_indented_on_a_terminal_and_one_line_in_a_pipe(self):
        fake = self.serve(chat([("1", -0.05), ("2", -3.0)]))
        args = ("-j", "-e", "Baggins,Gamgee", "q?", "x")
        self.assertEqual(run(fake.host, *args).stdout.count("\n"), 1)
        master, slave = pty.openpty()
        subprocess.run([str(SCRIPT), *args], stdout=slave, stderr=subprocess.DEVNULL,
                       env=env(fake.host, "/nonexistent"), timeout=10)
        os.close(slave)
        out = b""
        while True:
            try:
                chunk = os.read(master, 4096)
            except OSError:  # EIO once the child's end is closed
                break
            if not chunk:
                break
            out += chunk
        os.close(master)
        self.assertIn('\n  "label": "Baggins"', out.decode())
        self.assertEqual(json.loads(out)["label"], "Baggins")

    def test_enum_takes_a_newline_list_from_a_command(self):
        fake = self.serve(chat([("3", -0.05)]))
        r = run(fake.host, "-e", "main\nfeat/a\n\nfix/b\n", "which branch?", "x")
        self.assertEqual((r.returncode, r.stdout), (1, "fix/b 1.00\n"))
        self.assertIn("Options: 1=main, 2=feat/a, 3=fix/b, 0=none of these.", fake.requests[0]["messages"][1]["content"])

    def test_enum_option_description_goes_to_the_model_and_only_the_name_comes_back(self):
        fake = self.serve(chat([("2", -0.05), ("1", -3.0)]))
        r = run(fake.host, "-j", "-e", "prod=live customer traffic,staging=pre-release checks", "which env?", "x")
        out = json.loads(r.stdout)
        self.assertEqual((out["label"], sorted(out["p"])), ("staging", ["none", "prod", "staging"]))
        self.assertIn("Options: 1=prod (live customer traffic), 2=staging (pre-release checks), 0=none of these.",
                      fake.requests[0]["messages"][1]["content"])

    def test_a_newline_list_keeps_commas_inside_descriptions(self):
        fake = self.serve(chat([("1", -0.05)]))
        r = run(fake.host, "-e", "Gamgee=the gardener, Frodo's friend\nBaggins", "who?", "x")
        self.assertEqual(r.stdout, "Gamgee 1.00\n")
        self.assertIn("Options: 1=Gamgee (the gardener, Frodo's friend), 2=Baggins, 0=none of these.",
                      fake.requests[0]["messages"][1]["content"])

    def test_e_repeats_one_option_each_and_its_description_may_hold_commas(self):
        fake = self.serve(chat([("2", -0.05)]))
        r = run(fake.host, "did this page move?", "-e", "moved=the page says it moved, with a new address",
                "-e", "live=the page itself", "x")
        self.assertEqual((r.returncode, r.stdout), (1, "live 1.00\n"))
        self.assertIn("Options: 1=moved (the page says it moved, with a new address), 2=live (the page itself), "
                      "0=none of these.", fake.requests[0]["messages"][1]["content"])

    def test_repeated_e_mixes_name_lists_and_described_options(self):
        fake = self.serve(chat([("3", -0.05)]))
        r = run(fake.host, "-e", "a,b", "-e", "c=x, y", "q?", "x")
        self.assertEqual(r.stdout, "c 1.00\n")
        self.assertIn("Options: 1=a, 2=b, 3=c (x, y), 0=none of these.", fake.requests[0]["messages"][1]["content"])

    def test_a_description_runs_to_the_next_name(self):
        fake = self.serve(chat([("1", -0.05)]))
        run(fake.host, "-e", "a,prod=live traffic, mostly customers,staging=pre-release", "q?", "x")
        self.assertIn("Options: 1=a, 2=prod (live traffic, mostly customers), 3=staging (pre-release), 0=none of these.",
                      fake.requests[0]["messages"][1]["content"])

    def test_options_from_every_e_count_together(self):
        r = run(DEAD, "-e", "a=x", "-e", "A=y", "q?", "x")
        self.assertEqual(r.returncode, 2)
        self.assertIn("-e needs distinct named options, got a, A", r.stderr)
        fake = self.serve(chat([("no", -0.05), ("yes", -3.0)]))
        run(fake.host, "-e", "o1,o2,o3,o4,o5", "-e", "o6,o7,o8,o9,o10", "q?", "x")
        self.assertEqual(len(fake.requests), 10 + 1)

    def test_past_nine_options_each_is_asked_alone_and_the_three_likeliest_picked_among(self):
        def model(req):
            user = req["messages"][-1]["content"]
            if "Options:" in user:
                return chat([("2", -0.05), ("1", -3.0)])
            # o3, o7 and o11 are likelier than the rest, o7 most.
            y = {"o7": -0.1, "o3": -1.0, "o11": -1.5}.get(re.search(r"Is the answer (\w+)\?", user)[1], -4.0)
            return chat([("yes", y), ("no", math.log(1 - math.exp(y)))])
        fake = self.serve(model)
        opts = ",".join(f"o{i}" for i in range(1, 13))
        r = run(fake.host, "-j", "Which one?", "text", "-e", opts)
        self.assertEqual(r.returncode, 1)
        out = json.loads(r.stdout)
        self.assertEqual((out["label"], set(out["p"])), ("o7", {"o3", "o7", "o11", "none"}))
        self.assertEqual(set(out["screen"]), {f"o{i}" for i in range(1, 13)})
        self.assertEqual(out["screen"]["o7"], round(math.exp(-0.1), 3))
        self.assertIn("Which one? Is the answer o5? Answer yes, no or unknown.",
                      fake.requests[4]["messages"][-1]["content"])
        self.assertIn("Options: 1=o3, 2=o7, 3=o11, 0=none of these.", fake.requests[-1]["messages"][-1]["content"])
        self.assertEqual(len(fake.requests), 12 + 1)

    def test_past_nine_only_the_first_option_given_exits_0_not_the_first_finalist(self):
        def model(likely):
            def answer(req):
                user = req["messages"][-1]["content"]
                if "Options:" in user:
                    return chat([("1", -0.05), ("2", -3.0)])
                y = -0.1 if re.search(r"Is the answer (\w+)\?", user)[1] in likely else -4.0
                return chat([("yes", y), ("no", math.log(1 - math.exp(y)))])
            return answer
        opts = ",".join(f"o{i}" for i in range(1, 11))
        for likely, code in ((("o1", "o2", "o3"), 0), (("o5", "o6", "o7"), 1)):
            fake = self.serve(model(likely))
            r = run(fake.host, "Which one?", "text", "-e", opts)
            self.assertEqual((r.returncode, r.stdout.split()[0]), (code, likely[0]))

    def test_why_past_nine_options_is_refused(self):
        r = run(DEAD, "-w", "q?", "x", "-e", ",".join(f"o{i}" for i in range(10)))
        self.assertEqual(r.returncode, 2)
        self.assertIn("--why reads the text in pieces, which pick among at most 9 options; got 10", r.stderr)

    def test_the_input_file_flag_is_i_and_f_is_gone(self):
        out = run(DEAD, "--help").stdout
        self.assertIn("-i FILE, --input FILE", out)
        self.assertNotIn("--file", out)
        r = run(DEAD, "q?", "-f", "x.txt")
        self.assertEqual(r.returncode, 2)

    def test_progress_redraws_one_dim_line_and_clears_it(self):
        out = io.StringIO()
        progress = load().Progress(out, time.monotonic())
        progress(10, 100, "lines")
        progress(11, 100, "lines")      # within a quarter second: not drawn
        progress(100, 100, "lines")
        progress.clear()
        text = out.getvalue()
        self.assertEqual(text.count("\r"), 3)
        self.assertIn("read 100 of 100 lines", text)
        self.assertNotIn("read 11 of", text)
        self.assertTrue(text.endswith("\r\033[K"))

    def test_help_offers_e_and_hides_l(self):
        r = run(DEAD, "--help")
        self.assertIn("-e OPTION, --enum OPTION", r.stdout)
        self.assertNotIn("--labels", r.stdout)

    def test_help_shows_examples_as_written_exits_and_quoting(self):
        out = run(DEAD, "--help").stdout
        self.assertIn('\n    git log -1 --format=%B | classif "is this a bug fix?"\n', out)
        self.assertIn("Exit: 0 when the first option wins", " ".join(out.split()))
        self.assertIn("name=description, which the model reads as when to pick it", " ".join(out.split()))
        self.assertIn('\n    git diff | classif "what kind of change is this?" -e fix=repairs broken behaviour '
                      '-e feat=adds something new\n', out)

    def test_an_unquoted_e_description_split_by_the_shell_is_joined_again(self):
        fake = self.serve(chat([("1", -0.05)]))
        r = run(fake.host, "is this an AI project?", "-e", "LLM=machine", "learning", "-e", "BBM=bob", "learning",
                stdin="model.py\n")
        self.assertEqual((r.returncode, r.stdout), (0, "LLM 1.00\n"))
        msg = fake.requests[0]["messages"][1]["content"]
        self.assertTrue(msg.startswith("Text:\nmodel.py\n"))
        self.assertIn("Options: 1=LLM (machine learning), 2=BBM (bob learning), 0=none of these.", msg)

    def test_a_loose_word_after_a_described_option_no_longer_replaces_stdin(self):
        fake = self.serve(chat([("1", -0.05)]))
        run(fake.host, "did it move?", "-e", "moved=page", "moved", "-e", "live", stdin="<h1>301 Moved</h1>")
        msg = fake.requests[0]["messages"][1]["content"]
        self.assertTrue(msg.startswith("Text:\n<h1>301 Moved</h1>\n"))
        self.assertIn("Options: 1=moved (page moved), 2=live, 0=none of these.", msg)

    def test_a_one_word_text_joined_into_a_description_fails_loud(self):
        r = run(DEAD, "is it up?", "-e", "up=running", "active")
        self.assertEqual(r.returncode, 2)
        self.assertIn("no text is left", r.stderr)
        self.assertIn("put a one-word text before -e", r.stderr)

    def test_a_quoted_value_after_e_stays_the_text(self):
        for opt in ("a=x y", "a=x", "a,b"):
            fake = self.serve(chat([("1", -0.05)]))
            run(fake.host, "q?", "-e", opt, "the text")
            self.assertTrue(fake.requests[0]["messages"][1]["content"].startswith("Text:\nthe text\n"), opt)

    def run_at_terminal(self, hosts, *args):
        import pty
        master, slave = pty.openpty()
        self.addCleanup(os.close, master)
        try:
            return subprocess.run([str(SCRIPT), *args], stdin=slave, capture_output=True, text=True,
                                  timeout=10, env=env(hosts))
        finally:
            os.close(slave)

    def test_question_alone_at_a_terminal_is_judged_without_a_text(self):
        fake = self.serve(chat([("3", -0.05)]))
        r = self.run_at_terminal(fake.host, "-e", "Robert,Scundler,Baggins", "what is frodo's real surname?")
        self.assertEqual((r.returncode, r.stdout), (1, "Baggins 1.00\n"))
        self.assertEqual(fake.requests[0]["messages"][1]["content"], "what is frodo's real surname? "
                         "Options: 1=Robert, 2=Scundler, 3=Baggins, 0=none of these. Answer 1, 2, 3 or 0.")

    def test_no_question_and_no_text_at_a_terminal_is_rejected(self):
        r = self.run_at_terminal(DEAD, "-e", "a,b")
        self.assertEqual(r.returncode, 2)
        self.assertIn("no input: pass a question, a text, or both", r.stderr)

    def test_enum_escapes_to_none_when_no_option_fits(self):
        fake = self.serve(chat([("0", -0.05), ("3", -3.0)]))
        r = run(fake.host, "-e", "Robert,Scundler,Bagginskiskakis", "frodo's real surname?", "x")
        self.assertEqual((r.returncode, r.stdout), (1, "none 0.95\n"))
        r = run(fake.host, "-p", "-e", "Robert,Scundler", "q?", stdin="boom\n")
        self.assertEqual((r.returncode, r.stdout), (1, ""))

    def test_one_option_is_a_filter_against_none(self):
        fake = self.serve(chat([("1", -0.05), ("0", -3.0)]))
        r = run(fake.host, "-e", "hook", "about claude hooks?", "PreToolUse matcher")
        self.assertEqual((r.returncode, r.stdout), (0, "hook 0.95\n"))
        self.assertIn("Options: 1=hook, 0=none of these. Answer 1 or 0.", fake.requests[0]["messages"][1]["content"])

    def test_stdin_cut_mid_character_or_binary_is_still_judged(self):
        fake = self.serve(chat([("1", -0.05)]))
        r = subprocess.run([str(SCRIPT), "-e", "hook", "q?"], input=b"hook \xc3", capture_output=True,
                           env=env(fake.host), timeout=10)
        self.assertEqual((r.returncode, r.stdout, r.stderr), (0, b"hook 1.00\n", b""))
        self.assertIn("hook \ufffd", fake.requests[0]["messages"][1]["content"])

    def run_into_closed_pipe(self, hosts, *args, stdin=""):
        """stdout is a pipe nobody reads, as with `classif ... | head -c0`."""
        read_end, write_end = os.pipe()
        os.close(read_end)
        try:
            return subprocess.run([str(SCRIPT), *args], input=stdin, stdout=write_end, stderr=subprocess.PIPE,
                                  text=True, timeout=10, env=env(hosts))
        finally:
            os.close(write_end)

    def test_a_reader_that_leaves_early_gets_no_traceback_and_the_verdict_stands(self):
        fake = self.serve(chat([("No", -0.05), ("Yes", -3.0)]))
        r = self.run_into_closed_pipe(fake.host, "is it k8s?", "sourdough")
        self.assertEqual((r.returncode, r.stderr), (1, ""))
        fake.reply = chat([("Yes", -0.05), ("No", -3.0)])
        r = self.run_into_closed_pipe(fake.host, "-p", "is this an error?", stdin="boom\n" * 50000)
        self.assertEqual((r.returncode, r.stderr), (0, ""))

    def test_enum_needs_distinct_named_options(self):
        for opts in (",", "a,A", "a=x,A=y", "a,=orphan"):
            r = run(DEAD, "-e", opts, "q?", "x")
            self.assertEqual(r.returncode, 2, opts)
            self.assertIn("-e needs distinct named options", r.stderr, opts)

    def test_enum_and_labels_together_are_rejected(self):
        r = run(DEAD, "-e", "a,b", "-l", "c,d", "q?", "x")
        self.assertEqual(r.returncode, 2)
        self.assertIn("not allowed with argument", r.stderr)

    def test_duplicate_or_single_labels_are_rejected(self):
        for labels in ("yes,Yes", "yes"):
            r = run(DEAD, "-l", labels, "q?", "x")
            self.assertEqual(r.returncode, 2, labels)
            self.assertIn("need at least two distinct labels", r.stderr, labels)


class JudgeTests(unittest.TestCase):
    """judge() is the library entry a caller scoring many items uses: one
    call per item on a host it picked once, a dict back, nothing printed."""
    cls = load()

    def serve(self, reply):
        fake = FakeOllama(reply)
        self.addCleanup(fake.close)
        return fake

    def test_a_given_host_and_model_are_called_directly_and_the_result_comes_back(self):
        fake = self.serve(chat([("No", -0.05), ("Yes", -3.0)]))
        with mock.patch.object(self.cls, "route", side_effect=AssertionError("routed")), \
                mock.patch.dict(os.environ, {"CLASSIF_CALIBRATION": "/nonexistent"}):
            r = self.cls.judge("is it k8s?", "sourdough", ["yes", "no"], model="llama3.2:3b", host=fake.host)
        self.assertEqual((r["label"], r["host"], r["model"]), ("no", fake.host, "llama3.2:3b"))
        self.assertAlmostEqual(r["p"]["no"], math.exp(-0.05) / (math.exp(-0.05) + math.exp(-3.0)))
        self.assertEqual(fake.requests[0]["model"], "llama3.2:3b")
        self.assertIn("Text:\nsourdough", fake.requests[0]["messages"][1]["content"])

    def test_an_unscored_call_returns_the_reason_instead_of_printing_it(self):
        fake = self.serve(chat([("Maybe", -0.1), ("Yes", -3.0)]))
        err = io.StringIO()
        with mock.patch.object(sys, "stderr", err):
            r = self.cls.judge("q?", "t", ["yes", "no"], model="llama3.2:3b", host=fake.host)
        self.assertIsNone(r["label"])
        self.assertIn("label mass 0.05 < 0.5; model wanted: Maybe,Yes", r["unscored"])
        self.assertEqual(err.getvalue(), "")

    def test_keep_alive_comes_from_the_environment_and_minus_one_is_a_number(self):
        fake = self.serve(chat([("No", -0.05), ("Yes", -3.0)]))
        cfg = tempfile.mkdtemp()  # no keep_alive file, whatever this machine has
        self.addCleanup(shutil.rmtree, cfg)
        for env, want in (({}, "30m"), ({"CLASSIF_KEEP_ALIVE": "-1"}, -1), ({"CLASSIF_KEEP_ALIVE": "2h"}, "2h")):
            with mock.patch.dict(os.environ, dict(env, XDG_CONFIG_HOME=cfg)):
                if not env:
                    os.environ.pop("CLASSIF_KEEP_ALIVE", None)
                self.cls.judge("q?", "t", ["yes", "no"], model="m", host=fake.host)
            self.assertEqual(fake.requests[-1]["keep_alive"], want)

    def test_keep_alive_file_applies_when_the_environment_is_unset(self):
        fake = self.serve(chat([("No", -0.05), ("Yes", -3.0)]))
        cfg = Path(tempfile.mkdtemp())
        self.addCleanup(shutil.rmtree, cfg)
        (cfg / "classif").mkdir()
        (cfg / "classif" / "keep_alive").write_text("-1\n")
        with mock.patch.dict(os.environ, {"XDG_CONFIG_HOME": str(cfg)}):
            os.environ.pop("CLASSIF_KEEP_ALIVE", None)
            self.cls.judge("q?", "t", ["yes", "no"], model="m", host=fake.host)
            with mock.patch.dict(os.environ, {"CLASSIF_KEEP_ALIVE": "5m"}):
                self.cls.judge("q?", "t", ["yes", "no"], model="m", host=fake.host)
        self.assertEqual([r["keep_alive"] for r in fake.requests], [-1, "5m"])

    def test_a_caller_can_set_the_context_window_and_the_timeout(self):
        fake = self.serve(chat([("No", -0.05), ("Yes", -3.0)]))
        self.cls.judge("q?", "t", ["yes", "no"], model="m", host=fake.host, num_ctx=2048)
        self.assertEqual(fake.requests[0]["options"]["num_ctx"], 2048)
        seen = []
        real = self.cls.urllib.request.urlopen
        with mock.patch.object(self.cls.urllib.request, "urlopen",
                               lambda req, timeout: seen.append(timeout) or real(req, timeout=timeout)):
            self.cls.judge("q?", "t", ["yes", "no"], model="m", host=fake.host, timeout=0.7)
        self.assertEqual(seen, [0.7])

    def test_a_dead_host_is_unscored_not_raised(self):
        r = self.cls.judge("q?", "t", ["yes", "no"], model="llama3.2:3b", host=DEAD)
        self.assertIsNone(r["label"])
        self.assertTrue(r["unscored"].startswith(DEAD), r["unscored"])

class EmbedTests(unittest.TestCase):
    """The embedding client behind the search index."""
    cls = load()

    def serve(self, reply):
        fake = FakeOllama(reply)
        self.addCleanup(fake.close)
        return fake

    def test_embed_returns_unit_vectors_with_the_models_prefixes(self):
        fake = self.serve((200, {"embeddings": [[3.0, 4.0], [0.0, 2.0]], "prompt_eval_count": 7}))
        r = self.cls.embed(["a passage", "another"], host=fake.host)
        self.assertEqual(r, {"vectors": [[0.6, 0.8], [0.0, 1.0]], "tokens": 7})
        self.assertEqual(fake.requests[0]["model"], "embeddinggemma")
        self.assertEqual(fake.requests[0]["input"], ["title: none | text: a passage", "title: none | text: another"])
        self.cls.embed(["why?"], host=fake.host, query=True)
        self.assertEqual(fake.requests[1]["input"], ["task: search result | query: why?"])

    def test_embed_says_when_the_model_is_missing_or_the_reply_holds_no_vectors(self):
        fake = self.serve((404, {"error": "model 'embeddinggemma' not found"}))
        r = self.cls.embed(["x"], host=fake.host)
        self.assertEqual((r["vectors"], r["missing"]), (None, True))
        self.assertIn("404", r["unscored"])
        fake = self.serve(chat([("yes", -0.01)]))
        r = self.cls.embed(["x"], host=fake.host)
        self.assertEqual((r["vectors"], r.get("missing")), (None, None))
        self.assertIn("no embeddings", r["unscored"])


if __name__ == "__main__":
    unittest.main()
