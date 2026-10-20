"""Exercise the public CLI across a server-confirmed context overflow.

The HTTP stub scores evidence with canned labels. These tests verify routing,
coverage refusals, source offsets and shell behavior, not model accuracy.
"""
import json
import re
import math
import subprocess
import tempfile
import threading
import time
import unittest
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path

from test_judge import SCRIPT, chat, env

PUBLIC = SCRIPT.with_name("classif")


OVERFLOW = (400, {"error": "request (40000 tokens) exceeds the available context size (32768 tokens)"})


class SequencedOllama:
    def __init__(self, responder):
        self.requests = []
        fake = self

        class Handler(BaseHTTPRequestHandler):
            def do_POST(self):
                body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
                fake.requests.append(body)
                try:
                    status, reply = responder(body, len(fake.requests))
                except KeyError:
                    if "messages" in body:
                        raise
                    # A responder written for chat calls: the host has no embedding model.
                    status, reply = 404, {"error": f"model '{body.get('model')}' not found"}
                data = json.dumps(reply).encode()
                self.send_response(status)
                self.send_header("Content-Length", str(len(data)))
                self.end_headers()
                try:
                    self.wfile.write(data)
                except (BrokenPipeError, ConnectionResetError):
                    pass

            def log_message(self, *args):
                pass

        self.server = HTTPServer(("127.0.0.1", 0), Handler)
        self.host = f"127.0.0.1:{self.server.server_port}"
        threading.Thread(target=self.server.serve_forever, kwargs={"poll_interval": 0.01}, daemon=True).start()

    def close(self):
        self.server.shutdown()
        self.server.server_close()


def claim_reader(body, _):
    user = body["messages"][1]["content"]
    return chat([("fits" if "Taken alone" in user or "Does this passage" in user else "yes", -0.01)])


class AutoInputTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)

    def serve(self, responder):
        fake = SequencedOllama(responder)
        self.addCleanup(fake.close)
        return fake

    def call(self, fake, *args, text=None):
        """text=None gives the CLI no input at all, as at a terminal; "" is an empty pipe."""
        variables = env(fake.host)
        variables["XDG_CACHE_HOME"] = self.tmp.name
        feed = {"stdin": subprocess.DEVNULL} if text is None else {"input": text}
        return subprocess.run([str(PUBLIC), *args], text=True, capture_output=True, env=variables, timeout=10,
                              **feed)

    def test_short_input_stays_one_direct_call(self):
        fake = self.serve(lambda *_: chat([("yes", -0.01)]))
        result = self.call(fake, "-j", "Is this about installation?", text="Installation finished.")
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(json.loads(result.stdout)["mode"], "direct")
        self.assertEqual(len(fake.requests), 1)
        self.assertFalse(fake.requests[0]["truncate"])

    def test_overflow_uses_full_raw_source_and_named_file_offsets(self):
        fake = self.serve(lambda body, n: OVERFLOW if n == 1 else claim_reader(body, n))
        raw = " \r\n\u2615 Installation requires manual approval.\r\n  "
        path = Path(self.tmp.name) / "install.txt"
        path.write_bytes(raw.encode())
        result = self.call(fake, "-j", "-i", str(path), "Installation requires manual approval.")
        self.assertEqual(result.returncode, 0, result.stderr)
        data = json.loads(result.stdout)
        self.assertEqual((data["mode"], data["verdict"], data["label"]), ("memory", "supported", "yes"))
        self.assertEqual(data["read"]["file"], str(path))
        self.assertTrue(data["read"]["scan_complete"])
        self.assertEqual(data["read"]["unit"], "char")
        for source in data["read"]["sources"]:
            evidence = raw[source["start"]:source["end"]]
            self.assertIn("manual approval", evidence)
            self.assertIn(evidence.rstrip(), fake.requests[-1]["messages"][1]["content"])

    def test_nothing_is_written_to_disk_unless_cache_is_asked_for(self):
        store = Path(self.tmp.name) / "classif" / "mem.sqlite"
        for flags, expected in (((), False), (("--cache",), True)):
            fake = self.serve(lambda body, n: OVERFLOW if n == 1 else claim_reader(body, n))
            result = self.call(fake, *flags, "-j", "Installation requires manual approval.",
                               text="Installation requires manual approval.\n")
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertEqual(json.loads(result.stdout)["mode"], "memory")
            self.assertEqual(store.exists(), expected, flags)

    def test_default_gate_passes_raw_source_after_memory_verdict(self):
        fake = self.serve(lambda body, n: OVERFLOW if n == 1 else claim_reader(body, n))
        raw = " \nInstallation needs approval.\n\n"
        result = self.call(fake, "-p", "Installation needs approval.", text=raw)
        self.assertEqual((result.returncode, result.stdout), (0, raw), result.stderr)

    def test_unsettled_memory_judge_is_insufficient_and_passes_no_gate(self):
        def respond(body, n):
            if n == 1:
                return OVERFLOW
            user = body["messages"][1]["content"]
            return chat([("fits" if "Taken alone" in user else "unknown", -0.01)])

        fake = self.serve(respond)
        result = self.call(fake, "-p", "-j", "Installation is safe.", text="Installer started.\n")
        self.assertEqual((result.returncode, result.stdout), (3, ""))
        data = json.loads(result.stderr.splitlines()[0])
        self.assertEqual(data["verdict"], "insufficient")
        self.assertIsNone(data["label"])

    def test_threshold_blocks_a_memory_answer(self):
        def respond(body, n):
            if n == 1:
                return OVERFLOW
            user = body["messages"][1]["content"]
            return (chat([("fits", -0.01)]) if "Taken alone" in user else
                    chat([("yes", math.log(.6)), ("no", math.log(.4)), ("unknown", -100)]))

        fake = self.serve(respond)
        result = self.call(fake, "-p", "-j", "-t", ".8", "Installation is complete.",
                           text="Installation is complete.\n")
        self.assertEqual((result.returncode, result.stdout), (3, ""))
        self.assertTrue(json.loads(result.stderr.splitlines()[0])["unsure"])

    def test_enum_fallback_preserves_names_and_descriptions(self):
        fake = self.serve(lambda _, n: OVERFLOW if n == 1 else chat([("2", -0.01)]))
        result = self.call(fake, "-j", "-e", "routine=ordinary activity,review=serious harm",
                           "How should this discussion be routed?", text="A customer reports losing their savings.\n")
        self.assertEqual(result.returncode, 1, result.stderr)
        data = json.loads(result.stdout)
        self.assertEqual((data["mode"], data["label"], data["verdict"]), ("memory", "review", "answered"))
        self.assertEqual(set(data["p"]), {"routine", "review", "none"})
        self.assertEqual(data["read"]["checked"], data["read"]["passages"])
        for body in fake.requests:
            self.assertIn("serious harm", str(body["messages"]))

    def test_unrelated_http_error_and_low_mass_do_not_start_a_scan(self):
        replies = [(500, {"error": "runner unavailable"}),
                   (400, {"error": "context settings exceed supported GPU memory"}),
                   chat([("perhaps", -0.01)])]
        for reply in replies:
            with self.subTest(reply=reply):
                fake = self.serve(lambda *_, reply=reply: reply)
                result = self.call(fake, "q?", text="source")
                self.assertEqual(result.returncode, 2)
                self.assertEqual(len(fake.requests), 1)

    def test_custom_labels_overflow_is_explicitly_unscored(self):
        fake = self.serve(lambda *_: OVERFLOW)
        result = self.call(fake, "-l", "pass,stop", "May this comment be published?", text="comment")
        self.assertEqual(result.returncode, 2)
        self.assertIn("custom", result.stderr)
        self.assertEqual(len(fake.requests), 1)

    def test_past_the_window_a_claim_naming_a_line_is_answered_from_its_linked_lines(self):
        fake = self.serve(lambda body, n: OVERFLOW if n == 1 else claim_reader(body, n))
        long = "".join(f"Note {i}: routine check passed.\n" for i in range(200)) + "Invoice INV-42 was paid.\n"
        result = self.call(fake, "-j", "Invoice INV-42 is paid.", text=long)
        self.assertEqual(result.returncode, 0, result.stderr)
        read = json.loads(result.stdout)["read"]
        self.assertEqual((read["basis"], read["graph"]["component"]), ("component", 1))
        self.assertNotIn("tried", read)
        self.assertLess(len(fake.requests), 10)

    def test_past_the_window_a_name_on_too_many_lines_skips_the_linked_lines_attempt(self):
        # Linking every line that names Acme Ltd would cost more calls than reading every passage.
        def respond(body, n):
            if n == 1:
                return OVERFLOW
            if "The last line uses a word like" in str(body["messages"]):
                return chat([("0", -0.01)])
            return claim_reader(body, n)
        fake = self.serve(respond)
        long = "".join(f"Order {i} for Acme Ltd shipped and it arrived.\n" for i in range(300))
        result = self.call(fake, "-j", "Acme Ltd shipped an order.", text=long)
        self.assertEqual(result.returncode, 0, result.stderr)
        read = json.loads(result.stdout)["read"]
        # The first link call, if any, comes after the reading: the graph made none up front.
        asked = ["The last line uses a word like" in str(b["messages"]) for b in fake.requests]
        self.assertFalse(any(asked[:5]))
        self.assertIn("graph skipped", read["tried"]["why"])
        self.assertEqual(read["tried"]["calls"], 0)

    def test_past_the_window_an_exists_claim_is_answered_from_the_closest_passages(self):
        def respond(body, n):
            if "input" in body:
                # the index: a passage reads like the question when both say failure
                return 200, {"embeddings": [[1.0, 0.0] if "failure" in t else [0.0, 1.0] for t in body["input"]],
                             "prompt_eval_count": len(body["input"])}
            user = body["messages"][1]["content"]
            if len(user) > 40000:
                return OVERFLOW
            if "Which kind is it?" in user:
                return chat([(re.search(r"(\d)=one line", user).group(1), -0.01)])
            if "Taken alone" in user or "Does this passage" in user:
                return chat([("fits" if "failure" in user else "unrelated", -0.01)])
            return chat([("yes" if "failure: access denied" in user else "no", -0.01)])
        fake = self.serve(respond)
        long = "".join(f"Note {i}: routine check passed.\n" for i in range(2000)) + "Deployment failure: access denied.\n"
        result = self.call(fake, "-j", "An entry reports a deployment failure.", text=long)
        self.assertEqual(result.returncode, 0, result.stderr)
        read = json.loads(result.stdout)["read"]
        self.assertEqual((read["basis"], read["plan"], read["kind"]["kind"]), ("search", "search", "exists"))
        self.assertEqual(read["by"], "meaning and words")
        self.assertLess(read["checked"], read["passages"])
        self.assertNotIn("tried", read)
        self.assertLess(len(fake.requests), 10)
        # The question and every passage went through the embedding model once, with its prefixes.
        inputs = [t for b in fake.requests if "input" in b for t in b["input"]]
        self.assertEqual(sum(t.startswith("task: search result | query: An entry") for t in inputs), 1)
        self.assertEqual(sum(t.startswith("title: none | text: ") for t in inputs), read["passages"])
        # --why reads the judged passages line by line for the lines the answer rests on.
        result = self.call(fake, "-j", "--why", "An entry reports a deployment failure.", text=long)
        read = json.loads(result.stdout)["read"]
        self.assertEqual(read["evidence"], [{"line": 2001, "end": 2001, "text": "Deployment failure: access denied.",
                                            "label": "fits"}])
        self.assertEqual(read["lines"]["verdict"], "supported")
        # Printed, each line says how it read, where a short text's line says how far it moved p.
        result = self.call(fake, "--why", "An entry reports a deployment failure.", text=long)
        self.assertEqual(result.stdout.splitlines()[1:], ["  fits    2001: Deployment failure: access denied."])
        # From files, the label stays with the line in its own file.
        path = Path(self.tmp.name) / "long.log"
        path.write_text(long)
        result = self.call(fake, "--why", "-i", str(path), "An entry reports a deployment failure.")
        self.assertEqual(result.stdout.splitlines()[1:], [f"  fits    {path}:2001: Deployment failure: access denied."])

    def test_past_the_window_without_an_embedding_model_the_scan_stops_at_a_witness(self):
        def respond(body, n):
            if "input" in body:
                return 404, {"error": "model 'embeddinggemma' not found"}
            user = body["messages"][1]["content"]
            if n == 1:
                return OVERFLOW
            if "Which kind is it?" in user:
                # the root call: pick the option saying one line is enough
                return chat([(re.search(r"(\d)=one line", user).group(1), -0.01)])
            if "Taken alone" in user or "Does this passage" in user:
                return chat([("fits" if "failure" in user else "unrelated", -0.01)])
            # The search by words alone says no: not trusted, so the scan runs.
            return chat([("no" if "is this shown" in user else "yes", -0.01)])
        fake = self.serve(respond)
        long = "Deployment failure: access denied.\n" + "".join(f"Note {i}: routine check passed.\n" for i in range(200))
        result = self.call(fake, "-j", "An entry reports a deployment failure.", text=long)
        self.assertEqual(result.returncode, 0, result.stderr)
        read = json.loads(result.stdout)["read"]
        self.assertEqual((read["basis"], read["kind"]["kind"]), ("witness", "exists"))
        self.assertNotIn("tried", read)      # a witness claim skips the linked-lines attempt
        self.assertEqual((read["search"]["verdict"], read["search"]["index"]["missing"]), ("contradicted", True))
        self.assertIn("ollama pull embeddinggemma", result.stderr)
        self.assertLess(len(fake.requests), 20)
        self.assertEqual(read["evidence"], [{"line": 1, "end": 1, "text": "Deployment failure: access denied.",
                                            "label": "fits"}])

    @staticmethod
    def witness(body, n):
        """A reader that finds the failure line. It keys on the line's words,
        since the claim in every prompt says failure."""
        user = body["messages"][1]["content"]
        if "Which kind is it?" in user:
            return chat([(re.search(r"(\d)=one line", user).group(1), -0.01)])
        if "Taken alone" in user or "Does this passage" in user:
            return chat([("fits" if "access denied" in user else "unrelated", -0.01)])
        return chat([("yes", -0.01)])

    @staticmethod
    def settled(body, n):
        """One call on the whole text: yes while the failure line is in it."""
        return chat([("yes" if "access denied" in body["messages"][1]["content"] else "no", -0.01)])

    SHORT = "Note 0: routine check passed.\nDeployment failure: access denied.\nNote 2: routine check passed.\n"
    # Past 8 lines -w narrows: it leaves lines out in runs, not one at a time.
    LONG = "".join(f"Note {i}: routine check passed.\n" for i in range(61))
    MARKED = ("why: the lines whose removal moves p(yes) most", "  +1.000  2: Deployment failure: access denied.")

    @staticmethod
    def read_line_by_line(fake):
        return [r for r in fake.requests if "Which kind is it?" in r["messages"][1]["content"]]

    def test_why_keeps_the_one_call_answer_and_marks_the_line_that_moves_it(self):
        fake = self.serve(self.settled)
        plain = self.call(fake, "An entry reports a deployment failure.", text=self.SHORT)
        result = self.call(fake, "--why", "An entry reports a deployment failure.", text=self.SHORT)
        self.assertEqual(result.returncode, 0, result.stderr)
        verdict, *lines = result.stdout.splitlines()
        self.assertEqual(verdict, plain.stdout.strip())
        self.assertEqual(lines, list(self.MARKED))
        self.assertEqual((len(fake.requests), self.read_line_by_line(fake)), (1 + 1 + 3, []))

    def test_why_names_the_file_and_its_own_line_numbers(self):
        fake = self.serve(self.settled)
        a, b = Path(self.tmp.name) / "a.log", Path(self.tmp.name) / "b.log"
        a.write_text("Note 0: all fine.\nNote 1: all fine.")
        b.write_text("Note 2: all fine.\n  Deployment failure: access denied.\n")
        result = self.call(fake, "--why", "-i", f"{a},{b}", "An entry reports a deployment failure.")
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(result.stdout.splitlines()[2:], [f"  +1.000  {b}:2: Deployment failure: access denied."])
        result = self.call(fake, "-j", "--why", "-i", f"{a},{b}", "An entry reports a deployment failure.")
        self.assertEqual(json.loads(result.stdout)["marks"], [
            {"from": "text", "at": 2, "line": "  Deployment failure: access denied.", "delta": 1.0, "file": str(b)}])

    def test_why_numbers_lines_of_the_input_past_its_leading_blank_lines(self):
        fake = self.serve(self.settled)
        result = self.call(fake, "--why", "An entry reports a deployment failure.", text="\n\n" + self.SHORT)
        self.assertEqual(result.stdout.splitlines()[1:], [self.MARKED[0], self.MARKED[1].replace("2:", "4:")])

    def test_why_puts_the_marks_in_the_json_of_a_direct_answer(self):
        fake = self.serve(self.settled)
        result = self.call(fake, "-j", "-w", "An entry reports a deployment failure.", text=self.SHORT)
        self.assertEqual(result.returncode, 0, result.stderr)
        data = json.loads(result.stdout)
        self.assertEqual((data["mode"], data["label"]), ("direct", "yes"))
        self.assertEqual(data["marks"], [{"from": "text", "at": 2, "line": "Deployment failure: access denied.",
                                          "delta": 1.0}])

    def test_why_on_a_gate_keeps_the_passed_text_clean(self):
        fake = self.serve(self.settled)
        result = self.call(fake, "-p", "--why", "An entry reports a deployment failure.", text=self.SHORT)
        self.assertEqual((result.returncode, result.stdout), (0, self.SHORT), result.stderr)
        self.assertIn(self.MARKED[1], result.stderr)

    def test_why_on_one_line_says_it_is_not_marked(self):
        fake = self.serve(self.settled)
        result = self.call(fake, "--why", "An entry reports a deployment failure.", text="Note 0: all fine.\n")
        self.assertEqual(result.stdout.splitlines(), [
            "no 1.00", "why: not marked, there is only one line to leave out"])

    def test_why_on_a_one_line_c_judged_as_text_does_not_deny_the_c(self):
        # With no INPUT, -c becomes the text, so "there is no -c" was false.
        fake = self.serve(self.settled)
        result = self.call(fake, "--why", "-c", "Note 0: all fine.", "An entry reports a deployment failure.")
        self.assertEqual(result.stdout.splitlines(), [
            "no 1.00", "why: not marked, there is only one line to leave out"])

    def test_why_marks_the_option_that_won(self):
        fake = self.serve(lambda body, n: chat([("2" if "payment was reversed" in body["messages"][1]["content"] else "1",
                                                 -0.01)]))
        text = "Invoice INV-42 was issued.\nThat payment was reversed.\n"
        result = self.call(fake, "--why", "-e", "paid,reversed", "What is the status?", text=text)
        self.assertEqual(result.returncode, 1, result.stderr)
        self.assertEqual(result.stdout.splitlines(), ["reversed 1.00", "why: the lines whose removal moves p(reversed) most",
                                                      "  +1.000  2: That payment was reversed."])

    def test_why_on_a_long_text_narrows_to_the_line_in_a_few_calls(self):
        fake = self.serve(self.settled)
        text = self.LONG.replace("Note 37: routine check passed.", "Deployment failure: access denied.")
        result = self.call(fake, "--why", "An entry reports a deployment failure.", text=text)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(result.stdout.splitlines(), ["yes 1.00", self.MARKED[0],
                                                      "  +1.000  38: Deployment failure: access denied."])
        # The answer, then 4 runs of 16 lines, 4 of 4 and 4 single lines: 13 calls where one per line took 62.
        self.assertEqual((len(fake.requests), self.read_line_by_line(fake)), (1 + 12, []))

    def test_why_with_options_on_a_long_text_marks_the_line_for_the_option_that_won(self):
        fake = self.serve(lambda body, n: chat([("2" if "That payment was reversed." in body["messages"][1]["content"]
                                                 else "1", -0.01)]))
        text = "Invoice INV-42 was issued.\nIt was sent to the customer.\nThat payment was reversed.\n" + self.LONG
        result = self.call(fake, "--why", "-e", "paid,reversed", "What is the status?", text=text)
        self.assertEqual(result.returncode, 1, result.stderr)
        self.assertEqual(result.stdout.splitlines(), ["reversed 1.00", "why: the lines whose removal moves p(reversed) most",
                                                      "  +1.000  3: That payment was reversed."])

    def test_terminal_escapes_never_reach_the_model_and_a_gate_passes_them_on(self):
        fake = self.serve(self.settled)
        text = "\x1b[31mNote 0\x1b[0m: routine.\nDeployment \x1b]8;;http://x\x1b\\failure\x1b]8;;\x1b\\: access denied.\n"
        rule = Path(self.tmp.name) / "rule.txt"
        rule.write_text("\x1b[32mRule\x1b[0m: be brief.\n")
        result = self.call(fake, "-p", "-c", "\x1b[1mPolicy\x1b[0m: report failures.", "-c", str(rule),
                           "An entry reports a deployment failure.", text=text)
        self.assertEqual((result.returncode, result.stdout), (0, text), result.stderr)
        user = fake.requests[0]["messages"][1]["content"]
        self.assertIn("Context:\nPolicy: report failures.\n\nRule: be brief.\n\nText:\nNote 0: routine.\n"
                      "Deployment failure: access", user)

    def test_why_counts_lines_across_escapes_that_hold_or_end_a_line(self):
        fake = self.serve(self.settled)
        text = "\x1b[0m\nNote 1: \x1b]2;a title\nover two lines\x07all fine.\nDeployment failure: access denied.\n"
        result = self.call(fake, "--why", "An entry reports a deployment failure.", text=text)
        self.assertEqual(result.stdout.splitlines()[1:], [self.MARKED[0], self.MARKED[1].replace("2:", "4:")])

    def test_why_with_only_c_points_at_lines_of_c(self):
        # With no INPUT, -c is the text judged, so its lines are the ones cited.
        fake = self.serve(self.settled)
        result = self.call(fake, "--why", "-c", self.SHORT, "An entry reports a deployment failure.")
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(result.stdout.splitlines()[1:], list(self.MARKED))

    def test_without_input_c_is_the_text_judged_not_context(self):
        fake = self.serve(lambda *_: chat([("yes", -0.01)]))
        result = self.call(fake, "-j", "-c", "Usage: classif QUESTION\n  -w  print the lines", "Is this good help?")
        self.assertEqual(result.returncode, 0, result.stderr)
        user = fake.requests[0]["messages"][1]["content"]
        self.assertIn("Text:\nUsage: classif QUESTION", user)
        self.assertNotIn("Context:", user)
        self.assertNotIn("context", json.loads(result.stdout))

    def test_an_empty_pipe_or_argument_stays_an_empty_input_beside_c(self):
        # git diff with no changes must not get the policy judged in its place,
        # or passed down a -p gate.
        fake = self.serve(lambda *_: chat([("yes", -0.01)]))
        for args, text in ((("-p", "Does this touch secrets?"), ""), (("-p", "Does this touch secrets?", ""), None)):
            result = self.call(fake, *args, "-c", "Secrets policy: tokens never leave the vault.", text=text)
            self.assertEqual(result.stdout, "", args)
            user = fake.requests[-1]["messages"][1]["content"]
            self.assertIn("Context:\nSecrets policy", user)
            self.assertNotIn("Text:\nSecrets policy", user)

    def test_c_takes_the_text_itself_as_context(self):
        fake = self.serve(lambda *_: chat([("yes", -0.01)]))
        result = self.call(fake, "-c", "Change policy: no deploys on Fridays.", "Does this break the policy?",
                           "Deployed payments-api on Friday.", text="")
        self.assertEqual(result.returncode, 0, result.stderr)
        user = fake.requests[0]["messages"][1]["content"]
        self.assertIn("Context:\nChange policy: no deploys on Fridays.\n\nText:\nDeployed payments-api", user)

    def test_an_unquoted_command_split_into_words_says_to_quote_it(self):
        fake = self.serve(lambda *_: chat([("yes", -0.01)]))
        result = self.call(fake, "who am I", "-c", "user", "is:", "decoder", text="")
        self.assertEqual(result.returncode, 2)
        self.assertIn('got 3 arguments, want at most 2 (question, text). The shell split a text into words: quote '
                      'it, $(cmd) as "$(cmd)" too', result.stderr)
        self.assertEqual(fake.requests, [])

    def test_a_c_value_that_reads_as_a_missing_file_is_refused(self):
        fake = self.serve(lambda *_: chat([("yes", -0.01)]))
        for value in ("rulez.md", "./rules", "policies/deploy"):
            result = self.call(fake, "-c", value, "Is this fine?", text="fine")
            self.assertEqual(result.returncode, 2, value)
            self.assertIn(f"-c {value}: No such file or directory. -c takes a file, <(cmd) or the text itself",
                          result.stderr)
        self.assertEqual(fake.requests, [])

    def test_why_with_custom_labels_is_refused_before_any_call(self):
        fake = self.serve(lambda *_: chat([("a", -0.01)]))
        result = self.call(fake, "--why", "-l", "a,b", "Which?", text="x\n")
        self.assertEqual(result.returncode, 2)
        self.assertIn("--why", result.stderr)
        self.assertEqual(fake.requests, [])

    def test_positional_filename_remains_literal_text(self):
        fake = self.serve(lambda *_: chat([("yes", -0.01)]))
        path = Path(self.tmp.name) / "exists.txt"
        path.write_text("Hidden file content")
        result = self.call(fake, "q?", str(path))
        self.assertEqual(result.returncode, 0)
        user = fake.requests[0]["messages"][1]["content"]
        self.assertIn(str(path), user)
        self.assertNotIn("Hidden file content", user)

    def test_command_word_can_be_an_escaped_question(self):
        fake = self.serve(lambda *_: chat([("yes", -0.01)]))
        result = self.call(fake, "--", "unload", "source")
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("unload", fake.requests[0]["messages"][1]["content"])

    def test_invalid_memory_flags_are_rejected_before_reading_or_scoring(self):
        fake = self.serve(lambda *_: chat([("yes", -0.01)]))
        # How a long text is read, and by which model, is the machine's choice,
        # so -P, -g, --share, --no-cache and -m are not flags.
        for flags in (("--share", "-e", "a,b", "q?"), ("-P", "exists", "q?"), ("-g", "q?"), ("--no-cache", "q?"),
                      ("-m", "tiny:1b", "q?"), ("-i", "missing.txt", "q?", "text"), ("-f", "x.txt", "q?")):
            with self.subTest(flags=flags):
                result = self.call(fake, *flags, text="source")
                self.assertEqual(result.returncode, 2)
                self.assertIn("classif:", result.stderr)
        self.assertEqual(fake.requests, [])

    def test_deadline_includes_rejected_direct_attempt(self):
        def delayed_overflow(*_):
            time.sleep(.16)
            return OVERFLOW

        fake = self.serve(delayed_overflow)
        result = self.call(fake, "-d", ".08", "-p", "-j", "q?", text="source")
        self.assertEqual((result.returncode, result.stdout), (3, ""))
        self.assertEqual(len(fake.requests), 1)
        self.assertEqual(json.loads(result.stderr.splitlines()[0])["verdict"], "insufficient")


if __name__ == "__main__":
    unittest.main()
