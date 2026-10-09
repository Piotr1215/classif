"""A semantic if on a local model: one question, one text, one token.

    classif "is this sarcastic?" "great, another meeting"
    git log -1 --format=%B | classif "is this a bug fix?"
    curl -sL https://www.gutenberg.org/cache/epub/1342/pg1342.txt | classif "does Mr. Collins propose to Elizabeth?"
    crontab -l | classif "does anything run in the next hour?" -c <(echo "Now:"; date)
    make 2>&1 | classif -p "is this an error?" | ifne claude -p 'root cause?'

One /api/chat call for one token with logprobs. Each label's family (case
and leading-space variants, or the first piece of a multi-token label) is
summed and normalized; nothing is generated or parsed. Prints `<label> <p>`,
or JSON with -j: label, p, confidence ((n*pmax - 1)/(n - 1) over n labels,
0 at a tie and 1 when one label takes all), T, logp (each label's log mass),
mass, model, host, ms.

p is tempered when calibration.json beside the script holds a temperature
T for the model (evals/calibrate.py fits it): softmax(logp / T), with p_raw
in -j beside it. Raw gemma4:12b says 0.99 on cases it gets wrong; tempered,
its held-out ECE fell from 0.066 to 0.016 over 200 cases. No label changes
rank, so the winner and the exit code stay. -e questions use their own
temperature, T_enum. A model or a kind of question without one keeps its
raw p.

Labels default to yes,no,unknown: without unknown the model has to pick yes
or no about a fact the text never states ("is his friend older than me?"
answered no both ways). SemIf's missing-evidence option plays the same role.

Exit: 0 when the top label is the first label (yes by default), 1 for any
other label (no or unknown), 2 when unscored (no host, low mass, bad response),
3 when a long input's pieces did not settle the answer (insufficient).
With -t P (--min-p), a winner whose p is below P is unsure: exit 3 whichever
label leads, ` unsure` after the verdict, and a gate that passes nothing.

-e takes options by name and numbers them 1 to 9 behind the scenes: the
model answers a digit, classif prints the name, so any name scores, even one
the model would start with a one-character piece (Baggins starts "B").
Option 0, "none of these", is always added and prints `none`, so a list
without the answer is not forced onto its nearest name, the enum form of
unknown. Repeat -e for each option, or list names with commas or newlines,
so a command can generate the list: -e "$(git branch --format='%(refname:short)')".
`name=description` is an instruction to the model, not a note for the
reader: the model reads it as when to pick that option, and classif still
prints only the name. Unquoted, the shell splits it into words; classif
joins the words up to the next option or a value holding a space, the form
a quoted question or text has. A description runs to the next name=, so it may hold commas:
-e "prod=live customer traffic, mostly EU" -e "staging=pre-release checks".
In a newline list each line is one option.

With no text argument and nothing piped in, the question is judged alone.

-p makes classif a gate in a pipe: when the first label wins it prints the
input unchanged, otherwise nothing. At a terminal a dim verdict line goes to
stderr; off a terminal only unscored reasons do (and the result with -j). Put `ifne` (moreutils) before the next command so
it runs only on a pass.

Input goes in whole, up to a 32768-token window (about 80k chars of dense
text, more of prose). Nothing is judged on a fragment: when the server says
the input is past the window, classif hands it to mem.py beside it, which
reads it in pieces, keeps what it read outside the prompt, and answers only
when it read enough. The default labels then judge the question as a claim
(yes when the text supports it, no when it contradicts it) and -e picks
among the options over every passage. The hidden -l LABELS, kept for
evals/eval.py, scores the labels themselves without none, in one read only;
past the window it exits 2. `insufficient` (exit 3) means the
pieces did not settle it: the lines that bear on the claim leave it open, a
deadline cut the reading, or there is too much evidence for one final read.
It has no label and passes no gate. A claim that names something the text
names is answered first from the lines linked to it; when they do not settle
it, every line is read.

-c supplies the background the question needs, a policy, a reference or the
date, read before the text on every reader and judge call: a file, <(cmd), or
the text itself. Given no INPUT at all, no argument, -i or pipe, -c is the
text the question is asked of, so -c "$(cmd)" alone works like piping cmd in;
an empty pipe stays an empty INPUT. The context must fit the window with the text,
or with one passage on a long text; past that the server refuses and the
result is unscored. -i FILE reads the input from a file and names it in the
-j report; -i A,B or -i A -i B reads several as one input, each on its own
line after the last, and the report names the range of each. -d SECONDS
bounds everything, the whole read and the pieces after it. -j then
adds mode ("direct" or "memory"), verdict and read: what was read, the
source offsets of what the judge saw, calls and time.

Hosts, earliest reachable wins: CLASSIF_HOSTS=host:port[=model],... when set,
else ~/.config/classif/hosts with one such entry per line, else localhost:11434.
A host without a model runs DEFAULT_MODEL; -m overrides both. CLASSIF_TIMEOUT
(seconds, default 15) bounds the model call; a cold model load takes seconds.
CLASSIF_CALIBRATION names another calibration file.
"""
import argparse
import importlib.util
import json
import math
import os
import re
import socket
import stat
import sys
import threading
import time
import types
import urllib.error
import urllib.request

LOCAL_HOST = "localhost:11434"
# gemma4:12b over llama3.2:3b for ad-hoc questions: llama answers yes to both
# "is this good?" and "is this bad?" about the same text; gemma does not, and
# it scores higher in evals/eval.py (15/15 vs 13/15 synthetic, 66/70 vs 60/70
# RAG). Its raw p saturates near 0 or 1, so the threshold gate of plan #183
# pins llama3.2:3b with -m, whose raw p is graded and stays untempered.
# Winnow-12B (EldanRing, a Gemma 4 12B fine-tune for typed decisions) over
# gemma4:12b, both Q4_K_M through the same gemma4 renderer: 329/344 vs 325/344
# in evals/eval.py (7 won, 3 lost; email-action 54/60 vs 50/60), held-out NLL
# 0.166 vs 0.200 on -l, same latency (p50 222 vs 218 ms). Not in the Ollama
# library; README's setup builds it.
DEFAULT_MODEL = "winnow:12b-q4_K_M"
MASS_MIN = 0.5
# Tokens per request. Ollama's default here is 4096, and past its window it
# drops the start of the input silently. 32768 keeps whole pages and emails.
# Loaded on the laptop GPU: gemma4:12b 8.1GB, winnow:12b-q4_K_M 8.0GB,
# llama3.2:3b 6.0GB; short calls
# stay fast (~90ms and ~45ms); an 18k-token document took llama 5.4s.
NUM_CTX = 32768
# The search index past the window: a text's passages as vectors from a
# 300M embedding model, built once per text, 170k tokens in 8.5 s on this
# laptop's GPU against 100 s for the 12B to read them. Each model wants its
# own prefix on a query and on a passage; a model not listed gets none.
EMBED_MODEL = "embeddinggemma"
EMBED_PREFIX = {"embeddinggemma": ("task: search result | query: ", "title: none | text: "),
                "nomic-embed-text": ("search_query: ", "search_document: ")}
PROBE_TIMEOUT = 1.0
# Once any host answers, earlier ones get this long to answer too. It only
# matters with the first host down: a LAN host answered in 39-171ms over
# Wi-Fi, a wired one in under 1ms, and an unreachable name hangs.
GRACE = 0.3
# Fitted temperatures per model, written by evals/calibrate.py. Beside the
# script itself, so a symlink on PATH still finds it.
CALIBRATION = os.path.join(os.path.dirname(os.path.realpath(__file__)), "calibration.json")
DEFAULT_QUESTION = "Which one fits this text?"
DEFAULT_LABELS = "yes,no,unknown"
PROG = "classif"    # the name messages carry
# A whole read is prompt evaluation: 153k chars that fit the window took
# gemma4:12b 14.8s here, a hair under CLASSIF_TIMEOUT's default.
READ_RATE = 2000    # chars a second a whole read is allowed, on top of the default timeout


def config_dir():
    return os.path.join(os.environ.get("XDG_CONFIG_HOME") or os.path.expanduser("~/.config"), "classif")


def parse_hosts(entries):
    """host:port[=model] entries in order, the first of a repeated host kept.
    A host without a model runs DEFAULT_MODEL; # starts a comment."""
    out = {}
    for e in entries:
        host, _, model = e.split("#", 1)[0].strip().partition("=")
        if host.strip():
            out.setdefault(host.strip(), model.strip() or DEFAULT_MODEL)
    return list(out.items())


def hosts():
    """(host, model) in preference order: CLASSIF_HOSTS, comma separated, when
    set, else the hosts file in config_dir(), one entry per line, else the
    local server. The file reaches cron jobs and hooks, which an exported
    variable may not."""
    env = os.environ.get("CLASSIF_HOSTS", "").strip()
    if env:
        return parse_hosts(env.split(","))
    try:
        with open(os.path.join(config_dir(), "hosts")) as fh:
            listed = parse_hosts(fh.read().splitlines())
    except OSError:
        listed = []
    return listed or [(LOCAL_HOST, DEFAULT_MODEL)]


def route(forced=None):
    """Host to call and the model to run there: the caller's model when
    given, else the host's own. None for the host when none answered."""
    cands = hosts()
    host = pick_host([h for h, _ in cands])
    return host, forced or dict(cands).get(host)


def pick_host(candidates):
    """Earliest host in the list to accept a TCP connect. Probes run in
    parallel on daemon threads because an unresolvable .local name blocks in
    getaddrinfo for seconds, past any socket timeout. Once any host is up,
    earlier ones still pending get GRACE, then the earliest up host wins."""
    state = [None] * len(candidates)  # None pending, True up, False down
    cond = threading.Condition()

    def probe(i, h):
        name, _, port = h.rpartition(":")
        try:
            socket.create_connection((name, int(port)), timeout=PROBE_TIMEOUT).close()
            up = True
        except (OSError, ValueError):
            up = False
        with cond:
            state[i] = up
            cond.notify_all()

    for i, h in enumerate(candidates):
        threading.Thread(target=probe, args=(i, h), daemon=True).start()
    deadline = time.monotonic() + PROBE_TIMEOUT + 0.5
    with cond:
        while True:
            first = next((i for i, s in enumerate(state) if s is not False), None)
            if first is None:
                return None
            if state[first]:
                return candidates[first]
            if True in state:
                deadline = min(deadline, time.monotonic() + GRACE)
            left = deadline - time.monotonic()
            if left <= 0:
                return next((h for h, s in zip(candidates, state) if s), None)
            cond.wait(left)


def prompt(question, text, labels, options=None, context=None):
    """Text first, question last, one-word instruction in both turns. Beat a
    "You are a classifier" framing 13/15 to 9/15 on llama3.2:3b. With options
    the labels are digits and the question lists what each one stands for.
    context, a policy or reference the question is about, goes before the
    text: the same context on every call is then one cached prefix."""
    choices = " or ".join(labels) if len(labels) == 2 else ", ".join(labels[:-1]) + " or " + labels[-1]
    system = f"Answer with exactly one word: {choices}."
    question = question.strip()
    if options:
        question += " Options: " + ", ".join(f"{l}={o}" for l, o in zip(labels, options)) + "."
    body = (f"Context:\n{context}\n\n" if context else "") + (f"Text:\n{text}\n\n" if text else "")
    user = f"{body}{question} Answer {choices}."
    return system, user


def credit(tok, labels):
    """Label a first token belongs to, or None. A whole label matches in any
    case. A multi-token label ("garbage" is "gar"+"bage") only ever shows its
    first piece, so a piece of 2+ chars that starts exactly one label counts."""
    t = tok.strip().lower()
    if not t:
        return None
    exact = [l for l in labels if l.lower() == t]
    if exact:
        return exact[0]
    if len(t) < 2:
        return None
    pre = [l for l in labels if l.lower().startswith(t)]
    return pre[0] if len(pre) == 1 else None


def families(top, labels):
    """Each label's family mass: the probability of every token credited to it."""
    raw = {l: 0.0 for l in labels}
    for row in top:
        l = credit(str(row.get("token", "")), labels)
        if l:
            raw[l] += math.exp(float(row["logprob"]))
    return raw


def log_mass(top, labels):
    """Log family mass per label. A label with no token in the list gets the
    lowest logprob shown, an upper bound on any token past the list."""
    floor = min(float(row["logprob"]) for row in top)
    return {l: math.log(v) if v > 0 else floor for l, v in families(top, labels).items()}


def score(top, labels):
    """Family masses normalized over the labels. Returns (p per label, mass)."""
    raw = families(top, labels)
    mass = sum(raw.values())
    if mass <= 0:
        return {l: 0.0 for l in labels}, 0.0
    return {l: v / mass for l, v in raw.items()}, mass


def temperature(model, enum=False):
    """The model's fitted temperature, or None: an unfitted model keeps its
    raw p, so a threshold tuned on it (the #183 gate on llama3.2:3b) holds.
    -e questions have their own, T_enum: gemma4:12b's -l temperature left
    its -e answers further from the truth than raw p did."""
    try:
        with open(os.environ.get("CLASSIF_CALIBRATION", CALIBRATION)) as fh:
            t = json.load(fh)[model]["T_enum" if enum else "T"]
    except (OSError, ValueError, KeyError, TypeError):
        return None
    return float(t) if isinstance(t, (int, float)) and t > 0 else None


def calibrate(logp, t):
    """Temperature scaling: softmax of each label's log mass over t. t > 1
    softens a model that is surer than it is right, and no label changes
    rank, so the winner and the exit code stay."""
    top = max(logp.values())
    e = {l: math.exp((v - top) / t) for l, v in logp.items()}
    z = sum(e.values())
    return {l: v / z for l, v in e.items()}


def confidence(pmax, n):
    """Jev's rescaled top p: 0 when the n labels are a coin toss, 1 when one
    takes everything. A bare p reads differently with 2 labels than with 10."""
    return (n * pmax - 1) / (n - 1)


def server_error(body):
    """Ollama's error text. 0.34 wraps llama-server's JSON error as a string
    inside its own, so unwrap until a plain message is left."""
    msg = body
    while True:
        try:
            d = json.loads(msg)
        except (TypeError, ValueError):
            return msg
        if not isinstance(d, dict):
            return msg
        inner = d.get("error", d)
        nxt = inner.get("message") if isinstance(inner, dict) else inner
        if not isinstance(nxt, str):
            return msg
        msg = nxt


def keep_alive():
    """How long Ollama keeps the model loaded after a call: CLASSIF_KEEP_ALIVE,
    else the first line of the keep_alive file in the config dir, as a
    duration ("2h") or seconds, where -1 keeps it until something unloads it
    (classif unload does). The file reaches callers whose environment was set
    before it changed, such as hooks of a running session. Ollama reads "-1"
    as a bad duration, so a bare number goes out as a number. Unset, 30m."""
    v = os.environ.get("CLASSIF_KEEP_ALIVE", "").strip()
    if not v:
        try:
            with open(os.path.join(config_dir(), "keep_alive")) as fh:
                v = fh.readline().strip()
        except OSError:
            pass
    v = v or "30m"
    try:
        return int(v)
    except ValueError:
        return v


def chat(host, model, system, user, options, timeout=None, **extra):
    """One /api/chat call at temperature 0 with top_logprobs: the response with
    its ms, or {label: None, unscored, host[, overflow]}. extra goes into the
    request as is (tag's format)."""
    body = json.dumps({
        # think:false keeps a thinking model's position-0 token an answer, not
        # the start of a reasoning trace; non-thinking models accept it.
        # truncate:false makes an oversized input an error instead of a silent
        # cut (0.34.3 and 0.17.0 both honor it).
        "model": model, "stream": False, "think": False, "logprobs": True, "top_logprobs": 20,
        "truncate": False, "keep_alive": keep_alive(), "options": {"temperature": 0, **options},
        "messages": [{"role": "system", "content": system}, {"role": "user", "content": user}], **extra,
    }).encode()
    req = urllib.request.Request(f"http://{host}/api/chat", data=body,
                                 headers={"Content-Type": "application/json"})
    t0 = time.monotonic()
    try:
        with urllib.request.urlopen(req, timeout=timeout or float(os.environ.get("CLASSIF_TIMEOUT", "15"))) as r:
            resp = json.load(r)
    except urllib.error.HTTPError as e:
        detail = server_error(e.read().decode(errors="replace"))[:300]
        r = {"label": None, "unscored": f"{host} HTTP {e.code}: {detail}", "host": host}
        if e.code == 400 and "exceeds the available context size" in detail:
            # "request (193601 tokens) exceeds the available context size (32768 tokens)"
            r["overflow"] = True
        return r
    except (OSError, ValueError) as e:
        return {"label": None, "unscored": f"{host}: {e}", "host": host}
    if not isinstance(resp, dict):
        return {"label": None, "unscored": f"{host}: response is not a JSON object", "host": host}
    resp["ms"] = round((time.monotonic() - t0) * 1000)
    return resp


def judge(question, text, labels, options=None, model=None, host=None, num_ctx=NUM_CTX, timeout=None, context=None):
    """One scored call, for callers that score many items (classif): pick a
    host once, then judge each item on it. Without a host it routes like the
    CLI. num_ctx and timeout let a hot-path caller run a small window (llama
    at 2048 fits beside gemma4:12b on this laptop's GPU; at 32768 one evicts
    the other) inside its own budget. Returns {label, p, logp, mass, model,
    host, ms, T[, p_raw]}, keyed by the labels given, or {label: None,
    unscored: why[, host, mass]}."""
    if host is None:
        host, model = route(model)
        if not host:
            return {"label": None, "unscored": "no Ollama host answered: " + ",".join(h for h, _ in hosts())}
    model = model or dict(hosts()).get(host, DEFAULT_MODEL)
    system, user = prompt(question.strip() or DEFAULT_QUESTION, text, labels, options, context)
    resp = chat(host, model, system, user, {"num_predict": 1, "num_ctx": num_ctx}, timeout)
    if "unscored" in resp:
        return resp
    ms = resp.pop("ms")

    try:
        top = resp["logprobs"][0]["top_logprobs"]
        if not resp.get("done"):
            raise KeyError("done")
    except (KeyError, IndexError, TypeError):
        return {"label": None, "unscored": f"{host}: response has no logprobs (Ollama too old?)", "host": host}

    p, mass = score(top, labels)
    if mass < MASS_MIN:
        seen = ",".join(str(r.get("token", "")).strip() for r in top[:5])
        return {"label": None, "unscored": f"label mass {mass:.2f} < {MASS_MIN}; model wanted: {seen}",
                "host": host, "mass": round(mass, 3)}

    label = max(labels, key=lambda l: p[l])
    logp, t, p_raw = log_mass(top, labels), temperature(model, enum=options is not None), p
    res = {"label": label, "p": calibrate(logp, t) if t else p, "logp": logp, "mass": mass,
           "model": model, "host": host, "ms": ms, "T": t}
    if t:
        res["p_raw"] = p_raw
    return res


def tag_request(facets, text, context=None):
    """System, user and the JSON schema for several questions over one text.
    facets: [(name, [option, ...])]. The schema makes the grammar write each
    name as a key before its digit, which keeps the answers apart: with bare
    digits ("12") the second answer copied the first. On 55 emails, "needs a
    reply?" asked after "newsletter?" agreed with its own call 16 times
    bare and 51 times keyed. Each question numbers its options from 1; 0 is
    none of these, as with -e."""
    lines = [f"{name}: Options: " + ", ".join(f"{i}={o}" for i, o in enumerate(opts, 1)) + ", 0=none of these."
             for name, opts in facets]
    body = (f"Context:\n{context}\n\n" if context else "") + f"Text:\n{text}\n\n"
    schema = {"type": "object", "required": [n for n, _ in facets],
              "properties": {n: {"type": "integer", "enum": list(range(1, len(o) + 1)) + [0]} for n, o in facets}}
    # One line without spaces: pretty-printed, the model spent 28 generated
    # tokens on three questions instead of 15, at about 20 ms each.
    system = "Answer with a JSON object on one line, without spaces, giving each question's option number."
    return system, body + "\n".join(lines), schema


KEY_BEFORE = re.compile(r'"((?:[^"\\]|\\.)*)"\s*:\s*$')


def answers(steps):
    """Each key's answer position: {key: top_logprobs}. An answer is a
    one-digit token right after `"key":`, the key read back from the text
    before it, so neither the order the keys come in nor a digit inside a
    key misplaces one."""
    out, text = {}, ""
    for step in steps:
        tok = str(step.get("token", ""))
        key = KEY_BEFORE.search(text)
        if key and len(tok.strip()) == 1 and tok.strip().isdigit():
            out.setdefault(json.loads(f'"{key.group(1)}"'), step.get("top_logprobs") or [])
        text += tok
    return out


def tag(facets, text, model=None, host=None, timeout=None, context=None, num_ctx=NUM_CTX):
    """Several questions over one text in one call, the text read once.
    Returns {tags: {name: {label, p, mass} or {label: None, unscored}}, model,
    host, ms}, p keyed by option names plus none, or the failed call's
    {label: None, unscored[, overflow]}. p is raw: no temperature is fitted
    for an answer read this way."""
    if host is None:
        host, model = route(model)
        if not host:
            return {"label": None, "unscored": "no Ollama host answered: " + ",".join(h for h, _ in hosts())}
    model = model or dict(hosts()).get(host, DEFAULT_MODEL)
    system, user, schema = tag_request(facets, text, context)
    # The JSON's braces, quotes and keys are generated tokens too. A character
    # the vocabulary lacks is written a byte per token, or as a \u escape, so
    # a key costs at most its ASCII-escaped length, which is never shorter
    # than its UTF-8 bytes: twelve U+20000 took Winnow 57 tokens with kind.
    budget = 8 + sum(len(json.dumps(n)) + 6 for n, _ in facets)
    resp = chat(host, model, system, user, {"num_predict": budget, "num_ctx": num_ctx}, timeout, format=schema)
    if "unscored" in resp:
        return resp
    if not isinstance(resp.get("logprobs"), list) or not resp.get("done"):
        return {"label": None, "unscored": f"{host}: response has no logprobs (Ollama too old?)", "host": host}
    found = answers(resp["logprobs"])
    stopped, tokens = resp.get("done_reason"), resp.get("eval_count")
    missing = "no answer for it in the response"
    if stopped == "length":
        missing = f"the response stopped at its {tokens}-token budget before this answer"
    tags = {}
    for name, opts in facets:
        if name not in found:
            tags[name] = {"label": None, "unscored": missing}
            continue
        labels = [str(i) for i in range(1, len(opts) + 1)] + ["0"]
        p, mass = score(found[name], labels)
        if mass < MASS_MIN:
            tags[name] = {"label": None, "unscored": f"label mass {mass:.2f} < {MASS_MIN}", "mass": round(mass, 3)}
            continue
        p = {o: p[l] for l, o in zip(labels, opts + ["none"])}
        tags[name] = {"label": max(p, key=p.get), "p": p, "mass": mass}
    return {"tags": tags, "model": model, "host": host, "ms": resp["ms"], "done_reason": stopped, "tokens": tokens}


def embed(texts, model=EMBED_MODEL, host=None, timeout=None, query=False):
    """Unit vectors for the texts from an embedding model, one /api/embed
    call: {vectors, tokens}, or {vectors: None, unscored: why}, with missing
    True when the host has no such model, so a caller can say what to pull
    and read without it. query marks a question; a passage gets the other
    prefix."""
    host = host or LOCAL_HOST
    pre = EMBED_PREFIX.get(model.split(":")[0], ("", ""))[0 if query else 1]
    body = json.dumps({"model": model, "input": [pre + t for t in texts], "truncate": True,
                       "keep_alive": keep_alive()}).encode()
    req = urllib.request.Request(f"http://{host}/api/embed", data=body, headers={"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=timeout or float(os.environ.get("CLASSIF_TIMEOUT", "15")) * 4) as r:
            resp = json.load(r)
    except urllib.error.HTTPError as e:
        detail = server_error(e.read().decode(errors="replace"))[:300]
        return {"vectors": None, "unscored": f"{host} HTTP {e.code}: {detail}", "missing": e.code == 404}
    except (OSError, ValueError) as e:
        return {"vectors": None, "unscored": f"{host}: {e}"}
    vectors = resp.get("embeddings") if isinstance(resp, dict) else None
    if not isinstance(vectors, list) or len(vectors) != len(texts):
        return {"vectors": None, "unscored": f"{host}: response holds no embeddings for the {len(texts)} texts"}
    out = []
    for v in vectors:
        norm = math.sqrt(sum(x * x for x in v)) or 1.0
        out.append([x / norm for x in v])
    return {"vectors": out, "tokens": resp.get("prompt_eval_count", 0)}


def dump(obj):
    """-j output: indented on a terminal, one line in a pipe so a script can
    read one result per line."""
    print(json.dumps(obj, indent=2 if sys.stdout.isatty() else None))


def unscored(why, as_json, **extra):
    if as_json:
        dump({"label": None, "unscored": why, **extra})
    print(f"{PROG}: unscored: {why}", file=sys.stderr)
    return 2


def memory():
    """mem.py beside this script, as a module: the path for input past the window."""
    spec = importlib.util.spec_from_file_location(
        "classif_mem", os.path.join(os.path.dirname(os.path.realpath(__file__)), "mem.py"))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


class Help(argparse.HelpFormatter):
    """Wraps prose paragraphs to the terminal and keeps indented ones, the
    examples, as written."""

    def _fill_text(self, text, width, indent):
        return "\n\n".join(p if p.startswith(" ") else super(Help, self)._fill_text(p, width, indent)
                           for p in text.split("\n\n"))


EPILOG = """\
Output: LABEL P, the winner and its probability; -j for JSON. Exit: 0 when the first option wins (yes), 1 for another, 2 unscored, 3 insufficient or below -t.

Quote the question and a text argument. An -e description may go unquoted unless it holds ? * ' or #; its words run to the next option, so in scripts put the text before -e or pipe it in:

    git diff | classif "what kind of change is this?" -e fix=repairs broken behaviour -e feat=adds something new"""


def join_descriptions(argv):
    """The shell splits an unquoted -e description: -e fix=repairs a bug
    arrives as -e, fix=repairs, a, bug, and the loose words would become the
    text. Join the words after an unspaced name=... up to the next option or
    a value holding a space, which the caller quoted and so meant apart."""
    out, i = [], 0
    while i < len(argv):
        out.append(argv[i])
        i += 1
        if out[-1] in ("-e", "--enum") and i < len(argv) and "=" in argv[i] and not re.search(r"\s", argv[i]):
            words = [argv[i]]
            i += 1
            while i < len(argv) and argv[i] and not argv[i].startswith("-") and not re.search(r"\s", argv[i]):
                words.append(argv[i])
                i += 1
            out.append(" ".join(words))
    return out


def enum_options(values):
    """Every -e's options, each "name" or "name=description". A value with a
    newline holds one option per line, as a command prints them. Otherwise
    commas separate options, and a description runs to the next name=, so
    -e "moved=it moved, with a new address" is one option."""
    out = []
    for v in values:
        if "\n" in v:
            out += v.split("\n")
            continue
        first = len(out)
        for piece in v.split(","):
            if piece.strip() and len(out) > first and "=" in out[-1] and "=" not in piece:
                out[-1] += "," + piece
            else:
                out.append(piece)
    return [o for o in out if o.strip()]


def enum_labels(ap, values, most=9):
    """-e's options as (labels, names, options): the digits the model answers
    with, the names the output carries and the text the model reads, each
    ending in none. Past nine the labels are None: screen() narrows the
    options to FINALISTS before the model picks."""
    opts = [o.partition("=") for o in enum_options(values)]
    names = [n.strip() for n, _, _ in opts]
    # One name is enough: none is its alternative, so -e hook is a filter.
    if not 1 <= len(names) <= (most or len(names)) or not all(names) or len({n.lower() for n in names}) != len(names):
        ap.error(f"-e needs 1 to {most} distinct options, got {len(names)}" if most else
                 f"-e needs distinct named options, got {', '.join(names)}")
    # Digits are one token each, so any name scores; 10 and up may not be.
    labels = [str(i) for i in range(1, len(names) + 1)] + ["0"] if len(names) <= 9 else None
    # The model reads each description; the output carries only the name.
    options = [f"{n} ({d.strip()})" if d.strip() else n for n, (_, _, d) in zip(names, opts)]
    return labels, names + ["none"], options + ["none of these"]


def piped():
    """Whether stdin is a pipe, file or socket, empty or not. A script's
    empty pipe (git diff with no changes) is an empty input, never a reason
    to judge -c in its place; a terminal or /dev/null gives nothing."""
    try:
        mode = os.fstat(sys.stdin.fileno()).st_mode
    except (OSError, ValueError, AttributeError):
        return False
    return stat.S_ISFIFO(mode) or stat.S_ISREG(mode) or stat.S_ISSOCK(mode)


def read_context(ap, value):
    """-c's text, stripped, or None without -c: the file it names, <(cmd)
    included, else the value itself. A value that reads as a file name (one
    word holding a slash or ending in an extension) but names no file is a
    usage error, so a mistyped path never becomes the context. Empty is too."""
    if value is None:
        return None
    if os.path.exists(value) or (not re.search(r"\s", value) and re.search(r"/|\.\w{1,5}$", value)):
        try:
            with open(value, "rb") as fh:
                context = fh.read().decode("utf-8", errors="replace").strip()
        except OSError as e:
            ap.error(f"-c {value}: {e.strerror}. -c takes a file, <(cmd) or the text itself, and this reads as a "
                     "file name")
        if not context:
            ap.error(f"-c {value}: the file is empty")
        return context
    if not value.strip():
        ap.error("-c is empty")
    return value.strip()


def read_files(ap, groups):
    """-i's files as one text, and the range each occupies: each starts where
    the last ended, on its own line. A name with a comma in it is given with
    its own -i."""
    parts, files = [], []
    for path in [p for group in groups for p in (group.split(",") if not os.path.exists(group) else [group]) if p]:
        try:
            with open(path, "rb") as fh:
                part = fh.read().decode("utf-8", errors="replace")
        except OSError as e:
            ap.error(f"-i {path}: {e.strerror}")
        if parts and not parts[-1].endswith("\n"):
            parts[-1] += "\n"
        start = sum(len(p) for p in parts)
        files.append({"file": path, "start": start, "end": start + len(part)})
        parts.append(part)
    return "".join(parts), files


TAG_EPILOG = """\
Output: one line per question, NAME ANSWER P; -j for JSON. Exit: 0 when every question is answered, 2 when any is unscored, 3 when any answer is under -t.

    notmuch show --format=raw id:x | classif tag 'urgency=today,this week,no deadline' 'kind=asks me,fyi,newsletter'"""


def tag_main(argv, prog="classif tag"):
    """classif tag: several questions about one text, read once."""
    global PROG
    PROG = prog
    ap = argparse.ArgumentParser(
        prog=prog, formatter_class=Help, epilog=TAG_EPILOG,
        usage="cmd | %(prog)s [options] NAME=OPTIONS ...\n       %(prog)s [options] NAME=OPTIONS ... -i FILE",
        description="Ask several questions about one text in one call. The text is read once, so each question "
                    "after the first adds a few generated tokens, not another read. The model writes each "
                    "question's name before its answer, so the answers stay apart; p is raw, with no fitted "
                    "temperature.")
    ap.add_argument("facets", nargs="+", metavar="NAME=OPTIONS",
                    help="a question and its answers, urgency=today,this week,no deadline: the name is what the "
                         "model reads, a word or a whole question; 1 to 9 options, by commas or newlines, plus none")
    ap.add_argument("-i", "--input", dest="files", action="append", metavar="FILE",
                    help="read the text from FILE; repeat -i, or give a.log,b.log, to join several into one text")
    ap.add_argument("-c", "--context", metavar="TEXT",
                    help="background read first: a file, <(cmd) or the text itself; with no input, the text")
    ap.add_argument("-j", "--json", action="store_true", help="print the full result as JSON")
    ap.add_argument("-t", "--min-p", type=float, metavar="P",
                    help="mark an answer under this p unsure and exit 3")
    ap.add_argument("-d", "--deadline", type=float, metavar="SECONDS", help="bound the call")
    a = ap.parse_intermixed_args(argv)
    facets = []
    for f in a.facets:
        name, _, rest = f.partition("=")
        opts = [o.strip() for o in re.split(r"[,\n]", rest) if o.strip()]
        if not name.strip() or not opts:
            ap.error(f"{f!r}: want NAME=OPTION,OPTION...")
        if len(opts) > 9 or len({o.lower() for o in opts + ["none"]}) != len(opts) + 1:
            ap.error(f"{name.strip()}: want 1 to 9 distinct options other than none, got {', '.join(opts)}")
        facets.append((name.strip(), opts))
    if len({n.lower() for n, _ in facets}) != len(facets):
        ap.error("each question needs its own name")
    if a.min_p is not None and not 0 < a.min_p <= 1:
        ap.error(f"--min-p wants a p above 0 and at most 1, got {a.min_p}")
    if a.deadline is not None and a.deadline <= 0:
        ap.error(f"-d wants seconds above 0, got {a.deadline}")
    context = read_context(ap, a.context)
    if a.files is not None:
        raw = read_files(ap, a.files)[0]
    else:
        raw = "" if sys.stdin.isatty() else sys.stdin.buffer.read().decode("utf-8", errors="replace")
    if not raw.strip() and context and a.files is None and not piped():
        # Given no input at all, -c is what the questions are asked of.
        raw, context, a.context = context, None, None
    text = raw.strip()
    if not text:
        ap.error("no input: pipe the text in or pass -i FILE")
    wait = a.deadline
    if wait is None and "CLASSIF_TIMEOUT" not in os.environ:
        wait = 15 + (len(text) + len(context or "")) / READ_RATE
    r = tag(facets, text, timeout=wait, context=context)
    if "tags" not in r:
        why = r.pop("unscored") + (". tag reads the whole text in one call; past the window ask one question at "
                                   "a time" if r.get("overflow") else "")
        return unscored(why, a.json, **{k: v for k, v in r.items() if k != "label"})
    tags, code = r["tags"], 0
    for t in tags.values():
        if t["label"] is None:
            code = 2
            continue
        if a.min_p is not None:
            t["unsure"] = t["p"][t["label"]] < a.min_p
            code = code or (3 if t["unsure"] else 0)
        # Unrounded for programs: a p that prints as 1.00 can still be under -t 1.
        t["confidence"] = confidence(t["p"][t["label"]], len(t["p"]))
    if a.json:
        dump({**r, **({"context": a.context} if context else {})})
    else:
        w = max(len(n) for n in tags)
        lw = max(len(t["label"] or "") for t in tags.values())
        for n, t in tags.items():
            if t["label"] is None:
                print(f"{n:<{w}}  unscored")
            else:
                print(f"{n:<{w}}  {t['label']:<{lw}}  {t['p'][t['label']]:.2f}" + (" unsure" if t.get("unsure") else ""))
    for n, t in tags.items():
        if t["label"] is None:
            print(f"{PROG}: {n}: unscored: {t['unscored']}", file=sys.stderr)
    return code


RANK_EPILOG = """\
Output: one row per candidate, best first: the score, the answer with its p, the candidate; -j for JSON. Exit: 0 when something ranked, 2 when nothing could be scored. A candidate classif cannot score is left out with a note on stderr.

    classif rank "Is this the right next step?" -c goals.md "renew the passport" "reorganize the bookshelf"
    task export | jq -r '.[].description' | classif rank "What should happen to this task?" -c goals.md \\
        -e "prioritize=do it this week" -e "defer=worth doing, not now" -e "drop=serves no goal"
    classif rank "Does this offer pay above market?" -c market.md -i offers/*.md"""
MARK_LINES = 60     # one call per line, so longer texts and contexts are not marked
MARK_MIN = 0.005    # a smaller move is noise: identical runs differ by about 0.001
MARKS = 3


def candidates(ap, texts, files):
    """The candidates as [{name, text}]: each argument, then each -i file named
    by its file name; else each stdin line, plain text or JSON {"name", "text"}."""
    out = [{"name": t, "text": t} for t in texts]
    for path in files or []:
        try:
            with open(path, "rb") as fh:
                out.append({"name": os.path.basename(path), "text": fh.read().decode("utf-8", errors="replace")})
        except OSError as e:
            ap.error(f"-i {path}: {e.strerror}")
    if out or not piped():
        return out
    for line in sys.stdin.buffer.read().decode("utf-8", errors="replace").splitlines():
        try:
            obj = json.loads(line)
        except ValueError:
            obj = None
        if isinstance(obj, dict) and isinstance(obj.get("text"), str):
            out.append({"name": str(obj.get("name") or obj["text"]), "text": obj["text"]})
        elif line.strip():
            out.append({"name": line.strip(), "text": line.strip()})
    return out


def without_each_line(text, context):
    """(source, line, text, context) with one non-blank line left out: of the
    text when it has 2 to MARK_LINES, then of the context when it has 1 to
    MARK_LINES; None when neither may be marked."""
    lines, ctx = text.split("\n"), (context or "").split("\n")
    n, m = sum(1 for l in lines if l.strip()), sum(1 for l in ctx if l.strip())
    if not 1 < n <= MARK_LINES and not 0 < m <= MARK_LINES:
        return None
    out = []
    if 1 < n <= MARK_LINES:
        out += [("text", l, "\n".join(lines[:i] + lines[i + 1:]), context) for i, l in enumerate(lines) if l.strip()]
    if 0 < m <= MARK_LINES:
        out += [("context", l, text, "\n".join(ctx[:i] + ctx[i + 1:]).strip() or None)
                for i, l in enumerate(ctx) if l.strip()]
    return out


def short(s, n=60):
    s = " ".join(s.split())
    return s if len(s) <= n else s[:n - 1] + "…"


def rank_main(argv, prog="classif rank"):
    """classif rank: one question asked of each candidate, best first."""
    global PROG
    PROG = prog
    ap = argparse.ArgumentParser(
        prog=prog, formatter_class=Help, epilog=RANK_EPILOG,
        usage="%(prog)s [options] QUESTION CANDIDATE ...\n       cmd | %(prog)s [options] QUESTION\n"
              "       %(prog)s [options] QUESTION -i FILE ...",
        description="Ask one question of each candidate and rank them, best first: by p(yes), or with -e by p of "
                    "the first option. Each candidate is its own call, so any number rank; a model answers one "
                    "question about one text well and picks among many poorly.")
    ap.add_argument("question", metavar="QUESTION", help="a yes/no question, or with -e the question the options answer")
    ap.add_argument("texts", nargs="*", metavar="CANDIDATE", help="a candidate as text, one per argument")
    ap.add_argument("-e", "--enum", action="append", metavar="OPTION",
                    help="an answer to pick, as in classif -e: name or name=description, 1 to 9 plus none. The "
                         "first option is the one candidates rank by")
    ap.add_argument("-c", "--context", metavar="TEXT",
                    help="background read before every candidate, such as goals: a file, <(cmd) or the text itself. "
                         "With no candidates, it is the one judged")
    ap.add_argument("-i", "--input", dest="files", action="extend", nargs="+", metavar="FILE",
                    help="candidates, one per file, each named by its file name")
    ap.add_argument("-k", "--top", type=int, metavar="N", help="show only the top N")
    ap.add_argument("-w", "--why", action="store_true",
                    help="for the top pick, the lines of its text and of -c whose removal moves its score most. One "
                         "call per line, so a text or context over 60 lines is not marked")
    ap.add_argument("-j", "--json", action="store_true", help="print the ranking as JSON")
    a = ap.parse_intermixed_args(join_descriptions(argv))
    if a.top is not None and a.top < 1:
        ap.error(f"-k wants 1 or more, got {a.top}")
    labels, names, options = enum_labels(ap, a.enum) if a.enum else (DEFAULT_LABELS.split(","), None, None)
    show = dict(zip(labels, names or labels))
    context = read_context(ap, a.context)
    cands = candidates(ap, a.texts, a.files)
    if not cands and context and not piped():
        # As in classif: given no input, -c is what the question is asked of.
        name = os.path.basename(a.context) if os.path.isfile(a.context) else "context"
        cands, context = [{"name": name, "text": context}], None
    if not cands:
        ap.error("no candidates: pass them after the question, one per line on stdin, or -i FILE ...")

    host, model = route()
    if not host:
        return unscored("no Ollama host answered: " + ",".join(h for h, _ in hosts()), a.json)

    def ask(text, ctx):
        wait = None if "CLASSIF_TIMEOUT" in os.environ else 15 + (len(text) + len(ctx or "")) / READ_RATE
        return judge(a.question, text, labels, options, model=model, host=host, timeout=wait, context=ctx)

    ranked, missed = [], []
    for c in cands:
        r = ask(c["text"], context)
        if r["label"] is None:
            why = r["unscored"] + (". rank reads each candidate in one call, so it must fit the window"
                                   if r.get("overflow") else "")
            missed.append({"name": c["name"], "unscored": why})
            print(f"{PROG}: {short(c['name'])}: unscored: {why}", file=sys.stderr)
            continue
        ranked.append({"name": c["name"], "score": r["p"][labels[0]], "label": show[r["label"]],
                       "p": {show[l]: v for l, v in r["p"].items()}, "text": c["text"]})
    if not ranked:
        return 2
    # Stable: a tie keeps the order the candidates came in.
    ranked.sort(key=lambda row: -row["score"])
    top = ranked[0]
    if a.why:
        cut = without_each_line(top["text"], context)
        if cut is None:
            top["unmarked"] = f"its text has under 2 or over {MARK_LINES} lines and there is no -c of 1 to {MARK_LINES}"
        else:
            top["marks"] = []
            for source, line, text, ctx in cut:
                r = ask(text, ctx)
                if r["label"] is not None and abs(top["score"] - r["p"][labels[0]]) >= MARK_MIN:
                    top["marks"].append({"from": source, "line": line,
                                         "delta": round(top["score"] - r["p"][labels[0]], 4)})
            top["marks"] = sorted(top["marks"], key=lambda m: -abs(m["delta"]))[:MARKS]
    shown = ranked[:a.top] if a.top else ranked
    first = show[labels[0]]

    if a.json:
        rows = [{"name": row["name"], "score": round(row["score"], 4), "label": row["label"],
                 "p": {l: round(v, 4) for l, v in row["p"].items()},
                 **{k: row[k] for k in ("marks", "unmarked") if k in row}} for row in shown]
        dump({"question": a.question, **({"options": names[:-1]} if names else {}),
              **({"context": a.context} if context else {}), "ranking": rows,
              **({"unscored": missed} if missed else {}), "model": model, "host": host})
        return 0
    table = [[f"p({first})", "answer", "candidate"]] + [
        [f"{row['score']:.3f}", f"{row['label']} {row['p'][row['label']]:.3f}", short(row["name"])] for row in shown]
    widths = [max(len(row[i]) for row in table) for i in range(2)]
    for row in table:
        print(f"{row[0]:<{widths[0]}}  {row[1]:<{widths[1]}}  {row[2]}")
    if "unmarked" in top:
        print(f"\nwhy {short(top['name'])}: not marked, {top['unmarked']}")
    elif "marks" in top:
        print(f"\nwhy {short(top['name'])}: the lines whose removal moves p({first}) most")
        for m in top["marks"]:
            print(f"  {m['delta']:+.3f}  {'context: ' if m['from'] == 'context' else ''}{short(m['line'], 100)}")
        if not top["marks"]:
            print("  (no single line moves it)")
    return 0


def main(argv=None, prog="classif"):
    global PROG
    PROG = prog
    # INPUT comes one of three ways; the usage shows all three, so -i reads as filling that slot.
    ap = argparse.ArgumentParser(prog=prog, description="\n\n".join(__doc__.split("\n\n")[:2]), epilog=EPILOG,
                                 formatter_class=Help, usage="%(prog)s [options] QUESTION [INPUT]\n"
                                 "       cmd | %(prog)s [options] QUESTION\n"
                                 "       %(prog)s [options] QUESTION -i FILE")
    ap.add_argument("question", nargs="?", default="", metavar="QUESTION",
                    help=f'what to judge; omitted or "" means "{DEFAULT_QUESTION}"')
    ap.add_argument("input", nargs="?", metavar="INPUT",
                    help="what the question is about: a log, a page, a diff. One quoted argument, -i FILE, "
                         "or stdin")
    pick = ap.add_mutually_exclusive_group()
    pick.add_argument("-e", "--enum", action="append", metavar="OPTION",
                      help="an answer to pick: name, or name=description, which the model reads as when to pick it. "
                           "Repeat -e or list names with commas; plus none. Past 9, each is first asked alone and the "
                           "3 likeliest picked among. Without -e: yes, no, unknown")
    # The labels themselves, no none: the path the default question and the
    # specs score on. evals/eval.py and calibrate.py measure it through here.
    pick.add_argument("-l", "--labels", default=DEFAULT_LABELS, help=argparse.SUPPRESS)
    ap.add_argument("-j", "--json", action="store_true", help="print the full result as JSON")
    ap.add_argument("-p", "--pass", dest="gate", action="store_true",
                    help="gate: pass INPUT through when the first option wins")
    ap.add_argument("-t", "--min-p", type=float, metavar="P",
                    help="act only on an answer at least this sure, e.g. -t 0.8; a less sure one prints unsure and "
                         "exits 3, so && and -p do not fire")
    ap.add_argument("-i", "--input", dest="files", action="append", metavar="FILE",
                    help="read INPUT from FILE; repeat -i, or give a.log,b.log, to join several into one INPUT")
    ap.add_argument("-c", "--context", metavar="TEXT",
                    help="background the question needs, read first: a policy, a reference, the date. A file, "
                         "<(cmd) or the text itself; with no INPUT it is the text judged")
    ap.add_argument("-d", "--deadline", type=float, metavar="SECONDS",
                    help="bound all the work; a read cut short exits 3")
    ap.add_argument("-w", "--why", action="store_true",
                    help="print the lines the answer rests on; the text is read in pieces even when it fits, so "
                         "it is slower and the answer can differ from the one-call one")
    ap.add_argument("--cache", action="store_true",
                    help="save readings, links and the passage index under ~/.cache/classif, so a repeat on the same "
                         "text reuses them; nothing is written to disk without it")
    ap.add_argument("extra", nargs="*", help=argparse.SUPPRESS)
    # Intermixed: positionals may follow options, as in `classif "q?" -e a,b "text"`.
    given = sys.argv[1:] if argv is None else list(argv)
    joined = join_descriptions(given)
    a = ap.parse_intermixed_args(joined)
    claim = a.enum is None and a.labels == DEFAULT_LABELS
    if a.deadline is not None and a.deadline <= 0:
        ap.error(f"-d wants seconds above 0, got {a.deadline}")
    if a.files is not None and a.input is not None:
        ap.error("got an input argument and -i; pass one of them")
    if a.why and not claim and a.enum is None:
        ap.error("--why answers the default labels and -e; custom -l labels need one whole read")
    context = read_context(ap, a.context)
    if a.min_p is not None and not 0 < a.min_p <= 1:
        ap.error(f"--min-p wants a p above 0 and at most 1, got {a.min_p}")
    if a.extra:
        # A pasted text with its own double quotes reaches us already split by
        # the shell; nothing here can rejoin it faithfully.
        ap.error(f"got {2 + len(a.extra)} arguments, want at most 2 (question, text). The shell split a text "
                 "into words: quote it, $(cmd) as \"$(cmd)\" too, or send it on stdin: "
                 "xsel -ob | classif \"question\"")
    out = sys.stdout
    if a.gate:
        # stdout carries only the passed input; -j output goes to stderr, so
        # the next command in the pipe sees the data alone.
        sys.stdout = sys.stderr

    names = options = None
    if a.enum is not None:
        labels, names, options = enum_labels(ap, a.enum, most=None)
        if labels is None and a.why:
            ap.error(f"--why reads the text in pieces, which pick among at most 9 options; got {len(names) - 1}")
    else:
        labels = [l.strip() for l in a.labels.split(",") if l.strip()]
        if len(labels) < 2 or len({l.lower() for l in labels}) != len(labels):
            ap.error("need at least two distinct labels")
    first = (names or labels)[0]
    show = dict(zip(labels or [], names or labels or []))
    raw, files = a.input, []
    if a.files is not None:
        raw, files = read_files(ap, a.files)
    elif raw is None or raw == "-":
        if not sys.stdin.isatty():
            # head -c cuts mid-character and binaries are not UTF-8 at all;
            # neither should crash a loop over files.
            raw = sys.stdin.buffer.read().decode("utf-8", errors="replace")
        elif a.question.strip():
            # At a terminal nothing is piped in, so the question stands alone.
            raw = ""
        else:
            ap.error("no input: pass a question, a text, or both")
    text = raw.strip()
    if not text and len(joined) < len(given):
        # A one-word text after an unquoted description reads as its last word.
        ap.error("the words after an unquoted -e name= were read as its description, and no text is left. "
                 "Quote the description, or put a one-word text before -e")
    if not text and context and a.input in (None, "-") and a.files is None and not piped():
        # Given no INPUT at all, -c is what the question is asked of. An empty
        # pipe or argument stays an empty INPUT: judging -c in its place could
        # pass a policy down a -p gate.
        raw, text, context, a.context = context, context, None, None

    t0 = time.monotonic()
    if a.why:
        # The lines an answer rests on come from reading line by line, so a
        # short text goes to the reader too instead of one whole read.
        if not text:
            ap.error("--why points at lines of the INPUT, and there is none: pass it after the question, pipe it "
                     "in, use -i FILE or -c")
        host, model = route()
        if not host:
            return unscored("no Ollama host answered: " + ",".join(h for h, _ in hosts()), a.json)
        return long_input(a, raw, names, options, host, model, t0, out, context, files)
    wait = a.deadline
    if wait is None and "CLASSIF_TIMEOUT" not in os.environ:
        wait = 15 + (len(text) + len(context or "")) / READ_RATE
    r = yes = host = model = None
    if labels is None:
        host, model = route()
        if not host:
            return unscored("no Ollama host answered: " + ",".join(h for h, _ in hosts()), a.json)
        left = (lambda: wait) if a.deadline is None else (lambda: a.deadline - (time.monotonic() - t0))
        r, yes, names, options = screen(a.question, text, names, options, model, host, left, context)
        if r is None:
            labels = [str(i) for i in range(1, len(names))] + ["0"]
            show = dict(zip(labels, names))
        elif r.get("overflow"):
            return unscored(r["unscored"] + ". Past 9 options each is asked of the whole text, which must fit the "
                            "window", a.json, host=r["host"])
    if r is None:
        left = wait if a.deadline is None else a.deadline - (time.monotonic() - t0)
        r = judge(a.question, text, labels, options, model=model, host=host, timeout=left, context=context)
    if r["label"] is None and r.get("overflow"):
        if not text:
            # Only -c overflowed: past the window the input is read in pieces,
            # and with an empty input there is nothing to read, so no answer.
            return unscored(r["unscored"] + ". -c is context added to every call and must fit the window; "
                            "pass a long source as the input with -i", a.json, host=r["host"])
        if not claim and a.enum is None:
            return unscored(r["unscored"] + ". Past the window only the default labels and -e are answered; "
                            "custom -l labels need one whole read", a.json, host=r["host"])
        return long_input(a, raw, names, options, r["host"], dict(hosts()).get(r["host"], DEFAULT_MODEL),
                          t0, out, context, files)
    if r["label"] is None and a.deadline is not None and "timed out" in r["unscored"]:
        if a.json:
            dump({"label": None, "p": None, "mode": "direct", "verdict": "insufficient",
                 "why": "deadline passed during the whole read", "host": r["host"]})
        print(f"{PROG}: insufficient: deadline passed during the whole read", file=sys.stderr)
        return 3
    if r["label"] is None:
        return unscored(r.pop("unscored"), a.json, **{k: v for k, v in r.items() if k != "label"})
    label, p = r["label"], r["p"]
    # A winner under --min-p is neither a yes nor a no the caller should act
    # on: 0.49 and 0.51 would otherwise trigger opposite actions.
    unsure = a.min_p is not None and p[label] < a.min_p
    verdict = f"{show[label]} {p[label]:.2f}" + (" unsure" if unsure else "")
    if a.json:
        res = {"label": show[label], "p": {show[l]: round(v, 3) for l, v in p.items()},
               "confidence": round(confidence(p[label], len(labels)), 3),
               "T": r["T"], "logp": {show[l]: round(v, 3) for l, v in r["logp"].items()},
               "mass": round(r["mass"], 3), "model": r["model"], "host": r["host"], "ms": r["ms"], "mode": "direct"}
        if context:
            res["context"] = a.context
        if a.min_p is not None:
            res["unsure"] = unsure
        if r["T"]:
            res["p_raw"] = {show[l]: round(v, 3) for l, v in r["p_raw"].items()}
        if yes:
            res["screen"] = {n: round(v, 3) for n, v in yes.items()}
        dump(res)
    elif not a.gate:
        print(verdict)
    elif sys.stderr.isatty():
        # A silent gate at a prompt reads as broken: say what it decided, on
        # stderr so the data passing through stays clean. Scripts stay silent.
        sys.stderr.write(f"\033[2m{PROG}: {verdict}\033[0m\n")
    if unsure:
        return 3
    if show[label] != first:
        return 1
    if a.gate:
        out.write(raw)
    return 0


FINALISTS = 3   # options past nine narrow to these before the model picks


def screen(question, text, names, options, model, host, left, context):
    """Past nine -e options, whose digits stop being one token and among
    which a model picks poorly: ask of each option alone whether it is the
    answer, then keep the FINALISTS likeliest, in the order given, for one
    pick. Returns (None, {name: p(yes)}, names, options), the kept names and
    options ending in none, or (the failed call's result, ...). left() is the
    seconds the next call may take."""
    yes, q = {}, question.strip() or DEFAULT_QUESTION
    for n, o in zip(names[:-1], options[:-1]):
        t = left()
        if t is not None and t <= 0:
            return {"label": None, "unscored": f"{host}: timed out", "host": host}, None, None, None
        r = judge(f"{q} Is the answer {o}?", text, DEFAULT_LABELS.split(","), model=model, host=host, timeout=t,
                  context=context)
        if r["label"] is None:
            return r, None, None, None
        yes[n] = r["p"]["yes"]
    keep = sorted(sorted(range(len(names) - 1), key=lambda i: -yes[names[i]])[:FINALISTS])
    return None, yes, [names[i] for i in keep] + ["none"], [options[i] for i in keep] + ["none of these"]


class Progress:
    """How far a long read is, as one dim line on a terminal redrawn in
    place: `classif: read 640 of 804 passages, 9 s`. Drawn at most every
    quarter second, and always when done reaches total; clear() wipes it
    before the answer prints."""

    def __init__(self, out, t0, every=0.25):
        self.out, self.t0, self.every, self.last = out, t0, every, None

    def __call__(self, done, total, unit):
        now = time.monotonic()
        if done < total and self.last is not None and now - self.last < self.every:
            return
        self.last = now
        self.out.write(f"\r\033[K\033[2m{PROG}: read {done:,} of {total:,} {unit}, {now - self.t0:.0f} s\033[0m")
        self.out.flush()

    def clear(self):
        self.out.write("\r\033[K")
        self.out.flush()


def long_input(a, raw, names, options, host, model, t0, out, context=None, files=()):
    """Answer through mem.py and report in classif's own terms. A claim's
    verdict becomes a label: supported yes, contradicted no. Anything the
    pieces did not settle has no label: insufficient, exit 3, no gate passes."""
    mem = memory()
    left = None if a.deadline is None else a.deadline - (time.monotonic() - t0)
    if left is not None and left <= 0:
        r = {"verdict": "insufficient", "label": None, "p": None, "read": {"why": "deadline passed before reading"}}
    else:
        progress = Progress(sys.stderr, t0) if sys.stderr.isatty() else None
        try:
            r = mem.answer(types.SimpleNamespace(**globals()), raw, a.question, host, model,
                           names[:-1] if names else None, options[:-1] if options else None,
                           "auto", False, False, left, cache=a.cache, context=context, progress=progress,
                           evidence=a.why)
        finally:
            if progress:
                progress.clear()
    read, first = r["read"], names[0] if names else "yes"
    if ((read.get("search") or {}).get("index") or {}).get("missing"):
        print(f"{PROG}: ollama pull {EMBED_MODEL} lets a question read the passages closest to it first, "
              "seconds instead of minutes", file=sys.stderr)
    if files and read.get("evidence"):
        read["evidence"] = in_files(read["evidence"], raw, files)
    read["file"] = ",".join(f["file"] for f in files) if files else "-"
    if len(files) > 1:
        read["files"] = list(files)
    if names:
        label = r["label"] if r["verdict"] in ("answered", "none") else None
    else:
        label = {"supported": "yes", "contradicted": "no"}.get(r["verdict"])
    p = r["p"] if label and r["p"] else None
    unsure = bool(a.min_p is not None and p and p[label] < a.min_p)
    verdict = (label or r["verdict"]) + (f" {p[label]:.2f}" if p else "") + (" unsure" if unsure else "")
    if a.json:
        res = {"label": label, "p": {k: round(v, 3) for k, v in p.items()} if p else None,
               "confidence": round(confidence(p[label], len(p)), 3) if p else None,
               "mode": "memory", "verdict": r["verdict"], "read": read, "model": model, "host": host,
               "ms": round((time.monotonic() - t0) * 1000)}
        if context:
            res["context"] = a.context
        if r["verdict"] == "unscored":
            res["unscored"] = read["why"]
        if a.min_p is not None:
            res["unsure"] = unsure
        dump(res)
    elif not a.gate and r["verdict"] != "unscored":
        print(verdict)
    if r["verdict"] == "unscored":
        print(f"{PROG}: unscored: {read['why']}", file=sys.stderr)
        return 2
    if a.why and not a.json:
        # Under a gate stdout carries the text, so the lines go to stderr.
        print("\n".join(why_lines(read)), file=sys.stderr if a.gate else sys.stdout)
    if label is None or sys.stderr.isatty():
        # What the answer rests on, or why there is none. Scripts with an answer stay silent.
        line = mem.report(read) if "calls" in read else read["why"]
        dim = ("\033[2m", "\033[0m") if sys.stderr.isatty() else ("", "")
        sys.stderr.write(f"{dim[0]}{PROG}: {verdict}: {line}{dim[1]}\n")
    if label is None or unsure:
        return 3
    if label != first:
        return 1
    if a.gate:
        out.write(raw)
    return 0


def in_files(evidence, raw, files):
    """Evidence runs as lines of the files they came from: each run gets its
    file and that file's own line numbers, split where one file ends and the
    next begins. Each file starts on its own line of the joined text."""
    starts = [raw.count("\n", 0, f["start"]) + 1 for f in files]
    out = []
    for e in evidence:
        lines = e["text"].split("\n")
        for k, f in enumerate(files):
            lo, hi = starts[k], starts[k + 1] - 1 if k + 1 < len(files) else math.inf
            a, b = max(e["line"], lo), min(e["end"], hi)
            if a <= b:
                out.append({"file": f["file"], "line": a - lo + 1, "end": b - lo + 1,
                            "text": "\n".join(lines[a - e["line"]:b - e["line"] + 1])})
    return out


def why_lines(read):
    """--why's report: each run of lines the answer rests on as `  LINE: text`,
    or what it rests on when no line does."""
    if not read.get("evidence"):
        return ["  (answered from counted facts and a sample, not from lines)"
                if read.get("basis") == "sample" and "lines" not in read else "  (no line reads for or against it)"]
    out = []
    for e in read["evidence"]:
        at = str(e["line"]) if e["end"] == e["line"] else f"{e['line']}-{e['end']}"
        if e.get("file"):
            at = f"{e['file']}:{at}"
        first, *rest = e["text"].split("\n")
        out += [f"  {at}: {first}"] + [" " * (len(at) + 4) + l for l in rest]
    return out


def flush_all():
    """A reader that leaves early (`| head -1`) closes the pipe; the verdict
    is the exit code and stands without the rest of the output."""
    for stream in (sys.__stdout__, sys.stderr):
        try:
            stream.flush()
        except BrokenPipeError:
            pass
