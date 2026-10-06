#!/usr/bin/env python3
"""Score `classif tag` against one `classif -e` call per question.

    evals/tag_eval.py MODEL [HOST] [--pace SECONDS]

Each text asks two labeled questions through tag, in both orders, and asks
each question alone through -e with the same wording and option names, so
none is on offer to both. Cases: the authored144.jsonl states that carry
two questions (tracked), and email_set.json from ~/.local/state/classif
when present, its two questions once as whole questions and once as short
names. --pace sleeps between texts (default 0.5 s) to keep a laptop GPU off
its limit.

Reports per set and answer position: accuracy, agreement with the single
call, Brier on the true option, ECE on the winner's p, and model ms per text
for tag against the sum of its single calls. p is what each command prints:
raw for tag, tempered for -e on a model in calibration.json. Rows land in
~/.local/state/classif/tag-results-<model>.jsonl.
"""
import json
import os
import statistics
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.realpath(__file__))
CLASS = os.path.join(HERE, "..", "classif")
STATE = os.path.expanduser("~/.local/state/classif")
EMAIL_QS = {"whole": ("Is this email a newsletter?", "Does this email need a reply or action from me?"),
            "short": ("newsletter", "needs reply or action")}


def pair_cases(path):
    """(set, text, [(question, options, truth)]) for each state asked two
    questions. truth indexes the row's own option order, which varies."""
    by = {}
    with open(path) as fh:
        for line in fh:
            r = json.loads(line)
            names = [o["id"] for o in r["options"]]
            by.setdefault(r["state"], []).append((r["question"], names, names[r["label"]]))
    for text, qs in by.items():
        if len(qs) == 2:
            yield "pairs", text, qs


def email_cases(path):
    """Each mail twice, whole questions and short names. A flagged mail has
    no newsletter label, so it is left out."""
    with open(path) as fh:
        mails = json.load(fh)
    for m in mails:
        if m["src"] == "flagged":
            continue
        bulk, action = ("yes", "no") if m["y"] == "garbage" else ("no", "yes")
        for form, (qb, qa) in EMAIL_QS.items():
            yield f"email-{form}", m["x"], [(qb, ["yes", "no"], bulk), (qa, ["yes", "no"], action)]


def cases():
    yield from pair_cases(os.path.join(HERE, "cases", "authored144.jsonl"))
    emails = os.path.join(STATE, "email_set.json")
    if os.path.exists(emails):
        yield from email_cases(emails)


def call(cmd, text, env):
    r = subprocess.run(cmd, input=text, env=env, capture_output=True, text=True)
    try:
        return json.loads(r.stdout)
    except ValueError:
        return {"label": None, "unscored": r.stderr.strip()[-200:]}


def tag(qs, text, env):
    return call([CLASS, "tag", "-j", *[f"{q}={','.join(opts)}" for q, opts, _ in qs]], text, env)


def single(q, opts, text, env):
    return call([CLASS, "-j", "-e", ",".join(opts), q], text, env)


def rows_for(name, text, qs, env):
    alone = [single(q, opts, text, env) for q, opts, _ in qs]
    out = []
    for order in ([0, 1], [1, 0]):
        t = tag([qs[i] for i in order], text, env)
        for pos, i in enumerate(order, 1):
            q, _, truth = qs[i]
            got = (t.get("tags") or {}).get(q) or {"label": None}
            out.append({"set": name, "pos": pos, "q": q, "truth": truth,
                        "tag": {k: got.get(k) for k in ("label", "p", "unscored")},
                        "single": {k: alone[i].get(k) for k in ("label", "p", "unscored")},
                        "tag_ms": t.get("ms"), "single_ms": sum(a.get("ms") or 0 for a in alone)})
    return out


def ece(pairs, bins=10):
    """Expected calibration error over (winner p, right) pairs."""
    if not pairs:
        return 0.0
    total = 0.0
    for b in range(bins):
        inb = [(p, ok) for p, ok in pairs if min(int(p * bins), bins - 1) == b]
        if inb:
            total += len(inb) * abs(statistics.mean(ok for _, ok in inb) - statistics.mean(p for p, _ in inb))
    return total / len(pairs)


def measure(rows, side):
    scored = [r for r in rows if r[side]["label"]]
    right = [r[side]["label"] == r["truth"] for r in scored]
    return {"n": len(rows), "unscored": len(rows) - len(scored), "right": sum(right),
            "brier": statistics.mean((1 - r[side]["p"].get(r["truth"], 0.0)) ** 2 for r in scored) if scored else 0.0,
            "ece": ece([(r[side]["p"][r[side]["label"]], ok) for r, ok in zip(scored, right)])}


def report(model, rows):
    print(f"\n=== {model}  tag against one -e call per question")
    print(f"{'set':13} {'pos':>3} {'n':>4} {'unsc':>4} {'tag':>4} {'one':>4} {'agree':>5} "
          f"{'brier t/1':>11} {'ece t/1':>11} {'ms t/1 p50':>12}")
    for s in sorted({r["set"] for r in rows}):
        for pos in (1, 2, "all"):
            rs = [r for r in rows if r["set"] == s and (pos == "all" or r["pos"] == pos)]
            t, o = measure(rs, "tag"), measure(rs, "single")
            agree = sum(r["tag"]["label"] == r["single"]["label"] for r in rs if r["tag"]["label"])
            # One timing per tag call: the first position's row carries it.
            ms = [(r["tag_ms"], r["single_ms"]) for r in rs if r["pos"] == 1 and r["tag_ms"]]
            p50 = (f"{statistics.median(a for a, _ in ms):.0f}/{statistics.median(b for _, b in ms):.0f}"
                   if ms and pos != 2 else "")
            print(f"{s:13} {pos!s:>3} {t['n']:>4} {t['unscored']:>4} {t['right']:>4} {o['right']:>4} {agree:>5} "
                  f"{t['brier']:.3f}/{o['brier']:.3f} {t['ece']:.3f}/{o['ece']:.3f} {p50:>12}")


def main():
    args = sys.argv[1:]
    pace = 0.5
    if "--pace" in args:
        i = args.index("--pace")
        pace = float(args[i + 1])
        del args[i:i + 2]
    if not args:
        sys.exit(__doc__.strip().split("\n\n")[1])
    model = args[0]
    host = args[1] if len(args) > 1 else "localhost:11434"
    env = dict(os.environ, CLASSIF_HOSTS=f"{host}={model}", CLASSIF_TIMEOUT="300")
    subprocess.run([CLASS, "warm?", "x"], env=env, capture_output=True)
    rows = []
    for name, text, qs in cases():
        rows += rows_for(name, text, qs, env)
        time.sleep(pace)
    os.makedirs(STATE, exist_ok=True)
    out = os.path.join(STATE, f"tag-results-{model.replace('/', '_').replace(':', '_')}.jsonl")
    with open(out, "w") as fh:
        for r in rows:
            fh.write(json.dumps(r) + "\n")
    report(model, rows)


if __name__ == "__main__":
    main()
