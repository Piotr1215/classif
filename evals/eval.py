#!/usr/bin/env python3
"""Score a model as a semantic-if classifier through the real `classif` (plan #183).

    evals/eval.py MODEL [HOST]          # HOST defaults to localhost:11434

Cases: cases/synthetic.json and cases/authored144.jsonl (tracked; the second
is SemIf's, see cases/THIRD_PARTY.md, and runs through -e), plus two private
sets read from ~/.local/state/classif when present and never committed:
rag_set.json (RAG hits labeled by hand) and email_set.json (mail labeled from
notmuch replied/flagged/newsletter tags). Results land next to them as
results-<model>.jsonl.

Reports per task: accuracy, Brier score on the true label, confidently wrong
(p(true) < 0.1), label mass and latency.
p is what classif prints, tempered for a model in calibration.json; each row
also keeps logp, which evals/calibrate.py fits a temperature from.
"""
import json
import os
import statistics
import subprocess
import sys

HERE = os.path.dirname(os.path.realpath(__file__))
CLASS = os.path.join(HERE, "..", "classif")
STATE = os.path.expanduser("~/.local/state/classif")

RAG_Q = "Would this chunk help answer the query: {q}"
E_BULK = "Is this email a newsletter?"
E_ACTION = "Does this email need a reply or action from me?"


def load(path):
    with open(path) as fh:
        return json.load(fh)


def choice(path):
    """SemIf's authored decisions as -e cases. label indexes the row's own
    option order, which varies, and every option carries a description."""
    with open(path) as fh:
        for line in fh:
            r = json.loads(line)
            names = [o["id"] for o in r["options"]]
            enum = "\n".join(f"{o['id']}={o['description']}" for o in r["options"])
            yield "choice", r["question"], r["state"], names[r["label"]], names + ["none"], enum


def items():
    """(task, question, text, true label, labels, enum). enum is the -e
    option list for a case asked by name, None for one asked with -l."""
    for c in load(os.path.join(HERE, "cases", "synthetic.json")):
        yield "synthetic", c["q"], c["x"], c["y"], c["labels"], None
    yield from choice(os.path.join(HERE, "cases", "authored144.jsonl"))
    rag = os.path.join(STATE, "rag_set.json")
    if os.path.exists(rag):
        for r in load(rag):
            yield "rag", RAG_Q.format(q=r["q"]), r["x"], r["y"], ["yes", "no"], None
    emails = os.path.join(STATE, "email_set.json")
    if os.path.exists(emails):
        for e in load(emails):
            if e["src"] != "flagged":
                yield "email-bulk", E_BULK, e["x"], "yes" if e["y"] == "garbage" else "no", ["yes", "no"], None
            yield "email-action", E_ACTION, e["x"], "no" if e["y"] == "garbage" else "yes", ["yes", "no"], None


def score(task, q, x, y, labels, enum, env):
    ask = ["-e", enum] if enum else ["-l", ",".join(labels)]
    r = subprocess.run([CLASS, "-j", *ask, q, x], env=env, capture_output=True, text=True)
    try:
        d = json.loads(r.stdout)
    except ValueError:
        d = {"label": None, "unscored": r.stderr.strip()[-200:]}
    return {"task": task, "y": y, "labels": labels, "enum": bool(enum), **d}


def report(model, rows):
    print(f"\n=== {model}  ({len(rows)} items)")
    for task in ["synthetic", "choice", "rag", "email-bulk", "email-action", "ALL"]:
        rs = [r for r in rows if task == "ALL" or r["task"] == task]
        if not rs:
            continue
        sc = [r for r in rs if r.get("label")]
        pt = [r["p"][r["y"]] for r in sc]
        ms = sorted(r["ms"] for r in sc)
        line = (f"{task:13} acc={sum(r['label'] == r['y'] for r in sc)}/{len(rs)} "
                f"unscored={len(rs) - len(sc)} "
                f"brier={statistics.mean((1 - p) ** 2 for p in pt) if pt else 0:.3f} "
                f"mean_p_true={statistics.mean(pt) if pt else 0:.2f} "
                f"conf_wrong={sum(p < 0.1 for p in pt)} "
                f"min_mass={min((r['mass'] for r in sc), default=0):.2f} "
                f"ms_p50={ms[len(ms) // 2] if ms else 0} ms_p95={ms[int(len(ms) * .95)] if ms else 0}")
        print(line)


def main():
    if len(sys.argv) < 2:
        sys.exit(__doc__.strip().split("\n\n")[1])
    model = sys.argv[1]
    host = sys.argv[2] if len(sys.argv) > 2 else "localhost:11434"
    # The hosts entry names the model: which model runs is config, not a flag.
    env = dict(os.environ, CLASSIF_HOSTS=f"{host}={model}", CLASSIF_TIMEOUT="300")
    subprocess.run([CLASS, "warm?", "x"], env=env, capture_output=True)
    rows = [score(*it, env) for it in items()]
    os.makedirs(STATE, exist_ok=True)
    out = os.path.join(STATE, f"results-{model.replace('/', '_').replace(':', '_')}.jsonl")
    with open(out, "w") as fh:
        for r in rows:
            fh.write(json.dumps(r) + "\n")
    report(model, rows)


if __name__ == "__main__":
    main()
