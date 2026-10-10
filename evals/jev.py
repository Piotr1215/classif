#!/usr/bin/env python3
"""Ask Jev and classif the same cases and measure how far apart they answer.

    evals/jev.py MODEL [HOST]          # TYPESAFE_API_KEY set; HOST defaults to localhost:11434

Jev (TypeSafe's jev-latest) is the reference: classif borrows its primitives,
so the closer classif's distributions sit to Jev's, the more its answers can
be trusted, and a prompt, quantization or fine-tune can be judged by whether
it moves them closer. Each case goes to Jev as its primitive and to the real
`classif` with MODEL: the synthetic yes/no cases as a Noul (-l yes,no), the
other synthetic cases as a Choice (-l), SemIf's authored cases as a Choice
(-e, classif's none dropped and the options renormalized), and the license
cases as a Score (-s) when /usr/share/common-licenses exists.

Per case: the total variation between the two distributions (half the summed
absolute gap: 0 the same, 1 disjoint) and whether the top answer agrees. A
score also reports its gap from Jev's and whether that gap sits within Jev's
own spread, the standard deviation of its level distribution, at least FLOOR
of a level. Jev's answers are cached under ~/.local/state/classif/jev keyed
by request, so a rerun calls Jev only for a new or changed case; they stay
out of git. Rows land in ~/.local/state/classif/jev-<model>.jsonl.
"""
import hashlib
import json
import math
import os
import statistics
import subprocess
import sys
import time
import urllib.request

HERE = os.path.dirname(os.path.realpath(__file__))
CLASS = os.path.join(HERE, "..", "classif")
STATE = os.path.expanduser("~/.local/state/classif")
JEV_URL = "https://api.typesafe.ai/v1/systemone"
JEV_MODEL = "jev-latest"
FLOOR = 0.1     # a score gap this small is within, however sharp Jev's answer
LICENSES = "/usr/share/common-licenses"
LICENSE_Q = "How restrictive is this license for third-party code shipped inside a closed-source paid product?"
LICENSE_LEVELS = [
    "Permissive: use, change and sell freely, nothing to keep or share",
    "Notice: keep the copyright notice or license text with copies, nothing else",
    "File-level copyleft: changes to the licensed files must be shared under the same license, our own files stay closed",
    "Library copyleft: closed code may link to it, but changes to the library must be released under the same license",
    ("Strong copyleft: any program that includes or derives from it must be released whole under the same license "
     "with source"),
    "Network copyleft: strong copyleft, and users who reach it over a network can demand the source"]
LICENSE_FILES = ["BSD", "Apache-2.0", "Artistic", "CC0-1.0", "MPL-2.0", "LGPL-2.1", "LGPL-3", "GPL-2", "GPL-3",
                 "GFDL-1.3"]


def cases():
    """Each case as {task, kind, question, text or path, options or levels[, criteria]}."""
    with open(os.path.join(HERE, "cases", "synthetic.json")) as fh:
        for c in json.load(fh):
            if c["labels"] == ["yes", "no"]:
                yield {"task": "synthetic", "kind": "noul", "question": c["q"], "text": c["x"], "options": c["labels"]}
            else:
                yield {"task": "synthetic", "kind": "choice", "question": c["q"], "text": c["x"],
                       "options": c["labels"], "criteria": dict.fromkeys(c["labels"])}
    with open(os.path.join(HERE, "cases", "authored144.jsonl")) as fh:
        for line in fh:
            r = json.loads(line)
            yield {"task": "choice", "kind": "choice", "question": r["question"], "text": r["state"],
                   "options": [o["id"] for o in r["options"]],
                   "criteria": {o["id"]: o["description"] for o in r["options"]}}
    for name in LICENSE_FILES:
        path = os.path.join(LICENSES, name)
        if os.path.exists(path):
            yield {"task": "license", "kind": "score", "question": LICENSE_Q, "path": path, "levels": LICENSE_LEVELS}
    mit = os.path.join(HERE, "..", "LICENSE")
    if os.path.isdir(LICENSES):
        yield {"task": "license", "kind": "score", "question": LICENSE_Q, "path": mit, "levels": LICENSE_LEVELS}


def text_of(case):
    if "text" in case:
        return case["text"]
    with open(case["path"], errors="replace") as fh:
        return fh.read()


def jev_request(case, text):
    """The case as one Jev question under the id q."""
    q = {"type": case["kind"], "instructions": case["question"]}
    if case["kind"] == "choice":
        q["criteria"] = case["criteria"]
    elif case["kind"] == "score":
        q["criteria"] = case["levels"]
    return {"state": text, "model": JEV_MODEL, "questions": {"q": q}}


def post(req):
    """One request to Jev's API, the key from TYPESAFE_API_KEY."""
    body = json.dumps(req).encode()
    http = urllib.request.Request(JEV_URL, data=body, headers={
        "Authorization": "Bearer " + os.environ["TYPESAFE_API_KEY"], "Content-Type": "application/json"})
    t0 = time.monotonic()
    with urllib.request.urlopen(http, timeout=120) as r:
        resp = json.load(r)
    resp["ms"] = round((time.monotonic() - t0) * 1000)
    return resp


def jev_answer(req, cache, ask=post):
    """Jev's answer to req, from cache when the same request was asked before."""
    key = hashlib.sha256(json.dumps(req, sort_keys=True).encode()).hexdigest()
    path = os.path.join(cache, key + ".json")
    if os.path.exists(path):
        with open(path) as fh:
            resp = json.load(fh)
    else:
        resp = ask(req)
        os.makedirs(cache, exist_ok=True)
        with open(path, "w") as fh:
            json.dump(resp, fh)
    return {**resp["answers"]["q"], "model": resp.get("model"), "ms": resp.get("ms")}


def classif_args(case):
    """The classif arguments that ask the case as Jev's primitive asks it."""
    if case["kind"] == "score":
        return [x for level in case["levels"] for x in ("-s", level)] + [case["question"], "-i", case["path"]]
    if case["kind"] == "noul" or not any(case["criteria"].values()):
        return ["-l", ",".join(case["options"]), case["question"], case["text"]]
    enum = "\n".join(f"{o}={d}" for o, d in case["criteria"].items())
    return ["-e", enum, case["question"], case["text"]]


def jev_dist(answer, case):
    if case["kind"] == "noul":
        return {"yes": answer["noul"], "no": 1 - answer["noul"]}
    return answer["probabilities"]


def classif_dist(out, case):
    """classif's distribution over the case's own answers: a score's levels,
    or the options with -e's none dropped and the rest renormalized."""
    if case["kind"] == "score":
        return out["probabilities"]
    p = {o: out["p"][o] for o in case["options"]}
    z = sum(p.values())
    return {o: v / z for o, v in p.items()}


def tv(p, q):
    """Total variation: half the summed absolute gap, 0 to 1."""
    return sum(abs(p.get(k, 0.0) - q.get(k, 0.0)) for k in set(p) | set(q)) / 2


def level_sd(p):
    """The standard deviation of a level distribution keyed "0" to "n-1"."""
    mean = sum(int(k) * v for k, v in p.items())
    return math.sqrt(sum(v * (int(k) - mean) ** 2 for k, v in p.items()))


def compare(case, jev, ours):
    """How far classif's answer sits from Jev's on one case."""
    jd, cd = jev_dist(jev, case), classif_dist(ours, case)
    row = {"tv": tv(jd, cd), "agree": max(jd, key=jd.get) == max(cd, key=cd.get)}
    if case["kind"] == "score":
        gap, sd = abs(jev["score"] - ours["score"]), level_sd(jd)
        row.update({"gap": gap, "sd": sd, "within": gap <= max(sd, FLOOR)})
    return row


def quantile(xs, q):
    xs = sorted(xs)
    return xs[min(len(xs) - 1, int(len(xs) * q))] if xs else 0.0


def report(model, rows):
    print(f"\n=== {model} against Jev  ({len(rows)} cases)")
    for task in ["synthetic", "choice", "license", "ALL"]:
        rs = [r for r in rows if (task == "ALL" or r["task"] == task) and "tv" in r]
        missed = sum(1 for r in rows if (task == "ALL" or r["task"] == task) and "tv" not in r)
        if not rs and not missed:
            continue
        tvs = [r["tv"] for r in rs]
        line = (f"{task:10} n={len(rs)} unscored={missed} agree={sum(r['agree'] for r in rs)}/{len(rs)} "
                f"tv_mean={statistics.mean(tvs) if tvs else 0:.3f} tv_p50={quantile(tvs, .5):.3f} "
                f"tv_p90={quantile(tvs, .9):.3f} tv<=0.1={sum(t <= 0.1 for t in tvs)}/{len(tvs)}")
        scores = [r for r in rs if "gap" in r]
        if scores:
            line += (f" gap_mean={statistics.mean(r['gap'] for r in scores):.3f} "
                     f"within_sd={sum(r['within'] for r in scores)}/{len(scores)}")
        print(line)


def main():
    if len(sys.argv) < 2:
        sys.exit(__doc__.strip().split("\n\n")[1])
    if not os.environ.get("TYPESAFE_API_KEY"):
        sys.exit("evals/jev.py: set TYPESAFE_API_KEY to ask Jev")
    model = sys.argv[1]
    host = sys.argv[2] if len(sys.argv) > 2 else "localhost:11434"
    env = dict(os.environ, CLASSIF_HOSTS=f"{host}={model}", CLASSIF_TIMEOUT="300")
    subprocess.run([CLASS, "warm?", "x"], env=env, capture_output=True, check=False)
    rows = []
    for case in cases():
        jev = jev_answer(jev_request(case, text_of(case)), os.path.join(STATE, "jev"))
        r = subprocess.run([CLASS, "-j", *classif_args(case)], env=env, capture_output=True, text=True, check=False)
        try:
            ours = json.loads(r.stdout)
        except ValueError:
            ours = {"label": None, "unscored": r.stderr.strip()[-200:]}
        name = os.path.basename(case.get("path", "")) or case["question"][:60]
        row = {"task": case["task"], "kind": case["kind"], "case": name, "jev": jev_dist(jev, case),
               "jev_model": jev["model"], "jev_ms": jev["ms"], "ms": ours.get("ms")}
        if "unscored" not in ours:
            row.update({"classif": classif_dist(ours, case), **compare(case, jev, ours)})
        else:
            row["unscored"] = ours["unscored"]
        rows.append(row)
    os.makedirs(STATE, exist_ok=True)
    out = os.path.join(STATE, f"jev-{model.replace('/', '_').replace(':', '_')}.jsonl")
    with open(out, "w") as fh:
        for row in rows:
            fh.write(json.dumps(row) + "\n")
    report(model, rows)
    print(f"rows: {out}")


if __name__ == "__main__":
    main()
