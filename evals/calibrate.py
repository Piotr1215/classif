#!/usr/bin/env python3
"""Fit a model's temperatures so classif's p means what it says.

    evals/eval.py MODEL [HOST]           # first: score every case, keeping logp
    evals/calibrate.py MODEL             # fit T, cross-validate, report
    evals/calibrate.py MODEL --write     # also store T in calibration.json
    evals/jev.py MODEL                   # or: ask every case of Jev and classif
    evals/calibrate.py MODEL --jev       # fit each kind to Jev's distributions
    evals/calibrate.py MODEL --jev --write   # also store T_score in calibration.json

Reads the rows eval.py wrote for MODEL and fits -l questions and -e
questions apart, as T and T_enum: one temperature does not serve both. T
minimizes the negative log likelihood of the true labels under
softmax(logp / T), the function classif applies. Five folds report what a T
fit on four does to the fifth, against
T=1, the model's own log-odds: NLL, Brier on the true label, ECE over 10
bins of the top p. No label changes rank, so accuracy is the same both ways.

--jev reads the rows evals/jev.py wrote and fits the temperature that brings
classif's distributions closest to Jev's: the one minimizing the cross-entropy
of softmax(logp / T) under Jev's distribution, -e's none left out. Scores,
-e questions and -l questions are fitted apart, and five folds report held-out
mean total variation and cross-entropy at T=1, at the temperature classif
applies now, and at the fit. --write stores T_score only: T and T_enum stay
fitted to labels, and the report shows what a Jev fit would change.
"""
import datetime
import importlib.machinery
import importlib.util
import json
import math
import os
import statistics
import sys

HERE = os.path.dirname(os.path.realpath(__file__))
STATE = os.path.expanduser("~/.local/state/classif")
FOLDS = 5
# Log-spaced from 0.05 to 100: a grid finds the minimum without assuming the
# NLL curve is smooth, and 2001 points are 0.4% apart.
GRID = [0.05 * 2000 ** (i / 2000) for i in range(2001)]


def load_class():
    path = os.path.join(HERE, "..", "judge.py")
    loader = importlib.machinery.SourceFileLoader("class_cli", path)
    mod = importlib.util.module_from_spec(importlib.util.spec_from_loader("class_cli", loader))
    loader.exec_module(mod)
    return mod


CLS = load_class()


def nll(rows, t):
    return statistics.mean(-math.log(max(CLS.calibrate(r["logp"], t)[r["y"]], 1e-12)) for r in rows)


def fit(rows):
    return min(GRID, key=lambda t: nll(rows, t))


def metrics(pairs):
    """pairs: (p per label, true label). NLL, Brier on the true label, ECE."""
    bins = {}
    for p, y in pairs:
        conf = max(p.values())
        bins.setdefault(min(int(conf * 10), 9), []).append((conf, max(p, key=p.get) == y))
    mean = statistics.mean
    return {"nll": mean(-math.log(max(p[y], 1e-12)) for p, y in pairs),
            "brier": mean((1 - p[y]) ** 2 for p, y in pairs),
            "ece": sum(len(b) / len(pairs) * abs(mean(ok for _, ok in b) - mean(c for c, _ in b))
                       for b in bins.values()),
            "acc": mean(max(p, key=p.get) == y for p, y in pairs)}


def cross_validate(rows):
    """Out-of-fold predictions at T=1 and at the T fit on the other folds."""
    raw, fitted, ts = [], [], []
    for k in range(FOLDS):
        train = [r for i, r in enumerate(rows) if i % FOLDS != k]
        test = [r for i, r in enumerate(rows) if i % FOLDS == k]
        t = fit(train)
        ts.append(t)
        raw += [(CLS.calibrate(r["logp"], 1.0), r["y"]) for r in test]
        fitted += [(CLS.calibrate(r["logp"], t), r["y"]) for r in test]
    return metrics(raw), metrics(fitted), ts


def fits(rows):
    """(key, rows, T) for each kind of question with rows enough to fold:
    T over the -l rows, T_enum over the -e rows."""
    for key, enum in (("T", False), ("T_enum", True)):
        kind = [r for r in rows if bool(r.get("enum")) == enum]
        if len(kind) >= 2 * FOLDS:
            yield key, kind, fit(kind)


def at(row, t):
    """classif's distribution over the answers Jev gave at temperature t: the
    raw log mass tempered over Jev's keys, so -e's none is left out."""
    return CLS.calibrate({k: row["logp"][k] for k in row["jev"]}, t)


def tv(p, q):
    return sum(abs(p[k] - q[k]) for k in p) / 2


def cross_entropy(rows, t):
    return statistics.mean(-sum(v * math.log(max(at(r, t)[k], 1e-12)) for k, v in r["jev"].items()) for r in rows)


def fit_jev(rows):
    return min(GRID, key=lambda t: cross_entropy(rows, t))


def jev_fits(rows):
    """(key, rows, T) per kind with rows enough to fold: T_score over scores,
    T_enum over -e rows (their logp holds none), T over the rest (-l)."""
    kinds = {"T_score": [], "T_enum": [], "T": []}
    for r in rows:
        kinds["T_score" if r["kind"] == "score" else "T_enum" if "none" in r["logp"] else "T"].append(r)
    for key, kind in kinds.items():
        if len(kind) >= 2 * FOLDS:
            yield key, kind, fit_jev(kind)


def jev_cross_validate(rows, now):
    """Held-out mean TV and cross-entropy at T=1, at now and at the fit."""
    held = {"T=1": [], "now": [], "fitted": []}
    for k in range(FOLDS):
        train = [r for i, r in enumerate(rows) if i % FOLDS != k]
        test = [r for i, r in enumerate(rows) if i % FOLDS == k]
        t = fit_jev(train)
        for name, temp in (("T=1", 1.0), ("now", now or 1.0), ("fitted", t)):
            held[name] += [(tv(at(r, temp), r["jev"]), cross_entropy([r], temp)) for r in test]
    return {name: (statistics.mean(a for a, _ in v), statistics.mean(b for _, b in v)) for name, v in held.items()}


def write(path, model, t, n, enum=False, kind=None):
    """Set one model's temperature for one kind of question: "" for -l,
    "_enum" for -e, "_score" for -s. Its other kinds and every other model
    stay as they were."""
    try:
        with open(path) as fh:
            table = json.load(fh)
    except (OSError, ValueError):
        table = {}
    kind = kind if kind is not None else "_enum" if enum else ""
    table[model] = {**table.get(model, {}), "T" + kind: round(t, 3), "n" + kind: n,
                    "fitted": datetime.date.today().isoformat()}
    with open(path, "w") as fh:
        json.dump(table, fh, indent=1, sort_keys=True)
        fh.write("\n")


def jev_main(model):
    src = os.path.join(STATE, f"jev-{model.replace('/', '_').replace(':', '_')}.jsonl")
    try:
        with open(src) as fh:
            rows = [json.loads(line) for line in fh]
    except OSError:
        sys.exit(f"no Jev rows at {src}: run evals/jev.py {model} first")
    rows = [r for r in rows if "logp" in r and "jev" in r]
    for key, kind, t in jev_fits(rows):
        now = CLS.temperature(model, enum=key == "T_enum", scale=key == "T_score")
        held = jev_cross_validate(kind, now)
        print(f"{model} {key}  n={len(kind)}  fit on all rows {t:.2f}, classif applies {now or 1.0:.2f} now")
        print(f"{'held out':10} {'tv':>6} {'xent':>6}")
        for name, (a, b) in held.items():
            print(f"{name:10} {a:6.3f} {b:6.3f}")
        if "--write" in sys.argv and key == "T_score":
            write(CLS.CALIBRATION, model, t, len(kind), kind="_score")
            print(f"wrote T_score={t:.3f} for {model} to {os.path.realpath(CLS.CALIBRATION)}")


def main():
    if "--jev" in sys.argv:
        args = [a for a in sys.argv[1:] if a not in ("--write", "--jev")]
        if len(args) != 1:
            sys.exit(__doc__.strip().split("\n\n")[1])
        return jev_main(args[0])
    args = [a for a in sys.argv[1:] if a != "--write"]
    if len(args) != 1:
        sys.exit(__doc__.strip().split("\n\n")[1])
    model = args[0]
    src = os.path.join(STATE, f"results-{model.replace('/', '_').replace(':', '_')}.jsonl")
    try:
        with open(src) as fh:
            rows = [json.loads(line) for line in fh]
    except OSError:
        sys.exit(f"no results at {src}: run evals/eval.py {model} first")
    rows = [r for r in rows if r.get("label") and "logp" in r]
    found = list(fits(rows))
    if not found:
        sys.exit(f"{len(rows)} rows with logp in {src}: rerun evals/eval.py {model}")

    for key, kind, t in found:
        raw, fitted, ts = cross_validate(kind)
        tasks = {}
        for r in kind:
            tasks[r["task"]] = tasks.get(r["task"], 0) + 1
        flag = "-e" if key == "T_enum" else "-l"
        print(f"{model} {flag}  n={len(kind)} ({', '.join(f'{k} {v}' for k, v in tasks.items())})")
        print(f"{key}={t:.2f} on all rows; per fold {min(ts):.2f} to {max(ts):.2f}")
        print(f"{'held out':10} {'nll':>6} {'brier':>6} {'ece':>6} {'acc':>5}")
        for name, m in (("T=1", raw), ("fitted", fitted)):
            print(f"{name:10} {m['nll']:6.3f} {m['brier']:6.3f} {m['ece']:6.3f} {m['acc']:5.2f}")
        if "--write" in sys.argv:
            write(CLS.CALIBRATION, model, t, len(kind), enum=key == "T_enum")
            print(f"wrote {key}={t:.3f} for {model} to {os.path.realpath(CLS.CALIBRATION)}")


if __name__ == "__main__":
    main()
