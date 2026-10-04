#!/usr/bin/env python3
"""Fit a model's temperatures so classif's p means what it says.

    evals/eval.py MODEL [HOST]           # first: score every case, keeping logp
    evals/calibrate.py MODEL             # fit T, cross-validate, report
    evals/calibrate.py MODEL --write     # also store T in calibration.json

Reads the rows eval.py wrote for MODEL and fits -l questions and -e
questions apart, as T and T_enum: one temperature does not serve both. T
minimizes the negative log likelihood of the true labels under
softmax(logp / T), the function classif applies. Five folds report what a T
fit on four does to the fifth, against
T=1, the model's own log-odds: NLL, Brier on the true label, ECE over 10
bins of the top p. No label changes rank, so accuracy is the same both ways.
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


def write(path, model, t, n, enum=False):
    """Set one model's temperature for one kind of question; its other kind
    and every other model stay as they were."""
    try:
        with open(path) as fh:
            table = json.load(fh)
    except (OSError, ValueError):
        table = {}
    kind = "_enum" if enum else ""
    table[model] = {**table.get(model, {}), "T" + kind: round(t, 3), "n" + kind: n,
                    "fitted": datetime.date.today().isoformat()}
    with open(path, "w") as fh:
        json.dump(table, fh, indent=1, sort_keys=True)
        fh.write("\n")


def main():
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
