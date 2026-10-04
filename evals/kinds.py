#!/usr/bin/env python3
"""The root call's routing: what each question about a long text needs, as
the model picks it from the question alone, against the expected kind.

    evals/kinds.py [CLASSIF_DIR]    # one line per case, then the tally

A whole-text pick on a question one line settles is the costly mistake: the
sample can miss the one line and answer a confident no. Those are marked.
CLASSIF_DIR defaults to the checkout beside this directory.
"""

import os
import sys
import types

HERE = os.path.dirname(os.path.realpath(__file__))

# (expected kind, question). exists: one line settles it; all: every case
# must hold; state: a current or latest state, an order or a count; whole:
# about the text as a whole.
CASES = [
    ("exists", "Does this log ever show a disk error?"),
    ("exists", "Is there any mention of a telephone?"),
    ("exists", "At least one line refers to Elizabeth Bennet."),
    ("exists", "Did the deployment fail at some point?"),
    ("exists", "Does the text mention Kubernetes anywhere?"),
    ("exists", "rsync can skip files larger than a given size."),
    ("exists", "The document describes the docker run command."),
    ("exists", "Has anyone complained about the price?"),
    ("exists", "Is there a line with a stack trace?"),
    ("exists", "Invoice INV-0100 has been paid."),
    ("all", "Every invoice is paid."),
    ("all", "Did all the tests pass?"),
    ("all", "No request took longer than a second."),
    ("all", "Are all hosts healthy in this output?"),
    ("all", "Every shell variable the document lists is read-only."),
    ("all", "None of the pods restarted."),
    ("state", "Is invoice INV-42 currently paid?"),
    ("state", "What is the latest version mentioned?"),
    ("state", "Exactly two distinct invoices are paid."),
    ("state", "How many errors are in this log?"),
    ("state", "Is the service up as of the last entry?"),
    ("state", "Invoice INV-0300 is still valid."),
    ("whole", "Is this about Georgiana?"),
    ("whole", "Is this a good readme?"),
    ("whole", "Is this a romance novel?"),
    ("whole", "Is the tone of this document angry?"),
    ("whole", "Is this log mostly noise?"),
    ("whole", "Is this documentation for a command-line tool?"),
    ("whole", "Is this a financial ledger?"),
    ("whole", "Is this text written in English?"),
]
# Written before any tuning against CASES and never tuned on: a wording that
# only works on CASES shows here.
HELD = [
    ("exists", "Does the log contain a timeout?"),
    ("exists", "Was a refund issued for any order?"),
    ("exists", "Is there a recipe for pancakes in here?"),
    ("exists", "The manual explains how to configure a proxy."),
    ("exists", "Somebody mentions Paris."),
    ("all", "Are all the links valid?"),
    ("all", "Every chapter ends with a question."),
    ("all", "Nothing in this output is marked failed."),
    ("all", "Each request was authenticated."),
    ("state", "What is the final balance?"),
    ("state", "Is the build green right now?"),
    ("state", "How many users signed up?"),
    ("state", "Was the last deployment successful?"),
    ("whole", "Is this a legal contract?"),
    ("whole", "Is this boring?"),
    ("whole", "Is this mostly about cooking?"),
    ("whole", "Is this written for children?"),
]


def main():
    root = sys.argv[1] if len(sys.argv) > 1 else os.path.dirname(HERE)
    sys.path.insert(0, root)
    import judge
    import mem
    host, model = judge.route(None)
    klass = types.SimpleNamespace(**vars(judge))

    def ask(q, text, labels, options=None):
        return klass.judge(q, text, labels, options, model=model, host=host)
    tally = {}
    for name, want, q in [("tuned", w, q) for w, q in CASES] + [("held", w, q) for w, q in HELD]:
        got, p, r = mem.kind(q, ask)
        raw = mem.KINDS.get(r.get("label"), ("?",))[0]
        bad = got == "whole" and want in ("exists", "all")
        t = tally.setdefault(name, [0, 0, 0])
        t[0], t[1], t[2] = t[0] + (got == want), t[1] + 1, t[2] + bad
        print(f"{name:5} {'ok' if got == want else 'XX'}{' COSTLY' if bad else ''}  want {want:6} picked {raw:6} "
              f"p {p if p is None else round(p, 2)}  {q}", flush=True)
    for name, (right, n, costly) in tally.items():
        print(f"{name}: {right}/{n} routed as expected, {costly} one-line questions sent to the sample ({model})")


if __name__ == "__main__":
    main()
