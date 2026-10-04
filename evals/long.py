#!/usr/bin/env python3
"""Claims over inputs too long to read in one fast call: wall time and verdict
per case, to compare reading strategies.

    evals/long.py run [CLASSIF] [OUT] [FROM]   # score cases FROM on, one JSON row each
    evals/long.py report OUT [OUT...]   # accuracy and time per run

The texts are man pages rendered on this machine (`man bash`, `man rsync`), a
generated ledger and Pride and Prejudice from Project Gutenberg, fetched on
first use, so nothing third-party is stored here. Expected answers are this file's; a case
whose page lacks the fact on another system is a wrong case there, not a miss.
CLASSIF defaults to the classif beside this directory; pass another checkout's
to measure it. Each case gets an empty cache so saved reader scores do not
hide the cost.
"""

import json
import os
import subprocess
import sys
import tempfile
import time

HERE = os.path.dirname(os.path.realpath(__file__))
STATE = os.path.join(os.environ.get("XDG_STATE_HOME") or os.path.expanduser("~/.local/state"), "classif", "bench")

# (page, expected label, claim). yes and no are what a reader of the whole page
# would answer; no means the page contradicts the claim or shows it absent.
CASES = [
    ("rsync", "yes", "The text describes an option that deletes extraneous files from the destination."),
    ("rsync", "yes", "The text describes an option that limits the transfer bandwidth."),
    ("rsync", "yes", "rsync can skip files larger than a given size."),
    ("rsync", "no", "The text explains how to configure a Kubernetes ingress."),
    ("rsync", "no", "rsync needs a daemon running on both ends for every transfer."),
    ("bash", "yes", "The document describes an option that makes the shell exit when a command fails."),
    ("bash", "yes", "The document describes a way to make a pipeline fail when any command in it fails, not only the last."),
    ("bash", "yes", "There is a shell option that corrects minor spelling errors in directory names given to cd."),
    ("bash", "no", "The document describes the docker run command."),
    ("bash", "no", "The document states that bash is copyrighted by Microsoft."),
    ("bash", "no", "Every shell variable the document lists is read-only."),
    # Reworded: under Doc.find the answering line ranks 72nd or shares no term
    # with the claim, so only a read by meaning finds it. Two plain claims above
    # (exit on failure, pipefail) miss the search too.
    ("bash", "yes", "The shell can be told to stop at the first error."),
    ("bash", "yes", "You can make the prompt show the current working directory."),
    ("rsync", "yes", "rsync can throttle its network usage."),
    ("rsync", "yes", "A copy can be rehearsed without changing anything."),
    # The ledger's claims name invoices, so the lines linked to them can answer
    # before a full read. The spring batch line bears on INV-0300 without
    # naming it: only a full read meets it.
    ("ledger", "no", "Invoice INV-0042 is currently paid."),
    ("ledger", "yes", "Invoice INV-0100 has been paid."),
    ("ledger", "yes", "Invoice INV-0200 came from a supplier based in Berlin."),
    ("ledger", "no", "Invoice INV-0300 is still valid."),
    ("ledger", "no", "Every invoice issued by SUP-3 has been paid."),
    # The README's example: a witness early in a 772 KB novel.
    ("book", "yes", "At least one line refers to Elizabeth Bennet."),
    # About the text as a whole: answered from counted facts and a sample.
    ("book", "no", "Is this about Georgiana?"),
    ("book", "yes", "Is this a romance novel?"),
    ("bash", "yes", "Is this documentation for a command-line tool?"),
    ("bash", "no", "Is this a cookbook?"),
    ("ledger", "yes", "Is this a financial ledger?"),
    # The novel with its names swapped and one death written in, so a right
    # answer comes from the text, not from what the model remembers of the
    # book. Each yes has one scene behind it; each no is never in the text.
    ("novel", "yes", "does Mr. Pryce propose to Marisol?"),
    ("novel", "yes", "does Hollis propose to Marisol?"),
    ("novel", "yes", "does Prudence run off with Caldwell?"),
    ("novel", "yes", "is Odette ever ill?"),
    ("novel", "yes", "is Mr. Hollis rich?"),
    ("novel", "yes", "does Odette die?"),
    ("novel", "no", "does Marisol die?"),
    ("novel", "no", "does anyone fight a duel?"),
    ("novel", "no", "does Hollis travel to France?"),
]
BOOK = "https://www.gutenberg.org/cache/epub/1342/pg1342.txt"
RENAMED = [("Elizabeth", "Marisol"), ("Lizzy", "Mari"), ("Eliza", "Mari"), ("Darcy", "Hollis"), ("Bingley", "Fenwick"),
           ("Jane", "Odette"), ("Lydia", "Prudence"), ("Wickham", "Caldwell"), ("Collins", "Pryce"),
           ("Bennet", "Ashworth"), ("Netherfield", "Thornbury"), ("Longbourn", "Wrenfield")]
DEATH = ("\nOdette, who had never fully recovered her strength, died of a fever that winter, and was buried beside "
         "the church at Wrenfield.\n")


def novel():
    """The book under other names, with one death the book does not have."""
    text = open(page("book")).read()
    for old, new in RENAMED:
        text = text.replace(old, new)
    lines = text.split("\n")
    lines.insert(10900, DEATH)
    return "\n".join(lines)


def ledger():
    """A payments ledger past the model's window, the same on every machine."""
    out = []
    for n in range(1, 2501):
        inv, sup = f"INV-{n:04d}", "SUP-777" if n == 200 else f"SUP-{n % 40}"
        day = f"2026-{1 + n % 9:02d}-{1 + n % 28:02d}"
        out.append(f"{day}: Invoice {inv} issued by {sup} for {100 + n * 7 % 900} EUR.")
        if n == 300:
            out.append(f"{day}: Invoice {inv} was issued as part of the spring batch.")
        if n == 1803:
            continue    # the one SUP-3 invoice never paid
        out.append(f"{day}: Invoice {inv} paid in full.")
        if n == 42:
            out.append("That payment was reversed the next day.")
        if n == 1200:
            out.append("Supplier SUP-777 is based in Berlin.")
        if n == 2400:
            out.append("Every invoice in the spring batch was cancelled.")
    return "\n".join(out) + "\n"


def page(name):
    path = os.path.join(tempfile.gettempdir(), f"classif-long-{name}.txt")
    if not os.path.exists(path):
        if name == "book":
            text = subprocess.run(["curl", "-sfL", BOOK], capture_output=True, text=True, check=True).stdout
        elif name == "ledger":
            text = ledger()
        elif name == "novel":
            text = novel()
        else:
            text = subprocess.run(f"MANWIDTH=80 man {name} | col -b", shell=True, capture_output=True, text=True,
                                  check=True).stdout
        with open(path, "w") as fh:
            fh.write(text)
    return path


def run(classif=None, out=None, start=0):
    classif = classif or os.path.join(os.path.dirname(HERE), "classif")
    os.makedirs(STATE, exist_ok=True)
    out = out or os.path.join(STATE, f"long-{time.strftime('%Y%m%d-%H%M%S')}.jsonl")
    with open(out, "w") as fh:
        for name, want, claim in CASES[int(start):]:
            path = page(name)
            with tempfile.TemporaryDirectory() as cache:
                t0 = time.monotonic()
                r = subprocess.run([classif, "-j", "-i", path, claim], capture_output=True, text=True,
                                   env=dict(os.environ, XDG_CACHE_HOME=cache))
                wall = time.monotonic() - t0
            try:
                res = json.loads(r.stdout)
            except ValueError:
                res = {"label": None, "error": r.stderr.strip()[-300:]}
            read = res.get("read") or {}
            row = {"page": name, "chars": os.path.getsize(path), "claim": claim, "want": want, "got": res.get("label"),
                   "p": (res.get("p") or {}).get(res.get("label")), "mode": res.get("mode"),
                   "basis": read.get("basis"), "plan": read.get("plan"), "calls": read.get("calls"),
                   "wall": round(wall, 2), "exit": r.returncode}
            fh.write(json.dumps(row) + "\n")
            fh.flush()
            print(f"{row['wall']:7.1f}s  want {want:3}  got {str(row['got']):5}  {row['mode']}/{row['basis']}  "
                  f"{name}: {claim}", flush=True)
    print(out)


def report(paths):
    for path in paths:
        with open(path) as fh:
            rows = [json.loads(l) for l in fh]
        right = sum(r["got"] == r["want"] for r in rows)
        wrong = sum(r["got"] not in (None, r["want"]) for r in rows)
        walls = sorted(r["wall"] for r in rows)
        print(f"{os.path.basename(path)}: {right}/{len(rows)} right, {wrong} wrong, "
              f"{len(rows) - right - wrong} no answer; wall p50 {walls[len(walls) // 2]:.1f}s, "
              f"max {walls[-1]:.1f}s, total {sum(walls):.0f}s")


def main():
    if len(sys.argv) < 2 or sys.argv[1] not in ("run", "report"):
        sys.exit(__doc__)
    if sys.argv[1] == "run":
        run(*sys.argv[2:5])
    else:
        report(sys.argv[2:])


if __name__ == "__main__":
    main()
