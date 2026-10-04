#!/usr/bin/env python3
"""mem: judge a claim about a text too long to read in one call.

    mem.py "Every invoice is paid." ledger.txt
    journalctl -b | mem.py -P exists "A disk reported an I/O error."
    mem.py -j -r llama3.2:3b "The contract was renewed." contract.txt

The text stays outside the model. A root call reads the question alone and
says what kind of answer it needs. A question one scene or one state settles
goes to the search: the text's passages are indexed once (vectors from a
small embedding model, saved beside the readings, and BM25), and one
single-token call judges the passages closest to the question, in seconds
where a full read is minutes. What the search does not settle, and a claim
about every line, is read in full: one single-token call per span (the leaf
check) says whether the span fits the claim, breaks it, relates to it or has
nothing to do with it; the rows stay in a table outside any prompt; code
reduces them to a few facts and the spans that matter; one final
single-token call judges those. Nothing is generated.

Prints `<verdict> <p>` and one line on stderr saying what was read, or JSON
with -j. Verdicts: supported (exit 0), contradicted (1), unscored (2, a model
call failed), insufficient (3). A verdict other than insufficient needs every
span checked. Only a plan pinned with -P may stop early: `exists` on a span
that fits and that the judge confirms, `all` on one that breaks. `latest`
adds a check for two opposite spans on the same latest date, `join` one for
an identifier chain that reaches an unread span.

    doc = Doc(text)                   built once per text, keyed by sha256
    doc.find(terms, limit)            -> (span ids by BM25, total matching)
    doc.digest(ids)                   -> lines; repeats collapsed, no value lost
    gather(doc, question)             -> lexical evidence and its read report
    run(doc, question, ask, plan)     -> verdict, label, p and the read report

`ask(question, text, labels)` is judge.judge's shape: {label, p} or
{label: None, unscored}. Tests inject a fake; the CLI passes judge.judge.
"""
import argparse
import array
import bisect
import collections
import hashlib
import importlib.machinery
import importlib.util
import json
import math
import operator
import os
import re
import sqlite3
import sys
import textwrap
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor

SPAN_MAX = 400     # chars per span
BUDGET = 4000      # chars of evidence, about 900 tokens: one second of gemma4:12b
HOPS = 3
HEAD = 8           # best lexical matches read line by line before any passage
K1, B = 1.2, 0.75
PLANS = ("auto", "exists", "all", "count_distinct", "latest", "join", "semantic_check", "semantic_join")
LEAF = ("fits", "breaks", "related", "unrelated")
JUDGE = ("yes", "no", "unknown")
LEAF_QUESTION = ('Claim: "{claim}" Taken alone, does this line give a case that fits the claim, a case that breaks '
                 "it, something related that decides nothing by itself, or something unrelated?")
BLOCK_QUESTION = ('Claim: "{claim}" Does this passage hold a line that fits the claim, a line that breaks it, a line '
                  "related to it, or is all of it unrelated?")
LINK_QUESTION = ('The last line uses a word like "that", "the same", "it" or "they" for something an earlier line '
                 'names. Which numbered line names it? Answer 0 when none of them does.')
HUB = 20            # a name found on a linked line is not followed when more lines than this carry it
WINDOW = 8         # earlier lines offered as what a line points back to; one digit must name each
ENUM_QUESTION = "Going only by this passage: {question}"
ALL_QUESTION = 'Does any of these lines break this claim: "{claim}"'
INVERT = True
# The root call: past the window the model reads the question alone and picks
# how much of the text answering it needs, as RLM's root model picks its program.
KIND_QUESTION = ("The text above is a question or claim about a long document. Which kind is it? Pick the number "
                 "of the first option that fits.")
KINDS = {"1": ("state", "it asks for a current, latest or final state, a value now, an order, a total or a count "
                        "(currently, still, now, latest, last, as of, how many, exactly)"),
         "2": ("all", "it holds only if every case holds, so one line against it settles it (every, all, each, "
                      "only, always, none, no, never, not any)"),
         "3": ("whole", "it is about the document as a whole: its topic, subject, kind, genre, tone, language or "
                        "quality, or what most of it is"),
         "4": ("exists", "one line that shows it is enough: something is mentioned, described, exists or "
                         "happened at least once")}
KIND_MIN = 0.8      # a less sure pick reads every line
WHOLE_BUDGET = 12000    # chars of sampled passages for a whole-text question, about 3k tokens or 1.6 s
JUDGE_QUESTION = 'Going only by these lines and facts, is this claim true: "{claim}"'
# The search: the passages closest to the question, by vector and by words,
# in one judge call. The 12B reads 2,000 tokens a second, so a full read of a
# novel is minutes; the index reads it once at ten times that and a question
# then costs one call. A witness among the closest passages settles an
# exists question either way: found, or not shown in the text's closest
# passages.
CHUNK = 1200            # chars per indexed passage: a paragraph or a run of lines
SEARCH_BUDGET = 30000   # chars the search judge reads, about 7.5k tokens or 5 s on gemma 12B
# Passages judged first: the closest by words, few and exact (an identifier,
# a rare name), then the closest by vector. Words go first since the budget
# cuts the tail: a supplier's one line in a ledger ranked first by words and
# was cut after twenty look-alike passages by vector.
SEARCH_LEXICAL, SEARCH_DENSE = 6, 20
SEARCH_QUESTION = 'Going only by these passages from a longer text, is this shown: "{claim}"'
dot = getattr(math, "sumprod", lambda a, b: sum(map(operator.mul, a, b)))

LEVEL = re.compile(r"\b(FATAL|CRITICAL|ERROR|WARN(?:ING)?|INFO|DEBUG|TRACE)\b")
TOKEN = re.compile(r"\w[\w\-./:]*\w|\w")
DATE = re.compile(r"\b\d{4}-\d{2}-\d{2}\b")
NAME = re.compile(r"[A-Z][\w'-]*\.?(?:\s+[A-Z][\w'-]*\.?)*")
ANAPHOR = re.compile(r"\b(?:that|this|these|those|the same|the latter|the former|said|such|it|its|they|their|them|"
                     r"he|him|his|she|her)\b", re.IGNORECASE)
SENTENCE = re.compile(r"(?<=[.!?;:])\s+")
STOP = set("""a an and are as at be been but by can could did do does for from had has have how i if in into is it its
may might must no not of on or our shall should so such than that the their them then there these they this those to
was we were what when where which who whom whose why will with would you your about above after again all also any
because before being between both during each few further here just more most only other over own same some too under
until up very via while answer assess claim whether text following options option none one word exactly""".split())

Group = collections.namedtuple("Group", "template members rows")
Row = collections.namedtuple("Row", "span label p status via", defaults=["line"])
Reduction = collections.namedtuple("Reduction", "facts support counts conflict closure")


def terms(text):
    """Search terms of a text: identifiers verbatim (a token holding a digit
    and a letter, or all capitals), other words lowercased without stop words."""
    out = []
    for tok in TOKEN.findall(text):
        if is_id(tok):
            out.append(tok)
        elif tok.lower() not in STOP and len(tok) > 2:
            out.append(tok.lower())
    return out


def is_id(tok):
    return (any(c.isdigit() for c in tok) and any(c.isalpha() for c in tok)) or (tok.isupper() and len(tok) > 1)


def split(text):
    """(start, end) spans in order: a line is one span, an overlong line is
    packed sentence by sentence, an overlong sentence is cut at whitespace."""
    spans, pos = [], 0
    for line in text.split("\n"):
        start, end = pos, pos + len(line)
        pos = end + 1
        if not line.strip():
            continue
        if len(line) <= SPAN_MAX:
            spans.append((start, end))
            continue
        cur = start
        cuts = [start + m.end() for m in SENTENCE.finditer(line)] + [end]
        for i, cut in enumerate(cuts):
            nxt = cuts[i + 1] if i + 1 < len(cuts) else None
            if nxt is not None and nxt - cur <= SPAN_MAX:
                continue
            while cut - cur > SPAN_MAX:
                hard = text.rfind(" ", cur, cur + SPAN_MAX)
                hard = hard + 1 if hard > cur else cur + SPAN_MAX
                spans.append((cur, hard))
                cur = hard
            if cut > cur:
                spans.append((cur, cut))
                cur = cut
    return spans


class Doc:
    """One text, never changed. Spans are str offsets (unit "char"):
    text[start:end] is the span. complete=False marks a stream that may still
    grow, so no reading of it ever counts as having read everything."""
    unit = "char"

    def __init__(self, text, complete=True):
        self.text, self.complete = text, complete
        self.key = hashlib.sha256(text.encode()).hexdigest()
        self.spans = split(text)
        self.index = collections.defaultdict(dict)   # casefolded term -> {span id: count}
        self.length = []
        for i in range(len(self.spans)):
            ts = [t.casefold() for t in terms(self.span(i))]
            self.length.append(len(ts))
            for t in ts:
                self.index[t][i] = self.index[t].get(i, 0) + 1
        self.avg = (sum(self.length) / len(self.length)) if self.length else 0.0

    def span(self, i):
        if isinstance(i, bool) or not isinstance(i, int) or not 0 <= i < len(self.spans):
            raise IndexError(f"span id {i!r} outside 0..{len(self.spans) - 1}")
        s, e = self.spans[i]
        return self.text[s:e]

    def find(self, query, limit):
        """Span ids ranked by BM25 over the query terms, and how many spans
        matched at all, so a cut result is visible to the caller. Terms match
        in any case; the source keeps its own."""
        score, n = collections.defaultdict(float), len(self.spans)
        for t in {q.casefold() for q in query}:
            posting = self.index.get(t)
            if not posting:
                continue
            idf = math.log(1 + (n - len(posting) + 0.5) / (len(posting) + 0.5))
            for i, tf in posting.items():
                score[i] += idf * tf * (K1 + 1) / (tf + K1 * (1 - B + B * self.length[i] / (self.avg or 1)))
        ranked = sorted(score, key=lambda i: (-score[i], i))
        return ranked[:max(0, limit)], len(ranked)

    def read(self, ids, around=0):
        """Those span ids and their neighbours, in document order."""
        return sorted({j for i in ids for j in range(max(0, i - around), min(len(self.spans), i + around + 1))})

    def ids(self, i):
        return list(dict.fromkeys(t for t in TOKEN.findall(self.span(i)) if is_id(t)))

    def dates(self, i):
        return DATE.findall(self.span(i))

    def links(self, ids, seen):
        """Identifiers in those spans not seen yet, in the source's case.
        seen holds casefolded identifiers."""
        found = []
        for i in ids:
            for tok in self.ids(i):
                if tok.casefold() not in seen and tok not in found:
                    found.append(tok)
        return found

    def template(self, i):
        """(the span with every token that holds a digit replaced by #, those tokens)."""
        values = []

        def mask(m):
            if any(c.isdigit() for c in m.group()):
                values.append(m.group())
                return "#"
            return m.group()
        return TOKEN.sub(mask, self.span(i).strip()), tuple(values)

    def groups(self, ids):
        """Spans that differ only in tokens holding digits, grouped. rows keeps
        every member's values, so nothing a group stands for is lost."""
        by = collections.OrderedDict()
        for i in sorted(set(ids)):
            tpl, values = self.template(i)
            by.setdefault(tpl, []).append((i, values))
        return [Group(tpl, [i for i, _ in rows], rows) for tpl, rows in by.items()]

    def digest(self, ids):
        """Lines for the model, in document order. Three or more spans of one
        form become one line that lists every member's differing values."""
        out = []
        for g in self.groups(ids):
            if len(g.members) < 3:
                out.extend((i, self.span(i).strip()) for i in g.members)
                continue
            slots = list(zip(*(v for _, v in g.rows)))
            vary = [k for k, col in enumerate(slots) if len(set(col)) > 1]
            parts, k = g.template.split("#"), 0
            line = parts[0]
            for part in parts[1:]:
                line += ("#" if k in vary else slots[k][0]) + part
                k += 1
            values = "; ".join(" ".join(v[k] for k in vary) for _, v in g.rows)
            out.append((g.members[0], f"{line} [{len(g.members)} lines of this form" +
                        (f"; # in each: {values}]" if vary else ", identical]")))
        return [line for _, line in sorted(out)]


def evidence(doc, ids, top=20, chars=300):
    """Where span ids sit in the source, as [{line, end, text}]: 1-based
    line numbers a reader can open the source at, runs of touching lines
    merged, each text cut to chars. The first top runs are kept. A run is
    dedented as a whole, so its lines keep their indentation relative to
    each other."""
    if not ids:
        return []
    nl = [m.start() for m in re.finditer("\n", doc.text)]
    runs = []
    for i in sorted(set(ids)):
        s, e = doc.spans[i]
        e = s + len(doc.text[s:e].rstrip("\n"))
        first, last = bisect.bisect_left(nl, s) + 1, bisect.bisect_left(nl, e) + 1
        if runs and first <= runs[-1][1] + 1:
            runs[-1][1], runs[-1][3] = max(runs[-1][1], last), max(runs[-1][3], e)
        else:
            runs.append([first, last, s, e])
    return [{"line": a, "end": b, "text": textwrap.dedent(doc.text[s:e]).rstrip()[:chars]}
            for a, b, s, e in runs[:top]]


def pack(doc, ordered, budget):
    """The longest prefix of the ordered span ids whose digest fits the
    budget. Returns (text, ids kept, whether any id was left out)."""
    kept, text = [], ""
    for i in ordered:
        trial = "\n".join(doc.digest(kept + [i]))
        if len(trial) > budget:
            return text, kept, True
        kept, text = kept + [i], trial
    return text, kept, False


def closure(doc, question, hops=HOPS):
    """Spans holding the question's identifiers, then spans holding the
    identifiers those mention, hop by hop. An identifier met on the way that
    more than HUB spans carry is not followed: a supplier every invoice names
    would chain one invoice to the whole ledger. Returns (span ids, hops used)."""
    frontier = list(dict.fromkeys(t for t in terms(question) if is_id(t)))
    seen, picked, hop_n = {t.casefold() for t in frontier}, [], 0
    while frontier and hop_n < hops:
        hits = []
        for t in frontier:
            hits.extend(doc.index.get(t.casefold(), {}))
        hits = [i for i in dict.fromkeys(hits) if i not in picked]
        if not hits:
            break
        hop_n += 1
        picked.extend(hits)
        found = doc.links(hits, seen)
        seen.update(t.casefold() for t in found)
        frontier = [t for t in found if len(doc.index.get(t.casefold(), {})) <= HUB]
    return picked, hop_n


def gather(doc, question, budget=BUDGET, hops=HOPS):
    """Lexical evidence for a question: the identifier closure, then the best
    word matches with their neighbours, until the budget. Returns (evidence
    text, report). judged lists only spans whose text went into the evidence,
    and a lexical read never claims semantic coverage."""
    picked, hop_n = closure(doc, question, hops)
    ranked, matched = doc.find(terms(question), len(doc.spans))
    order = list(picked)
    for i in ranked:
        order.extend(j for j in doc.read([i], around=1) if j not in order)
    text, kept, cut = pack(doc, order, budget)
    coverage = "none" if not order else "truncated" if cut else "complete"
    return text, {"method": "lookup", "spans": len(doc.spans), "judged": sorted(kept), "hits": len(picked),
                  "matched": matched, "hops": hop_n, "coverage": coverage, "semantic_coverage": "none",
                  "chars": len(text)}


class Table:
    """Leaf results, kept outside every prompt. One row per checked span."""

    def __init__(self):
        self.rows = {}

    def add(self, row):
        self.rows[row.span] = row

    def spans(self, *labels, via=None):
        """Span ids with one of the labels; via="line" leaves out rows a whole
        passage gave, which were never read on their own."""
        return sorted(r.span for r in self.rows.values() if r.status == "ok" and r.label in labels
                      and (via is None or r.via != "block"))

    def failed(self):
        return sorted(r.span for r in self.rows.values() if r.status != "ok")


def cut_short(r):
    """True when a call failed because the caller's deadline stopped it. The
    caller's ask says so with deadline=True on the failed result: a call
    bounded by the time left returns just before the deadline, so the clock
    alone cannot tell it from a broken reader."""
    return r.get("label") is None and bool(r.get("deadline"))


def check(doc, ids, claim, ask, table=None, jobs=1):
    """One leaf call per span: fits, breaks, related or unrelated to the
    claim. A failed call is a row too, so it is never read as unrelated."""
    table = table if table is not None else Table()
    question = LEAF_QUESTION.format(claim=claim)

    def one(i):
        r = ask(question, doc.span(i).strip(), list(LEAF))
        if r.get("label") is None:
            return Row(i, None, None, "deadline" if cut_short(r) else str(r.get("unscored", "failed")))
        return Row(i, r["label"], r["p"][r["label"]], "ok")
    if jobs > 1 and len(ids) > 1:
        with ThreadPoolExecutor(jobs) as pool:
            rows = list(pool.map(one, ids))
    else:
        rows = [one(i) for i in ids]
    for row in rows:
        table.add(row)
    return table


def blocks(doc, ids, size):
    """Runs of consecutive span ids, each at most size chars of source."""
    out, cur = [], []
    for i in sorted(ids):
        if cur and (i != cur[-1] + 1 or doc.spans[i][1] - doc.spans[cur[0]][0] > size):
            out.append(cur)
            cur = []
        cur.append(i)
    return out + [cur] if cur else out


def chunks(doc, size=CHUNK):
    """Runs of consecutive spans of at most size chars of source, cut at a
    paragraph end (a blank line) once a run is half full: a novel's passage
    is a paragraph or a few, a log's a run of lines."""
    out, cur = [], []
    for i in range(len(doc.spans)):
        if cur:
            gap = doc.text[doc.spans[cur[-1]][1]:doc.spans[i][0]]
            full = doc.spans[i][1] - doc.spans[cur[0]][0] > size
            if full or (gap.count("\n") > 1 and doc.spans[cur[-1]][1] - doc.spans[cur[0]][0] >= size // 2):
                out.append(cur)
                cur = []
        cur.append(i)
    return out + [cur] if cur else out


def stem(word):
    """A crude stem for the word index: pay, pays and paying meet, invoice
    and invoices too. An identifier is left alone."""
    if not word.islower():
        return word
    for suffix in ("ing", "ed", "s"):
        if word.endswith(suffix) and len(word) - len(suffix) >= 3:
            return word[:-len(suffix)]
    return word


class Index:
    """A text's passages ranked for a question without a model call per
    passage: unit vectors from an embedding model, built once per text and
    kept, and BM25 over the same passages by stem. rank() puts the passages
    closest in meaning first, then the closest in words, so one judge call
    reads what a reader would skim to."""

    def __init__(self, doc, size=CHUNK):
        self.doc, self.blks = doc, chunks(doc, size)
        self.texts = [doc.text[doc.spans[b[0]][0]:doc.spans[b[-1]][1]] for b in self.blks]
        self.vectors, self.tokens = None, 0
        self.counts = [collections.Counter(stem(w) for w in terms(t)) for t in self.texts]
        self.df = collections.Counter(w for c in self.counts for w in c)
        self.length = [sum(c.values()) for c in self.counts]
        self.avg = (sum(self.length) / len(self.length)) if self.length else 0.0

    def embed(self, fn, progress=None, over=None, batch=64, jobs=4):
        """Vectors for every passage through fn(texts), batch texts a call
        and jobs calls at once. Returns None, or the call's failure: the
        index then ranks by words alone."""
        batches = [self.texts[i:i + batch] for i in range(0, len(self.texts), batch)]
        vectors = []
        with ThreadPoolExecutor(max(1, jobs)) as ex:
            for at in range(0, len(batches), max(1, jobs)):
                if over is not None and over():
                    return {"unscored": "deadline passed while indexing"}
                for r in ex.map(fn, batches[at:at + max(1, jobs)]):
                    if r.get("vectors") is None:
                        return r
                    vectors += r["vectors"]
                    self.tokens += r.get("tokens", 0)
                    if progress:
                        progress(len(vectors), len(self.texts), "passages into the index")
        self.vectors = vectors
        return None

    def lexical(self, question):
        """Ids of the passages sharing a stem with the question, by BM25,
        best first."""
        qt, n, scores = {stem(w) for w in terms(question)}, len(self.texts), {}
        for i, c in enumerate(self.counts):
            s = 0.0
            for w in qt:
                tf = c.get(w, 0)
                if tf:
                    idf = math.log(1 + (n - self.df[w] + 0.5) / (self.df[w] + 0.5))
                    s += idf * tf * (K1 + 1) / (tf + K1 * (1 - B + B * self.length[i] / (self.avg or 1)))
            if s > 0:
                scores[i] = s
        return sorted(scores, key=lambda i: (-scores[i], i))

    def rank(self, question, qvec=None):
        """Passage ids for a question: the SEARCH_LEXICAL closest by words,
        the SEARCH_DENSE closest by vector, then the rest by vector. Without
        vectors, by words alone, then in text order."""
        lex = self.lexical(question)
        if self.vectors is None or qvec is None:
            return list(dict.fromkeys(lex + list(range(len(self.texts)))))
        sims = [dot(qvec, v) for v in self.vectors]
        dense = sorted(range(len(sims)), key=lambda i: (-sims[i], i))
        return list(dict.fromkeys(lex[:SEARCH_LEXICAL] + dense[:SEARCH_DENSE] + dense))


def search(doc, index, question, ask, mode="exists", budget=SEARCH_BUDGET, qvec=None):
    """One judge call over the passages closest to the question, in text
    order, as many as fit the budget. mode exists asks yes or no: a witness
    among the closest passages settles it, and none means not shown. state
    asks yes, no or unknown, since the line that decides a state may sit
    anywhere. Basis "search"; the read says how many passages of how many."""
    t0, order, kept = time.monotonic(), index.rank(question, qvec), []
    # The lines chained to the question's identifiers go first: an invoice,
    # its supplier and the supplier's city sit together under the judge's
    # eye, where the closest passages hold them a hundred KB apart.
    chain, chained, _ = pack(doc, closure(doc, question)[0], budget // 8)
    used = len(chain) + 2 if chain else 0
    for i in order:
        if used + len(index.texts[i]) + 2 > budget:
            if kept:
                break
            continue
        kept.append(i)
        used += len(index.texts[i]) + 2
    kept.sort()
    labels = ["yes", "no"] if mode == "exists" else list(JUDGE)
    r = ask((SEARCH_QUESTION if mode == "exists" else JUDGE_QUESTION).format(claim=question),
            "\n\n".join(([chain] if chain else []) + [index.texts[i].strip() for i in kept]), labels)
    label = r.get("label")
    if label == "no" and not doc.complete:
        label, why = None, "the text may still grow"
    else:
        why = "" if label in ("yes", "no") else str(r.get("unscored") or "the closest passages do not settle it")
    verdict = {"yes": "supported", "no": "contradicted", "unknown": "insufficient", None: "insufficient"}[label]
    if r.get("label") is None:
        verdict = "unscored"
    read = {"plan": "search", "pinned": False, "doc_key": doc.key, "unit": doc.unit, "spans": len(doc.spans),
            "doc_complete": doc.complete, "passages": len(index.blks), "checked": len(kept),
            "failed": 0 if r.get("label") else 1, "scan_complete": False,
            "by": "meaning and words" if index.vectors is not None and qvec is not None else "words",
            "basis": "search" if label in ("yes", "no") else "none",
            "judged": sorted(set(chained) | {j for i in kept for j in index.blks[i]}),
            "sources": [{"span": [index.blks[i][0], index.blks[i][-1]], "start": doc.spans[index.blks[i][0]][0],
                         "end": doc.spans[index.blks[i][-1]][1]} for i in kept],
            "evidence": [], "why": why, "calls": 1, "ms": round((time.monotonic() - t0) * 1000)}
    return {"verdict": verdict, "label": label if label in ("yes", "no") else None,
            "p": r.get("p") if label in ("yes", "no") else None, "read": read}


def reduce(doc, table, plan, question="", allowed=None):
    """Code facts over the table and the spans the judge should read. Flagged
    spans come first (breaks, fits, related), then the question's identifier
    closure, so a line that decides nothing alone still reaches the judge.
    conflict: under `latest`, the latest date carries both a fits and a breaks
    span. closure: under `join`, closure spans the executor may not read."""
    fits, breaks, related = table.spans("fits"), table.spans("breaks"), table.spans("related", via="line")
    counts = {l: len(table.spans(l)) for l in LEAF}
    chain, _ = closure(doc, question)
    allowed = set(range(len(doc.spans))) if allowed is None else set(allowed)
    missing = [i for i in chain if i not in allowed] if plan in ("join", "auto") else []
    support = list(dict.fromkeys(breaks + fits + related + [i for i in chain if i in allowed]))
    # The judge gets facts about content only. How much was read is the gate's
    # business: told "4 of 2201 lines were checked", gemma4:12b answered unknown
    # (p 1.00) on a witness it called yes (p 1.00) when told "1 of 2201". The
    # fit and break tallies stay out too: one wrong leaf label would sway it.
    facts = []
    if plan in ("count_distinct", "auto"):
        # One line neither fits nor breaks a claim about how many, so count
        # what the flagged lines name.
        flagged = sorted(set(fits + breaks + related))
        keys = list(dict.fromkeys(k for i in flagged for k in doc.ids(i)))
        facts.append(f"Distinct identifiers in these lines: {', '.join(keys[:20]) or 'none'} ({len(keys)}).")
    dated = [(d, i) for i in fits + breaks for d in doc.dates(i)]
    conflict = False
    if dated:
        last = max(d for d, _ in dated)
        at = sorted({i for d, i in dated if d == last})
        facts.append(f"The latest date in these lines is {last}.")
        conflict = plan in ("latest", "auto") and any(i in fits for i in at) and any(i in breaks for i in at)
    return Reduction(facts, support, counts, conflict, missing)


def kind(question, ask):
    """The root call: what answering the question needs, read from the
    question alone. Returns (kind or None when unsure, p, the answer)."""
    labels = list(KINDS)
    r = ask(KIND_QUESTION, question.strip(), labels, [d for _, d in KINDS.values()])
    if r.get("label") is None:
        return None, None, r
    p = r["p"][r["label"]]
    return (KINDS[r["label"]][0] if p >= KIND_MIN else None), p, r


def notes(doc, top=10):
    """Facts counted over the whole text in code, in memory, in milliseconds:
    its size, the names and identifiers it carries most, its dates, log
    levels and the line forms it repeats most."""
    named = collections.Counter()
    for i in range(len(doc.spans)):
        for name in Graph.names(doc.span(i)):
            words = name.split()
            while len(words) > 1 and words[0].isupper():
                words = words[1:]       # "INFO Mr Darcy": a log level is not part of the name
            named[" ".join(words)] += 1
        named.update(t for t in doc.ids(i) if any(c.isdigit() for c in t))
    facts = [f"The text has {len(doc.text)} characters in {len(doc.spans)} lines."]
    common = [(a, c) for a, c in named.most_common(top) if c >= 2]
    if common:
        facts.append("Most named: " + ", ".join(f"{a} ({c})" for a, c in common) + ".")
    dates = sorted(d for i in range(len(doc.spans)) for d in doc.dates(i))
    if dates:
        facts.append(f"Dates run from {dates[0]} to {dates[-1]} ({len(dates)} in all).")
    levels = collections.Counter(LEVEL.findall(doc.text))
    if levels:
        facts.append("Log levels: " + ", ".join(f"{l} {c}" for l, c in levels.most_common()) + ".")
    forms = collections.Counter(re.sub(r"\d+", "#", doc.span(i).strip()) for i in range(len(doc.spans)))
    common = [(f, c) for f, c in forms.most_common(3) if c >= 3]
    if common:
        facts.append("Most repeated line forms: " + "; ".join(f'"{f[:120]}" ({c})' for f, c in common) + ".")
    return facts


def sample(doc, question, budget=WHOLE_BUDGET, n=12):
    """Span ids for a whole-text read: the question's best keyword matches
    and runs spread evenly over the text, in document order, within budget."""
    ranked, _ = doc.find(terms(question), 4)
    picked, size = list(ranked), sum(len(doc.span(i)) for i in ranked)
    per = max(1, (budget - size) // n)
    for k in range(n):
        i, run_ = len(doc.spans) * k // n, 0
        while i < len(doc.spans) and run_ < per and size + len(doc.span(i)) <= budget:
            if i not in picked:
                picked.append(i)
                run_, size = run_ + len(doc.span(i)), size + len(doc.span(i))
            i += 1
    return sorted(picked)


def whole(doc, question, ask, budget=WHOLE_BUDGET):
    """A whole-text question past the window: one call over the facts code
    counted and an even sample, the question as asked. Basis "sample"."""
    t0, facts, ids = time.monotonic(), notes(doc), sample(doc, question, budget)
    text = ("Facts counted over the whole text: " + " ".join(facts) + "\n\nPassages sampled evenly from the text:\n"
            + "\n".join(doc.span(i).strip() for i in ids))
    r = ask(question, text, list(JUDGE))
    label = r.get("label")
    verdict = {"yes": "supported", "no": "contradicted", "unknown": "insufficient", None: "unscored"}[label]
    read = {"plan": "whole", "pinned": False, "doc_key": doc.key, "unit": doc.unit, "spans": len(doc.spans),
            "doc_complete": doc.complete, "checked": len(ids), "failed": 0 if label else 1, "scan_complete": False,
            "basis": "sample" if label in ("yes", "no") else "none", "facts": facts, "judged": ids, "evidence": [],
            "sources": [{"span": i, "start": doc.spans[i][0], "end": doc.spans[i][1]} for i in ids],
            "why": "" if label in ("yes", "no") else str(r.get("unscored") or "the sample does not settle it"),
            "calls": 1, "ms": round((time.monotonic() - t0) * 1000)}
    return {"verdict": verdict, "label": label if label in ("yes", "no") else None,
            "p": r.get("p") if label in ("yes", "no") else None, "read": read}


def run(doc, question, ask, plan="auto", only=None, deadline=None, budget=BUDGET, jobs=1, leaf=None,
        block=0, graph=None, stop_at=None, progress=None):
    """Judge a claim about a document. ask judges; leaf (default ask) reads
    the spans. only limits the spans that may be read. block > 0 reads
    passages of that many chars first and goes line by line only inside one
    the reader does not call unrelated; a line dismissed that way is a row
    with via "block". graph, a Graph of doc, reads only the lines it links
    to what the claim names and may answer from them: the basis is then
    "component", never "complete", read.graph says what was linked, and its
    link calls count against calls and the deadline. Returns {verdict,
    label, p, read}; see the module docstring for what each verdict needs."""
    if plan not in PLANS:
        raise ValueError(f"plan {plan!r} is not one of {', '.join(PLANS)}")
    if graph is not None and plan in ("all", "count_distinct"):
        raise ValueError(f"plan {plan} speaks about every line, so it needs a full scan, not a component")
    if graph is not None and graph.doc.key != doc.key:
        raise ValueError("the graph was built from a different text; its line numbers do not fit this one")
    t0, leaf, calls, component = time.monotonic(), leaf or ask, 0, None

    def over():
        return deadline is not None and time.monotonic() - t0 > deadline

    if graph is not None:
        component = graph.component(question, leaf, over=over)
        only, calls = component[0], component[1]["link_calls"]
    allowed = list(range(len(doc.spans))) if only is None else sorted({i for i in only if doc.span(i) is not None})
    first, _ = closure(doc, question)
    ranked, _ = doc.find(terms(question), len(doc.spans))
    order = [i for i in dict.fromkeys(first + ranked + allowed) if i in set(allowed)]
    table, late, by_block, block_calls, related_blocks, link_failed = Table(), False, 0, 0, 0, []
    # pinned: the caller named the plan. Always true until a selector exists.
    read = {"plan": plan, "pinned": plan != "auto", "doc_key": doc.key, "unit": doc.unit, "spans": len(doc.spans),
            "doc_complete": doc.complete}
    if component is not None:
        read["graph"] = component[1]

    def done(verdict, label=None, p=None, basis="none", why="", red=None, judged=()):
        red = red or reduce(doc, table, plan, question, allowed)
        read.update(checked=len(table.rows) - len(table.failed()), failed=len(table.failed()),
                    scan_complete=len(table.rows) == len(doc.spans) and not table.failed(),
                    counts=red.counts, basis=basis, support=sorted(red.support), judged=sorted(judged),
                    sources=[{"span": i, "start": doc.spans[i][0], "end": doc.spans[i][1]} for i in sorted(judged)],
                    # The judged lines read one way or the other; an answer resting on none points at none.
                    evidence=evidence(doc, [i for i in judged if i in table.rows and table.rows[i].status == "ok"
                                            and table.rows[i].label in ("fits", "breaks")]),
                    blocks=block_calls, by_block=by_block, related_blocks=related_blocks,
                    by_link=sum(r.via == "link" for r in table.rows.values()),
                    why=why, calls=calls, ms=round((time.monotonic() - t0) * 1000))
        return {"verdict": verdict, "label": label, "p": p, "read": read}

    def judge(red, ask_what=JUDGE_QUESTION):
        """(answer, ids read, whether flagged lines were left out). budget
        bounds the whole text the judge reads, facts included; with no room
        for the facts no call is made and the answer is None."""
        nonlocal calls
        facts = "Facts: " + " ".join(red.facts) if red.facts else ""
        text, kept, cut = pack(doc, red.support, budget - len(facts) - 1)
        if len(facts) > budget or over():
            return None, [], True
        calls += 1
        return ask(ask_what.format(claim=question), "\n".join(filter(None, [text, facts])), list(JUDGE)), kept, cut

    if component is not None and component[1]["cut"]:
        return done("insufficient", why="deadline passed while linking lines")
    stop = {"exists": ("fits", "yes", "supported", "witness"), "all": ("breaks", "no", "contradicted", "counterexample")}
    tried, links = set(), graph

    def closed(ids, want):
        """Whether the lines in ids may end the read: every line up to the
        link window after each, and every line sharing an anchor with one
        (bar an anchor more than HUB lines carry), was read and none reads
        the other way. "That payment was reversed" follows its witness, so a
        witness is never taken without the lines that could undo it."""
        nonlocal calls, links
        against = "breaks" if want == "fits" else "fits"
        links = links or Graph(doc)
        near = set()
        for i in ids:
            near.update(range(i + 1, min(len(doc.spans), i + links.window + 1)))
            for a in links.anchors[i]:
                if len(links.by_anchor[a]) <= HUB:
                    near.update(links.by_anchor[a])
        near -= set(ids)
        if near - set(allowed):
            return False
        todo = sorted(j for j in near if j not in table.rows)
        runs = [[j] for j in todo[:1]]
        for j in todo[1:]:
            runs[-1].append(j) if j == runs[-1][-1] + 1 else runs.append([j])
        for run_ in runs:
            if over():
                return False
            if len(run_) > 1:
                # One call clears a run that holds nothing against the stop.
                r = leaf(BLOCK_QUESTION.format(claim=question),
                         doc.text[doc.spans[run_[0]][0]:doc.spans[run_[-1]][1]], list(LEAF))
                calls += 1
                if r.get("label") == "unrelated":
                    for j in run_:
                        table.add(Row(j, "unrelated", r["p"]["unrelated"], "ok", "block"))
                    continue
                if r.get("label") == want:
                    continue
                if r.get("label") is None:
                    return False
            check(doc, run_, question, leaf, table)
            calls += len(run_)
        return not any(table.rows[j].status != "ok" or table.rows[j].label == against
                       for j in near if j in table.rows)

    def stopped():
        """The early stop's verdict once a line read since the last try
        settles the claim, else None."""
        # A component has its own checks and judge, so auto stops early only on a scan.
        mode = plan if plan in stop else stop_at if plan == "auto" and graph is None and stop_at in stop else None
        if mode is None:
            return None
        want, label, verdict, basis = stop[mode]
        new = [i for i in table.spans(want) if i not in tried]
        if not new:
            return None
        tried.update(new)
        if plan == "auto" and not closed(new, want):
            return None
        red = reduce(doc, table, plan, question, allowed)
        r, kept, _ = judge(red)
        if r and r.get("label") == label and set(new) & set(kept):
            return done(verdict, label, r["p"], basis, red=red, judged=kept)
        return None

    def lines(ids):
        """Leaf-check those lines. Returns a verdict when an early stop
        fires, True when the deadline cut it short, else None."""
        nonlocal calls
        for at in range(0, len(ids), max(1, jobs)):
            if over():
                return True
            batch = ids[at:at + max(1, jobs)]
            check(doc, batch, question, leaf, table, jobs)
            calls += len(batch)
            if progress:
                progress(len(table.rows), len(doc.spans), "lines")
            out = stopped()
            if out is not None:
                return out

    # Lines the question's words point at go first, one by one, so a witness
    # is met in the first calls. The rest goes line by line, or by passage.
    head = order[:len([i for i in first if i in set(allowed)]) + HEAD] if block else order
    rest = sorted(set(order) - set(head))
    out = lines(head)
    def passage(blk):
        """Read a run of spans by narrowing. One call over the whole run:
        unrelated dismisses it; related keeps it as one passage, since a
        passage that decides nothing by itself has nothing for the judge and
        reading its lines one by one is where a loose claim turns a scan into
        thousands of calls; fits or breaks halves it and narrows each half,
        down to lines. Returns what lines() returns."""
        nonlocal calls, block_calls, by_block, related_blocks
        if len(blk) == 1:
            return lines(blk)
        if over():
            return True
        r = leaf(BLOCK_QUESTION.format(claim=question), doc.text[doc.spans[blk[0]][0]:doc.spans[blk[-1]][1]], list(LEAF))
        calls, block_calls = calls + 1, block_calls + 1
        label = r.get("label")
        if label in (None, "unrelated", "related"):
            for i in blk:
                table.add(Row(i, label, r["p"][label] if label else None,
                              "ok" if label else "deadline" if cut_short(r) else str(r.get("unscored", "failed")),
                              "block"))
            by_block += len(blk) if label == "unrelated" else 0
            related_blocks += 1 if label == "related" else 0
            if progress:
                progress(len(table.rows), len(doc.spans), "lines")
            return None
        mid = min(range(1, len(blk)), key=lambda k: abs((doc.spans[blk[k]][0] - doc.spans[blk[0]][0])
                                                        - (doc.spans[blk[-1]][1] - doc.spans[blk[k]][0])))
        flagged = set(table.spans(label))
        for half in (blk[:mid], blk[mid:]):
            out = passage(half)
            if out is not None:
                return out
        if set(table.spans(label)) - flagged:
            return None
        # No part says it alone: man bash wraps set -e over lines that, read
        # one by one, are related at most. The smallest part that says it is
        # kept whole, bar lines read the other way or not read.
        for i in blk:
            row = table.rows.get(i)
            if row is None or row.status == "ok" and row.label in ("related", "unrelated"):
                table.add(Row(i, label, r["p"][label], "ok", "block"))
        return stopped()

    # Passages holding the best keyword matches go first, so a witness is met
    # early; the rest keep document order. Coverage is the same either way.
    rank = {i: k for k, i in enumerate(ranked)}
    for blk in sorted(blocks(doc, rest, block), key=lambda b: min(rank.get(i, len(rank)) for i in b)) \
            if out is None else []:
        out = passage(blk)
        if out is not None:
            break
    if isinstance(out, dict):
        return out
    late = out is True or any(table.rows[i].status == "deadline" for i in table.failed())
    if not late:
        # "That payment was reversed", read alone, bears on no claim, and the
        # line it reverses then reaches the judge without it. So every line
        # near a flagged one that points back is asked what it points at, and
        # a line and its referent go to the judge together when either one was
        # flagged. The links are the graph's: saved, and the same for any claim.
        g = graph if graph is not None else Graph(doc)
        flagged, before = set(table.spans("fits", "breaks", "related", via="line")), g.calls
        todo = sorted(flagged, reverse=True)
        while todo and not g.cut:
            i = todo.pop()
            near = [j for j in range(i, min(len(doc.spans), i + g.window + 1))
                    if j in table.rows and g.points_back(j)]
            for j in near:
                back = g.link(j, leaf, over)
                if g.links.get(j, (None, None, "ok"))[2] != "ok":
                    link_failed.append(g.links[j][2])
                for one, other in ((j, back), (back, j)):
                    row = table.rows.get(other)
                    if one in flagged and other not in flagged and row is not None and row.status == "ok":
                        table.add(Row(other, "related", row.p, "ok", "link"))
                        flagged.add(other)
                        todo.append(other)
        calls, late = calls + g.calls - before, g.cut

    red = reduce(doc, table, plan, question, allowed)
    if late or over():
        return done("insufficient", why=f"deadline passed with {len(table.rows) - len(table.failed())} of "
                    f"{len(doc.spans)} lines read", red=red)
    if table.failed():
        return done("unscored", why=f"{len(table.failed())} leaf calls failed: {table.rows[table.failed()[0]].status}",
                    red=red)
    if link_failed:
        return done("unscored", why=f"{len(link_failed)} link calls failed: {link_failed[0]}", red=red)
    if red.closure:
        return done("insufficient", why=f"the identifier chain reaches {len(red.closure)} unread lines", red=red)
    if component is not None:
        # The caller asked for an answer from the lines the graph links to the
        # question. The judge is not told they are all there is, and the basis
        # says so.
        if not component[0]:
            return done("insufficient", why="the question names nothing the text names", red=red)
        if component[1]["failed"]:
            return done("unscored", why=f"{component[1]['failed']} link calls failed", red=red)
        if red.conflict:
            return done("insufficient", why="two lines on the latest date disagree", red=red)
        r, kept, cut = judge(red)
        if r is None or cut:
            return done("insufficient", why="the linked lines do not fit the judge's budget or the deadline passed",
                        red=red, judged=kept)
        if cut_short(r):
            return done("insufficient", why="deadline passed during the judge call", red=red)
        if r.get("label") is None:
            return done("unscored", why=f"judge call failed: {r.get('unscored')}", red=red)
        verdict = {"yes": "supported", "no": "contradicted"}.get(r["label"], "insufficient")
        return done(verdict, r["label"], r["p"], "component" if verdict != "insufficient" else "none",
                    "" if verdict != "insufficient" else "the linked lines do not settle it", red, kept)
    if len(table.rows) < len(doc.spans):
        return done("insufficient", why=f"{len(table.rows)} of {len(doc.spans)} lines read: only some lines "
                    "were allowed", red=red)
    if not doc.complete:
        return done("insufficient", why="the text may still grow", red=red)
    if red.conflict:
        return done("insufficient", why="two lines on the latest date disagree", red=red)
    # Every span was checked, so the judge may be told so: without it a claim
    # about all of something cannot be confirmed from the lines alone. An early
    # stop never says it.
    red = red._replace(facts=red.facts + ["No other line of the text bears on the claim."
                                          if not related_blocks else
                                          "Other passages of the text relate to the claim without settling it."])
    if plan == "all":
        # Every case fits when no line breaks the claim, so that is what the
        # judge is asked: shown two paid invoices and told there are no other
        # lines, gemma4:12b still answered unknown to "every invoice is paid".
        if not red.counts["fits"]:
            return done("insufficient", why="no line fits the claim", red=red)
        r, kept, cut = judge(red, ALL_QUESTION)
        if r and r.get("label"):
            flip = {"yes": "no", "no": "yes", "unknown": "unknown"}
            r = dict(r, label=flip[r["label"]], p={flip[l]: v for l, v in r["p"].items()})
    else:
        r, kept, cut = judge(red)
        if (plan == "auto" and INVERT and r and r.get("label") == "unknown" and not cut and red.counts["fits"]
                and not red.counts["breaks"]):
            # A full read with fitting lines and none breaking: a claim about
            # every case holds if none breaks it, which the judge answers
            # where it would not confirm the claim itself.
            r2, kept2, cut2 = judge(red, ALL_QUESTION)
            if r2 and r2.get("label") == "no":
                flip = {"yes": "no", "no": "yes", "unknown": "unknown"}
                r, kept, cut = dict(r2, label="yes", p={flip[l]: v for l, v in r2["p"].items()}), kept2, cut2
    if r is None and over():
        return done("insufficient", why="deadline passed before the judge call", red=red)
    if cut:
        # Nothing is judged on a part of the flagged lines: the missing ones
        # could reverse it.
        return done("insufficient", why=f"{len(red.support) - len(kept)} flagged lines and the facts do not fit "
                    f"the judge's budget of {budget} chars", red=red, judged=kept)
    if cut_short(r):
        return done("insufficient", why="deadline passed during the judge call", red=red)
    if r.get("label") is None:
        return done("unscored", why=f"judge call failed: {r.get('unscored')}", red=red)
    verdict = {"yes": "supported", "no": "contradicted"}.get(r["label"], "insufficient")
    return done(verdict, r["label"], r["p"], "complete" if verdict != "insufficient" else "none",
                "" if verdict != "insufficient" else "the judge found the lines do not settle it", red, kept)


class Graph:
    """Links between the lines of one document that hold whatever the
    question: which lines name the same thing, and which earlier line a line
    points back to. Naming is code. Pointing back is one call per line, made
    only for lines near the ones a question reaches, and it does not mention
    the question, so a saved link serves every later question. Pass an ask
    wrapped by Saved to keep links: the key is the whole line, the whole
    lines offered and the model, so editing any of them asks again.

    Anchors are identifiers (Doc.ids) and names: two or more capitalised
    words in a row, or one that does not open the line. A line points back
    when it has a word ANAPHOR matches, whether or not it also has an anchor
    of its own. What this cannot reach: a line with neither anchor nor such
    a word, a referent more than `window` lines back, and lines sharing only
    a name that more than HUB lines carry (the question's own names are always
    followed). So a component is where to look, never proof nothing else
    matters."""

    def __init__(self, doc, window=WINDOW):
        self.doc, self.window = doc, max(1, min(window, 9))
        self.by_anchor = collections.defaultdict(list)
        self.anchors = []
        for i in range(len(doc.spans)):
            found = list(dict.fromkeys(a.casefold() for a in doc.ids(i) + self.names(doc.span(i))))
            self.anchors.append(found)
            for a in found:
                self.by_anchor[a].append(i)
        self.links = {}     # span id -> (earlier span id or None, p, "ok" or why the call failed)
        self.cut = False    # the deadline stopped the last component before every link was asked
        self.calls = 0      # link calls made, over every question asked of this graph

    @staticmethod
    def names(text):
        out = []
        for m in NAME.finditer(text.strip()):
            words, opens = m.group().split(), m.start() == 0
            while words and (words[0].casefold() in STOP or ANAPHOR.fullmatch(words[0])):
                words, opens = words[1:], False
            name = " ".join(words).rstrip(".")
            if name and (len(words) > 1 or not opens) and not is_id(name):
                out.append(name)
        return out

    def points_back(self, i):
        return bool(ANAPHOR.search(self.doc.span(i)))

    def link(self, i, ask, over=None):
        """The earlier line span i points back to, or None. One call, kept;
        past the deadline no call starts and nothing is kept."""
        if i not in self.links:
            cands = list(range(max(0, i - self.window), i))
            if not cands:
                self.links[i] = (None, None, "ok")
                return None
            if self.cut or (over and over()):
                self.cut = True
                return None
            labels = [str(k) for k in range(1, len(cands) + 1)] + ["0"]
            text = "\n".join(f"{k}: {self.doc.span(c).strip()}" for k, c in enumerate(cands, 1))
            r = ask(LINK_QUESTION, f"{text}\nLast line: {self.doc.span(i).strip()}", labels)
            self.calls += 1
            if cut_short(r):
                self.cut = True
                return None
            if r.get("label") is None:
                self.links[i] = (None, None, str(r.get("unscored", "failed")))
            else:
                self.links[i] = (None if r["label"] == "0" else cands[int(r["label"]) - 1], r["p"][r["label"]], "ok")
        return self.links[i][0]

    def seeds(self, question):
        """The anchors a question names: its identifiers and names."""
        return list(dict.fromkeys(a.casefold() for a in [t for t in terms(question) if is_id(t)]
                                  + self.names(question)))

    def cost(self, question):
        """(link calls the first hop of question's component would make at
        most, lines naming its anchors), counted in code without a call: each
        line naming an anchor, and each within the window after one, that
        points back and is not linked yet."""
        lines = {i for a in self.seeds(question) for i in self.by_anchor.get(a, [])}
        near = {j for i in lines for j in range(i, min(len(self.doc.spans), i + self.window + 1))}
        return sum(self.points_back(j) and j not in self.links for j in near), len(lines)

    def component(self, question, ask, hops=HOPS, over=None):
        """Span ids a question's anchors reach within `hops` steps: lines
        that name them, lines that share a name with those, the line each
        points back to and the lines pointing back at them. over() true means
        no further call may start. Returns (ids, report); the report counts
        this question's links only, and pending is how many linked lines lay
        one step past the last hop."""
        seeds = self.seeds(question)
        comp, frontier, seen, mine, before, hubs = [], [], set(seeds), set(), self.calls, []
        self.cut = False
        for a in seeds:
            frontier.extend(self.by_anchor.get(a, []))
        for _ in range(hops + 1):
            new = [i for i in dict.fromkeys(frontier) if i not in comp]
            frontier = []
            if not new:
                break
            comp.extend(new)
            for i in new:
                for a in self.anchors[i]:
                    if a not in seen:
                        seen.add(a)
                        if len(self.by_anchor[a]) > HUB:
                            hubs.append(a)
                        else:
                            frontier.extend(self.by_anchor[a])
                if self.points_back(i):
                    mine.add(i)
                    back = self.link(i, ask, over)
                    if back is not None:
                        frontier.append(back)
                for j in range(i + 1, min(len(self.doc.spans), i + self.window + 1)):
                    if j not in comp and self.points_back(j):
                        mine.add(j)
                        if self.link(j, ask, over) in comp:
                            frontier.append(j)
        got = [self.links[i] for i in mine if i in self.links]
        return sorted(comp), {"anchors": sorted((seen & set(self.by_anchor)) - set(hubs)), "hubs": sorted(hubs),
                              "component": len(comp),
                              "window": self.window, "hops": hops, "link_calls": self.calls - before,
                              "linked": len(got), "resolved": sum(v[0] is not None for v in got),
                              "unresolved": sum(v[0] is None and v[2] == "ok" for v in got),
                              "failed": sum(v[2] != "ok" for v in got), "cut": self.cut,
                              "pending": len(set(frontier) - set(comp))}


def run_enum(doc, question, names, ask, deadline=None, budget=3 * BUDGET, block=3000, leaf=None, share=False,
             progress=None):
    """Pick one of the named options for a question about a document, or
    none. ask(question, text, labels, options) is judge.judge's -e shape: the
    labels are digits, 0 is "none of these". Every passage of block chars is
    asked the question; the ones that answer none are dismissed; one final
    call reads the rest. When they do not all fit the budget the result is
    insufficient: nine old passages saying paid would outvote the one that
    reverses it. share=True is the caller saying the question is about what
    most of the text is; then the judge reads the surest passages and its
    answer stands if it is also the option the passages voted for. Returns
    {verdict, label, p, read}: verdict answered, none, insufficient or
    unscored."""
    if not 1 <= len(names) <= 9:
        raise ValueError(f"need 1 to 9 options, got {len(names)}")
    t0, leaf, calls = time.monotonic(), leaf or ask, 0
    labels, options = [str(i) for i in range(1, len(names) + 1)] + ["0"], list(names) + ["none of these"]
    show = dict(zip(labels, list(names) + ["none"]))
    blks = blocks(doc, range(len(doc.spans)), block)
    read = {"doc_key": doc.key, "unit": doc.unit, "spans": len(doc.spans), "passages": len(blks),
            "doc_complete": doc.complete}
    rows, failed, late = [], [], False

    def text(blk):
        return doc.text[doc.spans[blk[0]][0]:doc.spans[blk[-1]][1]]

    def done(verdict, label=None, p=None, basis="none", why="", judged=()):
        votes = {n: 0.0 for n in names}
        for _, r in rows:
            if r["label"] != "0":
                for l, n in zip(labels, names):
                    votes[n] += r["p"][l]
        read.update(checked=len(rows), failed=len(failed), flagged=sum(r["label"] != "0" for _, r in rows),
                    votes={n: round(v, 3) for n, v in votes.items()}, basis=basis,
                    sources=[{"span": [b[0], b[-1]], "start": doc.spans[b[0]][0], "end": doc.spans[b[-1]][1]}
                             for b in sorted(judged)],
                    # The judged passages that themselves voted for the answer.
                    evidence=evidence(doc, [i for b, r in rows if b in judged and label not in (None, "none")
                                            and show.get(r["label"]) == label for i in b]),
                    why=why, calls=calls, ms=round((time.monotonic() - t0) * 1000))
        return {"verdict": verdict, "label": label, "p": p, "read": read}

    for blk in blks:
        if deadline is not None and time.monotonic() - t0 > deadline:
            late = True
            break
        r = leaf(ENUM_QUESTION.format(question=question), text(blk), labels, options)
        calls += 1
        if cut_short(r):
            late = True
            break
        (failed if r.get("label") is None else rows).append((blk, r))
        if progress:
            progress(len(rows) + len(failed), len(blks), "passages")
    if late or (deadline is not None and time.monotonic() - t0 > deadline):
        return done("insufficient", why=f"deadline passed with {len(rows)} of {len(blks)} passages read")
    if failed:
        return done("unscored", why=f"{len(failed)} passage calls failed: {failed[0][1].get('unscored')}")
    if not doc.complete:
        return done("insufficient", why="the text may still grow")
    flagged = sorted((x for x in rows if x[1]["label"] != "0"), key=lambda x: -max(x[1]["p"][l] for l in labels[:-1]))
    if not flagged:
        return done("none", "none", basis="complete", why="no passage answers the question")
    kept, used = [], 0
    for blk, _ in flagged:
        if used + len(text(blk)) + 1 > budget:
            break
        kept.append(blk)
        used += len(text(blk)) + 1
    if not kept or (len(kept) < len(flagged) and not share):
        return done("insufficient", why=f"{len(flagged) - len(kept)} of {len(flagged)} flagged passages do not fit "
                    f"the judge's budget of {budget} chars")
    if deadline is not None and time.monotonic() - t0 > deadline:
        return done("insufficient", why="deadline passed before the judge call")
    r = ask(question, "\n".join(text(b) for b in sorted(kept)), labels, options)
    calls += 1
    if cut_short(r):
        return done("insufficient", why="deadline passed during the judge call")
    if r.get("label") is None:
        return done("unscored", why=f"judge call failed: {r.get('unscored')}", judged=kept)
    label, p = show[r["label"]], {show[l]: v for l, v in r["p"].items()}
    if len(kept) < len(flagged):
        res = done("answered", label, p, "vote", judged=kept)
        winner = max(res["read"]["votes"], key=res["read"]["votes"].get)
        if winner != label:
            res.update(verdict="insufficient", label=None)
            res["read"].update(basis="none", why=f"the judge read {len(kept)} of {len(flagged)} flagged passages and "
                               f"said {label}; the passages voted {winner}")
        return res
    return done("none" if label == "none" else "answered", label, p, "complete", judged=kept)


class Saved:
    """Reader results on disk, so a claim asked again, or asked of a text
    that grew, reads only what is new. A row is keyed by the model's tag and
    digest, num_ctx, the exact prompt and the labels, and holds the raw log
    masses: p is tempered at
    lookup, so a recalibration never serves a stale p. A different claim
    shares nothing here; what carries across questions is the link graph."""

    def __init__(self, path, klass):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        self.db, self.klass, self.hits = sqlite3.connect(path, check_same_thread=False), klass, 0
        self.db.execute("create table if not exists leaf (key text primary key, logp text, mass real)")
        self.db.execute("create table if not exists vectors (key text primary key, dim integer, data blob)")

    def vectors(self, key):
        """A text's index vectors, or None. key names the text, the
        embedding model and digest and the passage size."""
        row = self.db.execute("select dim, data from vectors where key = ?", (key,)).fetchone()
        if not row:
            return None
        flat = array.array("f")
        flat.frombytes(row[1])
        return [list(flat[i:i + row[0]]) for i in range(0, len(flat), row[0])]

    def keep(self, key, vectors):
        flat = array.array("f", [x for v in vectors for x in v])
        with self.db:
            self.db.execute("insert or replace into vectors values (?, ?, ?)", (key, len(vectors[0]), flat.tobytes()))

    def key(self, ident, question, text, labels, options=None):
        """ident names what produced a reading: model tag, model digest and
        num_ctx. With the exact prompt and the labels that is everything a
        stored row depends on; temperature 0 and one token are fixed in judge.py."""
        return hashlib.sha256(json.dumps([ident, self.klass.prompt(question, text, labels, options), labels],
                                         sort_keys=True).encode()).hexdigest()

    def wrap(self, judge, model, ident, context=None):
        """context is part of every key but a link's: a reading under one
        policy says nothing under another, a link says the same under any."""
        def ask(question, text, labels, options=None):
            key = self.key(ident if question == LINK_QUESTION or not context else dict(ident, context=context),
                           question, text, labels, options)
            row = self.db.execute("select logp, mass from leaf where key = ?", (key,)).fetchone()
            if row:
                self.hits += 1
                logp = json.loads(row[0])
                p = self.klass.calibrate(logp, self.klass.temperature(model, enum=options is not None) or 1.0)
                return {"label": max(labels, key=lambda l: p[l]), "p": p, "logp": logp, "mass": row[1], "model": model}
            r = judge(question, text, labels, options)
            if r.get("label") is not None:
                with self.db:
                    self.db.execute("insert or replace into leaf values (?, ?, ?)", (key, json.dumps(r["logp"]), r["mass"]))
            return r
        return ask


def load_class():
    """The judge module beside this file, as a module."""
    here = os.path.dirname(os.path.realpath(__file__))
    loader = importlib.machinery.SourceFileLoader("classif_class", os.path.join(here, "judge.py"))
    mod = importlib.util.module_from_spec(importlib.util.spec_from_loader(loader.name, loader))
    loader.exec_module(mod)
    return mod


def answer(klass, text, question, host, model, names=None, shown=None, plan="auto", graph=False,
           share=False, deadline=None, block=3000, jobs=1, reader_model=None, reader_ctx=2048, cache=False,
           context=None, progress=None, evidence=False):
    """What both command lines run. Calls go through klass.judge on one
    host, reader results are saved only when cache is true, and the text is
    judged against a claim (run) or, with names, asked an enum question
    (run_enum); shown is what the model reads for each name. context goes
    before the text on every reader and judge call, never on a link call,
    so links stay the same under any context. A claim the root call reads
    as exists or state goes to the search first: the text's passages are
    indexed once (klass.embed, vectors saved beside the readings) and one
    judge call reads the closest; what it does not settle goes on to the
    graph and the full read. progress(done, total, unit) hears how far a
    read is; evidence asks a search answer for the lines it rests on.
    Returns that result with read.saved added and enum labels given as
    names."""
    doc, t0 = Doc(text), time.monotonic()
    ename, edigest, digest = getattr(klass, "EMBED_MODEL", "embeddinggemma"), None, None

    def left():
        # run starts no call past the deadline; this bounds one in flight.
        return 120 if deadline is None else max(0.2, deadline - (time.monotonic() - t0))

    def bounded(r):
        # With a deadline every call's timeout is the time left, so one that times out was cut by it.
        if r.get("label") is None and deadline is not None and "timed out" in str(r.get("unscored")):
            return dict(r, deadline=True)
        return r

    def ctx(q):
        return None if q in (LINK_QUESTION, KIND_QUESTION) else context

    def ask(q, txt, labels, options=None):
        return bounded(klass.judge(q, txt, labels, options, model=model, host=host, timeout=left(), context=ctx(q)))

    def leaf(q, txt, labels, options=None):
        return bounded(klass.judge(q, txt, labels, options, model=reader_model, host=host, num_ctx=reader_ctx,
                                   timeout=left(), context=ctx(q)))
    reader, saved = (leaf if reader_model else ask), None
    if cache:
        saved = Saved(os.path.join(os.environ.get("XDG_CACHE_HOME") or os.path.expanduser("~/.cache"), "classif",
                                   "mem.sqlite"), klass)
        name = reader_model or model
        try:
            with urllib.request.urlopen(f"http://{host}/api/tags", timeout=min(5, left())) as fh:
                tags = {m["name"]: m.get("digest") for m in json.load(fh)["models"]}
            digest, edigest = tags.get(name), tags.get(ename) or tags.get(f"{ename}:latest")
        except (OSError, ValueError, KeyError, TypeError):
            pass
        if digest:
            reader = saved.wrap(reader, name, {"model": name, "digest": digest,
                                               "num_ctx": reader_ctx if reader_model else klass.NUM_CTX},
                                context=context)
        else:
            saved = None    # an unidentified model's readings are not saved
    if deadline is not None:
        # The lookup above spent part of it; run and run_enum count from their own start.
        deadline = max(0.0, deadline - (time.monotonic() - t0))
        t0 = time.monotonic()
    if names:
        r = run_enum(doc, question, shown, ask, deadline=deadline, block=block or 3000, leaf=reader, share=share,
                     progress=progress)
        back = dict(zip(shown, names), none="none")
        r["label"] = back.get(r["label"])
        r["p"] = {back[k]: v for k, v in r["p"].items()} if r["p"] else None
        r["read"]["votes"] = {back[k]: v for k, v in r["read"]["votes"].items()}
    else:
        r, tried, picked = None, None, None
        if plan == "auto" and not graph:
            t1 = time.monotonic()
            # No call starts past the deadline, the root call included, and its time is the read's.
            picked = kind(question, ask) if deadline is None or deadline > 0 else (None, None, {})
            if deadline is not None:
                deadline = max(0.0, deadline - (time.monotonic() - t1))
            if picked[0] == "whole":
                r = whole(doc, question, ask)
        top = KINDS.get(picked[2].get("label"), (None,))[0] if picked and picked[2] else None
        # The search: index once, judge the closest passages in one call. The
        # root call's top pick names the mode even when it was unsure, and an
        # unsure pick of another kind searches as exists: a witness settles a
        # claim whatever its kind, and only a no needs the kind right.
        mode = top if top in ("exists", "state") else "exists" if top and picked[0] is None else None
        searched = None
        if r is None and plan == "auto" and not graph and mode and left() > 0.2:
            t1, idx, missing, kept = time.monotonic(), Index(doc), None, None
            key = hashlib.sha256(json.dumps([doc.key, ename, edigest, CHUNK]).encode()).hexdigest()
            if saved and edigest:
                kept = saved.vectors(key)
            if kept is not None:
                idx.vectors = kept
            elif getattr(klass, "embed", None) and (edigest or digest is None):
                missing = idx.embed(lambda ts: klass.embed(ts, model=ename, host=host, timeout=left()),
                                    progress=progress, over=lambda: left() <= 0.2)
                if idx.vectors is not None and saved and edigest:
                    saved.keep(key, idx.vectors)
            else:
                missing = {"unscored": f"no embedding model {ename} on {host}", "missing": True}
            index = {"ms": round((time.monotonic() - t1) * 1000), "saved": kept is not None, "tokens": idx.tokens,
                     "passages": len(idx.blks)}
            if missing:
                index.update(why=missing.get("unscored"), missing=bool(missing.get("missing")))
            qvec = None
            if idx.vectors is not None:
                qvec = (klass.embed([question], model=ename, host=host, timeout=left(), query=True).get("vectors")
                        or [None])[0]
            searched = search(doc, idx, question, ask, mode=mode, qvec=qvec)
            searched["read"]["index"] = index
            if deadline is not None:
                deadline = max(0.0, deadline - (time.monotonic() - t1))
            # A yes stands: the judge saw a witness. A no stands for an exists
            # claim when the passages were the closest by meaning, and for a
            # state claim when the lines about it said no.
            if searched["verdict"] == "supported" or (searched["verdict"] == "contradicted" and (
                    (top == "exists" and searched["read"]["by"] == "meaning and words") or top == "state")):
                r, searched = searched, None
                if evidence:
                    lines_ = run(doc, question, ask, "auto", only=r["read"]["judged"], deadline=deadline, jobs=jobs,
                                 leaf=reader, block=CHUNK, stop_at=top, progress=progress)
                    r["read"]["evidence"] = lines_["read"].get("evidence", [])
                    r["read"]["lines"] = {k: lines_["read"].get(k) for k in ("verdict", "why", "calls", "ms")}
                    r["read"]["lines"]["verdict"] = lines_["verdict"]
                    r["read"]["calls"] += lines_["read"]["calls"]
                    if deadline is not None:
                        deadline = max(0.0, deadline - (time.monotonic() - t1))
        if r is None and plan == "auto" and not graph and picked[0] not in ("exists", "all"):
            # The lines linked to what the claim names come first: a confident
            # answer from them stands, anything else falls back to the full read.
            # A name on hundreds of lines (a novel's heroine) would take more
            # link calls than reading every passage, so the full read goes first.
            t1, g = time.monotonic(), Graph(doc)
            cost, named = g.cost(question)
            passages = math.ceil(len(doc.text) / (block or 3000))
            if cost > passages:
                tried = {"read": {"why": f"graph skipped: the {named} lines naming {', '.join(g.seeds(question))} "
                                         f"would take {cost} link calls, more than the {passages} passages of a "
                                         "full read", "calls": 0, "ms": 0}}
            else:
                tried = run(doc, question, ask, plan, deadline=deadline, jobs=jobs, leaf=reader, block=0, graph=g,
                            progress=progress)
            if tried.get("verdict") in ("supported", "contradicted"):
                r, tried = tried, None
            elif deadline is not None:
                deadline = max(0.0, deadline - (time.monotonic() - t1))
        if r is None:
            r = run(doc, question, ask, plan, deadline=deadline, jobs=jobs, leaf=reader, block=0 if graph else block,
                    graph=Graph(doc) if graph else None, stop_at=picked[0] if picked else None, progress=progress)
        if tried is not None:
            r["read"]["tried"] = {k: tried["read"].get(k) for k in ("why", "calls", "ms")}
            r["read"]["calls"] += tried["read"]["calls"]
        if searched is not None:
            r["read"]["search"] = {k: searched["read"].get(k) for k in ("why", "calls", "ms", "index")}
            r["read"]["search"]["verdict"] = searched["verdict"]
            r["read"]["calls"] += searched["read"]["calls"]
        if picked:
            r["read"]["kind"] = {"kind": picked[0], "p": picked[1]}
            r["read"]["calls"] += 1
    r["read"]["saved"] = saved.hits if saved else 0
    return r


def report(read):
    """One line on what a result rests on, for stderr."""
    why = f"; {read['why']}" if read["why"] else ""
    if read.get("plan") == "search":
        idx = read.get("index") or {}
        built = f"index saved" if idx.get("saved") else f"index {idx.get('ms', 0)} ms"
        return (f"search: judged the {read['checked']} of {read['passages']} passages closest to the question by "
                f"{read['by']} ({built}), {read['calls']} calls, {read['ms']} ms" + why)
    if read.get("plan") == "whole":
        return (f"whole text: read {read['checked']}/{read['spans']} lines sampled and {len(read['facts'])} counted "
                f"facts, {read['calls'] + 1} calls, {read['ms']} ms" + why)
    if "passages" in read:
        return (f"enum: read {read['checked']}/{read['passages']} passages ({read['flagged']} answer, "
                f"{read['saved']} saved), judged {len(read['sources'])}, {read['calls']} calls, {read['ms']} ms" + why)
    c = read["counts"]
    return (f"{read['plan']}: read {read['checked']}/{read['spans']} lines ({c['fits']} fit, {c['breaks']} break, "
            f"{c['related']} related; {read['by_block']} dismissed in {read['blocks']} passages, {read['saved']} saved), "
            f"judged {len(read['judged'])}, {read['calls']} calls, {read['ms']} ms"
            + (f"; graph: {read['graph']['component']} linked lines, {read['graph']['link_calls']} link calls, not a "
               "full read" if "graph" in read else "") + why)


def main():
    ap = argparse.ArgumentParser(prog="mem.py", description=__doc__.split("\n\n")[0])
    ap.add_argument("claim", help="a statement the text supports or contradicts; with -e, a question")
    ap.add_argument("file", nargs="?", default="-", help="the text; stdin when omitted or -")
    ap.add_argument("-P", "--plan", default="auto", choices=PLANS,
                    help="pin a plan; exists and all may stop early (default auto)")
    ap.add_argument("-e", "--enum", help="answer a question with one of 1 to 9 comma-separated options, or none; "
                                         "name=description shows the model what an option means")
    ap.add_argument("-g", "--graph", action="store_true",
                    help="answer from the lines linked to what the claim names instead of scanning; links are saved "
                         "and serve later claims")
    ap.add_argument("--share", action="store_true",
                    help="with -e: the question is about what most of the text is, so a majority of passages may answer")
    ap.add_argument("-m", "--model", help="judge model (judge.py's default when omitted)")
    ap.add_argument("-r", "--reader", help="model for the per-line checks (default: the judge model)")
    ap.add_argument("--reader-ctx", type=int, default=2048, help="num_ctx for a separate reader (default 2048)")
    # Four at once bought nothing here: Ollama answered them one after another
    # (0.18, 0.35, 0.52, 0.71 s) and the first batch stalled 13 s.
    ap.add_argument("-n", "--jobs", type=int, default=1, help="leaf calls in flight (default 1)")
    ap.add_argument("-b", "--block", type=int, default=3000,
                    help="read passages of this many chars first, lines only inside a flagged one; 0 reads every "
                         "line (default 3000)")
    ap.add_argument("-d", "--deadline", type=float,
                    help="seconds; no call starts past it, one in flight is cut, and the verdict is insufficient")
    ap.add_argument("-j", "--json", action="store_true", help="print the full result as JSON")
    ap.add_argument("--cache", action="store_true",
                    help="read and write saved reader results, links and index vectors under $XDG_CACHE_HOME/classif; "
                         "nothing is written without it")
    a = ap.parse_args()
    if a.graph and a.enum:
        ap.error("-g answers claims; -e reads every passage")
    if a.share and not a.enum:
        ap.error("--share needs -e")
    raw = sys.stdin.buffer.read() if a.file == "-" else open(a.file, "rb").read()
    klass = load_class()
    host, model = klass.route(a.model)
    if not host:
        print("mem: unscored: no Ollama host answered", file=sys.stderr)
        return 2
    names = shown = None
    if a.enum:
        opts = [o.partition("=") for o in a.enum.split("\n" if "\n" in a.enum else ",") if o.strip()]
        names = [n.strip() for n, _, _ in opts]
        if not 1 <= len(names) <= 9 or len({n.lower() for n in names}) != len(names):
            ap.error(f"-e needs 1 to 9 distinct options, got {len(names)}")
        shown = [f"{n} ({d.strip()})" if d.strip() else n for n, (_, _, d) in zip(names, opts)]
    r = answer(klass, raw.decode("utf-8", errors="replace"), a.claim, host, model, names, shown, a.plan, a.graph,
               a.share, a.deadline, a.block, a.jobs, a.reader, a.reader_ctx, a.cache)
    read, p = r["read"], (r["p"] or {}).get(r["label"])
    read["file"] = a.file
    if a.json:
        print(json.dumps(r))
    elif a.enum:
        print((r["label"] or r["verdict"]) + (f" {p:.2f}" if p is not None else ""))
    else:
        print(r["verdict"] + (f" {p:.2f}" if p is not None else ""))
    print("mem: " + report(read), file=sys.stderr)
    if a.enum:
        if r["verdict"] in ("unscored", "insufficient"):
            return 2 if r["verdict"] == "unscored" else 3
        return 0 if r["label"] == names[0] else 1
    return {"supported": 0, "contradicted": 1, "unscored": 2}.get(r["verdict"], 3)


if __name__ == "__main__":
    sys.exit(main())
