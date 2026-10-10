# Reference

See the [README](../README.md) for the quick start and [architecture](architecture.md) for how a question is answered.

## Public command

`classif QUESTION [TEXT]` scores one question. Without `TEXT`, it reads stdin; at a terminal with no piped input, it scores the question alone. With no arguments at all it is a usage error, exit 2. `-i FILE` reads a file and records its name. A positional `TEXT` remains literal text even if it names an existing file.

Short inputs use one whole-input call. A server-confirmed context overflow switches default yes/no/unknown questions and enums to the external-memory executor. Connection failures, other HTTP errors and inadequate label mass stay unscored.

`classif specs` lists the saved classifier definitions, the fields each wants and the file it is read from. `classif classify SPEC field=value ...` and `classif smoke SPEC` use them. Other integration subcommands remain available through `classif --help`. If a question equals a subcommand word, put it after `--`, for example `classif -- unload "source text"`.

| Flag | Meaning |
| --- | --- |
| `-e OPTION` | An answer to pick: a name, or `name=description`. The model reads the description as when to pick that option; only the name is printed. Repeat `-e` for each option, or list names with commas or newlines. Any number of options, plus `none`; the first exits 0. Past nine, each option is first asked alone (`QUESTION Is the answer NAME (description)?`, yes or no), and the three likeliest, in the order given, go to one pick; see [past nine options](#past-nine-options). A description runs to the next `name=`, so it may hold commas. Unquoted, its words run to the next option or quoted value, so a one-word text goes before `-e`. Without `-e` the answers are `yes`, `no` and `unknown`. |
| `-p` | Print the input unchanged when the first label wins. |
| `-j` | Print scores and request metadata as JSON. Pipe through `python3 -m json.tool` to indent it. |
| `-t P` | Exit 3 and mark the result unsure when the winning probability is below P. A pipe gate then passes nothing. |
| `-i FILE[,FILE...]` | Read the input from these files, in order, as one text; repeatable. A name holding a comma gets its own `-i`. `read.files` names each file's range when there are several. |
| `-c TEXT` | Background the question needs, a policy, a reference or the date: a file, `<(cmd)`, or the text itself. A value that reads as a file name (one word with a slash or an extension) but names no file is refused, so a typo never becomes the context. With no input, `-c` is the text judged, so `-c "$(cmd)"` alone works like `cmd \| classif`. "No input" means stdin is a terminal or `/dev/null`: a script, hotkey or editor plugin that starts classif with an open pipe on stdin leaves it waiting for that pipe, so end such a call with `</dev/null` or pipe the text in. With an input it is read before the input on every reader and judge call, never on a link call. Must fit the window with the input or with one passage. On a short input it is the same decision as putting the file first in `-i`. |
| `--cache` | Save readings, links and the passage index under `$XDG_CACHE_HOME/classif/mem.sqlite` (`~/.cache/classif/mem.sqlite`) and reuse them on a repeat of the same text, claim or model. Without it nothing is written to disk and every run starts cold. |
| `-d SECONDS` | Limit the whole-input attempt, the model lookup and the bounded reads together; must be above 0. |
| `-w` | Print the lines the answer rests on, as `  LINE: text` (`  FILE:LINE: text` with `-i`) after the verdict, or on stderr under `-p`. The text goes to the reader even when it fits the window, so a short text takes longer, and the answer can differ from the one whole-text call: it may be `insufficient` when no line settles the question, which a `-p` gate does not pass. A yes or no rests on the lines that, read one at a time, fit or break the claim among those it was judged from, including an answer from a search or a sample. With `-e`, each passage that voted for the answer is narrowed to a run that gives that answer alone: the first half that still gives it is halved again, and a passage that gives it only as a whole is printed whole. When no line reads for or against the answer, that is printed instead; `read.lines` reports the extra calls. Memory results always carry the lines in `read.evidence` as 1-based `line`, `end` and `text`, plus `file` with `-i`, its line numbers that file's own. |

Exit 0 means the first label won; exit 1 means another label won. Exit 2 means unscored, including connection failure, inadequate label mass or a context overflow of `-c` alone. Exit 3 means insufficient evidence or coverage, a deadline cut, or an unmet supplied threshold. A memory result that is insufficient has no output label and passes no gate.

The direct path sends one Ollama `/api/chat` request with `num_predict: 1`, `temperature: 0`, `think: false` and `top_logprobs: 20`. It sums label token probabilities, including case and leading-space variants, then normalizes them into `p`. Labels holding less than half the reported probability yield an unscored result. The 32,768-token window is enforced with `truncate: false`. An overflow triggers bounded reads rather than judging a truncated fragment. Without `-d` or `CLASSIF_TIMEOUT`, the whole-input request allows 15 seconds plus one second per 2000 input characters; this is a timeout allowance, not a token-count estimate. The imported `judge()` function remains a single bounded call.

## Several questions, one text

`classif tag NAME=OPTIONS ...` asks every question about one text in a single call, so the text is read once. The text comes from stdin or `-i FILE`, never an argument.

```sh
classif tag 'urgency=today,this week,no deadline' 'kind=asks me,fyi,newsletter' -i mail.txt
urgency  today    0.98
kind     asks me  0.99
```

- `NAME=OPTIONS`: the name is what the model reads, a word or a whole question. Options are split by commas or newlines, 1 to 9 of them, plus `none`. Option descriptions are not supported.
- `-i`, `-c`, `-j`, `-t` and `-d` work as for a question. `-j` keys each question by its name, with `label`, `p`, `confidence` and `mass`, unrounded, and adds the response's `done_reason` and generated `tokens`. A response cut by its token budget (`done_reason: length`) names the cut on each question it left unanswered.
- Exit 0 when every question is answered, 2 when any is unscored, 3 when any answer is under `-t`. A text past the window is unscored: ask one question at a time to read it in pieces.

The call carries a JSON schema in `format`. The grammar writes each name as a key, the model writes a digit after it, and that digit's top 20 logprobs score the question as with `-e`. Answers are matched by key, not by position. `p` is raw: no temperature is fitted for this path, so it sits closer to 0 and 1 than a calibrated `-e` answer.

Each question adds about five generated tokens plus one per token of its name, about 20 ms each on the eval GPU, so short names are cheaper. A separate call reads the whole text again, so `tag` gains as the text grows.

Measured 2026-10-06 on `winnow:12b-q4_K_M` with `evals/tag_eval.py`: two labeled questions per text, `tag` in both orders, each question alone through `-e` with the same wording and option names.

| Case | `tag` | One `-e` call per question |
| --- | ---: | ---: |
| 55 emails, whole questions ("Is this email a newsletter?"), correct | 210/220 | 208/220 |
| 55 emails, short names (`newsletter`, `needs reply or action`), correct | 207/220 | 170/220 |
| 36 authored144 texts, two claims each, option names without descriptions, correct | 137/144 | 138/144 |
| Model ms per text: short names, whole questions, authored claims | 639, 906, 1068 | 945, 886, 360 |
| Wall time, three questions, 150-char mail | 900 ms | 660-800 ms |
| Wall time, three questions, 44K-char document | 2.2 s | 4.4 s |

The second answer held up in every set: 104, 103 and 68 right at position 2 against 103, 107 and 69 at position 1. With short names, `-e` got 37 fewer right than `tag`. Raw `tag` p sat further from the truth than tempered `-e` p on the authored claims (ECE 0.048 against 0.021) and closer on the emails (0.030 against 0.033). The authored claims are long names, so `tag` cost three times the two short-text calls.

Bare digits without keys ("12") let the second answer copy the first: "needs a reply?" asked after "newsletter?" agreed with its own call 16 times in 55, keyed 51. Pretty-printed JSON spent 28 generated tokens on three questions; the prompt asks for one line, which takes 15.

## One question, many items

`classif each QUESTION ITEM ...` asks the question of each item as its own call and prints them best first, so a list of any length is ordered. Items are arguments, lines on stdin (plain text, or JSON `{"name", "text"}`), or files with `-i FILE ...`, one item each, named by the file name.

```sh
classif each "Is this the right next step?" -c goals.md "renew the passport" "Sign up for a 10-week running plan"
p(yes)  answer     item
0.882   yes 0.882  Sign up for a 10-week running plan
0.003   no 0.986   renew the passport
```

- The score is p(yes), or with `-e` p of the first option; the answer column shows what each item got. `-e` takes any number of options plus `none`, as for a question; past nine, each item's options are screened, and its score is the first option's p(yes) from its own call.
- `-c` is read before every item. With no items and no pipe, `-c` is the one item, as for a question.
- `-k N` prints the top N. `-j` prints `{question, options, context, items, unscored, model, host}`, each item with `name`, `score`, `label`, `p` and, under `-w`, `marks`.
- `-w` marks the top pick: each non-blank line of its text (2 to 60 lines), then of `-c` (1 to 60 lines), is left out and the question asked again. Up to three lines whose removal moves the score by 0.005 or more are printed, `+` for a line that raised it. A mark is sensitivity from the same one-call reading as the answer, not the line-by-line read of `--why` on a question.
- Exit 0 when some item was scored, 2 when none could be. An item that cannot be scored, such as one past the window, is left out with a note on stderr and in `-j`'s `unscored`.

Each call is the direct path above with its own timeout; items are judged one after another on one host.

## A score

`-s LEVEL`, repeated, asks for a degree on ordered levels instead of a pick, modeled on [Jev's Score](https://docs.typesafe.ai/primitives/score). Give 2 to 10 levels, low to high, each a situation the model can match rather than a degree word: "Broken, but a workaround exists" places a report, "moderately severe" does not. One call reads the levels as digits 0 to n-1 and answers one; classif reads each level's probability from that token.

```sh
classif "How restrictive is this license for third-party code shipped inside a closed-source paid product?" \
  -i /usr/share/common-licenses/MPL-2.0 -s "Permissive: nothing to keep or share" -s "Notice: keep the copyright notice" \
  -s "File-level copyleft: changes to its files must be shared" -s "Library copyleft: changes to the library must be shared" \
  -s "Strong copyleft: the whole program must be shared" -w
2.06 0.87
where the probability went
  0  0.00  Permissive: nothing to keep or share
  1  0.04  Notice: keep the copyright notice
  2  0.85  File-level copyleft: changes to its files must be shared
  3  0.09  Library copyleft: changes to the library must be shared
  4  0.01  Strong copyleft: the whole program must be shared
why: not marked, its text has under 2 or over 60 lines and there is no -c of 1 to 60
```

- The output is `SCORE CONFIDENCE`. The score is each level's number times its probability, summed, so it runs from 0 to the top level and can land between two. The confidence is Jev's Score formula: 1 less the probability-weighted distance from the likeliest level, over the same for an even spread, floored at 0. Probability on a neighbouring level lowers it less than probability at the far end.
- `-j` carries Jev's answer fields, `type`, `score`, `confidence`, `legend` and `probabilities`, keyed by level number, beside `logp`, `mass`, `T`, `model`, `host` and `ms`.
- `-w` prints where the probability went, one line per level, then the lines of the text (2 to 60 lines) and of `-c` (1 to 60 lines) whose removal moves the score most, as `each -w` marks them.
- `-t C` gates on the confidence: under C the answer prints `unsure` and exits 3. `-p` is refused, since a score has no winner to pass the input on. The input must fit the window in one call.
- `classif each -s` sorts items by score, highest first, in a `score conf item` table; `-j` items carry `score`, `confidence` and `probabilities`, with the levels once as `legend`, and `-w` adds the top pick's levels to its marks.

Jev reads each level on its own. Asked of the 11 licenses in `/usr/share/common-licenses` and classif's own on six restrictiveness levels, reading each level alone put MIT and Apache-2.0 at "permissive" and averaged 0.44 from Jev's scores, in 2 to 8 s a license; the one-call read averaged 0.12 from Jev's in about 2.2 s, and 0.06 without GFDL-1.3, which Jev spreads over three levels (2026-10-10, `winnow:12b-q4_K_M` against `jev-1.13.0`). A score uses the model's `T_score` from `calibration.json`, else its `T_enum`.

## Past nine options

A model picks among a few options well and among many poorly, and option digits past 9 are no longer one token. Past nine `-e` options, classif asks each option alone whether it is the answer, keeps the three with the highest p(yes) in the order given, and asks the question once more with those three and `none`. The text and `-c` come first in every prompt, so after the first call each costs little more than its question: twelve options answered in about 1.3 s on this laptop, against about 0.2 s for one three-option call (2026-10-09, `winnow:12b-q4_K_M`, wall time including startup).

```sh
classif "Which command does this describe?" "stop process 1234" -e ls,cd,grep,find,sed,awk,tar,ssh,curl,chmod,ps,kill
kill 0.98
```

`p` is the final pick's, over the finalists and `none`, so it does not say how the other options fared; `-j` carries each option's p(yes) as `screen`. The exit code is 0 only when the first option given wins. A text past the window and `--why` read in pieces that pick among at most nine options, so past nine both are refused. An option missing from the three finalists cannot win, and the final p still reads high, so a `-t` floor past nine checks only the pick among finalists. No eval set covers more than nine options yet: live checks put six of six short command descriptions on the right one of twelve commands, and seven of eight on the right one of twenty. Options that overlap, such as `billing` and `invoice`, are untested.

## Long-input claims

```text
classif [-d SECONDS] [-j] -i FILE CLAIM
classif -e OPTIONS [-d SECONDS] [-j] -i FILE QUESTION
```

How a long input is read is the reader's choice, not a flag. Past the window one call reads the question alone and picks how much of the text it needs (`read.kind`): a claim one line settles (something exists or happened) or one about a state goes to the search; one about every case reads to a counterexample or every line; a question about the text as a whole is answered in one call from counted facts (`read.facts`) and an even sample (`basis: sample`). The search indexes the text's passages of 1,200 chars once, as vectors from `embeddinggemma` (saved beside the readings, keyed by the text and the model) and as a word index, then judges the passages closest to the question, by words first and then by meaning, about 30,000 chars in one call (`basis: search`, `read.checked` of `read.passages`, `read.index`). A yes stands. A no stands for an exists claim when the passages were the closest by meaning (`read.by`), and for a state claim when the lines about it said no. Without the embedding model the search ranks by words alone: a yes stands, a state no stands, an exists no does not, and the run says what to pull. Anything the search did not settle goes on (`read.search` reports it): a claim that names something the text names is answered from its linked component (`basis: component`), and any other outcome falls back to a complete scan, with `read.tried` reporting the component attempt. With `--why` a search or whole-text answer is followed by a line-by-line read of what it was judged from, for the lines it rests on (`read.lines`). A witness or counterexample stops the scan only once the eight lines after it and the lines sharing its identifiers are read and none reads the other way (`basis: witness` or `counterexample`). Code adds the distinct identifiers and latest date among flagged lines, refuses a latest date carried by both a fitting and a breaking line, and refuses an identifier chain reaching an unread line. The chain follows a name carried by more than 20 lines only when the claim uses it. When a complete scan finds fitting lines, none breaking, and the judge answers unknown, the judge is asked whether any line breaks the claim, and a no becomes yes.

The public command uses 3000-character passages and one reader call at a time. A passage the reader calls fits or breaks is narrowed by halving down to lines, or to the smallest part that still says it when no line does alone; one it calls related is kept as a passage (`read.related_blocks`) and its lines are not packed for the judge; one it calls unrelated is dismissed (`read.by_block`). The compatibility executor `python3 mem.py CLAIM FILE` additionally exposes `-r` for a separate reader, `--reader-ctx N`, `-b CHARS` (0 checks each line) and `-n JOBS`. `-d` sets a deadline, including the remaining time for an in-flight call.

The executor verdicts are supported, contradicted, unscored and insufficient. The public command maps supported to `yes` (exit 0) and contradicted to `no` (1); unscored exits 2 and insufficient exits 3 with a null label. JSON includes `read.doc_key`, `read.file`, `read.unit` and `read.sources`. For unit `char`, each source range is a half-open slice of the decoded input: `text[start:end]`. It also records operational coverage, passage dismissals and judged spans. Coverage does not certify the reader's semantic recall.

For enums past the window, `-e` takes one to nine options, including `name=description`; a text that fits takes any number. Every passage is checked; passages answering `none` are dismissed. The final judge reads the flagged passages. An over-budget set returns insufficient. Enum exits are 0 for the first option, 1 for other options or `none`, 2 for unscored and 3 for insufficient.

Nothing is written to disk without `--cache`. With it, saved reader scores live in `$XDG_CACHE_HOME/classif/mem.sqlite`, or `~/.cache/classif/mem.sqlite`. Keys include the exact prompt, labels, model tag and digest, context size and the `-c` text; a link's key leaves the `-c` text out. Raw log masses are recalibrated at lookup. A model without an identified digest is not cached. Saved readings do not cache the final verdict.

Every scan resolves lines near flagged ones that point back and sends them to the judge with their referent; `read.by_link` counts the lines added that way.

The component of a claim holds the lines linked to what it names. Identifier and name mentions create code links. References are resolved with a fixed question against up to eight preceding spans, so their saved scores serve different claims. A component answer does not imply a full scan. A failed required link or a deadline cut leaves the component unsettled, and the complete scan follows.

The graph reports its window, traversal hops, unresolved links and failures. `read.graph.pending` counts known linked spans one step beyond the last traversal hop; a nonzero value means those spans were not included in the component. Common anchors appearing on more than 20 spans are not followed transitively; explicit question anchors are followed and skipped hubs appear in the report. A reference beyond the candidate window or a statement with no recognized anchor may be missed: a confident component answer stands without the full scan that would meet it.

`read.calls` counts scoring-function invocations, including saved lookups; `read.saved` counts cache hits. Their difference gives fresh model calls. Graph link calls are included in the total. The public result's `ms` covers routing, the whole-input attempt and bounded execution; `read.ms` measures the executor itself.

## JSON output

```json
{
  "label": "yes",
  "mode": "direct",
  "p": {
    "yes": 0.948,
    "no": 0.022,
    "unknown": 0.031
  },
  "confidence": 0.922,
  "T": 4.332,
  "logp": {
    "yes": -0.0,
    "no": -16.357,
    "unknown": -14.872
  },
  "mass": 1.0,
  "model": "gemma4:12b",
  "host": "localhost:11434",
  "ms": 132,
  "p_raw": {
    "yes": 1.0,
    "no": 0.0,
    "unknown": 0.0
  }
}
```

- `confidence` is `(n * pmax - 1) / (n - 1)` for n labels: 0 for a tie and 1 when one label takes all the probability. The scale is the same for 2 labels and 10.
- `logp` is each label's log mass. A label absent from the top 20 receives the lowest reported logprob as an upper bound.
- `T` is the fitted temperature, or `null`. `p_raw` appears only when `T` is set.
- `unsure` appears only with `-t`.
- `context` is the `-c` path, present only when one was given.
- `screen` appears past nine `-e` options: each option's p(yes) from its own call. `p` then covers the three finalists and `none`.

## Calibration

A raw model can assign 0.99 to a wrong answer. If `calibration.json` defines a temperature for the model, `p` becomes `softmax(logp / T)`. A temperature above 1 reduces overconfidence without changing label rank or the winner. It can change whether a result passes `-t`.

Label questions (the default yes/no/unknown, specs and the hidden `-l` the evals ask with) use `T`; option questions (`-e`) use `T_enum`; scores (`-s`) use `T_score`, else `T_enum`. Models and question types without a temperature retain raw probabilities. The supplied `llama3.2:3b` gate threshold uses raw probabilities.

Five-fold held-out results, fitted on 200 labeled `-l` cases (synthetic, RAG chunks, private mail) and on the 144 public `-e` cases:

| Model | Questions | T | NLL | Brier | ECE | Accuracy |
| --- | --- | --- | --- | --- | --- | --- |
| `winnow:12b-q4_K_M` | `-l`, raw | 1 | 0.283 | 0.047 | 0.048 | 0.95 |
| `winnow:12b-q4_K_M` | `-l`, fitted | 2.79 | 0.166 | 0.042 | 0.029 | 0.95 |
| `winnow:12b-q4_K_M` | `-e`, raw | 1 | 0.240 | 0.035 | 0.034 | 0.97 |
| `winnow:12b-q4_K_M` | `-e`, fitted | 1.96 | 0.167 | 0.036 | 0.024 | 0.97 |
| `gemma4:12b` | `-l`, raw | 1 | 0.516 | 0.063 | 0.066 | 0.93 |
| `gemma4:12b` | `-l`, fitted | 4.33 | 0.200 | 0.055 | 0.016 | 0.93 |
| `gemma4:12b` | `-e`, raw | 1 | 0.331 | 0.035 | 0.036 | 0.97 |
| `gemma4:12b` | `-e`, fitted | 2.70 | 0.171 | 0.035 | 0.020 | 0.97 |

Per-fold temperatures ranged from 2.40 to 2.95 for `-l` and 1.81 to 2.10 for `-e` on `winnow:12b-q4_K_M`, and from 3.98 to 4.53 and 2.47 to 2.83 on `gemma4:12b`. On `gemma4:12b`, applying the `-l` temperature to `-e` cases gives ECE 0.095 and Brier 0.050, worse than raw probabilities. Each question type therefore has its own temperature.

These results cover yes/no questions and questions with three described options. They do not evaluate the `unknown` label or other option counts.

To fit a model:

```sh
evals/eval.py MODEL [HOST]          # score every case, save rows with logp
evals/calibrate.py MODEL            # fit T and T_enum, report held-out metrics
evals/calibrate.py MODEL --write    # store them in calibration.json
evals/jev.py MODEL [HOST]           # ask every case of Jev and classif, save rows with logp
evals/calibrate.py MODEL --jev      # fit each kind to Jev's distributions, report held-out TV
evals/calibrate.py MODEL --jev --write   # store T_score
```

`--jev` minimizes the cross-entropy of classif's tempered distribution under Jev's and reports five-fold held-out total variation and cross-entropy at T=1, at the temperature classif applies now, and at the fit. `--write` stores only `T_score`; `T` and `T_enum` stay fitted to labels.

The command reads `calibration.json` beside the real script, including when run through a symlink on PATH.

## Hosts

Ollama hosts come in preference order from `CLASSIF_HOSTS`, comma separated, or else from `~/.config/classif/hosts`, one per line with `#` comments. Each entry is `host:port[=model]`; a host without a model runs the default model, `winnow:12b-q4_K_M` (`DEFAULT_MODEL` in `judge.py`). With neither, the command uses `localhost:11434`. The model is the entry's; `CLASSIF_HOSTS=host:port=MODEL` picks one for a single call.

```
# ~/.config/classif/hosts
localhost:11434
gpu-box.lan:11434=llama3.2:3b
```

A spec that lists `hosts` uses those. Specs come from `CLASSIF_DIR` when set, else `~/.config/classif/specs`.

Connection probes run in parallel. The first host to accept a connection wins, with 0.3s extra for higher-priority hosts after the first response.

| Variable | Meaning |
| --- | --- |
| `CLASSIF_HOSTS` | Comma list of `host:port[=model]`; overrides the hosts file. |
| `CLASSIF_DIR` | The only spec directory, when set. |
| `CLASSIF_TIMEOUT` | Seconds for one model call. Unset, a whole read gets 15 s plus 1 s per 2000 chars; `-d` overrides both. |
| `CLASSIF_KEEP_ALIVE` | How long Ollama keeps the model loaded after a call: a duration (`2h`) or seconds, `-1` until `classif unload` or Ollama drops it. Unset, the first line of `~/.config/classif/keep_alive`, else `30m`. The file reaches callers started before it changed, such as hooks. |
| `CLASSIF_CALIBRATION` | Another calibration file. |

## Tests and evals

```sh
python3 -m unittest discover -s tests   # fake Ollama and injected readers; no model needed
python3 -m unittest discover -s tests/e2e   # the real command, replayed from tests/e2e/cassette.json; no model
CLASSIF_E2E=record python3 -m unittest discover -s tests/e2e   # record it again against the routed model
evals/eval.py MODEL [HOST]              # accuracy, Brier, latency per task
evals/tag_eval.py MODEL [HOST]          # tag against one -e call per question, per answer position
evals/jev.py MODEL [HOST]               # distance from Jev's answers on the same cases; needs TYPESAFE_API_KEY
evals/jev.py --diff A.jsonl B.jsonl     # run B against run A, paired by case, with a 95% interval
```

`evals/jev.py` asks every case of Jev and of classif as the same primitive: a synthetic yes/no case as a Noul, the other synthetic and the authored cases as a Choice, and the 41 score cases in `evals/cases/score.jsonl` (six domains: bug severity, customer frustration, candidate fit, change risk, email urgency, review comments, each with the level its author expected) and the license cases (when `/usr/share/common-licenses` exists) as a Score. Jev is the reference, so a prompt, quantization or fine-tune can be judged by whether it moves classif's distributions closer to Jev's. Per case it reports the total variation between the two distributions (half the summed absolute gap, 0 the same and 1 disjoint) and whether the top answer agrees; for a score, also the gap and whether it sits within Jev's own spread, the standard deviation of its level distribution, at least 0.1 of a level. classif's `none` is dropped from a `-e` answer and the options renormalized. Jev's answers are cached under `~/.local/state/classif/jev` by request, so a rerun calls Jev only for a new or changed case, and stay out of git with the per-case rows. Each row keeps classif's raw log mass, which `evals/calibrate.py --jev` fits a temperature from. `--diff` pairs two runs by request and puts a 95% bootstrap interval on the change in mean total variation: with about 200 cases and a handful of disagreements, a change of two or three cases is noise, and an interval that spans 0 says so.

The e2e tests run `classif` and its subcommands as a subprocess, each test in its own spec dir and log, through a local proxy that stands in for Ollama. By default the proxy replays `tests/e2e/cassette.json`, so the suite needs no model and no network and runs in about 6 seconds. A request the cassette does not hold fails its test: a changed prompt, option or text needs a new recording. `CLASSIF_E2E=record` forwards to the host and model classif routes to (`CLASSIF_HOSTS=host:port=model` picks one) and rewrites the cassette; `CLASSIF_E2E=live` forwards and keeps nothing. The cassette leaves out timestamps and durations, so two recordings of one model are byte-identical and a new recording diffs only where an answer changed.

Checks pinned to answers measured on the default model skip on any other model. The committed cassette holds Winnow, so they replay too. Known weaknesses run as expected failures, so a fix shows up as an unexpected success.

`evals/eval.py` scores 15 synthetic cases and 144 three-option decisions from [SemIf](https://github.com/TheoLeeCJ/SemIf). The latter use `-e` with option descriptions and are stored in `evals/cases/authored144.jsonl` under MIT terms; see `evals/cases/THIRD_PARTY.md`.

The evaluator also reads `rag_set.json` and `email_set.json` from `~/.local/state/classif` when present. These private sets stay out of git.

### Eval record

2026-10-03, Ollama 0.34.3 on an RTX 5070 Ti Laptop GPU (12 GB).

`eval.py`, 344 cases, `gemma4:12b` against `winnow:12b-q4_K_M`, both Q4_K_M through the gemma4 renderer. Result: Winnow became `DEFAULT_MODEL`.

| Measure | gemma4:12b | winnow:12b-q4_K_M |
| --- | ---: | ---: |
| Correct | 325 | 329 |
| email-action | 50/60 | 54/60 |
| Held-out NLL, `-l` | 0.200 | 0.166 |
| Latency p50 / p95 | 218 / 601 ms | 222 / 622 ms |

Paired: 7 won, 3 lost, 95% bootstrap interval for the accuracy gain -0.6 to +2.9 points.

2026-10-10, `jev.py`, 211 cases, `winnow:12b-q4_K_M` against `jev-1.13.0`.

| Task | Top answer agrees | Mean / median TV | TV at most 0.1 | Score within Jev's spread |
| --- | ---: | ---: | ---: | ---: |
| synthetic (15) | 15/15 | 0.033 / 0.034 | 15/15 | |
| choice (144) | 140/144 | 0.058 / 0.011 | 130/144 | |
| score (41) | 38/41 | 0.063 / 0.007 | 37/41 | 39/41, mean gap 0.06 |
| license (11) | 10/11 | 0.109 / 0.031 | 8/11 | 10/11, mean gap 0.17 |

Both got 139 of the 144 authored cases right. Of the four where their top answers differ, each matched the label twice; classif's two misses answered `insufficient` where the evidence settled it. On the score cases Jev's likeliest level matched the author's in 41 of 41 and classif's in 38, after two cases both had answered alike against the author were reworded to say what their level means (an email due in two days, a review asking for a retry cap before merging). Two of classif's misses are candidate fit, one level off in each direction and outside Jev's spread; the third rates a login outage critical where Jev and the author say blocking. The license outside Jev's spread is GFDL-1.3.

Fitted to Jev's distributions instead of labels (`calibrate.py --jev`, five folds), a temperature does not close the gap. On the 52 score and license cases, held-out TV is 0.073 at the `T_enum` of 1.96 classif applies now and 0.088 at the cross-entropy fit of 2.30, with cross-entropy tied at 0.282; a fit to TV itself reaches 0.071. Top-level agreement is 48 of 52 at every temperature from 1 to 3, since a temperature cannot change which level wins. The `-e` and `-l` cases behave alike: the fit trims cross-entropy and raises TV. The gap sits in which answer wins, so `calibration.json` carries no `T_score` for this model.
