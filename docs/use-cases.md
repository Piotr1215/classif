# Shell use cases

Setup: Python 3.12, Ollama with the default model from the README's quick start, and `CLASSIF_HOSTS=localhost:11434` (or your server) in the environment. Run the examples from the repository root. They use ordinary files and shell tools; `entr` and `ifne` (moreutils) are the only extras, each named where used.

The first label or option exits 0, another exits 1, a failed call exits 2, insufficient or below `--min-p` exits 3. `--json` prints the result as JSON.

## Choose an action in a game loop

Supply the world state as input, the rules as context, and ask for an action from the engine's allowed set:

```sh
./classif --json --enum "move_left,move_right,dodge,wait" --input world-state.json,game-rules.md \
  "What should the character do next?"
```

The engine consumes the label, checks whether the action is legal and applies it. The state holds the objective and the current situation; the rules file holds what does not change between turns. Both are the input, read as one text. If the state ever grows past the window, give the rules with `--context` instead, so they stay in front of every bounded read with the question. This is a semantic decision node inside the engine; frame scheduling and action execution remain engine code.

To reevaluate when the engine writes a new snapshot, use `entr`:

```sh
printf '%s\n' world-state.json |
  entr -n sh --context './classif --json --enum "seek_cover,advance,wait" "Which action best serves the objective in this world state?" < world-state.json'
```

The file must exist before starting the watcher. The engine replaces `world-state.json` on each turn; a history, if wanted, goes to a separate append-only JSONL that the decision never reads. Each run reads one snapshot and finishes. The watcher serializes runs; it does not make the model a frame-rate controller.

## Filter live comments

Supply the policy with the comment; the decision applies it. One JSON file holds both (two files, `--input policy.txt,comment.txt`, read the same):

```sh
cat > moderation-input.json <<'EOF'
{
  "policy": "Comments stay on the post's topic and address ideas, not people. Spam, advertising, slurs, threats and personal attacks are removed.",
  "comment": "Buy followers cheap at my site, link in bio!!!"
}
EOF
./classif --json --enum "pass,stop" \
  "Apply the supplied moderation policy to this comment. Should it pass or stop?" < moderation-input.json
```

The consumer publishes on `pass` (exit 0), rejects on `stop` (exit 1) and sends `none`, unsure or unscored results to review. The policy is the input, so changing it needs no retraining and no examples; a labelled sample is a way to check a policy's wording, not a prerequisite. `--min-p 0.9` adds a confidence floor for the review queue. This is the same shape as the game loop above: a state and a rule in one file, one bounded choice out.

## Assess a discussion's sentiment and severity

Ask about a window of comments together when their combined meaning matters:

```sh
./classif --json \
  --enum "negative_strong,negative_mild,neutral,positive_mild,positive_strong" \
  "Considering all these comments together, what sentiment strength does this discussion express about the post?" < comments-window.txt
```

A complaint about losing savings carries different meaning from “the product is boring.” Positive comments do not erase a serious harm report. Ask that as a separate decision when it should trigger review:

```sh
./classif --json --enum "review=a report of substantial harm or a credible threat requiring review" --enum "ordinary=no such report" \
  "Does any comment in this discussion require review under this policy?" < comments-window.txt
```

These are different questions: the overall tone and the presence of an exception. The policy is text supplied with the question, as in the moderation example; a labelled sample checks its wording and is optional. Label probability measures confidence in the selected band, not sentiment strength or harm. The consumer owns the time window, retention and actions; the CLI does not implement a continuous stream reducer. A long window whose evidence exceeds the judge budget returns insufficient.

## Ground a decision in the installed tool

A tool on this machine may be newer than the model's knowledge. Give its own help text as the context, the failure as the input, and let code map the answer to an action:

```sh
my-tool --help > tool-help.txt
./classif --json --input tool-help.txt,failure.log --enum "retry,correct_flags,investigate" \
  "Given this tool's help and the failure, what should happen next?"
```

The model fetches nothing and runs nothing; the consumer decides what `retry` or `correct_flags` means and does it.

## Select relevant files

When a collection contains more text than you want to read, select files that describe the topic you need:

```sh
for file in notes/*.txt; do
  ./classif --min-p 0.8 "Does this describe a database migration?" < "$file"
  status=$?
  case "$status" in
    0) printf 'match: %s\n' "$file" ;;
    2|3) printf 'review: %s\n' "$file" ;;
  esac
done
```

The script names matching files and keeps uncertain or failed decisions visible.

## Route a category

Use enums when the next step depends on one of several named outcomes:

```sh
decision=$(./classif --min-p 0.8 --enum "bug,feature,chore" \
  "What kind of change is this?" < change.txt)
status=$?
case "$status" in
  0|1)
    category=${decision%% *}
    case "$category" in
      bug) echo "Add to the fixes list." ;;
      feature) echo "Add to the features list." ;;
      chore) echo "Add to the maintenance list." ;;
      none) echo "Review the category." ;;
    esac
    ;;
  2|3) echo "Review before routing." ;;
esac
```

Both exit 0 and exit 1 can carry a valid enum result. Only the first option exits 0. This example uses single-word names; use JSON when names contain spaces or your script needs scores.

## Inspect a long document

Use a claim for a question that needs evidence from a large file:

```sh
./classif --json --deadline 20 --input agreement.txt "The agreement permits cancellation without a fee."
```

The read report states which spans reached the judge. `insufficient` means the available evidence or coverage did not settle the claim. A deadline stops further work; it does not turn an incomplete scan into an answer.

For a named answer over a large source:

```sh
./classif --json --enum "routine=ordinary operation" --enum "intervention=manual action needed" --enum "complete=installation finished" \
  --input install.log "What does this installation log indicate?"
```

An answer is refused if the flagged evidence exceeds the judge budget. Repeating the same question reuses saved reader scores; a final judge runs when passages supply an answer.

For related claims about the same entity:

```sh
./classif --json --input ledger.txt "Invoice INV-42 has been paid."
./classif --json --input ledger.txt "Invoice INV-42 was issued by SUP-1."
```

The claims share cached reference links but receive separate evidence checks. Each is answered from the lines linked to the invoice when they settle it (`basis: component`). A policy or exception that names no linked identifier sits outside the component and can change the source-wide answer.

## Watch installation logs

In one terminal, save the installer's [streaming Kubernetes logs](https://kubernetes.io/docs/reference/kubectl/generated/kubectl_logs/). Replace the workload and namespace with yours:

```sh
kubectl logs -f -n my-namespace deployment/my-installer | tee /tmp/install.log
```

After the file exists, watch snapshots in another terminal with [entr](https://eradman.com/entrproject/):

```sh
printf '%s\n' /tmp/install.log |
  entr -n sh -c './classif --json --deadline 20 --input /tmp/install.log "At least one log entry indicates the installation needs manual intervention."'
```

This asks about meaning, including failures that do not contain a fixed error keyword. Repeated runs reuse unchanged reader scores. An insufficient result does not establish that the installation is healthy. The claim concerns any entry in the snapshot; it does not compute the latest state.

The log is one file that grows; `kubectl logs -f` appends to it and each run reads the snapshot that exists at that moment. Unchanged lines reuse their saved readings, so a run after a few new lines costs those lines. `classif` reads stdin to EOF and has no watch mode of its own; `entr` serializes runs and can combine rapid change events, so it is not a promise to classify every individual event.

## Screen terms for deal-breakers

Keep the downloaded terms in `terms.txt`, then check a claim and inspect its cited evidence:

```sh
./classif --json --deadline 20 --input terms.txt "The terms allow selling personal data to third parties."
./classif --json --deadline 20 --input terms.txt "The terms impose a fee for cancelling the subscription."
```

The input should contain the full relevant terms. A supported result identifies evidence for that condition; contradicted and insufficient are separate outcomes. A deadline or ambiguous clause can require review. These are semantic screening checks, not a substitute for interpreting the legal effect of a contract.

Several checks can be scheduled with a parallel command. One local GPU may still execute inference serially, so higher concurrency alone does not make them faster. Preserve each result and its exit code when routing it.

## Reuse a classifier definition

An ad hoc command puts the question in the shell. A spec saves a repeated decision as JSON: its question, input template, labels, model, field limits and test examples. This keeps a support worker's policy consistent and lets you check changes before using them on real tickets.

Create your own spec directory:

```sh
mkdir -p /tmp/classif-specs
cat > /tmp/classif-specs/support-action.json <<'EOF'
{
  "question": "Does this support ticket require a response from the team?",
  "text": "Subject: {subject}\n\n{body}",
  "fields": {"subject": 200, "body": 3000},
  "labels": ["yes", "no"],
  "model": "winnow:12b-q4_K_M",
  "hosts": ["localhost:11434"],
  "min_p": 0.8,
  "smoke": [
    {"subject": "Cannot sign in", "body": "Please help me recover my account.", "expect": "yes"},
    {"subject": "Resolved", "body": "It works now. No help needed, thank you.", "expect": "no"}
  ]
}
EOF
export CLASSIF_DIR=/tmp/classif-specs
./classif smoke support-action
./classif classify support-action \
  subject="Cannot access my account" body="Resetting the password did not help. Can you investigate?"
```

`smoke` scores the examples and compares their labels with `expect`. `classify` supplies real field values, renders the template and returns a JSON decision. It also records a local classification log. The consumer decides whether to open a task, route a ticket or request review; the spec does not execute those actions.

The field numbers are character limits. This example uses only the first 3000 characters of the ticket body. Spec calls are bounded direct reads, so they do not inherit automatic whole-document scanning. For a large source, use the ordinary question form with `--input` instead. Expand the smoke set with representative cases before treating a passing check as evidence of production quality. Keep the spec in version control alongside the consuming application.

To keep your own specs without `CLASSIF_DIR`, put them in `~/.config/classif/specs`, and run `unset CLASSIF_DIR` after this example. `classif specs` lists every spec it finds, the fields each wants and its file. Run `./classif --help` for the available commands.

## Rank many candidates

`--enum` picks one of one to nine options in a single call. A to-do list, a set of offers or every repository you touched this month is longer than that, and you want them ordered, not one picked. [`examples/decide.py`](../examples/decide.py) asks each candidate its own questions, one classif call per candidate per question, so the list can be any length. Each call stays inside what a small model does well: one yes/no question about one short text, or a few described options.

Write the goals the tasks are judged against, then let each task take a verdict:

```sh
cat > goals.md <<'GOALS'
Goal: run a half marathon in April
Goal: ship the Rust side project to its first users by summer
Constraint: two free evenings a week
GOALS
printf '%s\n' "Sign up for a 10-week running plan" "Rewrite the side project's CLI in Go" \
  "Write the landing page for the side project" "Reorganize the bookshelf" |
  examples/decide.py "What should happen to this task?" -c goals.md -k 3 \
    -e "prioritize=do it this week, it moves a goal forward" \
    -e "defer=worth doing, not now" -e "drop=serves no goal"
```

```text
  verdict  one of      What should happen to this task? (prioritize, defer, drop)

score  verdict           candidate
0.979  prioritize 0.979  Sign up for a 10-week running plan
0.873  prioritize 0.873  Write the landing page for the side project
0.028  drop 0.914        Rewrite the side project's CLI in Go

why Sign up for a 10-week running plan: lines whose removal moves the answer most
  verdict (prioritize)
    +0.974  context: Goal: run a half marathon in April
    -0.005  context: Goal: ship the Rust side project to its first users by summer
    -0.005  context: Constraint: two free evenings a week
```

The question is asked as in classif. With `-e` it is the verdict's question, and the score is p of the first option, so the list is ordered by how surely each task should be prioritized. `-k 3` prints the top three; the rest stay in the record. Without `-e` the question is a yes/no criterion; `-y QUESTION` and `-n QUESTION` add more, and the score is the product of the p each one wants. Candidates can also follow the question as arguments, and `-i` takes files, one candidate each:

```sh
examples/decide.py "Is this the right next step?" -c goals.md "renew the passport" "reorganize the bookshelf"
examples/decide.py "Does this offer pay above market?" -n "Does it require relocating?" -c market.md -i offers/*.md
```

The `why` block explains the top pick (`-x N` explains more). Each line of the candidate's text, then of the context, is left out in turn and the question asked again; the lines whose removal moves p most are printed. They come from the same one-call reading as the answer, so they cannot disagree with it, as a line-by-line `--why` reading can. Here the running plan rests on the half-marathon goal: without that line, p(prioritize) falls by 0.974.

Every run writes the criteria with their context, each candidate's text, every p and the marks to `~/.local/state/decide/`, so a decision can be read back and rerun. A repeated decision goes in `~/.config/decide/NAME.json` with a command that lists its candidates and commands that fetch its context, and runs as `examples/decide.py -d NAME`; `python3 examples/decide.py --help` shows the format.

Keep the context short, a hand-written list of goals rather than a folder of notes: it must fit the model's window beside each candidate, and a context over 60 lines is not marked. The cost grows with the list: candidates times questions, plus one call per line for each explained pick. Each candidate is judged alone, so two tasks at 0.97 and 0.96 are a tie, not an order. A verdict's options still go to one `--enum` call, so it takes nine at most.
