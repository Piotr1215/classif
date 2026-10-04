# Claude Code integration

[Claude Code](https://code.claude.com/docs) uses classif through its skill to ask questions about text too large for its context. The integration lives in the Claude Code config (`~/.claude`), outside this repository. classif knows nothing about Claude Code.

## Oversized text

A CI log, a day of journald or `man bash` costs Claude more context than its answer is worth. The classif skill tells Claude to send such text to the local model instead:

```sh
journalctl --since today | classif "Did any unit fail to start?"
git diff | classif -p "Does this touch auth?" | less
classif --why -i build.log "The build failed because of a missing dependency."
```

Only the verdict enters Claude's context. Exit codes branch as in the [README](../README.md#results-and-exit-codes). `--why` returns the line numbers an answer rests on, and the skill tells Claude to open the source at those lines before acting on a verdict about text it has not read. `-c` carries what the model cannot know, such as today's date or a policy.

## Off switches

- `CLASSIF=0` turns classif off in one process.
- `classif pause` turns it off everywhere and unloads the models until `classif resume`, to give the GPU to something else.

## Removed RAG gate

A RAG relevance gate was tried and removed on 4 October 2026. A local 12B model sees the prompt but not the conversation. At its best cut the gate saved about 6,000 tokens a session while losing one relevant hit in nine and adding about 25 seconds of tool latency.
