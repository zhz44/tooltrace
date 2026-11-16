# tooltrace

Record every shell command a coding agent runs, then see where the time went.

`tooltrace` wraps [mini-swe-agent](https://github.com/SWE-agent/mini-swe-agent) without
changing its source. Each command the agent runs in its sandbox becomes one JSON line:
what ran, how long it took, how it ended, and how long the conversation was when the
model asked for it. A summary command then reports how the run's time splits between
tool calls and everything else, which calls were slowest, and which failed.

It is written against mini-swe-agent 1.16.0 and its Docker environment. It has no
dependencies beyond the Python standard library.

## Install

```bash
pip install -e .            # into the same environment as mini-swe-agent
pip install -e ".[agent]"   # or let it install mini-swe-agent 1.16.0 as well
```

## Use

Run the agent as usual, with `tooltrace run --out <file> --` in front:

```bash
tooltrace run --out traces/<task-id>.jsonl --instance <task-id> --model <model> -- \
    mini-extra swebench-single --instance <task-id> --model <model> --output traj.json
```

The agent prints one extra line on stderr, `[tooltrace] tracing to ...`. If that line
does not appear, the hooks were not installed. `--instance` and `--model` are labels
that are stored in every row.

Rows are appended to the file. Use a new file, or delete the old one, to start afresh.

Summarise one or more trace files:

```bash
tooltrace summary traces/*.jsonl          # text
tooltrace summary --json traces/*.jsonl   # the same figures as JSON
```

If the classifier changes, recompute `tool_name` from the stored commands without
running the agent again:

```bash
tooltrace reclassify --backup traces/*.jsonl
```

From Python, call `tooltrace.install(path)` once before the agent starts.

## Example

`tests/data/sample_trace.jsonl` is a made-up trace of two runs and ten calls, with
round numbers so the output can be checked by hand. `tooltrace summary` prints this
for it:

```text
10 calls in 2 traces

Time (2 traces with at least two calls)
  wall-clock, first call to last call              85.2 s
  inside tool calls                                44.2 s   51.9%
  between calls (model + agent overhead)           41.0 s   48.1%

Exit status
  ok 8   nonzero 1   timeout 1

By tool, ordered by total time
  tool             calls  failed   median ms      p95 ms      max ms   total s
  pip                  1       1     30000.0     30000.0     30000.0      30.0
  pytest               2       1      5500.0      6000.0      6000.0      11.0
  python               1       0      2000.0      2000.0      2000.0       2.0
  find                 1       0       500.0       500.0       500.0       0.5
  git                  1       0       200.0       200.0       200.0       0.2
  grep                 1       0       200.0       200.0       200.0       0.2
  cat                  1       0       100.0       100.0       100.0       0.1
  ls                   1       0       100.0       100.0       100.0       0.1
  sed                  1       0       100.0       100.0       100.0       0.1

Slowest calls
     30000.0 ms  pip           timeout  demo-2 step 2  pip install -e .
      6000.0 ms  pytest        nonzero  demo-1 step 3  cd /repo && pytest tests/test_dates.py -x
      5000.0 ms  pytest        ok       demo-1 step 5  cd /repo && pytest tests/test_dates.py -x
      2000.0 ms  python        ok       demo-2 step 1  python scripts/build_index.py
       500.0 ms  find          ok       demo-2 step 0  find . -name "*.cfg"

Prompt tokens when the command was issued (9 of 10 calls have them)
  median 2200   max 4300
```

## What a row contains

| field | meaning |
|---|---|
| `trace_id` | random id of the agent process; one per run |
| `instance_id` | the `--instance` label |
| `step_idx` | position of the command in the run, from 0 |
| `tool_name` | tool type given by the classifier, e.g. `pytest`, `grep`, `heredoc_write`, `other` |
| `command_raw` | the full command, unchanged |
| `args_size_bytes` | length of the command in bytes |
| `start_ts`, `end_ts` | wall-clock time before and after the call, seconds since the epoch |
| `duration_ms` | `end_ts - start_ts` in milliseconds |
| `result_size_bytes` | length of the output the agent received, in bytes |
| `result_truncated` | true if mini-swe-agent shortened the output |
| `exit_status` | `ok`, `nonzero` (non-zero return code), `timeout`, or `error` (the call raised) |
| `returncode` | the command's return code; -1 if the call raised |
| `model_name` | the `--model` label |
| `prompt_tokens`, `completion_tokens`, `total_tokens` | token usage of the most recent model turn; null for commands that ran before the first model turn |
| `model_usage` | the full usage record of that model turn |
| `extra` | reserved; empty |

## How it works

Two methods of mini-swe-agent are replaced by wrappers that call the originals:

- `DockerEnvironment.execute` runs one command in the sandbox. The wrapper takes the
  time before and after, and writes the row whether the command returned or raised.
  The result, or the exception, is passed on unchanged.
- `DefaultAgent.add_message` appends a message to the conversation. The wrapper reads
  the token usage attached to each assistant message, so the next row can record it.

`tooltrace run` sets three environment variables (`TOOLTRACE_PATH`,
`TOOLTRACE_INSTANCE_ID`, `TOOLTRACE_MODEL_NAME`), puts the package's `autoload`
directory at the front of `PYTHONPATH`, and starts the agent. Python imports a module
named `sitecustomize` at start-up if it finds one, and the one in `autoload` installs
the wrappers. Without `TOOLTRACE_PATH` it does nothing.

## Limits

**The agent.** Only mini-swe-agent 1.16.0 with its Docker environment has been tested.
The wrappers assume the method signatures of that version. If the environment already
has its own `sitecustomize` module, `tooltrace run` hides it for the agent process.

**The time split.** Time between calls is the model's turn plus the agent's own
overhead; the trace cannot separate the two. The model's first turn happens before the
first call and is not counted. A run with a single call is left out of the split. The
figures assume the commands of one run do not overlap, which holds for this agent.

**Percentiles.** `p95` is the nearest-rank percentile, with no interpolation. With few
calls it equals the maximum.

**The classifier.** It labels a command with one tool type by looking at the first
word of each part of the command and keeping the one expected to cost most. It is
deliberately simple, and it is wrong in known ways:

- A heredoc counts as a file write whenever the command also contains `>`, so
  `python - <<'EOF'` with a `>` in the script is labelled `heredoc_write`, not `python`.
- Names are matched by prefix: `truncate` is labelled `tr` and `lsof` is labelled `ls`.
- A command behind a wrapper is not seen: `timeout 60 pytest`, `sudo pip install` and
  `bash run_tests.sh` are all labelled `other`.
- If the only known word is a trivial one, it wins: `cd /repo && nosetests` is
  labelled `cd`.

Because `command_raw` is kept in every row, labels can be recomputed later with
`tooltrace reclassify`. Check the labels against the commands before relying on a
per-tool figure.

## Tests

```bash
pip install -e ".[test]"
pytest
```

The tests use stand-ins for the agent's classes, so they need neither the agent nor
Docker. With mini-swe-agent installed (`pip install -e ".[test,agent]"`), a further
set runs the wrappers on the real classes, using a stand-in for the `docker` command.

## Licence

BSD 2-Clause. See `LICENSE`.
