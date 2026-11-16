"""Command line: ``tooltrace run``, ``tooltrace summary``, ``tooltrace reclassify``."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

from .reclassify import reclassify_file
from .summary import format_text, load, summarise


def _build_parser():
    parser = argparse.ArgumentParser(
        prog="tooltrace",
        description="Record and summarise the shell commands a coding agent runs.")
    sub = parser.add_subparsers(dest="action", required=True)

    run = sub.add_parser(
        "run", help="run an agent command with tracing switched on",
        description="Example: tooltrace run --out traces/a.jsonl -- mini-extra swebench-single ...")
    run.add_argument("--out", required=True, help="trace file; rows are appended to it")
    run.add_argument("--instance", default="unknown", help="label for this run, e.g. a task id")
    run.add_argument("--model", default="unknown", help="model name to store in each row")
    run.add_argument("agent_command", nargs=argparse.REMAINDER,
                     help="the command to run, written after --")

    summary = sub.add_parser("summary", help="summarise one or more trace files")
    summary.add_argument("files", nargs="+")
    summary.add_argument("--json", action="store_true", help="print the figures as JSON")
    summary.add_argument("--top", type=int, default=5, help="how many slowest calls to list")

    reclassify = sub.add_parser(
        "reclassify", help="recompute tool_name in place from the stored commands")
    reclassify.add_argument("files", nargs="+")
    reclassify.add_argument("--backup", action="store_true", help="keep a .bak copy of each file")

    return parser


def _run(args, parser):
    command = args.agent_command
    if command and command[0] == "--":
        command = command[1:]
    if not command:
        parser.error("run: give the agent command after --")

    out = os.path.abspath(args.out)
    os.makedirs(os.path.dirname(out), exist_ok=True)

    env = dict(os.environ)
    env["TOOLTRACE_PATH"] = out
    env["TOOLTRACE_INSTANCE_ID"] = args.instance
    env["TOOLTRACE_MODEL_NAME"] = args.model
    # Python imports `sitecustomize` at start-up from the first place it finds
    # one. Putting our directory first makes the agent's Python install the hooks.
    autoload_dir = str(Path(__file__).parent / "autoload")
    old_path = env.get("PYTHONPATH")
    env["PYTHONPATH"] = autoload_dir + os.pathsep + old_path if old_path else autoload_dir

    try:
        os.execvpe(command[0], command, env)  # replaces this process; does not return
    except FileNotFoundError:
        print(f"tooltrace: command not found: {command[0]}", file=sys.stderr)
        return 127


def _summary(args):
    result = summarise(load(args.files), top=args.top)
    print(json.dumps(result, indent=2) if args.json else format_text(result))
    return 0


def _reclassify(args):
    total = changed = 0
    for path in args.files:
        rows, rows_changed = reclassify_file(path, backup=args.backup)
        total += rows
        changed += rows_changed
        print(f"{path}: {rows} rows, {rows_changed} changed")
    print(f"total: {changed} of {total} rows changed")
    return 0


def main(argv=None):
    parser = _build_parser()
    args = parser.parse_args(argv)
    if args.action == "run":
        return _run(args, parser)
    if args.action == "summary":
        return _summary(args)
    return _reclassify(args)


if __name__ == "__main__":
    sys.exit(main())
