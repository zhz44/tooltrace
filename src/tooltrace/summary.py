"""Summarise trace files: where the time went, which calls were slow, which failed.

The figures assume that one agent run issues its commands one after another,
which is how mini-swe-agent works. Within one trace:

    wall-clock    = end of the last call - start of the first call
    tool time     = sum of the call durations
    between calls = wall-clock - tool time

Time between calls is the model producing its next turn plus the agent's own
overhead. The trace cannot separate those two. The model's first turn happens
before the first call, so it is not counted at all.
"""

from __future__ import annotations

import json
import math
import statistics
from collections import Counter, defaultdict


def load(paths):
    """Read trace files into one list of rows. Blank lines are skipped."""
    rows = []
    for path in paths:
        with open(path, encoding="utf-8") as fp:
            for line_no, line in enumerate(fp, start=1):
                line = line.strip()
                if not line:
                    continue
                try:
                    rows.append(json.loads(line))
                except json.JSONDecodeError as exc:
                    raise ValueError(f"{path}:{line_no}: not valid JSON ({exc})") from exc
    return rows


def percentile(sorted_values, fraction):
    """Nearest-rank percentile of an ascending list.

    Returns the smallest value that has at least ``fraction`` of the values at
    or below it. No interpolation, so the result is always an observed value.
    """
    rank = max(1, math.ceil(fraction * len(sorted_values)))
    return sorted_values[rank - 1]


def _one_line(command, width=70):
    """First line of a command, shortened for display."""
    lines = command.strip().splitlines()
    first = lines[0] if lines else ""
    return first if len(first) <= width else first[: width - 3] + "..."


def summarise(rows, top=5):
    """Return a dict of summary figures for the given rows."""
    by_trace = defaultdict(list)
    for row in rows:
        by_trace[row.get("trace_id", "unknown")].append(row)

    per_trace = []
    for trace_id, calls in by_trace.items():
        calls.sort(key=lambda r: r["start_ts"])
        wall_s = calls[-1]["end_ts"] - calls[0]["start_ts"]
        tool_s = sum(r["end_ts"] - r["start_ts"] for r in calls)
        per_trace.append({
            "trace_id": trace_id,
            "instance_id": calls[0].get("instance_id", "unknown"),
            "calls": len(calls),
            "wall_s": wall_s,
            "tool_s": tool_s,
            "between_calls_s": wall_s - tool_s,
        })
    per_trace.sort(key=lambda t: (t["instance_id"], t["trace_id"]))

    # A trace with a single call has no time between calls, so it cannot say
    # how the time splits. It still counts in every other figure.
    timed = [t for t in per_trace if t["calls"] >= 2 and t["wall_s"] > 0]
    wall_s = sum(t["wall_s"] for t in timed)
    tool_s = sum(t["tool_s"] for t in timed)

    by_tool = defaultdict(list)
    for row in rows:
        by_tool[row.get("tool_name", "unknown")].append(row)
    tools = []
    for name, calls in by_tool.items():
        durations = sorted(r["duration_ms"] for r in calls)
        tools.append({
            "tool": name,
            "calls": len(calls),
            "failed": sum(1 for r in calls if r.get("exit_status") != "ok"),
            "median_ms": statistics.median(durations),
            "p95_ms": percentile(durations, 0.95),
            "max_ms": durations[-1],
            "total_s": sum(durations) / 1000.0,
        })
    tools.sort(key=lambda t: (-t["total_s"], t["tool"]))

    slowest = sorted(rows, key=lambda r: -r["duration_ms"])[:top]
    slowest = [{
        "duration_ms": r["duration_ms"],
        "tool": r.get("tool_name", "unknown"),
        "exit_status": r.get("exit_status", "unknown"),
        "instance_id": r.get("instance_id", "unknown"),
        "step_idx": r.get("step_idx"),
        "command": _one_line(r.get("command_raw", "")),
    } for r in slowest]

    prompt_tokens = sorted(r["prompt_tokens"] for r in rows
                           if r.get("prompt_tokens") is not None)

    return {
        "calls": len(rows),
        "traces": len(per_trace),
        "timed_traces": len(timed),
        "wall_s": wall_s,
        "tool_s": tool_s,
        "between_calls_s": wall_s - tool_s,
        "tool_fraction": tool_s / wall_s if wall_s > 0 else None,
        "exit_status": dict(Counter(r.get("exit_status", "unknown") for r in rows)),
        "tools": tools,
        "slowest": slowest,
        "prompt_tokens": {
            "calls": len(prompt_tokens),
            "median": statistics.median(prompt_tokens),
            "max": prompt_tokens[-1],
        } if prompt_tokens else None,
        "per_trace": per_trace,
    }


def format_text(summary):
    """Render the dict from ``summarise`` as plain text."""
    if summary["calls"] == 0:
        return "No calls in the given files."

    out = [f"{summary['calls']} calls in {summary['traces']} traces", ""]

    out.append(f"Time ({summary['timed_traces']} traces with at least two calls)")
    wall_s = summary["wall_s"]
    if wall_s > 0:
        tool_s = summary["tool_s"]
        between_s = summary["between_calls_s"]
        out.append(f"  wall-clock, first call to last call        {wall_s:>10.1f} s")
        out.append(f"  inside tool calls                          {tool_s:>10.1f} s"
                   f"  {100 * tool_s / wall_s:>5.1f}%")
        out.append(f"  between calls (model + agent overhead)     {between_s:>10.1f} s"
                   f"  {100 * between_s / wall_s:>5.1f}%")
    else:
        out.append("  no trace has two calls, so the split cannot be computed")
    out.append("")

    out.append("Exit status")
    counts = sorted(summary["exit_status"].items(), key=lambda item: (-item[1], item[0]))
    out.append("  " + "   ".join(f"{status} {n}" for status, n in counts))
    out.append("")

    out.append("By tool, ordered by total time")
    out.append(f"  {'tool':<16}{'calls':>6}{'failed':>8}{'median ms':>12}"
               f"{'p95 ms':>12}{'max ms':>12}{'total s':>10}")
    for t in summary["tools"]:
        out.append(f"  {t['tool']:<16}{t['calls']:>6}{t['failed']:>8}{t['median_ms']:>12.1f}"
                   f"{t['p95_ms']:>12.1f}{t['max_ms']:>12.1f}{t['total_s']:>10.1f}")
    out.append("")

    out.append("Slowest calls")
    for s in summary["slowest"]:
        out.append(f"  {s['duration_ms']:>10.1f} ms  {s['tool']:<14}{s['exit_status']:<9}"
                   f"{s['instance_id']} step {s['step_idx']}  {s['command']}")

    tokens = summary["prompt_tokens"]
    if tokens:
        out.append("")
        out.append(f"Prompt tokens when the command was issued "
                   f"({tokens['calls']} of {summary['calls']} calls have them)")
        out.append(f"  median {tokens['median']:.0f}   max {tokens['max']}")

    return "\n".join(out)
