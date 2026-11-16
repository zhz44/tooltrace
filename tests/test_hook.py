"""The two wrappers, tested against stand-ins for mini-swe-agent's classes.

The stand-ins have the same method signatures as mini-swe-agent 1.16.0, so
these tests need neither the agent nor Docker. The real classes are tested in
test_with_mini_swe_agent.py.
"""

from __future__ import annotations

import json
import subprocess

import pytest

from tooltrace.hook import Tracer, install_from_env, wrap_add_message, wrap_execute

FIELDS = [
    "trace_id", "instance_id", "step_idx", "tool_name", "command_raw",
    "args_size_bytes", "start_ts", "end_ts", "duration_ms", "result_size_bytes",
    "result_truncated", "exit_status", "returncode", "model_name",
    "prompt_tokens", "completion_tokens", "total_tokens", "model_usage", "extra",
]


class FakeEnvironment:
    """Stands in for DockerEnvironment: returns or raises whatever the test queued."""

    def __init__(self, outcomes):
        self.outcomes = list(outcomes)

    def execute(self, command, cwd="", *, timeout=None):
        outcome = self.outcomes.pop(0)
        if isinstance(outcome, Exception):
            raise outcome
        return outcome


class FakeAgent:
    """Stands in for DefaultAgent: keeps the conversation in ``messages``."""

    def __init__(self):
        self.messages = []

    def add_message(self, role, content, **kwargs):
        self.messages.append({"role": role, "content": content, **kwargs})


def read_rows(path):
    return [json.loads(line) for line in path.read_text().splitlines()]


def traced(tmp_path, outcomes):
    """A FakeEnvironment whose execute is wrapped, and the trace path."""
    path = tmp_path / "trace.jsonl"
    tracer = Tracer(path, instance_id="task-1", model_name="m")
    env = FakeEnvironment(outcomes)
    env.execute = wrap_execute(FakeEnvironment.execute, tracer).__get__(env)
    return env, tracer, path


def test_one_row_per_call_with_all_fields(tmp_path):
    env, tracer, path = traced(tmp_path, [
        {"output": "a.py\nb.py\n", "returncode": 0},
        {"output": "", "returncode": 0},
    ])
    result = env.execute("ls /repo")
    env.execute("cd /repo && pytest -x", timeout=60)

    assert result == {"output": "a.py\nb.py\n", "returncode": 0}  # passed through unchanged
    rows = read_rows(path)
    assert len(rows) == 2
    assert list(rows[0]) == FIELDS
    assert [r["step_idx"] for r in rows] == [0, 1]
    assert [r["tool_name"] for r in rows] == ["ls", "pytest"]
    assert rows[0]["command_raw"] == "ls /repo"
    assert rows[0]["args_size_bytes"] == len("ls /repo")
    assert rows[0]["result_size_bytes"] == len("a.py\nb.py\n")
    assert rows[0]["exit_status"] == "ok"
    assert rows[0]["instance_id"] == "task-1"
    assert rows[0]["trace_id"] == tracer.trace_id
    assert rows[0]["end_ts"] >= rows[0]["start_ts"]
    assert rows[0]["duration_ms"] == pytest.approx((rows[0]["end_ts"] - rows[0]["start_ts"]) * 1000)


def test_nonzero_return_code(tmp_path):
    env, _, path = traced(tmp_path, [{"output": "1 failed", "returncode": 1}])
    env.execute("pytest")
    row = read_rows(path)[0]
    assert (row["exit_status"], row["returncode"]) == ("nonzero", 1)


def test_timeout_is_recorded_and_raised_again(tmp_path):
    env, _, path = traced(tmp_path, [subprocess.TimeoutExpired("docker", 5)])
    with pytest.raises(subprocess.TimeoutExpired):
        env.execute("sleep 100", timeout=5)
    row = read_rows(path)[0]
    assert (row["exit_status"], row["returncode"]) == ("timeout", -1)


def test_other_error_is_recorded_and_raised_again(tmp_path):
    env, _, path = traced(tmp_path, [RuntimeError("container gone")])
    with pytest.raises(RuntimeError):
        env.execute("ls")
    assert read_rows(path)[0]["exit_status"] == "error"


def test_truncated_output_is_flagged(tmp_path):
    env, _, path = traced(tmp_path, [
        {"output": "head\n<warning>\n5000 characters elided\n</warning>\ntail", "returncode": 0},
    ])
    env.execute("cat big.log")
    assert read_rows(path)[0]["result_truncated"] is True


def test_token_usage_of_latest_model_turn_goes_on_the_next_row(tmp_path):
    env, tracer, path = traced(tmp_path, [{"output": "", "returncode": 0}] * 3)
    agent = FakeAgent()
    agent.add_message = wrap_add_message(FakeAgent.add_message, tracer).__get__(agent)
    usage = {"prompt_tokens": 1200, "completion_tokens": 80, "total_tokens": 1280}

    env.execute("ls")                                   # before any model turn
    agent.add_message("assistant", "run grep", extra={"response": {"usage": usage}})
    env.execute("grep -r foo .")
    agent.add_message("user", "output of grep")          # not a model turn
    env.execute("cat a.py")

    rows = read_rows(path)
    assert rows[0]["prompt_tokens"] is None and rows[0]["model_usage"] == {}
    assert rows[1]["prompt_tokens"] == 1200 and rows[1]["total_tokens"] == 1280
    assert rows[2]["model_usage"] == usage               # still the latest model turn
    assert len(agent.messages) == 2                      # add_message itself still worked


def test_second_tracer_on_the_same_file_appends(tmp_path):
    path = tmp_path / "trace.jsonl"
    first = Tracer(path)
    first.write_row("ls", 0.0, 1.0, {"output": "", "returncode": 0}, "ok")
    second = Tracer(path)                                # must not empty the file
    second.write_row("pwd", 2.0, 3.0, {"output": "", "returncode": 0}, "ok")
    rows = read_rows(path)
    assert [r["command_raw"] for r in rows] == ["ls", "pwd"]
    assert rows[0]["trace_id"] != rows[1]["trace_id"]


def test_nothing_happens_without_the_environment_variable(monkeypatch):
    monkeypatch.delenv("TOOLTRACE_PATH", raising=False)
    assert install_from_env() is None
