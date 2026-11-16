"""The hooks on the real mini-swe-agent classes.

Skipped when mini-swe-agent is not installed. Docker is not needed: the real
``DockerEnvironment.execute`` is pointed at a stand-in for the ``docker``
command that runs the shell command on this machine instead of in a container.
"""

from __future__ import annotations

import inspect
import json
import subprocess
import sys

import pytest

pytest.importorskip("minisweagent")

from minisweagent.agents.default import DefaultAgent  # noqa: E402
from minisweagent.environments.docker import (  # noqa: E402
    DockerEnvironment,
    DockerEnvironmentConfig,
)

from tooltrace.hook import install  # noqa: E402

STAND_IN_DOCKER = """#!/bin/sh
# Called as: docker exec -w <cwd> <container> bash -lc <command>
# Drop everything before "bash" and run the rest here.
while [ $# -gt 0 ] && [ "$1" != "bash" ]; do shift; done
[ $# -gt 0 ] && exec "$@"
exit 0
"""


def read_rows(path):
    return [json.loads(line) for line in path.read_text().splitlines()]


@pytest.fixture
def hooked(tmp_path, monkeypatch):
    """Install the hooks on the real classes; undo it when the test ends."""
    monkeypatch.setattr(DockerEnvironment, "execute", DockerEnvironment.execute)
    monkeypatch.setattr(DefaultAgent, "add_message", DefaultAgent.add_message)
    path = tmp_path / "trace.jsonl"
    tracer = install(path, instance_id="task-1", model_name="m")

    script = tmp_path / "docker"
    script.write_text(STAND_IN_DOCKER)
    script.chmod(0o755)
    env = DockerEnvironment.__new__(DockerEnvironment)  # skip __init__: it starts a container
    env.config = DockerEnvironmentConfig(image="unused", cwd=str(tmp_path),
                                         executable=str(script), timeout=30)
    env.container_id = "stand-in"
    yield env, path
    env.container_id = None  # so that the object's clean-up has nothing to stop
    tracer.close()


def test_methods_and_signature_are_what_the_wrappers_assume():
    assert callable(DefaultAgent.add_message)
    parameters = inspect.signature(DockerEnvironment.execute).parameters
    assert list(parameters) == ["self", "command", "cwd", "timeout"]
    assert parameters["timeout"].kind is inspect.Parameter.KEYWORD_ONLY


def test_real_execute_is_traced(hooked):
    env, path = hooked

    ok = env.execute("echo hello")
    failed = env.execute("exit 3")

    assert ok == {"output": "hello\n", "returncode": 0}  # result passed through unchanged
    assert failed["returncode"] == 3
    rows = read_rows(path)
    assert [(r["step_idx"], r["tool_name"], r["exit_status"], r["returncode"]) for r in rows] == [
        (0, "echo", "ok", 0),
        (1, "other", "nonzero", 3),
    ]
    assert rows[0]["result_size_bytes"] == len("hello\n")


def test_real_timeout_is_traced_and_raised_again(hooked):
    env, path = hooked
    with pytest.raises(subprocess.TimeoutExpired):
        env.execute("sleep 5", timeout=1)
    row = read_rows(path)[0]
    assert row["exit_status"] == "timeout"
    assert 900 <= row["duration_ms"] < 4000


def test_real_add_message_feeds_token_counts(hooked):
    env, path = hooked
    agent = DefaultAgent.__new__(DefaultAgent)  # skip __init__: it needs a model
    agent.messages = []
    usage = {"prompt_tokens": 700, "completion_tokens": 30, "total_tokens": 730}

    # The same call DefaultAgent.query makes after a model turn in 1.16.0.
    agent.add_message("assistant", "run ls", extra={"response": {"usage": usage}})
    env.execute("ls")

    assert agent.messages[0]["role"] == "assistant"
    row = read_rows(path)[0]
    assert (row["prompt_tokens"], row["completion_tokens"], row["total_tokens"]) == (700, 30, 730)


def test_tooltrace_run_installs_the_hooks_in_the_agent_process(tmp_path):
    out = tmp_path / "trace.jsonl"
    show = ("from minisweagent.environments.docker import DockerEnvironment; "
            "print(DockerEnvironment.execute.__name__)")

    result = subprocess.run(
        [sys.executable, "-m", "tooltrace", "run", "--out", str(out), "--", sys.executable, "-c", show],
        capture_output=True, text=True)

    assert result.returncode == 0
    assert result.stdout.splitlines()[-1] == "traced_execute"
    assert f"[tooltrace] tracing to {out}" in result.stderr
