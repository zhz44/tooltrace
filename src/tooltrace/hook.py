"""Record every shell command that mini-swe-agent runs in its sandbox.

Two methods of mini-swe-agent are wrapped:

* ``DockerEnvironment.execute`` runs one shell command in the sandbox container.
  The wrapper times the call and writes one JSON line that describes it.
* ``DefaultAgent.add_message`` appends a message to the conversation. The wrapper
  reads the token usage of each assistant message, so that the next command row
  records how long the conversation was when the model issued that command.

The agent's behaviour does not change: each wrapper calls the original method and
returns or re-raises exactly what the original does.

Written against mini-swe-agent 1.16.0. Other versions are untested.
"""

from __future__ import annotations

import json
import os
import sys
import time
import uuid

from .classifier import classify_tool


class Tracer:
    """The open trace file and the state the two wrappers share."""

    def __init__(self, path, instance_id="unknown", model_name="unknown"):
        self.path = str(path)
        self.trace_id = uuid.uuid4().hex[:8]  # one id per agent process
        self.instance_id = instance_id
        self.model_name = model_name
        self.step_idx = 0
        self.last_usage = None  # token usage of the most recent assistant turn
        # Append, one line per write. A second Python process started with the
        # same settings must not empty a trace that is still being written.
        self._fp = open(self.path, "a", buffering=1, encoding="utf-8")

    def note_usage(self, usage):
        """Remember the token usage of the model turn that just finished."""
        self.last_usage = dict(usage)

    def write_row(self, command, start_ts, end_ts, result, exit_status):
        """Write one JSON line for one command and return the row."""
        if isinstance(result, dict):
            output = result.get("output", "") or ""
            returncode = result.get("returncode", -1)
        else:
            # The command raised before it returned anything.
            output = "" if result is None else str(result)
            returncode = -1

        if exit_status == "ok" and returncode != 0:
            exit_status = "nonzero"

        usage = self.last_usage or {}
        row = {
            "trace_id": self.trace_id,
            "instance_id": self.instance_id,
            "step_idx": self.step_idx,
            "tool_name": classify_tool(command),
            "command_raw": command,
            "args_size_bytes": len(command.encode("utf-8", "replace")),
            "start_ts": start_ts,
            "end_ts": end_ts,
            "duration_ms": (end_ts - start_ts) * 1000.0,
            "result_size_bytes": len(output.encode("utf-8", "replace")),
            # mini-swe-agent marks output it has shortened with this text.
            "result_truncated": "<warning>" in output and "elided" in output,
            "exit_status": exit_status,
            "returncode": returncode,
            "model_name": self.model_name,
            # Usage of the most recent model turn. Null for commands that ran
            # before the first model turn.
            "prompt_tokens": usage.get("prompt_tokens"),
            "completion_tokens": usage.get("completion_tokens"),
            "total_tokens": usage.get("total_tokens"),
            "model_usage": usage,
            "extra": {},
        }
        self._fp.write(json.dumps(row) + "\n")
        self.step_idx += 1
        return row

    def close(self):
        self._fp.close()


def wrap_execute(original, tracer):
    """Return a replacement for ``DockerEnvironment.execute`` that also traces."""

    def traced_execute(self, command, cwd="", *, timeout=None):
        start_ts = time.time()
        exit_status = "ok"
        result = None
        try:
            result = original(self, command, cwd, timeout=timeout)
        except Exception as exc:
            timed_out = type(exc).__name__ == "TimeoutExpired"
            exit_status = "timeout" if timed_out else "error"
            raise
        finally:
            # Runs on success and on failure, so a failed command is recorded too.
            tracer.write_row(command, start_ts, time.time(), result, exit_status)
        return result

    return traced_execute


def _assistant_usage(message):
    """Token usage carried by an assistant message, or None."""
    if not isinstance(message, dict) or message.get("role") != "assistant":
        return None
    response = (message.get("extra") or {}).get("response") or {}
    return response.get("usage") or None


def wrap_add_message(original, tracer):
    """Return a replacement for ``DefaultAgent.add_message`` that notes token usage."""

    def traced_add_message(self, *args, **kwargs):
        result = original(self, *args, **kwargs)
        try:
            messages = getattr(self, "messages", None)
            if messages:
                usage = _assistant_usage(messages[-1])
                if usage:
                    tracer.note_usage(usage)
        except Exception:
            pass  # reading token counts must never break the agent
        return result

    return traced_add_message


def install(path, instance_id="unknown", model_name="unknown"):
    """Patch mini-swe-agent in this process and return the Tracer.

    Call it once, before the agent starts. Raises ImportError if mini-swe-agent
    is not installed.
    """
    from minisweagent.agents.default import DefaultAgent
    from minisweagent.environments.docker import DockerEnvironment

    tracer = Tracer(path, instance_id, model_name)
    DefaultAgent.add_message = wrap_add_message(DefaultAgent.add_message, tracer)
    DockerEnvironment.execute = wrap_execute(DockerEnvironment.execute, tracer)
    return tracer


def install_from_env():
    """Install the hooks if TOOLTRACE_PATH is set; otherwise do nothing.

    ``tooltrace run`` sets the variables, and the ``sitecustomize`` module in
    ``tooltrace/autoload`` calls this function when the agent's Python starts.
    """
    path = os.environ.get("TOOLTRACE_PATH")
    if not path:
        return None
    instance_id = os.environ.get("TOOLTRACE_INSTANCE_ID", "unknown")
    model_name = os.environ.get("TOOLTRACE_MODEL_NAME", "unknown")
    try:
        tracer = install(path, instance_id, model_name)
    except ImportError as exc:
        print(f"[tooltrace] mini-swe-agent not importable, tracing disabled: {exc}",
              file=sys.stderr)
        return None
    print(f"[tooltrace] tracing to {path} "
          f"(trace_id={tracer.trace_id}, instance={instance_id})", file=sys.stderr)
    return tracer
