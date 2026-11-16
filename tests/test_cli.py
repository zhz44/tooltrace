"""The three commands, run the way a user runs them."""

from __future__ import annotations

import json
import shutil
import subprocess
import sys
from pathlib import Path

SAMPLE = Path(__file__).parent / "data" / "sample_trace.jsonl"


def tooltrace(*args, **kwargs):
    return subprocess.run([sys.executable, "-m", "tooltrace", *args],
                          capture_output=True, text=True, **kwargs)


def test_summary_text():
    result = tooltrace("summary", str(SAMPLE))
    assert result.returncode == 0
    assert result.stdout.startswith("10 calls in 2 traces")


def test_summary_json():
    result = tooltrace("summary", "--json", str(SAMPLE))
    assert json.loads(result.stdout)["calls"] == 10


def test_reclassify_fixes_a_wrong_label_and_keeps_a_backup(tmp_path):
    trace = tmp_path / "trace.jsonl"
    rows = [json.loads(line) for line in SAMPLE.read_text().splitlines()]
    rows[0]["tool_name"] = "wrong"
    trace.write_text("".join(json.dumps(r) + "\n" for r in rows))

    result = tooltrace("reclassify", "--backup", str(trace))

    assert "1 of 10 rows changed" in result.stdout
    assert json.loads(trace.read_text().splitlines()[0])["tool_name"] == "ls"
    assert "wrong" in (tmp_path / "trace.jsonl.bak").read_text()


def test_run_sets_the_variables_and_runs_the_command(tmp_path):
    out = tmp_path / "traces" / "a.jsonl"                 # the folder does not exist yet
    show = ("import os; print(os.environ['TOOLTRACE_PATH']); "
            "print(os.environ['TOOLTRACE_INSTANCE_ID']); "
            "print(os.environ['PYTHONPATH'].split(os.pathsep)[0])")

    result = tooltrace("run", "--out", str(out), "--instance", "task-7",
                       "--", sys.executable, "-c", show)

    assert result.returncode == 0
    # Last three lines: mini-swe-agent, when installed, prints a banner first.
    path, instance, first_pythonpath_entry = result.stdout.splitlines()[-3:]
    assert path == str(out)
    assert instance == "task-7"
    assert (Path(first_pythonpath_entry) / "sitecustomize.py").exists()
    assert out.parent.is_dir()


def test_run_without_a_command_is_an_error(tmp_path):
    result = tooltrace("run", "--out", str(tmp_path / "a.jsonl"))
    assert result.returncode == 2
    assert "give the agent command after --" in result.stderr


def test_run_reports_a_missing_command(tmp_path):
    result = tooltrace("run", "--out", str(tmp_path / "a.jsonl"), "--", "no-such-command-xyz")
    assert result.returncode == 127
    assert shutil.which("no-such-command-xyz") is None
