"""Summary figures on tests/data/sample_trace.jsonl.

The sample is made up: two runs, ten calls, with round numbers so that every
expected value below can be checked by hand.

    demo-1: calls of 0.1, 0.2, 0.1, 6.0, 0.1, 5.0 s; first start 0.0, last end 37.5
    demo-2: calls of 0.5, 2.0, 30.0, 0.2 s;          first start 100.0, last end 147.7
"""

from __future__ import annotations

from pathlib import Path

import pytest

from tooltrace.summary import format_text, load, percentile, summarise

SAMPLE = Path(__file__).parent / "data" / "sample_trace.jsonl"


@pytest.fixture(scope="module")
def summary():
    return summarise(load([SAMPLE]), top=3)


def test_counts(summary):
    assert summary["calls"] == 10
    assert summary["traces"] == 2
    assert summary["timed_traces"] == 2
    assert summary["exit_status"] == {"ok": 8, "nonzero": 1, "timeout": 1}


def test_time_split(summary):
    assert summary["wall_s"] == pytest.approx(37.5 + 47.7)
    assert summary["tool_s"] == pytest.approx(11.5 + 32.7)
    assert summary["between_calls_s"] == pytest.approx(26.0 + 15.0)
    assert summary["tool_fraction"] == pytest.approx(44.2 / 85.2)


def test_per_trace(summary):
    demo_1, demo_2 = summary["per_trace"]
    assert (demo_1["instance_id"], demo_1["calls"]) == ("demo-1", 6)
    assert demo_1["wall_s"] == pytest.approx(37.5)
    assert demo_1["tool_s"] == pytest.approx(11.5)
    assert (demo_2["instance_id"], demo_2["calls"]) == ("demo-2", 4)
    assert demo_2["between_calls_s"] == pytest.approx(15.0)


def test_per_tool(summary):
    tools = {t["tool"]: t for t in summary["tools"]}
    assert summary["tools"][0]["tool"] == "pip"          # ordered by total time
    pytest_row = tools["pytest"]
    assert (pytest_row["calls"], pytest_row["failed"]) == (2, 1)
    assert pytest_row["median_ms"] == pytest.approx(5500.0)
    assert pytest_row["p95_ms"] == pytest.approx(6000.0)
    assert pytest_row["total_s"] == pytest.approx(11.0)
    assert tools["pip"]["failed"] == 1                   # the timeout


def test_slowest(summary):
    assert [s["tool"] for s in summary["slowest"]] == ["pip", "pytest", "pytest"]
    assert summary["slowest"][0]["exit_status"] == "timeout"
    assert summary["slowest"][0]["command"] == "pip install -e ."


def test_prompt_tokens(summary):
    # Nine calls carry token counts; the first call of demo-1 ran before any model turn.
    assert summary["prompt_tokens"] == {"calls": 9, "median": 2200, "max": 4300}


def test_single_call_trace_is_left_out_of_the_time_split():
    rows = load([SAMPLE])
    lone = dict(rows[0], trace_id="cccc3333", instance_id="demo-3")
    summary = summarise(rows + [lone])
    assert (summary["traces"], summary["timed_traces"], summary["calls"]) == (3, 2, 11)
    assert summary["wall_s"] == pytest.approx(85.2)


def test_percentile_is_nearest_rank():
    values = [10, 20, 30, 40]
    assert percentile(values, 0.5) == 20
    assert percentile(values, 0.95) == 40
    assert percentile([7], 0.95) == 7


def test_text_output_and_empty_input(summary):
    text = format_text(summary)
    assert "10 calls in 2 traces" in text
    assert "51.9%" in text                               # 44.2 / 85.2
    assert format_text(summarise([])) == "No calls in the given files."


def test_bad_line_is_reported_with_file_and_line(tmp_path):
    path = tmp_path / "broken.jsonl"
    path.write_text('{"trace_id": "x"}\n{"trace_id": \n')
    with pytest.raises(ValueError, match=r"broken\.jsonl:2"):
        load([path])
