"""Deterministic RoboDojo metric summarizer tests.

Covers the official result layout
``<task>/<policy>/<embodiment>/<run>/<timestamp>/_result.json`` (verified
against RoboDojo @ cb2a37d), the missing-episode (never 0%) rule, latency
merge from ``_latency.json``, and byte-identical output on re-runs.
"""
import json
import sys
from pathlib import Path

import pytest

BENCH = Path(__file__).resolve().parents[1] / "benchmarks" / "robodojo"
sys.path.insert(0, str(BENCH))

import summarize_metrics as sm  # noqa: E402


def make_run(root, task="block_hammER", policy="Pi05_RLT", emb="arx_x5",
             run="0_ckpt_name=pi05_rlt", ts="20261012_120000",
             successes=(1, 0, 1), latency_calls=None, with_result=True):
    d = root / task / policy / emb / run / ts
    d.mkdir(parents=True)
    if with_result:
        details = {str(i): {"layout_id": i, "success": bool(s),
                            "score": float(s)} for i, s in enumerate(successes)}
        (d / "_result.json").write_text(
            json.dumps({"task_name": task, "details": details})
        )
    if latency_calls is not None:
        (d / "_latency.json").write_text(json.dumps({"calls_ms": latency_calls}))
    return d


def test_success_rate_and_score(tmp_path):
    make_run(tmp_path, successes=(1, 0, 1))
    rows = sm.collect(tmp_path)
    assert len(rows) == 1
    row = rows[0]
    assert row["episodes"] == 3 and row["successes"] == 2
    assert row["success_rate"] == pytest.approx(66.666666, abs=1e-3)
    assert row["score"] == pytest.approx(66.666666, abs=1e-3)


def test_missing_result_not_counted_as_zero(tmp_path):
    make_run(tmp_path, with_result=False)
    make_run(tmp_path, task="other", successes=(1,))
    rows = sm.collect(tmp_path)
    assert [r["task"] for r in rows] == ["other"]


def test_result_json_unparsable_skipped(tmp_path):
    d = make_run(tmp_path)
    (d / "_result.json").write_text("{not json")
    assert sm.collect(tmp_path) == []


def test_latest_timestamp_wins(tmp_path):
    make_run(tmp_path, ts="20261011_000000", successes=(0,))
    make_run(tmp_path, ts="20261012_000000", successes=(1, 1))
    rows = sm.collect(tmp_path)
    assert rows[0]["episodes"] == 2 and rows[0]["successes"] == 2
    assert rows[0]["timestamp"] == "20261012_000000"


def test_latency_merged_from_sibling_json(tmp_path):
    make_run(tmp_path, latency_calls=[10.0, 20.0, 30.0, 40.0])
    row = sm.collect(tmp_path)[0]
    assert row["p50_ms"] == 20.0  # ceil(.5*4)-1 = idx 1
    assert row["p95_ms"] == 40.0
    assert row["mean_ms"] == pytest.approx(25.0)


def test_latency_missing_shows_dash(tmp_path):
    make_run(tmp_path)
    row = sm.collect(tmp_path)[0]
    assert row["p50_ms"] != row["p50_ms"]  # NaN → rendered as "-"


def test_deterministic_output(tmp_path, capsys):
    make_run(tmp_path, task="b_task", successes=(1, 0))
    make_run(tmp_path, task="a_task", successes=(1, 1, 1))
    outs = []
    for _ in range(2):
        assert sm.main([str(tmp_path)]) == 0
        outs.append(capsys.readouterr().out)
    assert outs[0] == outs[1]
    assert outs[0].splitlines()[2].startswith("| a_task |")
    assert outs[0].splitlines()[3].startswith("| b_task |")


def test_json_flag(tmp_path, capsys):
    make_run(tmp_path, successes=(1,))
    assert sm.main(["--json", str(tmp_path)]) == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload[0]["success_rate"] == pytest.approx(100.0)


def test_empty_root_errors(tmp_path, capsys):
    assert sm.main([str(tmp_path / "nope")]) == 1
    assert "no eval_result directory" in capsys.readouterr().err


def test_non_dict_details_ignored(tmp_path):
    d = make_run(tmp_path)
    result = json.loads((d / "_result.json").read_text())
    result["details"]["bad"] = "not-a-dict"
    (d / "_result.json").write_text(json.dumps(result))
    row = sm.collect(tmp_path)[0]
    assert row["episodes"] == 3  # only valid entries counted
