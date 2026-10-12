#!/usr/bin/env python3
"""Deterministic RoboDojo metric summarizer (success rates + latency).

Reads the per-run ``_result.json`` files that RoboDojo's eval client writes

    eval_result/RoboDojo/<task>/<policy>/<embodiment>/<seed>_ckpt_name=.../<timestamp>/_result.json

and prints a stable, sorted markdown table. This is a *local convenience*
view over results the official stack already produced — it is NOT the
official aggregator (``RoboDojo scripts/robodojo.sh summarize`` owns paired
25+25 generalization merging, 3-seed cells, and video-completeness checks;
run that for anything leaderboard-shaped) and it never fabricates a score:
runs without ``_result.json`` are reported as missing, never as 0%.

Latency: RoboDojo's own ``_result.json`` does not carry per-call policy
latency; when a sibling ``_latency.json`` (written by ``serve_policy.py
--latency-json``) exists next to it, its per-call server-side ``latency_ms``
percentiles are merged into the same row. Absent file → the latency columns
show ``-``, not invented numbers.

Determinism: inputs are walked in sorted order, floats are rounded, and only
mean/std (population) statistics are computed — same input tree, same bytes
out.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

STANDALONE_EPISODES = 50
PAIRED_HALF_EPISODES = 25


def load_json(path: Path):
    try:
        with path.open(encoding="utf-8") as fh:
            return json.load(fh)
    except (OSError, json.JSONDecodeError):
        return None


def details_entries(result: dict):
    """Yield (layout_id, success, score) sorted by layout id."""
    items = []
    for key, entry in (result.get("details") or {}).items():
        if not isinstance(entry, dict):
            continue
        try:
            layout_id = int(key)
        except (TypeError, ValueError):
            try:
                layout_id = int(entry.get("layout_id"))
            except (TypeError, ValueError):
                continue
        items.append((layout_id, bool(entry.get("success", False)),
                      float(entry.get("score", 0.0))))
    return sorted(items)


def percentile(values: list[float], q: float) -> float:
    if not values:
        return math.nan
    ordered = sorted(values)
    idx = min(len(ordered) - 1, max(0, math.ceil(q * len(ordered)) - 1))
    return ordered[idx]


def summarize_run(run_dir: Path) -> dict | None:
    ts_dirs = sorted(
        d for d in run_dir.iterdir()
        if d.is_dir() and (d / "_result.json").is_file()
    ) if run_dir.is_dir() else []
    if not ts_dirs:
        return None
    latest = ts_dirs[-1]
    result = load_json(latest / "_result.json")
    if not isinstance(result, dict):
        return None
    entries = details_entries(result)
    if not entries:
        return None
    successes = sum(1 for _, ok, _ in entries if ok)
    scores = [score for _, _, score in entries]

    latency = load_json(latest / "_latency.json") or {}
    calls = [float(v) for v in (latency.get("calls_ms") or []) if isinstance(v, (int, float))]

    return {
        "run_dir": str(run_dir),
        "episodes": len(entries),
        "successes": successes,
        "success_rate": 100.0 * successes / len(entries),
        "score": 100.0 * sum(scores) / len(scores),
        "p50_ms": percentile(calls, 0.50),
        "p95_ms": percentile(calls, 0.95),
        "mean_ms": (sum(calls) / len(calls)) if calls else math.nan,
        "timestamp": latest.name,
    }


def collect(root: Path) -> list[dict]:
    rows = []
    for task_dir in sorted(p for p in root.iterdir() if p.is_dir()):
        if task_dir.name.startswith("_"):
            continue
        for policy_dir in sorted(p for p in task_dir.iterdir() if p.is_dir()):
            for emb_dir in sorted(p for p in policy_dir.iterdir() if p.is_dir()):
                for run_dir in sorted(p for p in emb_dir.iterdir() if p.is_dir()):
                    row = summarize_run(run_dir)
                    if row:
                        row.update(task=task_dir.name, policy=policy_dir.name,
                                   embodiment=emb_dir.name, run=run_dir.name)
                        rows.append(row)
    return rows


def fmt(value: float, digits: int = 1) -> str:
    return "-" if value is None or math.isnan(value) else f"{value:.{digits}f}"


def render_markdown(rows: list[dict]) -> str:
    lines = [
        "| task | policy | embodiment | run | episodes | success | success rate % | score % | p50 ms | p95 ms | mean ms |",
        "| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |",
    ]
    for row in sorted(rows, key=lambda r: (r["task"], r["policy"], r["embodiment"], r["run"])):
        lines.append(
            f"| {row['task']} | {row['policy']} | {row['embodiment']} | {row['run']} "
            f"| {row['episodes']} | {row['successes']} | {fmt(row['success_rate'])} "
            f"| {fmt(row['score'])} | {fmt(row['p50_ms'])} | {fmt(row['p95_ms'])} "
            f"| {fmt(row['mean_ms'])} |"
        )
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("eval_root", nargs="?", type=Path,
                        default=Path("eval_result/RoboDojo"),
                        help="RoboDojo eval_result root (default: eval_result/RoboDojo)")
    parser.add_argument("--json", action="store_true", help="emit JSON instead of markdown")
    args = parser.parse_args(argv)

    root = args.eval_root
    if not root.is_dir():
        print(f"no eval_result directory at {root}", file=sys.stderr)
        return 1
    rows = collect(root)
    if not rows:
        print(f"no _result.json under {root}", file=sys.stderr)
        return 1
    if args.json:
        print(json.dumps(rows, indent=2, sort_keys=True))
    else:
        print(render_markdown(rows))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
