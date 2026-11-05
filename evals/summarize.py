"""results.jsonl -> results.md. Every number in the rendered page is computed
from the rows read in; nothing here is a literal figure.

    uv run python -m evals.summarize --results evals/results.jsonl --out evals/results.md
"""

import argparse
import json
import re
from collections import defaultdict
from collections.abc import Callable
from pathlib import Path

from evals import load_trips
from evals.metrics import RULES, aggregate

ARMS = ("A", "B", "C")

#: Load-bearing wording (docs/DECISIONS.md, docs/SPEC.md): no arm runs without
#: the validator -- it scores all three. Calling A "no validator" would claim
#: these numbers measure something they do not.
ARM_LABELS = {
    "A": "LLM only, unrepaired",
    "B": "ids, no repair",
    "C": "full system",
}

_CONSISTENCY_FIELDS = ("model", "reasoning_effort", "schema_version")


def _read_rows(path: Path) -> list[dict]:
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line:
            rows.append(json.loads(line))
    return rows


def _latest_by_pair(rows: list[dict]) -> list[dict]:
    """The last row written for each (trip_id, arm) -- a --force rerun
    supersedes the earlier row rather than duplicating it.
    """
    latest: dict[tuple[str, str], dict] = {}
    for row in rows:
        latest[(row["trip_id"], row["arm"])] = row
    return list(latest.values())


def _check_consistency(rows: list[dict]) -> list[str]:
    warnings = []
    for field in _CONSISTENCY_FIELDS:
        values = {row.get(field) for row in rows}
        if len(values) > 1:
            warnings.append(f"rows disagree on {field}: {sorted(str(v) for v in values)}")
    return warnings


def _paired_trip_ids(rows: list[dict]) -> set[str]:
    by_trip: dict[str, set[str]] = defaultdict(set)
    for row in rows:
        by_trip[row["trip_id"]].add(row["arm"])
    return {trip_id for trip_id, arms in by_trip.items() if arms == set(ARMS)}


def _rows_for_arm(rows: list[dict], arm: str) -> list[dict]:
    return [row for row in rows if row["arm"] == arm]


#: src/validator.py's V3 message ends in repr(interest); plain words like "food"
#: or "museums" repr with no escaping, so a trailing quoted token is enough.
_MISSING_INTEREST_RE = re.compile(r"'([^']*)'\s*$")


def _trips_interests(trips: dict) -> dict[str, list[str]]:
    """trip_id -> its distinct requested interests, read from trips.json."""
    return {t["trip_id"]: list(dict.fromkeys(t["request"]["interests"])) for t in trips["trips"]}


def _missing_interests(results: list[dict]) -> set[str]:
    missing = set()
    for r in results:
        if r["summary"] or r["status"] != "fail" or r["rule"] != "V3" or r["code"] != "missing_interest":
            continue
        match = _MISSING_INTEREST_RE.search(r["message"])
        if match:
            missing.add(match.group(1))
    return missing


def _with_interest_coverage(row: dict, trips_by_id: dict[str, list[str]]) -> dict:
    """row with interests_requested/interests_covered recomputed from trips.json
    and the row's own V3 results, ignoring whatever those two fields already held
    -- this is what keeps an old 0/0 row (and any future row) correct alike.
    """
    requested = set(trips_by_id.get(row["trip_id"], []))
    covered = requested - _missing_interests(row["results"])
    return {**row, "interests_requested": len(requested), "interests_covered": len(covered)}


def _fmt_pct(x: float) -> str:
    return f"{x * 100:.1f}%"


def _fmt_num(x: float) -> str:
    return f"{x:.2f}"


def _fmt_tokens(x: float) -> str:
    return f"{x:,.0f}"


def _fmt_pct_or_na(x: float | None) -> str:
    return "n/a" if x is None else _fmt_pct(x)


_METRIC_ROWS: tuple[tuple[str, str, Callable[[float], str]], ...] = (
    ("Pass rate", "pass_rate", _fmt_pct),
    ("Hallucinated-venue rate (V5 untraceable, pooled)", "hallucination_rate", _fmt_pct),
    ("Wrong-location stops (pooled)", "wrong_location_rate", _fmt_pct),
    ("Interest coverage (pooled)", "interest_coverage", _fmt_pct_or_na),
    ("Repairs per trip", "repairs_per_trip", _fmt_num),
    ("Tool calls per trip", "tool_calls_per_trip", _fmt_num),
    ("Tokens per trip (total, incl. shared gather)", "tokens_total_per_trip", _fmt_tokens),
    ("Tokens per trip (this arm only)", "tokens_arm_per_trip", _fmt_tokens),
    ("Seconds per trip (model + tool time)", "seconds_per_trip", _fmt_num),
    ("Average daily km", "avg_day_km", _fmt_num),
    ("Tool-step cap hit rate", "cap_hit_rate", _fmt_pct),
    ("Estimated cost share", "estimated_share", _fmt_pct),
    ("Unknown-cost stops per trip", "unknown_count_per_trip", _fmt_num),
    ("Priced share", "priced_share", _fmt_pct),
)


def _metrics_table_md(metrics_by_arm: dict[str, dict]) -> str:
    header = "| Metric | " + " | ".join(f"{arm} ({ARM_LABELS[arm]})" for arm in ARMS) + " |"
    sep = "|" + "---|" * (len(ARMS) + 1)
    lines = [header, sep]
    for label, key, fmt in _METRIC_ROWS:
        cells = " | ".join(fmt(metrics_by_arm[arm][key]) for arm in ARMS)
        lines.append(f"| {label} | {cells} |")
    n_cells = " | ".join(str(metrics_by_arm[arm]["n"]) for arm in ARMS)
    lines.append(f"| N | {n_cells} |")
    return "\n".join(lines)


def _rule_table_md(metrics_by_arm: dict[str, dict]) -> str:
    header = "| Rule | " + " | ".join(ARMS) + " |"
    sep = "|" + "---|" * (len(ARMS) + 1)
    lines = [header, sep]
    for rule in RULES:
        cells = []
        for arm in ARMS:
            m = metrics_by_arm[arm]
            cells.append(f"{_fmt_pct(m['rule_fail_share'][rule])} ({m['rule_detail_count'][rule]} detail(s))")
        lines.append(f"| {rule} | " + " | ".join(cells) + " |")
    return "\n".join(lines)


def render(rows: list[dict], *, trips_by_id: dict[str, list[str]] | None = None) -> str:
    if trips_by_id is None:
        trips_by_id = _trips_interests(load_trips())

    rows = _latest_by_pair(rows)
    invalid_rows = [r for r in rows if r["outcome"] == "rejected"]
    other_rows = [_with_interest_coverage(r, trips_by_id) for r in rows if r["outcome"] != "rejected"]
    invalid_trip_ids = {r["trip_id"] for r in invalid_rows}
    attempted_trip_ids = {r["trip_id"] for r in other_rows}

    warnings = _check_consistency(other_rows)

    paired_ids = _paired_trip_ids(other_rows)
    unpaired_ids = sorted(attempted_trip_ids - paired_ids)
    paired_rows = [r for r in other_rows if r["trip_id"] in paired_ids]

    paired_metrics = {arm: aggregate(_rows_for_arm(paired_rows, arm)) for arm in ARMS}
    all_by_arm = {arm: _rows_for_arm(other_rows, arm) for arm in ARMS}
    all_metrics = {arm: aggregate(all_by_arm[arm]) for arm in ARMS}
    error_counts = {arm: sum(1 for r in all_by_arm[arm] if r["outcome"] == "error") for arm in ARMS}

    lines: list[str] = []
    lines.append("# Eval results")
    lines.append("")
    lines.append(
        f"Generated from `results.jsonl`: {len(rows)} row(s) across "
        f"{len(invalid_trip_ids) + len(attempted_trip_ids)} trip(s)."
    )
    lines.append("")
    if warnings:
        lines.append(f"**Warning:** {'; '.join(warnings)}")
        lines.append("")
    lines.append(
        f"{len(invalid_trip_ids)} invalid trip(s) rejected before any API call; "
        "excluded from every rate below."
    )
    lines.append("")
    lines.append(
        "No arm runs without the validator -- it scores all three. Arm A is "
        '**"LLM only, unrepaired"**: the model writes whole stops and the validator '
        "scores them, but nothing repairs a failure. Arm B is arm C's own first "
        "attempt with repair switched off, so B and C are paired, not independent "
        "samples."
    )
    lines.append("")

    lines.append("## Headline: paired trips (all three arms finished)")
    lines.append("")
    per_arm_n = ", ".join(f"{arm}={all_metrics[arm]['n']}" for arm in ARMS)
    lines.append(f"Per-arm N: {per_arm_n}. Paired N: {len(paired_ids)}.")
    lines.append("")
    if unpaired_ids:
        lines.append(
            f"{len(unpaired_ids)} trip(s) did not finish on all three arms and are "
            f"excluded here (see the secondary table below): {', '.join(unpaired_ids)}."
        )
        lines.append("")
    lines.append(_metrics_table_md(paired_metrics))
    lines.append("")
    lines.append("### Failures by rule (paired trips)")
    lines.append("")
    lines.append(_rule_table_md(paired_metrics))
    lines.append("")

    lines.append("## All finished trips (unpaired; not comparable across arms)")
    lines.append("")
    lines.append(_metrics_table_md(all_metrics))
    lines.append("")
    lines.append("### Failures by rule (all finished trips)")
    lines.append("")
    lines.append(_rule_table_md(all_metrics))
    lines.append("")

    lines.append("## Errors")
    lines.append("")
    lines.append(
        "Infrastructure failures (AgentError / LLMUnavailable), reported separately "
        "so one is never counted as a model failure in disguise:"
    )
    lines.append("")
    lines.append(", ".join(f"{arm}: {error_counts[arm]}" for arm in ARMS))
    lines.append("")

    lines.append("## Known gaps")
    lines.append("")
    lines.append(
        "- Pricing coverage and estimated share are structurally 0.0: `src/pricing.py` "
        "is a documented stub that returns `UNPRICED` for every stop."
    )
    lines.append(
        "- The budget trips (V2) are untested: with no stop priced, V2 can only warn, "
        "never fail."
    )
    lines.append(
        "- No weather-replan metric is reported: no replan function exists in "
        "`src/agent.py` yet."
    )
    lines.append("")

    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Summarize evals/results.jsonl into results.md.")
    parser.add_argument("--results", default="evals/results.jsonl")
    parser.add_argument("--out", default="evals/results.md")
    args = parser.parse_args(argv)

    rows = _read_rows(Path(args.results))
    Path(args.out).write_text(render(rows), encoding="utf-8")
    print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
