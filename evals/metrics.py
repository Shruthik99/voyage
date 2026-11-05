"""Turn one trace dict into one comparable row, and rows into aggregate metrics.

Pure: no network, no I/O, no mutation of what is passed in. Everything here
operates on the JSON shape src.trace.Trace.to_dict() produces (or a truncated
prefix of its events), so a test can build a trace by hand with no agent run
and no disk at all.
"""

RULES = ("V1", "V2", "V3", "V4", "V5", "V6", "V7", "V8")

#: V1 detail codes that are stop-level ("a stop is in the wrong place"), as
#: opposed to bad_city/no_city_point, which are day-level and would inflate a
#: per-stop rate if pooled the same way.
_WRONG_LOCATION_CODES = ("state_mismatch", "out_of_radius")


def _truncate_at_validation(events: list[dict], upto_attempt: int) -> list[dict]:
    """Every event up to and including the first validation at this attempt."""
    truncated = []
    for event in events:
        truncated.append(event)
        if event["kind"] == "validation" and event["attempt"] == upto_attempt:
            break
    return truncated


def _tokens(llm_calls: list[dict], *, shared: bool) -> int:
    return sum(
        (e.get("prompt_tokens") or 0) + (e.get("completion_tokens") or 0)
        for e in llm_calls
        if bool(e.get("shared")) == shared
    )


def _empty_row(arm: str, *, tool_calls: int, tokens_gather: int, tokens_arm: int, seconds: float,
                cap_hit: bool, interests_requested: int) -> dict:
    """A row for a trace with no validation event at all (e.g. a geocode failure)."""
    return {
        "arm": arm,
        "ok": False,
        "repairs": 0,
        "tool_calls": tool_calls,
        "tokens_gather": tokens_gather,
        "tokens_arm": tokens_arm,
        "tokens_total": tokens_gather + tokens_arm,
        "seconds": seconds,
        "stops": 0,
        "stops_proposed": 0,
        "day_km": [],
        "avg_day_km": 0.0,
        "estimated_share": 0.0,
        "unknown_count": 0,
        "priced_share": 0.0,
        "cap_hit": cap_hit,
        "interests_requested": interests_requested,
        "interests_covered": 0,
        "results": [],
    }


def row_from_trace(
    trace: dict, *, arm: str, upto_attempt: int | None = None, interests: list[str] | None = None
) -> dict:
    """One comparable row from a trace dict.

    Arm B is arm C's own event stream, truncated at (and including) the first
    validation event: ``row_from_trace(c_trace, arm="B", upto_attempt=0)``.
    Arm A and arm C pass their own full trace with upto_attempt=None.

    ``interests`` is the request's requested-interest list. Pass it explicitly
    when the caller already has the TripRequest in hand -- ``trace["request"]``
    is only populated on the gather trace, not on the A/B/C traces that adopt
    its events, so falling back to it here always reads an empty list for them.
    """
    events = trace["events"]
    if upto_attempt is not None:
        events = _truncate_at_validation(events, upto_attempt)

    tool_calls = sum(1 for e in events if e["kind"] == "tool_call")
    llm_calls = [e for e in events if e["kind"] == "llm_call"]
    tokens_gather = _tokens(llm_calls, shared=True)
    tokens_arm = _tokens(llm_calls, shared=False)
    seconds = sum(
        e.get("duration_ms") or 0 for e in events if e["kind"] in ("tool_call", "llm_call")
    ) / 1000
    cap_hit = any(
        e["kind"] == "note" and "tool step cap" in e.get("message", "") for e in events
    )
    if interests is None:
        interests = trace.get("request", {}).get("interests", [])
    interests_requested = len(interests)

    validations = [e for e in events if e["kind"] == "validation"]
    if not validations:
        return _empty_row(
            arm,
            tool_calls=tool_calls,
            tokens_gather=tokens_gather,
            tokens_arm=tokens_arm,
            seconds=seconds,
            cap_hit=cap_hit,
            interests_requested=interests_requested,
        )

    validation = validations[-1]
    results = validation.get("results") or []

    missing_interests = sum(
        1
        for r in results
        if not r["summary"]
        and r["status"] == "fail"
        and r["rule"] == "V3"
        and r["code"] == "missing_interest"
    )
    interests_covered = max(interests_requested - missing_interests, 0)

    day_km = validation.get("day_km") or []
    avg_day_km = sum(day_km) / len(day_km) if day_km else 0.0

    stops = validation.get("stops") or 0
    unknown_count = validation.get("unknown_count") or 0
    priced_share = (stops - unknown_count) / stops if stops else 0.0

    return {
        "arm": arm,
        "ok": bool(validation["ok"]),
        "repairs": validation["attempt"],
        "tool_calls": tool_calls,
        "tokens_gather": tokens_gather,
        "tokens_arm": tokens_arm,
        "tokens_total": tokens_gather + tokens_arm,
        "seconds": seconds,
        "stops": stops,
        "stops_proposed": validation.get("stops_proposed") or 0,
        "day_km": day_km,
        "avg_day_km": avg_day_km,
        "estimated_share": validation.get("estimated_share") or 0.0,
        "unknown_count": unknown_count,
        "priced_share": priced_share,
        "cap_hit": cap_hit,
        "interests_requested": interests_requested,
        "interests_covered": interests_covered,
        "results": results,
    }


def _rule_detail_count(results: list[dict], rule: str, *, codes: tuple[str, ...] | None = None) -> int:
    return sum(
        1
        for r in results
        if not r["summary"]
        and r["status"] == "fail"
        and r["rule"] == rule
        and (codes is None or r["code"] in codes)
    )


def aggregate(rows: list[dict]) -> dict:
    """Metrics over the rows passed in -- the caller decides which rows (one
    arm, paired trips only, all finished trips, ...). An empty list reports
    n=0 and zeroed rates rather than raising or dividing by zero.
    """
    n = len(rows)
    if n == 0:
        return {
            "n": 0,
            "pass_rate": 0.0,
            "rule_fail_share": {rule: 0.0 for rule in RULES},
            "rule_detail_count": {rule: 0 for rule in RULES},
            "hallucination_rate": 0.0,
            "wrong_location_rate": 0.0,
            "interest_coverage": None,
            "repairs_per_trip": 0.0,
            "tool_calls_per_trip": 0.0,
            "tokens_total_per_trip": 0.0,
            "tokens_arm_per_trip": 0.0,
            "seconds_per_trip": 0.0,
            "avg_day_km": 0.0,
            "cap_hit_rate": 0.0,
            "estimated_share": 0.0,
            "unknown_count_per_trip": 0.0,
            "priced_share": 0.0,
        }

    def mean(key: str) -> float:
        return sum(r[key] for r in rows) / n

    rule_fail_share: dict[str, float] = {}
    rule_detail_count: dict[str, int] = {}
    for rule in RULES:
        fail_rows = 0
        detail_count = 0
        for row in rows:
            summary = next(
                (r for r in row["results"] if r["rule"] == rule and r["summary"]), None
            )
            if summary is not None and summary["status"] == "fail":
                fail_rows += 1
            detail_count += _rule_detail_count(row["results"], rule)
        rule_fail_share[rule] = fail_rows / n
        rule_detail_count[rule] = detail_count

    total_stops_proposed = sum(row["stops_proposed"] for row in rows)
    hallucinated = sum(
        _rule_detail_count(row["results"], "V5", codes=("untraceable",)) for row in rows
    )
    wrong_location = sum(
        _rule_detail_count(row["results"], "V1", codes=_WRONG_LOCATION_CODES) for row in rows
    )
    hallucination_rate = hallucinated / total_stops_proposed if total_stops_proposed else 0.0
    wrong_location_rate = wrong_location / total_stops_proposed if total_stops_proposed else 0.0

    total_requested = sum(row["interests_requested"] for row in rows)
    total_covered = sum(row["interests_covered"] for row in rows)
    interest_coverage = total_covered / total_requested if total_requested else None

    return {
        "n": n,
        "pass_rate": mean("ok"),
        "rule_fail_share": rule_fail_share,
        "rule_detail_count": rule_detail_count,
        "hallucination_rate": hallucination_rate,
        "wrong_location_rate": wrong_location_rate,
        "interest_coverage": interest_coverage,
        "repairs_per_trip": mean("repairs"),
        "tool_calls_per_trip": mean("tool_calls"),
        "tokens_total_per_trip": mean("tokens_total"),
        "tokens_arm_per_trip": mean("tokens_arm"),
        "seconds_per_trip": mean("seconds"),
        "avg_day_km": mean("avg_day_km"),
        "cap_hit_rate": mean("cap_hit"),
        "estimated_share": mean("estimated_share"),
        "unknown_count_per_trip": mean("unknown_count"),
        "priced_share": mean("priced_share"),
    }
