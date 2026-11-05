from evals.metrics import aggregate, row_from_trace


def _result(rule: str, code: str, status: str, *, summary: bool = False) -> dict:
    return {
        "rule": rule,
        "code": code,
        "status": status,
        "message": "m",
        "summary": summary,
        "day": None,
        "stop": None,
        "source_id": None,
    }


def _llm_call(phase: str, *, prompt: int, completion: int, shared: bool = False) -> dict:
    return {
        "kind": "llm_call",
        "phase": phase,
        "model": "m",
        "has_tools": phase == "tools",
        "has_schema": phase != "tools",
        "duration_ms": 10.0,
        "prompt_tokens": prompt,
        "completion_tokens": completion,
        "shared": shared,
    }


def _tool_call(*, shared: bool = False) -> dict:
    return {
        "kind": "tool_call",
        "name": "search_places",
        "arguments": {},
        "summary": {"found": 1},
        "error": None,
        "duration_ms": 5.0,
        "shared": shared,
    }


def _validation(attempt: int, *, ok: bool, results: list[dict], stops: int, stops_proposed: int,
                 day_km: list[float]) -> dict:
    return {
        "kind": "validation",
        "attempt": attempt,
        "ok": ok,
        "estimated_share": 0.0,
        "unknown_count": 0,
        "stops": stops,
        "stops_proposed": stops_proposed,
        "day_km": day_km,
        "results": results,
    }


def _repair(attempt: int) -> dict:
    return {"kind": "repair", "attempt": attempt, "failures": []}


# --- Arm B: arm C's event stream, truncated at the first validation ---


def _c_trace_with_parse_retry() -> dict:
    return {
        "request": {"interests": ["museums", "food"]},
        "events": [
            _tool_call(shared=True),
            _llm_call("tools", prompt=100, completion=20, shared=True),
            _llm_call("propose", prompt=50, completion=10),  # malformed JSON retry
            _llm_call("propose", prompt=60, completion=15),  # the one that parsed
            _validation(
                0,
                ok=False,
                results=[_result("V8", "sized", "fail", summary=True)],
                stops=2,
                stops_proposed=2,
                day_km=[5.0],
            ),
            _repair(1),
            _llm_call("repair", prompt=70, completion=20),
            _validation(
                1,
                ok=True,
                results=[_result("V8", "sized", "pass", summary=True)],
                stops=3,
                stops_proposed=3,
                day_km=[6.0],
            ),
        ],
    }


def test_arm_b_truncates_before_the_repair_and_counts_both_parse_retries():
    trace = _c_trace_with_parse_retry()

    row = row_from_trace(trace, arm="B", upto_attempt=0)

    assert row["arm"] == "B"
    assert row["ok"] is False
    assert row["repairs"] == 0
    assert row["tool_calls"] == 1
    assert row["tokens_gather"] == 120  # the shared tool-phase call
    assert row["tokens_arm"] == (50 + 10) + (60 + 15)  # both propose calls, not the repair
    assert row["stops"] == 2
    assert row["stops_proposed"] == 2
    assert row["day_km"] == [5.0]


def test_arm_c_uses_the_final_validation_and_counts_the_repair():
    trace = _c_trace_with_parse_retry()

    row = row_from_trace(trace, arm="C")

    assert row["ok"] is True
    assert row["repairs"] == 1
    assert row["tokens_gather"] == 120
    assert row["tokens_arm"] == (50 + 10) + (60 + 15) + (70 + 20)
    assert row["stops"] == 3
    assert row["stops_proposed"] == 3


def test_stops_proposed_counts_dropped_stops_back_in():
    trace = {
        "request": {"interests": []},
        "events": [
            _validation(
                0,
                ok=False,
                results=[],
                stops=2,
                stops_proposed=3,  # one id was untraceable and dropped during the build
                day_km=[4.0],
            ),
        ],
    }

    row = row_from_trace(trace, arm="A")

    assert row["stops"] == 2
    assert row["stops_proposed"] == 3


def test_row_with_no_validation_event_is_not_ok_and_has_zero_denominators():
    trace = {
        "request": {"interests": ["food"]},
        "events": [
            {
                "kind": "tool_call",
                "name": "geocode",
                "arguments": {},
                "summary": None,
                "error": "no city result",
                "duration_ms": 1.0,
            }
        ],
    }

    row = row_from_trace(trace, arm="C")

    assert row["ok"] is False
    assert row["stops"] == 0
    assert row["stops_proposed"] == 0
    assert row["tool_calls"] == 1
    assert row["results"] == []


def test_interests_covered_subtracts_v3_missing_interest_details():
    trace = {
        "request": {"interests": ["food", "museums", "hiking"]},
        "events": [
            _validation(
                0,
                ok=False,
                results=[
                    _result("V3", "covered", "fail", summary=True),
                    _result("V3", "missing_interest", "fail"),
                ],
                stops=1,
                stops_proposed=1,
                day_km=[0.0],
            ),
        ],
    }

    row = row_from_trace(trace, arm="A")

    assert row["interests_requested"] == 3
    assert row["interests_covered"] == 2


def test_interests_param_overrides_the_traces_own_empty_request():
    # adopt() never copies the gather trace's request onto arms A/B/C, so
    # trace["request"] is always {} for them; the caller must pass interests
    # explicitly rather than relying on the trace to carry it.
    trace = {
        "request": {},
        "events": [
            _validation(
                0,
                ok=False,
                results=[
                    _result("V3", "covered", "fail", summary=True),
                    _result("V3", "missing_interest", "fail"),
                ],
                stops=1,
                stops_proposed=1,
                day_km=[0.0],
            ),
        ],
    }

    row = row_from_trace(trace, arm="C", interests=["food", "museums"])

    assert row["interests_requested"] == 2
    assert row["interests_covered"] == 1


def test_cap_hit_detected_from_the_tool_step_cap_note():
    trace = {
        "request": {"interests": []},
        "events": [
            {"kind": "note", "message": "tool step cap (12) reached; proposing with what was gathered"},
            _validation(0, ok=True, results=[], stops=0, stops_proposed=0, day_km=[]),
        ],
    }

    row = row_from_trace(trace, arm="A")

    assert row["cap_hit"] is True


# --- aggregate(): pooled rates, not a mean of per-trip ratios ---


def _row(*, ok: bool, results: list[dict], stops_proposed: int, repairs: int = 0) -> dict:
    return {
        "arm": "C",
        "ok": ok,
        "repairs": repairs,
        "tool_calls": 2,
        "tokens_gather": 100,
        "tokens_arm": 50,
        "tokens_total": 150,
        "seconds": 1.0,
        "stops": stops_proposed,
        "stops_proposed": stops_proposed,
        "day_km": [10.0],
        "avg_day_km": 10.0,
        "estimated_share": 0.0,
        "unknown_count": 0,
        "priced_share": 0.0,
        "cap_hit": False,
        "interests_requested": 1,
        "interests_covered": 1,
        "results": results,
    }


def test_aggregate_pools_hallucination_rate_rather_than_averaging_per_trip_ratios():
    rows = [
        _row(
            ok=False,
            results=[_result("V5", "untraceable", "fail") for _ in range(1)],
            stops_proposed=2,
        ),
        _row(
            ok=False,
            results=[_result("V5", "untraceable", "fail") for _ in range(1)],
            stops_proposed=8,
        ),
    ]

    metrics = aggregate(rows)

    # pooled: (1 + 1) / (2 + 8) = 0.2, not mean(0.5, 0.125) = 0.3125
    assert metrics["hallucination_rate"] == 0.2
    assert metrics["n"] == 2
    assert metrics["pass_rate"] == 0.0


def test_aggregate_rule_fail_share_and_detail_count():
    rows = [
        _row(ok=False, results=[_result("V1", "located", "fail", summary=True), _result("V1", "state_mismatch", "fail")], stops_proposed=4),
        _row(ok=True, results=[_result("V1", "located", "pass", summary=True)], stops_proposed=4),
    ]

    metrics = aggregate(rows)

    assert metrics["rule_fail_share"]["V1"] == 0.5
    assert metrics["rule_detail_count"]["V1"] == 1
    assert metrics["wrong_location_rate"] == 1 / 8


def test_aggregate_excludes_day_level_v1_codes_from_wrong_location_rate():
    rows = [
        _row(ok=False, results=[_result("V1", "bad_city", "fail"), _result("V1", "no_city_point", "fail")], stops_proposed=4),
    ]

    metrics = aggregate(rows)

    assert metrics["wrong_location_rate"] == 0.0


def test_aggregate_empty_rows_reports_zero_n_with_no_division_errors():
    metrics = aggregate([])

    assert metrics["n"] == 0
    assert metrics["pass_rate"] == 0.0
    assert metrics["hallucination_rate"] == 0.0
    assert metrics["rule_fail_share"]["V1"] == 0.0


def test_aggregate_interest_coverage_is_none_when_nothing_was_requested():
    rows = [_row(ok=True, results=[], stops_proposed=2)]
    rows[0]["interests_requested"] = 0
    rows[0]["interests_covered"] = 0

    metrics = aggregate(rows)

    assert metrics["interest_coverage"] is None


def test_aggregate_interest_coverage_pools_covered_over_requested():
    rows = [
        {**_row(ok=True, results=[], stops_proposed=2), "interests_requested": 2, "interests_covered": 1},
        {**_row(ok=True, results=[], stops_proposed=2), "interests_requested": 1, "interests_covered": 1},
    ]

    metrics = aggregate(rows)

    # pooled: (1 + 1) / (2 + 1) = 2/3, not mean(0.5, 1.0) = 0.75
    assert metrics["interest_coverage"] == 2 / 3
