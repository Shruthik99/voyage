from evals.summarize import render


def _result(rule: str, code: str, status: str, *, summary: bool = False) -> dict:
    return {
        "rule": rule, "code": code, "status": status, "message": "m",
        "summary": summary, "day": None, "stop": None, "source_id": None,
    }


def _row(trip_id: str, arm: str, *, ok: bool, outcome: str | None = None, results: list[dict] | None = None) -> dict:
    return {
        "schema_version": 1,
        "trip_id": trip_id,
        "arm": arm,
        "model": "openai/gpt-oss-120b",
        "reasoning_effort": "low",
        "written_at": "2026-11-04T00:00:00+00:00",
        "outcome": outcome or ("ok" if ok else "unverified"),
        "tokens_billed": 100,
        "error": None,
        "ok": ok,
        "repairs": 0,
        "tool_calls": 2,
        "tokens_gather": 50,
        "tokens_arm": 50,
        "tokens_total": 100,
        "seconds": 1.0,
        "stops": 2,
        "stops_proposed": 2,
        "day_km": [5.0],
        "avg_day_km": 5.0,
        "estimated_share": 0.0,
        "unknown_count": 0,
        "priced_share": 0.0,
        "cap_hit": False,
        "interests_requested": 1,
        "interests_covered": 1,
        "results": results if results is not None else [],
    }


def _rejected_row(trip_id: str, arm: str) -> dict:
    return {
        "schema_version": 1,
        "trip_id": trip_id,
        "arm": arm,
        "model": "openai/gpt-oss-120b",
        "reasoning_effort": "low",
        "written_at": "2026-11-04T00:00:00+00:00",
        "outcome": "rejected",
        "tokens_billed": 0,
        "error": "invalid US state code",
        "ok": False,
        "repairs": 0,
        "tool_calls": 0,
        "tokens_gather": 0,
        "tokens_arm": 0,
        "tokens_total": 0,
        "seconds": 0.0,
        "stops": 0,
        "stops_proposed": 0,
        "day_km": [],
        "avg_day_km": 0.0,
        "estimated_share": 0.0,
        "unknown_count": 0,
        "priced_share": 0.0,
        "cap_hit": False,
        "interests_requested": 0,
        "interests_covered": 0,
        "results": [],
    }


def _fixture() -> list[dict]:
    return [
        # p1: paired, A passes, B fails, C passes
        _row("p1", "A", ok=True),
        _row("p1", "B", ok=False),
        _row("p1", "C", ok=True),
        # p2: paired, A fails, B passes, C passes
        _row("p2", "A", ok=False),
        _row("p2", "B", ok=True),
        _row("p2", "C", ok=True),
        # u1: unpaired -- arm A never ran (e.g. the daily budget stopped mid-trip)
        _row("u1", "B", ok=True),
        _row("u1", "C", ok=False),
        # inv: invalid, rejected before any API call
        _rejected_row("inv", "A"),
        _rejected_row("inv", "B"),
        _rejected_row("inv", "C"),
    ]


def test_arm_a_is_labelled_llm_only_unrepaired_not_no_validator():
    markdown = render(_fixture())

    assert "LLM only, unrepaired" in markdown
    assert "no validator" not in markdown.lower()


def test_headline_uses_only_paired_trips_and_reports_both_n_counts():
    markdown = render(_fixture())

    # per-arm N over all finished (non-invalid) rows: A=2 (p1,p2), B=3, C=3
    assert "A=2" in markdown
    assert "B=3" in markdown
    assert "C=3" in markdown
    assert "Paired N: 2" in markdown
    assert "u1" in markdown  # named as the unpaired trip


def test_headline_pass_rate_is_computed_from_paired_rows_only():
    markdown = render(_fixture())

    # Headline (paired: p1, p2 only) -- A: [True, False] -> 50.0%; C: [True, True] -> 100.0%
    # These must appear in the first (headline) table, not just anywhere in the doc,
    # so split on the secondary-table heading and check the headline half only.
    headline, _secondary = markdown.split("## All finished trips", 1)
    assert "50.0%" in headline
    assert "100.0%" in headline


def test_secondary_table_includes_the_unpaired_trips_rows():
    markdown = render(_fixture())
    _, secondary = markdown.split("## All finished trips", 1)

    # All finished: B over p1,p2,u1 -> ok=[False,True,True] -> 66.7%
    # C over p1,p2,u1 -> ok=[True,True,False] -> 66.7%
    assert "66.7%" in secondary


def test_invalid_trip_is_reported_once_and_excluded_from_rates():
    markdown = render(_fixture())

    assert "1 invalid trip(s)" in markdown


def test_inconsistent_model_or_reasoning_effort_is_warned_about():
    rows = _fixture()
    rows[0] = {**rows[0], "model": "a-different-model"}

    markdown = render(rows)

    assert "**Warning:**" in markdown
    assert "model" in markdown


def test_force_rerun_rows_are_deduplicated_to_the_latest():
    rows = _fixture()
    # a later row for the same (trip_id, arm), as --force would append
    rows.append(_row("p1", "A", ok=False))

    markdown = render(rows)
    headline, _secondary = markdown.split("## All finished trips", 1)

    # A's pass rate over paired trips is now [False, False] -> 0.0%, not 50.0%
    assert "0.0%" in headline


def test_rule_failures_are_tallied_by_summary_share_and_detail_count():
    rows = [
        _row(
            "p1", "C", ok=False,
            results=[_result("V5", "traceable", "fail", summary=True), _result("V5", "untraceable", "fail")],
        ),
        _row("p1", "A", ok=True, results=[_result("V5", "traceable", "pass", summary=True)]),
        _row("p1", "B", ok=True, results=[_result("V5", "traceable", "pass", summary=True)]),
    ]

    markdown = render(rows)

    assert "V5" in markdown
    assert "1 detail(s)" in markdown
