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


def _missing_interest_result(interest: str) -> dict:
    return {
        "rule": "V3", "code": "missing_interest", "status": "fail",
        "message": f"no stop anywhere in the trip covers the requested interest {interest!r}",
        "summary": False, "day": None, "stop": None, "source_id": None,
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


# --- interest coverage: recomputed from trips.json + V3 results, never the
# --- stored interests_requested/interests_covered fields on the row itself ---


def test_interest_coverage_drops_below_100_percent_on_a_v3_missing_interest_detail():
    trips_by_id = {"p1": ["food", "museums"]}
    rows = [
        _row("p1", "A", ok=False, results=[_missing_interest_result("food")]),
        _row("p1", "B", ok=False, results=[_missing_interest_result("food")]),
        _row("p1", "C", ok=False, results=[_missing_interest_result("food")]),
    ]

    markdown = render(rows, trips_by_id=trips_by_id)
    headline, _secondary = markdown.split("## All finished trips", 1)

    # requested=2 (food, museums), covered=1 (museums only) -> 50.0%
    assert "50.0%" in headline


def test_a_trip_with_no_requested_interests_reports_n_a_not_100_percent():
    trips_by_id = {"p1": []}
    rows = [
        _row("p1", "A", ok=True),
        _row("p1", "B", ok=True),
        _row("p1", "C", ok=True),
    ]

    markdown = render(rows, trips_by_id=trips_by_id)
    coverage_line = next(line for line in markdown.splitlines() if "Interest coverage" in line)

    assert "n/a" in coverage_line
    assert "100.0%" not in coverage_line


def test_old_format_zero_zero_row_is_recomputed_from_trips_json_not_trusted():
    trips_by_id = {"p1": ["history", "food"]}
    rows = [
        _row("p1", "A", ok=False, results=[_missing_interest_result("history")]),
        _row("p1", "B", ok=False, results=[_missing_interest_result("history")]),
        _row("p1", "C", ok=False, results=[_missing_interest_result("history")]),
    ]
    # the bug this guards against: every stored row already claims 0/0
    rows = [{**r, "interests_requested": 0, "interests_covered": 0} for r in rows]

    markdown = render(rows, trips_by_id=trips_by_id)
    headline, _secondary = markdown.split("## All finished trips", 1)

    # requested=2, covered=1 (food) -> 50.0%, not n/a and not 100%
    assert "50.0%" in headline
    assert "n/a" not in headline


def test_pooled_interest_coverage_sums_covered_and_requested_across_trips():
    trips_by_id = {
        "p1": ["food", "museums"],  # 1 of 2 covered
        "p2": ["history"],  # fully covered
    }
    rows = [
        _row("p1", "A", ok=False, results=[_missing_interest_result("food")]),
        _row("p1", "B", ok=False, results=[_missing_interest_result("food")]),
        _row("p1", "C", ok=False, results=[_missing_interest_result("food")]),
        _row("p2", "A", ok=True),
        _row("p2", "B", ok=True),
        _row("p2", "C", ok=True),
    ]

    markdown = render(rows, trips_by_id=trips_by_id)
    headline, _secondary = markdown.split("## All finished trips", 1)

    # pooled: covered (1+1) / requested (2+1) = 66.7%, not mean(50%, 100%) = 75.0%
    assert "66.7%" in headline
    assert "75.0%" not in headline


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


def test_rule_share_counts_a_detail_only_failure_with_a_passing_summary():
    # the bug this guards against: the agent merged a V5 "untraceable" detail in
    # while building stops, but the validator's own V5 summary result passed, so
    # the table printed 0.0% for a rule that in fact failed on this trip.
    rows = [
        _row(
            "p1", "B", ok=False,
            results=[_result("V5", "traceable", "pass", summary=True), _result("V5", "untraceable", "fail")],
        ),
        _row("p1", "A", ok=True, results=[_result("V5", "traceable", "pass", summary=True)]),
        _row("p1", "C", ok=True, results=[_result("V5", "traceable", "pass", summary=True)]),
        _row(
            "p2", "B", ok=True,
            results=[_result("V5", "traceable", "pass", summary=True)],
        ),
        _row("p2", "A", ok=True, results=[_result("V5", "traceable", "pass", summary=True)]),
        _row("p2", "C", ok=True, results=[_result("V5", "traceable", "pass", summary=True)]),
    ]

    markdown = render(rows)

    from evals.metrics import aggregate

    metrics = aggregate([r for r in rows if r["arm"] == "B"])
    assert metrics["rule_fail_share"]["V5"] == 0.5  # 1 of 2 B trips has a V5 failure

    headline, _secondary = markdown.split("## All finished trips", 1)
    rule_section = headline.split("### Failures by rule")[1]
    # B's V5 row must show the computed 50.0% share, not 0.0%, even though no
    # V5 summary result ever failed.
    v5_row = next(line for line in rule_section.splitlines() if line.startswith("| V5"))
    cells = [c.strip() for c in v5_row.strip("|").split("|")]
    assert cells[2] == "50.0% (1 detail(s))"  # columns: Rule, A, B, C


def test_rule_share_counts_a_v8_detail_only_failure_in_the_all_finished_table():
    rows = [
        _row(
            "p1", "C", ok=False,
            results=[_result("V8", "sized", "pass", summary=True), _result("V8", "bad_day_number", "fail")],
        ),
        _row("p1", "A", ok=True, results=[_result("V8", "sized", "pass", summary=True)]),
        _row("p1", "B", ok=True, results=[_result("V8", "sized", "pass", summary=True)]),
        # u1 is unpaired (arm A missing) so it only shows up in the all-finished table
        _row("u1", "C", ok=True, results=[_result("V8", "sized", "pass", summary=True)]),
        _row("u1", "B", ok=True, results=[_result("V8", "sized", "pass", summary=True)]),
    ]

    markdown = render(rows)

    from evals.metrics import aggregate

    all_c_rows = [r for r in rows if r["arm"] == "C"]
    metrics = aggregate(all_c_rows)
    assert metrics["rule_fail_share"]["V8"] == 0.5  # 1 of 2 C trips (p1, u1)

    _headline, secondary = markdown.split("## All finished trips", 1)
    rule_section = secondary.split("### Failures by rule")[1]
    v8_row = next(line for line in rule_section.splitlines() if line.startswith("| V8"))
    cells = [c.strip() for c in v8_row.strip("|").split("|")]
    assert cells[3] == "50.0% (1 detail(s))"  # columns: Rule, A, B, C
