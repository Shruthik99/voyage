from datetime import date

from src import validator as validator_module
from src.models import Day, GeoPoint, Itinerary, Stop, ValidationContext
from src.validator import validate

MIAMI_CITY = "Miami, FL"
MIAMI = GeoPoint(city="Miami", state="FL", lat=25.7617, lon=-80.1918)

# 0.22 degrees of latitude is ~24.5 km: inside RADIUS_KM of the centre, but two
# such stops on opposite sides are ~49 km apart, which is over MAX_DAY_KM.
NORTH_LAT = 25.9817
SOUTH_LAT = 25.5417
FAR_LAT = 26.5000  # ~82 km from the centre

STOP_FIELDS = {
    "name": "Perez Art Museum",
    "address": "1103 Biscayne Blvd, Miami, FL",
    "state": "FL",
    "lat": 25.7617,
    "lon": -80.1918,
    "interest": "museums",
    "source": "geoapify",
    "source_id": "p1",
    "cost_low_usd": 10.0,
    "cost_high_usd": 10.0,
    "cost_basis": "per_person",
    "cost_source": "osm:charge",
    "cost_confidence": "known",
}


def _stop(**over):
    return Stop(**{**STOP_FIELDS, **over})


def _stops(n, **over):
    return [_stop(source_id=f"p{i}", name=f"Stop {i}", **over) for i in range(1, n + 1)]


def _day(stops, **over):
    base = {"date": date(2026, 11, 3), "city": MIAMI_CITY, "stops": stops}
    return Day(**{**base, **over})


def _itinerary(days, **over):
    return Itinerary(days=days, **over)


def _context(itinerary=None, **over):
    base = {
        "city_points": {MIAMI_CITY: MIAMI},
        "travelers": 1,
        "budget_usd": 1000.0,
        "interests": [],
        "tool_results": (
            {(s.source, s.source_id) for d in itinerary.days for s in d.stops}
            if itinerary is not None
            else set()
        ),
    }
    return ValidationContext(**{**base, **over})


def _clean():
    """A two-stop day that passes every rule, with a context to match."""
    itinerary = _itinerary([_day(_stops(2))])
    return itinerary, _context(itinerary)


def _summary_for(report, rule):
    return next(r for r in report.results if r.rule == rule and r.summary)


def _details(report, rule):
    return [r for r in report.results if r.rule == rule and not r.summary]


def _codes(report, rule):
    return [r.code for r in _details(report, rule)]


# --- V1 Location ---


def test_v1_location_stop_in_state_and_radius_passes():
    report = validate(*_clean())
    assert _summary_for(report, "V1").status == "pass"


def test_v1_location_state_mismatch_fails():
    itinerary = _itinerary([_day([_stop(source_id="p1"), _stop(source_id="p2", state="GA")])])
    report = validate(itinerary, _context(itinerary))
    assert "state_mismatch" in _codes(report, "V1")
    assert _summary_for(report, "V1").status == "fail"
    assert report.ok is False


def test_v1_location_outside_radius_fails():
    itinerary = _itinerary([_day([_stop(source_id="p1"), _stop(source_id="p2", lat=FAR_LAT)])])
    report = validate(itinerary, _context(itinerary))
    assert "out_of_radius" in _codes(report, "V1")


def test_v1_location_none_state_within_radius_passes_with_warning():
    # SPEC: a None state falls back to the distance check and adds a warning.
    itinerary = _itinerary([_day([_stop(source_id="p1"), _stop(source_id="p2", state=None)])])
    report = validate(itinerary, _context(itinerary))
    assert _codes(report, "V1") == ["state_unverified"]
    assert _summary_for(report, "V1").status == "warn"
    assert report.ok is True
    assert any("no state" in w for w in report.warnings)


def test_v1_location_none_state_outside_radius_fails():
    # The missing state is not an excuse; distance alone then decides.
    itinerary = _itinerary(
        [_day([_stop(source_id="p1"), _stop(source_id="p2", state=None, lat=FAR_LAT)])]
    )
    report = validate(itinerary, _context(itinerary))
    assert "out_of_radius" in _codes(report, "V1")
    assert report.ok is False


def test_v1_location_missing_city_point_fails_without_raising():
    itinerary = _itinerary([_day(_stops(2))])
    report = validate(itinerary, _context(itinerary, city_points={}))
    assert "no_city_point" in _codes(report, "V1")
    assert report.ok is False


def test_v1_location_malformed_city_fails_without_raising():
    itinerary = _itinerary([_day(_stops(2), city="Miami")])
    report = validate(itinerary, _context(itinerary, city_points={}))
    assert "bad_city" in _codes(report, "V1")


def test_v1_location_failure_names_day_and_stop():
    itinerary = _itinerary(
        [_day(_stops(2)), _day([_stop(source_id="p3"), _stop(source_id="p4", state="GA")])]
    )
    report = validate(itinerary, _context(itinerary))
    failure = next(r for r in _details(report, "V1") if r.code == "state_mismatch")
    assert (failure.day, failure.stop, failure.source_id) == (1, 1, "p4")


# --- V2 Budget: four outcomes, three bases, unknown costs ---


def test_v2_budget_high_under_budget_passes_within():
    itinerary, _ = _clean()
    report = validate(itinerary, _context(itinerary, budget_usd=100.0))
    summary = _summary_for(report, "V2")
    assert (summary.code, summary.status) == ("within", "pass")


def test_v2_budget_low_over_budget_fails():
    itinerary = _itinerary([_day(_stops(2))])
    report = validate(itinerary, _context(itinerary, budget_usd=5.0))
    summary = _summary_for(report, "V2")
    assert (summary.code, summary.status) == ("over", "fail")
    assert report.ok is False


def test_v2_budget_straddling_budget_passes_with_uncertain_warning():
    itinerary = _itinerary([_day(_stops(2, cost_low_usd=10.0, cost_high_usd=100.0))])
    report = validate(itinerary, _context(itinerary, budget_usd=50.0))
    summary = _summary_for(report, "V2")
    assert (summary.code, summary.status) == ("uncertain", "warn")
    assert report.ok is True


def test_v2_budget_high_equal_to_budget_passes_within():
    itinerary = _itinerary([_day(_stops(2))])  # S_high = 20
    report = validate(itinerary, _context(itinerary, budget_usd=20.0))
    assert _summary_for(report, "V2").code == "within"


def test_v2_budget_low_equal_to_budget_does_not_fail():
    itinerary = _itinerary([_day(_stops(2, cost_low_usd=10.0, cost_high_usd=100.0))])
    report = validate(itinerary, _context(itinerary, budget_usd=20.0))
    summary = _summary_for(report, "V2")
    assert (summary.code, summary.status) == ("uncertain", "warn")


def test_v2_budget_per_person_multiplied_by_travelers():
    itinerary = _itinerary([_day(_stops(2))])  # 2 x $10 per person
    report = validate(itinerary, _context(itinerary, travelers=3, budget_usd=59.0))
    assert _summary_for(report, "V2").code == "over"  # 60 > 59


def test_v2_budget_per_group_counted_once():
    itinerary = _itinerary([_day(_stops(2, cost_basis="per_group"))])
    report = validate(itinerary, _context(itinerary, travelers=10, budget_usd=20.0))
    assert _summary_for(report, "V2").code == "within"  # 20, not 200


def test_v2_budget_per_vehicle_counted_once():
    itinerary = _itinerary([_day(_stops(2, cost_basis="per_vehicle"))])
    report = validate(itinerary, _context(itinerary, travelers=10, budget_usd=20.0))
    assert _summary_for(report, "V2").code == "within"


def test_v2_budget_mixed_bases_sum_correctly():
    itinerary = _itinerary(
        [
            _day(
                [
                    _stop(source_id="p1", cost_basis="per_person"),  # 10 x 2 = 20
                    _stop(source_id="p2", cost_basis="per_group"),  # 10
                ]
            )
        ]
    )
    report = validate(itinerary, _context(itinerary, travelers=2, budget_usd=30.0))
    assert _summary_for(report, "V2").code == "within"  # exactly 30


def test_v2_budget_unknown_cost_adds_incomplete_warning():
    itinerary = _itinerary(
        [
            _day(
                [
                    _stop(source_id="p1"),
                    _stop(
                        source_id="p2",
                        cost_low_usd=None,
                        cost_high_usd=None,
                        cost_confidence="unknown",
                    ),
                ]
            )
        ]
    )
    report = validate(itinerary, _context(itinerary, budget_usd=100.0))
    assert "incomplete" in _codes(report, "V2")
    assert report.unknown_count == 1


def test_v2_budget_incomplete_warning_lists_unknown_stops():
    itinerary = _itinerary(
        [
            _day(
                [
                    _stop(source_id="p1"),
                    _stop(
                        source_id="p2",
                        name="Mystery Tour",
                        cost_low_usd=None,
                        cost_high_usd=None,
                        cost_confidence="unknown",
                    ),
                ]
            )
        ]
    )
    report = validate(itinerary, _context(itinerary, budget_usd=100.0))
    listing = next(r for r in _details(report, "V2") if r.code == "incomplete" and r.stop is None)
    assert "Mystery Tour" in listing.message


def test_v2_budget_unknown_cost_still_emits_one_outcome_result():
    # Incomplete is additive: one unpriced stop must not mask a blown budget.
    itinerary = _itinerary(
        [
            _day(
                [
                    _stop(source_id="p1"),
                    _stop(
                        source_id="p2",
                        cost_low_usd=None,
                        cost_high_usd=None,
                        cost_confidence="unknown",
                    ),
                ]
            )
        ]
    )
    report = validate(itinerary, _context(itinerary, budget_usd=5.0))
    summary = _summary_for(report, "V2")
    assert (summary.code, summary.status) == ("over", "fail")
    assert report.ok is False
    assert "incomplete" in _codes(report, "V2")


def test_v2_budget_all_costs_unknown_warns_and_does_not_fail():
    itinerary = _itinerary(
        [_day(_stops(2, cost_low_usd=None, cost_high_usd=None, cost_confidence="unknown"))]
    )
    report = validate(itinerary, _context(itinerary, budget_usd=100.0))
    assert _summary_for(report, "V2").status == "pass"
    assert report.unknown_count == 2
    assert report.ok is True


def test_v2_budget_float_drift_does_not_flip_outcome():
    # In float arithmetic $0.01 + $0.14 sums to 15.000000000000002 cents, which
    # reads as over a $0.15 budget. Integer cents must land exactly on "within".
    itinerary = _itinerary(
        [
            _day(
                [
                    _stop(source_id="p1", cost_low_usd=0.01, cost_high_usd=0.01),
                    _stop(source_id="p2", cost_low_usd=0.14, cost_high_usd=0.14),
                ]
            )
        ]
    )
    report = validate(itinerary, _context(itinerary, budget_usd=0.15))
    assert _summary_for(report, "V2").code == "within"


def test_v2_budget_estimated_share_zero_labels_within_budget():
    itinerary, _ = _clean()
    report = validate(itinerary, _context(itinerary, budget_usd=100.0))
    assert report.estimated_share == 0.0
    assert "within budget" in _summary_for(report, "V2").message


def test_v2_budget_estimated_share_nonzero_labels_within_estimated_budget():
    itinerary = _itinerary([_day(_stops(2, cost_confidence="estimate"))])
    report = validate(itinerary, _context(itinerary, budget_usd=100.0))
    assert report.estimated_share == 1.0
    assert "within estimated budget" in _summary_for(report, "V2").message


def test_v2_budget_estimated_share_is_dollar_weighted():
    itinerary = _itinerary(
        [
            _day(
                [
                    _stop(source_id="p1", cost_low_usd=75.0, cost_high_usd=75.0),
                    _stop(
                        source_id="p2",
                        cost_low_usd=25.0,
                        cost_high_usd=25.0,
                        cost_confidence="estimate",
                    ),
                ]
            )
        ]
    )
    report = validate(itinerary, _context(itinerary, budget_usd=1000.0))
    assert report.estimated_share == 0.25  # by dollars, not by stop count


def test_v2_budget_all_free_stops_do_not_divide_by_zero():
    itinerary = _itinerary([_day(_stops(2, cost_low_usd=0.0, cost_high_usd=0.0))])
    report = validate(itinerary, _context(itinerary, budget_usd=10.0))
    assert report.estimated_share == 0.0


def test_v2_cost_summary_mismatch_warns():
    itinerary = _itinerary([_day(_stops(2, cost_confidence="estimate"))], estimated_share=0.0)
    report = validate(itinerary, _context(itinerary, budget_usd=100.0))
    assert "cost_summary_mismatch" in _codes(report, "V2")


# --- V3 Interests ---


def test_v3_interests_all_requested_covered_passes():
    itinerary, _ = _clean()
    report = validate(itinerary, _context(itinerary, interests=["museums"]))
    assert _summary_for(report, "V3").status == "pass"


def test_v3_interests_missing_interest_fails():
    itinerary, _ = _clean()
    report = validate(itinerary, _context(itinerary, interests=["museums", "hiking"]))
    assert _summary_for(report, "V3").status == "fail"


def test_v3_interests_failure_names_missing_interest():
    itinerary, _ = _clean()
    report = validate(itinerary, _context(itinerary, interests=["hiking"]))
    assert "hiking" in _details(report, "V3")[0].message


def test_v3_interests_no_requested_interests_passes():
    report = validate(*_clean())
    assert _summary_for(report, "V3").status == "pass"


# --- V4 No placeholders ---


def test_v4_placeholders_complete_fields_pass():
    report = validate(*_clean())
    assert _summary_for(report, "V4").status == "pass"


def test_v4_placeholders_empty_name_fails():
    itinerary = _itinerary([_day([_stop(source_id="p1"), _stop(source_id="p2", name="")])])
    report = validate(itinerary, _context(itinerary))
    assert _summary_for(report, "V4").status == "fail"


def test_v4_placeholders_whitespace_only_source_id_fails():
    itinerary = _itinerary([_day([_stop(source_id="p1"), _stop(source_id="   ")])])
    report = validate(itinerary, _context(itinerary))
    assert "placeholder" in _codes(report, "V4")


def test_v4_placeholders_empty_source_fails():
    itinerary = _itinerary([_day([_stop(source_id="p1"), _stop(source_id="p2", source="")])])
    report = validate(itinerary, _context(itinerary))
    assert "placeholder" in _codes(report, "V4")


# --- V5 Traceability ---


def test_v5_traceability_known_tool_result_passes():
    report = validate(*_clean())
    assert _summary_for(report, "V5").status == "pass"


def test_v5_traceability_unknown_source_id_fails():
    itinerary = _itinerary([_day(_stops(2))])
    report = validate(itinerary, _context(itinerary, tool_results={("geoapify", "p1")}))
    failure = _details(report, "V5")[0]
    assert (failure.code, failure.source_id) == ("untraceable", "p2")


def test_v5_traceability_matching_id_wrong_source_fails():
    itinerary = _itinerary([_day(_stops(2))])
    context = _context(itinerary, tool_results={("nps", "p1"), ("nps", "p2")})
    report = validate(itinerary, context)
    assert _summary_for(report, "V5").status == "fail"


def test_v5_traceability_empty_tool_results_fails_every_stop():
    itinerary = _itinerary([_day(_stops(2))])
    report = validate(itinerary, _context(itinerary, tool_results=set()))
    assert len(_details(report, "V5")) == 2


# --- V6 Daily distance ---


def test_v6_daily_distance_under_max_passes():
    report = validate(*_clean())
    assert _summary_for(report, "V6").status == "pass"


def test_v6_daily_distance_over_max_fails():
    itinerary = _itinerary(
        [_day([_stop(source_id="p1", lat=SOUTH_LAT), _stop(source_id="p2", lat=NORTH_LAT)])]
    )
    report = validate(itinerary, _context(itinerary))
    assert "day_too_far" in _codes(report, "V6")


def test_v6_daily_distance_single_stop_is_zero_km_passes():
    itinerary = _itinerary([_day([_stop(source_id="p1")])])
    report = validate(itinerary, _context(itinerary))
    assert _summary_for(report, "V6").status == "pass"  # V8 is what objects to a thin day


def test_v6_daily_distance_flags_only_the_failing_day():
    itinerary = _itinerary(
        [
            _day(_stops(2)),
            _day([_stop(source_id="p3", lat=SOUTH_LAT), _stop(source_id="p4", lat=NORTH_LAT)]),
        ]
    )
    report = validate(itinerary, _context(itinerary))
    failures = _details(report, "V6")
    assert len(failures) == 1 and failures[0].day == 1


def test_v6_can_fail_while_v1_passes():
    # RADIUS_KM=30 admits two stops ~49 km apart, over MAX_DAY_KM=40.
    itinerary = _itinerary(
        [_day([_stop(source_id="p1", lat=SOUTH_LAT), _stop(source_id="p2", lat=NORTH_LAT)])]
    )
    report = validate(itinerary, _context(itinerary))
    assert _summary_for(report, "V1").status == "pass"
    assert _summary_for(report, "V6").status == "fail"


# --- V7 No duplicates ---


def test_v7_duplicates_unique_ids_pass():
    report = validate(*_clean())
    assert _summary_for(report, "V7").status == "pass"


def test_v7_duplicates_repeated_id_fails():
    itinerary = _itinerary([_day([_stop(source_id="p1"), _stop(source_id="p1")])])
    report = validate(itinerary, _context(itinerary))
    assert "duplicate" in _codes(report, "V7")


def test_v7_duplicates_failure_points_at_second_occurrence():
    # Naming the first would have the repair step delete the wrong stop.
    itinerary = _itinerary([_day([_stop(source_id="p1"), _stop(source_id="p1")])])
    report = validate(itinerary, _context(itinerary))
    failure = _details(report, "V7")[0]
    assert (failure.day, failure.stop) == (0, 1)


def test_v7_duplicates_across_different_days_detected():
    itinerary = _itinerary(
        [
            _day([_stop(source_id="p1"), _stop(source_id="p2")]),
            _day([_stop(source_id="p3"), _stop(source_id="p1")]),
        ]
    )
    report = validate(itinerary, _context(itinerary))
    failure = _details(report, "V7")[0]
    assert (failure.day, failure.stop) == (1, 1)


# --- V8 Day size ---


def test_v8_day_size_within_bounds_passes():
    report = validate(*_clean())
    assert _summary_for(report, "V8").status == "pass"


def test_v8_day_size_below_min_stops_fails():
    itinerary = _itinerary([_day([_stop(source_id="p1")])])
    report = validate(itinerary, _context(itinerary))
    assert "too_few_stops" in _codes(report, "V8")


def test_v8_day_size_above_max_stops_fails():
    itinerary = _itinerary([_day(_stops(6))])
    report = validate(itinerary, _context(itinerary))
    assert "too_many_stops" in _codes(report, "V8")


def test_v8_day_size_empty_day_fails():
    itinerary = _itinerary([_day([])])
    report = validate(itinerary, _context(itinerary))
    assert "too_few_stops" in _codes(report, "V8")


# --- Report level ---


def test_validate_emits_one_summary_result_per_rule():
    report = validate(*_clean())
    summaries = [r.rule for r in report.results if r.summary]
    assert summaries == ["V1", "V2", "V3", "V4", "V5", "V6", "V7", "V8"]


def test_validate_clean_itinerary_ok_true():
    report = validate(*_clean())
    assert report.ok is True
    assert report.warnings == []


def test_validate_per_stop_results_only_on_failure():
    report = validate(*_clean())
    assert len(report.results) == 8


def test_validate_warnings_only_ok_true():
    itinerary = _itinerary([_day([_stop(source_id="p1"), _stop(source_id="p2", state=None)])])
    report = validate(itinerary, _context(itinerary))
    assert report.ok is True
    assert report.warnings


def test_validate_any_failure_ok_false():
    itinerary = _itinerary([_day([_stop(source_id="p1", state="GA"), _stop(source_id="p2")])])
    report = validate(itinerary, _context(itinerary))
    assert report.ok is False


def test_validate_empty_itinerary_fails():
    # Otherwise every rule passes vacuously and a trip of nothing is blessed.
    report = validate(_itinerary([]), _context())
    assert report.ok is False
    assert "no_days" in _codes(report, "V8")


def test_validate_day_with_zero_stops_does_not_raise():
    itinerary = _itinerary([_day([])])
    assert validate(itinerary, _context(itinerary)).ok is False


def test_validate_does_not_mutate_itinerary():
    itinerary, context = _clean()
    before = itinerary.model_dump()
    validate(itinerary, context)
    assert itinerary.model_dump() == before


def test_validate_result_order_is_deterministic():
    itinerary, context = _clean()
    first = [(r.rule, r.code, r.day, r.stop) for r in validate(itinerary, context).results]
    second = [(r.rule, r.code, r.day, r.stop) for r in validate(itinerary, context).results]
    assert first == second


def test_validate_report_serializes_to_json():
    report = validate(*_clean())
    assert '"ok":true' in report.model_dump_json().replace(" ", "")


# --- Constants are patched on the validator module, not on src.constants ---


def test_validate_radius_km_monkeypatch_changes_outcome(monkeypatch):
    itinerary, context = _clean()
    assert validate(itinerary, context).ok is True
    monkeypatch.setattr(validator_module, "RADIUS_KM", 0)
    assert "out_of_radius" not in _codes(validate(itinerary, context), "V1")  # exactly 0 km away

    far = _itinerary([_day([_stop(source_id="p1"), _stop(source_id="p2", lat=NORTH_LAT)])])
    assert "out_of_radius" in _codes(validate(far, _context(far)), "V1")


def test_validate_max_day_km_monkeypatch_changes_outcome(monkeypatch):
    itinerary = _itinerary(
        [_day([_stop(source_id="p1"), _stop(source_id="p2", lat=NORTH_LAT)])]
    )
    context = _context(itinerary)
    assert _summary_for(validate(itinerary, context), "V6").status == "pass"
    monkeypatch.setattr(validator_module, "MAX_DAY_KM", 1)
    assert _summary_for(validate(itinerary, context), "V6").status == "fail"
