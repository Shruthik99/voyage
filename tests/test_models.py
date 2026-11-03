import pytest
from pydantic import ValidationError

from src.models import GeoPoint, RuleResult, Stop, ValidationContext, ValidationReport

STOP_FIELDS = {
    "name": "Perez Art Museum",
    "address": "1103 Biscayne Blvd, Miami, FL",
    "state": "FL",
    "lat": 25.7857,
    "lon": -80.1865,
    "interest": "museums",
    "source": "geoapify",
    "source_id": "p1",
    "cost_low_usd": 16.0,
    "cost_high_usd": 16.0,
    "cost_basis": "per_person",
    "cost_source": "osm:charge",
    "cost_confidence": "known",
}


def _stop(**over):
    return Stop(**{**STOP_FIELDS, **over})


def test_stop_inherits_every_place_field():
    stop = _stop()
    assert (stop.name, stop.address, stop.state, stop.interest) == (
        "Perez Art Museum",
        "1103 Biscayne Blvd, Miami, FL",
        "FL",
        "museums",
    )


# --- Cost invariants: our pricing code writes these, so a violation is our bug ---


def test_stop_cost_low_without_high_rejected():
    # Would add to S_low but not S_high, letting V2 call an over-budget trip "within".
    with pytest.raises(ValidationError):
        _stop(cost_high_usd=None)


def test_stop_cost_high_without_low_rejected():
    with pytest.raises(ValidationError):
        _stop(cost_low_usd=None)


def test_stop_cost_low_above_high_rejected():
    with pytest.raises(ValidationError):
        _stop(cost_low_usd=50.0, cost_high_usd=10.0)


def test_stop_negative_cost_rejected():
    with pytest.raises(ValidationError):
        _stop(cost_low_usd=-1.0, cost_high_usd=10.0)


def test_stop_unknown_confidence_requires_null_costs():
    with pytest.raises(ValidationError):
        _stop(cost_confidence="unknown")


def test_stop_unknown_confidence_with_null_costs_accepted():
    stop = _stop(cost_low_usd=None, cost_high_usd=None, cost_confidence="unknown")
    assert stop.cost_low_usd is None and stop.cost_high_usd is None


def test_stop_free_entry_accepted():
    stop = _stop(cost_low_usd=0.0, cost_high_usd=0.0)
    assert stop.cost_low_usd == 0.0


def test_stop_rejects_unknown_cost_basis():
    with pytest.raises(ValidationError):
        _stop(cost_basis="per_household")


# --- Permissive where the LLM writes: V4 reports these, the model must not ---


def test_stop_allows_empty_name_so_v4_can_report_it():
    assert _stop(name="").name == ""


def test_stop_allows_empty_source_id_so_v4_can_report_it():
    assert _stop(source_id="  ").source_id == "  "


# --- ValidationContext guards ---


def test_context_zero_travelers_rejected():
    # travelers=0 would zero every per_person cost and report "within budget".
    with pytest.raises(ValidationError):
        ValidationContext(travelers=0, budget_usd=100.0)


def test_context_negative_budget_rejected():
    with pytest.raises(ValidationError):
        ValidationContext(travelers=2, budget_usd=-1.0)


def test_context_accepts_geo_points_and_tool_results():
    context = ValidationContext(
        city_points={"Miami, FL": GeoPoint(city="Miami", state="FL", lat=25.7, lon=-80.1)},
        tool_results={("geoapify", "p1")},
        travelers=2,
        budget_usd=500.0,
        interests=["museums"],
    )
    assert ("geoapify", "p1") in context.tool_results


# --- ValidationReport.ok is computed, never stored ---


def _result(status):
    return RuleResult(rule="V1", code="located", status=status, message="m")


def test_report_ok_true_when_no_failures():
    assert ValidationReport(results=[_result("pass"), _result("warn")]).ok is True


def test_report_ok_false_when_any_failure():
    assert ValidationReport(results=[_result("pass"), _result("fail")]).ok is False


def test_report_ok_is_not_settable():
    # ok is derived from results; an inbound value must never override it.
    report = ValidationReport(results=[_result("fail")], ok=True)
    assert report.ok is False


def test_report_ok_survives_a_json_round_trip():
    # The trace is written and read back as JSON, so ok must serialize.
    report = ValidationReport(results=[_result("fail")])
    dumped = report.model_dump_json()
    assert '"ok":false' in dumped.replace(" ", "")
    assert ValidationReport.model_validate_json(dumped).ok is False
