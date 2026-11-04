from datetime import date

from src.agent import TripRequest
from src.models import GeoPoint, Place, Stop, ValidationContext
from src.pricing import price_place
from src.validator import validate

PLACE_FIELDS = {
    "name": "Perez Art Museum",
    "address": "1103 Biscayne Blvd, Miami, FL",
    "state": "FL",
    "lat": 25.7617,
    "lon": -80.1918,
    "interest": "museums",
    "source": "geoapify",
    "source_id": "p1",
}


def _place(**over) -> Place:
    return Place(**{**PLACE_FIELDS, **over})


def _request(**over) -> TripRequest:
    base = {
        "route": ["Miami, FL"],
        "days_per_stop": 2,
        "travelers": 2,
        "budget_usd": 500.0,
        "meal_allowance_usd": 40.0,
        "interests": ["museums"],
        "start_date": date(2026, 11, 3),
    }
    return TripRequest(**{**base, **over})


def test_price_place_returns_unknown_for_every_interest():
    request = _request()
    for interest in ("museums", "history", "hiking", "food"):
        cost = price_place(_place(interest=interest), request)
        assert cost.cost_low_usd is None
        assert cost.cost_high_usd is None
        assert cost.cost_confidence == "unknown"


def test_price_place_result_builds_a_valid_stop():
    cost = price_place(_place(), _request())
    stop = Stop(**_place().model_dump(), **cost.model_dump())
    assert stop.cost_confidence == "unknown"
    assert stop.cost_low_usd is None
    assert stop.cost_high_usd is None


def test_unpriced_itinerary_reports_v2_unpriced_and_ok_true():
    request = _request()
    cost = price_place(_place(), request)
    stop = Stop(**_place().model_dump(), **cost.model_dump())
    second = _place(source_id="p2")
    stop2 = Stop(**second.model_dump(), **price_place(second, request).model_dump())

    from src.models import Day, Itinerary

    itinerary = Itinerary(days=[Day(date=date(2026, 11, 3), city="Miami, FL", stops=[stop, stop2])])
    context = ValidationContext(
        city_points={"Miami, FL": GeoPoint(city="Miami", state="FL", lat=25.7617, lon=-80.1918)},
        tool_results={(stop.source, stop.source_id), (stop2.source, stop2.source_id)},
        travelers=2,
        budget_usd=500.0,
        interests=["museums"],
    )

    report = validate(itinerary, context)
    summary = next(r for r in report.results if r.rule == "V2" and r.summary)

    assert (summary.code, summary.status) == ("unpriced", "warn")
    assert report.estimated_share == 0.0
    assert report.unknown_count == 2
    assert report.ok is True
