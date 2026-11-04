from datetime import date

import pytest
from pydantic import ValidationError

from evals import build_trip_request, load_trips
from src.parsing import LocationError
from src.tools import INTEREST_CATEGORIES

TODAY = date(2026, 11, 4)


def _trips() -> list[dict]:
    return load_trips()["trips"]


def test_trips_json_parses_with_schema_version_1():
    data = load_trips()
    assert data["schema_version"] == 1
    assert len(data["trips"]) == 30


def test_trip_ids_are_unique():
    ids = [t["trip_id"] for t in _trips()]
    assert len(ids) == len(set(ids))


def test_every_plan_trip_builds_a_trip_request():
    for trip in _trips():
        if trip["expect"] != "plan":
            continue
        request = build_trip_request(trip, today=TODAY)
        assert request.route == trip["request"]["route"]


def test_the_one_reject_trip_raises():
    reject_trips = [t for t in _trips() if t["expect"] == "reject"]
    assert len(reject_trips) == 1
    trip = reject_trips[0]
    assert trip["request"]["route"] == ["Toronto, ON"]
    with pytest.raises((LocationError, ValidationError)):
        build_trip_request(trip, today=TODAY)


# --- Coverage properties: the set must keep these, or the comparison silently narrows ---


def test_city_count_composition_is_12_12_5_1():
    plan_trips = [t for t in _trips() if t["expect"] == "plan"]
    reject_trips = [t for t in _trips() if t["expect"] == "reject"]

    by_city_count: dict[int, int] = {}
    for trip in plan_trips:
        n = len(trip["request"]["route"])
        by_city_count[n] = by_city_count.get(n, 0) + 1

    assert by_city_count == {1: 12, 2: 12, 3: 5}
    assert len(reject_trips) == 1


def test_ambiguous_city_pairs_are_present_as_separate_trips():
    routes = [t["request"]["route"] for t in _trips()]

    for city_a, city_b in [
        ("Portland, OR", "Portland, ME"),
        ("Springfield, IL", "Springfield, MA"),
        ("Columbus, OH", "Columbus, GA"),
        ("Kansas City, MO", "Kansas City, KS"),
    ]:
        assert [city_a] in routes, f"missing single-city trip for {city_a!r}"
        assert [city_b] in routes, f"missing single-city trip for {city_b!r}"

    assert ["Portland, OR", "Portland, ME"] in routes


def test_exactly_one_small_town_trip():
    small_town = [t for t in _trips() if "small-town" in t["tags"]]
    assert len(small_town) == 1
    assert small_town[0]["request"]["route"] == ["Marfa, TX"]


def _plan_trips() -> list[dict]:
    return [t for t in _trips() if t["expect"] == "plan"]


def test_every_interest_appears_alone_in_some_trip():
    for interest in INTEREST_CATEGORIES:
        assert any(
            t["request"]["interests"] == [interest] for t in _plan_trips()
        ), f"no trip covers {interest!r} alone"


def test_several_pair_and_all_four_and_empty_interest_trips_exist():
    pairs = [t for t in _plan_trips() if len(t["request"]["interests"]) == 2]
    all_four = [
        t for t in _plan_trips() if set(t["request"]["interests"]) == set(INTEREST_CATEGORIES)
    ]
    empty = [t for t in _plan_trips() if t["request"]["interests"] == []]

    assert len(pairs) >= 2
    assert len(all_four) == 1
    assert len(empty) == 1


def test_one_tight_and_one_loose_budget_trip_exist():
    tight = [t for t in _trips() if "budget-tight" in t["tags"]]
    loose = [t for t in _trips() if "budget-loose" in t["tags"]]
    assert len(tight) == 1
    assert len(loose) == 1
    assert tight[0]["request"]["budget_usd"] < loose[0]["request"]["budget_usd"]


def test_exactly_one_forecast_edge_trip_at_start_in_days_14():
    edge = [t for t in _trips() if t["request"].get("start_in_days") == 14]
    assert len(edge) == 1
    assert "forecast-edge" in edge[0]["tags"]
