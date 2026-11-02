import pytest

from src.parsing import LocationError, parse_location, parse_route


def test_parse_location_valid():
    assert parse_location("Miami, FL") == ("Miami", "FL")


def test_parse_location_lowercase_state():
    assert parse_location("Miami, fl") == ("Miami", "FL")


def test_parse_location_extra_whitespace():
    assert parse_location("  Miami  ,   FL  ") == ("Miami", "FL")


def test_parse_location_bad_state_code():
    with pytest.raises(LocationError):
        parse_location("Toronto, ON")


def test_parse_location_missing_state():
    with pytest.raises(LocationError):
        parse_location("Miami")


def test_parse_location_empty_city():
    with pytest.raises(LocationError):
        parse_location(", FL")


def test_parse_location_extra_commas():
    with pytest.raises(LocationError):
        parse_location("Miami, FL, USA")


def test_parse_route_three_stops():
    stops = ["San Jose, CA", "Miami, FL", "Boston, MA"]
    result = parse_route(stops)
    assert result == [("San Jose", "CA"), ("Miami", "FL"), ("Boston", "MA")]
    assert len(result) == 3
