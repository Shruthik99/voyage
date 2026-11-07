import html
from datetime import date

from src.mapview import _popup_html, build_map, main, save_map
from src.models import Day, ForecastDay, Itinerary, Stop

SCRIPT_NAME = "<script>alert(1)</script>"


def _stop(**over) -> Stop:
    base = {
        "name": "Denver Art Museum",
        "address": "100 W 14th Ave Pkwy",
        "state": "CO",
        "lat": 39.7372,
        "lon": -104.9897,
        "interest": "museums",
        "source": "geoapify",
        "source_id": "p1",
        "cost_low_usd": 10.0,
        "cost_high_usd": 20.0,
        "cost_basis": "per_person",
        "cost_source": "nps",
        "cost_confidence": "known",
    }
    return Stop(**{**base, **over})


def _unknown_cost_stop(**over) -> Stop:
    return _stop(cost_low_usd=None, cost_high_usd=None, cost_confidence="unknown", **over)


def _fixture_itinerary(verified: bool = False) -> Itinerary:
    stop_a = _stop(name="Denver Art Museum", source_id="p1", lat=39.7372, lon=-104.9897)
    stop_b = _unknown_cost_stop(
        name=SCRIPT_NAME, source_id="p2", lat=39.75, lon=-104.99, interest="hiking"
    )
    stop_c = _stop(
        name="Boulder Falls", source_id="p3", lat=40.0150, lon=-105.2705, state="CO"
    )

    day1 = Day(
        date=date(2026, 1, 1),
        city="Denver, CO",
        stops=[stop_a, stop_b],
        forecast=ForecastDay(date=date(2026, 1, 1), rain_chance=10.0, high_f=55.0, source="open-meteo"),
    )
    day2 = Day(
        date=date(2026, 1, 2),
        city="Boulder, CO",
        stops=[stop_c],
        forecast=None,
        warnings=["weather unavailable for Boulder, CO: timeout"],
    )
    return Itinerary(days=[day1, day2], verified=verified)


def test_stop_names_and_marker_count(tmp_path):
    itinerary = _fixture_itinerary()
    out = tmp_path / "map.html"
    save_map(build_map(itinerary), out)
    content = out.read_text(encoding="utf-8")

    assert "Denver Art Museum" in content
    assert "Boulder Falls" in content
    assert html.escape(SCRIPT_NAME) in content

    total_stops = sum(len(day.stops) for day in itinerary.days)
    assert content.count("L.AwesomeMarkers.icon(") == total_stops


def test_lat_lon_appear_exactly_no_jitter(tmp_path):
    itinerary = _fixture_itinerary()
    out = tmp_path / "map.html"
    save_map(build_map(itinerary), out)
    content = out.read_text(encoding="utf-8")

    for day in itinerary.days:
        for stop in day.stops:
            assert str(stop.lat) in content
            assert str(stop.lon) in content


def test_empty_day_does_not_crash_and_legend_says_no_stops(tmp_path):
    itinerary = Itinerary(days=[Day(date=date(2026, 1, 1), city="Denver, CO", stops=[])])
    out = tmp_path / "map.html"
    save_map(build_map(itinerary), out)
    content = out.read_text(encoding="utf-8")

    assert "no stops" in content
    assert content.count("L.AwesomeMarkers.icon(") == 0


def test_unknown_cost_stop_shows_unknown_no_dollar_figure():
    stop = _unknown_cost_stop()
    popup = _popup_html(stop, "Denver, CO", None)

    assert "cost: unknown" in popup
    assert "$" not in popup


def test_known_cost_stop_shows_dollar_figure():
    stop = _stop()
    popup = _popup_html(stop, "Denver, CO", None)

    assert "cost: $10-$20" in popup


def test_script_name_is_escaped_in_popup():
    stop = _unknown_cost_stop(name=SCRIPT_NAME)
    popup = _popup_html(stop, "Denver, CO", None)

    assert SCRIPT_NAME not in popup
    assert html.escape(SCRIPT_NAME) in popup


def test_unverified_shows_not_verified_in_legend(tmp_path):
    itinerary = _fixture_itinerary(verified=False)
    out = tmp_path / "map.html"
    save_map(build_map(itinerary), out)
    content = out.read_text(encoding="utf-8")

    assert "NOT verified" in content


def test_verified_shows_verified_in_legend(tmp_path):
    itinerary = _fixture_itinerary(verified=True)
    out = tmp_path / "map.html"
    save_map(build_map(itinerary), out)
    content = out.read_text(encoding="utf-8")

    assert "NOT verified" not in content
    assert "verified" in content


def test_forecast_shown_or_unavailable():
    stop = _stop()
    forecast = ForecastDay(date=date(2026, 1, 1), rain_chance=25.0, high_f=70.0, source="open-meteo")
    assert "70F" in _popup_html(stop, "Denver, CO", forecast)
    assert "25%" in _popup_html(stop, "Denver, CO", forecast)
    assert "weather unavailable" in _popup_html(stop, "Denver, CO", None)


def test_cli_roundtrips_itinerary_json_to_html(tmp_path):
    itinerary = _fixture_itinerary()
    itinerary_path = tmp_path / "itinerary.json"
    itinerary_path.write_text(itinerary.model_dump_json(indent=2), encoding="utf-8")
    out_path = tmp_path / "trip_map.html"

    exit_code = main([str(itinerary_path), str(out_path)])

    assert exit_code == 0
    assert out_path.exists()
    content = out_path.read_text(encoding="utf-8")
    assert "Denver Art Museum" in content
    assert "Boulder Falls" in content
