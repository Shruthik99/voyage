from datetime import date
from pathlib import Path

from src.agent import AgentError, AgentResult, TripRequest
from src.demo import main, parse_args, render_plan
from src.llm import LLMError
from src.models import (
    Day,
    ForecastDay,
    Itinerary,
    RuleResult,
    Stop,
    ValidationContext,
    ValidationReport,
)


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


def _fixture_result(verified: bool = False) -> AgentResult:
    stop_a = _stop(name="Denver Art Museum", source_id="p1")
    stop_b = _stop(name="Boulder Falls", source_id="p2", state="CO", lat=40.0150, lon=-105.2705)

    day1 = Day(
        date=date(2026, 1, 1),
        city="Denver, CO",
        stops=[stop_a],
        forecast=ForecastDay(date=date(2026, 1, 1), rain_chance=10.0, high_f=55.0, source="open-meteo"),
    )
    day2 = Day(
        date=date(2026, 1, 2),
        city="Boulder, CO",
        stops=[stop_b],
        warnings=["weather unavailable for Boulder, CO: timeout"],
    )
    itinerary = Itinerary(days=[day1, day2], verified=verified)

    report = ValidationReport(
        results=[
            RuleResult(rule="V2", code="over_budget", status="fail", message="trip exceeds budget by $50"),
            RuleResult(rule="V1", code="radius_ok", status="pass", message="ok", summary=True),
        ],
        warnings=["weather unavailable for Boulder, CO: timeout"],
    )
    context = ValidationContext(travelers=2, budget_usd=400.0, interests=["museums"])

    return AgentResult(
        itinerary=itinerary,
        report=report,
        context=context,
        repairs=1,
        tool_calls=5,
        trace_path=Path("traces/test.json"),
    )


def test_render_plan_contains_verified_flag_cities_stops_and_failures():
    result = _fixture_result(verified=False)
    text = render_plan(result)

    assert "verified: False" in text
    assert "Denver, CO" in text
    assert "Boulder, CO" in text
    assert "Denver Art Museum" in text
    assert "Boulder Falls" in text
    assert "over_budget" in text
    assert "radius_ok" not in text


def test_render_plan_shows_day_warning_when_forecast_missing():
    result = _fixture_result()
    text = render_plan(result)

    assert "weather unavailable for Boulder, CO: timeout" in text


def test_parse_args_defaults():
    args = parse_args([])

    assert args.route == ["Denver, CO"]
    assert args.days == 2
    assert args.travelers == 2
    assert args.budget == 400.0
    assert args.meal == 40.0
    assert args.interests == ["museums", "hiking"]
    assert args.start_in_days == 3
    assert args.reasoning_effort == "low"
    assert args.out_dir == "output"


def test_main_prints_error_and_exits_1_on_agent_error(monkeypatch, capsys):
    def _raise(*args, **kwargs):
        raise AgentError("could not produce a result")

    monkeypatch.setattr("src.demo.plan_trip", _raise)

    exit_code = main(["--route", "Denver, CO"])

    assert exit_code == 1
    out = capsys.readouterr().out
    assert "could not produce a result" in out


def test_main_prints_error_and_exits_1_on_llm_error(monkeypatch, capsys):
    def _raise(*args, **kwargs):
        raise LLMError("the model is unavailable")

    monkeypatch.setattr("src.demo.plan_trip", _raise)

    exit_code = main([])

    assert exit_code == 1
    assert "the model is unavailable" in capsys.readouterr().out


def test_main_writes_itinerary_and_map_on_success(monkeypatch, tmp_path, capsys):
    result = _fixture_result()

    def _fake_plan_trip(request: TripRequest, **kwargs):
        return result

    monkeypatch.setattr("src.demo.plan_trip", _fake_plan_trip)

    out_dir = tmp_path / "out"
    exit_code = main(["--out-dir", str(out_dir)])

    assert exit_code == 0
    itinerary_path = out_dir / "itinerary.json"
    map_path = out_dir / "trip_map.html"
    assert itinerary_path.exists()
    assert map_path.exists()

    out = capsys.readouterr().out
    assert str(itinerary_path) in out
    assert str(map_path) in out
