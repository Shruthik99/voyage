import json
from datetime import date, timedelta

import httpx
import pytest
from openai.types.chat import ChatCompletion
from pydantic import ValidationError as PydanticValidationError

from src import agent as agent_module
from src import tools as tools_module
from src.agent import (
    MAX_REPAIR_ITEMS,
    AgentError,
    PlaceRegistry,
    ProposalFull,
    ProposalIds,
    ProposalStopFull,
    TripRequest,
    _build_context,
    _build_itinerary,
    _build_skeleton,
    _execute_tool,
    _failure_payload,
    plan_trip,
)
from src.constants import FORECAST_MAX_DAYS
from src.llm import LLMError
from src.models import GeoPoint, Place, RuleResult
from src.validator import validate

PROPOSAL_SCHEMAS = (agent_module.PROPOSAL_SCHEMA_IDS, agent_module.PROPOSAL_SCHEMA_FULL)


@pytest.fixture(autouse=True)
def _isolated_cache(tmp_path, monkeypatch):
    monkeypatch.setattr(tools_module, "CACHE_DIR", tmp_path / "cache")


@pytest.fixture(autouse=True)
def _api_key(monkeypatch):
    monkeypatch.setenv("GEOAPIFY_KEY", "test-key")


# --- request / response builders ---


def _request(**over) -> TripRequest:
    base = {
        "route": ["Austin, TX"],
        "days_per_stop": 1,
        "travelers": 2,
        "budget_usd": 500.0,
        "meal_allowance_usd": 40.0,
        "interests": ["museums"],
        "start_date": date(2026, 11, 3),
    }
    return TripRequest(**{**base, **over})


def _place(**over) -> Place:
    base = {
        "name": "Museum A",
        "address": "1 Main St",
        "state": "TX",
        "lat": 30.001,
        "lon": -97.001,
        "interest": "museums",
        "source": "geoapify",
        "source_id": "p1",
    }
    return Place(**{**base, **over})


class _FakeCompletions:
    def __init__(self, script):
        self.script = list(script)
        self.calls: list[dict] = []

    def create(self, **kwargs):
        # messages is mutated in place by later loop iterations, so snapshot it now
        # rather than storing the live list by reference.
        snapshot = {**kwargs, "messages": list(kwargs.get("messages", []))}
        self.calls.append(snapshot)
        if not self.script:
            raise AssertionError("FakeClient script exhausted")
        item = self.script.pop(0)
        if isinstance(item, BaseException):
            raise item
        return item


class _FakeChat:
    def __init__(self, script):
        self.completions = _FakeCompletions(script)


class FakeClient:
    def __init__(self, script):
        self.chat = _FakeChat(script)

    @property
    def calls(self) -> list[dict]:
        return self.chat.completions.calls


def _completion(message: dict, usage: dict | None = None) -> ChatCompletion:
    data = {
        "id": "chatcmpl-test",
        "object": "chat.completion",
        "created": 0,
        "model": "m",
        "choices": [{"index": 0, "finish_reason": "stop", "message": message}],
    }
    if usage is not None:
        data["usage"] = usage
    return ChatCompletion.model_validate(data)


def _tool_call_message(calls: list[tuple[str, dict]]) -> ChatCompletion:
    tool_calls = [
        {
            "id": f"call_{i}",
            "type": "function",
            "function": {"name": name, "arguments": json.dumps(args)},
        }
        for i, (name, args) in enumerate(calls)
    ]
    return _completion({"role": "assistant", "content": None, "tool_calls": tool_calls})


def _no_tool_calls(content: str = "") -> ChatCompletion:
    return _completion({"role": "assistant", "content": content})


def _proposal_message(payload: dict) -> ChatCompletion:
    return _completion({"role": "assistant", "content": json.dumps(payload)})


def _places_json(entries: list[dict]) -> dict:
    return {
        "features": [
            {
                "properties": {
                    "name": e["name"],
                    "formatted": e.get("address", ""),
                    "state_code": e.get("state"),
                    "lat": e["lat"],
                    "lon": e["lon"],
                    "place_id": e["source_id"],
                }
            }
            for e in entries
        ]
    }


def _geocode_response(request: httpx.Request) -> httpx.Response:
    text = request.url.params.get("text", "")
    parts = [p.strip() for p in text.split(",")]
    city, state = parts[0], parts[1]
    return httpx.Response(
        200,
        json={
            "results": [
                {
                    "city": city,
                    "name": city,
                    "state_code": state,
                    "result_type": "city",
                    "lat": 30.0,
                    "lon": -97.0,
                }
            ]
        },
    )


def make_handler(places_json: dict | None = None, forecast_json: dict | None = None):
    def handler(request: httpx.Request) -> httpx.Response:
        url = str(request.url)
        if "geocode/search" in url:
            return _geocode_response(request)
        if "v2/places" in url:
            return httpx.Response(200, json=places_json or {"features": []})
        if "open-meteo" in url:
            return httpx.Response(
                200,
                json=forecast_json
                or {"daily": {"time": [], "temperature_2m_max": [], "precipitation_probability_max": []}},
            )
        return httpx.Response(404)

    return handler


TWO_MUSEUMS = _places_json(
    [
        {"source_id": "p1", "name": "Museum A", "lat": 30.001, "lon": -97.001, "state": "TX"},
        {"source_id": "p2", "name": "Museum B", "lat": 30.002, "lon": -97.002, "state": "TX"},
    ]
)


# --- 1. Happy path ---


def test_plan_trip_happy_path(tmp_path):
    handler = make_handler(places_json=TWO_MUSEUMS)
    client = FakeClient(
        [
            _tool_call_message([("search_places", {"city_state": "Austin, TX", "interest": "museums"})]),
            _no_tool_calls(),
            _proposal_message({"days": [{"day": 1, "stop_ids": ["p1", "p2"]}]}),
        ]
    )

    with httpx.Client(transport=httpx.MockTransport(handler)) as http_client:
        result = plan_trip(_request(), client=client, http_client=http_client, trace_dir=tmp_path)

    assert result.report.ok is True
    assert result.itinerary.verified is True
    assert result.repairs == 0
    assert len(result.itinerary.days[0].stops) == 2


# --- 2. Tool error fed back ---


def test_plan_trip_tool_error_is_fed_back_to_the_model(tmp_path):
    handler = make_handler(places_json=TWO_MUSEUMS)
    client = FakeClient(
        [
            _tool_call_message([("geocode", {"city_state": "Atlantis, XX"})]),
            _tool_call_message([("search_places", {"city_state": "Austin, TX", "interest": "museums"})]),
            _no_tool_calls(),
            _proposal_message({"days": [{"day": 1, "stop_ids": ["p1", "p2"]}]}),
        ]
    )

    with httpx.Client(transport=httpx.MockTransport(handler)) as http_client:
        result = plan_trip(_request(), client=client, http_client=http_client, trace_dir=tmp_path)

    second_call_messages = client.calls[1]["messages"]
    tool_messages = [m for m in second_call_messages if m.get("role") == "tool"]
    assert len(tool_messages) == 1
    content = json.loads(tool_messages[0]["content"])
    assert "invalid" in content["error"].lower()
    assert result.report.ok is True


# --- 3. Tool step cap ---


def test_plan_trip_tool_step_cap_still_proposes(tmp_path):
    handler = make_handler()
    client = FakeClient(
        [_tool_call_message([("geocode", {"city_state": "Austin, TX"})]) for _ in range(3)]
        + [_proposal_message({"days": [{"day": 1, "stop_ids": []}]})]
    )

    with httpx.Client(transport=httpx.MockTransport(handler)) as http_client:
        result = plan_trip(
            _request(),
            client=client,
            http_client=http_client,
            trace_dir=tmp_path,
            max_tool_steps=3,
            max_repairs=0,
        )

    assert result.tool_calls == 3
    assert any("tool step cap" in w for w in result.itinerary.warnings)
    assert len(client.calls) == 4


# --- 4. Hallucinated id ---


def test_build_itinerary_hallucinated_id_is_untraceable_not_fabricated():
    registry = PlaceRegistry()
    skeleton = [(date(2026, 11, 3), "Austin, TX")]
    proposal = ProposalIds(days=[{"day": 1, "stop_ids": ["ghost"]}])

    itinerary, failures = _build_itinerary(proposal, skeleton, registry, _request(), "ids", False)

    assert itinerary.days[0].stops == []
    assert any(f.rule == "V5" and f.code == "untraceable" and f.source_id == "ghost" for f in failures)


# --- 5. Repair succeeds ---


def test_plan_trip_repair_succeeds(tmp_path):
    handler = make_handler(places_json=TWO_MUSEUMS)
    client = FakeClient(
        [
            _tool_call_message([("search_places", {"city_state": "Austin, TX", "interest": "museums"})]),
            _no_tool_calls(),
            _proposal_message({"days": [{"day": 1, "stop_ids": ["p1"]}]}),  # too few stops
            _proposal_message({"days": [{"day": 1, "stop_ids": ["p1", "p2"]}]}),  # fixed
        ]
    )

    with httpx.Client(transport=httpx.MockTransport(handler)) as http_client:
        result = plan_trip(_request(), client=client, http_client=http_client, trace_dir=tmp_path)

    assert result.repairs == 1
    assert result.report.ok is True
    assert result.itinerary.verified is True


# --- 6. Repair exhausted ---


def test_plan_trip_repair_exhausted(tmp_path):
    handler = make_handler(places_json=TWO_MUSEUMS)
    failing_proposal = _proposal_message({"days": [{"day": 1, "stop_ids": ["p1"]}]})
    client = FakeClient(
        [
            _tool_call_message([("search_places", {"city_state": "Austin, TX", "interest": "museums"})]),
            _no_tool_calls(),
            failing_proposal,
            failing_proposal,
            failing_proposal,
            failing_proposal,
        ]
    )

    with httpx.Client(transport=httpx.MockTransport(handler)) as http_client:
        result = plan_trip(_request(), client=client, http_client=http_client, trace_dir=tmp_path)

    assert result.repairs == 3
    assert result.report.ok is False
    assert result.itinerary.verified is False
    proposal_calls = [c for c in client.calls if "response_format" in c]
    assert len(proposal_calls) == 4


# --- 7. Warnings never trigger repair ---


def test_plan_trip_warnings_alone_do_not_trigger_repair(tmp_path):
    handler = make_handler(places_json=TWO_MUSEUMS)
    client = FakeClient(
        [
            _tool_call_message([("search_places", {"city_state": "Austin, TX", "interest": "museums"})]),
            _no_tool_calls(),
            _proposal_message({"days": [{"day": 1, "stop_ids": ["p1", "p2"]}]}),
        ]
    )

    with httpx.Client(transport=httpx.MockTransport(handler)) as http_client:
        result = plan_trip(_request(), client=client, http_client=http_client, trace_dir=tmp_path)

    assert result.repairs == 0
    assert result.report.ok is True
    assert any("budget not checked" in w for w in result.report.warnings)


# --- 8. Malformed output ---


def test_plan_trip_malformed_proposal_then_valid_succeeds(tmp_path):
    handler = make_handler(places_json=TWO_MUSEUMS)
    client = FakeClient(
        [
            _tool_call_message([("search_places", {"city_state": "Austin, TX", "interest": "museums"})]),
            _no_tool_calls(),
            _completion({"role": "assistant", "content": "not json"}),
            _proposal_message({"days": [{"day": 1, "stop_ids": ["p1", "p2"]}]}),
        ]
    )

    with httpx.Client(transport=httpx.MockTransport(handler)) as http_client:
        result = plan_trip(_request(), client=client, http_client=http_client, trace_dir=tmp_path)

    assert result.report.ok is True
    assert len(client.calls) == 4


def test_plan_trip_all_malformed_raises_agent_error(tmp_path):
    handler = make_handler(places_json=TWO_MUSEUMS)
    bad = _completion({"role": "assistant", "content": "not json"})
    client = FakeClient(
        [
            _tool_call_message([("search_places", {"city_state": "Austin, TX", "interest": "museums"})]),
            _no_tool_calls(),
            bad,
            bad,
            bad,
        ]
    )

    with httpx.Client(transport=httpx.MockTransport(handler)) as http_client, pytest.raises(AgentError):
        plan_trip(_request(), client=client, http_client=http_client, trace_dir=tmp_path)


# --- 9. content=None on a tool-call turn round-trips ---


def test_tool_call_message_content_none_round_trips_as_empty_string(tmp_path):
    handler = make_handler(places_json=TWO_MUSEUMS)
    client = FakeClient(
        [
            _tool_call_message([("search_places", {"city_state": "Austin, TX", "interest": "museums"})]),
            _no_tool_calls(),
            _proposal_message({"days": [{"day": 1, "stop_ids": ["p1", "p2"]}]}),
        ]
    )

    with httpx.Client(transport=httpx.MockTransport(handler)) as http_client:
        plan_trip(_request(), client=client, http_client=http_client, trace_dir=tmp_path)

    second_call_messages = client.calls[1]["messages"]
    assistant_entries = [m for m in second_call_messages if m.get("role") == "assistant" and "tool_calls" in m]
    assert len(assistant_entries) == 1
    assert assistant_entries[0]["content"] == ""
    assert assistant_entries[0]["tool_calls"][0]["function"]["name"] == "search_places"


# --- 10. content=None with no tool calls in the proposal phase ---


def test_proposal_phase_content_none_counts_as_parse_failure_not_a_crash(tmp_path):
    handler = make_handler(places_json=TWO_MUSEUMS)
    client = FakeClient(
        [
            _tool_call_message([("search_places", {"city_state": "Austin, TX", "interest": "museums"})]),
            _no_tool_calls(),
            _completion({"role": "assistant", "content": None}),
            _proposal_message({"days": [{"day": 1, "stop_ids": ["p1", "p2"]}]}),
        ]
    )

    with httpx.Client(transport=httpx.MockTransport(handler)) as http_client:
        result = plan_trip(_request(), client=client, http_client=http_client, trace_dir=tmp_path)

    assert result.report.ok is True
    assert len(client.calls) == 4


# --- 11. Phase separation ---


def test_tool_phase_and_proposal_phase_kwargs_are_disjoint(tmp_path):
    handler = make_handler(places_json=TWO_MUSEUMS)
    client = FakeClient(
        [
            _tool_call_message([("search_places", {"city_state": "Austin, TX", "interest": "museums"})]),
            _no_tool_calls(),
            _proposal_message({"days": [{"day": 1, "stop_ids": ["p1", "p2"]}]}),
        ]
    )

    with httpx.Client(transport=httpx.MockTransport(handler)) as http_client:
        plan_trip(_request(), client=client, http_client=http_client, trace_dir=tmp_path)

    tool_phase_calls = client.calls[:2]
    proposal_calls = client.calls[2:]

    for call in tool_phase_calls:
        assert "tools" in call
        assert "response_format" not in call

    for call in proposal_calls:
        assert "tools" not in call
        assert call["response_format"]["json_schema"]["strict"] is True


# --- 12. Repair payload ---


def test_failure_payload_only_fail_details_capped_with_six_fields():
    fails = [
        RuleResult(rule="V1", code="out_of_radius", status="fail", message=f"m{i}", day=0, stop=i, source_id=f"p{i}")
        for i in range(25)
    ]
    extras = [
        RuleResult(rule="V1", code="located", status="pass", message="ok", summary=True),
        RuleResult(rule="V2", code="unpriced", status="warn", message="warn", summary=True),
    ]

    payload = _failure_payload(fails + extras)

    assert len(payload) == MAX_REPAIR_ITEMS + 1
    assert payload[-1] == {"note": "...and 5 more"}
    for item in payload[:-1]:
        assert set(item.keys()) == {"rule", "code", "day", "stop", "source_id", "message"}


def test_failure_payload_sorts_none_day_and_stop_first():
    results = [
        RuleResult(rule="V8", code="too_many_stops", status="fail", message="m1", day=1, stop=None),
        RuleResult(rule="V8", code="no_days", status="fail", message="m0"),
        RuleResult(rule="V1", code="state_mismatch", status="fail", message="m2", day=0, stop=1),
    ]

    payload = _failure_payload(results)

    assert [p["code"] for p in payload] == ["state_mismatch", "no_days", "too_many_stops"]


# --- 13. No prices from the LLM ---


def test_proposal_schemas_never_mention_cost_fields():
    for schema in PROPOSAL_SCHEMAS:
        assert "cost" not in json.dumps(schema).lower()


def test_proposal_stop_full_rejects_a_smuggled_cost_field():
    with pytest.raises(PydanticValidationError):
        ProposalStopFull(
            name="A",
            address="1 Main St",
            state="TX",
            lat=30.0,
            lon=-97.0,
            interest="museums",
            source="geoapify",
            source_id="p1",
            cost_low_usd=10.0,
        )


# --- 14. stop_mode="full" exercises the real rules ---


def test_full_mode_wrong_state_stop_fails_v1():
    registry = PlaceRegistry()
    registry.record_point("Austin, TX", GeoPoint(city="Austin", state="TX", lat=30.0, lon=-97.0))
    skeleton = [(date(2026, 11, 3), "Austin, TX")]
    request = _request()
    proposal = ProposalFull(
        days=[
            {
                "day": 1,
                "stops": [
                    {
                        "name": "A",
                        "address": "123 Main",
                        "state": "TX",
                        "lat": 30.001,
                        "lon": -97.001,
                        "interest": "museums",
                        "source": "geoapify",
                        "source_id": "p1",
                    },
                    {
                        "name": "B",
                        "address": "456 Main",
                        "state": "GA",
                        "lat": 30.002,
                        "lon": -97.002,
                        "interest": "museums",
                        "source": "geoapify",
                        "source_id": "p2",
                    },
                ],
            }
        ]
    )

    itinerary, failures = _build_itinerary(proposal, skeleton, registry, request, "full", False)
    assert failures == []
    context = _build_context(registry, request)
    report = validate(itinerary, context)

    assert any(r.code == "state_mismatch" for r in report.results)


# --- 15. Skeleton ---


def test_build_skeleton_expands_route_times_days_per_stop():
    request = _request(route=["Austin, TX", "Dallas, TX"], days_per_stop=2, start_date=date(2026, 11, 3))

    skeleton = _build_skeleton(request)

    assert skeleton == [
        (date(2026, 11, 3), "Austin, TX"),
        (date(2026, 11, 4), "Austin, TX"),
        (date(2026, 11, 5), "Dallas, TX"),
        (date(2026, 11, 6), "Dallas, TX"),
    ]


def test_build_itinerary_bogus_day_number_fails():
    registry = PlaceRegistry()
    skeleton = [(date(2026, 11, 3), "Austin, TX")]
    proposal = ProposalIds(days=[{"day": 99, "stop_ids": []}])

    _, failures = _build_itinerary(proposal, skeleton, registry, _request(), "ids", False)

    assert any(f.code == "bad_day_number" for f in failures)


# --- 16. Context wiring ---


def test_registry_separates_tool_results_from_city_points():
    registry = PlaceRegistry()
    registry.record_places("Austin, TX", [_place()])
    registry.record_point("Austin, TX", GeoPoint(city="Austin", state="TX", lat=30.0, lon=-97.0))

    assert registry.tool_results == {("geoapify", "p1")}
    assert "Austin, TX" in registry.city_points
    assert ("geoapify", "Austin, TX") not in registry.tool_results


# --- Correction 1: catalog shows every searched interest, capped per interest ---


def test_catalog_caps_per_interest_but_shows_every_interest():
    registry = PlaceRegistry()
    museums = [_place(source_id=f"m{i}", name=f"M{i}") for i in range(10)]
    history = [_place(source_id=f"h{i}", name=f"H{i}", interest="history") for i in range(2)]
    registry.record_places("Austin, TX", museums)
    registry.record_places("Austin, TX", history)

    catalog = registry.catalog("Austin, TX")
    museum_ids = [c["source_id"] for c in catalog if c["interest"] == "museums"]
    history_ids = [c["source_id"] for c in catalog if c["interest"] == "history"]

    assert len(museum_ids) == agent_module.CATALOG_PER_INTEREST
    assert len(history_ids) == 2


# --- Correction 2: canonical registry keys, route-spelling city_points ---


def test_registry_keys_are_canonical_but_city_points_keep_route_spelling():
    registry = PlaceRegistry()
    registry.record_places("austin,   tx", [_place()])

    assert registry.catalog("Austin, TX") == registry.catalog("austin,   tx")
    assert registry.get("p1") is not None

    registry.record_point("Austin, TX", GeoPoint(city="Austin", state="TX", lat=30.0, lon=-97.0))

    assert "Austin, TX" in registry.city_points
    assert "austin, TX" not in registry.city_points
    assert "austin,   tx" not in registry.city_points


# --- Correction 3: Phase 0 geocodes every route city before any LLM call ---


def test_geocode_route_failure_raises_agent_error_before_any_llm_call(tmp_path):
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"results": []})

    client = FakeClient([])

    with httpx.Client(transport=httpx.MockTransport(handler)) as http_client, pytest.raises(AgentError):
        plan_trip(_request(), client=client, http_client=http_client, trace_dir=tmp_path)

    assert client.calls == []


def test_geocode_route_runs_before_the_tool_phase(tmp_path):
    handler = make_handler(places_json=TWO_MUSEUMS)
    client = FakeClient(
        [
            _no_tool_calls(),
            _proposal_message({"days": [{"day": 1, "stop_ids": []}]}),
        ]
    )

    with httpx.Client(transport=httpx.MockTransport(handler)) as http_client:
        result = plan_trip(
            _request(), client=client, http_client=http_client, trace_dir=tmp_path, max_repairs=0
        )

    assert "Austin, TX" in result.context.city_points


# --- Correction 4: _execute_tool catches bad arguments; null limit means default ---


def test_execute_tool_missing_required_argument_is_reported_not_raised():
    registry = PlaceRegistry()
    payload, error = _execute_tool("geocode", {}, registry, None)
    assert payload == {}
    assert error is not None


def test_execute_tool_unexpected_keyword_argument_is_reported_not_raised():
    registry = PlaceRegistry()
    payload, error = _execute_tool(
        "geocode", {"city_state": "Austin, TX", "bogus": "x"}, registry, None
    )
    assert payload == {}
    assert error is not None


def test_search_places_tool_message_is_compact_but_registry_keeps_all(monkeypatch):
    places = [_place(source_id=f"p{i}", name=f"Place {i}") for i in range(20)]

    def fake_search_places(city_state, interest, limit, client=None):
        return places

    monkeypatch.setitem(agent_module.TOOL_FUNCTIONS, "search_places", fake_search_places)
    registry = PlaceRegistry()

    payload, error = _execute_tool(
        "search_places",
        {"city_state": "Austin, TX", "interest": "museums", "limit": None},
        registry,
        None,
    )

    assert error is None
    content = json.dumps(payload)
    assert "source_id" not in content
    assert "lat" not in content
    assert len(content) < 600
    assert payload == {"found": 20, "interest": "museums", "sample": ["Place 0", "Place 1", "Place 2"]}
    assert len(registry.tool_results) == 20


def test_geocode_tool_message_has_only_lat_lon(monkeypatch):
    registry = PlaceRegistry()

    def fake_geocode(city_state, client=None):
        return GeoPoint(city="Austin", state="TX", lat=30.0, lon=-97.0)

    monkeypatch.setitem(agent_module.TOOL_FUNCTIONS, "geocode", fake_geocode)

    payload, error = _execute_tool("geocode", {"city_state": "Austin, TX"}, registry, None)

    assert error is None
    assert payload == {"lat": 30.0, "lon": -97.0}


def test_get_forecast_tool_message_has_date_high_and_rain_chance_per_day(monkeypatch):
    from datetime import date as _date

    from src.models import ForecastDay

    forecasts = [
        ForecastDay(date=_date(2026, 11, 3), rain_chance=0.2, high_f=75.0, source="open-meteo"),
        ForecastDay(date=_date(2026, 11, 4), rain_chance=0.5, high_f=70.0, source="open-meteo"),
    ]

    def fake_get_forecast(city_state, days, client=None):
        return forecasts

    monkeypatch.setitem(agent_module.TOOL_FUNCTIONS, "get_forecast", fake_get_forecast)
    registry = PlaceRegistry()

    payload, error = _execute_tool(
        "get_forecast", {"city_state": "Austin, TX", "days": 2}, registry, None
    )

    assert error is None
    assert payload == {
        "days": [
            {"date": "2026-11-03", "high_f": 75.0, "rain_chance": 0.2},
            {"date": "2026-11-04", "high_f": 70.0, "rain_chance": 0.5},
        ]
    }


def test_proposal_prompt_still_contains_the_full_catalog():
    registry = PlaceRegistry()
    places = [_place(source_id=f"p{i}", name=f"P{i}") for i in range(5)]
    registry.record_places("Austin, TX", places)
    skeleton = [(date(2026, 11, 3), "Austin, TX")]

    messages = agent_module._initial_proposal_messages(_request(), skeleton, registry, "ids")
    content = messages[1]["content"]

    assert "source_id" in content
    assert "lat" in content
    assert "p0" in content and "p4" in content


def test_plan_trip_records_usage_tokens_in_trace(tmp_path):
    handler = make_handler(places_json=TWO_MUSEUMS)
    client = FakeClient(
        [
            _tool_call_message(
                [("search_places", {"city_state": "Austin, TX", "interest": "museums"})]
            ),
            _no_tool_calls(),
            _completion(
                {"role": "assistant", "content": json.dumps({"days": [{"day": 1, "stop_ids": ["p1", "p2"]}]})},
                usage={"prompt_tokens": 111, "completion_tokens": 22, "total_tokens": 133},
            ),
        ]
    )

    with httpx.Client(transport=httpx.MockTransport(handler)) as http_client:
        result = plan_trip(_request(), client=client, http_client=http_client, trace_dir=tmp_path)

    data = json.loads(result.trace_path.read_text(encoding="utf-8"))
    llm_events = [e for e in data["events"] if e["kind"] == "llm_call"]
    assert llm_events[0]["prompt_tokens"] is None  # the first tool-phase call carried no usage
    propose_event = next(e for e in llm_events if e["phase"] == "propose")
    assert propose_event["prompt_tokens"] == 111
    assert propose_event["completion_tokens"] == 22


def test_execute_tool_null_limit_uses_search_limit_default(monkeypatch):
    captured = {}

    def fake_search_places(city_state, interest, limit, client=None):
        captured["limit"] = limit
        return []

    monkeypatch.setitem(agent_module.TOOL_FUNCTIONS, "search_places", fake_search_places)
    registry = PlaceRegistry()

    _, error = _execute_tool(
        "search_places",
        {"city_state": "Austin, TX", "interest": "museums", "limit": None},
        registry,
        None,
    )

    assert error is None
    assert captured["limit"] == agent_module.SEARCH_LIMIT


# --- Weather: forecasts are plain Python's job, not the model's ---

FIXED_TODAY = date(2026, 11, 4)


def _patch_today(monkeypatch, today: date = FIXED_TODAY) -> None:
    monkeypatch.setattr(tools_module, "_today", lambda: today)


def _forecast_json(start: date, n: int) -> dict:
    dates = [(start + timedelta(days=i)).isoformat() for i in range(n)]
    return {
        "daily": {
            "time": dates,
            "temperature_2m_max": [70.0] * n,
            "precipitation_probability_max": [10.0] * n,
        }
    }


def test_tool_schemas_exclude_get_forecast():
    names = {schema["function"]["name"] for schema in agent_module.TOOL_SCHEMAS}
    assert names == {"geocode", "search_places"}


def test_forecast_fetched_for_every_route_city_before_first_model_call(tmp_path, monkeypatch):
    _patch_today(monkeypatch)
    handler = make_handler(places_json=TWO_MUSEUMS)
    client = FakeClient(
        [
            _tool_call_message([("search_places", {"city_state": "Austin, TX", "interest": "museums"})]),
            _no_tool_calls(),
            _proposal_message({"days": [{"day": 1, "stop_ids": ["p1", "p2"]}]}),
        ]
    )

    real_get_forecast = tools_module.get_forecast

    def spy_get_forecast(*args, **kwargs):
        assert client.calls == []  # forecast fetched before any model call
        return real_get_forecast(*args, **kwargs)

    monkeypatch.setattr(tools_module, "get_forecast", spy_get_forecast)

    with httpx.Client(transport=httpx.MockTransport(handler)) as http_client:
        plan_trip(_request(start_date=FIXED_TODAY), client=client, http_client=http_client, trace_dir=tmp_path)


def test_plan_trip_forecast_present_for_every_day_within_window(tmp_path, monkeypatch):
    _patch_today(monkeypatch)
    handler = make_handler(forecast_json=_forecast_json(FIXED_TODAY, 3))
    client = FakeClient(
        [
            _no_tool_calls(),
            _proposal_message(
                {"days": [{"day": i, "stop_ids": []} for i in range(1, 4)]}
            ),
        ]
    )

    with httpx.Client(transport=httpx.MockTransport(handler)) as http_client:
        result = plan_trip(
            _request(days_per_stop=3, start_date=FIXED_TODAY),
            client=client,
            http_client=http_client,
            trace_dir=tmp_path,
            max_repairs=0,
        )

    assert len(result.itinerary.days) == 3
    for day in result.itinerary.days:
        assert day.forecast is not None
        assert day.warnings == []


def test_plan_trip_day_beyond_forecast_window_gets_window_warning_not_unverified(tmp_path, monkeypatch):
    _patch_today(monkeypatch)
    handler = make_handler(forecast_json=_forecast_json(FIXED_TODAY, FORECAST_MAX_DAYS))
    days_per_stop = FORECAST_MAX_DAYS + 4
    proposal_days = [{"day": i, "stop_ids": []} for i in range(1, days_per_stop + 1)]
    client = FakeClient([_no_tool_calls(), _proposal_message({"days": proposal_days})])

    with httpx.Client(transport=httpx.MockTransport(handler)) as http_client:
        result = plan_trip(
            _request(days_per_stop=days_per_stop, start_date=FIXED_TODAY),
            client=client,
            http_client=http_client,
            trace_dir=tmp_path,
            max_repairs=0,
        )

    last_in_window = result.itinerary.days[FORECAST_MAX_DAYS - 1]
    assert last_in_window.warnings == []
    assert last_in_window.forecast is not None

    first_beyond_window = result.itinerary.days[FORECAST_MAX_DAYS]
    assert first_beyond_window.forecast is None
    assert len(first_beyond_window.warnings) == 1
    assert f"outside the forecast window ({FORECAST_MAX_DAYS} days)" in first_beyond_window.warnings[0]
    assert "Austin, TX" in first_beyond_window.warnings[0]


def test_plan_trip_forecast_failure_adds_unavailable_warning_and_completes(tmp_path, monkeypatch):
    _patch_today(monkeypatch)

    def handler(request: httpx.Request) -> httpx.Response:
        url = str(request.url)
        if "geocode/search" in url:
            return _geocode_response(request)
        if "open-meteo" in url:
            return httpx.Response(500, text="boom")
        return httpx.Response(200, json={"features": []})

    client = FakeClient([_no_tool_calls(), _proposal_message({"days": [{"day": 1, "stop_ids": []}]})])

    with httpx.Client(transport=httpx.MockTransport(handler)) as http_client:
        result = plan_trip(
            _request(start_date=FIXED_TODAY),
            client=client,
            http_client=http_client,
            trace_dir=tmp_path,
            max_repairs=0,
        )

    assert any("weather unavailable for Austin, TX" in w for w in result.itinerary.warnings)
    day = result.itinerary.days[0]
    assert day.forecast is None
    assert len(day.warnings) == 1
    assert "weather unavailable for Austin, TX" in day.warnings[0]


def test_plan_trip_past_trip_date_gets_past_date_warning(tmp_path, monkeypatch):
    _patch_today(monkeypatch, today=date(2026, 11, 10))
    handler = make_handler(forecast_json=_forecast_json(date(2026, 11, 10), FORECAST_MAX_DAYS))
    client = FakeClient([_no_tool_calls(), _proposal_message({"days": [{"day": 1, "stop_ids": []}]})])

    with httpx.Client(transport=httpx.MockTransport(handler)) as http_client:
        result = plan_trip(
            _request(start_date=date(2026, 11, 5)),
            client=client,
            http_client=http_client,
            trace_dir=tmp_path,
            max_repairs=0,
        )

    day = result.itinerary.days[0]
    assert day.warnings == ["date is in the past for Austin, TX"]
    assert day.forecast is None


# --- Regression: Open-Meteo counts days from the destination's local date, which can
# be a day ahead of (or behind) the machine's date. Coverage must be judged from the
# dates the API actually returned, never from arithmetic against the machine date. ---

REGRESSION_TODAY = date(2026, 11, 4)
REGRESSION_FORECAST_START = date(2026, 11, 3)  # a day BEFORE the patched "today"
REGRESSION_FORECAST_DAYS = 16  # == FORECAST_MAX_DAYS


def _forecast_json_for_dates(dates: list[date]) -> dict:
    n = len(dates)
    return {
        "daily": {
            "time": [d.isoformat() for d in dates],
            "temperature_2m_max": [70.0] * n,
            "precipitation_probability_max": [10.0] * n,
        }
    }


def _regression_handler():
    return make_handler(forecast_json=_forecast_json(REGRESSION_FORECAST_START, REGRESSION_FORECAST_DAYS))


def test_plan_trip_covers_trip_day_when_destination_local_date_is_ahead_of_machine_today(
    tmp_path, monkeypatch
):
    """The exact reported bug: machine date 2026-11-04, Open-Meteo's local-date window
    starts 2026-11-03. A trip day that the old today-based arithmetic would have pushed
    outside the window must still get its forecast, because the API actually returned it.
    """
    _patch_today(monkeypatch, today=REGRESSION_TODAY)
    handler = _regression_handler()
    client = FakeClient(
        [_no_tool_calls(), _proposal_message({"days": [{"day": 1, "stop_ids": []}, {"day": 2, "stop_ids": []}]})]
    )

    with httpx.Client(transport=httpx.MockTransport(handler)) as http_client:
        result = plan_trip(
            _request(days_per_stop=2, start_date=date(2026, 11, 5)),
            client=client,
            http_client=http_client,
            trace_dir=tmp_path,
            max_repairs=0,
        )

    assert [d.date for d in result.itinerary.days] == [date(2026, 11, 5), date(2026, 11, 6)]
    for day in result.itinerary.days:
        assert day.forecast is not None
        assert day.warnings == []


def test_plan_trip_date_after_last_returned_forecast_gets_window_warning_with_actual_count(
    tmp_path, monkeypatch
):
    _patch_today(monkeypatch, today=REGRESSION_TODAY)
    handler = _regression_handler()
    client = FakeClient([_no_tool_calls(), _proposal_message({"days": [{"day": 1, "stop_ids": []}]})])

    with httpx.Client(transport=httpx.MockTransport(handler)) as http_client:
        result = plan_trip(
            _request(start_date=date(2026, 11, 19)),  # one day past the last returned date (11-18)
            client=client,
            http_client=http_client,
            trace_dir=tmp_path,
            max_repairs=0,
        )

    day = result.itinerary.days[0]
    assert day.forecast is None
    assert day.warnings == [
        f"outside the forecast window ({REGRESSION_FORECAST_DAYS} days) for Austin, TX on 2026-11-19"
    ]


def test_plan_trip_date_before_first_returned_forecast_gets_past_date_warning(tmp_path, monkeypatch):
    _patch_today(monkeypatch, today=REGRESSION_TODAY)
    handler = _regression_handler()
    client = FakeClient([_no_tool_calls(), _proposal_message({"days": [{"day": 1, "stop_ids": []}]})])

    with httpx.Client(transport=httpx.MockTransport(handler)) as http_client:
        result = plan_trip(
            _request(start_date=date(2026, 11, 2)),  # one day before the first returned date (11-03)
            client=client,
            http_client=http_client,
            trace_dir=tmp_path,
            max_repairs=0,
        )

    day = result.itinerary.days[0]
    assert day.forecast is None
    assert day.warnings == ["date is in the past for Austin, TX"]


def test_plan_trip_last_day_null_high_still_covers_complete_days(tmp_path, monkeypatch):
    """Regression: Open-Meteo's last day in a 16-day response can have a null high (not
    yet forecast). get_forecast must drop that one day, not discard the whole response,
    so the window shrinks to the days actually complete rather than vanishing entirely.
    """
    today = date(2026, 11, 3)
    _patch_today(monkeypatch, today=today)
    dates = [(today + timedelta(days=i)).isoformat() for i in range(REGRESSION_FORECAST_DAYS)]
    highs = [70.0] * REGRESSION_FORECAST_DAYS
    highs[-1] = None
    forecast_json = {
        "daily": {
            "time": dates,
            "temperature_2m_max": highs,
            "precipitation_probability_max": [10.0] * REGRESSION_FORECAST_DAYS,
        }
    }
    handler = make_handler(forecast_json=forecast_json)
    proposal_days = [{"day": i, "stop_ids": []} for i in range(1, REGRESSION_FORECAST_DAYS + 1)]
    client = FakeClient([_no_tool_calls(), _proposal_message({"days": proposal_days})])

    with httpx.Client(transport=httpx.MockTransport(handler)) as http_client:
        result = plan_trip(
            _request(days_per_stop=REGRESSION_FORECAST_DAYS, start_date=today),
            client=client,
            http_client=http_client,
            trace_dir=tmp_path,
            max_repairs=0,
        )

    complete_days = REGRESSION_FORECAST_DAYS - 1  # last day dropped for the null high
    for day in result.itinerary.days[:complete_days]:
        assert day.forecast is not None
        assert day.warnings == []

    last_day = result.itinerary.days[complete_days]
    assert last_day.forecast is None
    expected_warning = (
        f"outside the forecast window ({complete_days} days) "
        f"for Austin, TX on {last_day.date.isoformat()}"
    )
    assert last_day.warnings == [expected_warning]


# --- reasoning_effort: threaded through every llm.chat call, recorded in the trace ---


def test_plan_trip_reasoning_effort_passed_to_every_llm_call(tmp_path):
    handler = make_handler(places_json=TWO_MUSEUMS)
    client = FakeClient(
        [
            _tool_call_message([("search_places", {"city_state": "Austin, TX", "interest": "museums"})]),
            _no_tool_calls(),
            _proposal_message({"days": [{"day": 1, "stop_ids": ["p1"]}]}),  # too few stops
            _proposal_message({"days": [{"day": 1, "stop_ids": ["p1", "p2"]}]}),  # fixed
        ]
    )

    with httpx.Client(transport=httpx.MockTransport(handler)) as http_client:
        result = plan_trip(
            _request(),
            client=client,
            http_client=http_client,
            trace_dir=tmp_path,
            reasoning_effort="low",
        )

    assert result.repairs == 1
    for call in client.calls:
        assert call["reasoning_effort"] == "low"


def test_plan_trip_reasoning_effort_none_omits_kwarg_from_every_call(tmp_path):
    handler = make_handler(places_json=TWO_MUSEUMS)
    client = FakeClient(
        [
            _tool_call_message([("search_places", {"city_state": "Austin, TX", "interest": "museums"})]),
            _no_tool_calls(),
            _proposal_message({"days": [{"day": 1, "stop_ids": ["p1", "p2"]}]}),
        ]
    )

    with httpx.Client(transport=httpx.MockTransport(handler)) as http_client:
        plan_trip(_request(), client=client, http_client=http_client, trace_dir=tmp_path)

    for call in client.calls:
        assert "reasoning_effort" not in call


def test_plan_trip_invalid_reasoning_effort_raises_llm_error(tmp_path):
    handler = make_handler(places_json=TWO_MUSEUMS)
    client = FakeClient([])

    with httpx.Client(transport=httpx.MockTransport(handler)) as http_client, pytest.raises(LLMError):
        plan_trip(
            _request(),
            client=client,
            http_client=http_client,
            trace_dir=tmp_path,
            reasoning_effort="extreme",
        )


def test_plan_trip_records_reasoning_effort_in_trace(tmp_path):
    handler = make_handler(places_json=TWO_MUSEUMS)
    client = FakeClient(
        [
            _tool_call_message([("search_places", {"city_state": "Austin, TX", "interest": "museums"})]),
            _no_tool_calls(),
            _proposal_message({"days": [{"day": 1, "stop_ids": ["p1", "p2"]}]}),
        ]
    )

    with httpx.Client(transport=httpx.MockTransport(handler)) as http_client:
        result = plan_trip(
            _request(),
            client=client,
            http_client=http_client,
            trace_dir=tmp_path,
            reasoning_effort="medium",
        )

    data = json.loads(result.trace_path.read_text(encoding="utf-8"))
    assert data["request"]["reasoning_effort"] == "medium"


def test_plan_trip_gap_inside_forecast_range_gets_weather_unverified(tmp_path, monkeypatch):
    _patch_today(monkeypatch, today=REGRESSION_TODAY)
    handler = make_handler(
        forecast_json=_forecast_json_for_dates([date(2026, 11, 3), date(2026, 11, 5)])  # 11-04 missing
    )
    client = FakeClient([_no_tool_calls(), _proposal_message({"days": [{"day": 1, "stop_ids": []}]})])

    with httpx.Client(transport=httpx.MockTransport(handler)) as http_client:
        result = plan_trip(
            _request(start_date=date(2026, 11, 4)),
            client=client,
            http_client=http_client,
            trace_dir=tmp_path,
            max_repairs=0,
        )

    day = result.itinerary.days[0]
    assert day.forecast is None
    assert day.warnings == ["weather unverified"]
