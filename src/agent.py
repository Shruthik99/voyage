"""The tool-calling loop: plan, validate, repair.

Two phases per docs/SPEC.md ("Agent behavior"): Groq rejects `tools` and
`response_format` in the same request, and gpt-oss-120b does not call tools in
parallel. Phase 1 calls tools (geocode, search_places) with no schema; phase 2
asks for a strict json_schema proposal with no tools. Repairs reuse phase 2.

Plain Python, not the LLM, expands route x days_per_stop into dated days
(`_build_skeleton`), geocodes every route city (`_geocode_route`), and fetches
weather for every route city (`_forecast_route`) before phase 1 ever runs --
weather coverage can't depend on the model remembering to call a tool. The LLM
names stops by source_id; `PlaceRegistry` resolves each id to the `Place`
recorded from a real tool result, so a stop's coordinates are never retyped by
a model. An id that was never returned is reported, never invented. Plain
Python also prices every stop (`src.pricing`).
"""

import json
import time
import uuid
from datetime import date, timedelta
from pathlib import Path
from typing import Literal

import httpx
from openai import OpenAI
from pydantic import BaseModel, ConfigDict, Field, model_validator
from pydantic import ValidationError as PydanticValidationError

from src import llm, pricing, tools
from src.constants import FORECAST_MAX_DAYS, MAX_DAY_KM, MAX_STOPS, MIN_STOPS, RADIUS_KM
from src.models import (
    Day,
    ForecastDay,
    GeoPoint,
    Itinerary,
    Place,
    RuleResult,
    Stop,
    ValidationContext,
    ValidationReport,
)
from src.parsing import LocationError, parse_location
from src.tools import INTEREST_CATEGORIES, ToolError
from src.trace import TRACE_DIR, Trace
from src.validator import validate

DEFAULT_MODEL = llm.DEFAULT_MODEL

MAX_TOOL_STEPS = 12  # total tool calls per run (gpt-oss-120b cannot batch them)
MAX_REPAIRS = 3
MAX_PARSE_RETRIES = 2  # malformed JSON, separate budget from repairs
MAX_REPAIR_ITEMS = 20

SEARCH_LIMIT = 20  # requested from the places API per search_places call
CATALOG_PER_INTEREST = 6  # shown to the model, per interest per city


class AgentError(RuntimeError):
    """Raised when the agent cannot produce a result at all."""


class TripRequest(BaseModel):
    route: list[str] = Field(min_length=1)
    days_per_stop: int = Field(ge=1)
    travelers: int = Field(ge=1)
    budget_usd: float = Field(ge=0)
    meal_allowance_usd: float = Field(ge=0)
    interests: list[str] = Field(default_factory=list)
    start_date: date

    @model_validator(mode="after")
    def _check_route_and_interests(self) -> "TripRequest":
        for city_state in self.route:
            parse_location(city_state)
        unknown = [i for i in self.interests if i not in INTEREST_CATEGORIES]
        if unknown:
            raise ValueError(f"unknown interest(s): {unknown!r}")
        return self


class AgentResult(BaseModel):
    itinerary: Itinerary
    report: ValidationReport
    context: ValidationContext
    repairs: int
    tool_calls: int
    trace_path: Path | None


# --- PlaceRegistry: the traceability spine ---


def _canon_city_key(city_state: str) -> str:
    """A case/whitespace-insensitive city key: parse_location, casefolded city + state."""
    city, state = parse_location(city_state)
    return f"{city.casefold()}, {state}"


class PlaceRegistry:
    """Single source of truth for both the resolver and the validator.

    Keyed on source_id alone for lookups; the authoritative source comes from
    the stored Place, so the LLM never retypes it. Places and forecasts are
    bucketed by a canonical city key so a tool call's own spelling of a city
    always finds what another tool call recorded for it. city_points is the
    exception: it stays keyed by the exact string passed in (the route's own
    spelling), because that is what ValidationContext.city_points is looked up
    by, via Day.city.
    """

    def __init__(self) -> None:
        self._by_id: dict[str, Place] = {}
        self._places_by_city: dict[str, dict[str, Place]] = {}
        self._city_points: dict[str, GeoPoint] = {}
        self._tool_results: set[tuple[str, str]] = set()
        self._forecasts: dict[str, list[ForecastDay]] = {}

    def record_places(self, city_state: str, places: list[Place]) -> None:
        key = _canon_city_key(city_state)
        bucket = self._places_by_city.setdefault(key, {})
        for place in places:
            existing = self._by_id.get(place.source_id)
            if existing is not None and existing.source != place.source:
                raise AgentError(
                    f"source_id collision: {place.source_id!r} already recorded from "
                    f"{existing.source!r}, now from {place.source!r}"
                )
            self._by_id[place.source_id] = place
            bucket[place.source_id] = place
            self._tool_results.add((place.source, place.source_id))

    def record_point(self, city_state: str, point: GeoPoint) -> None:
        self._city_points[city_state] = point

    def record_forecast(self, city_state: str, forecasts: list[ForecastDay]) -> None:
        self._forecasts[_canon_city_key(city_state)] = forecasts

    def get(self, source_id: str) -> Place | None:
        return self._by_id.get(source_id)

    def forecast_for(self, city_state: str, target_date: date) -> ForecastDay | None:
        for forecast in self._forecasts.get(_canon_city_key(city_state), []):
            if forecast.date == target_date:
                return forecast
        return None

    def forecast_range(self, city_state: str) -> tuple[date, date] | None:
        """Earliest and latest dates actually recorded for this city, or None if none."""
        forecasts = self._forecasts.get(_canon_city_key(city_state))
        if not forecasts:
            return None
        dates = [f.date for f in forecasts]
        return min(dates), max(dates)

    @property
    def tool_results(self) -> set[tuple[str, str]]:
        return set(self._tool_results)

    @property
    def city_points(self) -> dict[str, GeoPoint]:
        return dict(self._city_points)

    def catalog(self, city_state: str) -> list[dict]:
        """Up to CATALOG_PER_INTEREST places per interest searched for this city."""
        bucket = self._places_by_city.get(_canon_city_key(city_state), {})
        by_interest: dict[str, list[Place]] = {}
        for place in bucket.values():
            by_interest.setdefault(place.interest, []).append(place)
        return [
            _place_summary(place)
            for places in by_interest.values()
            for place in places[:CATALOG_PER_INTEREST]
        ]


def _place_summary(place: Place) -> dict:
    return {
        "source_id": place.source_id,
        "name": place.name,
        "interest": place.interest,
        "state": place.state,
        "lat": place.lat,
        "lon": place.lon,
    }


# --- Phase 1: tool schemas and dispatch ---

TOOL_SCHEMAS = [
    {
        "type": "function",
        "function": {
            "name": "geocode",
            "description": "Geocode a US city to its coordinates.",
            "parameters": {
                "type": "object",
                "properties": {
                    "city_state": {
                        "type": "string",
                        "description": "A 'City, ST' string, e.g. 'Austin, TX'",
                    },
                },
                "required": ["city_state"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "search_places",
            "description": "Search real places of one interest category near a US city.",
            "parameters": {
                "type": "object",
                "properties": {
                    "city_state": {"type": "string"},
                    "interest": {
                        "type": "string",
                        "enum": sorted(INTEREST_CATEGORIES),
                    },
                    "limit": {
                        "type": ["integer", "null"],
                        "description": f"max results; null means the default ({SEARCH_LIMIT})",
                    },
                },
                "required": ["city_state", "interest"],
            },
        },
    },
]

TOOL_FUNCTIONS = {
    "geocode": tools.geocode,
    "search_places": tools.search_places,
    "get_forecast": tools.get_forecast,
}


def _parse_arguments(raw: str) -> tuple[dict, str | None]:
    try:
        args = json.loads(raw) if raw else {}
    except json.JSONDecodeError as exc:
        return {}, f"invalid JSON arguments: {exc}"
    if not isinstance(args, dict):
        return {}, "tool arguments must be a JSON object"
    return args, None


def _execute_tool(
    name: str,
    args: dict,
    registry: PlaceRegistry,
    http_client: httpx.Client | None,
) -> tuple[dict, str | None]:
    """Run one tool call. Never raises: bad args or a real failure become (payload, error)."""
    fn = TOOL_FUNCTIONS.get(name)
    if fn is None:
        return {}, f"unknown tool: {name!r}"

    call_args = dict(args)
    if name == "search_places" and call_args.get("limit") is None:
        call_args["limit"] = SEARCH_LIMIT
    call_args["client"] = http_client

    try:
        result = fn(**call_args)
    except (ToolError, LocationError) as exc:
        return {}, str(exc)
    except (TypeError, ValueError) as exc:
        return {}, str(exc)

    if name == "geocode":
        registry.record_point(call_args["city_state"], result)
        payload = {"lat": result.lat, "lon": result.lon}
    elif name == "search_places":
        # Full place records (with source_id, lat/lon) go into the catalog the
        # proposal phase reads from the registry; the tool phase only needs to
        # know enough to decide whether to search again.
        registry.record_places(call_args["city_state"], result)
        payload = {
            "found": len(result),
            "interest": call_args.get("interest"),
            "sample": [p.name for p in result[:3]],
        }
    else:  # get_forecast
        registry.record_forecast(call_args["city_state"], result)
        payload = {
            "days": [
                {"date": f.date.isoformat(), "high_f": f.high_f, "rain_chance": f.rain_chance}
                for f in result
            ]
        }

    return payload, None


# --- Phase 2: proposal schemas and models ---

PROPOSAL_SCHEMA_IDS = {
    "type": "object",
    "properties": {
        "days": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "day": {"type": "integer"},
                    "stop_ids": {"type": "array", "items": {"type": "string"}},
                },
                "required": ["day", "stop_ids"],
                "additionalProperties": False,
            },
        },
    },
    "required": ["days"],
    "additionalProperties": False,
}

PROPOSAL_SCHEMA_FULL = {
    "type": "object",
    "properties": {
        "days": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "day": {"type": "integer"},
                    "stops": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "properties": {
                                "name": {"type": "string"},
                                "address": {"type": "string"},
                                "state": {"type": "string"},
                                "lat": {"type": "number"},
                                "lon": {"type": "number"},
                                "interest": {"type": "string"},
                                "source": {"type": "string"},
                                "source_id": {"type": "string"},
                            },
                            "required": [
                                "name",
                                "address",
                                "state",
                                "lat",
                                "lon",
                                "interest",
                                "source",
                                "source_id",
                            ],
                            "additionalProperties": False,
                        },
                    },
                },
                "required": ["day", "stops"],
                "additionalProperties": False,
            },
        },
    },
    "required": ["days"],
    "additionalProperties": False,
}


def _proposal_schema(stop_mode: str) -> dict:
    return PROPOSAL_SCHEMA_IDS if stop_mode == "ids" else PROPOSAL_SCHEMA_FULL


class ProposalDayIds(BaseModel):
    model_config = ConfigDict(extra="forbid")
    day: int
    stop_ids: list[str]


class ProposalIds(BaseModel):
    model_config = ConfigDict(extra="forbid")
    days: list[ProposalDayIds]


class ProposalStopFull(BaseModel):
    model_config = ConfigDict(extra="forbid")
    name: str
    address: str
    state: str
    lat: float
    lon: float
    interest: str
    source: str
    source_id: str


class ProposalDayFull(BaseModel):
    model_config = ConfigDict(extra="forbid")
    day: int
    stops: list[ProposalStopFull]


class ProposalFull(BaseModel):
    model_config = ConfigDict(extra="forbid")
    days: list[ProposalDayFull]


def _proposal_model(stop_mode: str) -> type[BaseModel]:
    return ProposalIds if stop_mode == "ids" else ProposalFull


# --- Skeleton (plain Python, no LLM) ---


def _build_skeleton(request: TripRequest) -> list[tuple[date, str]]:
    """route x days_per_stop, consecutive dates from start_date."""
    skeleton: list[tuple[date, str]] = []
    current = request.start_date
    for city in request.route:
        for _ in range(request.days_per_stop):
            skeleton.append((current, city))
            current += timedelta(days=1)
    return skeleton


# --- Message builders ---


def _tool_phase_messages(request: TripRequest) -> list[dict]:
    interests = ", ".join(request.interests) or "none"
    system = (
        "You are planning a US road trip. Call geocode and search_places to gather "
        "real places for each city on the route. Search every requested interest in "
        "every city before you stop. Never invent a place, a coordinate, or a price "
        "-- only use tool results. "
        f"Route: {', '.join(request.route)}. Requested interests: {interests}. "
        "When you have gathered enough information, stop calling tools."
    )
    return [
        {"role": "system", "content": system},
        {"role": "user", "content": "Gather places and forecasts for this trip, then stop."},
    ]


def _usage_tokens(response) -> tuple[int | None, int | None]:
    usage = getattr(response, "usage", None)
    if usage is None:
        return None, None
    return usage.prompt_tokens, usage.completion_tokens


def _skeleton_payload(skeleton: list[tuple[date, str]]) -> list[dict]:
    return [{"day": i + 1, "date": d.isoformat(), "city": c} for i, (d, c) in enumerate(skeleton)]


def _catalog_payload(registry: PlaceRegistry, request: TripRequest) -> dict:
    return {city: registry.catalog(city) for city in dict.fromkeys(request.route)}


def _proposal_system_prompt(request: TripRequest, *, repair: bool) -> str:
    interests = ", ".join(request.interests) or "none"
    base = (
        "Build the itinerary for the trip below. Pick stops only by source_id from the "
        "catalog provided; never invent a place, a coordinate, or a source_id. Never "
        "output a price -- cost fields are not part of your schema. "
        f"Radius limit: {RADIUS_KM} km from each city's center. "
        f"Max straight-line distance per day: {MAX_DAY_KM} km. "
        f"Stops per day: between {MIN_STOPS} and {MAX_STOPS}. "
        f"Requested interests to cover somewhere in the trip: {interests}."
    )
    if repair:
        base += (
            " This is a repair. Change only the days and stops named in the failure list "
            "below. Every other day and every other stop must come back byte-identical."
        )
    return base


def _initial_proposal_messages(
    request: TripRequest,
    skeleton: list[tuple[date, str]],
    registry: PlaceRegistry,
    stop_mode: str,
) -> list[dict]:
    payload = {
        "days": _skeleton_payload(skeleton),
        "catalog": _catalog_payload(registry, request),
        "stop_mode": stop_mode,
    }
    return [
        {"role": "system", "content": _proposal_system_prompt(request, repair=False)},
        {"role": "user", "content": json.dumps(payload)},
    ]


def _repair_messages(
    request: TripRequest,
    registry: PlaceRegistry,
    prior_proposal: dict,
    failures: list[dict],
) -> list[dict]:
    return [
        {"role": "system", "content": _proposal_system_prompt(request, repair=True)},
        {"role": "user", "content": json.dumps({"catalog": _catalog_payload(registry, request)})},
        {"role": "assistant", "content": json.dumps(prior_proposal)},
        {"role": "user", "content": json.dumps({"failures": failures})},
    ]


# --- Phase 1: tool loop ---


def _run_tool_phase(
    client: OpenAI,
    model: str,
    request: TripRequest,
    registry: PlaceRegistry,
    trace: Trace,
    http_client: httpx.Client | None,
    max_tool_steps: int,
) -> int:
    messages = _tool_phase_messages(request)
    steps = 0

    while steps < max_tool_steps:
        start = time.monotonic()
        response = llm.chat(client, messages, model=model, tools=TOOL_SCHEMAS, tool_choice="auto")
        duration_ms = (time.monotonic() - start) * 1000
        message = response.choices[0].message
        reasoning = getattr(message, "reasoning", None) or ""
        prompt_tokens, completion_tokens = _usage_tokens(response)
        trace.llm_call(
            "tools",
            model=model,
            has_tools=True,
            has_schema=False,
            reasoning_chars=len(reasoning),
            duration_ms=duration_ms,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
        )

        tool_calls = getattr(message, "tool_calls", None) or []
        if not tool_calls:
            break

        messages.append(
            {
                "role": "assistant",
                "content": message.content or "",
                "tool_calls": [
                    {
                        "id": tc.id,
                        "type": "function",
                        "function": {"name": tc.function.name, "arguments": tc.function.arguments},
                    }
                    for tc in tool_calls
                ],
            }
        )

        for tc in tool_calls:
            if steps >= max_tool_steps:
                break
            args, parse_error = _parse_arguments(tc.function.arguments)
            tool_start = time.monotonic()
            if parse_error:
                payload, error = {}, parse_error
            else:
                payload, error = _execute_tool(tc.function.name, args, registry, http_client)
            tool_duration_ms = (time.monotonic() - tool_start) * 1000
            trace.tool_call(
                tc.function.name,
                args,
                summary=None if error else payload,
                error=error,
                duration_ms=tool_duration_ms,
            )
            content = json.dumps({"error": error} if error else payload)
            messages.append(
                {
                    "role": "tool",
                    "tool_call_id": tc.id,
                    "name": tc.function.name,
                    "content": content,
                }
            )
            steps += 1

    if steps >= max_tool_steps:
        trace.note(f"tool step cap ({max_tool_steps}) reached; proposing with what was gathered")

    return steps


# --- Phase 0: geocode every route city before any LLM call ---


def _geocode_route(
    request: TripRequest,
    registry: PlaceRegistry,
    http_client: httpx.Client | None,
    trace: Trace,
) -> None:
    for city in dict.fromkeys(request.route):
        start = time.monotonic()
        try:
            point = tools.geocode(city, client=http_client)
        except (ToolError, LocationError) as exc:
            duration_ms = (time.monotonic() - start) * 1000
            trace.tool_call("geocode", {"city_state": city}, error=str(exc), duration_ms=duration_ms)
            raise AgentError(str(exc)) from exc
        duration_ms = (time.monotonic() - start) * 1000
        trace.tool_call(
            "geocode",
            {"city_state": city},
            summary={"lat": point.lat, "lon": point.lon},
            duration_ms=duration_ms,
        )
        registry.record_point(city, point)


def _forecast_route(
    request: TripRequest,
    registry: PlaceRegistry,
    http_client: httpx.Client | None,
    trace: Trace,
) -> tuple[dict[str, str], list[str]]:
    """Fetch weather for every distinct route city. A failure is a warning, not a raise.

    Open-Meteo counts forecast days from the destination's local date, which can
    differ from the machine's date by a day. So every city gets the max window,
    and coverage is judged later from the dates actually returned
    (PlaceRegistry.forecast_range), never from the machine date.
    """
    failures: dict[str, str] = {}
    run_warnings: list[str] = []

    for city in dict.fromkeys(request.route):
        start = time.monotonic()
        try:
            forecasts = tools.get_forecast(city, FORECAST_MAX_DAYS, client=http_client)
        except ToolError as exc:
            duration_ms = (time.monotonic() - start) * 1000
            trace.tool_call(
                "get_forecast",
                {"city_state": city, "days": FORECAST_MAX_DAYS},
                error=str(exc),
                duration_ms=duration_ms,
            )
            failures[_canon_city_key(city)] = str(exc)
            run_warnings.append(f"weather unavailable for {city}: {exc}")
            continue
        duration_ms = (time.monotonic() - start) * 1000
        trace.tool_call(
            "get_forecast",
            {"city_state": city, "days": FORECAST_MAX_DAYS},
            summary={"count": len(forecasts)},
            duration_ms=duration_ms,
        )
        registry.record_forecast(city, forecasts)

    return failures, run_warnings


def _weather_warning(
    city: str,
    day_date: date,
    forecast: ForecastDay | None,
    forecast_range: tuple[date, date] | None,
    forecast_failures: dict[str, str],
) -> str | None:
    if forecast is not None:
        return None
    reason = forecast_failures.get(_canon_city_key(city))
    if reason is not None:
        return f"weather unavailable for {city}: {reason}"
    if forecast_range is None:
        return "weather unverified"
    earliest, latest = forecast_range
    if day_date < earliest:
        return f"date is in the past for {city}"
    if day_date > latest:
        window_days = (latest - earliest).days + 1
        return f"outside the forecast window ({window_days} days) for {city} on {day_date.isoformat()}"
    return "weather unverified"


# --- Phase 2: proposal (and repair, which reuses it) ---


def _run_proposal_phase(
    client: OpenAI,
    model: str,
    messages: list[dict],
    trace: Trace,
    phase: str,
    stop_mode: str,
) -> BaseModel:
    schema = _proposal_schema(stop_mode)
    proposal_model = _proposal_model(stop_mode)
    retries = 0

    while True:
        start = time.monotonic()
        response = llm.chat(
            client,
            messages,
            model=model,
            response_format={
                "type": "json_schema",
                "json_schema": {"name": "itinerary_proposal", "strict": True, "schema": schema},
            },
        )
        duration_ms = (time.monotonic() - start) * 1000
        message = response.choices[0].message
        reasoning = getattr(message, "reasoning", None) or ""
        prompt_tokens, completion_tokens = _usage_tokens(response)
        trace.llm_call(
            phase,
            model=model,
            has_tools=False,
            has_schema=True,
            reasoning_chars=len(reasoning),
            duration_ms=duration_ms,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
        )

        content = message.content
        try:
            if not content or not content.strip():
                raise ValueError("empty response content")
            data = json.loads(content)
            return proposal_model.model_validate(data)
        except (json.JSONDecodeError, ValueError, PydanticValidationError) as exc:
            retries += 1
            if retries > MAX_PARSE_RETRIES:
                raise AgentError(
                    f"the model did not produce a valid proposal after "
                    f"{MAX_PARSE_RETRIES} retries: {exc}"
                ) from exc
            messages.append({"role": "assistant", "content": content or ""})
            messages.append(
                {
                    "role": "user",
                    "content": f"your reply was not valid JSON for the schema: {exc}",
                }
            )


# --- Phase 3: build, price, validate ---


def _place_to_stop(place: Place, request: TripRequest) -> Stop:
    cost = pricing.price_place(place, request)
    return Stop(**place.model_dump(), **cost.model_dump())


def _cost_summary(days: list[Day], travelers: int) -> tuple[float, int]:
    """Mirrors src.validator's own math, so the itinerary's stated summary agrees with it."""
    high_cents = estimate_high_cents = 0
    unknown_count = 0
    for day in days:
        for stop in day.stops:
            if stop.cost_confidence == "unknown":
                unknown_count += 1
                continue
            multiplier = travelers if stop.cost_basis == "per_person" else 1
            high = round(stop.cost_high_usd * 100) * multiplier
            high_cents += high
            if stop.cost_confidence == "estimate":
                estimate_high_cents += high
    share = estimate_high_cents / high_cents if high_cents > 0 else 0.0
    return share, unknown_count


def _build_itinerary(
    proposal: BaseModel,
    skeleton: list[tuple[date, str]],
    registry: PlaceRegistry,
    request: TripRequest,
    stop_mode: str,
    cap_hit: bool,
    *,
    forecast_failures: dict[str, str] | None = None,
) -> tuple[Itinerary, list[RuleResult]]:
    forecast_failures = forecast_failures or {}
    failures: list[RuleResult] = []
    valid_day_numbers = set(range(1, len(skeleton) + 1))
    by_day_number: dict[int, BaseModel] = {}

    for pd in proposal.days:
        if pd.day not in valid_day_numbers:
            failures.append(
                RuleResult(
                    rule="V8",
                    code="bad_day_number",
                    status="fail",
                    message=f"day number {pd.day} does not correspond to a day in the trip",
                )
            )
            continue
        if pd.day in by_day_number:
            failures.append(
                RuleResult(
                    rule="V8",
                    code="bad_day_number",
                    status="fail",
                    message=f"day number {pd.day} is repeated in the proposal",
                )
            )
            continue
        by_day_number[pd.day] = pd

    days: list[Day] = []
    for idx, (day_date, city) in enumerate(skeleton):
        day_number = idx + 1
        pd = by_day_number.get(day_number)
        stops: list[Stop] = []

        if pd is None:
            failures.append(
                RuleResult(
                    rule="V8",
                    code="bad_day_number",
                    status="fail",
                    day=idx,
                    message=f"no proposed stops for day {day_number}",
                )
            )
        elif stop_mode == "ids":
            for sid in pd.stop_ids:
                place = registry.get(sid)
                if place is None:
                    failures.append(
                        RuleResult(
                            rule="V5",
                            code="untraceable",
                            status="fail",
                            day=idx,
                            source_id=sid,
                            message=(
                                f"day {day_number}: source_id {sid!r} was never "
                                "returned by a tool"
                            ),
                        )
                    )
                    continue
                stops.append(_place_to_stop(place, request))
        else:  # full
            for s in pd.stops:
                try:
                    place = Place(
                        name=s.name,
                        address=s.address,
                        state=s.state,
                        lat=s.lat,
                        lon=s.lon,
                        interest=s.interest,
                        source=s.source,
                        source_id=s.source_id,
                    )
                except PydanticValidationError as exc:
                    failures.append(
                        RuleResult(
                            rule="V4",
                            code="malformed_stop",
                            status="fail",
                            day=idx,
                            message=f"day {day_number}: {exc}",
                        )
                    )
                    continue
                stops.append(_place_to_stop(place, request))

        forecast = registry.forecast_for(city, day_date)
        forecast_range = registry.forecast_range(city)
        warning = _weather_warning(city, day_date, forecast, forecast_range, forecast_failures)
        warnings = [warning] if warning else []
        days.append(Day(date=day_date, city=city, stops=stops, forecast=forecast, warnings=warnings))

    share, unknown_count = _cost_summary(days, request.travelers)
    itinerary_warnings = []
    if cap_hit:
        itinerary_warnings.append("tool step cap reached before the model finished gathering places")

    itinerary = Itinerary(
        days=days,
        warnings=itinerary_warnings,
        estimated_share=share,
        unknown_count=unknown_count,
    )
    return itinerary, failures


def _build_context(registry: PlaceRegistry, request: TripRequest) -> ValidationContext:
    return ValidationContext(
        city_points=registry.city_points,
        tool_results=registry.tool_results,
        travelers=request.travelers,
        budget_usd=request.budget_usd,
        interests=request.interests,
    )


def _merge_build_failures(
    report: ValidationReport, build_failures: list[RuleResult]
) -> ValidationReport:
    if not build_failures:
        return report
    return ValidationReport(
        results=[*report.results, *build_failures],
        warnings=[*report.warnings, *(r.message for r in build_failures if r.status == "warn")],
        estimated_share=report.estimated_share,
        unknown_count=report.unknown_count,
    )


# --- Phase 4: repair ---


def _failure_payload(results: list[RuleResult]) -> list[dict]:
    details = [r for r in results if not r.summary and r.status == "fail"]
    details.sort(key=lambda r: (r.rule, r.day if r.day is not None else -1, r.stop if r.stop is not None else -1))

    payload = [
        {
            "rule": r.rule,
            "code": r.code,
            "day": r.day,
            "stop": r.stop,
            "source_id": r.source_id,
            "message": r.message,
        }
        for r in details[:MAX_REPAIR_ITEMS]
    ]
    if len(details) > MAX_REPAIR_ITEMS:
        payload.append({"note": f"...and {len(details) - MAX_REPAIR_ITEMS} more"})
    return payload


def _propose_and_repair(
    client: OpenAI,
    model: str,
    request: TripRequest,
    registry: PlaceRegistry,
    skeleton: list[tuple[date, str]],
    trace: Trace,
    stop_mode: str,
    max_repairs: int,
    cap_hit: bool,
    forecast_failures: dict[str, str],
) -> tuple[Itinerary, ValidationReport, ValidationContext, int]:
    messages = _initial_proposal_messages(request, skeleton, registry, stop_mode)
    proposal = _run_proposal_phase(client, model, messages, trace, "propose", stop_mode)

    itinerary, build_failures = _build_itinerary(
        proposal,
        skeleton,
        registry,
        request,
        stop_mode,
        cap_hit,
        forecast_failures=forecast_failures,
    )
    context = _build_context(registry, request)
    report = _merge_build_failures(validate(itinerary, context), build_failures)
    trace.validation(0, report)

    repairs = 0
    last_proposal = proposal.model_dump()

    while not report.ok and repairs < max_repairs:
        failures = _failure_payload(report.results)
        trace.repair(repairs + 1, failures)

        messages = _repair_messages(request, registry, last_proposal, failures)
        proposal = _run_proposal_phase(client, model, messages, trace, "repair", stop_mode)

        itinerary, build_failures = _build_itinerary(
            proposal,
            skeleton,
            registry,
            request,
            stop_mode,
            cap_hit,
            forecast_failures=forecast_failures,
        )
        context = _build_context(registry, request)
        report = _merge_build_failures(validate(itinerary, context), build_failures)
        repairs += 1
        last_proposal = proposal.model_dump()
        trace.validation(repairs, report)

    itinerary.verified = report.ok
    return itinerary, report, context, repairs


# --- Entry point ---


def plan_trip(
    request: TripRequest,
    *,
    client: OpenAI | None = None,
    http_client: httpx.Client | None = None,
    model: str = DEFAULT_MODEL,
    stop_mode: Literal["ids", "full"] = "ids",
    trace_dir: Path | None = TRACE_DIR,
    max_tool_steps: int = MAX_TOOL_STEPS,
    max_repairs: int = MAX_REPAIRS,
) -> AgentResult:
    run_id = uuid.uuid4().hex
    trace = Trace(run_id, request.model_dump(mode="json"))
    registry = PlaceRegistry()
    llm_client = client or llm.build_client()

    try:
        skeleton = _build_skeleton(request)
        _geocode_route(request, registry, http_client, trace)
        forecast_failures, forecast_warnings = _forecast_route(request, registry, http_client, trace)
        tool_calls = _run_tool_phase(
            llm_client, model, request, registry, trace, http_client, max_tool_steps
        )
        cap_hit = tool_calls >= max_tool_steps
        itinerary, report, context, repairs = _propose_and_repair(
            llm_client,
            model,
            request,
            registry,
            skeleton,
            trace,
            stop_mode,
            max_repairs,
            cap_hit,
            forecast_failures,
        )
        itinerary.warnings = [*itinerary.warnings, *forecast_warnings]
        trace.finish("ok" if report.ok else "unverified", repairs=repairs, tool_calls=tool_calls)
    except Exception:
        trace.finish("error")
        raise
    finally:
        trace_path = trace.write(trace_dir)

    return AgentResult(
        itinerary=itinerary,
        report=report,
        context=context,
        repairs=repairs,
        tool_calls=tool_calls,
        trace_path=trace_path,
    )
