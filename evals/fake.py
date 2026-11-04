"""Dry run: a fake LLM client and an httpx.MockTransport handler good for any
trip in evals/trips.json, so ``evals.run --dry-run`` makes no network call of
any kind.

Unlike tests/test_agent.py's script-driven FakeClient, this one is
*responsive*: the runner makes a variable number of calls (city count, parse
retries, repairs), so the fake reacts to each call's kwargs instead of
replaying a fixed script.
"""

import hashlib
import json
import re
from datetime import date, timedelta
from itertools import count
from types import SimpleNamespace

import httpx
from openai import RateLimitError
from openai.types.chat import ChatCompletion

from src.constants import MAX_STOPS, MIN_STOPS

_FILTER_RE = re.compile(r"circle:([-\d.]+),([-\d.]+),")


# --- Chat completions: schema-valid, catalog-grounded, reacting to kwargs ---


def _completion(message: dict, usage: dict | None = None) -> ChatCompletion:
    data = {
        "id": "chatcmpl-dry-run",
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


def _no_tool_calls() -> ChatCompletion:
    return _completion({"role": "assistant", "content": ""})


def _synthetic_usage(messages: list[dict], content: str) -> dict:
    prompt = max(1, round(len(json.dumps(messages)) / 3.5))
    completion = max(1, round(len(content) / 3.5))
    return {"prompt_tokens": prompt, "completion_tokens": completion, "total_tokens": prompt + completion}


def _tpd_429() -> RateLimitError:
    request = httpx.Request("POST", "https://api.groq.com/openai/v1/chat/completions")
    message = "Rate limit reached for tokens per day (TPD): limit reached"
    response = httpx.Response(429, request=request, json={"error": {"message": message}})
    return RateLimitError(message, response=response, body=None)


def _try_json(content: str | None) -> dict | None:
    if not content:
        return None
    try:
        parsed = json.loads(content)
    except json.JSONDecodeError:
        return None
    return parsed if isinstance(parsed, dict) else None


def _initial_payload(messages: list[dict]) -> dict | None:
    """The one user message carrying stop_mode -- present only on the first
    proposal attempt, never on a repair.
    """
    for m in messages:
        if m["role"] == "user":
            parsed = _try_json(m["content"])
            if parsed is not None and "stop_mode" in parsed:
                return parsed
    return None


def _assistant_payload(messages: list[dict]) -> dict:
    return next(_try_json(m["content"]) for m in messages if m["role"] == "assistant")


def _mode_from_assistant(messages: list[dict]) -> str:
    payload = _assistant_payload(messages)
    first_day = payload["days"][0] if payload.get("days") else {}
    return "ids" if "stop_ids" in first_day else "full"


def _repair_catalog_by_city(messages: list[dict]) -> dict[str, list[dict]]:
    for m in messages:
        if m["role"] != "user":
            continue
        parsed = _try_json(m["content"])
        if parsed is not None and "catalog" in parsed:
            return parsed["catalog"]
    return {}


_GHOST_ID = "ghost-hallucinated"


def _stops_per_day(fail: bool) -> int:
    return 1 if fail else min(MAX_STOPS, max(MIN_STOPS, 3))


def _round_robin_by_interest(catalog: list[dict]) -> list[dict]:
    """Reorder so consecutive items cycle through every interest present --
    otherwise the first few picks are all one interest's catalog block, and a
    multi-interest trip fails V3 even though the catalog had coverage.
    """
    groups: dict[str, list[dict]] = {}
    for place in catalog:
        groups.setdefault(place["interest"], []).append(place)
    order = []
    while any(groups.values()):
        for places in groups.values():
            if places:
                order.append(places.pop(0))
    return order


def _pick_for_day(
    day: int, day_city: dict[int, str], catalog_by_city: dict[str, list[dict]],
    city_offset: dict[str, int], needed: int,
) -> list[dict]:
    """needed items from this day's own city catalog, rotating per city so two
    days in the same city do not always propose the identical stops.
    """
    city = day_city.get(day, "")
    catalog = _round_robin_by_interest(catalog_by_city.get(city, []))
    offset = city_offset.get(city, 0)
    city_offset[city] = offset + needed
    if not catalog or needed <= 0:
        return []
    return [catalog[(offset + i) % len(catalog)] for i in range(needed)]


def _ids_proposal(
    days: list[int], day_city: dict[int, str], catalog_by_city: dict[str, list[dict]],
    *, fail: bool, hallucinate: bool,
) -> dict:
    needed = _stops_per_day(fail)
    city_offset: dict[str, int] = {}
    out_days = []
    for idx, day in enumerate(days):
        picks = _pick_for_day(day, day_city, catalog_by_city, city_offset, needed)
        stop_ids = [p["source_id"] for p in picks]
        if hallucinate and idx == 0:
            stop_ids.append(_GHOST_ID)
        out_days.append({"day": day, "stop_ids": stop_ids})
    return {"days": out_days}


def _full_proposal(
    days: list[int], day_city: dict[int, str], catalog_by_city: dict[str, list[dict]],
    *, fail: bool, hallucinate: bool,
) -> dict:
    needed = _stops_per_day(fail)
    city_offset: dict[str, int] = {}
    out_days = []
    for idx, day in enumerate(days):
        picks = _pick_for_day(day, day_city, catalog_by_city, city_offset, needed)
        stops = [
            {
                "name": p["name"],
                "address": "123 Synthetic St",
                "state": p["state"],
                "lat": p["lat"],
                "lon": p["lon"],
                "interest": p["interest"],
                "source": "geoapify",
                "source_id": p["source_id"],
            }
            for p in picks
        ]
        if hallucinate and idx == 0:
            stops.append(
                {
                    "name": "Invented Place",
                    "address": "nowhere",
                    "state": "ZZ",
                    "lat": 0.0,
                    "lon": 0.0,
                    "interest": "museums",
                    "source": "made-up",
                    "source_id": _GHOST_ID,
                }
            )
        out_days.append({"day": day, "stops": stops})
    return {"days": out_days}


class FakeLLMClient:
    """Duck-types client.chat.completions.create(**kwargs) for one trip.

    route/interests are passed in rather than parsed back out of the prompt
    text, so this has no dependency on src.agent's exact wording. A fresh
    instance per trip is the intended usage (the runner already builds a new
    TripRequest per trip); raise_tpd_429_after is the one piece of state that
    persists across calls, to simulate the daily limit landing mid-trip.
    """

    def __init__(
        self,
        *,
        route: list[str],
        interests: list[str],
        max_tool_steps: int = 12,
        hallucinate_id: bool = False,
        fail_first_attempt: bool = False,
        raise_tpd_429_after: int | None = None,
    ) -> None:
        self.route = list(dict.fromkeys(route))
        self.interests = list(dict.fromkeys(interests))
        self.max_tool_steps = max_tool_steps
        self.hallucinate_id = hallucinate_id
        self.fail_first_attempt = fail_first_attempt
        self.raise_tpd_429_after = raise_tpd_429_after
        self._calls = 0
        self._day_city: dict[int, str] = {}  # set from the initial skeleton, reused by repairs
        self.chat = SimpleNamespace(completions=self)

    def create(self, **kwargs):
        self._calls += 1
        if self.raise_tpd_429_after is not None and self._calls > self.raise_tpd_429_after:
            raise _tpd_429()
        if "tools" in kwargs:
            return self._tool_response(kwargs["messages"])
        if "response_format" in kwargs:
            return self._proposal_response(kwargs["messages"])
        raise AssertionError("fake LLM client received neither tools nor response_format")

    def _tool_response(self, messages: list[dict]) -> ChatCompletion:
        queue = [(city, interest) for city in self.route for interest in self.interests]
        already_called = sum(1 for m in messages if m.get("role") == "tool")

        if already_called < len(queue) and already_called < self.max_tool_steps:
            city, interest = queue[already_called]
            return _tool_call_message(
                [("search_places", {"city_state": city, "interest": interest, "limit": None})]
            )
        return _no_tool_calls()

    def _proposal_response(self, messages: list[dict]) -> ChatCompletion:
        initial = _initial_payload(messages)
        if initial is not None:
            mode = initial["stop_mode"]
            days = [d["day"] for d in initial["days"]]
            self._day_city = {d["day"]: d["city"] for d in initial["days"]}
            catalog_by_city = initial["catalog"]
            is_first_attempt = True
        else:
            mode = _mode_from_assistant(messages)
            days = [d["day"] for d in _assistant_payload(messages)["days"]]
            catalog_by_city = _repair_catalog_by_city(messages)
            is_first_attempt = False

        fail = self.fail_first_attempt and is_first_attempt
        builder = _ids_proposal if mode == "ids" else _full_proposal
        payload = builder(
            days, self._day_city, catalog_by_city, fail=fail, hallucinate=self.hallucinate_id
        )

        content = json.dumps(payload)
        return _completion(
            {"role": "assistant", "content": content}, usage=_synthetic_usage(messages, content)
        )


# --- HTTP: geocode, search_places, get_forecast, with no real network ---


def _city_point(city: str, state: str) -> tuple[float, float]:
    """A deterministic, distinct point per city -- not geographically real,
    but stable across calls so a day's stops land near its own city.
    """
    digest = hashlib.sha256(f"{city},{state}".encode()).hexdigest()
    h = int(digest, 16)
    lat = 25.0 + (h % 2000) / 100.0  # 25.00 .. 44.99
    lon = -125.0 + ((h // 2000) % 6000) / 100.0  # -125.00 .. -65.01
    return lat, lon


def _geocode_response(request: httpx.Request, points: dict[tuple[float, float], str]) -> httpx.Response:
    text = request.url.params.get("text", "")
    parts = [p.strip() for p in text.split(",")]
    city, state = parts[0], parts[1]
    lat, lon = _city_point(city, state)
    points[(lon, lat)] = state
    return httpx.Response(
        200,
        json={
            "results": [
                {
                    "city": city,
                    "name": city,
                    "state_code": state,
                    "result_type": "city",
                    "lat": lat,
                    "lon": lon,
                }
            ]
        },
    )


def _places_response(
    request: httpx.Request, points: dict[tuple[float, float], str], ids: "count[int]"
) -> httpx.Response:
    match = _FILTER_RE.search(request.url.params.get("filter", ""))
    limit = int(request.url.params.get("limit", "20"))
    if match is None:
        return httpx.Response(200, json={"features": []})

    lon, lat = float(match.group(1)), float(match.group(2))
    state = points.get((lon, lat), "ZZ")
    category = request.url.params.get("categories", "unknown").split(",")[0]

    features = []
    for _ in range(min(limit, 6)):
        idx = next(ids)
        jitter = (idx % 5) * 0.01  # well under RADIUS_KM, so V1's distance check passes
        features.append(
            {
                "properties": {
                    "name": f"Synthetic {category} {idx}",
                    "formatted": f"{idx} Synthetic St",
                    "state_code": state,
                    "lat": lat + jitter,
                    "lon": lon + jitter,
                    "place_id": f"dry-{idx}",
                }
            }
        )
    return httpx.Response(200, json={"features": features})


def _forecast_response(request: httpx.Request) -> httpx.Response:
    forecast_days = int(request.url.params.get("forecast_days", "16"))
    today = date.today()
    dates = [(today + timedelta(days=i)).isoformat() for i in range(forecast_days)]
    return httpx.Response(
        200,
        json={
            "daily": {
                "time": dates,
                "temperature_2m_max": [70.0] * forecast_days,
                "precipitation_probability_max": [10.0] * forecast_days,
            }
        },
    )


def make_dry_run_handler():
    """An httpx.MockTransport handler good for any trip: deterministic
    synthetic places inside the real radius/day-distance limits, and a full
    forecast window. Reuse one instance across a whole dry-run campaign so
    source_ids stay unique and a city's own geocoded point is remembered.
    """
    points: dict[tuple[float, float], str] = {}
    ids = count(1)

    def handler(request: httpx.Request) -> httpx.Response:
        url = str(request.url)
        if "geocode/search" in url:
            return _geocode_response(request, points)
        if "v2/places" in url:
            return _places_response(request, points, ids)
        if "open-meteo" in url:
            return _forecast_response(request)
        return httpx.Response(404)

    return handler
