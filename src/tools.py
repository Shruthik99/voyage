"""External API tools: geocoding, place search, weather forecast.

Each tool accepts an optional httpx.Client so tests can inject a mock transport.
Tools raise ToolError on failure; they never return placeholder or fallback data.
"""

import hashlib
import json
import os
import re
from collections.abc import Callable
from datetime import date
from pathlib import Path

import httpx
from dotenv import load_dotenv
from pydantic import BaseModel, ValidationError
from tenacity import retry, retry_if_exception_type, stop_after_attempt, wait_exponential

from src.parsing import parse_location

load_dotenv()

CACHE_DIR = Path(__file__).resolve().parent.parent / "cache"
TIMEOUT = 15.0
RETRY_WAIT_MAX = 2.0

GEOAPIFY_GEOCODE_URL = "https://api.geoapify.com/v1/geocode/search"
GEOAPIFY_PLACES_URL = "https://api.geoapify.com/v2/places"
OPEN_METEO_URL = "https://api.open-meteo.com/v1/forecast"

INTEREST_CATEGORIES = {
    "museums": "entertainment.museum",
    "history": "heritage",
    "hiking": "leisure.park,national_park,natural.mountain",
    "food": "catering.restaurant",
}

CITY_RESULT_TYPES = {"city"}

_API_KEY_PATTERN = re.compile(r"(apiKey=)[^&]*")


class ToolError(RuntimeError):
    """Raised when a tool cannot produce a real result."""


class _RetryableStatus(Exception):
    """Internal signal that a response was 429/5xx and should be retried."""

    def __init__(self, status_code: int, safe_url: str) -> None:
        self.status_code = status_code
        self.safe_url = safe_url
        super().__init__(f"{safe_url} returned {status_code}")


class GeoPoint(BaseModel):
    city: str
    state: str
    lat: float
    lon: float


class Place(BaseModel):
    name: str
    address: str
    state: str | None
    lat: float
    lon: float
    interest: str
    source: str
    source_id: str


class ForecastDay(BaseModel):
    date: date
    rain_chance: float
    high_f: float
    source: str


def _today() -> date:
    return date.today()


def _redact(url: str | httpx.URL) -> str:
    """Strip the apiKey query parameter from a URL for safe logging."""
    return _API_KEY_PATTERN.sub(r"\1REDACTED", str(url))


def _geoapify_key() -> str:
    key = os.environ.get("GEOAPIFY_KEY")
    if not key:
        raise ToolError("GEOAPIFY_KEY is not set in the environment")
    return key


def _cache_key(url: str, params: dict, extra: dict | None = None) -> str:
    safe_params = {k: v for k, v in params.items() if k != "apiKey"}
    payload: dict = {"url": url, "params": safe_params}
    if extra:
        payload["extra"] = extra
    raw = json.dumps(payload, sort_keys=True, default=str)
    return hashlib.sha256(raw.encode()).hexdigest()


def _cache_get(key: str) -> dict | None:
    path = CACHE_DIR / f"{key}.json"
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text())
    except (OSError, UnicodeDecodeError, json.JSONDecodeError):
        return None


def _cache_set(key: str, data: dict) -> None:
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    path = CACHE_DIR / f"{key}.json"
    tmp_path = path.with_name(f"{key}.tmp")
    tmp_path.write_text(json.dumps(data))
    tmp_path.replace(path)


def _raise_for_status(response: httpx.Response) -> None:
    if response.status_code < 400:
        return
    safe_url = _redact(response.request.url)
    if response.status_code == 429 or response.status_code >= 500:
        raise _RetryableStatus(response.status_code, safe_url)
    raise ToolError(
        f"{safe_url} returned {response.status_code}: {_redact(response.text)[:200]}"
    ) from None


def _wait_transient(retry_state):
    return wait_exponential(multiplier=0.1, min=0, max=RETRY_WAIT_MAX)(retry_state)


@retry(
    retry=retry_if_exception_type((httpx.RequestError, _RetryableStatus)),
    stop=stop_after_attempt(3),
    wait=_wait_transient,
    reraise=True,
)
def _request(client: httpx.Client, url: str, params: dict) -> httpx.Response:
    response = client.get(url, params=params, timeout=TIMEOUT)
    _raise_for_status(response)
    return response


def _get_json(
    client: httpx.Client,
    url: str,
    params: dict,
    cache_extra: dict | None = None,
    is_empty: Callable[[dict], bool] | None = None,
) -> dict:
    key = _cache_key(url, params, cache_extra)
    cached = _cache_get(key)
    if cached is not None:
        return cached

    safe_url = _redact(url)
    try:
        response = _request(client, url, params)
    except _RetryableStatus as exc:
        raise ToolError(f"{exc.safe_url} failed after retries: {exc.status_code}") from None
    except httpx.RequestError as exc:
        request_url = _redact(exc.request.url) if exc.request is not None else safe_url
        raise ToolError(f"{exc.__class__.__name__} calling {request_url}") from None

    try:
        data = response.json()
    except (json.JSONDecodeError, ValueError):
        raise ToolError(f"invalid JSON response from {safe_url}") from None

    if not isinstance(data, dict):
        raise ToolError(f"unexpected JSON response from {safe_url}: expected an object")

    if is_empty is None or not is_empty(data):
        _cache_set(key, data)
    return data


def geocode(city_state: str, client: httpx.Client | None = None) -> GeoPoint:
    city, state = parse_location(city_state)
    key = _geoapify_key()
    params = {
        "text": f"{city}, {state}, USA",
        "filter": "countrycode:us",
        "format": "json",
        "apiKey": key,
    }

    owns_client = client is None
    client = client or httpx.Client()
    try:
        data = _get_json(
            client,
            GEOAPIFY_GEOCODE_URL,
            params,
            is_empty=lambda d: not d.get("results"),
        )
    finally:
        if owns_client:
            client.close()

    results = data.get("results", [])
    requested_city = city.casefold().strip()
    match = next(
        (
            r
            for r in results
            if r.get("result_type") in CITY_RESULT_TYPES
            and (r.get("state_code") or "").upper() == state
            and (r.get("city") or r.get("name") or "").casefold().strip() == requested_city
        ),
        None,
    )
    if match is None:
        raise ToolError(f"no city result for {city_state!r} matched {city!r} in {state!r}")

    matched_city = match.get("city") or match.get("name")
    try:
        return GeoPoint(city=matched_city, state=state, lat=match.get("lat"), lon=match.get("lon"))
    except ValidationError:
        raise ToolError(f"malformed geocode result for {city_state!r}") from None


def search_places(
    city_state: str,
    interest: str,
    limit: int = 20,
    radius_m: int = 5000,
    client: httpx.Client | None = None,
) -> list[Place]:
    if interest not in INTEREST_CATEGORIES:
        raise ToolError(f"unknown interest: {interest!r}")

    owns_client = client is None
    client = client or httpx.Client()
    try:
        point = geocode(city_state, client=client)
        key = _geoapify_key()
        params = {
            "categories": INTEREST_CATEGORIES[interest],
            "filter": f"circle:{point.lon},{point.lat},{radius_m}",
            "bias": f"proximity:{point.lon},{point.lat}",
            "limit": limit,
            "apiKey": key,
        }
        data = _get_json(
            client,
            GEOAPIFY_PLACES_URL,
            params,
            is_empty=lambda d: not d.get("features"),
        )
    finally:
        if owns_client:
            client.close()

    places: list[Place] = []
    for feature in data.get("features", []):
        props = feature.get("properties", {})
        name = props.get("name")
        place_id = props.get("place_id")
        lat = props.get("lat")
        lon = props.get("lon")
        if not name or place_id is None or lat is None or lon is None:
            continue

        state_code = props.get("state_code")
        state = state_code.upper() if isinstance(state_code, str) else None

        try:
            places.append(
                Place(
                    name=name,
                    address=props.get("formatted", ""),
                    state=state,
                    lat=lat,
                    lon=lon,
                    interest=interest,
                    source="geoapify",
                    source_id=place_id,
                )
            )
        except ValidationError:
            raise ToolError(f"malformed place result from geoapify for {city_state!r}") from None

    return places


def get_forecast(
    city_state: str, days: int, client: httpx.Client | None = None
) -> list[ForecastDay]:
    owns_client = client is None
    client = client or httpx.Client()
    try:
        point = geocode(city_state, client=client)
        params = {
            "latitude": point.lat,
            "longitude": point.lon,
            "daily": "temperature_2m_max,precipitation_probability_max",
            "temperature_unit": "fahrenheit",
            "forecast_days": days,
            "timezone": "auto",
        }
        data = _get_json(
            client,
            OPEN_METEO_URL,
            params,
            cache_extra={"date": _today().isoformat()},
        )
    finally:
        if owns_client:
            client.close()

    daily = data.get("daily", {})
    dates = daily.get("time", [])
    highs = daily.get("temperature_2m_max", [])
    rain = daily.get("precipitation_probability_max", [])

    if not (len(dates) == len(highs) == len(rain)):
        raise ToolError("open-meteo returned mismatched daily arrays")

    try:
        return [
            ForecastDay(date=d, rain_chance=r, high_f=h, source="open-meteo")
            for d, h, r in zip(dates, highs, rain)
        ]
    except ValidationError:
        raise ToolError(f"malformed forecast data for {city_state!r}") from None
