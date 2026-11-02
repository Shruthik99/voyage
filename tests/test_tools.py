import time
import traceback
from datetime import date

import httpx
import pytest

from src import tools as tools_module
from src.tools import ForecastDay, GeoPoint, ToolError, geocode, get_forecast, search_places

GEOCODE_OK = {"results": [{"lat": 25.7617, "lon": -80.1918, "state_code": "FL"}]}


@pytest.fixture(autouse=True)
def _isolated_cache(tmp_path, monkeypatch):
    monkeypatch.setattr(tools_module, "CACHE_DIR", tmp_path / "cache")


@pytest.fixture(autouse=True)
def _api_key(monkeypatch):
    monkeypatch.setenv("GEOAPIFY_KEY", "test-key")


def _client(handler):
    return httpx.Client(transport=httpx.MockTransport(handler))


def test_geocode_success():
    def handler(request):
        return httpx.Response(200, json=GEOCODE_OK)

    with _client(handler) as client:
        point = geocode("Miami, FL", client=client)

    assert point == GeoPoint(city="Miami", state="FL", lat=25.7617, lon=-80.1918)


def test_geocode_state_mismatch_raises():
    def handler(request):
        return httpx.Response(
            200, json={"results": [{"lat": 25.7617, "lon": -80.1918, "state_code": "GA"}]}
        )

    with _client(handler) as client, pytest.raises(ToolError):
        geocode("Miami, FL", client=client)


def test_search_places_empty_result_returns_empty_list():
    def handler(request):
        if "geocode" in str(request.url):
            return httpx.Response(200, json=GEOCODE_OK)
        return httpx.Response(200, json={"features": []})

    with _client(handler) as client:
        places = search_places("Miami, FL", "museums", client=client)

    assert places == []


def test_search_places_skips_unnamed_results():
    def handler(request):
        if "geocode" in str(request.url):
            return httpx.Response(200, json=GEOCODE_OK)
        return httpx.Response(
            200,
            json={
                "features": [
                    {"properties": {"formatted": "no name", "lat": 1, "lon": 2, "place_id": "a"}},
                    {
                        "properties": {
                            "name": "History Museum",
                            "formatted": "123 Main St",
                            "state_code": "FL",
                            "lat": 25.7,
                            "lon": -80.1,
                            "place_id": "b",
                        }
                    },
                ]
            },
        )

    with _client(handler) as client:
        places = search_places("Miami, FL", "museums", client=client)

    assert len(places) == 1
    assert places[0].name == "History Museum"
    assert places[0].source == "geoapify"
    assert places[0].source_id == "b"


def test_search_places_unknown_interest_raises():
    with _client(lambda request: httpx.Response(500)) as client, pytest.raises(ToolError):
        search_places("Miami, FL", "skydiving", client=client)


def test_retry_on_503_then_success():
    attempts = {"n": 0}

    def handler(request):
        attempts["n"] += 1
        if attempts["n"] == 1:
            return httpx.Response(503)
        return httpx.Response(200, json=GEOCODE_OK)

    with _client(handler) as client:
        point = geocode("Miami, FL", client=client)

    assert attempts["n"] == 2
    assert point.lat == 25.7617


def test_no_retry_on_400():
    attempts = {"n": 0}

    def handler(request):
        attempts["n"] += 1
        return httpx.Response(400, json={"error": "bad request"})

    with _client(handler) as client, pytest.raises(ToolError):
        geocode("Miami, FL", client=client)

    assert attempts["n"] == 1


def test_cache_hit_avoids_second_request():
    attempts = {"n": 0}

    def handler(request):
        attempts["n"] += 1
        return httpx.Response(200, json=GEOCODE_OK)

    with _client(handler) as client:
        geocode("Miami, FL", client=client)
        geocode("Miami, FL", client=client)

    assert attempts["n"] == 1


def test_missing_api_key_raises(monkeypatch):
    monkeypatch.delenv("GEOAPIFY_KEY", raising=False)

    with _client(lambda request: httpx.Response(500)) as client, pytest.raises(ToolError):
        geocode("Miami, FL", client=client)


def test_get_forecast_success():
    def handler(request):
        if "geocode" in str(request.url):
            return httpx.Response(200, json=GEOCODE_OK)
        return httpx.Response(
            200,
            json={
                "daily": {
                    "time": ["2026-11-03", "2026-11-04"],
                    "temperature_2m_max": [70.5, 68.2],
                    "precipitation_probability_max": [10, 40],
                }
            },
        )

    with _client(handler) as client:
        forecast = get_forecast("Miami, FL", 2, client=client)

    assert forecast == [
        ForecastDay(date="2026-11-03", rain_chance=10, high_f=70.5, source="open-meteo"),
        ForecastDay(date="2026-11-04", rain_chance=40, high_f=68.2, source="open-meteo"),
    ]


# --- 1. API key never appears in error messages or tracebacks ---


def test_api_key_never_leaks_on_401(monkeypatch):
    monkeypatch.setenv("GEOAPIFY_KEY", "SECRET123")

    def handler(request):
        assert "apiKey=SECRET123" in str(request.url)
        return httpx.Response(401, text="unauthorized")

    with _client(handler) as client, pytest.raises(ToolError) as excinfo:
        geocode("Miami, FL", client=client)

    assert "SECRET123" not in str(excinfo.value)
    full_traceback = "".join(
        traceback.format_exception(excinfo.type, excinfo.value, excinfo.tb)
    )
    assert "SECRET123" not in full_traceback


def test_api_key_never_leaks_on_retryable_5xx(monkeypatch):
    monkeypatch.setenv("GEOAPIFY_KEY", "SECRET123")

    def handler(request):
        assert "apiKey=SECRET123" in str(request.url)
        return httpx.Response(503, text="server error")

    with _client(handler) as client, pytest.raises(ToolError) as excinfo:
        geocode("Miami, FL", client=client)

    full_traceback = "".join(
        traceback.format_exception(excinfo.type, excinfo.value, excinfo.tb)
    )
    assert "SECRET123" not in str(excinfo.value)
    assert "SECRET123" not in full_traceback


# --- 2. apiKey excluded from the cache key ---


def test_cache_key_ignores_api_key(monkeypatch):
    attempts = {"n": 0}

    def handler(request):
        attempts["n"] += 1
        return httpx.Response(200, json=GEOCODE_OK)

    with _client(handler) as client:
        geocode("Miami, FL", client=client)
        monkeypatch.setenv("GEOAPIFY_KEY", "different-key")
        geocode("Miami, FL", client=client)

    assert attempts["n"] == 1


# --- 3. forecast cache key includes today's date ---


def test_forecast_cache_key_changes_with_date(monkeypatch):
    attempts = {"n": 0}

    def handler(request):
        attempts["n"] += 1
        if "geocode" in str(request.url):
            return httpx.Response(200, json=GEOCODE_OK)
        return httpx.Response(
            200,
            json={
                "daily": {
                    "time": ["2026-11-03"],
                    "temperature_2m_max": [70.5],
                    "precipitation_probability_max": [10],
                }
            },
        )

    monkeypatch.setattr(tools_module, "_today", lambda: date(2026, 11, 2))
    with _client(handler) as client:
        get_forecast("Miami, FL", 1, client=client)
        get_forecast("Miami, FL", 1, client=client)

    assert attempts["n"] == 2  # geocode cached after first call, forecast not re-fetched

    monkeypatch.setattr(tools_module, "_today", lambda: date(2026, 11, 3))
    with _client(handler) as client:
        get_forecast("Miami, FL", 1, client=client)

    assert attempts["n"] == 3  # new day -> forecast re-fetched despite same params


# --- 4. every failure raises ToolError ---


def test_connect_error_raises_tool_error():
    def handler(request):
        raise httpx.ConnectError("connection refused", request=request)

    with _client(handler) as client, pytest.raises(ToolError):
        geocode("Miami, FL", client=client)


def test_invalid_json_raises_tool_error():
    def handler(request):
        return httpx.Response(200, text="not json")

    with _client(handler) as client, pytest.raises(ToolError):
        geocode("Miami, FL", client=client)


def test_search_places_skips_feature_missing_place_id():
    def handler(request):
        if "geocode" in str(request.url):
            return httpx.Response(200, json=GEOCODE_OK)
        return httpx.Response(
            200,
            json={
                "features": [
                    {
                        "properties": {
                            "name": "No ID Place",
                            "formatted": "1 Main St",
                            "state_code": "FL",
                            "lat": 25.7,
                            "lon": -80.1,
                        }
                    },
                    {
                        "properties": {
                            "name": "History Museum",
                            "formatted": "123 Main St",
                            "state_code": "FL",
                            "lat": 25.7,
                            "lon": -80.1,
                            "place_id": "b",
                        }
                    },
                ]
            },
        )

    with _client(handler) as client:
        places = search_places("Miami, FL", "museums", client=client)

    assert len(places) == 1
    assert places[0].source_id == "b"


# --- 5. Place.state is str | None, never defaulted to the searched city's state ---


def test_place_state_none_when_missing_from_result():
    def handler(request):
        if "geocode" in str(request.url):
            return httpx.Response(200, json=GEOCODE_OK)
        return httpx.Response(
            200,
            json={
                "features": [
                    {
                        "properties": {
                            "name": "History Museum",
                            "formatted": "123 Main St",
                            "lat": 25.7,
                            "lon": -80.1,
                            "place_id": "b",
                        }
                    }
                ]
            },
        )

    with _client(handler) as client:
        places = search_places("Miami, FL", "museums", client=client)

    assert places[0].state is None


# --- 6. geocode picks the first result whose state_code matches ---


def test_geocode_skips_wrong_state_picks_matching_result():
    def handler(request):
        return httpx.Response(
            200,
            json={
                "results": [
                    {"lat": 1.0, "lon": 2.0, "state_code": "GA"},
                    {"lat": 25.7617, "lon": -80.1918, "state_code": "FL"},
                ]
            },
        )

    with _client(handler) as client:
        point = geocode("Miami, FL", client=client)

    assert point == GeoPoint(city="Miami", state="FL", lat=25.7617, lon=-80.1918)


# --- 7. RETRY_WAIT_MAX controls the retry backoff ceiling ---


def test_retry_wait_max_near_zero_is_fast(monkeypatch):
    monkeypatch.setattr(tools_module, "RETRY_WAIT_MAX", 0.0)
    attempts = {"n": 0}

    def handler(request):
        attempts["n"] += 1
        if attempts["n"] < 3:
            return httpx.Response(503)
        return httpx.Response(200, json=GEOCODE_OK)

    start = time.monotonic()
    with _client(handler) as client:
        geocode("Miami, FL", client=client)
    elapsed = time.monotonic() - start

    assert attempts["n"] == 3
    assert elapsed < 1.0
