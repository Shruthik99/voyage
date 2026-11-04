import importlib.util
import time
import traceback
from datetime import date
from pathlib import Path

import httpx
import pytest

from src import tools as tools_module
from src.tools import ForecastDay, GeoPoint, ToolError, geocode, get_forecast, search_places

GEOCODE_OK = {
    "results": [
        {
            "city": "Miami",
            "name": "Miami",
            "state_code": "FL",
            "result_type": "city",
            "lat": 25.7617,
            "lon": -80.1918,
        }
    ]
}


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
            200,
            json={
                "results": [
                    {
                        "city": "Miami",
                        "name": "Miami",
                        "state_code": "GA",
                        "result_type": "city",
                        "lat": 25.7617,
                        "lon": -80.1918,
                    }
                ]
            },
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


def test_get_forecast_drops_trailing_incomplete_day():
    dates = [f"2026-11-{3 + i:02d}" for i in range(16)]
    highs = [70.0 + i for i in range(16)]
    highs[-1] = None
    rain = [10 + i for i in range(16)]

    def handler(request):
        if "geocode" in str(request.url):
            return httpx.Response(200, json=GEOCODE_OK)
        return httpx.Response(
            200,
            json={
                "daily": {
                    "time": dates,
                    "temperature_2m_max": highs,
                    "precipitation_probability_max": rain,
                }
            },
        )

    with _client(handler) as client:
        forecast = get_forecast("Miami, FL", 16, client=client)

    assert len(forecast) == 15
    assert forecast[-1].date.isoformat() == "2026-11-17"


def test_get_forecast_drops_middle_day_missing_rain_chance():
    def handler(request):
        if "geocode" in str(request.url):
            return httpx.Response(200, json=GEOCODE_OK)
        return httpx.Response(
            200,
            json={
                "daily": {
                    "time": ["2026-11-03", "2026-11-04", "2026-11-05"],
                    "temperature_2m_max": [70.5, 68.2, 65.0],
                    "precipitation_probability_max": [10, None, 30],
                }
            },
        )

    with _client(handler) as client:
        forecast = get_forecast("Miami, FL", 3, client=client)

    assert [f.date.isoformat() for f in forecast] == ["2026-11-03", "2026-11-05"]


def test_get_forecast_mismatched_array_lengths_raises():
    def handler(request):
        if "geocode" in str(request.url):
            return httpx.Response(200, json=GEOCODE_OK)
        return httpx.Response(
            200,
            json={
                "daily": {
                    "time": ["2026-11-03", "2026-11-04"],
                    "temperature_2m_max": [70.5],
                    "precipitation_probability_max": [10, 40],
                }
            },
        )

    with _client(handler) as client, pytest.raises(ToolError):
        get_forecast("Miami, FL", 2, client=client)


def test_get_forecast_all_days_incomplete_raises():
    def handler(request):
        if "geocode" in str(request.url):
            return httpx.Response(200, json=GEOCODE_OK)
        return httpx.Response(
            200,
            json={
                "daily": {
                    "time": ["2026-11-03", "2026-11-04"],
                    "temperature_2m_max": [None, None],
                    "precipitation_probability_max": [10, 40],
                }
            },
        )

    with _client(handler) as client, pytest.raises(ToolError):
        get_forecast("Miami, FL", 2, client=client)


def test_get_forecast_missing_daily_key_raises():
    def handler(request):
        if "geocode" in str(request.url):
            return httpx.Response(200, json=GEOCODE_OK)
        return httpx.Response(200, json={})

    with _client(handler) as client, pytest.raises(ToolError):
        get_forecast("Miami, FL", 2, client=client)


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
                    {
                        "city": "Miami",
                        "name": "Miami",
                        "lat": 1.0,
                        "lon": 2.0,
                        "state_code": "GA",
                        "result_type": "city",
                    },
                    {
                        "city": "Miami",
                        "name": "Miami",
                        "lat": 25.7617,
                        "lon": -80.1918,
                        "state_code": "FL",
                        "result_type": "city",
                    },
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


# --- Fix 1: geocode verifies the matched place, not only the state ---


DENVR_CO_RESULTS = {
    "results": [
        {
            "city": "Denver",
            "name": "Denver",
            "state_code": "CO",
            "result_type": "city",
            "lat": 39.7392,
            "lon": -104.9903,
        },
        {
            "city": "Thornton",
            "name": "Hampton Inn Denvr Thorntn",
            "state_code": "CO",
            "result_type": "amenity",
            "lat": 39.87,
            "lon": -104.97,
        },
    ]
}


def test_geocode_rejects_misspelled_city_with_no_matching_result():
    def handler(request):
        return httpx.Response(200, json=DENVR_CO_RESULTS)

    with _client(handler) as client, pytest.raises(ToolError):
        geocode("Denvr, CO", client=client)


def test_geocode_accepts_correctly_spelled_city():
    def handler(request):
        return httpx.Response(200, json=DENVR_CO_RESULTS)

    with _client(handler) as client:
        point = geocode("Denver, CO", client=client)

    assert point == GeoPoint(city="Denver", state="CO", lat=39.7392, lon=-104.9903)


def test_geocode_rejects_amenity_only_match():
    def handler(request):
        return httpx.Response(200, json=DENVR_CO_RESULTS)

    with _client(handler) as client, pytest.raises(ToolError):
        geocode("Thornton, CO", client=client)


def test_geocode_picks_city_result_over_amenity_for_same_city():
    def handler(request):
        return httpx.Response(
            200,
            json={
                "results": [
                    {
                        "city": "Thornton",
                        "name": "Hampton Inn Thornton",
                        "state_code": "CO",
                        "result_type": "amenity",
                        "lat": 1.0,
                        "lon": 2.0,
                    },
                    {
                        "city": "Thornton",
                        "name": "Thornton",
                        "state_code": "CO",
                        "result_type": "city",
                        "lat": 39.87,
                        "lon": -104.97,
                    },
                ]
            },
        )

    with _client(handler) as client:
        point = geocode("Thornton, CO", client=client)

    assert point == GeoPoint(city="Thornton", state="CO", lat=39.87, lon=-104.97)


def test_geocode_matches_case_insensitively():
    def handler(request):
        return httpx.Response(200, json=GEOCODE_OK)

    with _client(handler) as client:
        point = geocode("miami, fl", client=client)

    assert point.city == "Miami"


# --- Fix 2: empty results/features are never cached ---


def test_geocode_empty_results_not_cached():
    attempts = {"n": 0}

    def handler(request):
        attempts["n"] += 1
        return httpx.Response(200, json={"results": []})

    with _client(handler) as client:
        with pytest.raises(ToolError):
            geocode("Miami, FL", client=client)
        with pytest.raises(ToolError):
            geocode("Miami, FL", client=client)

    assert attempts["n"] == 2


def test_search_places_empty_features_not_cached():
    attempts = {"n": 0}

    def handler(request):
        if "geocode" in str(request.url):
            return httpx.Response(200, json=GEOCODE_OK)
        attempts["n"] += 1
        return httpx.Response(200, json={"features": []})

    with _client(handler) as client:
        search_places("Miami, FL", "museums", client=client)
        search_places("Miami, FL", "museums", client=client)

    assert attempts["n"] == 2


# --- Fix 3: corrupt cache file is treated as a miss, cache writes are atomic ---


def test_corrupt_cache_file_treated_as_miss(tmp_path, monkeypatch):
    monkeypatch.setattr(tools_module, "CACHE_DIR", tmp_path / "cache")
    (tmp_path / "cache").mkdir(parents=True)

    key = tools_module._cache_key(
        tools_module.GEOAPIFY_GEOCODE_URL,
        {"text": "Miami, FL, USA", "filter": "countrycode:us", "format": "json"},
    )
    corrupt_path = (tmp_path / "cache") / f"{key}.json"
    corrupt_path.write_text("{not valid json")

    attempts = {"n": 0}

    def handler(request):
        attempts["n"] += 1
        return httpx.Response(200, json=GEOCODE_OK)

    with _client(handler) as client:
        point = geocode("Miami, FL", client=client)

    assert attempts["n"] == 1
    assert point.lat == 25.7617


# --- Fix 4: 4xx response body is redacted before it reaches the error ---


def test_4xx_body_redacted_in_error_and_traceback(monkeypatch):
    monkeypatch.setenv("GEOAPIFY_KEY", "test-key")

    def handler(request):
        return httpx.Response(401, text="unauthorized: apiKey=SECRET123")

    with _client(handler) as client, pytest.raises(ToolError) as excinfo:
        geocode("Miami, FL", client=client)

    assert "SECRET123" not in str(excinfo.value)
    full_traceback = "".join(
        traceback.format_exception(excinfo.type, excinfo.value, excinfo.tb)
    )
    assert "SECRET123" not in full_traceback


# --- Fix 5: a valid JSON body that isn't an object raises ToolError ---


def test_non_object_json_response_raises_tool_error():
    def handler(request):
        return httpx.Response(200, text="[]")

    with _client(handler) as client, pytest.raises(ToolError):
        geocode("Miami, FL", client=client)


# --- Fix 6: CACHE_DIR is anchored to the project root ---


def test_cache_dir_anchored_to_project_root():
    # Load a fresh copy of the module (separate from sys.modules["src.tools"], which the
    # autouse _isolated_cache fixture has already patched) so CACHE_DIR reflects the
    # unpatched, real computation.
    module_path = Path(tools_module.__file__)
    spec = importlib.util.spec_from_file_location("tools_fresh_for_test", module_path)
    fresh = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(fresh)

    expected = module_path.resolve().parent.parent / "cache"
    assert fresh.CACHE_DIR == expected
