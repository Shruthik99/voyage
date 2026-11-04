import json

from src.models import RuleResult, ValidationReport
from src.trace import Trace


def _report(status: str = "pass") -> ValidationReport:
    results = [RuleResult(rule="V1", code="located", status=status, message="ok", summary=True)]
    return ValidationReport(results=results, warnings=[] if status != "warn" else ["careful"])


def test_trace_write_produces_one_json_file_with_expected_shape(tmp_path):
    trace = Trace("run-123", {"route": ["Austin, TX"]})
    trace.tool_call(
        "search_places",
        {"city_state": "Austin, TX", "interest": "museums"},
        summary={"places": 3},
        error=None,
        duration_ms=12.3,
    )
    trace.validation(0, _report())
    trace.repair(1, [{"rule": "V8", "code": "too_few_stops", "day": 0}])
    trace.finish("ok", repairs=1, tool_calls=1)

    path = trace.write(tmp_path)

    files = list(tmp_path.glob("*.json"))
    assert files == [path]

    data = json.loads(path.read_text(encoding="utf-8"))
    assert data["status"] == "ok"
    assert data["run_id"] == "run-123"
    assert data["repairs"] == 1
    assert data["tool_calls"] == 1

    kinds = [e["kind"] for e in data["events"]]
    assert kinds == ["tool_call", "validation", "repair"]

    tool_event = data["events"][0]
    assert tool_event["name"] == "search_places"
    assert tool_event["summary"] == {"places": 3}
    assert tool_event["error"] is None
    assert tool_event["duration_ms"] == 12.3

    validation_event = data["events"][1]
    assert validation_event["attempt"] == 0
    assert "results" in validation_event


def test_trace_write_returns_none_when_trace_dir_is_none():
    trace = Trace("run-123", {})
    assert trace.write(None) is None


def test_trace_secret_scrub_removes_api_keys_from_file_bytes(tmp_path, monkeypatch):
    monkeypatch.setenv("GROQ_API_KEY", "SECRET123")
    monkeypatch.setenv("GEOAPIFY_KEY", "SECRET456")

    trace = Trace("run-456", {"note": "key is SECRET123"})
    trace.tool_call(
        "geocode",
        {"city_state": "Austin, TX"},
        summary={"url": "https://api.geoapify.com/v1/geocode/search?apiKey=SECRET456"},
        error=None,
        duration_ms=1.0,
    )
    trace.tool_call(
        "geocode",
        {"city_state": "Dallas, TX"},
        summary=None,
        error="auth failed with SECRET123",
        duration_ms=1.0,
    )
    trace.finish("ok")

    path = trace.write(tmp_path)
    raw = path.read_bytes().decode("utf-8")

    assert "SECRET123" not in raw
    assert "SECRET456" not in raw
    assert "apiKey=REDACTED" in raw


def test_trace_reasoning_recorded_as_char_count_not_text():
    trace = Trace("run-789", {})
    trace.llm_call(
        "propose",
        model="openai/gpt-oss-120b",
        has_tools=False,
        has_schema=True,
        reasoning_chars=42,
        duration_ms=5.0,
    )

    event = trace.events[0]
    assert event["reasoning_chars"] == 42
    assert "reasoning" not in event or event.get("reasoning") is None
    dumped = json.dumps(trace.to_dict())
    assert "chain of thought" not in dumped.lower()


def test_trace_writes_error_status_when_run_raises(tmp_path):
    trace = Trace("run-err", {})

    try:
        try:
            raise RuntimeError("boom")
        except RuntimeError:
            trace.finish("error")
            raise
    except RuntimeError:
        pass
    finally:
        path = trace.write(tmp_path)

    data = json.loads(path.read_text(encoding="utf-8"))
    assert data["status"] == "error"


def test_trace_tool_call_accepts_optional_summary_and_error():
    trace = Trace("run-opt", {})
    trace.tool_call("geocode", {"city_state": "Austin, TX"}, duration_ms=1.0)
    event = trace.events[0]
    assert event["summary"] is None
    assert event["error"] is None
