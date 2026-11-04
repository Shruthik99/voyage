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


def test_trace_llm_call_records_usage_tokens():
    trace = Trace("run-usage", {})
    trace.llm_call(
        "propose",
        model="openai/gpt-oss-120b",
        has_tools=False,
        has_schema=True,
        duration_ms=5.0,
        prompt_tokens=123,
        completion_tokens=45,
    )

    event = trace.events[0]
    assert event["prompt_tokens"] == 123
    assert event["completion_tokens"] == 45


def test_trace_llm_call_usage_defaults_to_none_when_not_given():
    trace = Trace("run-usage-none", {})
    trace.llm_call(
        "tools", model="openai/gpt-oss-120b", has_tools=True, has_schema=False, duration_ms=5.0
    )

    event = trace.events[0]
    assert event["prompt_tokens"] is None
    assert event["completion_tokens"] is None


def test_trace_validation_records_stop_counts_and_day_km():
    trace = Trace("run-stats", {})
    trace.validation(0, _report(), stops=4, stops_proposed=6, day_km=[1.5, 2.0])

    event = trace.events[0]
    assert event["stops"] == 4
    assert event["stops_proposed"] == 6
    assert event["day_km"] == [1.5, 2.0]


def test_trace_validation_stats_default_to_none():
    trace = Trace("run-stats-none", {})
    trace.validation(0, _report())

    event = trace.events[0]
    assert event["stops"] is None
    assert event["stops_proposed"] is None
    assert event["day_km"] is None


def test_trace_adopt_copies_events_marked_shared():
    gather = Trace("gather", {})
    gather.tool_call("geocode", {"city_state": "Austin, TX"}, duration_ms=1.0)
    gather.llm_call("tools", model="m", has_tools=True, has_schema=False, duration_ms=2.0)

    arm = Trace("arm", {})
    arm.adopt(gather)
    arm.llm_call("propose", model="m", has_tools=False, has_schema=True, duration_ms=3.0)

    assert [e["kind"] for e in arm.events] == ["tool_call", "llm_call", "llm_call"]
    assert [e.get("shared") for e in arm.events] == [True, True, None]
    # The adopted events are copies: the arm must not be able to edit the source.
    arm.events[0]["name"] = "changed"
    assert gather.events[0]["name"] == "geocode"
    assert "shared" not in gather.events[0]


def test_trace_adopt_lets_two_traces_share_one_gather():
    gather = Trace("gather", {})
    gather.tool_call("search_places", {"city_state": "Austin, TX"}, duration_ms=1.0)

    first, second = Trace("a", {}), Trace("b", {})
    for arm in (first, second):
        arm.adopt(gather)

    assert len(first.events) == len(second.events) == 1
    assert first.events[0] is not second.events[0]
