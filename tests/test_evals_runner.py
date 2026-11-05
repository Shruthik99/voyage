import json
from pathlib import Path
from types import SimpleNamespace

import pytest
from openai.types.chat import ChatCompletion

from evals import fake as fake_module
from evals import run as run_module
from src import tools as tools_module
from src.llm import LLMError


@pytest.fixture(autouse=True)
def _isolated_cache(tmp_path, monkeypatch):
    # --dry-run always overrides this with its own temp dir anyway; this just
    # guards against ever touching the real project cache/ from a test.
    monkeypatch.setattr(tools_module, "CACHE_DIR", tmp_path / "unused-cache")


@pytest.fixture(autouse=True)
def _api_key(monkeypatch):
    monkeypatch.setenv("GEOAPIFY_KEY", "test-key")


def _valid_trip(trip_id: str, *, start_in_days: int = 3) -> dict:
    return {
        "trip_id": trip_id,
        "label": "test fixture",
        "tags": ["single-city"],
        "expect": "plan",
        "request": {
            "route": ["Austin, TX"],
            "days_per_stop": 1,
            "travelers": 2,
            "budget_usd": 500,
            "meal_allowance_usd": 40,
            "interests": ["museums"],
            "start_in_days": start_in_days,
        },
    }


_INVALID_TRIP = {
    "trip_id": "invalid-trip",
    "label": "invalid fixture",
    "tags": ["invalid"],
    "expect": "reject",
    "request": {
        "route": ["Toronto, ON"],
        "days_per_stop": 1,
        "travelers": 1,
        "budget_usd": 100,
        "meal_allowance_usd": 10,
        "interests": [],
        "start_in_days": 3,
    },
}


def _write_trips(tmp_path: Path, trips: list[dict]) -> Path:
    path = tmp_path / "trips.json"
    path.write_text(json.dumps({"schema_version": 1, "trips": trips}))
    return path


def _read(path: Path) -> list[dict]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def _common_args(tmp_path: Path, trips_path: Path, out_path: Path) -> list[str]:
    return [
        "--dry-run",
        "--trips", str(trips_path),
        "--out", str(out_path),
        "--trace-dir", str(tmp_path / "traces"),
        # generous pacing so TokenBucket never actually sleeps in a test
        "--tpm", "10000000",
        "--daily-token-budget", "10000000",
    ]


# --- dry-run end to end ---


def test_dry_run_three_trips_gives_6_executed_plus_3_derived_rows(tmp_path):
    trips_path = _write_trips(tmp_path, [_valid_trip("t1"), _valid_trip("t2"), _valid_trip("t3")])
    out_path = tmp_path / "results.jsonl"

    code = run_module.main(_common_args(tmp_path, trips_path, out_path))

    assert code == 0
    rows = _read(out_path)
    assert len(rows) == 9

    by_trip: dict[str, set[str]] = {}
    for row in rows:
        by_trip.setdefault(row["trip_id"], set()).add(row["arm"])
    assert by_trip == {"t1": {"A", "B", "C"}, "t2": {"A", "B", "C"}, "t3": {"A", "B", "C"}}

    for trip_id in by_trip:
        c = next(r for r in rows if r["trip_id"] == trip_id and r["arm"] == "C")
        a = next(r for r in rows if r["trip_id"] == trip_id and r["arm"] == "A")
        b = next(r for r in rows if r["trip_id"] == trip_id and r["arm"] == "B")
        # same shared gather: same tool calls, same gather tokens
        assert c["tool_calls"] == a["tool_calls"]
        assert c["tokens_gather"] == a["tokens_gather"]
        assert b["tokens_billed"] == 0
        assert c["tokens_billed"] == c["tokens_total"]
        assert a["tokens_billed"] == a["tokens_arm"]


def test_resume_skips_done_pairs_and_force_reruns_them(tmp_path):
    trips_path = _write_trips(tmp_path, [_valid_trip("t1")])
    out_path = tmp_path / "results.jsonl"
    args = _common_args(tmp_path, trips_path, out_path)

    assert run_module.main(args) == 0
    first_rows = _read(out_path)
    assert len(first_rows) == 3

    assert run_module.main(args) == 0  # resume: already done, nothing to add
    assert _read(out_path) == first_rows

    assert run_module.main([*args, "--force"]) == 0
    forced_rows = _read(out_path)
    assert len(forced_rows) == 6


def test_invalid_trip_is_rejected_with_zero_calls_for_every_arm(tmp_path, monkeypatch):
    constructed = []
    original_init = fake_module.FakeLLMClient.__init__

    def counting_init(self, **kwargs):
        constructed.append(kwargs)
        original_init(self, **kwargs)

    monkeypatch.setattr(fake_module.FakeLLMClient, "__init__", counting_init)

    trips_path = _write_trips(tmp_path, [_INVALID_TRIP])
    out_path = tmp_path / "results.jsonl"

    code = run_module.main(_common_args(tmp_path, trips_path, out_path))

    assert code == 0
    assert constructed == []  # no LLM client was ever built for a trip that can't be planned

    rows = _read(out_path)
    assert len(rows) == 3
    assert {r["arm"] for r in rows} == {"A", "B", "C"}
    for row in rows:
        assert row["outcome"] == "rejected"
        assert row["tool_calls"] == 0
        assert row["tokens_billed"] == 0


def test_a_tpd_429_mid_campaign_stops_cleanly_with_prior_rows_intact(tmp_path, monkeypatch):
    trips_path = _write_trips(tmp_path, [_valid_trip("t1"), _valid_trip("t2"), _valid_trip("t3")])
    out_path = tmp_path / "results.jsonl"

    # Allow exactly trip t1's calls to succeed (2 gather + 1 arm C + 1 arm A),
    # then raise a real TPD 429 on every call after that -- a true cross-trip,
    # cross-instance budget exhaustion, not a per-instance counter.
    budget = {"calls": 0, "limit": 4}
    real_build_dry_run = run_module._build_dry_run

    def patched_build_dry_run(args):
        http_client, _ = real_build_dry_run(args)

        def inner_client_factory(route, interests):
            inner = fake_module.FakeLLMClient(route=route, interests=interests)
            real_create = inner.create

            def counted_create(**kwargs):
                budget["calls"] += 1
                if budget["calls"] > budget["limit"]:
                    raise fake_module._tpd_429()
                return real_create(**kwargs)

            inner.create = counted_create
            return inner

        return http_client, inner_client_factory

    monkeypatch.setattr(run_module, "_build_dry_run", patched_build_dry_run)

    code = run_module.main(_common_args(tmp_path, trips_path, out_path))

    assert code == 0
    rows = _read(out_path)
    assert {r["trip_id"] for r in rows} == {"t1"}  # t2 and t3 never got a single row
    assert {r["arm"] for r in rows} == {"A", "B", "C"}

    # Re-running (same budget object persists across this test's monkeypatch
    # still exhausted) proves prior rows are untouched and valid JSONL.
    assert run_module.main(_common_args(tmp_path, trips_path, out_path)) == 0
    assert _read(out_path) == rows


def test_no_row_contains_the_api_key_value(tmp_path, monkeypatch):
    monkeypatch.setenv("GEOAPIFY_KEY", "zz-not-a-real-secret-zz")
    trips_path = _write_trips(tmp_path, [_valid_trip("t1"), _INVALID_TRIP])
    out_path = tmp_path / "results.jsonl"

    run_module.main(_common_args(tmp_path, trips_path, out_path))

    raw = out_path.read_text(encoding="utf-8")
    assert "zz-not-a-real-secret-zz" not in raw


# --- Error rows must bill real tokens, not 0 ---
#
# Evidence: a live campaign trip's arm C got a Groq 400 on a repair call after
# the gather and two repairs had already spent real tokens. The row written
# for that run recorded outcome "error" with tokens_billed=0, even though the
# trace it was built from carried thousands of tokens' worth of llm_call
# events -- silently hiding real spend from the daily budget accounting.
#
# fake_module.FakeLLMClient never puts usage on its tool-phase completions
# (only its proposal completions carry synthetic usage), so these tests
# script their own completions to pin down exactly how many tokens a
# successful call "spent" before the failure, and assert the error row's
# tokens_billed against that exact number.


class _ScriptedClient:
    """A minimal client.chat.completions.create(**kwargs) stand-in: pops one
    scripted item per call, raising it if it's an exception.
    """

    def __init__(self, script: list):
        self._script = list(script)
        self.chat = SimpleNamespace(completions=self)

    def create(self, **kwargs):
        if not self._script:
            raise AssertionError("scripted client's script exhausted")
        item = self._script.pop(0)
        if isinstance(item, BaseException):
            raise item
        return item


def _completion(message: dict, *, prompt_tokens: int, completion_tokens: int) -> ChatCompletion:
    return ChatCompletion.model_validate(
        {
            "id": "chatcmpl-test",
            "object": "chat.completion",
            "created": 0,
            "model": "m",
            "choices": [{"index": 0, "finish_reason": "stop", "message": message}],
            "usage": {
                "prompt_tokens": prompt_tokens,
                "completion_tokens": completion_tokens,
                "total_tokens": prompt_tokens + completion_tokens,
            },
        }
    )


def _tool_call_completion(name: str, args: dict, *, prompt_tokens: int, completion_tokens: int) -> ChatCompletion:
    return _completion(
        {
            "role": "assistant",
            "content": None,
            "tool_calls": [
                {"id": "call_0", "type": "function", "function": {"name": name, "arguments": json.dumps(args)}}
            ],
        },
        prompt_tokens=prompt_tokens,
        completion_tokens=completion_tokens,
    )


def _trace_llm_call_tokens(trace_path: Path) -> int:
    data = json.loads(trace_path.read_text(encoding="utf-8"))
    return sum(
        (e.get("prompt_tokens") or 0) + (e.get("completion_tokens") or 0)
        for e in data["events"]
        if e["kind"] == "llm_call"
    )


def test_c_error_row_tokens_billed_matches_the_trace_llm_call_events(tmp_path, monkeypatch):
    """The gather spends real tokens and succeeds; arm C's first (and only,
    since a repair never starts) proposal then gets a Groq-400-like LLMError
    on every attempt, exhausting the parse-retry budget and raising
    AgentError -- a repair failing this way now falls back to the last parsed
    proposal instead (see tests/test_agent.py), but a FIRST proposal that
    never parses still raises, same as malformed JSON always has. The tokens
    already spent on the gather must not be reported as 0.
    """
    trips_path = _write_trips(tmp_path, [_valid_trip("t1")])
    out_path = tmp_path / "results.jsonl"

    gather_script = [
        _tool_call_completion(
            "search_places", {"city_state": "Austin, TX", "interest": "museums"},
            prompt_tokens=30, completion_tokens=5,
        ),
        _completion({"role": "assistant", "content": ""}, prompt_tokens=20, completion_tokens=5),
    ]
    schema_rejected = LLMError("groq 400: Generated JSON does not match the expected schema")
    c_script = [schema_rejected, schema_rejected, schema_rejected]
    real_build_dry_run = run_module._build_dry_run

    def patched_build_dry_run(args):
        http_client, _ = real_build_dry_run(args)
        calls = {"n": 0}

        def inner_client_factory(route, interests):
            calls["n"] += 1
            return _ScriptedClient(gather_script) if calls["n"] == 1 else _ScriptedClient(c_script)

        return http_client, inner_client_factory

    monkeypatch.setattr(run_module, "_build_dry_run", patched_build_dry_run)

    code = run_module.main([*_common_args(tmp_path, trips_path, out_path), "--arms", "C"])

    assert code == 0
    rows = _read(out_path)
    c_row = next(r for r in rows if r["arm"] == "C")
    b_row = next(r for r in rows if r["arm"] == "B")

    assert c_row["outcome"] == "error"
    assert b_row["outcome"] == "error"

    trace_path = next((tmp_path / "traces").glob("*.json"))  # only arm C writes a trace here
    expected_tokens = _trace_llm_call_tokens(trace_path)

    assert expected_tokens == (30 + 5) + (20 + 5)  # the two gather calls; C never parsed anything
    assert c_row["tokens_billed"] == expected_tokens
    assert b_row["tokens_billed"] == 0  # B never makes its own LLM calls, as on success


def test_gather_failure_error_rows_bill_the_gather_tokens_to_one_arm(tmp_path, monkeypatch):
    """When gather itself fails, nothing has produced a result yet, but a
    tool-phase call before the failure may still have spent real tokens.
    """
    trips_path = _write_trips(tmp_path, [_valid_trip("t1")])
    out_path = tmp_path / "results.jsonl"

    gather_script = [
        _tool_call_completion(
            "search_places", {"city_state": "Austin, TX", "interest": "museums"},
            prompt_tokens=30, completion_tokens=5,
        ),
        LLMError("boom"),
    ]
    real_build_dry_run = run_module._build_dry_run

    def patched_build_dry_run(args):
        http_client, _ = real_build_dry_run(args)

        def inner_client_factory(route, interests):
            return _ScriptedClient(gather_script)

        return http_client, inner_client_factory

    monkeypatch.setattr(run_module, "_build_dry_run", patched_build_dry_run)

    code = run_module.main([*_common_args(tmp_path, trips_path, out_path), "--arms", "C"])

    assert code == 0
    rows = _read(out_path)
    assert {r["outcome"] for r in rows} == {"error"}
    assert {r["arm"] for r in rows} == {"A", "B", "C"}  # error rows always cover all three

    billed = {r["arm"]: r["tokens_billed"] for r in rows}
    # Exactly one arm (C, by convention -- the one that would have billed the
    # gather on success) carries the real spend; the others are 0, not double-billed.
    assert billed["C"] == 30 + 5
    assert billed["A"] == 0
    assert billed["B"] == 0


def test_limit_counts_only_trips_actually_processed(tmp_path):
    trips_path = _write_trips(tmp_path, [_valid_trip("t1"), _valid_trip("t2"), _valid_trip("t3")])
    out_path = tmp_path / "results.jsonl"
    args = _common_args(tmp_path, trips_path, out_path)

    assert run_module.main([*args, "--limit", "1"]) == 0
    assert {r["trip_id"] for r in _read(out_path)} == {"t1"}

    # Resuming with --limit 1 again must process the NEXT trip, not re-skip-count t1.
    assert run_module.main([*args, "--limit", "1"]) == 0
    assert {r["trip_id"] for r in _read(out_path)} == {"t1", "t2"}
