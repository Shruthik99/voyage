"""The eval-harness runner.

One trip at a time: build the TripRequest, gather once, run arm C then arm A,
derive arm B from arm C's own trace. Every row is appended to results.jsonl
immediately (flush + fsync), so a kill at any point leaves valid JSONL and a
clean resume point.

    uv run python -m evals.run --dry-run
    uv run python -m evals.run --limit 2 --reasoning-effort low
"""

import argparse
import json
import os
import tempfile
import uuid
from datetime import UTC, date, datetime
from pathlib import Path

import httpx
from pydantic import ValidationError as PydanticValidationError

from evals import build_trip_request, load_trips
from evals import fake as fake_module
from evals import metrics as metrics_module
from evals.pacing import DailyLimitReached, PacedClient, TokenBucket
from src import llm, tools
from src.agent import MAX_TOOL_STEPS, AgentError, gather, plan_from_gathered
from src.llm import LLMError
from src.parsing import LocationError
from src.trace import Trace, scrub

SCHEMA_VERSION = 1
ARMS = ("A", "B", "C")  # the canonical three -- rejected/error rows always cover all of them

_ZERO_METRICS = {
    "ok": False,
    "repairs": 0,
    "tool_calls": 0,
    "tokens_gather": 0,
    "tokens_arm": 0,
    "tokens_total": 0,
    "seconds": 0.0,
    "stops": 0,
    "stops_proposed": 0,
    "day_km": [],
    "avg_day_km": 0.0,
    "estimated_share": 0.0,
    "unknown_count": 0,
    "priced_share": 0.0,
    "cap_hit": False,
    "interests_requested": 0,
    "interests_covered": 0,
    "results": [],
}


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the three-arm eval campaign.")
    parser.add_argument("--trips", default=None, help="trip file (default: evals/trips.json)")
    parser.add_argument("--out", default="evals/results.jsonl")
    parser.add_argument("--trace-dir", default="traces/evals")
    parser.add_argument("--arms", default="A,C", help="arms to execute; B is always derived")
    parser.add_argument("--trip-id", action="append", default=None, help="repeatable")
    parser.add_argument("--limit", type=int, default=None, help="stop after N trips this session")
    parser.add_argument("--model", default=llm.DEFAULT_MODEL)
    parser.add_argument("--reasoning-effort", default="low", help="low/medium/high, or 'none'")
    parser.add_argument("--tpm", type=int, default=8000)
    parser.add_argument("--daily-token-budget", type=int, default=120_000)
    parser.add_argument("--max-tool-steps", type=int, default=MAX_TOOL_STEPS)
    parser.add_argument("--cache-dir", default=str(tools.CACHE_DIR))
    parser.add_argument("--dry-run", action="store_true", help="fake client + MockTransport")
    parser.add_argument("--force", action="store_true", help="re-run pairs already in --out")
    args = parser.parse_args(argv)

    args.arms_list = [a.strip().upper() for a in args.arms.split(",") if a.strip()]
    effort = args.reasoning_effort
    args.reasoning_effort = None if effort.lower() == "none" else effort
    if args.reasoning_effort is not None and args.reasoning_effort not in llm.REASONING_EFFORTS:
        parser.error(f"--reasoning-effort must be one of {llm.REASONING_EFFORTS} or 'none'")
    return args


# --- results.jsonl: reading state, writing rows ---


def _read_rows(path: Path) -> list[dict]:
    if not path.exists():
        return []
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line:
            rows.append(json.loads(line))
    return rows


def _done_pairs(rows: list[dict]) -> set[tuple[str, str]]:
    return {(r["trip_id"], r["arm"]) for r in rows}


def _daily_used(rows: list[dict], *, today: date) -> int:
    """Real spend so far today, from this harness's own rows only -- any other
    same-day use of the key is invisible to it (see pacing.py).
    """
    total = 0
    for row in rows:
        written_at = row.get("written_at")
        if written_at and datetime.fromisoformat(written_at).date() == today:
            total += row.get("tokens_billed", 0)
    return total


def _append_row(path: Path, row: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(row) + "\n")
        f.flush()
        os.fsync(f.fileno())


def _base_row(trip_id: str, arm: str, *, model: str, reasoning_effort: str | None) -> dict:
    return {
        "schema_version": SCHEMA_VERSION,
        "trip_id": trip_id,
        "arm": arm,
        "model": model,
        "reasoning_effort": reasoning_effort,
        "written_at": datetime.now(UTC).isoformat(),
    }


def _placeholder_row(
    trip_id, arm, *, outcome: str, model, reasoning_effort, message: str, tokens_billed: int = 0
) -> dict:
    return {
        **_base_row(trip_id, arm, model=model, reasoning_effort=reasoning_effort),
        "outcome": outcome,
        "tokens_billed": tokens_billed,
        "error": scrub(message),
        **_ZERO_METRICS,
    }


def _outcome_row(trip_id, arm, metrics_row: dict, *, model, reasoning_effort, billed: int) -> dict:
    return {
        **_base_row(trip_id, arm, model=model, reasoning_effort=reasoning_effort),
        "outcome": "ok" if metrics_row["ok"] else "unverified",
        "tokens_billed": billed,
        "error": None,
        **metrics_row,
    }


# --- one trip ---


def _process_trip(
    trip: dict,
    *,
    args: argparse.Namespace,
    bucket: TokenBucket,
    http_client: httpx.Client,
    inner_client_factory,
    results_path: Path,
    trace_dir: Path,
    done: set[tuple[str, str]],
    today: date,
) -> bool:
    """Process one trip. Returns True iff a row was actually written (what
    --limit counts against).
    """
    trip_id = trip["trip_id"]
    model, reasoning_effort = args.model, args.reasoning_effort
    wrote_any = False

    def write(arm: str, row: dict) -> None:
        nonlocal wrote_any
        _append_row(results_path, row)
        done.add((trip_id, arm))
        wrote_any = True

    try:
        request = build_trip_request(trip, today=today)
    except (LocationError, PydanticValidationError) as exc:
        message = str(exc)
        for arm in ARMS:
            if (trip_id, arm) in done and not args.force:
                continue
            write(arm, _placeholder_row(trip_id, arm, outcome="rejected", model=model, reasoning_effort=reasoning_effort, message=message))
        return wrote_any

    configured = set(args.arms_list)
    need_c = "C" in configured and (args.force or (trip_id, "C") not in done or (trip_id, "B") not in done)
    need_a = "A" in configured and (args.force or (trip_id, "A") not in done)
    if not need_c and not need_a:
        return False

    gather_trace = Trace(
        f"gather-{uuid.uuid4().hex}",
        {**request.model_dump(mode="json"), "reasoning_effort": reasoning_effort},
    )
    gather_client = PacedClient(inner_client_factory(request.route, request.interests), bucket)
    try:
        gathered = gather(
            request,
            client=gather_client,
            http_client=http_client,
            model=model,
            trace=gather_trace,
            max_tool_steps=args.max_tool_steps,
            reasoning_effort=reasoning_effort,
        )
    except (AgentError, LLMError) as exc:
        message = str(exc)
        # Nobody has produced a result yet, but the gather itself may have
        # spent real tokens before it failed -- bill them to whichever arm
        # would normally have absorbed the gather's cost (C by default, A if
        # C is not running this session), never to 0.
        gather_tokens = metrics_module.row_from_trace(gather_trace.to_dict(), arm="gather")["tokens_total"]
        billed_to = "C" if need_c else ("A" if need_a else None)
        for arm in ARMS:
            if (trip_id, arm) in done and not args.force:
                continue
            tokens_billed = gather_tokens if arm == billed_to else 0
            write(
                arm,
                _placeholder_row(
                    trip_id, arm, outcome="error", model=model, reasoning_effort=reasoning_effort,
                    message=message, tokens_billed=tokens_billed,
                ),
            )
        return wrote_any

    if need_c:
        c_trace = Trace(f"C-{uuid.uuid4().hex}", {})
        c_trace.adopt(gather_trace)
        c_client = PacedClient(inner_client_factory(request.route, request.interests), bucket)
        try:
            result = plan_from_gathered(
                request, gathered, client=c_client, model=model, trace=c_trace,
                stop_mode="ids", max_repairs=3, reasoning_effort=reasoning_effort,
            )
        except (AgentError, LLMError) as exc:
            c_trace.finish("error")
            c_trace.write(trace_dir)
            message = str(exc)
            # c_trace already carries the adopted gather events, so its total
            # is every token spent on this trip so far, partial proposal/repair
            # attempts included -- never 0 just because the run ended in error.
            c_tokens = metrics_module.row_from_trace(c_trace.to_dict(), arm="C")["tokens_total"]
            if args.force or (trip_id, "C") not in done:
                write(
                    "C",
                    _placeholder_row(
                        trip_id, "C", outcome="error", model=model, reasoning_effort=reasoning_effort,
                        message=message, tokens_billed=c_tokens,
                    ),
                )
            if args.force or (trip_id, "B") not in done:
                write("B", _placeholder_row(trip_id, "B", outcome="error", model=model, reasoning_effort=reasoning_effort, message=message))
        else:
            c_trace.finish(
                "ok" if result.report.ok else "unverified",
                repairs=result.repairs, tool_calls=result.tool_calls,
            )
            c_trace.write(trace_dir)
            c_trace_dict = c_trace.to_dict()
            if args.force or (trip_id, "C") not in done:
                c_metrics = metrics_module.row_from_trace(c_trace_dict, arm="C")
                write("C", _outcome_row(trip_id, "C", c_metrics, model=model, reasoning_effort=reasoning_effort, billed=c_metrics["tokens_total"]))
            if args.force or (trip_id, "B") not in done:
                b_metrics = metrics_module.row_from_trace(c_trace_dict, arm="B", upto_attempt=0)
                write("B", _outcome_row(trip_id, "B", b_metrics, model=model, reasoning_effort=reasoning_effort, billed=0))

    if need_a:
        a_trace = Trace(f"A-{uuid.uuid4().hex}", {})
        a_trace.adopt(gather_trace)
        a_client = PacedClient(inner_client_factory(request.route, request.interests), bucket)
        try:
            result = plan_from_gathered(
                request, gathered, client=a_client, model=model, trace=a_trace,
                stop_mode="full", max_repairs=0, reasoning_effort=reasoning_effort,
            )
        except (AgentError, LLMError) as exc:
            a_trace.finish("error")
            a_trace.write(trace_dir)
            # Same split as the success path below: A bills its own attempt,
            # plus the gather only if C did not already bill it this session.
            a_metrics_partial = metrics_module.row_from_trace(a_trace.to_dict(), arm="A")
            billed_a = a_metrics_partial["tokens_arm"] + (0 if need_c else a_metrics_partial["tokens_gather"])
            write(
                "A",
                _placeholder_row(
                    trip_id, "A", outcome="error", model=model, reasoning_effort=reasoning_effort,
                    message=str(exc), tokens_billed=billed_a,
                ),
            )
        else:
            a_trace.finish(
                "ok" if result.report.ok else "unverified",
                repairs=result.repairs, tool_calls=result.tool_calls,
            )
            a_trace.write(trace_dir)
            a_metrics = metrics_module.row_from_trace(a_trace.to_dict(), arm="A")
            # The gather is billed by arm C; if C did not also run this
            # session, nobody else paid for it, so attribute it to A instead.
            billed_a = a_metrics["tokens_arm"] + (0 if need_c else a_metrics["tokens_gather"])
            write("A", _outcome_row(trip_id, "A", a_metrics, model=model, reasoning_effort=reasoning_effort, billed=billed_a))

    return wrote_any


# --- campaign ---


def _build_dry_run(args: argparse.Namespace):
    tools.CACHE_DIR = Path(tempfile.mkdtemp(prefix="voyage-eval-dry-run-"))
    handler = fake_module.make_dry_run_handler()
    http_client = httpx.Client(transport=httpx.MockTransport(handler))

    def inner_client_factory(route: list[str], interests: list[str]):
        return fake_module.FakeLLMClient(route=route, interests=interests, max_tool_steps=args.max_tool_steps)

    return http_client, inner_client_factory


def _build_live_run(args: argparse.Namespace):
    tools.CACHE_DIR = Path(args.cache_dir)
    http_client = httpx.Client()
    real_client = llm.build_client()

    def inner_client_factory(route: list[str], interests: list[str]):
        return real_client

    return http_client, inner_client_factory


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)

    trips_path = Path(args.trips) if args.trips else None
    trips = (load_trips(trips_path) if trips_path else load_trips())["trips"]
    if args.trip_id:
        wanted = set(args.trip_id)
        trips = [t for t in trips if t["trip_id"] in wanted]

    results_path = Path(args.out)
    trace_dir = Path(args.trace_dir)
    rows = _read_rows(results_path)
    done = _done_pairs(rows)
    today = datetime.now(UTC).date()
    daily_used = _daily_used(rows, today=today)

    http_client, inner_client_factory = (
        _build_dry_run(args) if args.dry_run else _build_live_run(args)
    )
    bucket = TokenBucket(args.tpm, args.daily_token_budget, daily_used=daily_used)

    processed = 0
    try:
        for trip in trips:
            if args.limit is not None and processed >= args.limit:
                break
            did_work = _process_trip(
                trip, args=args, bucket=bucket, http_client=http_client,
                inner_client_factory=inner_client_factory, results_path=results_path,
                trace_dir=trace_dir, done=done, today=today,
            )
            if did_work:
                processed += 1
    except DailyLimitReached as exc:
        print(f"daily token budget reached: {exc}")
        print(f"completed {processed} trip(s) this session; prior rows are intact")
        return 0
    finally:
        http_client.close()

    print(f"completed {processed} trip(s) this session")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
