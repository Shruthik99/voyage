"""The JSON record of one run: what was called, what the validator said, what was repaired.

Buffered in memory and written once, so a run that raises still leaves a trace
(plan_trip writes from a finally). DECISIONS: "Files now, SQLite when the monitor
exists".

Two rules about what goes in:

- No secrets, ever. Nothing here records a URL, a header, or a raw provider
  payload in the first place; _scrub is the second line of defence, replacing any
  live key value found in the environment before the bytes hit disk.
- Summaries, not payloads. A tool entry holds the arguments the LLM chose and a
  count-shaped summary of what came back. The point of a trace is to answer "what
  did the agent do", not to mirror Geoapify. Reasoning is recorded as a character
  count for the same reason.
"""

import json
import os
import re
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from src.models import ValidationReport
from src.tools import _redact

TRACE_DIR = Path(__file__).resolve().parent.parent / "traces"

SECRET_ENV_VARS = ("GROQ_API_KEY", "GEOAPIFY_KEY")
_MIN_SECRET_LEN = 8  # below this, a "key" is a test stub and redacting it hides real text


def _now() -> datetime:
    return datetime.now(UTC)


def scrub(value: Any) -> Any:
    """Walk a JSON-shaped value, removing API keys and key-bearing URLs from strings.

    Public because anything else that writes run output to disk needs the same
    guarantee, and a second implementation is a second thing to get wrong.
    """
    if isinstance(value, dict):
        return {k: scrub(v) for k, v in value.items()}
    if isinstance(value, list):
        return [scrub(v) for v in value]
    if not isinstance(value, str):
        return value

    text = _redact(value)
    for name in SECRET_ENV_VARS:
        secret = os.environ.get(name) or ""
        if len(secret) >= _MIN_SECRET_LEN:
            text = text.replace(secret, "REDACTED")
    return text


_scrub = scrub  # the name this module used before scrub went public


class Trace:
    """One run's events, in order. Every method appends; nothing is ever rewritten."""

    def __init__(self, run_id: str, request: dict) -> None:
        self.run_id = run_id
        self.started_at = _now()
        self.request = request
        self.events: list[dict] = []
        self.status: str = "incomplete"
        self.extra: dict = {}

    def _add(self, kind: str, **fields) -> None:
        self.events.append({"kind": kind, **fields})

    def tool_call(
        self,
        name: str,
        arguments: dict,
        *,
        summary: dict | None = None,
        error: str | None = None,
        duration_ms: float,
    ) -> None:
        self._add(
            "tool_call",
            name=name,
            arguments=arguments,
            summary=summary,
            error=error,
            duration_ms=round(duration_ms, 1),
        )

    def llm_call(
        self,
        phase: str,
        *,
        model: str,
        has_tools: bool,
        has_schema: bool,
        reasoning_chars: int = 0,
        duration_ms: float,
        prompt_tokens: int | None = None,
        completion_tokens: int | None = None,
    ) -> None:
        self._add(
            "llm_call",
            phase=phase,
            model=model,
            has_tools=has_tools,
            has_schema=has_schema,
            reasoning_chars=reasoning_chars,
            duration_ms=round(duration_ms, 1),
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
        )

    def validation(
        self,
        attempt: int,
        report: ValidationReport,
        *,
        stops: int | None = None,
        stops_proposed: int | None = None,
        day_km: list[float] | None = None,
    ) -> None:
        """One attempt's verdict.

        stops/stops_proposed/day_km are the denominators a report cannot carry:
        how many stops survived the build, how many the model actually named
        (the difference is what was dropped as untraceable or malformed), and
        each day's straight-line km. Recording them here is what lets a reader
        of the trace alone compute a per-stop rate, without the itinerary.
        """
        self._add(
            "validation",
            attempt=attempt,
            ok=report.ok,
            estimated_share=report.estimated_share,
            unknown_count=report.unknown_count,
            stops=stops,
            stops_proposed=stops_proposed,
            day_km=day_km,
            results=[r.model_dump() for r in report.results],
        )

    def repair(self, attempt: int, failures: list[dict]) -> None:
        self._add("repair", attempt=attempt, failures=failures)

    def note(self, message: str) -> None:
        self._add("note", message=message)

    def adopt(self, other: "Trace") -> None:
        """Append another trace's events, each marked shared=True.

        One gathering phase can back more than one proposal (two eval arms
        proposing from the same catalog). Each proposal still gets its own
        trace, and adopting the gather's events keeps that trace a complete,
        self-contained record of the tool results every stop traces back to.
        The flag is what stops a reader double-counting shared cost as if each
        trace had paid it.
        """
        self.events.extend({**event, "shared": True} for event in other.events)

    def finish(self, status: str, **extra) -> None:
        self.status = status
        self.extra.update(extra)

    def to_dict(self) -> dict:
        return _scrub(
            {
                "run_id": self.run_id,
                "started_at": self.started_at.isoformat(),
                "finished_at": _now().isoformat(),
                "status": self.status,
                "request": self.request,
                "events": self.events,
                **self.extra,
            }
        )

    def write(self, trace_dir: Path | None = TRACE_DIR) -> Path | None:
        """Write the trace, or do nothing if trace_dir is None. Returns the path."""
        if trace_dir is None:
            return None
        trace_dir = Path(trace_dir)
        trace_dir.mkdir(parents=True, exist_ok=True)

        stamp = self.started_at.strftime("%Y%m%d-%H%M%S")
        safe_id = re.sub(r"[^A-Za-z0-9_-]", "", self.run_id)[:8]
        path = trace_dir / f"{stamp}-{safe_id}.json"
        tmp_path = path.with_suffix(".tmp")
        tmp_path.write_text(json.dumps(self.to_dict(), indent=2, default=str), encoding="utf-8")
        tmp_path.replace(path)
        return path
