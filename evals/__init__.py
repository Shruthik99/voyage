"""The eval harness: a three-arm comparison scored by src.validator.

Shared trip-loading helpers live here, not in run.py, so evals/trips.json's
own test can build a TripRequest from a trip entry without importing the
runner. run.py reuses the same functions.
"""

import json
from datetime import date, timedelta
from pathlib import Path

from src.agent import TripRequest

TRIPS_PATH = Path(__file__).resolve().parent / "trips.json"


def load_trips(path: Path = TRIPS_PATH) -> dict:
    """The parsed trips.json document: {"schema_version": ..., "trips": [...]}."""
    return json.loads(Path(path).read_text(encoding="utf-8"))


def resolve_start_date(trip: dict, *, today: date | None = None) -> date:
    """The trip's start_date: the explicit one if given, else today + start_in_days.

    A committed absolute date goes stale, so trips.json normally carries
    start_in_days; an explicit start_date is only for a reproducible re-run.
    """
    request = trip["request"]
    if "start_date" in request:
        return date.fromisoformat(request["start_date"])
    return (today or date.today()) + timedelta(days=request["start_in_days"])


def build_trip_request(trip: dict, *, today: date | None = None) -> TripRequest:
    """The TripRequest for one trips.json entry.

    Raises LocationError or pydantic ValidationError for a malformed trip --
    the one expect="reject" trip is supposed to raise here.
    """
    request = trip["request"]
    fields = {k: v for k, v in request.items() if k not in ("start_in_days", "start_date")}
    return TripRequest(**fields, start_date=resolve_start_date(trip, today=today))
