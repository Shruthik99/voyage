"""Rules V1-V8 from docs/SPEC.md, as plain Python.

The LLM proposes; this decides. Pure functions: no network, no LLM, no disk,
and no mutation of the itinerary passed in. validate() never raises -- a
degraded input (a city that failed to geocode, a malformed city string) is
reported as a failing rule, because a stop that cannot be verified has not
passed.

Each rule emits exactly one summary result (summary=True) so a trace can assert
all eight ran, plus one detail result per failing or warning item. A clean trip
yields 8 results; the detail list is what goes back to the LLM for repair.

Constants are imported by name, so tests patch them on *this* module:
``monkeypatch.setattr(validator_module, "RADIUS_KM", 5)``.
"""

from itertools import pairwise

from src.constants import MAX_DAY_KM, MAX_STOPS, MIN_STOPS, RADIUS_KM
from src.geo import haversine_km
from src.models import (
    Itinerary,
    RuleResult,
    RuleStatus,
    Stop,
    ValidationContext,
    ValidationReport,
)
from src.parsing import LocationError, parse_location


def _status_of(details: list[RuleResult]) -> RuleStatus:
    if any(d.status == "fail" for d in details):
        return "fail"
    if any(d.status == "warn" for d in details):
        return "warn"
    return "pass"


def _summary(rule: str, code: str, details: list[RuleResult], ok_message: str) -> RuleResult:
    """Build the one always-present result for a rule."""
    status = _status_of(details)
    if status == "pass":
        message = ok_message
    else:
        message = f"{len(details)} problem(s) found; see the detail results for {rule}"
    return RuleResult(rule=rule, code=code, status=status, message=message, summary=True)


def _expected_state(city: str) -> str | None:
    """The two-letter state a day's stops should be in, or None if unparseable."""
    try:
        return parse_location(city)[1]
    except LocationError:
        return None


# --- V1 Location: right state, and within RADIUS_KM of the city point ---


def _check_v1(itinerary: Itinerary, context: ValidationContext) -> list[RuleResult]:
    details: list[RuleResult] = []

    for di, day in enumerate(itinerary.days):
        expected = _expected_state(day.city)
        point = context.city_points.get(day.city)

        if expected is None:
            details.append(
                RuleResult(
                    rule="V1",
                    code="bad_city",
                    status="fail",
                    message=f"day {di + 1}: {day.city!r} is not a valid 'City, ST' string",
                    day=di,
                )
            )
        if point is None:
            details.append(
                RuleResult(
                    rule="V1",
                    code="no_city_point",
                    status="fail",
                    message=(
                        f"day {di + 1}: no geocoded point for {day.city!r}; "
                        "its stops cannot be verified"
                    ),
                    day=di,
                )
            )

        for si, stop in enumerate(day.stops):
            where = f"day {di + 1} stop {si + 1} ({stop.name!r})"

            if stop.state is None:
                # SPEC: a None state falls back to the distance check and warns.
                details.append(
                    RuleResult(
                        rule="V1",
                        code="state_unverified",
                        status="warn",
                        message=f"{where}: source reported no state; using the distance check",
                        day=di,
                        stop=si,
                        source_id=stop.source_id or None,
                    )
                )
            elif expected is not None and stop.state != expected:
                details.append(
                    RuleResult(
                        rule="V1",
                        code="state_mismatch",
                        status="fail",
                        message=f"{where}: state {stop.state} does not match {day.city}",
                        day=di,
                        stop=si,
                        source_id=stop.source_id or None,
                    )
                )

            if point is not None:
                km = haversine_km(stop.lat, stop.lon, point.lat, point.lon)
                if km > RADIUS_KM:
                    details.append(
                        RuleResult(
                            rule="V1",
                            code="out_of_radius",
                            status="fail",
                            message=(
                                f"{where}: {km:.1f} km from {day.city}, "
                                f"over the {RADIUS_KM} km limit"
                            ),
                            day=di,
                            stop=si,
                            source_id=stop.source_id or None,
                        )
                    )

    ok = f"every stop is in its day's state and within {RADIUS_KM} km of the city"
    return [_summary("V1", "located", details, ok), *details]


# --- V2 Budget ---


def _group_cents(stop: Stop, travelers: int) -> tuple[int, int] | None:
    """A stop's (low, high) cost to the whole group, in cents. None if unknown.

    Cents because V2's verdict turns on exact comparisons at the budget, and a
    sum of floats drifts enough to move S_high == budget into "uncertain".
    """
    if stop.cost_confidence == "unknown":
        return None
    multiplier = travelers if stop.cost_basis == "per_person" else 1
    low = round(stop.cost_low_usd * 100) * multiplier
    high = round(stop.cost_high_usd * 100) * multiplier
    return low, high


def _cost_totals(itinerary: Itinerary, context: ValidationContext) -> dict:
    low_cents = high_cents = estimate_high_cents = 0
    priced_count = 0
    unknown: list[tuple[int, int, Stop]] = []

    for di, day in enumerate(itinerary.days):
        for si, stop in enumerate(day.stops):
            group = _group_cents(stop, context.travelers)
            if group is None:
                unknown.append((di, si, stop))
                continue
            priced_count += 1
            low, high = group
            low_cents += low
            high_cents += high
            if stop.cost_confidence == "estimate":
                estimate_high_cents += high

    # Dollar-weighted, over priced stops only; unknown_count carries the rest.
    share = estimate_high_cents / high_cents if high_cents > 0 else 0.0
    return {
        "low_cents": low_cents,
        "high_cents": high_cents,
        "unknown": unknown,
        "priced_count": priced_count,
        "estimated_share": share,
        "unknown_count": len(unknown),
    }


def _check_v2(context: ValidationContext, totals: dict) -> list[RuleResult]:
    budget_cents = round(context.budget_usd * 100)
    low, high = totals["low_cents"], totals["high_cents"]
    share = totals["estimated_share"]

    if totals["priced_count"] == 0:
        # First, so it can never be masked: with nothing priced, S_low and S_high
        # are both 0, which would otherwise read as a comfortable "within budget".
        # Keyed on priced_count rather than unknown_count so an itinerary with no
        # stops at all lands here too, instead of passing on a $0.00 total.
        code, status = "unpriced", "warn"
        message = "budget not checked: no stop has a known cost"
    elif low > budget_cents:
        code, status = "over", "fail"
        message = (
            f"over budget: ${low / 100:,.2f} at the low end exceeds ${context.budget_usd:,.2f}"
        )
    elif high <= budget_cents:
        estimated = share > 0
        code, status = ("within_estimated" if estimated else "within"), "pass"
        label = "within estimated budget" if estimated else "within budget"
        message = f"{label}: ${high / 100:,.2f} at the high end fits ${context.budget_usd:,.2f}"
    else:
        code, status = "uncertain", "warn"
        message = (
            f"uncertain: ${low / 100:,.2f}-${high / 100:,.2f} straddles "
            f"${context.budget_usd:,.2f}"
        )

    results = [RuleResult(rule="V2", code=code, status=status, message=message, summary=True)]

    # Additive, not exclusive: one unpriced stop must not mask a blown budget.
    unknown = totals["unknown"]
    if unknown:
        named = ", ".join(f"day {di + 1} stop {si + 1} ({s.name!r})" for di, si, s in unknown)
        results.append(
            RuleResult(
                rule="V2",
                code="incomplete",
                status="warn",
                message=f"{len(unknown)} stop(s) have no known cost: {named}",
            )
        )
        results.extend(
            RuleResult(
                rule="V2",
                code="incomplete",
                status="warn",
                message=f"day {di + 1} stop {si + 1} ({s.name!r}): cost unknown",
                day=di,
                stop=si,
                source_id=s.source_id or None,
            )
            for di, si, s in unknown
        )
    return results


# --- V3 Interests ---


def _check_v3(itinerary: Itinerary, context: ValidationContext) -> list[RuleResult]:
    covered = {stop.interest for day in itinerary.days for stop in day.stops}
    details = [
        RuleResult(
            rule="V3",
            code="missing_interest",
            status="fail",
            message=f"no stop anywhere in the trip covers the requested interest {interest!r}",
        )
        for interest in context.interests
        if interest not in covered
    ]
    ok = "every requested interest has at least one stop"
    return [_summary("V3", "covered", details, ok), *details]


# --- V4 No placeholders ---


def _check_v4(itinerary: Itinerary) -> list[RuleResult]:
    details: list[RuleResult] = []
    for di, day in enumerate(itinerary.days):
        for si, stop in enumerate(day.stops):
            empty = [
                field for field in ("name", "source", "source_id") if not getattr(stop, field).strip()
            ]
            if empty:
                details.append(
                    RuleResult(
                        rule="V4",
                        code="placeholder",
                        status="fail",
                        message=f"day {di + 1} stop {si + 1}: empty {', '.join(empty)}",
                        day=di,
                        stop=si,
                        source_id=stop.source_id or None,
                    )
                )
    ok = "every stop has a name, a source and a source_id"
    return [_summary("V4", "complete", details, ok), *details]


# --- V5 Traceability ---


def _check_v5(itinerary: Itinerary, context: ValidationContext) -> list[RuleResult]:
    details: list[RuleResult] = []
    for di, day in enumerate(itinerary.days):
        for si, stop in enumerate(day.stops):
            if (stop.source, stop.source_id) not in context.tool_results:
                details.append(
                    RuleResult(
                        rule="V5",
                        code="untraceable",
                        status="fail",
                        message=(
                            f"day {di + 1} stop {si + 1} ({stop.name!r}): "
                            f"({stop.source}, {stop.source_id}) is not in any tool result "
                            "from this run"
                        ),
                        day=di,
                        stop=si,
                        source_id=stop.source_id or None,
                    )
                )
    ok = "every stop traces to a tool result from this run"
    return [_summary("V5", "traceable", details, ok), *details]


# --- V6 Daily distance ---


def day_km(stops: list[Stop]) -> float:
    """Straight-line km along a day's stops, in visit order."""
    return sum(haversine_km(a.lat, a.lon, b.lat, b.lon) for a, b in pairwise(stops))


def _check_v6(itinerary: Itinerary) -> list[RuleResult]:
    details: list[RuleResult] = []
    for di, day in enumerate(itinerary.days):
        km = day_km(day.stops)
        if km > MAX_DAY_KM:
            details.append(
                RuleResult(
                    rule="V6",
                    code="day_too_far",
                    status="fail",
                    message=(
                        f"day {di + 1}: {km:.1f} km between stops, "
                        f"over the {MAX_DAY_KM} km limit"
                    ),
                    day=di,
                )
            )
    ok = f"every day stays within {MAX_DAY_KM} km"
    return [_summary("V6", "compact", details, ok), *details]


# --- V7 No duplicates ---


def _check_v7(itinerary: Itinerary) -> list[RuleResult]:
    details: list[RuleResult] = []
    # SPEC keys V7 on source_id alone; V5 is the rule that uses the pair.
    first_seen: dict[str, tuple[int, int, str]] = {}

    for di, day in enumerate(itinerary.days):
        for si, stop in enumerate(day.stops):
            seen = first_seen.get(stop.source_id)
            if seen is None:
                first_seen[stop.source_id] = (di, si, stop.source)
                continue
            fdi, fsi, fsource = seen
            clash = f" (source {stop.source} vs {fsource})" if stop.source != fsource else ""
            # Flag the later occurrence: naming the first would have the repair
            # step delete the wrong stop.
            details.append(
                RuleResult(
                    rule="V7",
                    code="duplicate",
                    status="fail",
                    message=(
                        f"day {di + 1} stop {si + 1} ({stop.name!r}) repeats source_id "
                        f"{stop.source_id}, already used on day {fdi + 1} stop {fsi + 1}{clash}"
                    ),
                    day=di,
                    stop=si,
                    source_id=stop.source_id or None,
                )
            )
    ok = "no source_id is used twice"
    return [_summary("V7", "unique", details, ok), *details]


# --- V8 Day size ---


def _check_v8(itinerary: Itinerary) -> list[RuleResult]:
    details: list[RuleResult] = []

    if not itinerary.days:
        # Without this, an empty itinerary passes every other rule vacuously.
        details.append(
            RuleResult(
                rule="V8",
                code="no_days",
                status="fail",
                message="the itinerary has no days",
            )
        )

    for di, day in enumerate(itinerary.days):
        count = len(day.stops)
        if count < MIN_STOPS:
            details.append(
                RuleResult(
                    rule="V8",
                    code="too_few_stops",
                    status="fail",
                    message=f"day {di + 1} has {count} stop(s), fewer than {MIN_STOPS}",
                    day=di,
                )
            )
        elif count > MAX_STOPS:
            details.append(
                RuleResult(
                    rule="V8",
                    code="too_many_stops",
                    status="fail",
                    message=f"day {di + 1} has {count} stop(s), more than {MAX_STOPS}",
                    day=di,
                )
            )
    ok = f"every day has between {MIN_STOPS} and {MAX_STOPS} stops"
    return [_summary("V8", "sized", details, ok), *details]


def _check_cost_summary(itinerary: Itinerary, totals: dict) -> list[RuleResult]:
    """The LLM never supplies prices, so a stale cost summary is worth saying."""
    matches = (
        abs(itinerary.estimated_share - totals["estimated_share"]) <= 1e-9
        and itinerary.unknown_count == totals["unknown_count"]
    )
    if matches:
        return []
    return [
        RuleResult(
            rule="V2",
            code="cost_summary_mismatch",
            status="warn",
            message=(
                f"itinerary reports estimated_share={itinerary.estimated_share:.4f} / "
                f"unknown_count={itinerary.unknown_count}; recomputed "
                f"{totals['estimated_share']:.4f} / {totals['unknown_count']}"
            ),
        )
    ]


def validate(itinerary: Itinerary, context: ValidationContext) -> ValidationReport:
    """Check an itinerary against rules V1-V8. Never raises; never mutates."""
    totals = _cost_totals(itinerary, context)

    results: list[RuleResult] = [
        *_check_v1(itinerary, context),
        *_check_v2(context, totals),
        *_check_v3(itinerary, context),
        *_check_v4(itinerary),
        *_check_v5(itinerary, context),
        *_check_v6(itinerary),
        *_check_v7(itinerary),
        *_check_v8(itinerary),
        *_check_cost_summary(itinerary, totals),
    ]

    return ValidationReport(
        results=results,
        warnings=[r.message for r in results if r.status == "warn"],
        estimated_share=totals["estimated_share"],
        unknown_count=totals["unknown_count"],
    )
