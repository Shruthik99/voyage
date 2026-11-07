"""End-to-end demo: build a TripRequest, plan it, print the result, save outputs.

Calls live APIs by design (geocoding, places, weather, the LLM) -- this is a
script, not a test; see tests/test_demo.py for its fixture-only tests.

    uv run python -m src.demo --route "Denver, CO" "Boulder, CO" --days 2
"""

import argparse
from datetime import date, timedelta
from pathlib import Path

from src.agent import AgentError, AgentResult, TripRequest, plan_trip
from src.llm import LLMError
from src.mapview import build_map, save_map


def render_plan(result: AgentResult) -> str:
    """Pure text rendering of a planned trip, so tests can call it without a live run."""
    lines = [
        f"verified: {result.itinerary.verified}",
        f"repairs: {result.repairs}",
        f"tool calls: {result.tool_calls}",
        f"trace: {result.trace_path}",
        "",
    ]

    for day_index, day in enumerate(result.itinerary.days):
        lines.append(f"Day {day_index + 1} ({day.date.isoformat()}): {day.city}")
        if day.forecast is not None:
            lines.append(
                f"  forecast: {day.forecast.high_f:.0f}F high, "
                f"{day.forecast.rain_chance:.0f}% rain chance"
            )
        elif day.warnings:
            lines.append(f"  forecast: {day.warnings[0]}")
        else:
            lines.append("  forecast: weather unavailable")
        for stop_index, stop in enumerate(day.stops):
            lines.append(f"  {stop_index + 1}. {stop.name} ({stop.interest}) - {stop.address}")
        lines.append("")

    failing = [r for r in result.report.results if r.status == "fail"]
    lines.append("failing checks:" if failing else "failing checks: none")
    for r in failing:
        lines.append(f"  [{r.rule}] {r.code}: {r.message}")

    if result.report.warnings:
        lines.append("warnings:")
        for w in result.report.warnings:
            lines.append(f"  - {w}")

    return "\n".join(lines)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plan a demo trip and render a map.")
    parser.add_argument("--route", nargs="+", default=["Denver, CO"], help='one or more "City, ST"')
    parser.add_argument("--days", type=int, default=2, help="days per stop")
    parser.add_argument("--travelers", type=int, default=2)
    parser.add_argument("--budget", type=float, default=400.0)
    parser.add_argument("--meal", type=float, default=40.0, help="meal allowance (USD)")
    parser.add_argument("--interests", nargs="+", default=["museums", "hiking"])
    parser.add_argument("--start-in-days", type=int, default=3)
    parser.add_argument("--reasoning-effort", default="low")
    parser.add_argument("--out-dir", default="output")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    reasoning_effort = None if args.reasoning_effort.lower() == "none" else args.reasoning_effort

    request = TripRequest(
        route=args.route,
        days_per_stop=args.days,
        travelers=args.travelers,
        budget_usd=args.budget,
        meal_allowance_usd=args.meal,
        interests=args.interests,
        start_date=date.today() + timedelta(days=args.start_in_days),
    )

    try:
        result = plan_trip(request, reasoning_effort=reasoning_effort)
    except (AgentError, LLMError) as exc:
        print(str(exc))
        return 1

    print(render_plan(result))

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    itinerary_path = out_dir / "itinerary.json"
    itinerary_path.write_text(result.itinerary.model_dump_json(indent=2), encoding="utf-8")

    map_path = out_dir / "trip_map.html"
    save_map(build_map(result.itinerary, city_points=result.context.city_points), map_path)

    print(itinerary_path)
    print(map_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
