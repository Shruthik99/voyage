"""Render an Itinerary as an interactive Folium map.

Every marker sits at the real lat/lon a Stop carries from its tool result --
never a city-centre placeholder, never a jittered coordinate. Builds from
plain data (Itinerary, optional GeoPoint city_points); no network calls.
"""

import argparse
import html
from pathlib import Path

import folium

from src.models import Day, ForecastDay, GeoPoint, Itinerary, Stop

#: Cycled by day index so each day's markers and polyline share one color.
DAY_COLORS = ["red", "blue", "green", "purple", "orange", "darkred", "lightblue"]

#: Geographic center of the contiguous US, used only when there is nothing to fit.
US_CENTER = (39.8283, -98.5795)
US_DEFAULT_ZOOM = 4


def _day_color(day_index: int) -> str:
    return DAY_COLORS[day_index % len(DAY_COLORS)]


def _forecast_text(forecast: ForecastDay | None) -> str:
    if forecast is None:
        return "weather unavailable"
    return f"{forecast.high_f:.0f}F high, {forecast.rain_chance:.0f}% rain chance"


def _cost_text(stop: Stop) -> str:
    if stop.cost_confidence == "unknown":
        return "cost: unknown"
    return f"cost: ${stop.cost_low_usd:.0f}-${stop.cost_high_usd:.0f} ({stop.cost_basis})"


def _popup_html(stop: Stop, city: str, forecast: ForecastDay | None) -> str:
    rows = [
        ("name", stop.name),
        ("city", city),
        ("interest", stop.interest),
        ("address", stop.address),
        ("state", stop.state or ""),
        ("source", stop.source),
        ("source_id", stop.source_id),
    ]
    lines = [f"<b>{html.escape(label)}:</b> {html.escape(str(value))}" for label, value in rows]
    lines.append(f"<b>weather:</b> {html.escape(_forecast_text(forecast))}")
    lines.append(f"<b>{html.escape(_cost_text(stop))}</b>")
    return "<br>".join(lines)


def _add_day(m: folium.Map, day: Day, day_index: int) -> None:
    color = _day_color(day_index)
    group = folium.FeatureGroup(name=f"Day {day_index + 1}")

    coords: list[tuple[float, float]] = []
    for stop_index, stop in enumerate(day.stops):
        tooltip = html.escape(f"Day {day_index + 1}, stop {stop_index + 1}: {stop.name}")
        popup = folium.Popup(_popup_html(stop, day.city, day.forecast), max_width=300)
        folium.Marker(
            location=[stop.lat, stop.lon],
            tooltip=tooltip,
            popup=popup,
            icon=folium.Icon(color=color, icon="star"),
        ).add_to(group)
        coords.append((stop.lat, stop.lon))

    if len(coords) >= 2:
        folium.PolyLine(coords, color=color).add_to(group)

    group.add_to(m)


def _add_route(m: folium.Map, city_points: dict[str, GeoPoint]) -> None:
    coords = [(p.lat, p.lon) for p in city_points.values()]
    if len(coords) >= 2:
        folium.PolyLine(coords, color="darkblue", weight=3).add_to(m)
    for city_name, point in city_points.items():
        folium.Marker(
            location=[point.lat, point.lon],
            tooltip=html.escape(city_name),
            icon=folium.Icon(color="darkblue", icon="plane", prefix="fa"),
        ).add_to(m)


def _legend_html(itinerary: Itinerary) -> str:
    rows = []
    for day_index, day in enumerate(itinerary.days):
        color = _day_color(day_index)
        swatch = (
            f'<span style="display:inline-block;width:10px;height:10px;'
            f'background:{color};margin-right:4px;"></span>'
        )
        if not day.stops:
            label = "no stops"
        else:
            status = "verified" if itinerary.verified else "NOT verified"
            label = f"{status}, {len(day.warnings)} warning(s)"
        rows.append(f"<div>{swatch}Day {day_index + 1}: {html.escape(label)}</div>")

    body = "".join(rows) if rows else "<div>no days</div>"
    return (
        '<div style="position: fixed; bottom: 20px; left: 20px; z-index: 9999; '
        "background: white; padding: 10px; border: 2px solid grey; border-radius: 5px; "
        'font-size: 12px; max-height: 40vh; overflow-y: auto;">'
        f"<b>Legend</b>{body}</div>"
    )


def build_map(itinerary: Itinerary, city_points: dict[str, GeoPoint] | None = None) -> folium.Map:
    """Build a Folium map from an itinerary. Never invents or jitters a coordinate."""
    city_points = city_points or {}
    stop_coords = [(stop.lat, stop.lon) for day in itinerary.days for stop in day.stops]
    city_coords = [(p.lat, p.lon) for p in city_points.values()]
    bounds = stop_coords or city_coords

    if bounds:
        m = folium.Map(location=bounds[0], tiles="OpenStreetMap")
    else:
        m = folium.Map(location=list(US_CENTER), zoom_start=US_DEFAULT_ZOOM, tiles="OpenStreetMap")

    for day_index, day in enumerate(itinerary.days):
        _add_day(m, day, day_index)

    if city_points:
        _add_route(m, city_points)

    folium.LayerControl().add_to(m)

    if bounds:
        m.fit_bounds(bounds)

    m.get_root().html.add_child(folium.Element(_legend_html(itinerary)))
    return m


def save_map(m: folium.Map, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    m.save(str(path))


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Build a Folium map from a saved Itinerary JSON.")
    parser.add_argument("itinerary_json", help="path to an Itinerary JSON file")
    parser.add_argument("out_html", help="path to write the map HTML to")
    args = parser.parse_args(argv)

    data = Path(args.itinerary_json).read_text(encoding="utf-8")
    itinerary = Itinerary.model_validate_json(data)
    m = build_map(itinerary)
    save_map(m, args.out_html)
    print(args.out_html)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
