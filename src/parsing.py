"""Parsing of "City, ST" location strings into structured stops."""


class LocationError(ValueError):
    """Raised when a location string is non-US or malformed."""


US_STATES = {
    "AL", "AK", "AZ", "AR", "CA", "CO", "CT", "DE", "FL", "GA",
    "HI", "ID", "IL", "IN", "IA", "KS", "KY", "LA", "ME", "MD",
    "MA", "MI", "MN", "MS", "MO", "MT", "NE", "NV", "NH", "NJ",
    "NM", "NY", "NC", "ND", "OH", "OK", "OR", "PA", "RI", "SC",
    "SD", "TN", "TX", "UT", "VT", "VA", "WA", "WV", "WI", "WY",
    "DC",
}


def parse_location(location: str) -> tuple[str, str]:
    """Parse a single "City, ST" string into (city, state_code).

    Raises LocationError on missing/invalid state, empty city, or extra commas.
    """
    parts = location.split(",")
    if len(parts) != 2:
        raise LocationError(f"expected exactly one comma in 'City, ST' format: {location!r}")

    city, state = parts[0].strip(), parts[1].strip().upper()

    if not city:
        raise LocationError(f"missing city: {location!r}")
    if not state:
        raise LocationError(f"missing state: {location!r}")
    if state not in US_STATES:
        raise LocationError(f"invalid US state code: {state!r}")

    return city, state


def parse_route(stops: list[str]) -> list[tuple[str, str]]:
    """Parse a list of "City, ST" strings into (city, state_code) tuples.

    Each list entry is parsed as one combined string; never split across entries.
    """
    return [parse_location(stop) for stop in stops]
