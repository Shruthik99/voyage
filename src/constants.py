"""Tunable limits shared by the validator and the planner.

Kept in one module so a change to a limit is a change in one place. Modules
import the names they use (``from src.constants import RADIUS_KM``), which
binds them as globals on the importing module -- so tests patch the *consumer*,
e.g. ``monkeypatch.setattr(validator_module, "RADIUS_KM", 5)``. Patching
``src.constants`` itself has no effect on an already-imported module.
"""

# V1: how far a stop may sit from its day's geocoded city point.
RADIUS_KM = 30

# V6: cap on the straight-line path walked/driven within a single day.
MAX_DAY_KM = 40

# V8: how many stops a day must have.
MIN_STOPS = 2
MAX_STOPS = 5

# Weather replan: rain_chance at or above this swaps a day's outdoor stops.
# Percent, matching ForecastDay.rain_chance (open-meteo precipitation_probability_max).
RAIN_THRESHOLD = 50.0
