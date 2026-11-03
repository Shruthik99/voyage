"""Straight-line distance between two points on the earth.

Pure math, no dependencies: V1 and V6 both need it, and so will the day
grouping and the weather replan.
"""

from math import asin, cos, radians, sin, sqrt

# Mean earth radius (IUGG), km. Pinned because the choice moves a 30 km
# boundary by roughly 0.1%.
EARTH_RADIUS_KM = 6371.0088


def haversine_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Great-circle distance in km between two (lat, lon) pairs in degrees."""
    phi1, phi2 = radians(lat1), radians(lat2)
    d_phi = phi2 - phi1
    d_lambda = radians(lon2 - lon1)

    h = sin(d_phi / 2) ** 2 + cos(phi1) * cos(phi2) * sin(d_lambda / 2) ** 2
    # asin(sqrt(h)) is stable at both ends: h never exceeds 1 by more than a
    # rounding step, and min() keeps asin in domain for antipodal points.
    return 2 * EARTH_RADIUS_KM * asin(min(1.0, sqrt(h)))
