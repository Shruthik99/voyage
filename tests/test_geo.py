import pytest

from src.geo import haversine_km

MIAMI = (25.7617, -80.1918)
NYC = (40.7128, -74.0060)


def test_haversine_same_point_is_zero():
    assert haversine_km(*MIAMI, *MIAMI) == 0.0


def test_haversine_known_city_pair_matches_reference():
    # Miami to New York is ~1757 km great-circle.
    assert haversine_km(*MIAMI, *NYC) == pytest.approx(1757.0, abs=5.0)


def test_haversine_is_symmetric():
    assert haversine_km(*MIAMI, *NYC) == haversine_km(*NYC, *MIAMI)


def test_haversine_antipodal_does_not_raise():
    # The acos form leaves its domain here; the asin form must not.
    assert haversine_km(0.0, 0.0, 0.0, 180.0) == pytest.approx(20015.1, abs=1.0)


def test_haversine_near_identical_points_does_not_raise():
    assert haversine_km(25.7617, -80.1918, 25.76170000001, -80.19180000001) < 0.001
