"""Lightweight geographic coordinate calculations."""

import math

EARTH_RADIUS_M = 6371008.8


def validate_coordinates(latitude, longitude):
    if not math.isfinite(latitude) or not math.isfinite(longitude):
        raise ValueError("Coordinates must be finite")
    if not -90 <= latitude <= 90:
        raise ValueError("Invalid latitude")
    if not -180 <= longitude <= 180:
        raise ValueError("Invalid longitude")


def destination_point(latitude, longitude, east_m, north_m):
    """Approximate destination from local east/north offsets."""
    validate_coordinates(latitude, longitude)

    if not all(math.isfinite(v) for v in (east_m, north_m)):
        raise ValueError("Offsets must be finite")

    lat_rad = math.radians(latitude)
    new_lat = latitude + math.degrees(north_m / EARTH_RADIUS_M)

    cosine = math.cos(lat_rad)
    if abs(cosine) < 1e-12:
        raise ValueError("Longitude offset is unstable near the poles")

    new_lon = longitude + math.degrees(
        east_m / (EARTH_RADIUS_M * cosine)
    )

    if not -90 <= new_lat <= 90:
        raise ValueError("Destination latitude is outside valid range")

    new_lon = (new_lon + 180) % 360 - 180
    return new_lat, new_lon


def distance_m(lat1, lon1, lat2, lon2):
    validate_coordinates(lat1, lon1)
    validate_coordinates(lat2, lon2)

    p1 = math.radians(lat1)
    p2 = math.radians(lat2)
    dp = p2 - p1
    dl = math.radians(lon2 - lon1)

    a = (
        math.sin(dp / 2) ** 2
        + math.cos(p1) * math.cos(p2) * math.sin(dl / 2) ** 2
    )
    return 2 * EARTH_RADIUS_M * math.asin(min(1.0, math.sqrt(a)))


def bearing_deg(lat1, lon1, lat2, lon2):
    validate_coordinates(lat1, lon1)
    validate_coordinates(lat2, lon2)

    p1 = math.radians(lat1)
    p2 = math.radians(lat2)
    dl = math.radians(lon2 - lon1)

    y = math.sin(dl) * math.cos(p2)
    x = (
        math.cos(p1) * math.sin(p2)
        - math.sin(p1) * math.cos(p2) * math.cos(dl)
    )

    return (math.degrees(math.atan2(y, x)) + 360) % 360
