"""Estimate target coordinates from UAV position and camera projection."""

from dataclasses import dataclass

from .camera_projection import pixel_to_ground, rotate_ground_offset
from .coordinates import destination_point


@dataclass(frozen=True)
class GeolocatedTarget:
    track_id: int
    latitude: float
    longitude: float
    east_offset_m: float
    north_offset_m: float
    uncertainty_m: float
    method: str = "downward_camera_ground_projection"
    geolocation_valid: bool = True

    def to_dict(self):
        return {
            "track_id": self.track_id,
            "latitude": self.latitude,
            "longitude": self.longitude,
            "east_offset_m": self.east_offset_m,
            "north_offset_m": self.north_offset_m,
            "estimated_uncertainty_m": self.uncertainty_m,
            "geolocation_method": self.method,
            "geolocation_valid": self.geolocation_valid,
        }


def geolocate_target(
    track_id,
    uav_latitude,
    uav_longitude,
    altitude_m,
    heading_deg,
    pixel_x,
    pixel_y,
    image_width,
    image_height,
    horizontal_fov_deg,
    vertical_fov_deg,
    uncertainty_m=5.0,
):
    if not isinstance(track_id, int) or isinstance(track_id, bool) or track_id < 0:
        raise ValueError("track_id must be a non-negative integer")

    if uncertainty_m < 0:
        raise ValueError("uncertainty_m cannot be negative")

    east, north = pixel_to_ground(
        pixel_x, pixel_y,
        image_width, image_height,
        altitude_m,
        horizontal_fov_deg,
        vertical_fov_deg,
    )

    east, north = rotate_ground_offset(east, north, heading_deg)

    latitude, longitude = destination_point(
        uav_latitude, uav_longitude, east, north
    )

    return GeolocatedTarget(
        track_id=track_id,
        latitude=latitude,
        longitude=longitude,
        east_offset_m=east,
        north_offset_m=north,
        uncertainty_m=uncertainty_m,
    )
