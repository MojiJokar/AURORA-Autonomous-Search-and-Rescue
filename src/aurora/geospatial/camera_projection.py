"""Ground projection for a calibrated downward-facing camera."""

import math


def pixel_to_ground(
    pixel_x,
    pixel_y,
    image_width,
    image_height,
    altitude_m,
    horizontal_fov_deg,
    vertical_fov_deg,
):
    """Return camera-relative east and north offsets in metres.

    Assumes a nadir-facing camera, flat ground and no terrain variation.
    Camera image right is east; image down is south before heading rotation.
    """
    values = (
        pixel_x, pixel_y, image_width, image_height, altitude_m,
        horizontal_fov_deg, vertical_fov_deg,
    )
    if not all(math.isfinite(v) for v in values):
        raise ValueError("Camera inputs must be finite")

    if image_width <= 0 or image_height <= 0 or altitude_m <= 0:
        raise ValueError("Image dimensions and altitude must be positive")

    if not 0 < horizontal_fov_deg < 180:
        raise ValueError("Invalid horizontal field of view")

    if not 0 < vertical_fov_deg < 180:
        raise ValueError("Invalid vertical field of view")

    if not 0 <= pixel_x < image_width or not 0 <= pixel_y < image_height:
        raise ValueError("Pixel is outside the image")

    center_x = (image_width - 1) / 2
    center_y = (image_height - 1) / 2

    ground_width = 2 * altitude_m * math.tan(
        math.radians(horizontal_fov_deg) / 2
    )
    ground_height = 2 * altitude_m * math.tan(
        math.radians(vertical_fov_deg) / 2
    )

    east = (pixel_x - center_x) / image_width * ground_width
    north = -(pixel_y - center_y) / image_height * ground_height

    return east, north


def rotate_ground_offset(east, north, heading_deg):
    """Rotate camera-relative offsets into north/east world coordinates."""
    if not all(math.isfinite(v) for v in (east, north, heading_deg)):
        raise ValueError("Offsets and heading must be finite")

    angle = math.radians(heading_deg)
    world_east = east * math.cos(angle) + north * math.sin(angle)
    world_north = -east * math.sin(angle) + north * math.cos(angle)

    return world_east, world_north
