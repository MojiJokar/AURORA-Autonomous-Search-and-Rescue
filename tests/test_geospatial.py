import math
import pytest

from aurora.geospatial import (
    bearing_deg,
    destination_point,
    distance_m,
    pixel_to_ground,
    rotate_ground_offset,
    geolocate_target,
)


def test_destination_point_east():
    lat, lon = destination_point(49.0, 8.0, 100, 0)
    assert lat == pytest.approx(49.0, abs=0.00001)
    assert lon > 8.0


def test_destination_point_north():
    lat, lon = destination_point(49.0, 8.0, 0, 100)
    assert lat > 49.0
    assert lon == pytest.approx(8.0)


def test_distance_zero():
    assert distance_m(49, 8, 49, 8) == pytest.approx(0)


def test_distance_one_hundred_metres():
    lat, lon = destination_point(49, 8, 100, 0)
    assert distance_m(49, 8, lat, lon) == pytest.approx(100, abs=0.1)


def test_bearing_east():
    lat, lon = destination_point(49, 8, 100, 0)
    assert bearing_deg(49, 8, lat, lon) == pytest.approx(90, abs=0.1)


def test_invalid_latitude():
    with pytest.raises(ValueError):
        destination_point(95, 8, 10, 10)


def test_camera_center_projects_to_zero():
    east, north = pixel_to_ground(
        319.5, 239.5, 640, 480, 30, 70, 55
    )
    assert east == pytest.approx(0)
    assert north == pytest.approx(0)


def test_camera_right_is_east():
    east, north = pixel_to_ground(
        500, 239.5, 640, 480, 30, 70, 55
    )
    assert east > 0
    assert north == pytest.approx(0)


def test_camera_lower_pixel_is_south():
    east, north = pixel_to_ground(
        319.5, 350, 640, 480, 30, 70, 55
    )
    assert north < 0


def test_heading_rotation():
    east, north = rotate_ground_offset(10, 0, 90)
    assert east == pytest.approx(0, abs=1e-9)
    assert north == pytest.approx(-10)


def test_geolocate_target():
    target = geolocate_target(
        track_id=1,
        uav_latitude=49.0,
        uav_longitude=8.0,
        altitude_m=30,
        heading_deg=0,
        pixel_x=319.5,
        pixel_y=239.5,
        image_width=640,
        image_height=480,
        horizontal_fov_deg=70,
        vertical_fov_deg=55,
    )
    assert target.latitude == pytest.approx(49.0)
    assert target.longitude == pytest.approx(8.0)
    assert target.geolocation_valid


def test_target_serialization():
    target = geolocate_target(
        7, 49, 8, 30, 0, 319.5, 239.5,
        640, 480, 70, 55,
    )
    data = target.to_dict()
    assert data["track_id"] == 7
    assert data["geolocation_valid"] is True


def test_invalid_pixel():
    with pytest.raises(ValueError):
        pixel_to_ground(
            900, 200, 640, 480, 30, 70, 55
        )


def test_invalid_fov():
    with pytest.raises(ValueError):
        pixel_to_ground(
            300, 200, 640, 480, 30, 180, 55
        )


def test_mission_geolocation_builder_and_geojson(tmp_path):
    from aurora.geospatial import MissionGeolocationBuilder

    builder = MissionGeolocationBuilder(
        image_width=640,
        image_height=480,
        horizontal_fov_deg=70,
        vertical_fov_deg=55,
    )

    telemetry = {
        "latitude": 49.87284,
        "longitude": 8.65124,
        "relative_altitude_m": 30,
        "heading_deg": 90,
        "source": "simulated",
    }
    bbox = {
        "x_min": 300,
        "y_min": 180,
        "x_max": 340,
        "y_max": 260,
    }

    record = builder.add_observation(
        track_id=4,
        bbox=bbox,
        telemetry=telemetry,
        confidence=0.91,
        priority="HIGH",
        frame_number=12,
    )

    assert record["track_id"] == 4
    assert record["priority"] == "HIGH"
    assert record["geolocation_valid"] is True

    geojson = builder.to_geojson()
    assert geojson["type"] == "FeatureCollection"
    assert len(geojson["features"]) == 2

    target_feature = geojson["features"][0]
    uav_feature = geojson["features"][1]

    assert target_feature["geometry"]["type"] == "Point"
    assert target_feature["properties"]["track_id"] == 4
    assert uav_feature["geometry"]["type"] == "Point"
    assert uav_feature["properties"]["role"] == "UAV"

    json_path = tmp_path / "targets.json"
    geojson_path = tmp_path / "targets.geojson"
    builder.save(json_path, geojson_path)

    assert json_path.is_file()
    assert geojson_path.is_file()


def test_mission_geolocation_rejects_invalid_bbox():
    from aurora.geospatial import MissionGeolocationBuilder

    builder = MissionGeolocationBuilder()
    telemetry = {
        "latitude": 49.0,
        "longitude": 8.0,
        "relative_altitude_m": 30,
        "heading_deg": 90,
        "source": "simulated",
    }
    bbox = {
        "x_min": 300,
        "y_min": 200,
        "x_max": 250,
        "y_max": 100,
    }

    with pytest.raises(ValueError):
        builder.add_observation(1, bbox, telemetry)
