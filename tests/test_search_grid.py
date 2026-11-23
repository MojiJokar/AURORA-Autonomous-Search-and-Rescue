import json
import pytest

from aurora.geospatial.search_grid import SearchGridPlanner


@pytest.fixture
def planner():
    return SearchGridPlanner(
        southwest=(49.8720, 8.6500),
        northeast=(49.8730, 8.6520),
        lane_spacing_m=20,
        altitude_m=30,
        speed_m_s=3,
    )


def test_generates_ordered_waypoints(planner):
    data = planner.to_dict()

    assert data["execution_status"] == "PLANNED_NOT_EXECUTED"
    assert data["waypoint_count"] == len(planner.waypoints)
    assert data["waypoint_count"] == data["settings"]["lane_count"] * 2

    sequences = [p["sequence"] for p in planner.waypoints]
    assert sequences == list(range(1, len(sequences) + 1))


def test_route_alternates_direction(planner):
    points = planner.waypoints

    assert points[0]["north_offset_m"] < points[1]["north_offset_m"]
    assert points[2]["north_offset_m"] > points[3]["north_offset_m"]


def test_waypoints_have_requested_altitude_and_speed(planner):
    for point in planner.waypoints:
        assert point["altitude_m"] == 30
        assert point["speed_m_s"] == 3


def test_lane_spacing_does_not_exceed_requested_spacing(planner):
    positions = planner.lane_positions
    for first, second in zip(positions, positions[1:]):
        assert second - first <= 20.000001


def test_geojson_contains_boundary_route_and_waypoints(planner):
    data = planner.to_geojson()

    assert data["type"] == "FeatureCollection"
    roles = [feature["properties"]["role"] for feature in data["features"]]
    assert "SEARCH_AREA" in roles
    assert "SEARCH_ROUTE" in roles
    assert roles.count("WAYPOINT") == planner.to_dict()["waypoint_count"]


def test_save_writes_json_and_geojson(planner, tmp_path):
    json_path = tmp_path / "plan" / "grid.json"
    geojson_path = tmp_path / "plan" / "grid.geojson"

    planner.save(json_path, geojson_path)

    saved_json = json.loads(json_path.read_text())
    saved_geojson = json.loads(geojson_path.read_text())

    assert saved_json["waypoint_count"] > 0
    assert saved_geojson["type"] == "FeatureCollection"


@pytest.mark.parametrize(
    "kwargs",
    [
        {"lane_spacing_m": 0},
        {"lane_spacing_m": -5},
        {"altitude_m": 0},
        {"altitude_m": -1},
        {"speed_m_s": 0},
    ],
)
def test_rejects_invalid_settings(kwargs):
    settings = {
        "southwest": (49.8720, 8.6500),
        "northeast": (49.8730, 8.6520),
    }
    settings.update(kwargs)

    with pytest.raises(ValueError):
        SearchGridPlanner(**settings)


def test_rejects_reversed_rectangle():
    with pytest.raises(ValueError):
        SearchGridPlanner(
            southwest=(49.8730, 8.6520),
            northeast=(49.8720, 8.6500),
        )
