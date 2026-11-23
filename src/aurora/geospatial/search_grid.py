"""Simulation-first rectangular lawnmower search-grid planner."""

import json
import math
from pathlib import Path

from .coordinates import destination_point, validate_coordinates

EARTH_RADIUS_M = 6371008.8


def _rectangle_dimensions(southwest, northeast):
    south, west = southwest
    north, east = northeast

    if north <= south:
        raise ValueError("Northeast latitude must be north of southwest latitude.")
    if east <= west:
        raise ValueError(
            "Northeast longitude must be east of southwest longitude. "
            "Antimeridian-crossing rectangles are not supported."
        )

    middle_latitude = math.radians((south + north) / 2.0)

    height_m = EARTH_RADIUS_M * math.radians(north - south)
    width_m = (
        EARTH_RADIUS_M
        * math.cos(middle_latitude)
        * math.radians(east - west)
    )

    if width_m <= 0 or height_m <= 0:
        raise ValueError("Search area must have positive dimensions.")

    return width_m, height_m


class SearchGridPlanner:
    """Generate a serpentine route inside a rectangular GPS search area."""

    def __init__(
        self,
        southwest,
        northeast,
        lane_spacing_m=20.0,
        altitude_m=30.0,
        speed_m_s=3.0,
    ):
        validate_coordinates(*southwest)
        validate_coordinates(*northeast)

        # validate_coordinates checks validity; it does not return
        # the coordinate pair.
        self.southwest = (float(southwest[0]), float(southwest[1]))
        self.northeast = (float(northeast[0]), float(northeast[1]))

        self.width_m, self.height_m = _rectangle_dimensions(
            self.southwest, self.northeast
        )

        self.lane_spacing_m = float(lane_spacing_m)
        self.altitude_m = float(altitude_m)
        self.speed_m_s = float(speed_m_s)

        if not math.isfinite(self.lane_spacing_m) or self.lane_spacing_m <= 0:
            raise ValueError("lane_spacing_m must be positive and finite.")
        if not math.isfinite(self.altitude_m) or self.altitude_m <= 0:
            raise ValueError("altitude_m must be positive and finite.")
        if not math.isfinite(self.speed_m_s) or self.speed_m_s <= 0:
            raise ValueError("speed_m_s must be positive and finite.")

        # Include both boundaries while ensuring adjacent lanes do not
        # exceed the requested maximum spacing.
        lane_count = max(2, math.ceil(self.width_m / self.lane_spacing_m) + 1)
        self.lane_positions = [
            self.width_m * i / (lane_count - 1)
            for i in range(lane_count)
        ]

        self.waypoints = self._generate_waypoints()

    def _coordinate_at(self, east_m, north_m):
        return destination_point(
            self.southwest[0],
            self.southwest[1],
            east_m,
            north_m,
        )

    def _generate_waypoints(self):
        waypoints = []

        for lane_index, east_m in enumerate(self.lane_positions):
            if lane_index % 2 == 0:
                north_positions = (0.0, self.height_m)
            else:
                north_positions = (self.height_m, 0.0)

            for point_index, north_m in enumerate(north_positions):
                latitude, longitude = self._coordinate_at(east_m, north_m)

                waypoints.append({
                    "sequence": len(waypoints) + 1,
                    "lane": lane_index + 1,
                    "latitude": latitude,
                    "longitude": longitude,
                    "altitude_m": self.altitude_m,
                    "speed_m_s": self.speed_m_s,
                    "east_offset_m": east_m,
                    "north_offset_m": north_m,
                    "turnaround": point_index == 1 and lane_index < len(
                        self.lane_positions
                    ) - 1,
                })

        return waypoints

    def to_dict(self):
        return {
            "schema_version": 1,
            "planner": "rectangular_lawnmower",
            "execution_status": "PLANNED_NOT_EXECUTED",
            "search_area": {
                "southwest": {
                    "latitude": self.southwest[0],
                    "longitude": self.southwest[1],
                },
                "northeast": {
                    "latitude": self.northeast[0],
                    "longitude": self.northeast[1],
                },
                "width_m": self.width_m,
                "height_m": self.height_m,
                "approximate_area_m2": self.width_m * self.height_m,
            },
            "settings": {
                "lane_spacing_m": self.lane_spacing_m,
                "altitude_m": self.altitude_m,
                "speed_m_s": self.speed_m_s,
                "lane_count": len(self.lane_positions),
            },
            "waypoint_count": len(self.waypoints),
            "waypoints": self.waypoints,
            "limitations": [
                "Flat rectangular search area approximation",
                "No obstacle or terrain avoidance",
                "No airspace or geofence validation",
                "No battery or endurance validation",
                "No aircraft commands are sent",
                "Waypoints are planning estimates, not flight authorization",
            ],
        }

    def to_geojson(self):
        coordinates = [
            [point["longitude"], point["latitude"]]
            for point in self.waypoints
        ]

        boundary = [
            self._coordinate_at(0.0, 0.0),
            self._coordinate_at(self.width_m, 0.0),
            self._coordinate_at(self.width_m, self.height_m),
            self._coordinate_at(0.0, self.height_m),
            self._coordinate_at(0.0, 0.0),
        ]

        features = [
            {
                "type": "Feature",
                "geometry": {
                    "type": "Polygon",
                    "coordinates": [[
                        [longitude, latitude]
                        for latitude, longitude in boundary
                    ]],
                },
                "properties": {
                    "role": "SEARCH_AREA",
                    "label": "Planned search boundary",
                },
            },
            {
                "type": "Feature",
                "geometry": {
                    "type": "LineString",
                    "coordinates": coordinates,
                },
                "properties": {
                    "role": "SEARCH_ROUTE",
                    "label": "Planned lawnmower route",
                    "lane_count": len(self.lane_positions),
                    "waypoint_count": len(self.waypoints),
                    "execution_status": "PLANNED_NOT_EXECUTED",
                },
            },
        ]

        for point in self.waypoints:
            features.append({
                "type": "Feature",
                "geometry": {
                    "type": "Point",
                    "coordinates": [
                        point["longitude"],
                        point["latitude"],
                    ],
                },
                "properties": {
                    "role": "WAYPOINT",
                    "sequence": point["sequence"],
                    "lane": point["lane"],
                    "altitude_m": point["altitude_m"],
                    "speed_m_s": point["speed_m_s"],
                },
            })

        return {
            "type": "FeatureCollection",
            "name": "AURORA planned search grid",
            "features": features,
        }

    def save(self, json_path, geojson_path):
        json_path = Path(json_path)
        geojson_path = Path(geojson_path)

        json_path.parent.mkdir(parents=True, exist_ok=True)
        geojson_path.parent.mkdir(parents=True, exist_ok=True)

        json_path.write_text(
            json.dumps(self.to_dict(), indent=2),
            encoding="utf-8",
        )
        geojson_path.write_text(
            json.dumps(self.to_geojson(), indent=2),
            encoding="utf-8",
        )
