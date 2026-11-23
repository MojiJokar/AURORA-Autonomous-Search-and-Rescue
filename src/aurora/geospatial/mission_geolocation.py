"""Build geolocated target records and GeoJSON mission exports."""

import json
from pathlib import Path

from .target_geolocation import geolocate_target


class MissionGeolocationBuilder:
    def __init__(
        self,
        image_width=768,
        image_height=432,
        horizontal_fov_deg=70.0,
        vertical_fov_deg=55.0,
        uncertainty_m=5.0,
    ):
        self.image_width = image_width
        self.image_height = image_height
        self.horizontal_fov_deg = horizontal_fov_deg
        self.vertical_fov_deg = vertical_fov_deg
        self.uncertainty_m = uncertainty_m
        self.targets = {}

    def add_observation(
        self,
        track_id,
        bbox,
        telemetry,
        confidence=None,
        priority="UNKNOWN",
        frame_number=None,
    ):
        """Add one tracked-person observation.

        bbox may be an object with x_min/y_min/x_max/y_max attributes
        or a dictionary containing those keys.
        telemetry must provide latitude, longitude, relative_altitude_m
        (or altitude_m), and heading_deg.
        """
        def value(obj, key):
            if isinstance(obj, dict):
                return obj[key]
            return getattr(obj, key)

        if not isinstance(track_id, int) or isinstance(track_id, bool) or track_id < 0:
            raise ValueError("track_id must be a non-negative integer")

        latitude = float(value(telemetry, "latitude"))
        longitude = float(value(telemetry, "longitude"))
        altitude = value(telemetry, "relative_altitude_m")
        if altitude is None:
            altitude = value(telemetry, "altitude_m")
        heading = float(value(telemetry, "heading_deg"))

        x_min = float(value(bbox, "x_min"))
        y_min = float(value(bbox, "y_min"))
        x_max = float(value(bbox, "x_max"))
        y_max = float(value(bbox, "y_max"))

        if x_max <= x_min or y_max <= y_min:
            raise ValueError("Bounding box must have positive width and height")

        # Use the bottom-centre of the person bounding box as the
        # approximate ground-contact point.
        pixel_x = (x_min + x_max) / 2.0
        pixel_y = y_max

        target = geolocate_target(
            track_id=track_id,
            uav_latitude=latitude,
            uav_longitude=longitude,
            altitude_m=float(altitude),
            heading_deg=heading,
            pixel_x=pixel_x,
            pixel_y=pixel_y,
            image_width=self.image_width,
            image_height=self.image_height,
            horizontal_fov_deg=self.horizontal_fov_deg,
            vertical_fov_deg=self.vertical_fov_deg,
            uncertainty_m=self.uncertainty_m,
        )

        record = target.to_dict()
        record.update({
            "confidence": confidence,
            "priority": str(priority).upper(),
            "frame_number": frame_number,
            "telemetry_source": (
                telemetry.get("source", "unknown")
                if isinstance(telemetry, dict)
                else getattr(telemetry, "source", "unknown")
            ),
            "uav_latitude": latitude,
            "uav_longitude": longitude,
            "uav_altitude_m": float(altitude),
            "uav_heading_deg": heading,
            "pixel_x": pixel_x,
            "pixel_y": pixel_y,
        })

        # Keep the most recent observation for each track.
        self.targets[track_id] = record
        return record

    def to_dict(self):
        return {
            "schema_version": 1,
            "method": "downward_camera_ground_projection",
            "assumptions": [
                "Downward-facing camera",
                "Flat ground plane",
                "Known camera field of view",
                "Synchronized UAV telemetry",
                "Bounding-box bottom-centre approximates ground contact",
            ],
            "targets": [
                self.targets[key] for key in sorted(self.targets)
            ],
        }

    def to_geojson(self):
        features = []
        for track_id in sorted(self.targets):
            target = self.targets[track_id]
            features.append({
                "type": "Feature",
                "geometry": {
                    "type": "Point",
                    "coordinates": [
                        target["longitude"],
                        target["latitude"],
                    ],
                },
                "properties": {
                    key: value
                    for key, value in target.items()
                    if key not in ("latitude", "longitude")
                },
            })

        if self.targets:
            latest = self.targets[sorted(self.targets)[-1]]
            features.append({
                "type": "Feature",
                "geometry": {
                    "type": "Point",
                    "coordinates": [
                        latest["uav_longitude"],
                        latest["uav_latitude"],
                    ],
                },
                "properties": {
                    "role": "UAV",
                    "label": "UAV position",
                    "telemetry_source": latest["telemetry_source"],
                    "altitude_m": latest["uav_altitude_m"],
                },
            })

        return {
            "type": "FeatureCollection",
            "name": "AURORA mission map",
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
