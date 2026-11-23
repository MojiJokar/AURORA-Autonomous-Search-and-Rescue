"""Generate an explicitly simulated M14 mapping demonstration."""

import argparse
import json
from pathlib import Path

from aurora.geospatial import MissionGeolocationBuilder


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--telemetry",
        default="datasets/telemetry_latest.json",
    )
    parser.add_argument(
        "--json-output",
        default="datasets/geolocated_targets.json",
    )
    parser.add_argument(
        "--geojson-output",
        default="datasets/geolocated_targets.geojson",
    )
    args = parser.parse_args()

    telemetry = json.loads(Path(args.telemetry).read_text(encoding="utf-8"))

    if telemetry.get("source") != "simulated":
        raise SystemExit(
            "This demonstration expects simulated telemetry. "
            "Use a dedicated test telemetry file instead."
        )

    builder = MissionGeolocationBuilder(
        image_width=640,
        image_height=480,
        horizontal_fov_deg=70,
        vertical_fov_deg=55,
        uncertainty_m=5,
    )

    # Demonstration bounding box only; not a real person detection.
    builder.add_observation(
        track_id=1,
        bbox={
            "x_min": 380,
            "y_min": 180,
            "x_max": 420,
            "y_max": 270,
        },
        telemetry=telemetry,
        confidence=0.91,
        priority="HIGH",
        frame_number=1,
    )

    builder.save(args.json_output, args.geojson_output)
    print("Sample mission mapping files generated.")
    print(f"JSON: {args.json_output}")
    print(f"GeoJSON: {args.geojson_output}")
    print("WARNING: target is a simulated demonstration, not a real detection.")


if __name__ == "__main__":
    main()
