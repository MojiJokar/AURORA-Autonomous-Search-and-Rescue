"""Generate an AURORA rectangular search-grid plan."""

import argparse

from aurora.geospatial.search_grid import SearchGridPlanner


def main():
    parser = argparse.ArgumentParser(
        description="Generate a simulation-only lawnmower search route."
    )
    parser.add_argument("--south-lat", type=float, required=True)
    parser.add_argument("--west-lon", type=float, required=True)
    parser.add_argument("--north-lat", type=float, required=True)
    parser.add_argument("--east-lon", type=float, required=True)
    parser.add_argument("--spacing", type=float, default=20.0)
    parser.add_argument("--altitude", type=float, default=30.0)
    parser.add_argument("--speed", type=float, default=3.0)
    parser.add_argument(
        "--json-output",
        default="datasets/search_grid.json",
    )
    parser.add_argument(
        "--geojson-output",
        default="datasets/search_grid.geojson",
    )
    args = parser.parse_args()

    planner = SearchGridPlanner(
        southwest=(args.south_lat, args.west_lon),
        northeast=(args.north_lat, args.east_lon),
        lane_spacing_m=args.spacing,
        altitude_m=args.altitude,
        speed_m_s=args.speed,
    )

    planner.save(args.json_output, args.geojson_output)

    result = planner.to_dict()
    print("AURORA search grid generated.")
    print("Execution status: PLANNED_NOT_EXECUTED")
    print(f"Search area: {result['search_area']['approximate_area_m2']:.1f} m²")
    print(f"Width: {result['search_area']['width_m']:.1f} m")
    print(f"Height: {result['search_area']['height_m']:.1f} m")
    print(f"Lanes: {result['settings']['lane_count']}")
    print(f"Waypoints: {result['waypoint_count']}")
    print(f"JSON: {args.json_output}")
    print(f"GeoJSON: {args.geojson_output}")


if __name__ == "__main__":
    main()
