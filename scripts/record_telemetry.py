"""Record simulated or MAVLink telemetry to JSON files."""

import argparse
import json
import time
from pathlib import Path

from aurora.telemetry import SimulatedTelemetrySource


def main():
    parser = argparse.ArgumentParser(
        description="Record AURORA UAV telemetry."
    )
    parser.add_argument(
        "--source",
        choices=("simulated", "mavlink"),
        default="simulated",
    )
    parser.add_argument("--duration", type=float, default=10.0)
    parser.add_argument("--interval", type=float, default=1.0)
    parser.add_argument(
        "--output",
        default="datasets/telemetry.jsonl",
    )
    parser.add_argument(
        "--latest",
        default="datasets/telemetry_latest.json",
    )
    parser.add_argument(
        "--endpoint",
        default="udp:127.0.0.1:14550",
    )
    parser.add_argument("--baud", type=int, default=57600)
    args = parser.parse_args()

    if args.duration <= 0:
        parser.error("--duration must be greater than zero")
    if args.interval <= 0:
        parser.error("--interval must be greater than zero")

    output_path = Path(args.output)
    latest_path = Path(args.latest)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    latest_path.parent.mkdir(parents=True, exist_ok=True)

    if output_path.resolve() == latest_path.resolve():
        parser.error("--output and --latest must be different files")

    if args.source == "simulated":
        source = SimulatedTelemetrySource()
        close_source = None
    else:
        from aurora.telemetry.mavlink_source import MavlinkTelemetrySource

        source = MavlinkTelemetrySource(
            endpoint=args.endpoint,
            baud=args.baud,
        )
        close_source = source.close

    start = time.monotonic()
    next_sample = start
    sample_count = 0
    simulated_index = 0

    try:
        with output_path.open("w", encoding="utf-8") as log_file:
            while time.monotonic() - start < args.duration:
                if args.source == "simulated":
                    snapshot = source.get_snapshot(simulated_index)
                    simulated_index += 1
                else:
                    snapshot = source.poll(timeout=args.interval)

                if snapshot is not None:
                    data = snapshot.to_dict()
                    log_file.write(json.dumps(data, ensure_ascii=False) + "\n")
                    log_file.flush()

                    latest_path.write_text(
                        json.dumps(data, indent=2, ensure_ascii=False) + "\n",
                        encoding="utf-8",
                    )

                    sample_count += 1
                    print(
                        f"Sample {sample_count}: "
                        f"lat={data['latitude']:.6f}, "
                        f"lon={data['longitude']:.6f}, "
                        f"alt={data['relative_altitude_m']}"
                    )

                next_sample += args.interval
                delay = next_sample - time.monotonic()
                if delay > 0:
                    time.sleep(delay)
                else:
                    next_sample = time.monotonic()

    finally:
        if close_source is not None:
            close_source()

    print(f"\nTelemetry recording complete: {sample_count} samples")
    print(f"JSONL log: {output_path}")
    print(f"Latest snapshot: {latest_path}")

    if sample_count == 0:
        raise SystemExit(
            "No telemetry samples were received. Check the source or connection."
        )


if __name__ == "__main__":
    main()
