"""Simulated telemetry source for development and testing."""

from dataclasses import replace

from aurora.telemetry.models import TelemetrySnapshot


class SimulatedTelemetrySource:
    """Generate a simple sequence of simulated UAV positions."""

    def __init__(
        self,
        latitude=49.8728,
        longitude=8.6512,
        altitude_m=30.0,
        relative_altitude_m=30.0,
        ground_speed_m_s=2.0,
        heading_deg=90.0,
    ):
        self.initial = TelemetrySnapshot(
            latitude=latitude,
            longitude=longitude,
            altitude_m=altitude_m,
            relative_altitude_m=relative_altitude_m,
            ground_speed_m_s=ground_speed_m_s,
            heading_deg=heading_deg,
            armed=False,
            flight_mode="SIMULATION",
            gps_fix_type=3,
            satellites_visible=10,
            source="simulated",
        )

    def get_snapshot(self, index=0):
        """Return a simulated snapshot at the requested sample index."""
        if isinstance(index, bool) or not isinstance(index, int) or index < 0:
            raise ValueError("index must be a non-negative integer")

        # Small north-east movement for demonstration purposes.
        latitude_offset = index * 0.00001
        longitude_offset = index * 0.00001

        return replace(
            self.initial,
            latitude=self.initial.latitude + latitude_offset,
            longitude=self.initial.longitude + longitude_offset,
        )

    def samples(self, count):
        """Yield a finite sequence of simulated snapshots."""
        if isinstance(count, bool) or not isinstance(count, int) or count < 0:
            raise ValueError("count must be a non-negative integer")

        for index in range(count):
            yield self.get_snapshot(index)
