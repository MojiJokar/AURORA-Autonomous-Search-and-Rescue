"""Telemetry freshness and connection-health monitoring."""

from enum import Enum
import time


class TelemetryStatus(str, Enum):
    LIVE = "LIVE"
    STALE = "STALE"
    DISCONNECTED = "DISCONNECTED"


class TelemetryHealthMonitor:
    """Monitor incoming telemetry using local monotonic receive time."""

    def __init__(self, stale_after=3.0, disconnect_after=10.0):
        if stale_after <= 0:
            raise ValueError("stale_after must be positive")
        if disconnect_after <= stale_after:
            raise ValueError(
                "disconnect_after must be greater than stale_after"
            )

        self.stale_after = float(stale_after)
        self.disconnect_after = float(disconnect_after)
        self.last_received_at = None
        self.last_snapshot = None
        self.connected = False

    def update(self, snapshot, now=None):
        """Record a successfully received telemetry snapshot."""
        received_at = time.monotonic() if now is None else float(now)

        if self.last_received_at is not None:
            if received_at < self.last_received_at:
                raise ValueError("receive time cannot move backwards")

        self.last_received_at = received_at
        self.last_snapshot = snapshot
        self.connected = True

        return self.get_status(now=received_at)

    def mark_disconnected(self):
        """Explicitly mark the telemetry connection as lost."""
        self.connected = False

    def get_status(self, now=None):
        """Return LIVE, STALE or DISCONNECTED."""
        current_time = time.monotonic() if now is None else float(now)

        if not self.connected or self.last_received_at is None:
            return TelemetryStatus.DISCONNECTED

        age = current_time - self.last_received_at

        if age < 0:
            raise ValueError("current time precedes last receive time")

        if age >= self.disconnect_after:
            return TelemetryStatus.DISCONNECTED

        if age >= self.stale_after:
            return TelemetryStatus.STALE

        return TelemetryStatus.LIVE

    def to_dict(self, now=None):
        """Return a JSON-compatible health summary."""
        current_time = time.monotonic() if now is None else float(now)
        status = self.get_status(now=current_time)

        age = None
        if self.last_received_at is not None:
            age = max(0.0, current_time - self.last_received_at)

        return {
            "status": status.value,
            "connected": self.connected,
            "age_seconds": age,
            "stale_after_seconds": self.stale_after,
            "disconnect_after_seconds": self.disconnect_after,
            "last_snapshot_timestamp": (
                self.last_snapshot.timestamp
                if self.last_snapshot is not None
                else None
            ),
            "source": (
                self.last_snapshot.source
                if self.last_snapshot is not None
                else None
            ),
        }


def classify_telemetry_source(
    source,
    age_seconds,
    stale_after=3.0,
    disconnect_after=10.0,
):
    """Classify telemetry without confusing recorded data with live data."""
    if not isinstance(source, str) or not source:
        raise ValueError("source must be a non-empty string")

    if age_seconds < 0:
        raise ValueError("age_seconds cannot be negative")

    if disconnect_after <= stale_after or stale_after <= 0:
        raise ValueError("invalid telemetry freshness thresholds")

    if source == "simulated":
        return "SIMULATED"

    if not source.startswith("mavlink:"):
        return "RECORDED"

    if age_seconds >= disconnect_after:
        return "DISCONNECTED"

    if age_seconds >= stale_after:
        return "STALE"

    return "LIVE"
