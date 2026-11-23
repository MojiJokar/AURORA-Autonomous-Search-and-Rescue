"""Telemetry data models for AURORA."""

from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
import math


@dataclass(frozen=True)
class TelemetrySnapshot:
    """One validated snapshot of UAV telemetry."""

    latitude: float
    longitude: float
    altitude_m: float | None = None
    relative_altitude_m: float | None = None
    ground_speed_m_s: float | None = None
    heading_deg: float | None = None
    armed: bool | None = None
    flight_mode: str | None = None
    gps_fix_type: int | None = None
    satellites_visible: int | None = None
    timestamp: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )
    source: str = "unknown"

    def __post_init__(self):
        numeric_fields = (
            "latitude",
            "longitude",
            "altitude_m",
            "relative_altitude_m",
            "ground_speed_m_s",
            "heading_deg",
        )

        for name in numeric_fields:
            value = getattr(self, name)
            if value is None:
                continue
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                raise ValueError(f"{name} must be a number or None")
            if not math.isfinite(value):
                raise ValueError(f"{name} must be finite")

        if not -90 <= self.latitude <= 90:
            raise ValueError("latitude must be between -90 and 90")

        if not -180 <= self.longitude <= 180:
            raise ValueError("longitude must be between -180 and 180")

        if self.ground_speed_m_s is not None and self.ground_speed_m_s < 0:
            raise ValueError("ground_speed_m_s cannot be negative")

        if self.heading_deg is not None and not 0 <= self.heading_deg < 360:
            raise ValueError("heading_deg must be between 0 and 360")

        for name in ("gps_fix_type", "satellites_visible"):
            value = getattr(self, name)
            if value is not None:
                if isinstance(value, bool) or not isinstance(value, int):
                    raise ValueError(f"{name} must be an integer or None")
                if value < 0:
                    raise ValueError(f"{name} cannot be negative")

        if self.armed is not None and not isinstance(self.armed, bool):
            raise ValueError("armed must be a boolean or None")

        if self.flight_mode is not None and not isinstance(self.flight_mode, str):
            raise ValueError("flight_mode must be a string or None")

        if not isinstance(self.timestamp, str) or not self.timestamp:
            raise ValueError("timestamp must be a non-empty string")

        if not isinstance(self.source, str) or not self.source:
            raise ValueError("source must be a non-empty string")

    def to_dict(self):
        """Convert the snapshot into a JSON-compatible dictionary."""
        return asdict(self)

    @classmethod
    def from_dict(cls, data):
        """Create a snapshot from a dictionary."""
        return cls(**data)
