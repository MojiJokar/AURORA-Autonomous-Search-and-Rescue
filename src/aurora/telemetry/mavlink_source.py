"""Read-only MAVLink telemetry adapter for ArduPilot."""

from datetime import datetime, timezone

from aurora.telemetry.models import TelemetrySnapshot


class MavlinkTelemetrySource:
    """Receive UAV telemetry through pymavlink."""

    def __init__(
        self,
        endpoint="udp:127.0.0.1:14550",
        baud=57600,
        heartbeat_timeout=10,
    ):
        try:
            from pymavlink import mavutil
        except ImportError as exc:
            raise RuntimeError(
                "pymavlink is required for MAVLink telemetry. "
                "Install it with: python -m pip install pymavlink"
            ) from exc

        self.mavutil = mavutil
        self.endpoint = endpoint

        self.connection = mavutil.mavlink_connection(
            endpoint,
            baud=baud,
        )

        heartbeat = self.connection.wait_heartbeat(
            timeout=heartbeat_timeout
        )

        if heartbeat is None:
            self.connection.close()
            raise TimeoutError(
                f"No MAVLink heartbeat received from {endpoint}"
            )

        self.state = {
            "latitude": None,
            "longitude": None,
            "altitude_m": None,
            "relative_altitude_m": None,
            "ground_speed_m_s": None,
            "heading_deg": None,
            "armed": None,
            "flight_mode": None,
            "gps_fix_type": None,
            "satellites_visible": None,
        }

    def poll(self, timeout=1.0):
        """Read one MAVLink message and return the latest known state."""

        message = self.connection.recv_match(
            blocking=True,
            timeout=timeout,
        )

        if message is None:
            return None

        message_type = message.get_type()

        if message_type == "GLOBAL_POSITION_INT":
            self.state["latitude"] = message.lat / 1e7
            self.state["longitude"] = message.lon / 1e7
            self.state["altitude_m"] = message.alt / 1000.0
            self.state["relative_altitude_m"] = (
                message.relative_alt / 1000.0
            )

            if message.hdg != 65535:
                self.state["heading_deg"] = message.hdg / 100.0

        elif message_type == "VFR_HUD":
            self.state["ground_speed_m_s"] = float(message.groundspeed)
            self.state["heading_deg"] = float(message.heading)

        elif message_type == "GPS_RAW_INT":
            self.state["gps_fix_type"] = int(message.fix_type)
            self.state["satellites_visible"] = int(message.satellites_visible)

        elif message_type == "HEARTBEAT":
            armed_flag = (
                self.mavutil.mavlink.MAV_MODE_FLAG_SAFETY_ARMED
            )

            self.state["armed"] = bool(message.base_mode & armed_flag)

            try:
                self.state["flight_mode"] = self.connection.flightmode
            except AttributeError:
                self.state["flight_mode"] = None

        if self.state["latitude"] is None:
            return None

        if self.state["longitude"] is None:
            return None

        return TelemetrySnapshot(
            timestamp=datetime.now(timezone.utc).isoformat(),
            source=f"mavlink:{self.endpoint}",
            **self.state,
        )

    def close(self):
        """Close the MAVLink connection."""
        self.connection.close()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.close()
