import pytest

from aurora.telemetry.health import (
    TelemetryHealthMonitor,
    TelemetryStatus,
)


class FakeSnapshot:
    timestamp = "2026-11-23T12:00:00+00:00"
    source = "test"


def test_initial_status_is_disconnected():
    monitor = TelemetryHealthMonitor()

    assert monitor.get_status(now=100) == TelemetryStatus.DISCONNECTED


def test_new_snapshot_sets_live_status():
    monitor = TelemetryHealthMonitor()

    status = monitor.update(FakeSnapshot(), now=100)

    assert status == TelemetryStatus.LIVE
    assert monitor.get_status(now=102.9) == TelemetryStatus.LIVE


def test_status_becomes_stale_after_threshold():
    monitor = TelemetryHealthMonitor()

    monitor.update(FakeSnapshot(), now=100)

    assert monitor.get_status(now=103) == TelemetryStatus.STALE


def test_status_becomes_disconnected_after_threshold():
    monitor = TelemetryHealthMonitor()

    monitor.update(FakeSnapshot(), now=100)

    assert monitor.get_status(now=110) == TelemetryStatus.DISCONNECTED


def test_new_snapshot_restores_live_status():
    monitor = TelemetryHealthMonitor()

    monitor.update(FakeSnapshot(), now=100)
    monitor.get_status(now=105)

    monitor.update(FakeSnapshot(), now=106)

    assert monitor.get_status(now=106) == TelemetryStatus.LIVE


def test_explicit_disconnect():
    monitor = TelemetryHealthMonitor()

    monitor.update(FakeSnapshot(), now=100)
    monitor.mark_disconnected()

    assert monitor.get_status(now=101) == TelemetryStatus.DISCONNECTED


def test_health_summary_is_json_compatible():
    monitor = TelemetryHealthMonitor()
    monitor.update(FakeSnapshot(), now=100)

    result = monitor.to_dict(now=101)

    assert result["status"] == "LIVE"
    assert result["connected"] is True
    assert result["age_seconds"] == 1
    assert result["source"] == "test"


def test_invalid_thresholds_are_rejected():
    with pytest.raises(ValueError):
        TelemetryHealthMonitor(stale_after=0)

    with pytest.raises(ValueError):
        TelemetryHealthMonitor(
            stale_after=5,
            disconnect_after=5,
        )


def test_receive_time_cannot_move_backwards():
    monitor = TelemetryHealthMonitor()

    monitor.update(FakeSnapshot(), now=100)

    with pytest.raises(ValueError):
        monitor.update(FakeSnapshot(), now=99)


from aurora.telemetry.health import classify_telemetry_source


def test_simulated_source_is_not_reported_as_live():
    assert classify_telemetry_source("simulated", 0) == "SIMULATED"


def test_old_non_mavlink_snapshot_is_recorded():
    assert classify_telemetry_source("unknown", 100) == "RECORDED"


def test_fresh_mavlink_data_is_live():
    assert classify_telemetry_source("mavlink:udp:127.0.0.1:14550", 1) == "LIVE"


def test_delayed_mavlink_data_is_stale():
    assert classify_telemetry_source("mavlink:test", 5) == "STALE"


def test_old_mavlink_data_is_disconnected():
    assert classify_telemetry_source("mavlink:test", 12) == "DISCONNECTED"


def test_negative_age_is_rejected():
    with pytest.raises(ValueError):
        classify_telemetry_source("mavlink:test", -1)
