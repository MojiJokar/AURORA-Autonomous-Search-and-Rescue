import json

import pytest

from aurora.operator.mission_report import MissionReportBuilder
from aurora.telemetry import SimulatedTelemetrySource


def test_report_starts_with_empty_telemetry():
    report = MissionReportBuilder("input.mp4", "output.mp4").to_dict()

    assert report["telemetry"]["sample_count"] == 0
    assert report["telemetry"]["latest"] is None


def test_report_accepts_telemetry_snapshot():
    builder = MissionReportBuilder("input.mp4", "output.mp4")
    snapshot = SimulatedTelemetrySource().get_snapshot(0)

    builder.add_telemetry(snapshot)
    report = builder.to_dict()

    assert report["telemetry"]["sample_count"] == 1
    assert report["telemetry"]["latest"]["source"] == "simulated"
    assert report["telemetry"]["latest"]["latitude"] == pytest.approx(49.8728)


def test_report_loads_jsonl_telemetry(tmp_path):
    builder = MissionReportBuilder("input.mp4", "output.mp4")
    source = SimulatedTelemetrySource()
    log_path = tmp_path / "telemetry.jsonl"

    with log_path.open("w", encoding="utf-8") as file:
        for snapshot in source.samples(3):
            file.write(json.dumps(snapshot.to_dict()) + "\n")

    count = builder.load_telemetry_log(log_path)

    assert count == 3
    assert builder.to_dict()["telemetry"]["sample_count"] == 3
    assert builder.to_dict()["telemetry"]["latest"]["source"] == "simulated"


def test_report_rejects_invalid_jsonl_record(tmp_path):
    builder = MissionReportBuilder("input.mp4", "output.mp4")
    log_path = tmp_path / "bad.jsonl"
    log_path.write_text("{invalid json}\n", encoding="utf-8")

    with pytest.raises(ValueError, match="line 1"):
        builder.load_telemetry_log(log_path)


def test_report_rejects_missing_telemetry_log(tmp_path):
    builder = MissionReportBuilder("input.mp4", "output.mp4")

    with pytest.raises(FileNotFoundError):
        builder.load_telemetry_log(tmp_path / "missing.jsonl")
