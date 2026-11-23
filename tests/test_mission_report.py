from types import SimpleNamespace

import json
import pytest

from aurora.operator.mission_report import MissionReportBuilder


def make_result(
    track_id=4,
    priority="MEDIUM",
    confidence=0.9,
    score=0.65,
    observed=True,
    status="PERSON_CANDIDATE",
):
    track = SimpleNamespace(
        track_id=track_id,
        class_name="person",
        confidence=confidence,
        observed=observed,
    )
    assessment = SimpleNamespace(
        track_id=track_id,
        status=status,
        priority=priority,
        score=score,
        reasons=("Test assessment reason.",),
    )
    return SimpleNamespace(
        detections=[object()],
        tracks=[track],
        assessments=[assessment],
    )


def test_empty_report_has_zero_counts():
    builder = MissionReportBuilder("input.mp4", "output.mp4")
    report = builder.to_dict()

    assert report["frames_processed"] == 0
    assert report["person_detections"] == 0
    assert report["unique_track_ids"] == 0


def test_add_frame_records_mission_counts():
    builder = MissionReportBuilder("input.mp4", "output.mp4")
    builder.add_frame(make_result(), 1)

    report = builder.to_dict()

    assert report["frames_processed"] == 1
    assert report["person_detections"] == 1
    assert report["unique_track_ids"] == 1
    assert report["observed_track_frame_assessments"] == 1
    assert report["priority_track_frames"]["MEDIUM"] == 1


def test_same_track_is_counted_once_as_unique_track():
    builder = MissionReportBuilder("input.mp4", "output.mp4")
    builder.add_frame(make_result(), 1)
    builder.add_frame(make_result(priority="HIGH", score=0.8), 2)

    report = builder.to_dict()
    track = report["tracks"][0]

    assert report["unique_track_ids"] == 1
    assert track["observed_frames"] == 2
    assert track["first_seen_frame"] == 1
    assert track["last_seen_frame"] == 2
    assert track["highest_priority"] == "HIGH"


def test_unobserved_track_is_not_added_to_candidate_summary():
    builder = MissionReportBuilder("input.mp4", "output.mp4")
    builder.add_frame(make_result(observed=False), 1)

    report = builder.to_dict()

    assert report["unique_track_ids"] == 0
    assert report["observed_track_frame_assessments"] == 0


def test_non_candidate_assessment_is_not_counted():
    builder = MissionReportBuilder("input.mp4", "output.mp4")
    builder.add_frame(make_result(status="NOT_VISIBLE"), 1)

    report = builder.to_dict()

    assert report["unique_track_ids"] == 0
    assert report["observed_track_frame_assessments"] == 0


def test_report_saves_valid_json(tmp_path):
    builder = MissionReportBuilder("input.mp4", "output.mp4")
    builder.add_frame(make_result(), 1)

    path = builder.save(tmp_path / "reports" / "mission.json")
    loaded = json.loads(path.read_text(encoding="utf-8"))

    assert loaded["frames_processed"] == 1
    assert loaded["tracks"][0]["track_id"] == 4


@pytest.mark.parametrize("frame_number", [0, -1, True, 1.5])
def test_invalid_frame_number_is_rejected(frame_number):
    builder = MissionReportBuilder("input.mp4", "output.mp4")

    with pytest.raises(ValueError):
        builder.add_frame(make_result(), frame_number)


def test_empty_video_paths_are_rejected():
    with pytest.raises(ValueError):
        MissionReportBuilder("", "output.mp4")
