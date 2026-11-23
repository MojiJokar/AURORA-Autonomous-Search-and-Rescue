import pytest

from aurora.perception.detection.detection_result import (
    BoundingBox,
    DetectionResult,
)
from aurora.perception.tracking import (
    IoUTracker,
    intersection_over_union,
)


def detection(x_min, y_min, x_max, y_max, class_name="person"):
    return DetectionResult(
        class_name=class_name,
        confidence=0.9,
        bounding_box=BoundingBox(
            x_min=x_min,
            y_min=y_min,
            x_max=x_max,
            y_max=y_max,
        ),
    )


def test_iou_identical_boxes_is_one():
    box = BoundingBox(10, 10, 50, 50)
    assert intersection_over_union(box, box) == pytest.approx(1.0)


def test_iou_non_overlapping_boxes_is_zero():
    first = BoundingBox(0, 0, 10, 10)
    second = BoundingBox(20, 20, 30, 30)
    assert intersection_over_union(first, second) == 0.0


def test_iou_partial_overlap():
    first = BoundingBox(0, 0, 10, 10)
    second = BoundingBox(5, 0, 15, 10)
    assert intersection_over_union(first, second) == pytest.approx(1 / 3)


def test_first_detection_creates_track():
    tracker = IoUTracker()
    tracks = tracker.update([detection(10, 10, 50, 50)])

    assert len(tracks) == 1
    assert tracks[0].track_id == 1
    assert tracks[0].hits == 1
    assert tracks[0].observed is True


def test_matching_detection_keeps_same_id():
    tracker = IoUTracker()

    first = tracker.update([detection(10, 10, 50, 50)])
    second = tracker.update([detection(12, 10, 52, 50)])

    assert first[0].track_id == second[0].track_id
    assert second[0].hits == 2
    assert second[0].observed is True


def test_new_detection_gets_new_id():
    tracker = IoUTracker()

    tracker.update([detection(10, 10, 50, 50)])
    tracks = tracker.update([
        detection(10, 10, 50, 50),
        detection(200, 200, 240, 240),
    ])

    assert len(tracks) == 2
    assert {track.track_id for track in tracks} == {1, 2}


def test_missed_track_is_temporarily_retained():
    tracker = IoUTracker(max_missed_frames=2)

    tracker.update([detection(10, 10, 50, 50)])
    tracks = tracker.update([])

    assert len(tracks) == 1
    assert tracks[0].missed_frames == 1
    assert tracks[0].observed is False


def test_track_expires_after_maximum_missed_frames():
    tracker = IoUTracker(max_missed_frames=2)

    tracker.update([detection(10, 10, 50, 50)])
    tracker.update([])
    tracker.update([])
    tracks = tracker.update([])

    assert tracks == []


def test_detection_after_expiration_gets_new_id():
    tracker = IoUTracker(max_missed_frames=1)

    tracker.update([detection(10, 10, 50, 50)])
    tracker.update([])
    tracker.update([])

    tracks = tracker.update([detection(10, 10, 50, 50)])

    assert len(tracks) == 1
    assert tracks[0].track_id == 2


def test_different_classes_do_not_match():
    tracker = IoUTracker()

    first = tracker.update([detection(10, 10, 50, 50, "person")])
    second = tracker.update([detection(10, 10, 50, 50, "car")])

    assert len(second) == 2
    assert first[0].track_id != second[1].track_id


def test_reset_clears_tracks_and_restarts_ids():
    tracker = IoUTracker()

    tracker.update([detection(10, 10, 50, 50)])
    tracker.reset()

    tracks = tracker.update([detection(10, 10, 50, 50)])

    assert len(tracks) == 1
    assert tracks[0].track_id == 1


@pytest.mark.parametrize("value", [-1, 1.5, True])
def test_invalid_max_missed_frames(value):
    with pytest.raises(ValueError):
        IoUTracker(max_missed_frames=value)


@pytest.mark.parametrize("value", [-0.1, 1.1, float("nan"), True])
def test_invalid_min_iou(value):
    with pytest.raises(ValueError):
        IoUTracker(min_iou=value)


def test_invalid_detection_list_type():
    tracker = IoUTracker()

    with pytest.raises(TypeError):
        tracker.update(())


def test_invalid_detection_item_type():
    tracker = IoUTracker()

    with pytest.raises(TypeError):
        tracker.update(["not a detection"])
