import numpy as np
import pytest

from aurora.autonomy.perception_pipeline import PerceptionPipeline
from aurora.perception.detection.detection_result import (
    BoundingBox,
    DetectionResult,
)
from aurora.perception.image_frame import ImageFrame
from aurora.perception.tracking import IoUTracker


class FakeDetector:
    def __init__(self, detections):
        self.detections = detections

    def detect(self, frame):
        return self.detections


def make_frame(frame_id="test_001"):
    return ImageFrame(
        image=np.zeros((100, 100, 3), dtype=np.uint8),
        timestamp=1.0,
        frame_id=frame_id,
    )


def make_person_detection():
    return DetectionResult(
        class_name="person",
        confidence=0.90,
        bounding_box=BoundingBox(
            x_min=10,
            y_min=10,
            x_max=90,
            y_max=90,
        ),
    )


def test_pipeline_connects_detection_tracking_and_assessment():
    detector = FakeDetector([make_person_detection()])
    tracker = IoUTracker()
    pipeline = PerceptionPipeline(detector, tracker)

    result = pipeline.process_frame(make_frame())

    assert len(result.detections) == 1
    assert len(result.tracks) == 1
    assert len(result.assessments) == 1
    assert result.tracks[0].track_id == 1
    assert result.assessments[0].status == "PERSON_CANDIDATE"
    assert result.assessments[0].priority == "MEDIUM"


def test_pipeline_keeps_track_id_across_frames():
    detector = FakeDetector([make_person_detection()])
    tracker = IoUTracker()
    pipeline = PerceptionPipeline(detector, tracker)

    first = pipeline.process_frame(make_frame("frame_1"))
    second = pipeline.process_frame(make_frame("frame_2"))

    assert first.tracks[0].track_id == 1
    assert second.tracks[0].track_id == 1
    assert second.tracks[0].hits == 2
    assert second.assessments[0].priority == "HIGH"


def test_pipeline_draws_annotations_without_modifying_input():
    detector = FakeDetector([make_person_detection()])
    pipeline = PerceptionPipeline(detector, IoUTracker())
    frame = make_frame()

    result = pipeline.process_frame(frame)

    assert result.image.shape == frame.image.shape
    assert not np.array_equal(result.image, frame.image)
    assert np.count_nonzero(frame.image) == 0


def test_pipeline_handles_frames_without_detections():
    detector = FakeDetector([])
    pipeline = PerceptionPipeline(detector, IoUTracker())

    result = pipeline.process_frame(make_frame())

    assert result.detections == ()
    assert result.tracks == ()
    assert result.assessments == ()


def test_pipeline_rejects_invalid_frame_type():
    pipeline = PerceptionPipeline(FakeDetector([]), IoUTracker())

    with pytest.raises(TypeError):
        pipeline.process_frame(object())


def test_pipeline_rejects_grayscale_frame():
    pipeline = PerceptionPipeline(FakeDetector([]), IoUTracker())

    frame = ImageFrame(
        image=np.zeros((100, 100), dtype=np.uint8),
        timestamp=1.0,
        frame_id="gray_001",
    )

    with pytest.raises(ValueError):
        pipeline.process_frame(frame)


def test_pipeline_rejects_detector_without_detect_method():
    with pytest.raises(TypeError):
        PerceptionPipeline(object(), IoUTracker())


def test_pipeline_requires_iou_tracker():
    with pytest.raises(TypeError):
        PerceptionPipeline(FakeDetector([]), object())
