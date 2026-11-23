from dataclasses import FrozenInstanceError

import numpy as np
import pytest

from aurora.perception.detection import (
    BoundingBox,
    DetectionResult,
    Detector,
    validate_confidence,
)
from aurora.perception.image_frame import ImageFrame


@pytest.mark.parametrize("value", [0.0, 0.5, 1.0])
def test_valid_confidence(value):
    assert validate_confidence(value) == value


@pytest.mark.parametrize("value", [-0.1, 1.1])
def test_confidence_out_of_range_is_rejected(value):
    with pytest.raises(ValueError):
        validate_confidence(value)


@pytest.mark.parametrize("value", [float("nan"), float("inf")])
def test_non_finite_confidence_is_rejected(value):
    with pytest.raises(ValueError):
        validate_confidence(value)


@pytest.mark.parametrize("value", [True, "0.5", None])
def test_invalid_confidence_type_is_rejected(value):
    with pytest.raises(TypeError):
        validate_confidence(value)


def test_valid_bounding_box():
    box = BoundingBox(10, 20, 100, 120)

    assert box.x_min == 10
    assert box.y_min == 20
    assert box.x_max == 100
    assert box.y_max == 120


def test_bounding_box_rejects_zero_width():
    with pytest.raises(ValueError):
        BoundingBox(10, 20, 10, 120)


def test_bounding_box_rejects_zero_height():
    with pytest.raises(ValueError):
        BoundingBox(10, 20, 100, 20)


def test_bounding_box_rejects_negative_coordinates():
    with pytest.raises(ValueError):
        BoundingBox(-1, 20, 100, 120)


def test_bounding_box_rejects_non_finite_coordinates():
    with pytest.raises(ValueError):
        BoundingBox(10, 20, float("inf"), 120)


def test_bounding_box_rejects_boolean_coordinates():
    with pytest.raises(TypeError):
        BoundingBox(True, 20, 100, 120)


def test_valid_detection_result():
    box = BoundingBox(10, 20, 100, 120)
    result = DetectionResult(
        class_name="person",
        confidence=0.95,
        bounding_box=box,
    )

    assert result.class_name == "person"
    assert result.confidence == 0.95
    assert result.bounding_box == box


def test_detection_result_rejects_empty_class_name():
    with pytest.raises(ValueError):
        DetectionResult(
            class_name=" ",
            confidence=0.95,
            bounding_box=BoundingBox(10, 20, 100, 120),
        )


def test_detection_result_rejects_invalid_bounding_box_type():
    with pytest.raises(TypeError):
        DetectionResult(
            class_name="person",
            confidence=0.95,
            bounding_box=(10, 20, 100, 120),
        )


def test_detection_result_is_immutable():
    result = DetectionResult(
        class_name="person",
        confidence=0.95,
        bounding_box=BoundingBox(10, 20, 100, 120),
    )

    with pytest.raises(FrozenInstanceError):
        result.class_name = "car"


def test_detector_is_abstract():
    with pytest.raises(TypeError):
        Detector()


def test_concrete_detector_implements_interface():
    class ExampleDetector(Detector):
        def detect(self, frame):
            return [
                DetectionResult(
                    class_name="person",
                    confidence=0.95,
                    bounding_box=BoundingBox(10, 20, 100, 120),
                )
            ]

    detector = ExampleDetector()

    frame = ImageFrame(
        image=np.zeros((120, 160, 3), dtype=np.uint8),
        timestamp=1.0,
        frame_id="test_frame",
    )

    results = detector.detect(frame)

    assert len(results) == 1
    assert results[0].class_name == "person"
    assert results[0].confidence == 0.95
