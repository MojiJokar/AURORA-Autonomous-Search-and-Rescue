"""AURORA object detection foundation."""

from aurora.perception.detection.confidence import validate_confidence
from aurora.perception.detection.detection_result import (
    BoundingBox,
    DetectionResult,
)
from aurora.perception.detection.detector import Detector

__all__ = [
    "BoundingBox",
    "DetectionResult",
    "Detector",
    "validate_confidence",
]
