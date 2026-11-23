"""Object detection interfaces and implementations."""

from aurora.perception.detection.confidence import validate_confidence
from aurora.perception.detection.detection_result import (
    BoundingBox,
    DetectionResult,
)
from aurora.perception.detection.detector import Detector
from aurora.perception.detection.yolo_detector import YOLOPersonDetector

__all__ = [
    "BoundingBox",
    "DetectionResult",
    "Detector",
    "YOLOPersonDetector",
    "validate_confidence",
]
