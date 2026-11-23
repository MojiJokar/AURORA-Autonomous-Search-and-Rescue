"""Data structures representing object detection results."""

from dataclasses import dataclass
from math import isfinite
from numbers import Real

from aurora.perception.detection.confidence import validate_confidence


@dataclass(frozen=True)
class BoundingBox:
    """An axis-aligned bounding box in image pixel coordinates."""

    x_min: float
    y_min: float
    x_max: float
    y_max: float

    def __post_init__(self):
        coordinates = (
            self.x_min,
            self.y_min,
            self.x_max,
            self.y_max,
        )

        for coordinate in coordinates:
            if isinstance(coordinate, bool) or not isinstance(coordinate, Real):
                raise TypeError("bounding box coordinates must be real numbers")

            if not isfinite(coordinate):
                raise ValueError("bounding box coordinates must be finite")

            if coordinate < 0:
                raise ValueError("bounding box coordinates cannot be negative")

        if self.x_max <= self.x_min:
            raise ValueError("x_max must be greater than x_min")

        if self.y_max <= self.y_min:
            raise ValueError("y_max must be greater than y_min")


@dataclass(frozen=True)
class DetectionResult:
    """A detected object's class, confidence, and bounding box."""

    class_name: str
    confidence: float
    bounding_box: BoundingBox

    def __post_init__(self):
        if not isinstance(self.class_name, str) or not self.class_name.strip():
            raise ValueError("class_name must be a non-empty string")

        object.__setattr__(
            self,
            "confidence",
            validate_confidence(self.confidence),
        )

        if not isinstance(self.bounding_box, BoundingBox):
            raise TypeError("bounding_box must be a BoundingBox instance")
