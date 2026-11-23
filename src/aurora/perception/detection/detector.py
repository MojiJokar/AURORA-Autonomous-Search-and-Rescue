"""Common interface for AURORA object detection implementations."""

from abc import ABC, abstractmethod

from aurora.perception.image_frame import ImageFrame
from aurora.perception.detection.detection_result import DetectionResult


class Detector(ABC):
    """Base interface for image-based object detectors."""

    @abstractmethod
    def detect(self, frame: ImageFrame) -> list[DetectionResult]:
        """Detect objects in a validated image frame.

        Implementations should return one DetectionResult per detected object.
        """
        raise NotImplementedError
