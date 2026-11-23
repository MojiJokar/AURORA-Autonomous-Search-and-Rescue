"""Lightweight multi-object tracking using greedy IoU matching."""

from dataclasses import dataclass
from numbers import Real

from aurora.perception.detection.detection_result import (
    BoundingBox,
    DetectionResult,
)


def intersection_over_union(
    first: BoundingBox,
    second: BoundingBox,
) -> float:
    """Calculate the intersection-over-union of two bounding boxes."""
    x_left = max(first.x_min, second.x_min)
    y_top = max(first.y_min, second.y_min)
    x_right = min(first.x_max, second.x_max)
    y_bottom = min(first.y_max, second.y_max)

    intersection_width = max(0.0, x_right - x_left)
    intersection_height = max(0.0, y_bottom - y_top)
    intersection = intersection_width * intersection_height

    first_area = (
        (first.x_max - first.x_min)
        * (first.y_max - first.y_min)
    )
    second_area = (
        (second.x_max - second.x_min)
        * (second.y_max - second.y_min)
    )

    union = first_area + second_area - intersection

    if union <= 0:
        return 0.0

    return intersection / union


@dataclass(frozen=True)
class Track:
    """Public snapshot of a tracked object."""

    track_id: int
    class_name: str
    confidence: float
    bounding_box: BoundingBox
    age: int
    hits: int
    missed_frames: int
    observed: bool


@dataclass
class _TrackState:
    """Internal mutable state for one tracked object."""

    track_id: int
    class_name: str
    confidence: float
    bounding_box: BoundingBox
    age: int = 1
    hits: int = 1
    missed_frames: int = 0
    observed: bool = True


class IoUTracker:
    """Track detections across frames using greedy IoU matching.

    This tracker is deliberately lightweight and does not require
    an additional machine-learning model.
    """

    def __init__(
        self,
        max_missed_frames: int = 5,
        min_iou: float = 0.3,
    ) -> None:
        if (
            isinstance(max_missed_frames, bool)
            or not isinstance(max_missed_frames, int)
            or max_missed_frames < 0
        ):
            raise ValueError("max_missed_frames must be a non-negative integer")

        if (
            isinstance(min_iou, bool)
            or not isinstance(min_iou, Real)
            or not 0.0 <= float(min_iou) <= 1.0
        ):
            raise ValueError("min_iou must be between 0 and 1")

        self.max_missed_frames = max_missed_frames
        self.min_iou = float(min_iou)
        self._tracks: list[_TrackState] = []
        self._next_track_id = 1

    def reset(self) -> None:
        """Remove all tracks and restart ID assignment."""
        self._tracks.clear()
        self._next_track_id = 1

    def update(
        self,
        detections: list[DetectionResult],
    ) -> list[Track]:
        """Update tracks from detections in the current frame."""
        if not isinstance(detections, list):
            raise TypeError("detections must be a list")

        if any(not isinstance(item, DetectionResult) for item in detections):
            raise TypeError("every detection must be a DetectionResult")

        for track in self._tracks:
            track.age += 1
            track.missed_frames += 1
            track.observed = False

        candidates = []

        for track_index, track in enumerate(self._tracks):
            for detection_index, detection in enumerate(detections):
                if track.class_name != detection.class_name:
                    continue

                overlap = intersection_over_union(
                    track.bounding_box,
                    detection.bounding_box,
                )

                if overlap >= self.min_iou:
                    candidates.append(
                        (overlap, track_index, detection_index)
                    )

        # Highest-overlap matches are assigned first.
        candidates.sort(reverse=True)

        matched_tracks = set()
        matched_detections = set()

        for _, track_index, detection_index in candidates:
            if track_index in matched_tracks:
                continue
            if detection_index in matched_detections:
                continue

            track = self._tracks[track_index]
            detection = detections[detection_index]

            track.bounding_box = detection.bounding_box
            track.confidence = detection.confidence
            track.hits += 1
            track.missed_frames = 0
            track.observed = True

            matched_tracks.add(track_index)
            matched_detections.add(detection_index)

        # Each unmatched detection starts a new track.
        for detection_index, detection in enumerate(detections):
            if detection_index in matched_detections:
                continue

            self._tracks.append(
                _TrackState(
                    track_id=self._next_track_id,
                    class_name=detection.class_name,
                    confidence=detection.confidence,
                    bounding_box=detection.bounding_box,
                )
            )
            self._next_track_id += 1

        # Expire tracks after too many consecutive missed frames.
        self._tracks = [
            track
            for track in self._tracks
            if track.missed_frames <= self.max_missed_frames
        ]

        return [
            Track(
                track_id=track.track_id,
                class_name=track.class_name,
                confidence=track.confidence,
                bounding_box=track.bounding_box,
                age=track.age,
                hits=track.hits,
                missed_frames=track.missed_frames,
                observed=track.observed,
            )
            for track in self._tracks
        ]
