"""Lightweight visual prioritisation of tracked person detections.

This module ranks visual detections for operator attention.
It does not determine whether a person is injured or needs rescue.
"""

from dataclasses import dataclass
from typing import Iterable


@dataclass(frozen=True)
class TargetAssessment:
    track_id: int
    status: str
    priority: str
    score: float
    confidence: float
    persistence: float
    visibility: float
    reasons: tuple[str, ...]


def _clamp(value: float) -> float:
    return max(0.0, min(1.0, float(value)))


def _box_coordinates(box):
    """Read common bounding-box coordinate formats."""
    if all(hasattr(box, name) for name in ("x_min", "y_min", "x_max", "y_max")):
        return (
            float(box.x_min), float(box.y_min),
            float(box.x_max), float(box.y_max),
        )

    if all(hasattr(box, name) for name in ("x1", "y1", "x2", "y2")):
        return float(box.x1), float(box.y1), float(box.x2), float(box.y2)

    if all(hasattr(box, name) for name in ("left", "top", "right", "bottom")):
        return (
            float(box.left), float(box.top),
            float(box.right), float(box.bottom),
        )

    if all(hasattr(box, name) for name in ("x", "y", "width", "height")):
        return (
            float(box.x), float(box.y),
            float(box.x + box.width), float(box.y + box.height),
        )

    raise ValueError("Unsupported bounding-box coordinate format")


def assess_track(track, frame_width: int, frame_height: int) -> TargetAssessment:
    """Assess one tracked object and assign an operator-attention priority."""
    if frame_width <= 0 or frame_height <= 0:
        raise ValueError("Frame dimensions must be positive")

    track_id = int(track.track_id)
    class_name = str(track.class_name).lower()
    confidence = _clamp(track.confidence)

    if class_name != "person":
        return TargetAssessment(
            track_id, "NOT_PERSON", "NONE", 0.0,
            confidence, 0.0, 0.0,
            ("Tracked object is not classified as a person.",),
        )

    if not track.observed or track.missed_frames > 0:
        return TargetAssessment(
            track_id, "NOT_VISIBLE", "NONE", 0.0,
            confidence, 0.0, 0.0,
            ("Person track is temporarily not observed.",),
        )

    hits = max(0, int(track.hits))
    persistence = _clamp(hits / 10.0)

    x1, y1, x2, y2 = _box_coordinates(track.bounding_box)
    box_width = max(0.0, min(frame_width, x2) - max(0.0, x1))
    box_height = max(0.0, min(frame_height, y2) - max(0.0, y1))

    frame_area = float(frame_width * frame_height)
    box_area = box_width * box_height
    visibility = _clamp(box_area / (frame_area * 0.08))

    # These weights provide a simple, explainable visual ranking.
    score = (
        0.50 * confidence
        + 0.30 * persistence
        + 0.20 * visibility
    )
    score = _clamp(score)

    if score >= 0.70:
        priority = "HIGH"
    elif score >= 0.45:
        priority = "MEDIUM"
    else:
        priority = "LOW"

    reasons = (
        f"Person detection confidence: {confidence:.2f}.",
        f"Track persistence: {hits} successful detections.",
        f"Relative bounding-box visibility: {visibility:.2f}.",
    )

    return TargetAssessment(
        track_id=track_id,
        status="PERSON_CANDIDATE",
        priority=priority,
        score=score,
        confidence=confidence,
        persistence=persistence,
        visibility=visibility,
        reasons=reasons,
    )


def assess_tracks(
    tracks: Iterable,
    frame_width: int,
    frame_height: int,
) -> list[TargetAssessment]:
    """Assess tracks and return highest-priority candidates first."""
    assessments = [
        assess_track(track, frame_width, frame_height)
        for track in tracks
    ]

    priority_order = {"HIGH": 0, "MEDIUM": 1, "LOW": 2, "NONE": 3}
    return sorted(
        assessments,
        key=lambda item: (priority_order[item.priority], -item.score),
    )
