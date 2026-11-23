"""Integrated AURORA detection, tracking, and target-assessment pipeline."""

from dataclasses import dataclass

import cv2
import numpy as np

from aurora.autonomy.target_assessment import (
    TargetAssessment,
    assess_tracks,
)
from aurora.perception.detection.detection_result import DetectionResult
from aurora.perception.image_frame import ImageFrame
from aurora.perception.tracking import IoUTracker, Track


@dataclass(frozen=True)
class PerceptionFrameResult:
    """Results produced by processing one camera frame."""

    frame_id: str
    image: np.ndarray
    detections: tuple[DetectionResult, ...]
    tracks: tuple[Track, ...]
    assessments: tuple[TargetAssessment, ...]


class PerceptionPipeline:
    """Connect person detection, tracking, and visual assessment."""

    def __init__(self, detector, tracker: IoUTracker):
        if not callable(getattr(detector, "detect", None)):
            raise TypeError("detector must provide a detect(frame) method")

        if not isinstance(tracker, IoUTracker):
            raise TypeError("tracker must be an IoUTracker")

        self.detector = detector
        self.tracker = tracker

    def process_frame(self, frame: ImageFrame) -> PerceptionFrameResult:
        """Process one frame and return annotated image and assessments."""

        if not isinstance(frame, ImageFrame):
            raise TypeError("frame must be an ImageFrame")

        if frame.image.ndim != 3 or frame.image.shape[2] != 3:
            raise ValueError("Pipeline requires a 3-channel color image")

        detections = self.detector.detect(frame)
        tracks = self.tracker.update(detections)

        assessments = assess_tracks(
            tracks,
            frame_width=frame.width,
            frame_height=frame.height,
        )

        assessment_by_id = {
            assessment.track_id: assessment
            for assessment in assessments
        }

        annotated = frame.image.copy()

        visible_assessments = [
            item for item in assessments
            if item.status == "PERSON_CANDIDATE"
        ]

        priority_counts = {
            "HIGH": sum(item.priority == "HIGH" for item in visible_assessments),
            "MEDIUM": sum(item.priority == "MEDIUM" for item in visible_assessments),
            "LOW": sum(item.priority == "LOW" for item in visible_assessments),
        }

        # Keep bounding boxes green, with priority shown in the label.
        for track in tracks:
            if not track.observed or track.class_name.lower() != "person":
                continue

            assessment = assessment_by_id[track.track_id]
            box = track.bounding_box

            x1 = max(0, min(frame.width - 1, int(box.x_min)))
            y1 = max(0, min(frame.height - 1, int(box.y_min)))
            x2 = max(0, min(frame.width - 1, int(box.x_max)))
            y2 = max(0, min(frame.height - 1, int(box.y_max)))

            if x2 <= x1 or y2 <= y1:
                continue

            cv2.rectangle(
                annotated,
                (x1, y1),
                (x2, y2),
                (0, 255, 0),
                2,
            )

            label = (
                f"ID {track.track_id} | "
                f"{track.confidence:.2f} | "
                f"{assessment.priority} | "
                f"{assessment.score:.2f}"
            )

            text_y = max(20, y1 - 8)

            cv2.putText(
                annotated,
                label,
                (x1, text_y),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.48,
                (0, 255, 0),
                2,
                cv2.LINE_AA,
            )

        header = (
            f"DETECTIONS: {len(detections)}  "
            f"VISIBLE: {len(visible_assessments)}  "
            f"HIGH: {priority_counts['HIGH']}  "
            f"MED: {priority_counts['MEDIUM']}  "
            f"LOW: {priority_counts['LOW']}"
        )

        cv2.rectangle(
            annotated,
            (0, 0),
            (frame.width, 30),
            (25, 25, 25),
            -1,
        )

        cv2.putText(
            annotated,
            header,
            (8, 21),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.48,
            (255, 255, 255),
            1,
            cv2.LINE_AA,
        )

        return PerceptionFrameResult(
            frame_id=frame.frame_id,
            image=annotated,
            detections=tuple(detections),
            tracks=tuple(tracks),
            assessments=tuple(assessments),
        )
