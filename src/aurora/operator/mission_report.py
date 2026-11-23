"""Build a lightweight JSON mission report from perception results."""

import json
from pathlib import Path


class MissionReportBuilder:
    """Collect mission statistics and per-track summaries."""

    PRIORITY_ORDER = {"HIGH": 0, "MEDIUM": 1, "LOW": 2}

    def __init__(self, source_video: str, output_video: str):
        if not source_video or not output_video:
            raise ValueError("Video paths must not be empty")

        self.source_video = str(source_video)
        self.output_video = str(output_video)
        self.frames_processed = 0
        self.person_detections = 0
        self.observed_track_frame_assessments = 0
        self.priority_track_frames = {
            "HIGH": 0,
            "MEDIUM": 0,
            "LOW": 0,
        }
        self._tracks = {}

    def add_frame(self, result, frame_number: int):
        """Add one processed frame to the mission report."""
        if (
            isinstance(frame_number, bool)
            or not isinstance(frame_number, int)
            or frame_number < 1
        ):
            raise ValueError("frame_number must be a positive integer")

        self.frames_processed += 1
        self.person_detections += len(result.detections)

        observed_tracks = {
            track.track_id: track
            for track in result.tracks
            if track.observed and track.class_name.lower() == "person"
        }

        for assessment in result.assessments:
            if assessment.status != "PERSON_CANDIDATE":
                continue

            track = observed_tracks.get(assessment.track_id)
            if track is None:
                continue

            priority = assessment.priority
            if priority not in self.PRIORITY_ORDER:
                continue

            self.observed_track_frame_assessments += 1
            self.priority_track_frames[priority] += 1

            summary = self._tracks.get(assessment.track_id)

            if summary is None:
                summary = {
                    "track_id": assessment.track_id,
                    "first_seen_frame": frame_number,
                    "last_seen_frame": frame_number,
                    "observed_frames": 0,
                    "max_confidence": 0.0,
                    "max_score": 0.0,
                    "highest_priority": priority,
                    "latest_reasons": [],
                }
                self._tracks[assessment.track_id] = summary

            summary["last_seen_frame"] = frame_number
            summary["observed_frames"] += 1
            summary["max_confidence"] = max(
                summary["max_confidence"], float(track.confidence)
            )
            summary["max_score"] = max(
                summary["max_score"], float(assessment.score)
            )

            current = summary["highest_priority"]
            if self.PRIORITY_ORDER[priority] < self.PRIORITY_ORDER[current]:
                summary["highest_priority"] = priority

            summary["latest_reasons"] = list(assessment.reasons)

    def to_dict(self):
        """Return the report in JSON-compatible form."""
        return {
            "schema_version": 1,
            "source_video": self.source_video,
            "output_video": self.output_video,
            "frames_processed": self.frames_processed,
            "person_detections": self.person_detections,
            "unique_track_ids": len(self._tracks),
            "observed_track_frame_assessments": (
                self.observed_track_frame_assessments
            ),
            "priority_track_frames": dict(self.priority_track_frames),
            "tracks": sorted(
                self._tracks.values(),
                key=lambda item: item["track_id"],
            ),
        }

    def save(self, path):
        """Save the mission report as formatted JSON."""
        report_path = Path(path)
        report_path.parent.mkdir(parents=True, exist_ok=True)
        report_path.write_text(
            json.dumps(self.to_dict(), indent=2),
            encoding="utf-8",
        )
        return report_path
