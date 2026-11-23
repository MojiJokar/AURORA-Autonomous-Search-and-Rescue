"""Run the complete AURORA perception pipeline on recorded video."""

import argparse
from pathlib import Path

import cv2

from aurora.autonomy.perception_pipeline import PerceptionPipeline
from aurora.perception.camera_input import iter_video_frames
from aurora.perception.detection.yolo_detector import YOLOPersonDetector
from aurora.perception.tracking import IoUTracker


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Run YOLO person detection, multi-object tracking, "
            "and visual target assessment on a recorded video."
        )
    )

    parser.add_argument("--model", required=True, help="YOLO ONNX model")
    parser.add_argument("--video", required=True, help="Input video")
    parser.add_argument(
        "--output",
        default="datasets/aurora_pipeline_output.mp4",
        help="Annotated output video",
    )
    parser.add_argument(
        "--confidence",
        type=float,
        default=0.40,
        help="Person detection confidence threshold",
    )
    parser.add_argument(
        "--min-iou",
        type=float,
        default=0.30,
        help="Minimum IoU for track matching",
    )
    parser.add_argument(
        "--max-missed",
        type=int,
        default=5,
        help="Maximum missed frames before a track expires",
    )
    parser.add_argument(
        "--output-fps",
        type=float,
        default=20.0,
        help="Frames per second for output video",
    )
    parser.add_argument(
        "--max-frames",
        type=int,
        default=0,
        help="Optional frame limit; 0 processes the whole video",
    )

    return parser.parse_args()


def main():
    args = parse_args()

    video_path = Path(args.video)
    model_path = Path(args.model)
    output_path = Path(args.output)

    if not video_path.is_file():
        raise FileNotFoundError(f"Input video not found: {video_path}")

    if not model_path.is_file():
        raise FileNotFoundError(f"YOLO model not found: {model_path}")

    if video_path.resolve() == output_path.resolve():
        raise ValueError("Output video must not overwrite the input video")

    if args.output_fps <= 0:
        raise ValueError("--output-fps must be greater than zero")

    if args.max_frames < 0:
        raise ValueError("--max-frames cannot be negative")

    output_path.parent.mkdir(parents=True, exist_ok=True)

    detector = YOLOPersonDetector(
        model_path=model_path,
        confidence_threshold=args.confidence,
    )

    tracker = IoUTracker(
        max_missed_frames=args.max_missed,
        min_iou=args.min_iou,
    )

    pipeline = PerceptionPipeline(detector, tracker)

    writer = None
    frame_count = 0
    detection_count = 0
    observed_track_frames = 0
    unique_track_ids = set()

    priority_frame_counts = {
        "HIGH": 0,
        "MEDIUM": 0,
        "LOW": 0,
    }

    try:
        for frame in iter_video_frames(video_path):
            result = pipeline.process_frame(frame)

            if writer is None:
                height, width = result.image.shape[:2]
                fourcc = cv2.VideoWriter_fourcc(*"mp4v")

                writer = cv2.VideoWriter(
                    str(output_path),
                    fourcc,
                    args.output_fps,
                    (width, height),
                )

                if not writer.isOpened():
                    raise RuntimeError(
                        f"Could not create output video: {output_path}"
                    )

            writer.write(result.image)

            frame_count += 1
            detection_count += len(result.detections)

            for track in result.tracks:
                unique_track_ids.add(track.track_id)

            for assessment in result.assessments:
                if assessment.status != "PERSON_CANDIDATE":
                    continue

                observed_track_frames += 1
                priority_frame_counts[assessment.priority] += 1

            if frame_count % 10 == 0:
                visible = sum(
                    item.status == "PERSON_CANDIDATE"
                    for item in result.assessments
                )

                print(
                    f"Processed {frame_count} frames | "
                    f"detections: {len(result.detections)} | "
                    f"visible people: {visible} | "
                    f"unique track IDs: {len(unique_track_ids)}"
                )

            if args.max_frames and frame_count >= args.max_frames:
                print("Reached requested frame limit.")
                break

    finally:
        if writer is not None:
            writer.release()

    if frame_count == 0:
        raise RuntimeError("No frames were processed")

    print()
    print("AURORA PERCEPTION PIPELINE FINISHED")
    print("-----------------------------------")
    print(f"Frames processed: {frame_count}")
    print(f"Person detections: {detection_count}")
    print(f"Unique track IDs observed: {len(unique_track_ids)}")
    print(f"Observed track-frame assessments: {observed_track_frames}")
    print(f"HIGH priority track-frames: {priority_frame_counts['HIGH']}")
    print(f"MEDIUM priority track-frames: {priority_frame_counts['MEDIUM']}")
    print(f"LOW priority track-frames: {priority_frame_counts['LOW']}")
    print(f"Output video: {output_path}")


if __name__ == "__main__":
    main()
