"""Run YOLOv8 person detection and IoU tracking on a video."""

import argparse
from pathlib import Path

import cv2

from aurora.perception.camera_input import iter_video_frames
from aurora.perception.detection.yolo_detector import YOLOPersonDetector
from aurora.perception.tracking import IoUTracker


def parse_args():
    parser = argparse.ArgumentParser(
        description="Detect and track people in a recorded video."
    )
    parser.add_argument("--model", required=True, help="YOLOv8 ONNX model path")
    parser.add_argument("--video", required=True, help="Input video path")
    parser.add_argument(
        "--output",
        default="datasets/tracked_output.mp4",
        help="Output annotated video path",
    )
    parser.add_argument("--confidence", type=float, default=0.40)
    parser.add_argument("--min-iou", type=float, default=0.30)
    parser.add_argument("--max-missed", type=int, default=5)
    return parser.parse_args()


def main():
    args = parse_args()

    video_path = Path(args.video)
    model_path = Path(args.model)
    output_path = Path(args.output)

    if not video_path.is_file():
        raise FileNotFoundError(f"Input video not found: {video_path}")

    if not model_path.is_file():
        raise FileNotFoundError(f"Model not found: {model_path}")

    output_path.parent.mkdir(parents=True, exist_ok=True)

    detector = YOLOPersonDetector(
        model_path=str(model_path),
        confidence_threshold=args.confidence,
    )
    tracker = IoUTracker(
        max_missed_frames=args.max_missed,
        min_iou=args.min_iou,
    )

    writer = None
    frame_count = 0

    try:
        for frame in iter_video_frames(str(video_path)):
            detections = detector.detect(frame)
            tracks = tracker.update(detections)

            image = frame.image.copy()

            for track in tracks:
                if not track.observed:
                    continue

                box = track.bounding_box
                x1 = int(box.x_min)
                y1 = int(box.y_min)
                x2 = int(box.x_max)
                y2 = int(box.y_max)

                label = (
                    f"Person ID {track.track_id} "
                    f"{track.confidence:.2f}"
                )

                cv2.rectangle(
                    image,
                    (x1, y1),
                    (x2, y2),
                    (0, 255, 0),
                    2,
                )
                cv2.putText(
                    image,
                    label,
                    (x1, max(20, y1 - 10)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (0, 255, 0),
                    2,
                )

            if writer is None:
                height, width = image.shape[:2]
                fourcc = cv2.VideoWriter_fourcc(*"mp4v")
                writer = cv2.VideoWriter(
                    str(output_path),
                    fourcc,
                    20.0,
                    (width, height),
                )

                if not writer.isOpened():
                    raise RuntimeError(
                        f"Could not create output video: {output_path}"
                    )

            writer.write(image)
            frame_count += 1

            if frame_count % 10 == 0:
                print(
                    f"Processed {frame_count} frames; "
                    f"active tracks: {len(tracks)}"
                )

    finally:
        if writer is not None:
            writer.release()

    if frame_count == 0:
        raise RuntimeError(
            "No video frames were processed. Check the input video."
        )

    print(f"Finished. Processed {frame_count} frames.")
    print(f"Tracked video saved to: {output_path}")


if __name__ == "__main__":
    main()
