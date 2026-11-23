"""Run YOLOv8 person detection on a still image."""

import argparse
from pathlib import Path

import cv2

from aurora.perception.camera_input import load_image
from aurora.perception.detection import YOLOPersonDetector


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Detect people in an image using YOLOv8 ONNX."
    )
    parser.add_argument("--model", required=True, help="Path to YOLOv8 ONNX model")
    parser.add_argument("--image", required=True, help="Path to input image")
    parser.add_argument(
        "--confidence",
        type=float,
        default=0.40,
        help="Minimum person confidence (default: 0.40)",
    )
    parser.add_argument(
        "--output",
        default="",
        help="Optional path to save an annotated image",
    )
    args = parser.parse_args()

    frame = load_image(args.image)
    detector = YOLOPersonDetector(
        model_path=args.model,
        confidence_threshold=args.confidence,
    )

    detections = detector.detect(frame)

    print(f"Image: {args.image}")
    print(f"Detections: {len(detections)}")

    annotated = frame.image.copy()

    for number, detection in enumerate(detections, start=1):
        box = detection.bounding_box
        x1, y1 = int(box.x_min), int(box.y_min)
        x2, y2 = int(box.x_max), int(box.y_max)

        cv2.rectangle(annotated, (x1, y1), (x2, y2), (0, 255, 0), 2)
        label = f"person {detection.confidence:.2f}"
        cv2.putText(
            annotated,
            label,
            (x1, max(20, y1 - 8)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 255, 0),
            2,
        )

        print(
            f"{number}. person confidence={detection.confidence:.3f} "
            f"box=({x1}, {y1}, {x2}, {y2})"
        )

    if args.output:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        if not cv2.imwrite(str(output_path), annotated):
            raise RuntimeError(f"Could not write output image: {output_path}")
        print(f"Annotated image saved to: {output_path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
