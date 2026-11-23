"""CPU-based YOLOv8 person detector using OpenCV DNN."""

from pathlib import Path

import cv2
import numpy as np

from aurora.perception.detection.confidence import validate_confidence
from aurora.perception.detection.detection_result import (
    BoundingBox,
    DetectionResult,
)
from aurora.perception.detection.detector import Detector
from aurora.perception.image_frame import ImageFrame


class YOLOPersonDetector(Detector):
    """Detect people with a YOLOv8 COCO ONNX model."""

    PERSON_CLASS_ID = 0
    COCO_CLASS_COUNT = 80

    def __init__(
        self,
        model_path: str | Path,
        confidence_threshold: float = 0.40,
        nms_threshold: float = 0.45,
        input_size: tuple[int, int] = (640, 480),
    ) -> None:
        self.model_path = Path(model_path)

        self.confidence_threshold = validate_confidence(
            confidence_threshold
        )
        self.nms_threshold = validate_confidence(nms_threshold)

        if (
            not isinstance(input_size, tuple)
            or len(input_size) != 2
        ):
            raise TypeError(
                "input_size must be a (width, height) tuple"
            )

        width, height = input_size

        if any(
            isinstance(value, bool) or not isinstance(value, int)
            for value in input_size
        ):
            raise TypeError("Input dimensions must be integers")

        if width <= 0 or height <= 0:
            raise ValueError("Input dimensions must be positive")

        if not self.model_path.is_file():
            raise FileNotFoundError(
                f"YOLO model not found: {self.model_path}"
            )

        self.input_width = width
        self.input_height = height

        self.net = cv2.dnn.readNetFromONNX(
            str(self.model_path)
        )

        self.net.setPreferableBackend(
            cv2.dnn.DNN_BACKEND_OPENCV
        )
        self.net.setPreferableTarget(
            cv2.dnn.DNN_TARGET_CPU
        )

    def _prepare_input(self, image: np.ndarray):
        """Resize and pad an image to the model's input dimensions."""

        original_height, original_width = image.shape[:2]

        scale = min(
            self.input_width / original_width,
            self.input_height / original_height,
        )

        resized_width = int(round(original_width * scale))
        resized_height = int(round(original_height * scale))

        resized = cv2.resize(
            image,
            (resized_width, resized_height),
            interpolation=cv2.INTER_LINEAR,
        )

        pad_x = (self.input_width - resized_width) / 2.0
        pad_y = (self.input_height - resized_height) / 2.0

        left = int(round(pad_x - 0.1))
        right = int(round(pad_x + 0.1))
        top = int(round(pad_y - 0.1))
        bottom = int(round(pad_y + 0.1))

        padded = cv2.copyMakeBorder(
            resized,
            top,
            bottom,
            left,
            right,
            cv2.BORDER_CONSTANT,
            value=(114, 114, 114),
        )

        blob = cv2.dnn.blobFromImage(
            padded,
            scalefactor=1.0 / 255.0,
            size=(self.input_width, self.input_height),
            swapRB=True,
            crop=False,
        )

        return blob, scale, float(left), float(top)

    def _parse_output(self, output: np.ndarray) -> np.ndarray:
        """Convert YOLOv8 raw output into prediction rows."""

        predictions = np.asarray(output)

        if predictions.ndim == 3:
            if predictions.shape[0] != 1:
                raise ValueError(
                    "Only batch size 1 is supported"
                )
            predictions = predictions[0]

        if predictions.ndim != 2:
            raise ValueError(
                f"Unexpected YOLO output dimensions: {predictions.shape}"
            )

        expected_columns = 4 + self.COCO_CLASS_COUNT

        if predictions.shape[0] == expected_columns:
            predictions = predictions.T
        elif predictions.shape[1] != expected_columns:
            raise ValueError(
                "Expected YOLOv8 COCO output with 84 values "
                f"per prediction; received shape {predictions.shape}"
            )

        return predictions

    def detect(
        self,
        frame: ImageFrame,
    ) -> list[DetectionResult]:

        if not isinstance(frame, ImageFrame):
            raise TypeError(
                "frame must be an ImageFrame"
            )

        image = frame.image

        if image.ndim != 3 or image.shape[2] != 3:
            raise ValueError(
                "YOLO person detection requires a 3-channel image"
            )

        original_height, original_width = image.shape[:2]

        blob, scale, pad_x, pad_y = self._prepare_input(image)

        self.net.setInput(blob)
        raw_output = self.net.forward()

        predictions = self._parse_output(raw_output)

        boxes_for_nms = []
        confidences = []
        original_boxes = []

        for prediction in predictions:

            confidence = float(
                prediction[4 + self.PERSON_CLASS_ID]
            )

            if not np.isfinite(confidence):
                continue

            if confidence < self.confidence_threshold:
                continue

            center_x, center_y, box_width, box_height = map(
                float,
                prediction[:4],
            )

            if not all(
                np.isfinite(value)
                for value in (
                    center_x,
                    center_y,
                    box_width,
                    box_height,
                )
            ):
                continue

            if box_width <= 0 or box_height <= 0:
                continue

            x_min = (
                center_x - box_width / 2.0 - pad_x
            ) / scale

            y_min = (
                center_y - box_height / 2.0 - pad_y
            ) / scale

            x_max = (
                center_x + box_width / 2.0 - pad_x
            ) / scale

            y_max = (
                center_y + box_height / 2.0 - pad_y
            ) / scale

            x_min = max(
                0.0,
                min(float(original_width), x_min),
            )
            y_min = max(
                0.0,
                min(float(original_height), y_min),
            )
            x_max = max(
                0.0,
                min(float(original_width), x_max),
            )
            y_max = max(
                0.0,
                min(float(original_height), y_max),
            )

            if x_max <= x_min or y_max <= y_min:
                continue

            boxes_for_nms.append(
                [
                    int(round(x_min)),
                    int(round(y_min)),
                    max(1, int(round(x_max - x_min))),
                    max(1, int(round(y_max - y_min))),
                ]
            )

            confidences.append(confidence)

            original_boxes.append(
                (x_min, y_min, x_max, y_max)
            )

        if not boxes_for_nms:
            return []

        indices = cv2.dnn.NMSBoxes(
            boxes_for_nms,
            confidences,
            self.confidence_threshold,
            self.nms_threshold,
        )

        if indices is None or len(indices) == 0:
            return []

        results = []

        for index in np.asarray(indices).reshape(-1):

            x_min, y_min, x_max, y_max = (
                original_boxes[int(index)]
            )

            results.append(
                DetectionResult(
                    class_name="person",
                    confidence=confidences[int(index)],
                    bounding_box=BoundingBox(
                        x_min=x_min,
                        y_min=y_min,
                        x_max=x_max,
                        y_max=y_max,
                    ),
                )
            )

        return results
