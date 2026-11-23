"""Image and recorded-video input for the AURORA perception pipeline."""

from pathlib import Path
import time
from collections.abc import Iterator

import cv2
import numpy as np

from aurora.perception.image_frame import ImageFrame


def load_image(
    path: str | Path,
    frame_id: str = "image_000001",
    timestamp: float | None = None,
) -> ImageFrame:
    """Load an image file and return it as a validated ImageFrame."""
    image_path = Path(path)

    if not image_path.is_file():
        raise FileNotFoundError(f"Image file not found: {image_path}")

    image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)

    if image is None:
        raise ValueError(f"Could not read image file: {image_path}")

    frame_timestamp = time.time() if timestamp is None else timestamp

    return ImageFrame(
        image=image,
        timestamp=frame_timestamp,
        frame_id=frame_id,
    )


def iter_video_frames(
    video_path: str | Path,
    frame_prefix: str = "video",
) -> Iterator[ImageFrame]:
    """Read a recorded video and yield its frames as ImageFrame objects."""
    path = Path(video_path)

    if not path.is_file():
        raise FileNotFoundError(f"Video file not found: {path}")

    if not isinstance(frame_prefix, str) or not frame_prefix.strip():
        raise ValueError("frame_prefix must be a non-empty string")

    capture = cv2.VideoCapture(str(path))

    if not capture.isOpened():
        capture.release()
        raise OSError(f"Could not open video file: {path}")

    frame_index = 0

    try:
        fps = capture.get(cv2.CAP_PROP_FPS)
        valid_fps = np.isfinite(fps) and fps > 0

        while True:
            success, image = capture.read()

            if not success:
                break

            position_ms = capture.get(cv2.CAP_PROP_POS_MSEC)

            if (
                np.isfinite(position_ms)
                and position_ms >= 0
                and (position_ms > 0 or frame_index == 0)
            ):
                timestamp = position_ms / 1000.0
            elif valid_fps:
                timestamp = frame_index / fps
            else:
                timestamp = float(frame_index)

            yield ImageFrame(
                image=image,
                timestamp=float(timestamp),
                frame_id=f"{frame_prefix}_{frame_index:06d}",
            )

            frame_index += 1

    finally:
        capture.release()
