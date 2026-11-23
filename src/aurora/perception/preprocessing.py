"""Image preprocessing utilities for the AURORA perception pipeline."""

import cv2
import numpy as np


def _validate_image(image: np.ndarray) -> None:
    """Validate supported camera image formats."""
    if not isinstance(image, np.ndarray):
        raise TypeError("image must be a NumPy array")

    if image.size == 0:
        raise ValueError("image cannot be empty")

    if image.dtype != np.uint8:
        raise ValueError("image must use uint8 data")

    if image.ndim not in (2, 3):
        raise ValueError("image must be grayscale or a color image")

    if image.ndim == 3 and image.shape[2] not in (1, 3, 4):
        raise ValueError("image must have 1, 3, or 4 channels")


def resize_image(image: np.ndarray, width: int, height: int) -> np.ndarray:
    """Resize an image to the requested width and height."""
    _validate_image(image)

    if not isinstance(width, int) or not isinstance(height, int):
        raise TypeError("width and height must be integers")

    if width <= 0 or height <= 0:
        raise ValueError("width and height must be positive")

    interpolation = (
        cv2.INTER_AREA
        if width < image.shape[1] or height < image.shape[0]
        else cv2.INTER_LINEAR
    )

    return cv2.resize(image, (width, height), interpolation=interpolation)


def to_rgb(image: np.ndarray) -> np.ndarray:
    """Convert a three-channel OpenCV BGR image to RGB."""
    _validate_image(image)

    if image.ndim != 3 or image.shape[2] != 3:
        raise ValueError("to_rgb requires a three-channel BGR image")

    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)


def to_grayscale(image: np.ndarray) -> np.ndarray:
    """Convert a BGR image to grayscale, or copy an existing grayscale image."""
    _validate_image(image)

    if image.ndim == 2:
        return image.copy()

    if image.shape[2] == 1:
        return image[:, :, 0].copy()

    if image.shape[2] != 3:
        raise ValueError("to_grayscale requires a grayscale or three-channel BGR image")

    return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)


def normalize_image(image: np.ndarray) -> np.ndarray:
    """Convert uint8 pixel values from the range 0–255 to float32 values 0–1."""
    _validate_image(image)

    return image.astype(np.float32) / 255.0
