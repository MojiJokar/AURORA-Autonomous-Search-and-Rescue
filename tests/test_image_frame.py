import numpy as np
import pytest

from aurora.perception.image_frame import ImageFrame


def test_valid_color_frame():
    image = np.zeros((240, 320, 3), dtype=np.uint8)

    frame = ImageFrame(
        image=image,
        timestamp=1.5,
        frame_id="camera_01",
    )

    assert frame.width == 320
    assert frame.height == 240
    assert frame.frame_id == "camera_01"
    assert frame.timestamp == 1.5


def test_valid_grayscale_frame():
    image = np.zeros((240, 320), dtype=np.uint8)

    frame = ImageFrame(image, 2.0, "camera_01")

    assert frame.width == 320
    assert frame.height == 240


def test_empty_image_is_rejected():
    with pytest.raises(ValueError):
        ImageFrame(np.array([]), 1.0, "camera_01")


def test_invalid_image_type_is_rejected():
    with pytest.raises(TypeError):
        ImageFrame("not an image", 1.0, "camera_01")


def test_invalid_image_dtype_is_rejected():
    image = np.zeros((240, 320, 3), dtype=np.float32)

    with pytest.raises(ValueError):
        ImageFrame(image, 1.0, "camera_01")


def test_invalid_timestamp_is_rejected():
    image = np.zeros((240, 320, 3), dtype=np.uint8)

    with pytest.raises(ValueError):
        ImageFrame(image, -1.0, "camera_01")


def test_empty_frame_id_is_rejected():
    image = np.zeros((240, 320, 3), dtype=np.uint8)

    with pytest.raises(ValueError):
        ImageFrame(image, 1.0, "")
