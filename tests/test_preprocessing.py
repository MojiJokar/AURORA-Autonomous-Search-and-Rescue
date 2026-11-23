import numpy as np
import pytest

from aurora.perception.preprocessing import (
    resize_image,
    to_rgb,
    to_grayscale,
    normalize_image,
)


def test_resize_image():
    image = np.zeros((240, 320, 3), dtype=np.uint8)

    result = resize_image(image, 160, 120)

    assert result.shape == (120, 160, 3)
    assert result.dtype == np.uint8


def test_resize_rejects_invalid_dimensions():
    image = np.zeros((100, 100, 3), dtype=np.uint8)

    with pytest.raises(ValueError):
        resize_image(image, 0, 100)


def test_resize_rejects_non_integer_dimensions():
    image = np.zeros((100, 100, 3), dtype=np.uint8)

    with pytest.raises(TypeError):
        resize_image(image, 100.5, 100)


def test_bgr_to_rgb_conversion():
    image = np.zeros((10, 10, 3), dtype=np.uint8)
    image[0, 0] = [10, 20, 30]

    result = to_rgb(image)

    assert result[0, 0].tolist() == [30, 20, 10]


def test_grayscale_conversion():
    image = np.zeros((10, 10, 3), dtype=np.uint8)

    result = to_grayscale(image)

    assert result.shape == (10, 10)
    assert result.dtype == np.uint8


def test_existing_grayscale_image_is_copied():
    image = np.zeros((10, 10), dtype=np.uint8)

    result = to_grayscale(image)

    assert result.shape == image.shape
    assert result is not image


def test_normalization():
    image = np.array([[0, 127, 255]], dtype=np.uint8)

    result = normalize_image(image)

    assert result.dtype == np.float32
    assert result[0, 0] == 0.0
    assert result[0, 2] == 1.0
    assert 0.0 <= result.min() <= result.max() <= 1.0


def test_original_image_is_not_modified():
    image = np.zeros((10, 10, 3), dtype=np.uint8)
    image[0, 0] = [10, 20, 30]
    original = image.copy()

    to_rgb(image)
    normalize_image(image)
    resize_image(image, 5, 5)

    assert np.array_equal(image, original)


def test_invalid_image_dtype_is_rejected():
    image = np.zeros((10, 10, 3), dtype=np.float32)

    with pytest.raises(ValueError):
        normalize_image(image)


def test_empty_image_is_rejected():
    image = np.array([], dtype=np.uint8)

    with pytest.raises(ValueError):
        to_grayscale(image)


def test_rgb_rejects_grayscale_image():
    image = np.zeros((10, 10), dtype=np.uint8)

    with pytest.raises(ValueError):
        to_rgb(image)


def test_rgb_rejects_four_channel_image():
    image = np.zeros((10, 10, 4), dtype=np.uint8)

    with pytest.raises(ValueError):
        to_rgb(image)
