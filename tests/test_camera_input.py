from unittest.mock import patch

import cv2
import numpy as np
import pytest

from aurora.perception.camera_input import load_image, iter_video_frames
from aurora.perception.image_frame import ImageFrame


def test_load_image_success(tmp_path):
    image_path = tmp_path / "test.jpg"
    image_path.touch()

    image = np.zeros((120, 160, 3), dtype=np.uint8)

    with patch("aurora.perception.camera_input.cv2.imread", return_value=image):
        frame = load_image(image_path, timestamp=12.5)

    assert isinstance(frame, ImageFrame)
    assert frame.width == 160
    assert frame.height == 120
    assert frame.timestamp == 12.5
    assert frame.frame_id == "image_000001"


def test_load_image_missing_file(tmp_path):
    image_path = tmp_path / "missing.jpg"

    with pytest.raises(FileNotFoundError):
        load_image(image_path)


def test_load_image_unreadable_file(tmp_path):
    image_path = tmp_path / "unreadable.jpg"
    image_path.touch()

    with patch("aurora.perception.camera_input.cv2.imread", return_value=None):
        with pytest.raises(ValueError):
            load_image(image_path)


class FakeVideoCapture:
    """Small fake video source for tests without a real video codec."""

    def __init__(self):
        self.frames = [
            np.zeros((120, 160, 3), dtype=np.uint8),
            np.ones((120, 160, 3), dtype=np.uint8),
            np.full((120, 160, 3), 2, dtype=np.uint8),
        ]
        self.index = 0
        self.released = False

    def isOpened(self):
        return True

    def get(self, property_id):
        if property_id == cv2.CAP_PROP_FPS:
            return 2.0

        if property_id == cv2.CAP_PROP_POS_MSEC:
            return 0.0

        return 0.0

    def read(self):
        if self.index >= len(self.frames):
            return False, None

        frame = self.frames[self.index]
        self.index += 1
        return True, frame

    def release(self):
        self.released = True


def test_iter_video_frames_returns_frames_and_releases_capture(tmp_path):
    video_path = tmp_path / "test_video.avi"
    video_path.touch()

    fake_capture = FakeVideoCapture()

    with patch(
        "aurora.perception.camera_input.cv2.VideoCapture",
        return_value=fake_capture,
    ):
        frames = list(iter_video_frames(video_path))

    assert len(frames) == 3
    assert all(isinstance(frame, ImageFrame) for frame in frames)

    assert [frame.frame_id for frame in frames] == [
        "video_000000",
        "video_000001",
        "video_000002",
    ]

    assert [frame.timestamp for frame in frames] == [
        0.0,
        0.5,
        1.0,
    ]

    assert frames[0].width == 160
    assert frames[0].height == 120
    assert fake_capture.released is True


def test_iter_video_frames_custom_prefix(tmp_path):
    video_path = tmp_path / "test_video.avi"
    video_path.touch()

    fake_capture = FakeVideoCapture()

    with patch(
        "aurora.perception.camera_input.cv2.VideoCapture",
        return_value=fake_capture,
    ):
        frames = list(iter_video_frames(video_path, frame_prefix="rescue"))

    assert frames[0].frame_id == "rescue_000000"


def test_iter_video_frames_missing_file(tmp_path):
    video_path = tmp_path / "missing.avi"

    with pytest.raises(FileNotFoundError):
        list(iter_video_frames(video_path))


def test_iter_video_frames_rejects_empty_prefix(tmp_path):
    video_path = tmp_path / "test_video.avi"
    video_path.touch()

    with pytest.raises(ValueError):
        list(iter_video_frames(video_path, frame_prefix=" "))


def test_iter_video_frames_rejects_unopenable_video(tmp_path):
    video_path = tmp_path / "invalid.avi"
    video_path.touch()

    fake_capture = FakeVideoCapture()
    fake_capture.isOpened = lambda: False

    with patch(
        "aurora.perception.camera_input.cv2.VideoCapture",
        return_value=fake_capture,
    ):
        with pytest.raises(OSError):
            list(iter_video_frames(video_path))

    assert fake_capture.released is True
