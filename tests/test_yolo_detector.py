import cv2
import numpy as np
import pytest

from aurora.perception.detection.yolo_detector import YOLOPersonDetector
from aurora.perception.image_frame import ImageFrame


class FakeNet:
    def __init__(self, output):
        self.output = output
        self.input = None
        self.backend = None
        self.target = None

    def setPreferableBackend(self, backend):
        self.backend = backend

    def setPreferableTarget(self, target):
        self.target = target

    def setInput(self, blob):
        self.input = blob

    def forward(self):
        return self.output


def make_output(person_confidence=0.90, other_confidence=0.0):

    output = np.zeros((1, 84, 1), dtype=np.float32)

    output[0, 0, 0] = 320.0
    output[0, 1, 0] = 240.0
    output[0, 2, 0] = 200.0
    output[0, 3, 0] = 200.0

    output[0, 4, 0] = person_confidence
    output[0, 5, 0] = other_confidence

    return output


@pytest.fixture
def detector(tmp_path, monkeypatch):

    model = tmp_path / "fake.onnx"
    model.write_bytes(b"test model")

    net = FakeNet(make_output())

    monkeypatch.setattr(
        cv2.dnn,
        "readNetFromONNX",
        lambda _: net,
    )

    instance = YOLOPersonDetector(model)

    return instance, net


def make_frame(width=640, height=480):

    image = np.zeros(
        (height, width, 3),
        dtype=np.uint8,
    )

    return ImageFrame(
        image=image,
        timestamp=1.0,
        frame_id="test-frame",
    )


def test_detector_uses_opencv_cpu_backend(detector):

    _, net = detector

    assert net.backend == cv2.dnn.DNN_BACKEND_OPENCV
    assert net.target == cv2.dnn.DNN_TARGET_CPU


def test_detect_returns_person_detection(detector):

    instance, _ = detector

    results = instance.detect(make_frame())

    assert len(results) == 1
    assert results[0].class_name == "person"
    assert results[0].confidence == pytest.approx(0.90)

    box = results[0].bounding_box

    assert box.x_min == pytest.approx(220.0)
    assert box.y_min == pytest.approx(140.0)
    assert box.x_max == pytest.approx(420.0)
    assert box.y_max == pytest.approx(340.0)


def test_detect_filters_low_confidence(detector):

    instance, net = detector

    net.output = make_output(person_confidence=0.10)

    assert instance.detect(make_frame()) == []


def test_detect_ignores_non_person_classes(detector):

    instance, net = detector

    net.output = make_output(
        person_confidence=0.0,
        other_confidence=0.99,
    )

    assert instance.detect(make_frame()) == []


def test_detect_rejects_wrong_frame_type(detector):

    instance, _ = detector

    with pytest.raises(TypeError):
        instance.detect(
            np.zeros((10, 10, 3), dtype=np.uint8)
        )


def test_detect_rejects_grayscale_frame(detector):

    instance, _ = detector

    frame = ImageFrame(
        image=np.zeros((20, 20), dtype=np.uint8),
        timestamp=1.0,
        frame_id="gray",
    )

    with pytest.raises(ValueError, match="3-channel"):
        instance.detect(frame)


def test_detector_rejects_missing_model(tmp_path):

    with pytest.raises(FileNotFoundError):
        YOLOPersonDetector(tmp_path / "missing.onnx")


def test_detector_rejects_invalid_input_size(tmp_path, monkeypatch):

    model = tmp_path / "fake.onnx"
    model.write_bytes(b"test model")

    monkeypatch.setattr(
        cv2.dnn,
        "readNetFromONNX",
        lambda _: FakeNet(make_output()),
    )

    with pytest.raises(ValueError):
        YOLOPersonDetector(
            model,
            input_size=(0, 480),
        )


def test_detector_rejects_invalid_output_shape(detector):

    instance, net = detector

    net.output = np.zeros(
        (1, 10, 10),
        dtype=np.float32,
    )

    with pytest.raises(ValueError, match="84 values"):
        instance.detect(make_frame())


def test_input_blob_has_expected_shape(detector):

    instance, net = detector

    instance.detect(make_frame())

    assert net.input.shape == (1, 3, 480, 640)
