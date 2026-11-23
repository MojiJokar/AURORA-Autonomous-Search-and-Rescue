from dataclasses import dataclass
import numpy as np


@dataclass(frozen=True)
class ImageFrame:
    """Represents a validated camera frame and its metadata."""

    image: np.ndarray
    timestamp: float
    frame_id: str

    def __post_init__(self):
        if not isinstance(self.image, np.ndarray):
            raise TypeError("image must be a NumPy array")

        if self.image.size == 0:
            raise ValueError("image cannot be empty")

        if self.image.dtype != np.uint8:
            raise ValueError("image must use uint8 data")

        if self.image.ndim not in (2, 3):
            raise ValueError("image must be grayscale or a color image")

        if self.image.ndim == 3 and self.image.shape[2] not in (1, 3, 4):
            raise ValueError("image must have 1, 3, or 4 channels")

        if not np.isfinite(self.timestamp) or self.timestamp < 0:
            raise ValueError("timestamp must be finite and non-negative")

        if not isinstance(self.frame_id, str) or not self.frame_id.strip():
            raise ValueError("frame_id must be a non-empty string")

    @property
    def height(self) -> int:
        return self.image.shape[0]

    @property
    def width(self) -> int:
        return self.image.shape[1]
