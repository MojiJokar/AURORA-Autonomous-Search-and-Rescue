"""Confidence score validation for object detection."""

from math import isfinite
from numbers import Real


def validate_confidence(confidence: float) -> float:
    """Validate a confidence score in the inclusive range 0.0 to 1.0."""
    if isinstance(confidence, bool) or not isinstance(confidence, Real):
        raise TypeError("confidence must be a real number")

    if not isfinite(confidence):
        raise ValueError("confidence must be finite")

    if not 0.0 <= confidence <= 1.0:
        raise ValueError("confidence must be between 0.0 and 1.0")

    return float(confidence)
