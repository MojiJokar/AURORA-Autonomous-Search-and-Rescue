from types import SimpleNamespace

import pytest

from aurora.autonomy.target_assessment import assess_track, assess_tracks


def make_track(
    track_id=1,
    class_name="person",
    confidence=0.9,
    hits=10,
    missed_frames=0,
    observed=True,
    box=None,
):
    if box is None:
        box = SimpleNamespace(x1=100, y1=50, x2=300, y2=350)

    return SimpleNamespace(
        track_id=track_id,
        class_name=class_name,
        confidence=confidence,
        hits=hits,
        missed_frames=missed_frames,
        observed=observed,
        bounding_box=box,
    )


def test_person_track_is_assessed():
    result = assess_track(make_track(), 640, 480)

    assert result.track_id == 1
    assert result.status == "PERSON_CANDIDATE"
    assert result.priority in {"LOW", "MEDIUM", "HIGH"}


def test_high_confidence_persistent_track_gets_high_priority():
    result = assess_track(
        make_track(confidence=0.95, hits=10),
        640,
        480,
    )

    assert result.priority == "HIGH"
    assert result.score >= 0.70


def test_non_person_is_not_a_rescue_candidate():
    result = assess_track(
        make_track(class_name="car"),
        640,
        480,
    )

    assert result.status == "NOT_PERSON"
    assert result.priority == "NONE"


def test_unobserved_track_is_not_visible():
    result = assess_track(
        make_track(observed=False, missed_frames=1),
        640,
        480,
    )

    assert result.status == "NOT_VISIBLE"
    assert result.priority == "NONE"


def test_low_confidence_track_has_low_priority():
    result = assess_track(
        make_track(confidence=0.2, hits=1),
        640,
        480,
    )

    assert result.priority == "LOW"


def test_score_is_between_zero_and_one():
    result = assess_track(make_track(), 640, 480)

    assert 0.0 <= result.score <= 1.0


def test_persistence_is_capped_at_one():
    result = assess_track(
        make_track(hits=100),
        640,
        480,
    )

    assert result.persistence == 1.0


def test_invalid_frame_dimensions_are_rejected():
    with pytest.raises(ValueError):
        assess_track(make_track(), 0, 480)


def test_tracks_are_sorted_by_priority():
    tracks = [
        make_track(track_id=1, confidence=0.2, hits=1),
        make_track(track_id=2, confidence=0.95, hits=10),
        make_track(track_id=3, confidence=0.7, hits=6),
    ]

    results = assess_tracks(tracks, 640, 480)

    assert results[0].priority == "HIGH"
    assert results[-1].priority == "LOW"


def test_assessment_contains_explanations():
    result = assess_track(make_track(), 640, 480)

    assert len(result.reasons) >= 3
