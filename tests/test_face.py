from app.face import best_profile_match, normalize_embedding, stable_identity, validate_embeddings


def test_face_matching_requires_a_clear_runner_up_margin():
    probe = normalize_embedding([1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
    profiles = [
        {"profile_id": "maria", "embeddings": [[1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]] * 3},
        {"profile_id": "joan", "embeddings": [[0.99, 0.01, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]] * 3},
    ]
    assert best_profile_match(probe, profiles) is None


def test_face_matching_can_match_and_identity_requires_three_stable_hits():
    probe = normalize_embedding([1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
    profiles = [
        {"profile_id": "maria", "care_recipient_id": "recipient-1", "embeddings": [probe] * 3},
        {"profile_id": "joan", "care_recipient_id": "recipient-2", "embeddings": [[0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]] * 3},
    ]
    candidate = best_profile_match(probe, profiles)
    assert candidate and candidate["profile_id"] == "maria"

    state = {}
    key = ("home-1", "camera-1", 7)
    assert stable_identity(state, key, candidate) is None
    assert stable_identity(state, key, candidate) is None
    stable = stable_identity(state, key, candidate)
    assert stable and stable["care_recipient_id"] == "recipient-1"


def test_face_enrollment_rejects_malformed_or_inconsistent_templates():
    assert len(validate_embeddings([[1.0] + [0.0] * 7] * 3)) == 3
    try:
        validate_embeddings([[1.0] + [0.0] * 7, [1.0] + [0.0] * 6, [1.0] + [0.0] * 7])
    except ValueError as error:
        assert "invalid dimension" in str(error)
    else:  # pragma: no cover - assertion guard
        raise AssertionError("inconsistent face template dimensions were accepted")
