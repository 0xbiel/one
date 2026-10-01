import math

from app.face import (
    best_profile_match,
    cancelable_template,
    new_protected_template_payload,
    protected_template_payload,
    normalize_embedding,
    stable_identity,
    validate_embeddings,
)


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


def test_cancelable_template_is_revocable_and_not_the_raw_embedding():
    embedding = normalize_embedding([1.0] + [0.0] * 15)
    first = new_protected_template_payload([embedding] * 3, secret=b"a" * 32, profile_id="profile")
    second = new_protected_template_payload([embedding] * 3, secret=b"a" * 32, profile_id="profile")
    assert first["scheme"] == "cancelable-random-projection-v1"
    assert first["transform_salt"] != second["transform_salt"]
    parsed = protected_template_payload(first)
    assert parsed is not None
    templates, salt = parsed
    assert salt == first["transform_salt"]
    assert templates[0] != embedding
    profile = {"profile_id": "profile", "embeddings": templates, "transform_salt": salt}
    assert best_profile_match(embedding, [profile], transform_secret=b"a" * 32) is not None
    recomputed = cancelable_template(embedding, secret=b"a" * 32, profile_id="profile", transform_salt=salt)
    assert all(math.isclose(left, right, rel_tol=1e-12, abs_tol=1e-12) for left, right in zip(recomputed, templates[0]))
