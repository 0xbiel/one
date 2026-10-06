from geometry_service.face import active_liveness_challenge


def test_active_liveness_challenge_requires_opposite_side_motion():
    assert active_liveness_challenge([0.02, 0.29, -0.28])['status'] == 'passed'
    failed = active_liveness_challenge([-0.44, -0.43, -0.45])
    assert failed['status'] == 'failed'
    assert failed['yaw_span'] < 0.32


def test_active_liveness_challenge_requires_centered_first_sample():
    failed = active_liveness_challenge([0.31, 0.0, -0.31])

    assert failed['status'] == 'failed'
    assert failed['center_seen'] is False
    assert 'first_sample_not_centered' in failed['failure_reasons']


def test_active_liveness_challenge_requires_ordered_opposite_turn():
    failed = active_liveness_challenge([0.01, 0.28, 0.31, 0.27])

    assert failed['status'] == 'failed'
    assert failed['ordered_motion'] is False
    assert 'ordered_side_to_opposite_motion_missing' in failed['failure_reasons']
