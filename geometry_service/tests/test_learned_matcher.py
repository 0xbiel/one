import unittest

import numpy as np

from geometry_service.learned_matcher import FEATURE_DIM, MODEL_VERSION, get_or_fit_matcher, pair_features


class LearnedMatcherTests(unittest.TestCase):
    def test_pair_features_are_bounded_and_deterministic(self) -> None:
        first = np.zeros((2, 32), dtype=np.uint8)
        second = np.zeros((2, 32), dtype=np.uint8)
        second[1, 0] = 255

        features = pair_features(first, second, [0.0, 0.02], [0.02, 0.0])

        self.assertEqual(features.shape, (2, FEATURE_DIM))
        self.assertTrue(np.isfinite(features).all())
        self.assertTrue((features >= 0.0).all())
        self.assertTrue((features <= 1.0).all())
        np.testing.assert_array_equal(features, pair_features(first, second, [0.0, 0.02], [0.02, 0.0]))

    def test_map_specific_model_can_score_nearby_landmarks(self) -> None:
        rng = np.random.default_rng(44)
        base_descriptors = rng.integers(0, 256, size=(15, 32), dtype=np.uint8)
        points: list[np.ndarray] = []
        descriptors: list[np.ndarray] = []
        view_ids: list[str] = []
        responses: list[float] = []
        for cluster in range(15):
            center = np.asarray([float(cluster) * 0.5, 0.0, 0.0], dtype=np.float32)
            for view in range(3):
                points.append(center + rng.normal(0.0, 0.006, size=3).astype(np.float32))
                descriptor = base_descriptors[cluster].copy()
                descriptor ^= rng.integers(0, 2, size=32, dtype=np.uint8) * np.uint8(3)
                descriptors.append(descriptor)
                view_ids.append(f"scan-view-{view}")
                responses.append(0.4 + 0.05 * view)

        matcher = get_or_fit_matcher(
            np.asarray(points, dtype=np.float32),
            np.asarray(descriptors, dtype=np.uint8),
            np.asarray(responses, dtype=np.float32),
            view_ids,
            seed=44,
        )

        self.assertIsNotNone(matcher)
        assert matcher is not None
        positive_scores = matcher.score_pairs(
            np.asarray(descriptors[0:3], dtype=np.uint8),
            np.asarray(descriptors[1:4], dtype=np.uint8),
            np.asarray(responses[0:3], dtype=np.float32),
            np.asarray(responses[1:4], dtype=np.float32),
        )
        negative_scores = matcher.score_pairs(
            np.asarray(descriptors[0:3], dtype=np.uint8),
            np.asarray(descriptors[30:33], dtype=np.uint8),
            np.asarray(responses[0:3], dtype=np.float32),
            np.asarray(responses[30:33], dtype=np.float32),
        )
        self.assertGreater(float(np.mean(positive_scores)), float(np.mean(negative_scores)))
        self.assertEqual(matcher.diagnostics["model_version"], MODEL_VERSION)


if __name__ == "__main__":
    unittest.main()
