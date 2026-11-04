import unittest
from datetime import datetime
from src.aki.state import Patient
from src.aki.features import compute_features
from src.aki.config import FEATURE_ORDER


class TestFeatures(unittest.TestCase):
    def setUp(self):
        self.p = Patient(mrn="123", dob=datetime(1990, 1, 1), sex='F')

    def test_full_feature_vector_logic(self):
        """Verifies the features are correctly computed for the ML model."""
        # (a) add history
        self.p.add_result(datetime(2024, 1, 1), 100.0)
        self.p.add_result(datetime(2024, 1, 2), 120.0)

        # (b) add current result
        self.p.add_result(datetime(2024, 1, 4), 110.0)

        # (c) compute features
        # Returns a (1, 10) array
        feats = compute_features(self.p)[0]

        # Map values to names for clearer assertions
        f_map = dict(zip(FEATURE_ORDER, feats))

        # age (approx 36 in 2026)
        self.assertEqual(f_map["age"], 36.0)
        # sex (1.0 for Female)
        self.assertEqual(f_map["is_female"], 1.0)
        # baseline (min)
        self.assertEqual(f_map["baseline"], 100.0)
        # current
        self.assertEqual(f_map["current"], 110.0)
        # peak (max)
        self.assertEqual(f_map["peak"], 120.0)
        # volatility (std dev sample)
        self.assertAlmostEqual(f_map["volatility"], 10.0)
        # count
        self.assertEqual(f_map["count"], 3.0)
        # velocity (change per day from last result)
        # Delta: -10 units / 2 days = -5.0
        self.assertAlmostEqual(f_map["velocity"], -5.0)
        # ratio
        self.assertAlmostEqual(f_map["ratio"], 1.1)
        # absolute increase
        self.assertEqual(f_map["abs_increase"], 10.0)

    def test_single_value_stats(self):
        """Edge case: first result (no history)."""
        p = Patient(mrn="999", sex='M')
        p.add_result(datetime(2024, 1, 1), 50.0)

        feats = compute_features(p)[0]
        f_map = dict(zip(FEATURE_ORDER, feats))

        self.assertEqual(f_map["volatility"], 0.0)
        self.assertEqual(f_map["velocity"], 0.0)
        self.assertEqual(f_map["count"], 1.0)
