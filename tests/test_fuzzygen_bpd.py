import unittest

import numpy as np
import pandas as pd

from src.fuzzygen_bpd import (
    FEATURES,
    normalized_weights,
    score_candidates,
    validate_candidates,
)


class FuzzyGenBpdTests(unittest.TestCase):
    def setUp(self):
        self.data = pd.DataFrame(
            [
                ("V1", "A", 0.1, 0.02, 0.05, True, True, True),
                ("V1", "B", 0.2, 0.03, 0.1, False, True, True),
                ("V2", "A", 0.4, 0.1, 0.4, True, True, True),
            ],
            columns=[
                "vulnerability_id",
                "candidate_id",
                *FEATURES,
                "build_pass",
                "pov_pass",
                "functional_pass",
            ],
        )

    def test_rejects_empty_or_whitespace_identifiers(self):
        for column, value in (
            ("vulnerability_id", ""),
            ("vulnerability_id", "  "),
            ("candidate_id", ""),
            ("candidate_id", "\t"),
        ):
            with self.subTest(column=column, value=repr(value)):
                data = self.data.astype(object).copy()
                data.loc[0, column] = value
                with self.assertRaisesRegex(ValueError, "identifiers cannot be blank"):
                    validate_candidates(data)

    def test_rejects_missing_identifiers(self):
        data = self.data.copy()
        data.loc[0, "candidate_id"] = None
        with self.assertRaisesRegex(ValueError, "identifiers cannot be blank"):
            validate_candidates(data)

    def test_normalizes_nonnegative_finite_weights(self):
        weights = normalized_weights((0.6, 0.2, 0.2))
        self.assertAlmostEqual(sum(weights), 1.0)
        self.assertEqual(weights, (0.6, 0.2, 0.2))

    def test_rejects_negative_nonfinite_or_zero_weights(self):
        for weights in (
            (-0.1, 0.6, 0.5),
            (np.nan, 0.5, 0.5),
            (np.inf, 0.5, 0.5),
            (0.0, 0.0, 0.0),
        ):
            with self.subTest(weights=weights):
                with self.assertRaises(ValueError):
                    normalized_weights(weights)

    def test_rejects_wrong_number_of_weights(self):
        with self.assertRaisesRegex(ValueError, "exactly 3"):
            normalized_weights((0.5, 0.5))

    def test_scoring_preserves_gate_and_grouped_rank_behavior(self):
        validated = validate_candidates(self.data)
        results = score_candidates(validated, (0.6, 0.2, 0.2), 0.3)

        self.assertEqual(len(results), 3)
        self.assertEqual(int(results["eligible"].sum()), 2)
        blocked = results.loc[results["candidate_id"] == "B"].iloc[0]
        self.assertFalse(blocked["eligible"])
        self.assertTrue(pd.isna(blocked["BPD"]))
        self.assertEqual(
            int(results.loc[results["vulnerability_id"] == "V2", "rank"].iloc[0]),
            1,
        )
        with self.assertRaisesRegex(ValueError, "cannot be negative"):
            score_candidates(validated, (-0.1, 0.6, 0.5), 0.3)


if __name__ == "__main__":
    unittest.main()
