import unittest

import pandas as pd

from src.fuzzygen_bpd import FEATURES, score_candidates, validate_candidates
from src.uploaded_figures import (
    LABEL_COLUMN,
    generate_uploaded_figures,
    validate_acceptability_labels,
)


class UploadedFiguresTests(unittest.TestCase):
    def setUp(self):
        rows = []
        for vulnerability_index in range(2):
            for candidate_index in range(8):
                acceptable = candidate_index % 2
                rows.append(
                    (
                        f"V{vulnerability_index}",
                        f"C{candidate_index}",
                        0.1 + candidate_index * 0.08,
                        0.01 + candidate_index * 0.015,
                        0.03 + candidate_index * 0.06,
                        True,
                        True,
                        True,
                        bool(acceptable),
                    )
                )
        self.data = pd.DataFrame(
            rows,
            columns=[
                "vulnerability_id",
                "candidate_id",
                *FEATURES,
                "build_pass",
                "pov_pass",
                "functional_pass",
                LABEL_COLUMN,
            ],
        )

    def test_requires_binary_labels(self):
        with self.assertRaisesRegex(ValueError, "must include"):
            validate_acceptability_labels(self.data.drop(columns=LABEL_COLUMN))

        invalid = self.data.astype({LABEL_COLUMN: object})
        invalid.loc[0, LABEL_COLUMN] = "unknown"
        with self.assertRaisesRegex(ValueError, "true/false or 1/0"):
            validate_acceptability_labels(invalid)

    def test_generates_upload_specific_figure_collection(self):
        data = validate_candidates(validate_acceptability_labels(self.data))
        results = score_candidates(data, (0.6, 0.2, 0.2), 0.3)
        figures = generate_uploaded_figures(results, (0.6, 0.2, 0.2))

        self.assertEqual(len(figures), 14)
        for title in (
            "Fig. 2 · Fuzzy Membership Profiles of Patch Disturbance Features",
            "Fig. 3 · Fuzzy Inference and Disturbance Score Transformation",
            "Fig. 4 · Impact of Feature Weights on Candidate Patch Ranking",
            "Fig. 5 · Convergence Behaviour of Hyperparameter Optimization Algorithms",
            "Fig. 6 · Comparison of Optimized Feature Weights Across Algorithms",
            "Fig. 7 · Performance–Consistency Trade-off in Weight Optimization",
        ):
            with self.subTest(title=title):
                self.assertIn(title, figures)
                self.assertGreater(len(figures[title][1].data), 0)

    def test_requires_both_labels_among_eligible_candidates(self):
        data = validate_candidates(validate_acceptability_labels(self.data))
        data.loc[data[LABEL_COLUMN] == 0, "build_pass"] = False
        results = score_candidates(data, (0.6, 0.2, 0.2), 0.3)
        with self.assertRaisesRegex(ValueError, "both `patch_acceptable` classes"):
            generate_uploaded_figures(results, (0.6, 0.2, 0.2))


if __name__ == "__main__":
    unittest.main()
