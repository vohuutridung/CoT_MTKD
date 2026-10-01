import unittest

from cot_mtkd.evaluation.metrics import pass_metrics


class PassMetricTests(unittest.TestCase):
    def test_pass_at_1_averages_all_three_samples(self) -> None:
        records = [
            {"correct": [True, False, False]},
            {"correct": [False, False, False]},
            {"correct": [True, True, False]},
        ]
        metrics = pass_metrics(records)
        self.assertEqual(metrics["problems"], 3)
        self.assertAlmostEqual(metrics["pass_at_1"], (1 / 3 + 0 + 2 / 3) / 3)
        self.assertAlmostEqual(metrics["pass_at_1"], 3 / (3 * 3))
        self.assertAlmostEqual(metrics["pass_at_3"], 2 / 3)


if __name__ == "__main__":
    unittest.main()
