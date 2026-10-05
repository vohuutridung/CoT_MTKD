import unittest

from cot_mtkd.evaluation.metrics import pass_metrics
from cot_mtkd.evaluation.runner import generation_token_limit


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

    def test_pass_metrics_reject_a_record_without_three_samples(self) -> None:
        with self.assertRaises(ValueError):
            pass_metrics([{"correct": [True]}])


class GenerationLimitTests(unittest.TestCase):
    def test_completion_is_capped_by_the_model_length(self) -> None:
        self.assertEqual(generation_token_limit(100, 4096, 4096), 3996)
        self.assertEqual(generation_token_limit(10, 4096, 64), 64)

    def test_prompt_that_fills_the_context_is_rejected(self) -> None:
        with self.assertRaises(RuntimeError):
            generation_token_limit(4096, 4096, 4096)


if __name__ == "__main__":
    unittest.main()
