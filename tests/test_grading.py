from __future__ import annotations

import unittest

from cot_mtkd.evaluation.grading import extract_boxed, grade_math


class GradingTest(unittest.TestCase):
    def test_nested_boxed_answer(self) -> None:
        self.assertEqual(extract_boxed(r"work \boxed{\frac{1}{2}}"), r"\frac{1}{2}")

    def test_numeric_fallback_handles_dataset_float(self) -> None:
        self.assertTrue(
            grade_math(r"The result is \boxed{142}", 142.0, prefer_math_verify=False)
        )


if __name__ == "__main__":
    unittest.main()
