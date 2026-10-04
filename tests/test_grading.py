from __future__ import annotations

import unittest

from cot_mtkd.evaluation.grading import HAS_MATH_VERIFY, extract_boxed, grade_answer


class GradingTest(unittest.TestCase):
    def test_nested_boxed_answer(self) -> None:
        self.assertEqual(extract_boxed(r"work \boxed{\frac{1}{2}}"), r"\frac{1}{2}")

    def test_numeric_fallback_handles_dataset_float(self) -> None:
        self.assertTrue(grade_answer(r"The result is \boxed{142}", 142.0))

    def test_last_number_when_unboxed(self) -> None:
        self.assertTrue(grade_answer("so the answer is 70", "70"))
        self.assertFalse(grade_answer("so the answer is 71", "70"))

    @unittest.skipUnless(HAS_MATH_VERIFY, "math-verify not installed")
    def test_raw_latex_gold_is_wrapped_before_parsing(self) -> None:
        # MATH-500 gold; rejected when parsed without \boxed{}.
        gold = r"\left( 3, \frac{\pi}{2} \right)"
        self.assertTrue(grade_answer(r"Thus \boxed{(3, \frac{\pi}{2})}", gold))
        self.assertTrue(grade_answer(r"\boxed{\left(3, \frac{\pi}{2}\right)}", gold))
        self.assertFalse(grade_answer(r"\boxed{(3, \pi)}", gold))

    @unittest.skipUnless(HAS_MATH_VERIFY, "math-verify not installed")
    def test_equivalent_fraction(self) -> None:
        self.assertTrue(grade_answer(r"\boxed{\dfrac{14}{3}}", r"\frac{14}{3}"))


if __name__ == "__main__":
    unittest.main()
