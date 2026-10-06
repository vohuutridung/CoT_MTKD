import math
import unittest

from cot_mtkd.stage2.cot_prune import (
    ScoreCache,
    binary_search_prefix,
    candidate_token_ids,
    fidelity_threshold,
    sample_compression,
    token_cut_summary,
)
from cot_mtkd.stage2.cot_prune_run import reasoning_step_texts


def _prefix_score(valid_from: int):
    def score(kept: tuple[int, ...]) -> float:
        if kept != tuple(range(len(kept))):
            raise AssertionError(f"non-prefix scored: {kept}")
        return 0.0 if len(kept) >= valid_from else -1.0

    return score


class TokenCutSummaryTests(unittest.TestCase):
    def test_mean_is_unweighted_and_overall_weights_by_length(self) -> None:
        summary = token_cut_summary([(100, 20), (50, 25), (0, 0)])
        self.assertAlmostEqual(summary["mean_token_cut_percent"], (20.0 + 50.0 + 0.0) / 3)
        self.assertAlmostEqual(summary["overall_token_cut_percent"], 100.0 * 45 / 150)
        self.assertEqual(summary["total_deleted_reasoning_tokens"], 45.0)

    def test_deleted_tokens_cannot_exceed_the_original(self) -> None:
        with self.assertRaises(ValueError):
            token_cut_summary([(4, 5)])


class ReasoningSplitTests(unittest.TestCase):
    def test_blank_lines_are_the_step_boundaries(self) -> None:
        self.assertEqual(
            reasoning_step_texts("alpha\n\nbeta\n\n\ngamma", r"\n\n+"),
            ["alpha", "beta", "gamma"],
        )


class BinaryPrefixPruneTests(unittest.TestCase):
    def test_original_is_kept_when_nothing_shorter_is_valid(self) -> None:
        result = binary_search_prefix(8, _prefix_score(8), 0.95)
        self.assertEqual(result.kept, tuple(range(8)))

    def test_shortest_valid_prefix_is_selected(self) -> None:
        result = binary_search_prefix(8, _prefix_score(4), 0.95)
        self.assertEqual(result.kept, tuple(range(4)))

    def test_score_evaluations_are_logarithmic(self) -> None:
        calls: list[tuple[int, ...]] = []

        def score(kept: tuple[int, ...]) -> float:
            calls.append(kept)
            return _prefix_score(20)(kept)

        num_steps = 64
        result = binary_search_prefix(num_steps, score, 0.95)
        self.assertEqual(result.kept, tuple(range(20)))
        self.assertEqual(result.num_score_evaluations, len(calls))
        self.assertLessEqual(len(calls), math.floor(math.log2(num_steps)) + 3)
        self.assertLess(len(calls), num_steps)

    def test_threshold_stays_at_the_original_score(self) -> None:
        def score(kept: tuple[int, ...]) -> float:
            return -0.01 * (8 - len(kept))

        result = binary_search_prefix(8, score, 0.95)
        self.assertLess(len(result.kept), 8)
        self.assertEqual(result.threshold, result.original_score + math.log(0.95))
        self.assertEqual(result.threshold, fidelity_threshold(result.original_score, 0.95))
        self.assertNotEqual(result.threshold, result.final_score + math.log(0.95))

    def test_every_candidate_keeps_the_original_prompt(self) -> None:
        prompt = [7, 8, 9]
        steps = [[10], [11, 11], [12], [13]]
        for length in range(0, len(steps) + 1):
            sequence = candidate_token_ids(prompt, steps, tuple(range(length)), [4], [5, 6])
            self.assertEqual(sequence[: len(prompt)], prompt)

    def test_every_candidate_scores_the_same_answer_target(self) -> None:
        solution = [5, 6]
        steps = [[10], [11, 11], [12], [13]]
        for length in range(0, len(steps) + 1):
            sequence = candidate_token_ids([7, 8, 9], steps, tuple(range(length)), [4], solution)
            self.assertEqual(sequence[-len(solution) :], solution)

    def test_eta_one_rejects_any_drop(self) -> None:
        def score(kept: tuple[int, ...]) -> float:
            return 0.0 if len(kept) == 4 else -1.0e-9

        self.assertEqual(binary_search_prefix(4, score, 1.0).kept, tuple(range(4)))

        result = binary_search_prefix(4, lambda kept: 0.0, 1.0, min_steps=1)
        self.assertEqual(result.kept, (0,))

    def test_one_and_two_step_traces_without_a_shorter_prefix(self) -> None:
        self.assertEqual(binary_search_prefix(1, _prefix_score(1), 0.95).kept, (0,))
        self.assertEqual(binary_search_prefix(2, _prefix_score(2), 0.95).kept, (0, 1))
        self.assertEqual(binary_search_prefix(0, lambda kept: 0.0, 0.95, min_steps=1).kept, ())

    def test_repeated_prefix_does_not_call_the_scorer_again(self) -> None:
        calls = 0

        def raw(kept: tuple[int, ...]) -> float:
            nonlocal calls
            calls += 1
            return 0.0 if len(kept) >= 2 else -1.0

        cached = ScoreCache(raw)
        first = binary_search_prefix(4, cached, 0.95)
        second = binary_search_prefix(4, cached, 0.95)
        self.assertEqual(second.num_score_evaluations, 0)
        self.assertEqual(calls, first.num_score_evaluations)
        self.assertGreater(len(second.evaluated_prefix_lengths), 0)

    def test_final_prefix_meets_the_fidelity_bound(self) -> None:
        eta = 0.95
        result = binary_search_prefix(8, _prefix_score(4), eta)
        self.assertGreaterEqual(result.final_score, result.original_score + math.log(eta))
        metrics = sample_compression(
            8, len(result.kept), 80, 40, result.original_score, result.final_score
        )
        self.assertGreaterEqual(metrics["fidelity"], eta - 1e-12)
        self.assertAlmostEqual(metrics["step_reduction"], 0.5)
        self.assertAlmostEqual(metrics["token_reduction"], 0.5)


if __name__ == "__main__":
    unittest.main()
