import unittest

from cot_mtkd.stage2.cot_prune import (
    ScoreCache,
    candidate_token_ids,
    hierarchical_group_prune,
    sample_compression,
    token_cut_summary,
    within_nll_budget,
)
from cot_mtkd.stage2.cot_prune_run import reasoning_step_texts


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


def _from_nll(table: dict[tuple[int, ...], float]):
    def score(kept: tuple[int, ...]) -> float:
        if kept != tuple(sorted(kept)):
            raise AssertionError(f"steps were reordered: {kept}")
        return -table[kept]

    return score


class NllBudgetTests(unittest.TestCase):
    def test_accepts_a_deletion_inside_the_budget(self) -> None:
        self.assertTrue(within_nll_budget(10.04, 10.0, 0.05))
        result = hierarchical_group_prune(
            2,
            _from_nll({(0, 1): 10.0, (1,): 10.04, (0,): 11.0}),
            0.05,
        )
        self.assertEqual(result.kept, (1,))
        self.assertAlmostEqual(result.final_nll, 10.04)
        self.assertAlmostEqual(result.nll_delta_from_original, 0.04)

    def test_rejects_a_deletion_past_the_budget(self) -> None:
        self.assertFalse(within_nll_budget(10.06, 10.0, 0.05))
        result = hierarchical_group_prune(
            2,
            _from_nll({(0, 1): 10.0, (1,): 10.06, (0,): 10.06}),
            0.05,
        )
        self.assertEqual(result.kept, (0, 1))
        self.assertAlmostEqual(result.final_nll, 10.0)
        self.assertAlmostEqual(result.nll_delta_from_original, 0.0)

    def test_accepts_an_nll_improvement(self) -> None:
        self.assertTrue(within_nll_budget(9.0, 10.0, 0.05))
        result = hierarchical_group_prune(
            2,
            _from_nll({(0, 1): 10.0, (1,): 9.0, (0,): 12.0}),
            0.05,
        )
        self.assertEqual(result.kept, (1,))
        self.assertAlmostEqual(result.final_nll, 9.0)
        self.assertAlmostEqual(result.nll_delta_from_original, -1.0)

    def test_budget_is_measured_from_the_current_trace(self) -> None:
        # 10.07 exceeds the original budget of 10.05, but fits 10.03 + 0.05.
        score = _from_nll(
            {
                (0, 1, 2, 3): 10.0,
                (2, 3): 10.03,
                (0, 1): 12.0,
                (3,): 10.07,
                (2,): 12.0,
                (0, 2, 3): 12.0,
                (1, 2, 3): 12.0,
                (0, 1, 2): 12.0,
                (0, 1, 3): 12.0,
            }
        )
        result = hierarchical_group_prune(4, score, 0.05)
        self.assertEqual(result.kept, (3,))
        self.assertAlmostEqual(result.final_nll, 10.07)
        self.assertAlmostEqual(result.nll_delta_from_original, 0.07)
        self.assertGreater(result.nll_delta_from_original, result.nll_budget)

    def test_a_rejected_large_group_still_allows_a_better_subgroup(self) -> None:
        score = _from_nll(
            {
                (0, 1, 2, 3): 10.0,
                (): 20.0,
                (2, 3): 9.5,
                (0, 1): 15.0,
                (2,): 15.0,
                (3,): 15.0,
                (0,): 15.0,
                (1,): 15.0,
                (0, 2, 3): 15.0,
                (1, 2, 3): 15.0,
                (0, 1, 2): 15.0,
                (0, 1, 3): 15.0,
            }
        )
        result = hierarchical_group_prune(4, score, 0.05, min_steps=0)
        self.assertEqual(result.kept, (2, 3))
        self.assertAlmostEqual(result.final_nll, 9.5)
        self.assertLess(result.nll_delta_from_original, 0.0)

    def test_min_steps_blocks_an_empty_trace(self) -> None:
        calls: list[tuple[int, ...]] = []

        def score(kept: tuple[int, ...]) -> float:
            calls.append(kept)
            return 0.0

        result = hierarchical_group_prune(4, score, 0.05, min_steps=1)
        self.assertGreaterEqual(len(result.kept), 1)
        self.assertNotIn((), calls)

    def test_every_candidate_keeps_the_original_prompt(self) -> None:
        prompt = [7, 8, 9]
        steps = [[10], [11], [12], [13]]
        for kept in ((0, 1, 2, 3), (0, 3), (1, 3), (2,)):
            sequence = candidate_token_ids(prompt, steps, kept, [4], [5, 6])
            self.assertEqual(sequence[: len(prompt)], prompt)

    def test_every_candidate_scores_the_same_answer_target(self) -> None:
        solution = [5, 6]
        steps = [[10], [11], [12], [13]]
        for kept in ((0, 1, 2, 3), (0, 3), (1,), ()):
            sequence = candidate_token_ids([7, 8, 9], steps, kept, [4], solution)
            self.assertEqual(sequence[-len(solution) :], solution)

    def test_deleted_steps_keep_their_original_indices(self) -> None:
        score = _from_nll(
            {
                (0, 1, 2, 3): 10.0,
                (0, 2, 3): 10.01,
                (0, 3): 10.02,
                (1, 2, 3): 12.0,
                (0, 1, 2): 12.0,
                (0, 1, 3): 12.0,
                (0, 1): 12.0,
                (2, 3): 12.0,
                (0,): 12.0,
                (3,): 12.0,
                (0, 2): 12.0,
            }
        )
        result = hierarchical_group_prune(4, score, 0.05)
        self.assertEqual(result.kept, (0, 3))
        self.assertEqual(result.deleted_indices, (1, 2))

    def test_repeated_subset_uses_the_cache(self) -> None:
        calls = 0

        def raw(kept: tuple[int, ...]) -> float:
            nonlocal calls
            calls += 1
            return 0.0 if len(kept) == 4 else -1.0

        cached = ScoreCache(raw)
        first = hierarchical_group_prune(4, cached, 0.05)
        second = hierarchical_group_prune(4, cached, 0.05)
        self.assertEqual(second.num_score_evaluations, 0)
        self.assertGreater(second.cache_hits, 0)
        self.assertEqual(calls, first.num_score_evaluations)

    def test_batched_scores_match_sequential_scores(self) -> None:
        def raw(kept: tuple[int, ...]) -> float:
            if kept == (0, 1, 2, 3):
                return -10.0
            if kept == (2, 3):
                return -10.01
            return -12.0

        batches: list[int] = []

        def score_many(keys: list[tuple[int, ...]]) -> list[float]:
            batches.append(len(keys))
            return [raw(key) for key in keys]

        batched = hierarchical_group_prune(4, raw, 0.05, score_many=score_many)
        sequential = hierarchical_group_prune(4, raw, 0.05)
        self.assertEqual(batched.kept, sequential.kept)
        self.assertEqual(batched.final_nll, sequential.final_nll)
        self.assertGreater(max(batches), 1)

    def test_three_scripted_samples_report_nll_and_compression(self) -> None:
        def run(num_steps: int, table: dict[tuple[int, ...], float]):
            def score(kept: tuple[int, ...]) -> float:
                return -table.get(kept, 100.0)

            result = hierarchical_group_prune(num_steps, score, 0.05)
            ratio = sample_compression(
                num_steps,
                len(result.kept),
                10 * num_steps,
                10 * len(result.kept),
                -result.original_nll,
                -result.final_nll,
            )["token_reduction"]
            self.assertGreaterEqual(len(result.kept), 1)
            return result, ratio

        kept_half, half_ratio = run(4, {(0, 1, 2, 3): 8.0, (2, 3): 8.04})
        improved, improved_ratio = run(2, {(0, 1): 10.0, (1,): 9.2})
        unchanged, unchanged_ratio = run(
            3, {(0, 1, 2): 4.0, (1, 2): 4.06, (0, 2): 4.06, (0, 1): 4.06}
        )
        self.assertEqual(kept_half.kept, (2, 3))
        self.assertAlmostEqual(kept_half.nll_delta_from_original, 0.04)
        self.assertAlmostEqual(half_ratio, 0.5)
        self.assertEqual(improved.kept, (1,))
        self.assertAlmostEqual(improved.nll_delta_from_original, -0.8)
        self.assertAlmostEqual(improved_ratio, 0.5)
        self.assertEqual(unchanged.kept, (0, 1, 2))
        self.assertAlmostEqual(unchanged.nll_delta_from_original, 0.0)
        self.assertAlmostEqual(unchanged_ratio, 0.0)


if __name__ == "__main__":
    unittest.main()
