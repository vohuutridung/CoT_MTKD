import math
import unittest

from cot_mtkd.stage2.cot_prune import (
    ScoreCache,
    candidate_token_ids,
    fidelity_threshold,
    hierarchical_group_prune,
    sample_compression,
    token_cut_summary,
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


def _score_from(table: dict[tuple[int, ...], float]):
    def score(kept: tuple[int, ...]) -> float:
        if kept != tuple(sorted(kept)):
            raise AssertionError(f"steps were reordered: {kept}")
        return table[kept]

    return score


class HierarchicalGroupPruneTests(unittest.TestCase):
    def test_a_valid_group_is_removed(self) -> None:
        score = _score_from(
            {
                (0, 1, 2, 3): 0.0,
                (): -1.0,
                (2, 3): -0.01,
                (0, 1): -1.0,
                (0, 2, 3): -1.0,
                (1, 2, 3): -1.0,
                (0, 1, 2): -1.0,
                (0, 1, 3): -1.0,
                (2,): -1.0,
                (3,): -1.0,
                (0,): -1.0,
                (1,): -1.0,
            }
        )
        result = hierarchical_group_prune(4, score, 0.95, min_steps=0)
        self.assertNotIn(0, result.kept)
        self.assertNotIn(1, result.kept)
        self.assertEqual(result.deleted_indices[:2], (0, 1))

    def test_failed_parent_group_still_searches_a_valid_half(self) -> None:
        calls: list[tuple[int, ...]] = []

        def score(kept: tuple[int, ...]) -> float:
            calls.append(kept)
            if kept == (0, 1, 2, 3):
                return 0.0
            if kept == ():
                return -1.0
            if kept == (2, 3):
                return -0.01
            return -1.0

        result = hierarchical_group_prune(4, score, 0.95, min_steps=0)
        self.assertIn((), calls)
        self.assertEqual(result.kept, (2, 3))

    def test_pair_is_removed_when_each_step_alone_is_invalid(self) -> None:
        def score(kept: tuple[int, ...]) -> float:
            if kept == (0, 1, 2, 3):
                return 0.0
            if kept == (2, 3):
                return -0.01
            return -1.0

        result = hierarchical_group_prune(4, score, 0.95)
        self.assertEqual(result.kept, (2, 3))
        self.assertEqual(result.deleted_indices, (0, 1))

    def test_invalid_large_group_does_not_block_a_smaller_group(self) -> None:
        seen: set[tuple[int, ...]] = set()

        def score(kept: tuple[int, ...]) -> float:
            seen.add(kept)
            if kept == (0, 1, 2, 3):
                return 0.0
            if len(kept) == 0:
                return -1.0
            if kept == (2, 3):
                return -0.01
            return -1.0

        result = hierarchical_group_prune(4, score, 0.95, min_steps=0)
        self.assertIn((), seen)
        self.assertEqual(result.kept, (2, 3))
        self.assertLess(len(result.kept), 4)

    def test_threshold_stays_at_the_original_score(self) -> None:
        def score(kept: tuple[int, ...]) -> float:
            return 0.0 if len(kept) == 4 else -0.01

        result = hierarchical_group_prune(4, score, 0.95, min_steps=0)
        self.assertEqual(result.threshold, result.original_score + math.log(0.95))
        self.assertEqual(result.threshold, fidelity_threshold(result.original_score, 0.95))
        self.assertNotEqual(result.threshold, result.final_score + math.log(0.95))
        self.assertLess(len(result.kept), 4)

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
        def score(kept: tuple[int, ...]) -> float:
            if kept == (0, 1, 2, 3):
                return 0.0
            if kept in {(0, 2, 3), (0, 3)}:
                return -0.01
            return -1.0

        result = hierarchical_group_prune(4, score, 0.95)
        self.assertEqual(result.kept, (0, 3))
        self.assertEqual(result.deleted_indices, (1, 2))

    def test_min_steps_never_returns_an_empty_trace(self) -> None:
        calls: list[tuple[int, ...]] = []

        def score(kept: tuple[int, ...]) -> float:
            calls.append(kept)
            return 0.0

        result = hierarchical_group_prune(4, score, 0.95, min_steps=1)
        self.assertGreaterEqual(len(result.kept), 1)
        self.assertNotIn((), calls)

    def test_repeated_subset_uses_the_cache(self) -> None:
        calls = 0

        def raw(kept: tuple[int, ...]) -> float:
            nonlocal calls
            calls += 1
            return 0.0 if len(kept) == 4 else -1.0

        cached = ScoreCache(raw)
        first = hierarchical_group_prune(4, cached, 0.95)
        second = hierarchical_group_prune(4, cached, 0.95)
        self.assertEqual(second.num_score_evaluations, 0)
        self.assertGreater(second.cache_hits, 0)
        self.assertEqual(calls, first.num_score_evaluations)

    def test_batched_scores_match_sequential_scores(self) -> None:
        def raw(kept: tuple[int, ...]) -> float:
            if kept == (0, 1, 2, 3):
                return 0.0
            if kept == (2, 3):
                return -0.01
            return -1.0

        batches: list[int] = []

        def score_many(keys: list[tuple[int, ...]]) -> list[float]:
            batches.append(len(keys))
            return [raw(key) for key in keys]

        batched = hierarchical_group_prune(4, raw, 0.95, score_many=score_many)
        sequential = hierarchical_group_prune(4, raw, 0.95)
        self.assertEqual(batched.kept, sequential.kept)
        self.assertEqual(batched.final_score, sequential.final_score)
        self.assertGreater(max(batches), 1)

    def test_final_subset_meets_the_fidelity_bound(self) -> None:
        eta = 0.95

        def score(kept: tuple[int, ...]) -> float:
            if kept == (0, 1, 2, 3):
                return 0.0
            if kept == (2, 3):
                return -0.01
            return -1.0

        result = hierarchical_group_prune(4, score, eta)
        self.assertGreaterEqual(result.final_score, result.original_score + math.log(eta))
        metrics = sample_compression(
            4, len(result.kept), 40, 20, result.original_score, result.final_score
        )
        self.assertGreaterEqual(metrics["fidelity"], eta - 1e-12)


if __name__ == "__main__":
    unittest.main()
