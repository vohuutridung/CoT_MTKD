import math
import unittest

from cot_mtkd.stage2.cot_prune import (
    ScoreCache,
    beam_search_subset,
    candidate_token_ids,
    fidelity_threshold,
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


def _table(values: dict[tuple[int, ...], float]):
    def score(kept: tuple[int, ...]) -> float:
        if kept != tuple(sorted(set(kept))):
            raise AssertionError(f"unordered subset scored: {kept}")
        return values[kept]

    return score


class BeamPruneTests(unittest.TestCase):
    def test_original_is_kept_when_no_deletion_is_valid(self) -> None:
        def score(kept: tuple[int, ...]) -> float:
            return 0.0 if kept == (0, 1, 2) else -1.0

        result = beam_search_subset(3, score, 0.95, beam_width=4)
        self.assertEqual(result.kept, (0, 1, 2))
        self.assertEqual(result.deleted_indices, ())
        self.assertEqual(result.search_depth, 0)

    def test_one_redundant_step_is_removed(self) -> None:
        calls: list[tuple[int, ...]] = []

        def score(kept: tuple[int, ...]) -> float:
            calls.append(kept)
            if kept == (0, 1, 2):
                return 0.0
            if kept == (0, 2):
                return -0.01
            return -1.0

        result = beam_search_subset(3, score, 0.95, beam_width=4)
        self.assertEqual(result.kept, (0, 2))
        self.assertEqual(result.deleted_indices, (1,))
        self.assertEqual(result.search_depth, 1)
        self.assertEqual(result.num_score_evaluations, 4)
        self.assertNotIn((0,), calls)
        self.assertNotIn((2,), calls)

    def test_two_deletions_can_succeed_when_one_does_not(self) -> None:
        def score(kept: tuple[int, ...]) -> float:
            if kept == (0, 1, 2):
                return 0.0
            if kept == (1, 2):
                return -0.2
            if kept == (2,):
                return -0.01
            return -1.0

        result = beam_search_subset(3, score, 0.95, beam_width=1)
        self.assertEqual(result.kept, (2,))
        self.assertEqual(result.deleted_indices, (0, 1))
        self.assertEqual(result.search_depth, 2)
        blocked = beam_search_subset(3, score, 0.95, beam_width=1, max_deletions=1)
        self.assertEqual(blocked.kept, (0, 1, 2))

    def test_paths_that_meet_share_one_cached_score(self) -> None:
        calls: list[tuple[int, ...]] = []

        def score(kept: tuple[int, ...]) -> float:
            calls.append(kept)
            return 0.0 if len(kept) == 3 else -1.0

        result = beam_search_subset(3, score, 0.95, beam_width=3)
        self.assertEqual(len(calls), len(set(calls)))
        self.assertEqual(result.num_score_evaluations, 7)
        self.assertEqual(calls.count((2,)), 1)
        self.assertEqual(calls.count((0,)), 1)

    def test_non_monotonic_path_can_still_be_found(self) -> None:
        seen: dict[tuple[int, ...], float] = {}

        def score(kept: tuple[int, ...]) -> float:
            if kept == (0, 1, 2):
                value = 0.0
            elif kept == (1, 2):
                value = -0.2
            elif kept == (2,):
                value = -0.01
            else:
                value = -1.0
            seen[kept] = value
            return value

        result = beam_search_subset(3, score, 0.95, beam_width=4)
        self.assertEqual(result.kept, (2,))
        self.assertGreater(seen[(0, 1, 2)], seen[(1, 2)])
        self.assertGreater(seen[(2,)], seen[(1, 2)])

    def test_threshold_stays_at_the_original_score(self) -> None:
        def score(kept: tuple[int, ...]) -> float:
            return 0.0 if len(kept) == 3 else -0.01

        result = beam_search_subset(3, score, 0.95, beam_width=4)
        self.assertEqual(result.threshold, result.original_score + math.log(0.95))
        self.assertEqual(result.threshold, fidelity_threshold(result.original_score, 0.95))
        self.assertNotEqual(result.threshold, result.final_score + math.log(0.95))

    def test_every_candidate_keeps_the_original_prompt(self) -> None:
        prompt = [7, 8, 9]
        steps = [[10], [11, 11], [12], [13]]
        for kept in ((0, 1, 2, 3), (0, 2), (1, 3), ()):
            sequence = candidate_token_ids(prompt, steps, kept, [4], [5, 6])
            self.assertEqual(sequence[: len(prompt)], prompt)

    def test_every_candidate_scores_the_same_answer_target(self) -> None:
        solution = [5, 6]
        steps = [[10], [11, 11], [12], [13]]
        for kept in ((0, 1, 2, 3), (0, 2), (3,), ()):
            sequence = candidate_token_ids([7, 8, 9], steps, kept, [4], solution)
            self.assertEqual(sequence[-len(solution) :], solution)

    def test_min_steps_blocks_shorter_states(self) -> None:
        calls: list[tuple[int, ...]] = []

        def score(kept: tuple[int, ...]) -> float:
            calls.append(kept)
            if len(kept) <= 1:
                raise AssertionError("state below min_steps was scored")
            return 0.0 if len(kept) == 3 else -1.0

        result = beam_search_subset(3, score, 0.95, beam_width=4, min_steps=2)
        self.assertEqual(result.kept, (0, 1, 2))
        self.assertTrue(all(len(kept) >= 2 for kept in calls))

    def test_eta_one_rejects_any_drop(self) -> None:
        def dropping(kept: tuple[int, ...]) -> float:
            return 0.0 if len(kept) == 3 else -1.0e-9

        self.assertEqual(beam_search_subset(3, dropping, 1.0, beam_width=4).kept, (0, 1, 2))
        tied = beam_search_subset(3, lambda kept: 0.0, 1.0, beam_width=4)
        self.assertEqual(tied.deleted_indices, (0,))
        self.assertGreaterEqual(tied.final_score, tied.original_score)

    def test_duplicate_subset_is_scored_once(self) -> None:
        calls: list[tuple[int, ...]] = []

        def raw(kept: tuple[int, ...]) -> float:
            calls.append(kept)
            return 0.0 if len(kept) == 4 else -1.0

        cached = ScoreCache(raw)
        batches: list[int] = []

        def score_many(keys: list[tuple[int, ...]]) -> list[float]:
            batches.append(len(keys))
            return [raw(key) for key in keys]

        result = beam_search_subset(4, cached, 0.95, beam_width=4, score_many=score_many)
        self.assertEqual(result.num_unique_subsets_evaluated, len(set(calls)))
        self.assertEqual(calls.count((0, 1)), 1)
        self.assertGreater(max(batches), 1)

    def test_retained_beam_does_not_exceed_beam_width(self) -> None:
        calls: list[tuple[int, ...]] = []

        def score(kept: tuple[int, ...]) -> float:
            calls.append(kept)
            if kept == (0, 1, 2, 3):
                return 0.0
            if kept == (1, 2, 3):
                return -0.1
            if kept == (2, 3):
                return -0.01
            return -5.0

        result = beam_search_subset(4, score, 0.95, beam_width=1)
        self.assertEqual(result.kept, (2, 3))
        self.assertEqual(result.retained_beam_sizes, (1,))
        self.assertNotIn((0, 2), calls)
        wide = beam_search_subset(
            5, lambda kept: 0.0 if len(kept) == 5 else -1.0, 0.95, beam_width=2
        )
        self.assertTrue(wide.retained_beam_sizes)
        self.assertTrue(all(size <= 2 for size in wide.retained_beam_sizes))
        self.assertEqual(wide.retained_beam_sizes[0], 2)

    def test_final_subset_meets_the_fidelity_bound(self) -> None:
        eta = 0.95
        result = beam_search_subset(
            3,
            _table({(0, 1, 2): 0.0, (0, 2): -0.01, (1, 2): -1.0, (0, 1): -1.0}),
            eta,
            beam_width=4,
        )
        self.assertGreaterEqual(result.final_score, result.original_score + math.log(eta))
        metrics = sample_compression(
            3, len(result.kept), 30, 20, result.original_score, result.final_score
        )
        self.assertGreaterEqual(metrics["fidelity"], eta - 1e-12)


if __name__ == "__main__":
    unittest.main()
