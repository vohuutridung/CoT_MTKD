import math
import unittest

from cot_mtkd.stage2.cot_prune_run import majority_vote_correct, reasoning_step_texts
from cot_mtkd.stage2.cot_prune import (
    ScoreCache,
    candidate_token_ids,
    choose_fallback,
    fidelity_threshold,
    greedy_delete,
)


class ReasoningSplitTests(unittest.TestCase):
    def test_blank_lines_are_the_step_boundaries(self) -> None:
        self.assertEqual(
            reasoning_step_texts("alpha\n\nbeta\n\n\ngamma", r"\n\n+"),
            ["alpha", "beta", "gamma"],
        )

    def test_majority_vote_uses_the_unique_winner(self) -> None:
        self.assertEqual(majority_vote_correct(["4", "4", "5"], "4", 1)[0], True)
        self.assertEqual(majority_vote_correct(["4", "5", "6"], "4", 1)[0], False)


class GreedyPruneTests(unittest.TestCase):
    def test_nothing_can_be_deleted(self) -> None:
        def score(kept: tuple[int, ...]) -> float:
            return 0.0 if kept == (0, 1) else -1.0

        result = greedy_delete(2, score, 0.95)
        self.assertEqual(result.kept, (0, 1))
        self.assertEqual(result.history, ())

    def test_one_redundant_step_is_removed(self) -> None:
        def score(kept: tuple[int, ...]) -> float:
            if kept == (0, 1):
                return 0.0
            if kept == (0,):
                return -0.01
            return -1.0

        result = greedy_delete(2, score, 0.95)
        self.assertEqual(result.kept, (0,))
        self.assertEqual(result.history[0].deleted_step_index, 1)

    def test_reranks_after_the_first_deletion(self) -> None:
        calls: list[tuple[int, ...]] = []

        def score(kept: tuple[int, ...]) -> float:
            calls.append(kept)
            table = {
                (0, 1, 2): 0.0,
                (1, 2): 0.0,
                (0, 2): -1.0,
                (0, 1): -1.0,
                (2,): -10.0,
                (1,): -0.01,
                (0,): -10.0,
                (): -10.0,
            }
            return table[kept]

        result = greedy_delete(3, score, 0.95)
        self.assertEqual(result.kept, (1,))
        self.assertEqual(
            [item.deleted_step_index for item in result.history],
            [0, 2],
        )
        second_round = [
            kept for kept in calls if set(kept) <= {1, 2} and len(kept) == 1
        ]
        self.assertIn((1,), second_round)
        self.assertIn((2,), second_round)

    def test_threshold_stays_at_the_original_score(self) -> None:
        result = greedy_delete(
            2,
            lambda kept: 0.0 if len(kept) == 2 else -0.01,
            0.95,
        )
        self.assertEqual(result.threshold, result.original_score + math.log(0.95))
        self.assertTrue(result.history)
        self.assertEqual(result.threshold, fidelity_threshold(result.original_score, 0.95))

    def test_correctness_is_not_called_while_scoring_candidates(self) -> None:
        checked: list[tuple[int, ...]] = []

        def score(kept: tuple[int, ...]) -> float:
            if checked:
                raise AssertionError("correctness ran during candidate scoring")
            return 0.0

        result = greedy_delete(2, score, 0.5)
        self.assertEqual(result.kept, ())

        def is_correct(kept: tuple[int, ...]) -> bool:
            checked.append(kept)
            return True

        accepted, correct, fallback = choose_fallback(result.states, is_correct)
        self.assertEqual(accepted, ())
        self.assertTrue(correct)
        self.assertFalse(fallback)
        self.assertEqual(checked, [()])

    def test_fallback_walks_back_when_the_final_trace_is_wrong(self) -> None:
        states = [(0, 1), (0,), ()]

        def is_correct(kept: tuple[int, ...]) -> bool:
            return kept == (0,)

        accepted, correct, fallback = choose_fallback(states, is_correct)
        self.assertEqual(accepted, (0,))
        self.assertTrue(correct)
        self.assertTrue(fallback)

    def test_eta_one_rejects_any_drop(self) -> None:
        def score(kept: tuple[int, ...]) -> float:
            return 0.0 if len(kept) == 2 else -1.0e-6

        self.assertEqual(greedy_delete(2, score, 1.0).kept, (0, 1))

        def tied(kept: tuple[int, ...]) -> float:
            return 0.0

        self.assertEqual(greedy_delete(2, tied, 1.0).kept, ())

    def test_lower_eta_prunes_more(self) -> None:
        def score(kept: tuple[int, ...]) -> float:
            return {2: 0.0, 1: -0.01, 0: -0.4}[len(kept)]

        strict = greedy_delete(2, score, 0.99)
        loose = greedy_delete(2, score, 0.5)
        self.assertEqual(strict.kept, (1,))
        self.assertEqual(loose.kept, ())
        self.assertLess(len(loose.kept), len(strict.kept))

    def test_score_cache_uses_step_indices(self) -> None:
        calls = 0

        def raw(kept: tuple[int, ...]) -> float:
            nonlocal calls
            calls += 1
            return float(len(kept))

        cached = ScoreCache(raw)
        self.assertEqual(cached((1, 0)), cached(tuple([1, 0])))
        self.assertEqual(calls, 1)

    def test_batched_scores_match_one_at_a_time(self) -> None:
        def raw(kept: tuple[int, ...]) -> float:
            return float(sum(kept))

        sequential = greedy_delete(3, raw, 0.5)
        batched = greedy_delete(3, raw, 0.5, score_many=lambda keys: [raw(key) for key in keys])
        self.assertEqual(batched.kept, sequential.kept)
        self.assertEqual(batched.history, sequential.history)

    def test_prompt_tokens_are_copied_unchanged(self) -> None:
        prefix = [7, 8, 9]
        steps = [[1, 1], [2, 2], [3]]
        original = candidate_token_ids(prefix, steps, (0, 1, 2), [4], [5])
        shorter = candidate_token_ids(prefix, steps, (0, 2), [4], [5])
        self.assertEqual(original[: len(prefix)], prefix)
        self.assertEqual(shorter[: len(prefix)], prefix)
        self.assertEqual(shorter, [7, 8, 9, 1, 1, 3, 4, 5])


if __name__ == "__main__":
    unittest.main()
