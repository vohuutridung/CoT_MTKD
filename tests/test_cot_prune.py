import json
import tempfile
import unittest
from pathlib import Path

from cot_mtkd.data.schema import PreparedRecord
from cot_mtkd.stage2.cot_prune import (
    ScoreCache,
    candidate_token_ids,
    hierarchical_group_prune,
    minimum_kept_steps,
    sample_compression,
    token_cut_summary,
    within_nll_budget,
)
from cot_mtkd.stage2.cot_prune_run import (
    load_prune_metrics,
    reasoning_step_texts,
    row_from_saved_pruning,
)


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
            target_keep_ratio=0.0,
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
            target_keep_ratio=0.0,
        )
        self.assertEqual(result.kept, (1,))
        self.assertAlmostEqual(result.final_nll, 9.0)
        self.assertAlmostEqual(result.nll_delta_from_original, -1.0)

    def test_budget_is_measured_from_the_original_trace(self) -> None:
        # 10.07 would fit a local budget of 10.03 + 0.05, but not 10.00 + 0.05.
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
        result = hierarchical_group_prune(4, score, 0.05, target_keep_ratio=0.0)
        self.assertEqual(result.kept, (2, 3))
        self.assertAlmostEqual(result.final_nll, 10.03)
        self.assertLessEqual(result.final_nll, result.original_nll + result.nll_budget)
        self.assertAlmostEqual(result.nll_delta_from_original, 0.03)

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
        result = hierarchical_group_prune(4, score, 0.05, min_steps=0, target_keep_ratio=0.0)
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
        result = hierarchical_group_prune(4, score, 0.05, target_keep_ratio=0.0)
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

            result = hierarchical_group_prune(num_steps, score, 0.05, target_keep_ratio=0.0)
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


class RetentionFloorTests(unittest.TestCase):
    def test_ceil_examples(self) -> None:
        expected = {101: 71, 476: 334, 154: 108, 250: 175, 181: 127}
        for steps, floor in expected.items():
            self.assertEqual(minimum_kept_steps(steps, 1, 0.70), floor)
            self.assertGreaterEqual(floor / steps, 0.70)

    def test_min_steps_can_only_raise_the_floor(self) -> None:
        self.assertEqual(minimum_kept_steps(10, 8, 0.70), 8)
        self.assertEqual(minimum_kept_steps(10, 1, 0.70), 7)

    def test_improving_deletions_stop_at_the_floor(self) -> None:
        def score(kept: tuple[int, ...]) -> float:
            return -float(len(kept))

        seen: list[tuple[int, ...]] = []

        def counting(kept: tuple[int, ...]) -> float:
            seen.append(kept)
            return score(kept)

        result = hierarchical_group_prune(10, counting, 0.05, target_keep_ratio=0.70)
        self.assertEqual(len(result.kept), 7)
        self.assertGreaterEqual(len(result.kept) / 10, 0.70)
        self.assertLess(result.final_nll, result.original_nll)
        self.assertLessEqual(result.final_nll, result.original_nll + 0.05)
        self.assertTrue(all(len(kept) >= 7 for kept in seen))

    def test_stricter_min_steps_is_respected(self) -> None:
        def score(kept: tuple[int, ...]) -> float:
            return -float(len(kept))

        result = hierarchical_group_prune(10, score, 0.05, min_steps=8, target_keep_ratio=0.70)
        self.assertEqual(len(result.kept), 8)

    def test_scripted_subset_stays_inside_both_constraints(self) -> None:
        def prune(num_steps: int, score):
            return hierarchical_group_prune(num_steps, score, 0.05, target_keep_ratio=0.70)

        def shorter_is_better(kept: tuple[int, ...]) -> float:
            return -float(len(kept))

        def any_deletion_is_over_budget(kept: tuple[int, ...]) -> float:
            return -4.0 if len(kept) == 8 else -6.0

        def only_one_step_fits_the_budget(kept: tuple[int, ...]) -> float:
            deleted = 10 - len(kept)
            if deleted <= 1:
                return -(5.0 + 0.04 * deleted)
            return -6.0

        improving = prune(10, shorter_is_better)
        blocked = prune(8, any_deletion_is_over_budget)
        one_step = prune(10, only_one_step_fits_the_budget)
        self.assertEqual(len(improving.kept), 7)
        self.assertLess(improving.final_nll, improving.original_nll)
        self.assertEqual(len(blocked.kept), 8)
        self.assertAlmostEqual(blocked.nll_delta_from_original, 0.0)
        self.assertEqual(len(one_step.kept), 9)
        self.assertAlmostEqual(one_step.nll_delta_from_original, 0.04)
        for result, width in ((improving, 10), (blocked, 8), (one_step, 10)):
            self.assertGreaterEqual(len(result.kept) / width, 0.70)
            self.assertLessEqual(result.final_nll, result.original_nll + result.nll_budget)


class SavedPruneResumeTests(unittest.TestCase):
    def test_partial_final_line_is_dropped_and_a_row_can_be_rebuilt(self) -> None:
        record = PreparedRecord(
            sample_id="s1k-0000",
            input_ids=[1],
            labels=[1],
            attention_mask=[1],
            offset_mapping=[(0, 1)],
            region_ids=[0],
            step_ids=[0],
            question="q",
            thinking="alpha\n\nbeta\n\ngamma",
            solution="ans",
            deepseek_grade="yes",
            original_length=1,
            kept_length=1,
            original_steps=3,
            kept_steps=3,
            truncated=False,
            answer_start=0,
            reasoning_start=0,
            tokenizer_fingerprint="x",
        )
        saved = {
            "time": 1.0,
            "event": "cot_prune",
            "sample_id": "s1k-0000",
            "deleted_indices": [1],
            "original_num_steps": 3,
            "final_num_steps": 2,
            "original_reasoning_tokens": 30,
            "deleted_reasoning_tokens": 10,
        }
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "metrics.jsonl"
            path.write_text(json.dumps(saved) + "\n{\"sample_id\":", encoding="utf-8")
            loaded = load_prune_metrics(path)
            self.assertEqual(list(loaded), ["s1k-0000"])
            self.assertNotIn("time", loaded["s1k-0000"])
            text = path.read_text(encoding="utf-8")
            self.assertTrue(text.endswith("\n"))
            self.assertNotIn('{"sample_id":', text)
        row = row_from_saved_pruning(
            record,
            r"\n\n+",
            {"output": {"save_original_reasoning": True, "save_pruned_reasoning": True}},
            loaded["s1k-0000"],
        )
        self.assertEqual(row["deepseek_thinking_trajectory"], "alpha\n\ngamma")
        self.assertEqual(row["pruning"]["deleted_indices"], [1])
        self.assertEqual(row["question"], "q")
        self.assertEqual(row["solution"], "ans")


if __name__ == "__main__":
    unittest.main()
