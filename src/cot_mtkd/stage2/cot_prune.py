"""Hierarchical group pruning for Phase-2 CoT compression.

Contiguous groups of original reasoning steps are tested for deletion. A group
that still clears the fixed threshold ``S(R) + log(eta)`` is removed. A group
that misses the threshold is split in half, and both halves are searched.
Failure to delete a group does not prune away its subgroups. Surviving steps
keep their original indices and order. The prompt and answer are not edited.

The partition tree is about ``log T`` levels deep. Both children can be scored,
so the total number of ensemble evaluations is not guaranteed to be ``O(log T)``.
The procedure is a heuristic, not an exhaustive shortest-subset search. No
answer is generated.
"""

from __future__ import annotations

import math
from collections.abc import Callable, Sequence
from dataclasses import dataclass


@dataclass(frozen=True)
class PruneResult:
    kept: tuple[int, ...]
    deleted_indices: tuple[int, ...]
    original_score: float
    final_score: float
    threshold: float
    eta: float
    num_score_evaluations: int
    num_unique_subsets_evaluated: int
    cache_hits: int


class ScoreCache:
    """Deterministic cache keyed by the kept original step indices."""

    def __init__(self, score: Callable[[tuple[int, ...]], float]) -> None:
        self._score = score
        self._values: dict[tuple[int, ...], float] = {}
        self.misses = 0
        self.hits = 0

    def _store(self, key: tuple[int, ...], value: float) -> float:
        value = float(value)
        if not math.isfinite(value):
            raise ValueError(f"Ensemble score is not finite for steps {key}")
        self._values[key] = value
        return value

    def __call__(self, kept: Sequence[int]) -> float:
        key = tuple(int(index) for index in kept)
        if key not in self._values:
            self.misses += 1
            self._store(key, self._score(key))
        else:
            self.hits += 1
        return self._values[key]

    def ensure(
        self,
        keys: Sequence[Sequence[int]],
        score_many: Callable[[list[tuple[int, ...]]], Sequence[float]] | None = None,
    ) -> None:
        missing = [
            tuple(int(index) for index in key)
            for key in keys
            if tuple(int(index) for index in key) not in self._values
        ]
        if not missing:
            return
        if score_many is None or len(missing) == 1:
            for key in missing:
                self(key)
            return
        values = list(score_many(missing))
        if len(values) != len(missing):
            raise RuntimeError("Batched ensemble scoring returned the wrong number of scores")
        self.misses += len(missing)
        for key, value in zip(missing, values, strict=True):
            self._store(key, value)


def fidelity_threshold(original_score: float, eta: float) -> float:
    if not math.isfinite(original_score):
        raise ValueError("original_score must be finite")
    if not math.isfinite(eta) or not 0.0 < eta <= 1.0:
        raise ValueError("eta must be in (0, 1]")
    return original_score + math.log(eta)


def candidate_token_ids(
    prefix: Sequence[int],
    steps: Sequence[Sequence[int]],
    kept: Sequence[int],
    answer_prefix: Sequence[int] = (),
    solution: Sequence[int] = (),
) -> list[int]:
    """Prompt tokens stay put. Kept steps stay in their original order."""
    ids = [int(token) for token in prefix]
    for index in kept:
        ids.extend(int(token) for token in steps[int(index)])
    ids.extend(int(token) for token in answer_prefix)
    ids.extend(int(token) for token in solution)
    return ids


def _without(active: tuple[int, ...], group: Sequence[int]) -> tuple[int, ...]:
    banned = {int(index) for index in group}
    return tuple(index for index in active if index not in banned)


def _prefer(
    current: tuple[int, ...],
    current_score: float,
    challenger: tuple[int, ...],
    challenger_score: float,
) -> bool:
    """Fewer steps win. Equal length prefers the higher score, then smaller indices."""
    if len(challenger) != len(current):
        return len(challenger) < len(current)
    if challenger_score != current_score:
        return challenger_score > current_score
    return challenger < current


def hierarchical_group_prune(
    num_steps: int,
    score: Callable[[tuple[int, ...]], float],
    eta: float,
    min_steps: int = 1,
    max_depth: int | None = None,
    score_many: Callable[[list[tuple[int, ...]]], Sequence[float]] | None = None,
) -> PruneResult:
    """Delete contiguous groups while the original fidelity threshold still holds.

    ``score`` receives the surviving original step indices in increasing order.
    After one sibling is removed, the other sibling is scored against that
    updated active set. The threshold is never recomputed.
    """
    if isinstance(num_steps, bool) or not isinstance(num_steps, int) or num_steps < 0:
        raise ValueError("num_steps must be a nonnegative integer")
    if isinstance(min_steps, bool) or not isinstance(min_steps, int) or min_steps < 0:
        raise ValueError("min_steps must be a nonnegative integer")
    if max_depth is not None and (
        isinstance(max_depth, bool) or not isinstance(max_depth, int) or max_depth < 0
    ):
        raise ValueError("max_depth must be null or a nonnegative integer")
    cached = score if isinstance(score, ScoreCache) else ScoreCache(score)
    misses_before = cached.misses
    hits_before = cached.hits
    original = tuple(range(num_steps))
    original_score = cached(original)
    threshold = fidelity_threshold(original_score, eta)
    best = original
    best_score = original_score

    def consider(candidate: tuple[int, ...], candidate_score: float) -> None:
        nonlocal best, best_score
        if _prefer(best, best_score, candidate, candidate_score):
            best = candidate
            best_score = candidate_score

    def prune(active: tuple[int, ...], group: tuple[int, ...], depth: int) -> tuple[int, ...]:
        present = tuple(index for index in group if index in set(active))
        if not present:
            return active
        candidate = _without(active, present)
        if len(candidate) >= min_steps:
            candidate_score = cached(candidate)
            if candidate_score >= threshold:
                consider(candidate, candidate_score)
                return candidate
        if len(present) == 1:
            return active
        if max_depth is not None and depth >= max_depth:
            return active
        midpoint = len(present) // 2
        halves = (present[:midpoint], present[midpoint:])
        pending = [
            _without(active, half)
            for half in halves
            if len(_without(active, half)) >= min_steps
        ]
        if pending:
            cached.ensure(pending, score_many)
        for half in halves:
            active = prune(active, half, depth + 1)
        return active

    final = prune(original, original, 0)
    final_score = cached(final)
    if _prefer(final, final_score, best, best_score):
        final, final_score = best, best_score
    elif _prefer(best, best_score, final, final_score):
        best, best_score = final, final_score
    if final_score < threshold:
        raise RuntimeError("Selected subset is below the fixed fidelity threshold")
    evaluations = cached.misses - misses_before
    return PruneResult(
        kept=final,
        deleted_indices=_without(original, final),
        original_score=original_score,
        final_score=final_score,
        threshold=threshold,
        eta=float(eta),
        num_score_evaluations=evaluations,
        num_unique_subsets_evaluated=evaluations,
        cache_hits=cached.hits - hits_before,
    )


def sample_compression(
    original_steps: int,
    final_steps: int,
    original_tokens: int,
    final_tokens: int,
    original_score: float,
    final_score: float,
) -> dict[str, float]:
    """Step and token reduction, plus probability-space fidelity ``exp(S_final - S0)``."""
    if original_steps < 0 or not 0 <= final_steps <= original_steps:
        raise ValueError("final step count must lie within the original count")
    if original_tokens < 0 or not 0 <= final_tokens <= original_tokens:
        raise ValueError("final token count must lie within the original count")
    if not math.isfinite(original_score) or not math.isfinite(final_score):
        raise ValueError("scores must be finite")
    step_reduction = 0.0 if original_steps == 0 else 1.0 - final_steps / original_steps
    token_reduction = 0.0 if original_tokens == 0 else 1.0 - final_tokens / original_tokens
    return {
        "step_reduction": step_reduction,
        "token_reduction": token_reduction,
        "fidelity": math.exp(final_score - original_score),
    }


def token_cut_summary(cuts: Sequence[tuple[int, int]]) -> dict[str, float]:
    """Average and corpus-level share of deleted reasoning tokens.

    Each pair is ``(original_reasoning_tokens, deleted_reasoning_tokens)`` for
    one sample. The mean is unweighted across samples. The overall share weights
    each sample by how many reasoning tokens it started with.
    """
    if not cuts:
        raise ValueError("token cut summary needs at least one sample")
    percents: list[float] = []
    total_original = 0
    total_deleted = 0
    for original, deleted in cuts:
        if original < 0 or deleted < 0 or deleted > original:
            raise ValueError("deleted tokens must lie within the original count")
        total_original += original
        total_deleted += deleted
        percents.append(0.0 if original == 0 else 100.0 * deleted / original)
    overall = 0.0 if total_original == 0 else 100.0 * total_deleted / total_original
    return {
        "mean_token_cut_percent": sum(percents) / len(percents),
        "overall_token_cut_percent": overall,
        "total_original_reasoning_tokens": float(total_original),
        "total_deleted_reasoning_tokens": float(total_deleted),
    }
