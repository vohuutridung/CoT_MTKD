"""Hierarchical group pruning for Phase-2 CoT compression.

Contiguous groups of original reasoning steps are tested for deletion. A group
is removed when its answer-token NLL stays within ``nll_budget`` of the NLL of
the trace accepted so far. A rejected group is split in half and both halves
are searched. Deleted steps are not put back, and a deleted group is not
searched again. Surviving steps keep their original indices and order.

NLL may rise or fall when steps are removed. The budget is local to the
current trace, so the total change from the original trace can exceed one
budget. The procedure is a heuristic for distillation, not a proof that the
deleted steps are redundant. No answer is generated.
"""

from __future__ import annotations

import math
from collections.abc import Callable, Sequence
from dataclasses import dataclass


@dataclass(frozen=True)
class PruneResult:
    kept: tuple[int, ...]
    deleted_indices: tuple[int, ...]
    original_nll: float
    final_nll: float
    nll_budget: float
    nll_delta_from_original: float
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


def within_nll_budget(candidate_nll: float, current_nll: float, nll_budget: float) -> bool:
    """Accept a deletion when it does not raise NLL by more than the local budget."""
    if not math.isfinite(candidate_nll) or not math.isfinite(current_nll):
        raise ValueError("NLL must be finite")
    if not math.isfinite(nll_budget) or nll_budget < 0.0:
        raise ValueError("nll_budget must be finite and nonnegative")
    return candidate_nll <= current_nll + nll_budget


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


def hierarchical_group_prune(
    num_steps: int,
    score: Callable[[tuple[int, ...]], float],
    nll_budget: float,
    min_steps: int = 1,
    max_depth: int | None = None,
    score_many: Callable[[list[tuple[int, ...]]], Sequence[float]] | None = None,
) -> PruneResult:
    """Delete contiguous groups while answer NLL stays inside the local budget.

    ``score`` returns the ensemble answer log-probability, so NLL is its
    negation. Each accepted deletion replaces the NLL used for the next
    comparison. ``score`` receives surviving original step indices in order.
    """
    if isinstance(num_steps, bool) or not isinstance(num_steps, int) or num_steps < 0:
        raise ValueError("num_steps must be a nonnegative integer")
    if isinstance(min_steps, bool) or not isinstance(min_steps, int) or min_steps < 0:
        raise ValueError("min_steps must be a nonnegative integer")
    if not math.isfinite(nll_budget) or nll_budget < 0.0:
        raise ValueError("nll_budget must be finite and nonnegative")
    if max_depth is not None and (
        isinstance(max_depth, bool) or not isinstance(max_depth, int) or max_depth < 0
    ):
        raise ValueError("max_depth must be null or a nonnegative integer")
    cached = score if isinstance(score, ScoreCache) else ScoreCache(score)
    misses_before = cached.misses
    hits_before = cached.hits
    original = tuple(range(num_steps))
    original_nll = -cached(original)
    current_nll = original_nll

    def prune(active: tuple[int, ...], group: tuple[int, ...], depth: int) -> tuple[int, ...]:
        nonlocal current_nll
        if len(active) <= min_steps:
            return active
        present = tuple(index for index in group if index in set(active))
        if not present:
            return active
        candidate = _without(active, present)
        if len(candidate) >= min_steps:
            candidate_nll = -cached(candidate)
            if within_nll_budget(candidate_nll, current_nll, nll_budget):
                current_nll = candidate_nll
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
    final_nll = -cached(final)
    if final_nll != current_nll:
        raise RuntimeError("Accepted trace NLL does not match the scored active set")
    evaluations = cached.misses - misses_before
    return PruneResult(
        kept=final,
        deleted_indices=_without(original, final),
        original_nll=original_nll,
        final_nll=final_nll,
        nll_budget=float(nll_budget),
        nll_delta_from_original=final_nll - original_nll,
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
