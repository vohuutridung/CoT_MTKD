"""Beam search over reasoning subsets for Phase-2 CoT pruning.

Any original step may be deleted. Surviving steps keep their original order.
The prompt is not edited and no new reasoning is generated.

This is an approximation, not an exact shortest subset. Search increases the
deletion count. It stops at the first depth that contains a candidate whose
ensemble answer log-probability still clears the fixed original threshold
``S(R_original) + log(eta)``. Deeper subsets are left unexplored. When a depth
has no valid candidate, the ``beam_width`` highest-scoring children are kept
so a later combination can still recover. No answer is generated.
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
    beam_width: int
    num_score_evaluations: int
    num_unique_subsets_evaluated: int
    search_depth: int
    retained_beam_sizes: tuple[int, ...]


@dataclass(frozen=True)
class _BeamState:
    active: tuple[int, ...]
    score: float
    deleted: tuple[int, ...]


class ScoreCache:
    """Deterministic cache keyed by the kept original step indices."""

    def __init__(self, score: Callable[[tuple[int, ...]], float]) -> None:
        self._score = score
        self._values: dict[tuple[int, ...], float] = {}
        self.misses = 0

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


def _deleted_indices(num_steps: int, active: tuple[int, ...]) -> tuple[int, ...]:
    present = set(active)
    return tuple(index for index in range(num_steps) if index not in present)


def _rank(state: _BeamState) -> tuple[float, tuple[int, ...]]:
    return (-state.score, state.deleted)


def beam_search_subset(
    num_steps: int,
    score: Callable[[tuple[int, ...]], float],
    eta: float,
    beam_width: int = 4,
    min_steps: int = 1,
    max_deletions: int | None = None,
    score_many: Callable[[list[tuple[int, ...]]], Sequence[float]] | None = None,
) -> PruneResult:
    """Delete whole steps until one beam depth still clears the original threshold.

    ``score`` receives original step indices in increasing order. Equal scores
    keep the lexicographically smaller ``deleted_indices``. Duplicate subsets
    are scored once. The retained beam never exceeds ``beam_width``.
    """
    if isinstance(num_steps, bool) or not isinstance(num_steps, int) or num_steps < 0:
        raise ValueError("num_steps must be a nonnegative integer")
    if isinstance(min_steps, bool) or not isinstance(min_steps, int) or min_steps < 0:
        raise ValueError("min_steps must be a nonnegative integer")
    if isinstance(beam_width, bool) or not isinstance(beam_width, int) or beam_width < 1:
        raise ValueError("beam_width must be a positive integer")
    if max_deletions is not None and (
        isinstance(max_deletions, bool) or not isinstance(max_deletions, int) or max_deletions < 0
    ):
        raise ValueError("max_deletions must be null or a nonnegative integer")
    cached = score if isinstance(score, ScoreCache) else ScoreCache(score)
    misses_before = cached.misses
    original = tuple(range(num_steps))
    original_score = cached(original)
    threshold = fidelity_threshold(original_score, eta)
    best = _BeamState(original, original_score, ())
    beam = [best]
    retained: list[int] = []
    room = max(0, num_steps - min_steps)
    max_depth = room if max_deletions is None else min(room, max_deletions)
    for _depth in range(1, max_depth + 1):
        unique: dict[tuple[int, ...], None] = {}
        for state in beam:
            if len(state.active) - 1 < min_steps:
                continue
            for index in state.active:
                child = tuple(step for step in state.active if step != index)
                unique.setdefault(child, None)
        if not unique:
            break
        children = list(unique)
        cached.ensure(children, score_many)
        scored = [
            _BeamState(child, cached(child), _deleted_indices(num_steps, child))
            for child in children
        ]
        scored.sort(key=_rank)
        valid = [state for state in scored if state.score >= threshold]
        if valid:
            best = valid[0]
            break
        beam = scored[:beam_width]
        retained.append(len(beam))
        if not beam:
            break
    if best.score < threshold:
        raise RuntimeError("Selected subset is below the fixed fidelity threshold")
    evaluations = cached.misses - misses_before
    return PruneResult(
        kept=best.active,
        deleted_indices=best.deleted,
        original_score=original_score,
        final_score=best.score,
        threshold=threshold,
        eta=float(eta),
        beam_width=beam_width,
        num_score_evaluations=evaluations,
        num_unique_subsets_evaluated=evaluations,
        search_depth=len(best.deleted),
        retained_beam_sizes=tuple(retained),
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
