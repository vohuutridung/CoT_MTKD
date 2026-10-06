"""Shortest reasoning-prefix search for Phase-2 CoT pruning.

Only a prefix of the original steps may be kept. Steps in the middle are never
deleted, reordered, or rewritten, and the prompt is not part of the search.

Binary search assumes prefix validity is monotone: once a prefix clears the
fixed threshold, every longer prefix is treated as valid too. The threshold is
``S(R_original) + log(eta)`` and is not updated from later scores. This finds
the shortest valid prefix under that assumption, not the shortest arbitrary
subset. The score is the ensemble log-probability of the ground-truth answer
only. No answer is generated.
"""

from __future__ import annotations

import math
from collections.abc import Callable, Sequence
from dataclasses import dataclass


@dataclass(frozen=True)
class PruneResult:
    kept: tuple[int, ...]
    original_score: float
    final_score: float
    threshold: float
    eta: float
    num_score_evaluations: int
    evaluated_prefix_lengths: tuple[int, ...]


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
    """Prompt tokens stay the original prefix. Only a whole-step suffix is dropped."""
    ids = [int(token) for token in prefix]
    for index in kept:
        ids.extend(int(token) for token in steps[int(index)])
    ids.extend(int(token) for token in answer_prefix)
    ids.extend(int(token) for token in solution)
    return ids


def binary_search_prefix(
    num_steps: int,
    score: Callable[[tuple[int, ...]], float],
    eta: float,
    min_steps: int = 1,
) -> PruneResult:
    """Return the shortest prefix whose ensemble score clears the original threshold.

    ``score`` receives ``(0, 1, ..., k-1)`` and must be the geometric-mean
    log-probability of the unchanged ground-truth answer. Validity is assumed
    to be monotone in ``k``: a short prefix may fail, and lengthening it is
    what makes the threshold pass. The search therefore takes ``O(log T)``
    score evaluations. It does not try non-prefix subsets.
    """
    if isinstance(num_steps, bool) or not isinstance(num_steps, int) or num_steps < 0:
        raise ValueError("num_steps must be a nonnegative integer")
    if isinstance(min_steps, bool) or not isinstance(min_steps, int) or min_steps < 0:
        raise ValueError("min_steps must be a nonnegative integer")
    cached = score if isinstance(score, ScoreCache) else ScoreCache(score)
    misses_before = cached.misses
    evaluated: list[int] = []

    def cached_prefix(length: int) -> float:
        evaluated.append(length)
        return cached(tuple(range(length)))

    original_score = cached_prefix(num_steps)
    threshold = fidelity_threshold(original_score, eta)
    low = min_steps
    high = num_steps
    best_k = num_steps
    while low <= high:
        mid = (low + high) // 2
        if cached_prefix(mid) >= threshold:
            best_k = mid
            high = mid - 1
        else:
            low = mid + 1
    final_score = cached(tuple(range(best_k)))
    if final_score < threshold:
        raise RuntimeError("Selected prefix is below the fixed fidelity threshold")
    return PruneResult(
        kept=tuple(range(best_k)),
        original_score=original_score,
        final_score=final_score,
        threshold=threshold,
        eta=float(eta),
        num_score_evaluations=cached.misses - misses_before,
        evaluated_prefix_lengths=tuple(evaluated),
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
