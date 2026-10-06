"""Greedy deletion of existing reasoning steps.

The search only calls an ensemble score. Correctness generation is a separate
step that runs after no further deletion is accepted.
"""

from __future__ import annotations

import math
from collections.abc import Callable, Sequence
from dataclasses import dataclass


@dataclass(frozen=True)
class Deletion:
    iteration: int
    deleted_step_index: int
    score_before: float
    score_after: float
    score_drop: float
    remaining_num_steps: int


@dataclass(frozen=True)
class PruneResult:
    kept: tuple[int, ...]
    original_score: float
    final_score: float
    threshold: float
    eta: float
    history: tuple[Deletion, ...]
    states: tuple[tuple[int, ...], ...]


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
    """Prompt tokens stay the original prefix. Only whole steps are removed."""
    ids = [int(token) for token in prefix]
    for index in kept:
        ids.extend(int(token) for token in steps[int(index)])
    ids.extend(int(token) for token in answer_prefix)
    ids.extend(int(token) for token in solution)
    return ids


def greedy_delete(
    num_steps: int,
    score: Callable[[tuple[int, ...]], float],
    eta: float,
    score_many: Callable[[list[tuple[int, ...]]], Sequence[float]] | None = None,
) -> PruneResult:
    """Delete the best remaining step, then rank every survivor again.

    ``score`` receives original step indices in increasing order. It must be
    the ensemble answer log-probability, not a generated-answer check.
    """
    if isinstance(num_steps, bool) or not isinstance(num_steps, int) or num_steps < 0:
        raise ValueError("num_steps must be a nonnegative integer")
    cached = score if isinstance(score, ScoreCache) else ScoreCache(score)
    current = tuple(range(num_steps))
    original_score = cached(current)
    threshold = fidelity_threshold(original_score, eta)
    states = [current]
    history: list[Deletion] = []
    while current:
        trials_keys = [tuple(step for step in current if step != index) for index in current]
        cached.ensure(trials_keys, score_many)
        trials: list[tuple[float, int, tuple[int, ...]]] = []
        for index, trial in zip(current, trials_keys, strict=True):
            trials.append((cached(trial), index, trial))
        best_score, best_index, best_trial = max(trials, key=lambda item: (item[0], -item[1]))
        if best_score < threshold:
            break
        score_before = cached(current)
        history.append(
            Deletion(
                iteration=len(history),
                deleted_step_index=best_index,
                score_before=score_before,
                score_after=best_score,
                score_drop=score_before - best_score,
                remaining_num_steps=len(best_trial),
            )
        )
        current = best_trial
        states.append(current)
    return PruneResult(
        kept=current,
        original_score=original_score,
        final_score=cached(current),
        threshold=threshold,
        eta=float(eta),
        history=tuple(history),
        states=tuple(states),
    )


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


def choose_fallback(
    states: Sequence[Sequence[int]],
    is_correct: Callable[[tuple[int, ...]], bool],
) -> tuple[tuple[int, ...], bool, bool]:
    """Check the shortest trace first, then walk back toward the original.

    The original trace is kept when every pruned trace fails. Samples are
    never dropped. ``is_correct`` is not used by ``greedy_delete``.
    """
    if not states:
        raise ValueError("fallback requires the original reasoning state")
    ordered = [tuple(int(index) for index in state) for state in states]
    final = ordered[-1]
    for state in reversed(ordered):
        if is_correct(state):
            return state, True, state != final
    return ordered[0], False, final != ordered[0]
