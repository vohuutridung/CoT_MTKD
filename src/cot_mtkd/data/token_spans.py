from __future__ import annotations

from collections.abc import Sequence

from .schema import CharacterSegment, TokenRegion


def assign_token_regions(
    offsets: Sequence[tuple[int, int]], segments: Sequence[CharacterSegment]
) -> tuple[list[int], list[int]]:
    if not segments:
        raise ValueError("At least one character segment is required")
    regions: list[int] = []
    steps: list[int] = []
    last_segment = segments[0]
    next_nonempty_start = [0] * len(offsets)
    next_start = segments[-1].end
    for index in range(len(offsets) - 1, -1, -1):
        start, end = offsets[index]
        if end > start:
            next_start = start
        next_nonempty_start[index] = next_start
    previous_end = segments[0].start
    for index, (start, end) in enumerate(offsets):
        if end > start:
            best = max(segments, key=lambda item: item.overlap(start, end))
            if best.overlap(start, end) == 0:
                best = min(segments, key=lambda item: abs(item.start - start))
            last_segment = best
            previous_end = max(previous_end, end)
        else:
            # Some fast tokenizers report (0, 0) for literal special tokens.
            # The omitted character interval between neighboring ordinary
            # tokens provides a deterministic structural state transition.
            gap_end = next_nonempty_start[index]
            if gap_end > previous_end:
                best = max(
                    segments,
                    key=lambda item: item.overlap(previous_end, gap_end),
                )
                if best.overlap(previous_end, gap_end) == 0:
                    best = min(
                        segments, key=lambda item: abs(item.start - previous_end)
                    )
                last_segment = best
            else:
                best = last_segment
        regions.append(int(best.region))
        steps.append(best.step_id)
    return regions, steps


def complete_step_truncate(
    input_ids: list[int],
    labels: list[int],
    regions: list[int],
    steps: list[int],
    max_length: int,
) -> tuple[list[int], list[int], list[int], list[int], dict[str, int | bool]]:
    length = len(input_ids)
    if not (length == len(labels) == len(regions) == len(steps)):
        raise ValueError("Token arrays must have identical lengths")
    original_steps = len({step for step in steps if step >= 0})
    if length <= max_length:
        return (
            input_ids,
            labels,
            regions,
            steps,
            {
                "original_length": length,
                "kept_length": length,
                "original_steps": original_steps,
                "kept_steps": original_steps,
                "truncated": False,
            },
        )

    try:
        answer_start = next(
            index
            for index, region in enumerate(regions)
            if region == int(TokenRegion.ANSWER_MARKER)
        )
    except StopIteration as error:
        raise ValueError(
            "Answer marker is required for complete-step truncation"
        ) from error
    reasoning_indices = [index for index, step in enumerate(steps) if step >= 0]
    reasoning_start = min(reasoning_indices, default=answer_start)
    fixed_suffix_length = length - answer_start
    if reasoning_start + fixed_suffix_length > max_length:
        raise ValueError(
            "Prompt/control plus complete answer block exceeds max_length; cannot apply safe truncation"
        )

    prefix_end = reasoning_start
    for step_id in sorted({step for step in steps if step >= 0}):
        step_end = (
            max(index for index, value in enumerate(steps) if value == step_id) + 1
        )
        if step_end + fixed_suffix_length <= max_length:
            prefix_end = step_end
        else:
            break
    keep = list(range(prefix_end)) + list(range(answer_start, length))
    truncated_ids = [input_ids[index] for index in keep]
    truncated_labels = [labels[index] for index in keep]
    truncated_regions = [regions[index] for index in keep]
    truncated_steps = [steps[index] for index in keep]
    kept_steps = len({step for step in truncated_steps if step >= 0})
    return (
        truncated_ids,
        truncated_labels,
        truncated_regions,
        truncated_steps,
        {
            "original_length": length,
            "kept_length": len(keep),
            "original_steps": original_steps,
            "kept_steps": kept_steps,
            "truncated": True,
        },
    )


def first_region_index(
    regions: Sequence[int], region: TokenRegion, default: int
) -> int:
    return next(
        (index for index, value in enumerate(regions) if value == int(region)), default
    )


def validate_token_contract(
    input_ids: Sequence[int],
    labels: Sequence[int],
    regions: Sequence[int],
    steps: Sequence[int],
) -> None:
    if not (len(input_ids) == len(labels) == len(regions) == len(steps)):
        raise ValueError("Prepared token arrays have inconsistent lengths")
    for label, region, step in zip(labels, regions, steps, strict=True):
        if region == int(TokenRegion.PROMPT) and label != -100:
            raise ValueError("Prompt token is not masked from hard loss")
        if (
            region != int(TokenRegion.PROMPT)
            and region != int(TokenRegion.PADDING)
            and label == -100
        ):
            raise ValueError("Assistant-response token is unexpectedly masked")
        if (
            region in (int(TokenRegion.REASONING), int(TokenRegion.DELIMITER))
            and step < 0
        ):
            raise ValueError("Reasoning/delimiter token is missing a step id")
        if (
            region not in (int(TokenRegion.REASONING), int(TokenRegion.DELIMITER))
            and step >= 0
        ):
            raise ValueError("Non-reasoning token has a reasoning step id")
    all_steps = sorted({step for step in steps if step >= 0})
    content_steps = sorted(
        {
            step
            for region, step in zip(regions, steps, strict=True)
            if region == int(TokenRegion.REASONING)
        }
    )
    if all_steps != content_steps:
        raise ValueError("Every tokenized reasoning step must contain a content token")
    if all_steps != list(range(len(all_steps))):
        raise ValueError("Reasoning step ids must be contiguous from zero")
