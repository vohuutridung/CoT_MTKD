from __future__ import annotations

import re
from dataclasses import dataclass

from .schema import CharacterSegment, TokenRegion

DEFAULT_SYSTEM_PROMPT = (
    "You are Qwen, created by Alibaba Cloud. You are a helpful assistant."
)
DEFAULT_STEP_PATTERN = r"\r?\n[ \t]*\r?\n+"


@dataclass(frozen=True)
class SerializedResponse:
    text: str
    segments: tuple[CharacterSegment, ...]
    reasoning_start_char: int
    answer_start_char: int
    step_count: int


class _Builder:
    def __init__(self) -> None:
        self.parts: list[str] = []
        self.length = 0
        self.segments: list[CharacterSegment] = []

    def append(
        self, text: str, region: TokenRegion, step_id: int = -1
    ) -> tuple[int, int]:
        start = self.length
        self.parts.append(text)
        self.length += len(text)
        if text:
            self.segments.append(CharacterSegment(start, self.length, region, step_id))
        return start, self.length

    def append_reasoning(
        self, reasoning: str, step_pattern: str
    ) -> tuple[int, int, int]:
        start = self.length
        self.parts.append(reasoning)
        self.length += len(reasoning)
        relative = reasoning_character_segments(reasoning, step_pattern)
        self.segments.extend(
            CharacterSegment(
                start + item.start, start + item.end, item.region, item.step_id
            )
            for item in relative
        )
        step_count = 0 if not relative else max(item.step_id for item in relative) + 1
        return start, self.length, step_count

    def build(self) -> str:
        return "".join(self.parts)


def reasoning_character_segments(
    reasoning: str, step_pattern: str = DEFAULT_STEP_PATTERN
) -> tuple[CharacterSegment, ...]:
    if not reasoning:
        return ()
    boundaries = list(re.finditer(step_pattern, reasoning))
    raw_blocks: list[tuple[int, int]] = []
    cursor = 0
    for boundary in boundaries:
        raw_blocks.append((cursor, boundary.start()))
        cursor = boundary.end()
    raw_blocks.append((cursor, len(reasoning)))

    content_spans: list[tuple[int, int]] = []
    for start, end in raw_blocks:
        while start < end and reasoning[start].isspace():
            start += 1
        while end > start and reasoning[end - 1].isspace():
            end -= 1
        if start < end:
            content_spans.append((start, end))
    if not content_spans:
        return ()

    segments: list[CharacterSegment] = []
    if content_spans[0][0] > 0:
        segments.append(
            CharacterSegment(0, content_spans[0][0], TokenRegion.DELIMITER, 0)
        )
    for step_id, (start, end) in enumerate(content_spans):
        segments.append(CharacterSegment(start, end, TokenRegion.REASONING, step_id))
        next_start = (
            content_spans[step_id + 1][0]
            if step_id + 1 < len(content_spans)
            else len(reasoning)
        )
        if end < next_start:
            segments.append(
                CharacterSegment(end, next_start, TokenRegion.DELIMITER, step_id)
            )
    return tuple(segments)


def serialize_record(
    question: str,
    thinking: str,
    attempt: str,
    system_prompt: str = DEFAULT_SYSTEM_PROMPT,
    step_pattern: str = DEFAULT_STEP_PATTERN,
) -> SerializedResponse:
    builder = _Builder()
    builder.append(
        f"<|im_start|>system\n{system_prompt}\n<|im_end|>\n"
        f"<|im_start|>user\n{question}\n<|im_end|>\n",
        TokenRegion.PROMPT,
    )
    builder.append(
        "<|im_start|>assistant\n<|im_start|>think\n", TokenRegion.ASSISTANT_CONTROL
    )
    reasoning_start, _, step_count = builder.append_reasoning(thinking, step_pattern)
    answer_start, _ = builder.append(
        "\n<|im_start|>answer\nAnswer: ", TokenRegion.ANSWER_MARKER
    )
    builder.append(attempt, TokenRegion.ANSWER)
    builder.append("\n<|im_end|>", TokenRegion.EOS)
    return SerializedResponse(
        text=builder.build(),
        segments=tuple(builder.segments),
        reasoning_start_char=reasoning_start,
        answer_start_char=answer_start,
        step_count=step_count,
    )


def pag_answer_prefix() -> str:
    return "\n<|im_start|>answer\nAnswer: "
