from __future__ import annotations

from dataclasses import asdict, dataclass
from enum import IntEnum
from typing import Any


class TokenRegion(IntEnum):
    PROMPT = 0
    ASSISTANT_CONTROL = 1
    REASONING = 2
    DELIMITER = 3
    ANSWER_MARKER = 4
    ANSWER = 5
    EOS = 6
    PADDING = 7


@dataclass
class PreparedRecord:
    sample_id: str
    input_ids: list[int]
    labels: list[int]
    attention_mask: list[int]
    offset_mapping: list[tuple[int, int]]
    region_ids: list[int]
    step_ids: list[int]
    question: str
    thinking: str
    solution: str
    deepseek_grade: str | None
    original_length: int
    kept_length: int
    original_steps: int
    kept_steps: int
    truncated: bool
    # Legacy suffix-boundary name: in CoT-only records this points to EOS.
    # Gold-answer probes construct their own continuation from `solution`.
    answer_start: int
    reasoning_start: int
    tokenizer_fingerprint: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> "PreparedRecord":
        if "attempt" in value:
            raise ValueError(
                "Prepared data with an attempt field is obsolete; rerun prepare "
                "to build the filtered CoT-only corpus"
            )
        normalized = dict(value)
        normalized["offset_mapping"] = [
            tuple(int(item) for item in pair) for pair in normalized["offset_mapping"]
        ]
        return cls(**normalized)


@dataclass(frozen=True)
class CharacterSegment:
    start: int
    end: int
    region: TokenRegion
    step_id: int = -1

    def overlap(self, start: int, end: int) -> int:
        return max(0, min(self.end, end) - max(self.start, start))
