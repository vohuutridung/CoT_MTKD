"""Answer grading identical to the exp_s1k P-ALIGN evaluation (``eval/grade.py``).

The reference answer is wrapped in ``\\boxed{}`` before ``math_verify`` parses it,
because raw LaTeX such as ``\\left( 3, \\frac{\\pi}{2} \\right)`` is otherwise
parsed incompletely and correct predictions are rejected. When ``math_verify``
does not confirm a match, a normalized string comparison of the last boxed
answer (or the last number) is used.
"""

from __future__ import annotations

import math
import re
from typing import Any, Optional, Sequence

_BOXED = re.compile(r"\\boxed\s*{")
_NUMBER = re.compile(r"-?\d+(?:\.\d+)?")

try:
    from math_verify import parse as mv_parse
    from math_verify import verify as mv_verify

    HAS_MATH_VERIFY = True
except Exception:  # pragma: no cover - math-verify is a declared dependency
    HAS_MATH_VERIFY = False


def extract_boxed(text: str) -> Optional[str]:
    """Return the content of the last ``\\boxed{...}``, matching nested braces."""
    matches = list(_BOXED.finditer(text))
    if not matches:
        return None
    start = matches[-1].end()
    depth, out = 1, []
    for char in text[start:]:
        if char == "{":
            depth += 1
        elif char == "}":
            depth -= 1
            if depth == 0:
                break
        out.append(char)
    return "".join(out).strip() or None


def normalize(value: str) -> str:
    value = value.strip().rstrip(".").replace(" ", "").replace(",", "")
    value = value.replace("\\left", "").replace("\\right", "")
    value = value.replace("\\!", "").replace("\\,", "").replace("$", "")
    value = re.sub(r"\\text\{([^}]*)\}", r"\1", value)
    if value.endswith("%"):
        value = value[:-1]
    try:
        number = float(value)
        if math.isinf(number) or math.isnan(number):
            return value
        return str(int(number)) if number == int(number) else str(number)
    except (ValueError, OverflowError):
        return value


def grade_answer(prediction: str, gold: Any) -> bool:
    """True when ``prediction`` contains an answer equivalent to ``gold``."""
    gold = str(gold)
    if HAS_MATH_VERIFY:
        try:
            gold_parsed = mv_parse(gold if "\\boxed" in gold else f"\\boxed{{{gold}}}")
            pred_parsed = mv_parse(prediction)
            if gold_parsed and pred_parsed and mv_verify(gold_parsed, pred_parsed):
                return True
        except Exception:
            pass
    pred_boxed = extract_boxed(prediction)
    if pred_boxed is None:
        numbers = _NUMBER.findall(prediction)
        pred_boxed = numbers[-1] if numbers else None
    if pred_boxed is None:
        return False
    gold_boxed = extract_boxed(gold) or gold
    return normalize(pred_boxed) == normalize(gold_boxed)


def grade_samples(predictions: Sequence[str], gold: Any) -> list[bool]:
    return [grade_answer(prediction, gold) for prediction in predictions]
