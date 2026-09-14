from __future__ import annotations

import contextlib
import re
import signal
import threading
from decimal import Decimal, InvalidOperation
from typing import Any


class GradingTimeout(TimeoutError):
    pass


@contextlib.contextmanager
def grading_deadline(seconds: float):
    """Bound symbolic grading on Unix while remaining safe off the main thread."""
    can_alarm = (
        seconds > 0
        and threading.current_thread() is threading.main_thread()
        and hasattr(signal, "setitimer")
    )
    if not can_alarm:
        yield
        return

    def _raise_timeout(signum, frame):
        del signum, frame
        raise GradingTimeout("Mathematical grading exceeded its deadline")

    previous_handler = signal.getsignal(signal.SIGALRM)
    previous_timer = signal.getitimer(signal.ITIMER_REAL)
    signal.signal(signal.SIGALRM, _raise_timeout)
    signal.setitimer(signal.ITIMER_REAL, float(seconds))
    try:
        yield
    finally:
        signal.setitimer(signal.ITIMER_REAL, *previous_timer)
        signal.signal(signal.SIGALRM, previous_handler)


def extract_boxed(text: str) -> str | None:
    marker = r"\boxed{"
    start = text.rfind(marker)
    if start < 0:
        return None
    cursor = start + len(marker)
    depth = 1
    while cursor < len(text):
        if text[cursor] == "{":
            depth += 1
        elif text[cursor] == "}":
            depth -= 1
            if depth == 0:
                return text[start + len(marker) : cursor].strip()
        cursor += 1
    return None


def normalize_answer(value: Any) -> str:
    text = str(value).strip()
    boxed = extract_boxed(text)
    if boxed is not None:
        text = boxed
    text = text.replace("$", "").replace("\\,", "").strip()
    text = re.sub(r"\s+", "", text)
    text = text.removeprefix("Answer:").removeprefix("answer:")
    return text


def _equivalent_normalized_numbers(left: str, right: str) -> bool:
    try:
        return Decimal(left) == Decimal(right)
    except InvalidOperation:
        return False


def grade_math(
    candidate: str,
    reference: Any,
    prefer_math_verify: bool = True,
    timeout_seconds: float = 10.0,
) -> bool:
    if prefer_math_verify:
        try:
            from math_verify import parse, verify

            with grading_deadline(timeout_seconds):
                reference_parsed = parse(str(reference))
                candidate_parsed = parse(candidate)
                if reference_parsed and candidate_parsed:
                    return bool(verify(reference_parsed, candidate_parsed))
        except (ImportError, RuntimeError, ValueError, TypeError, GradingTimeout):
            pass
    normalized_candidate = normalize_answer(candidate)
    normalized_reference = normalize_answer(reference)
    return (
        normalized_candidate == normalized_reference
        or _equivalent_normalized_numbers(normalized_candidate, normalized_reference)
    )
