from __future__ import annotations

import contextlib
import hashlib
import random
from collections.abc import Iterator

import numpy as np
import torch


def seed_everything(seed: int, deterministic_algorithms: bool = False) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(deterministic_algorithms, warn_only=True)


def derived_seed(base_seed: int, *parts: object) -> int:
    payload = ":".join([str(base_seed), *(str(part) for part in parts)]).encode("utf-8")
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "big") % (2**31 - 1)


@contextlib.contextmanager
def deterministic_rng(seed: int, device: torch.device) -> Iterator[None]:
    devices = [device.index or 0] if device.type == "cuda" else []
    with torch.random.fork_rng(devices=devices, enabled=True):
        torch.manual_seed(seed)
        if device.type == "cuda":
            torch.cuda.manual_seed(seed)
        yield
