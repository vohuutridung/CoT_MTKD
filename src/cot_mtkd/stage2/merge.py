from __future__ import annotations

from typing import Any, Sequence

import torch

MERGE_METHODS = ("ta", "ties", "dare_ties", "tsv", "iso_c")


def normalize_merge_method(name: str) -> str:
    aliases = {
        "task_arithmetic": "ta",
        "task-arithmetic": "ta",
        "dare-ties": "dare_ties",
        "dareties": "dare_ties",
        "iso-c": "iso_c",
        "isoc": "iso_c",
        "iso_c": "iso_c",
    }
    key = str(name).strip().lower().replace(" ", "_")
    key = aliases.get(key, key)
    if key not in MERGE_METHODS:
        raise ValueError(
            f"Unknown merge method {name!r}; expected one of {list(MERGE_METHODS)}"
        )
    return key


def lora_scaling(rank: int, alpha: float) -> float:
    if int(rank) <= 0:
        raise ValueError("LoRA rank must be positive")
    return float(alpha) / float(rank)


def _lora_pairs(keys: Sequence[str]) -> list[tuple[str, str]]:
    pairs: list[tuple[str, str]] = []
    for key in keys:
        if ".lora_A." not in key:
            continue
        b_key = key.replace(".lora_A.", ".lora_B.")
        if b_key not in keys:
            raise RuntimeError(f"LoRA A matrix {key!r} has no matching B matrix")
        pairs.append((key, b_key))
    if not pairs:
        raise RuntimeError("Adapter state does not contain LoRA A/B matrices")
    return pairs


def _factorize(
    delta: torch.Tensor, rank: int, scaling: float
) -> tuple[torch.Tensor, torch.Tensor]:
    matrix = delta.detach().float()
    scale = float(scaling) if float(scaling) != 0.0 else 1.0
    matrix = matrix / scale
    rows, cols = matrix.shape
    kept = min(int(rank), rows, cols)
    if kept <= 0:
        raise ValueError("Cannot factorize an empty LoRA delta")
    if min(rows, cols) <= 512:
        left, values, right = torch.linalg.svd(matrix, full_matrices=False)
        left, values, right = left[:, :kept], values[:kept], right[:kept]
        return right.contiguous(), (left * values).contiguous()
    query = min(max(kept * 2, kept), min(rows, cols))
    left, values, right = torch.svd_lowrank(matrix, q=query)
    left, values, right = left[:, :kept], values[:kept], right[:, :kept]
    return right.transpose(0, 1).contiguous(), (left * values).contiguous()


def _reconstruct(
    a_matrix: torch.Tensor, b_matrix: torch.Tensor, scaling: float
) -> torch.Tensor:
    return float(scaling) * (b_matrix.float() @ a_matrix.float())


def _ta(deltas: Sequence[torch.Tensor]) -> torch.Tensor:
    stacked = torch.stack([delta.float() for delta in deltas], dim=0)
    return stacked.mean(dim=0)


def _topk_magnitude_mask(values: torch.Tensor, density: float) -> torch.Tensor:
    if not 0.0 < float(density) <= 1.0:
        raise ValueError("ties_density must be in (0, 1]")
    flat = values.reshape(-1)
    keep = max(1, int(round(float(density) * flat.numel())))
    keep = min(keep, flat.numel())
    threshold = torch.topk(flat.abs(), keep, largest=True).values[-1]
    return values.abs() >= threshold


def _ties(deltas: Sequence[torch.Tensor], density: float) -> torch.Tensor:
    trimmed = []
    for delta in deltas:
        current = delta.float()
        mask = _topk_magnitude_mask(current, density).to(current.dtype)
        trimmed.append(current * mask)
    stacked = torch.stack(trimmed, dim=0)
    elected = stacked.sum(dim=0).sign()
    elected = torch.where(elected == 0, torch.ones_like(elected), elected)
    agree = stacked.sign() == elected.unsqueeze(0)
    weight = agree.to(stacked.dtype)
    numerator = (stacked * weight).sum(dim=0)
    denominator = weight.sum(dim=0).clamp_min(1.0)
    return numerator / denominator


def _dare(
    delta: torch.Tensor, drop_probability: float, generator: torch.Generator
) -> torch.Tensor:
    if not 0.0 <= float(drop_probability) < 1.0:
        raise ValueError("dare_drop_prob must be in [0, 1)")
    mask = (
        torch.rand(delta.shape, generator=generator, dtype=torch.float32)
        > float(drop_probability)
    )
    kept = 1.0 - float(drop_probability)
    return delta.float() * mask.to(delta.dtype) / kept


def _thin_svd(
    delta: torch.Tensor, max_rank: int
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    matrix = delta.float()
    query = max(1, min(int(max_rank), *matrix.shape))
    if min(matrix.shape) <= 512:
        left, values, right = torch.linalg.svd(matrix, full_matrices=False)
        return left[:, :query], values[:query], right[:query]
    left, values, right = torch.svd_lowrank(matrix, q=query)
    return left[:, :query], values[:query], right[:, :query].transpose(0, 1)


def _tsv(
    deltas: Sequence[torch.Tensor], reduction: float, max_rank: int
) -> torch.Tensor:
    if not 0.0 < float(reduction) <= 1.0:
        raise ValueError("tsv_reduction must be in (0, 1]")
    components: list[tuple[torch.Tensor, torch.Tensor, torch.Tensor]] = []
    kept = 0
    for delta in deltas:
        left, values, right = _thin_svd(delta, max_rank)
        width = max(1, int(values.numel() * float(reduction)))
        width = min(width, values.numel())
        components.append((left[:, :width], values[:width], right[:width]))
        kept += width
    rows, cols = deltas[0].shape
    concatenated_u = torch.zeros(rows, kept, dtype=torch.float32)
    concatenated_s = torch.zeros(kept, dtype=torch.float32)
    concatenated_v = torch.zeros(kept, cols, dtype=torch.float32)
    cursor = 0
    for left, values, right in components:
        width = values.numel()
        concatenated_u[:, cursor : cursor + width] = left
        concatenated_s[cursor : cursor + width] = values
        concatenated_v[cursor : cursor + width, :] = right
        cursor += width
    left_u, _, right_u = torch.linalg.svd(concatenated_u, full_matrices=False)
    left_v, _, right_v = torch.linalg.svd(concatenated_v, full_matrices=False)
    return torch.linalg.multi_dot(
        (left_u, right_u, torch.diag(concatenated_s), left_v, right_v)
    )


def _iso_c(deltas: Sequence[torch.Tensor], max_rank: int) -> torch.Tensor:
    merged = _ta(deltas)
    left, values, right = _thin_svd(merged, max_rank)
    if values.numel() == 0:
        return merged
    isotropic = torch.full_like(values, float(values.mean().item()))
    return (left * isotropic) @ right


def _merge_deltas(
    deltas: Sequence[torch.Tensor],
    method: str,
    ties_density: float,
    dare_drop_prob: float,
    tsv_reduction: float,
    generator: torch.Generator,
    lora_rank: int,
) -> torch.Tensor:
    max_rank = int(lora_rank) * len(deltas)
    if method == "ta":
        return _ta(deltas)
    if method == "ties":
        return _ties(deltas, ties_density)
    if method == "dare_ties":
        dropped = [_dare(delta, dare_drop_prob, generator) for delta in deltas]
        return _ties(dropped, ties_density)
    if method == "tsv":
        return _tsv(deltas, tsv_reduction, max_rank=int(lora_rank))
    if method == "iso_c":
        return _iso_c(deltas, max_rank=max_rank)
    raise ValueError(f"Unsupported merge method {method!r}")


def merge_adapter_states(
    expert_states: Sequence[dict[str, torch.Tensor]],
    *,
    method: str,
    rank: int,
    alpha: float,
    seed: int = 42,
    merge_scaling: float = 1.0,
    ties_density: float = 0.2,
    dare_drop_prob: float = 0.5,
    tsv_reduction: float | None = None,
) -> dict[str, torch.Tensor]:
    """Merge LoRA experts into one student adapter.

    Each expert is reconstructed as ΔW_m = s B_m A_m. The default `ta`
    operator is TruncSVD_r(mean_m ΔW_m), then factorized back to rank-r
    LoRA so that s B_eff A_eff ≈ ΔW_merge.
    """
    if not expert_states:
        raise ValueError("At least one expert adapter is required to merge")
    method = normalize_merge_method(method)
    reference = expert_states[0]
    for state in expert_states[1:]:
        if set(state) != set(reference):
            raise RuntimeError("Expert adapter states do not share the same keys")
    scaling = lora_scaling(rank, alpha)
    reduction = (
        1.0 / len(expert_states) if tsv_reduction is None else float(tsv_reduction)
    )
    generator = torch.Generator(device="cpu")
    generator.manual_seed(int(seed))
    merged: dict[str, torch.Tensor] = {}
    for a_key, b_key in _lora_pairs(list(reference)):
        deltas = [
            _reconstruct(state[a_key], state[b_key], scaling) for state in expert_states
        ]
        delta = float(merge_scaling) * _merge_deltas(
            deltas,
            method,
            ties_density=float(ties_density),
            dare_drop_prob=float(dare_drop_prob),
            tsv_reduction=reduction,
            generator=generator,
            lora_rank=int(rank),
        )
        a_matrix, b_matrix = _factorize(delta, rank, scaling)
        merged[a_key] = a_matrix.to(dtype=reference[a_key].dtype)
        merged[b_key] = b_matrix.to(dtype=reference[b_key].dtype)
    missing = set(reference) - set(merged)
    if missing:
        raise RuntimeError(f"Unmerged adapter keys: {sorted(missing)}")
    return merged


def merge_config_values(config: dict[str, Any]) -> dict[str, Any]:
    stage2 = config["stage2"]
    reduction = stage2.get("tsv_reduction")
    return {
        "method": normalize_merge_method(str(stage2["merge_method"])),
        "rank": int(config["lora"]["rank"]),
        "alpha": float(config["lora"]["alpha"]),
        "seed": int(config["seed"]),
        "merge_scaling": float(stage2.get("merge_scaling", 1.0)),
        "ties_density": float(stage2.get("ties_density", 0.2)),
        "dare_drop_prob": float(stage2.get("dare_drop_prob", 0.5)),
        "tsv_reduction": None if reduction is None else float(reduction),
    }
