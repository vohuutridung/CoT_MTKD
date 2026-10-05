"""Medoid Cloning: pick the council adapter closest to all others.

The distance is the Phase-1 projection distance between LoRA subspaces:
``X_A = A`` (``r x d_in``) and ``X_B = B^T`` (``r x d_out``) span row spaces,
``C_pq = X_p X_q^T``, ``f_pq = tr(C~_pp^{-1} C_pq C~_qq^{-1} C_qp)`` and
``d_X^2 = r - f_pq``; ``d(phi_p, phi_q)^2`` averages ``d_A^2 + d_B^2`` over
the target modules. Gram matrices use a relative/absolute ridge in fp32 under
``no_grad``. The ``B`` side participates only when ``min_m ||B_m||_F > tau_B``.
The medoid minimizes ``D_m = sum_{q != m} d(phi_m, phi_q)^2``.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any

import torch

ADAPTER_KEY = re.compile(r"^(?P<module>.+)\.lora_(?P<side>[AB])\.(?P<adapter>\{adapter\}|[^.]+)\.weight$")


@dataclass(frozen=True)
class MedoidResult:
    """Pairwise squared distances, per-expert sums and the selected medoid."""

    adapter_names: tuple[str, ...]
    squared_distances: torch.Tensor  # [M, M] float64
    sums: torch.Tensor  # [M] float64, D_m
    medoid_index: int
    use_b_side: bool
    min_b_norm: float
    module_count: int

    @property
    def medoid_name(self) -> str:
        return self.adapter_names[self.medoid_index]

    def to_json(self) -> dict[str, Any]:
        return {
            "adapter_names": list(self.adapter_names),
            "squared_distances": self.squared_distances.tolist(),
            "sums": self.sums.tolist(),
            "medoid_index": self.medoid_index,
            "medoid_name": self.medoid_name,
            "use_b_side": self.use_b_side,
            "min_b_norm": self.min_b_norm,
            "module_count": self.module_count,
        }


def validate_medoid_config(medoid: dict[str, Any]) -> None:
    rel, absolute, tau_b = (
        float(medoid["epsilon_rel"]),
        float(medoid["epsilon_abs"]),
        float(medoid.get("tau_b", 0.0)),
    )
    if rel < 0 or absolute <= 0:
        raise ValueError("medoid.epsilon_rel must be >= 0 and medoid.epsilon_abs > 0")
    if tau_b < 0:
        raise ValueError("medoid.tau_b must be nonnegative")


def _ridge_gram(x: torch.Tensor, epsilon_rel: float, epsilon_abs: float) -> torch.Tensor:
    gram = x @ x.transpose(0, 1)
    rank = gram.shape[0]
    ridge = epsilon_rel * torch.trace(gram) / rank + epsilon_abs
    return gram + ridge * torch.eye(rank, dtype=gram.dtype, device=gram.device)


@torch.no_grad()
def projection_distance_matrix(
    matrices: list[torch.Tensor], epsilon_rel: float, epsilon_abs: float
) -> torch.Tensor:
    """``d_X(p, q)^2`` for every pair of ``r x d`` matrices (fp32 Gram algebra)."""
    if not matrices:
        raise ValueError("Need at least one matrix")
    rank = matrices[0].shape[0]
    xs = [matrix.detach().float() for matrix in matrices]
    if any(x.ndim != 2 or x.shape[0] != rank for x in xs):
        raise ValueError("All matrices must share the same rank r along dim 0")
    count = len(xs)
    inverses = [torch.linalg.inv(_ridge_gram(x, epsilon_rel, epsilon_abs)) for x in xs]
    distances = torch.zeros((count, count), dtype=torch.float64, device=xs[0].device)
    for p in range(count):
        for q in range(p + 1, count):
            cross = xs[p] @ xs[q].transpose(0, 1)
            f = torch.trace(inverses[p] @ cross @ inverses[q] @ cross.transpose(0, 1))
            value = float(rank) - f.double()
            distances[p, q] = value
            distances[q, p] = value
    return distances


def group_adapter_matrices(
    adapter_states: dict[str, dict[str, torch.Tensor]],
) -> dict[tuple[str, str], list[torch.Tensor]]:
    """Map ``(module, side)`` to the ``r x d`` subspace matrix of every adapter.

    ``adapter_states`` maps adapter name to its canonical state dict
    (``{module}.lora_{A,B}.{adapter}.weight``, where the adapter slot is the
    literal ``{adapter}`` placeholder or the adapter's own name). ``A`` is
    used as-is (``r x d_in``) and ``B`` transposed (``r x d_out``).
    """
    names = list(adapter_states)
    grouped: dict[tuple[str, str], dict[str, torch.Tensor]] = {}
    for name in names:
        for key, weight in adapter_states[name].items():
            match = ADAPTER_KEY.match(key)
            if match is None:
                raise ValueError(f"Unexpected adapter key {key!r}")
            if match.group("adapter") not in ("{adapter}", name):
                raise ValueError(f"Adapter state for {name!r} contains foreign key {key!r}")
            side = match.group("side")
            matrix = weight if side == "A" else weight.transpose(0, 1)
            grouped.setdefault((match.group("module"), side), {})[name] = matrix
    ordered: dict[tuple[str, str], list[torch.Tensor]] = {}
    for module_side, per_adapter in grouped.items():
        if set(per_adapter) != set(names):
            raise ValueError(f"Module {module_side} is missing in some adapters")
        ordered[module_side] = [per_adapter[name] for name in names]
    if not ordered:
        raise ValueError("Adapter states contain no LoRA weights")
    return ordered


@torch.no_grad()
def select_medoid(
    adapter_states: dict[str, dict[str, torch.Tensor]], medoid: dict[str, Any]
) -> MedoidResult:
    """Compute ``d(phi_p, phi_q)^2`` over the target modules and pick ``argmin D_m``."""
    validate_medoid_config(medoid)
    names = tuple(adapter_states)
    if len(names) < 2:
        raise ValueError("Medoid selection needs at least two adapters")
    epsilon_rel, epsilon_abs = float(medoid["epsilon_rel"]), float(medoid["epsilon_abs"])
    tau_b = float(medoid.get("tau_b", 0.0))
    grouped = group_adapter_matrices(adapter_states)
    b_norms = [
        min(float(torch.linalg.norm(m.float())) for m in matrices)
        for (_, side), matrices in grouped.items()
        if side == "B"
    ]
    min_b_norm = min(b_norms) if b_norms else 0.0
    use_b_side = bool(b_norms) and min_b_norm > tau_b
    modules = {module for module, _ in grouped}
    total = torch.zeros((len(names), len(names)), dtype=torch.float64)
    for (module, side), matrices in grouped.items():
        if side == "B" and not use_b_side:
            continue
        total += projection_distance_matrix(matrices, epsilon_rel, epsilon_abs).cpu()
    total /= float(len(modules))
    sums = total.sum(dim=1)
    index = int(torch.argmin(sums).item())
    return MedoidResult(
        adapter_names=names,
        squared_distances=total,
        sums=sums,
        medoid_index=index,
        use_b_side=use_b_side,
        min_b_norm=min_b_norm,
        module_count=len(modules),
    )
