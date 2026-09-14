from __future__ import annotations

import torch


def functional_medoid(scores: torch.Tensor) -> int:
    if scores.ndim != 1 or scores.numel() == 0:
        raise ValueError("Medoid scores must be a non-empty vector")
    if not torch.isfinite(scores).all():
        raise FloatingPointError("Functional-medoid scores contain non-finite values")
    return int(scores.argmin().item())


def medoid_kl_contributions(
    probabilities: list[torch.Tensor], log_probabilities: list[torch.Tensor]
) -> torch.Tensor:
    mixture = torch.stack(probabilities, dim=0).mean(dim=0)
    log_mixture = mixture.clamp_min(1.0e-30).log()
    return torch.stack(
        [
            (mixture * (log_mixture - log_probability)).sum(dim=-1).sum()
            for log_probability in log_probabilities
        ]
    )
