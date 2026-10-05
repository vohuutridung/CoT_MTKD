"""Phase-2 step pooling and training-corpus calibration, separate from token JSD."""

from __future__ import annotations

import math
from typing import Any

import torch

POOLING_METHOD = "power_mean_token_js"
RHO_MAPPING = "training_quantile_saturation"


@torch.no_grad()
def pool_token_disagreement(token_js: torch.Tensor, power: float = 4.0) -> torch.Tensor:
    """Power mean in nats; scaling avoids under/overflow for tiny/large values."""
    if not math.isfinite(power) or power <= 0:
        raise ValueError("disagreement_pooling_power must be finite and positive")
    values = token_js.detach().double().flatten()
    if not len(values) or not bool(torch.isfinite(values).all()) or bool((values < 0).any()):
        raise ValueError("Token JSD values must be nonempty, finite and nonnegative")
    if power == 1:
        return values.mean()
    scale = values.max()
    if float(scale) == 0:
        return scale
    return scale * ((values / scale).pow(power).mean()).pow(1 / power)


@torch.no_grad()
def saturation_rho(disagreement: torch.Tensor, tau: float) -> torch.Tensor:
    if not math.isfinite(tau) or tau <= 0:
        raise ValueError("A fitted finite positive tau is required; run stage2-cache")
    values = disagreement.detach().double()
    if not bool(torch.isfinite(values).all()) or bool((values < 0).any()):
        raise ValueError("Step disagreement must be finite and nonnegative")
    rho = values / (values + tau)
    # A finite positive tau gives rho < 1 analytically. Preserve that bound
    # if the addition rounds away tau at extreme scales.
    upper = torch.nextafter(torch.ones_like(rho), torch.zeros_like(rho))
    return rho.minimum(upper)


@torch.no_grad()
def fit_disagreement_calibration(
    training_steps: torch.Tensor,
    power: float = 4.0,
    quantile: float = 0.75,
) -> dict[str, Any]:
    if not math.isfinite(power) or power <= 0:
        raise ValueError("disagreement_pooling_power must be finite and positive")
    if not math.isfinite(quantile) or not 0 < quantile < 1:
        raise ValueError("tau_quantile must lie strictly between 0 and 1")
    values = training_steps.detach().double().flatten()
    if not len(values) or not bool(torch.isfinite(values).all()) or bool((values < 0).any()):
        raise ValueError("Calibration requires valid training reasoning-step disagreements")
    tau = float(torch.quantile(values, quantile, interpolation="linear"))
    if tau <= 0:
        raise ValueError(
            "Training disagreement quantile is zero: cannot fit a positive tau. "
            "Check frozen council/data; no evaluation tuning or artificial tau floor is applied."
        )
    return {
        "tau": tau,
        "tau_quantile": quantile,
        "disagreement_pooling_power": power,
        "disagreement_pooling_method": POOLING_METHOD,
        "rho_mapping": RHO_MAPPING,
        "num_reasoning_steps": len(values),
        "source": "stage2_training_corpus",
        "quantile_interpolation": "linear",
    }


def validate_calibration(calibration: dict, aggregation: dict | None = None) -> None:
    try:
        tau = float(calibration["tau"])
        q = float(calibration["tau_quantile"])
        power = float(calibration["disagreement_pooling_power"])
        valid = (
            math.isfinite(tau)
            and tau > 0
            and math.isfinite(q)
            and 0 < q < 1
            and math.isfinite(power)
            and power > 0
            and calibration["disagreement_pooling_method"] == POOLING_METHOD
            and calibration["rho_mapping"] == RHO_MAPPING
            and calibration["source"] == "stage2_training_corpus"
            and calibration["num_reasoning_steps"] > 0
        )
        if aggregation is not None:
            valid = valid and q == float(aggregation["tau_quantile"])
            valid = valid and power == float(aggregation["disagreement_pooling_power"])
    except (KeyError, TypeError, ValueError):
        valid = False
    if not valid:
        raise RuntimeError(
            "Missing/invalid fitted Phase-2 tau or calibration; rebuild stage2-cache"
        )
