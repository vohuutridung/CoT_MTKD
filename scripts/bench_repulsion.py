"""Benchmark projection-distance repulsion against the legacy geodesic force.

CPU runs a reduced grid. Pass ``--full`` for
M in {2,4,8}, r in {8,16,64}, d in {1024,4096}, L in {32,224}.
The script reports timings and cosine similarity. It does not claim accuracy.
"""

from __future__ import annotations

import argparse
from collections import OrderedDict

import torch

from cot_mtkd.stage1.grassmann import grassmann_repulsion_updates
from cot_mtkd.stage1.repulsion import gram_stats, kernel_from_d2, pairwise_d2, repulsion_force


def _projection_call(factors: torch.Tensor) -> torch.Tensor:
    stats = gram_stats(factors)
    distance = pairwise_d2(stats[3], None, rank=factors.shape[2], n_modules=factors.shape[0])
    kernel, _bandwidth = kernel_from_d2(distance)
    return repulsion_force(factors, stats, kernel, n_modules=factors.shape[0])


def _legacy_call(factors: torch.Tensor) -> list[list[torch.Tensor]]:
    layers, experts, rank, width = factors.shape
    groups = []
    for expert in range(experts):
        group: OrderedDict[str, torch.nn.Parameter] = OrderedDict()
        for layer in range(layers):
            matrix = factors[layer, expert].detach().to(dtype=torch.float32)
            group[f"m{layer}.lora_A.weight"] = torch.nn.Parameter(matrix.clone())
            group[f"m{layer}.lora_B.weight"] = torch.nn.Parameter(
                torch.eye(width, rank, dtype=torch.float32)
            )
        groups.append(group)
    updates, _kernel, _distances, _bandwidth = grassmann_repulsion_updates(groups)
    return updates


def _time_call(function, repeats: int, warmup: int) -> float:
    import time

    for _ in range(warmup):
        function()
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    samples = []
    for _ in range(repeats):
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        clock = time.perf_counter()
        function()
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        samples.append((time.perf_counter() - clock) * 1.0e3)
    samples.sort()
    return samples[len(samples) // 2]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--full", action="store_true")
    parser.add_argument("--repeats", type=int, default=20)
    args = parser.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda" or args.full:
        grid = [
            (experts, rank, width, layers)
            for experts in (2, 4, 8)
            for rank in (8, 16, 64)
            for width in (1024, 4096)
            for layers in (32, 224)
        ]
    else:
        print("CPU detected; running a reduced grid. Pass --full for the paper grid.")
        grid = [(2, 8, 128, 4), (4, 16, 256, 8)]
        args.repeats = min(args.repeats, 5)
    print(f"device={device} repeats={args.repeats}")
    print(f"{'M':>3} {'r':>4} {'d':>6} {'L':>4} {'proj_ms':>10} {'legacy_ms':>10} {'cosine':>8}")
    for experts, rank, width, layers in grid:
        factors = torch.randn(layers, experts, rank, width, device=device)
        projection_ms = _time_call(lambda: _projection_call(factors), args.repeats, warmup=2)
        if layers <= 8 and width <= 1024:
            legacy_ms = _time_call(lambda: _legacy_call(factors.cpu()), max(2, args.repeats // 4), warmup=1)
            new_force = _projection_call(factors.cpu())
            old = _legacy_call(factors.cpu())
            new_flat = torch.cat([new_force[:, expert].reshape(-1) for expert in range(experts)])
            old_flat = torch.cat(
                [old[expert][2 * layer].reshape(-1) for expert in range(experts) for layer in range(layers)]
            )
            cosine = float(
                torch.nn.functional.cosine_similarity(new_flat, old_flat, dim=0)
            )
        else:
            legacy_ms = float("nan")
            cosine = float("nan")
        print(
            f"{experts:3d} {rank:4d} {width:6d} {layers:4d} "
            f"{projection_ms:10.2f} {legacy_ms:10.2f} {cosine:8.3f}"
        )

    profile_factors = torch.randn(4, 2, 8, 64, device=device)
    for name, function in (
        ("projection_closed_form", lambda: _projection_call(profile_factors)),
        ("geodesic_autograd", lambda: _legacy_call(profile_factors.detach().cpu())),
    ):
        with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU]) as profile:
            function()
        print(f"\nTop ops for {name}")
        print(profile.key_averages().table(sort_by="cpu_time_total", row_limit=15))


if __name__ == "__main__":
    main()
