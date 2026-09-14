from __future__ import annotations

import os
from dataclasses import dataclass
from datetime import timedelta
from typing import Iterable

import torch
import torch.distributed as dist


@dataclass(frozen=True)
class DistributedContext:
    rank: int
    local_rank: int
    world_size: int
    device: torch.device

    @property
    def is_main(self) -> bool:
        return self.rank == 0


def initialize_distributed(timeout_minutes: int = 60) -> DistributedContext:
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    rank = int(os.environ.get("RANK", "0"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    if torch.cuda.is_available():
        device = torch.device("cuda", local_rank)
        torch.cuda.set_device(device)
        backend = "nccl"
    else:
        device = torch.device("cpu")
        backend = "gloo"
    if world_size > 1 and not dist.is_initialized():
        dist.init_process_group(
            backend=backend, timeout=timedelta(minutes=timeout_minutes)
        )
    return DistributedContext(rank, local_rank, world_size, device)


def barrier() -> None:
    if dist.is_available() and dist.is_initialized():
        dist.barrier()


def all_reduce_tensor(tensor: torch.Tensor, average: bool = False) -> torch.Tensor:
    if dist.is_available() and dist.is_initialized():
        dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
        if average:
            tensor.div_(dist.get_world_size())
    return tensor


def all_reduce_grad_lists(grad_lists: Iterable[Iterable[torch.Tensor]]) -> None:
    if not (dist.is_available() and dist.is_initialized()):
        return
    for gradients in grad_lists:
        for gradient in gradients:
            dist.all_reduce(gradient, op=dist.ReduceOp.SUM)


def shard_indices(length: int, rank: int, world_size: int) -> range:
    return range(rank, length, world_size)


def cleanup_distributed() -> None:
    if dist.is_available() and dist.is_initialized():
        dist.destroy_process_group()
