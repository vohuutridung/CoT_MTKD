from __future__ import annotations

from ..stage1.trainer import train_stage1
from ..utils.distributed import cleanup_distributed, initialize_distributed
from ..utils.local_logging import configure_logging
from ..utils.seed import seed_everything
from .common import parsed_config


def main() -> None:
    config = parsed_config("Train five good-and-diverse GAC-CoT LoRA experts")
    distributed = initialize_distributed()
    configure_logging(distributed.rank)
    seed_everything(
        int(config["seed"]) + distributed.rank,
        bool(config["runtime"].get("deterministic_algorithms", False)),
    )
    try:
        train_stage1(config, distributed)
    finally:
        cleanup_distributed()


if __name__ == "__main__":
    main()
