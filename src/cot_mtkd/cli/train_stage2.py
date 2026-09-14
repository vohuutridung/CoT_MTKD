from __future__ import annotations

from ..stage2.trainer import train_stage2
from ..utils.distributed import cleanup_distributed, initialize_distributed
from ..utils.local_logging import configure_logging
from ..utils.seed import seed_everything
from .common import parsed_config


def main() -> None:
    config = parsed_config("Train the single dual-source GAC-CoT-MTKD student")
    distributed = initialize_distributed()
    configure_logging(distributed.rank)
    seed_everything(int(config["seed"]) + distributed.rank)
    try:
        train_stage2(config, distributed)
    finally:
        cleanup_distributed()


if __name__ == "__main__":
    main()
