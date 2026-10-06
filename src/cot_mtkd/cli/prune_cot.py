from __future__ import annotations

from ..stage2.cot_prune_run import prune_dataset
from ..utils.distributed import cleanup_distributed, initialize_distributed
from ..utils.local_logging import configure_logging
from ..utils.seed import seed_everything
from .common import parsed_config


def main() -> None:
    config = parsed_config("Prune Phase-2 CoT traces with the frozen expert council")
    distributed = initialize_distributed()
    configure_logging(distributed.rank)
    seed_everything(int(config["seed"]) + distributed.rank)
    try:
        prune_dataset(config, distributed)
    finally:
        cleanup_distributed()


if __name__ == "__main__":
    main()
