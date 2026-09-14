from __future__ import annotations

from ..evaluation.runner import evaluate
from ..utils.distributed import cleanup_distributed, initialize_distributed
from ..utils.local_logging import configure_logging
from ..utils.seed import seed_everything
from .common import parsed_config


def main() -> None:
    config = parsed_config("Evaluate the final adapter with P-ALIGN settings")
    distributed = initialize_distributed()
    configure_logging(distributed.rank)
    seed_everything(int(config["seed"]) + distributed.rank)
    try:
        evaluate(config, distributed)
    finally:
        cleanup_distributed()


if __name__ == "__main__":
    main()
