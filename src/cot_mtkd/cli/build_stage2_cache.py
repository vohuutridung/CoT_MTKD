from __future__ import annotations

from ..stage2.council_cache import build_council_cache
from ..utils.distributed import cleanup_distributed, initialize_distributed
from ..utils.local_logging import configure_logging
from ..utils.seed import seed_everything
from .common import parsed_config


def main() -> None:
    config = parsed_config("Precompute Phase-2 support+tail targets and best-expert initialization")
    distributed = initialize_distributed()
    configure_logging(distributed.rank)
    seed_everything(int(config["seed"]) + distributed.rank)
    try:
        build_council_cache(config, distributed)
    finally:
        cleanup_distributed()


if __name__ == "__main__":
    main()
