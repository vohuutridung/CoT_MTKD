from __future__ import annotations

from ..stage2.cache_builder import build_teacher_cache
from ..utils.distributed import cleanup_distributed, initialize_distributed
from ..utils.local_logging import configure_logging
from ..utils.seed import seed_everything
from .common import parsed_config


def main() -> None:
    config = parsed_config("Compile Top-512 plus tail-bucket teacher targets")
    distributed = initialize_distributed()
    configure_logging(distributed.rank)
    seed_everything(int(config["seed"]) + distributed.rank)
    try:
        build_teacher_cache(config, distributed)
    finally:
        cleanup_distributed()


if __name__ == "__main__":
    main()
