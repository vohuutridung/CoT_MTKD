from __future__ import annotations

from ..config import load_config
from ..stage2.merge import MERGE_METHODS
from ..stage2.trainer import train_stage2
from ..utils.distributed import cleanup_distributed, initialize_distributed
from ..utils.local_logging import configure_logging
from ..utils.seed import seed_everything
from .common import config_parser


def main() -> None:
    parser = config_parser(
        "Train the merged LoRA student with council-weighted NLL distillation"
    )
    parser.add_argument(
        "--merge-method",
        choices=list(MERGE_METHODS),
        default=None,
        help="Override stage2.merge_method (ta, ties, dare_ties, tsv, iso_c)",
    )
    arguments = parser.parse_args()
    overrides = list(arguments.set)
    if arguments.merge_method is not None:
        overrides.append(f"stage2.merge_method={arguments.merge_method}")
    config = load_config(arguments.config, overrides)
    distributed = initialize_distributed()
    configure_logging(distributed.rank)
    seed_everything(int(config["seed"]) + distributed.rank)
    try:
        train_stage2(config, distributed)
    finally:
        cleanup_distributed()


if __name__ == "__main__":
    main()
