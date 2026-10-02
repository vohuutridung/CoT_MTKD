from __future__ import annotations

from ..config import load_config
from ..evaluation.stage1_runner import evaluate_stage1_experts
from ..utils.local_logging import configure_logging
from .common import config_parser


def main() -> None:
    parser = config_parser("Evaluate each Stage-1 LoRA expert in turn with vLLM")
    parser.add_argument(
        "--experts",
        default="",
        help="Comma-separated expert names (default: every expert in the Stage-1 manifest)",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Regenerate even when cached predictions match the current settings",
    )
    arguments = parser.parse_args()
    config = load_config(arguments.config, arguments.set)
    configure_logging(0)
    experts = [name.strip() for name in arguments.experts.split(",") if name.strip()]
    evaluate_stage1_experts(config, experts or None, force=arguments.force)


if __name__ == "__main__":
    main()
