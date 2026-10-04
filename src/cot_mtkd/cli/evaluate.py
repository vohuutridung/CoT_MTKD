from __future__ import annotations

from ..evaluation.runner import evaluate
from ..utils.local_logging import configure_logging
from .common import parsed_config


def main() -> None:
    config = parsed_config("Evaluate the Stage-2 student with the exp_s1k P-ALIGN protocol")
    configure_logging()
    evaluate(config)


if __name__ == "__main__":
    main()
