from __future__ import annotations

from ..evaluation.runner import evaluate
from ..utils.local_logging import configure_logging
from .common import parsed_config


def main() -> None:
    config = parsed_config("Evaluate the Phase-2 student with vLLM")
    configure_logging(0)
    evaluate(config)


if __name__ == "__main__":
    main()
