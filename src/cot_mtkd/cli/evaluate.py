from __future__ import annotations

import os

from ..evaluation.runner import evaluate
from ..utils.local_logging import configure_logging
from .common import parsed_config


def main() -> None:
    title = os.environ.get("PROC_TITLE")
    if title:
        try:
            import setproctitle

            setproctitle.setproctitle(title)
        except ImportError:
            pass
    config = parsed_config("Evaluate the Stage-2 student with the exp_s1k P-ALIGN protocol")
    configure_logging()
    evaluate(config)


if __name__ == "__main__":
    main()
