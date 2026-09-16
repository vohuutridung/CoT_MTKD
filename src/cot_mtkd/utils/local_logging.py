from __future__ import annotations

import json
import logging
import time
from pathlib import Path
from typing import Any

LOGGER = logging.getLogger(__name__)


def configure_logging(rank: int = 0) -> None:
    level = logging.INFO if rank == 0 else logging.WARNING
    logging.basicConfig(
        level=level,
        format=f"%(asctime)s | rank={rank} | %(levelname)s | %(name)s | %(message)s",
        force=True,
    )


class JsonlLogger:
    def __init__(
        self, path: str | Path, enabled: bool = True, truncate: bool = False
    ) -> None:
        self.path = Path(path)
        self.enabled = enabled
        if enabled:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            if truncate:
                self.path.write_text("", encoding="utf-8")

    def log(self, event: str, **values: Any) -> None:
        if not self.enabled:
            return
        record = {"time": time.time(), "event": event, **values}
        with self.path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record, ensure_ascii=False, default=str) + "\n")
        summary_keys = (
            "step",
            "epoch",
            "sft_nll",
            "dpp_loss",
            "learning_rate",
            "mean_selected_k",
            "mean_grassmann_distance",
            "hard_loss",
            "kd_loss",
            "total_loss",
        )
        summary = {key: values[key] for key in summary_keys if key in values}
        LOGGER.info("%s %s", event, summary)
