from __future__ import annotations

from pathlib import Path

from datasets import load_dataset

from ..data.prepare import write_prepared_dataset
from ..models.multi_adapter import load_tokenizer
from ..utils.local_logging import configure_logging
from .common import parsed_config


def _records(config):
    dataset = config["dataset"]
    local_json = dataset.get("local_json")
    if local_json:
        return load_dataset(
            "json",
            data_files=str(Path(local_json)),
            split=dataset.get("split", "train"),
        )
    return load_dataset(
        dataset["name"],
        split=dataset.get("split", "train"),
        revision=dataset.get("revision"),
    )


def main() -> None:
    config = parsed_config("Prepare canonical s1K-1.1 Long-CoT records")
    configure_logging()
    tokenizer = load_tokenizer(config["model"])
    write_prepared_dataset(_records(config), tokenizer, config)


if __name__ == "__main__":
    main()
