from __future__ import annotations

import argparse
from typing import Any

from ..config import load_config


def config_parser(description: str) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument(
        "--config", required=True, help="Path to a YAML configuration file"
    )
    parser.add_argument(
        "--set",
        action="append",
        default=[],
        metavar="KEY=VALUE",
        help="Override a dotted configuration key; may be repeated",
    )
    return parser


def parsed_config(description: str) -> dict[str, Any]:
    arguments = config_parser(description).parse_args()
    return load_config(arguments.config, arguments.set)
