from __future__ import annotations

import argparse
import os
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


def set_process_title() -> None:
    """Show GPU jobs under PROC_TITLE (e.g. in nvitop) when setproctitle is installed."""
    title = os.environ.get("PROC_TITLE")
    if not title:
        return
    try:
        import setproctitle
    except ImportError:
        return
    setproctitle.setproctitle(title)


def parsed_config(description: str) -> dict[str, Any]:
    set_process_title()
    arguments = config_parser(description).parse_args()
    return load_config(arguments.config, arguments.set)
