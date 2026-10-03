from __future__ import annotations

from ..stage2.teachers import ensure_stage2_teachers
from ..utils.local_logging import configure_logging
from .common import parsed_config


def main() -> None:
    config = parsed_config("Download/import trained Stage-1 teachers for Phase 2")
    configure_logging(0)
    ensure_stage2_teachers(config)


if __name__ == "__main__":
    main()
