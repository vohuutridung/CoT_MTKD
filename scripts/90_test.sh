#!/usr/bin/env bash
source "$(dirname "${BASH_SOURCE[0]}")/_common.sh"
"$PYTHON_BIN" -m unittest discover -s tests -v

