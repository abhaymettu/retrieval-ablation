#!/usr/bin/env bash
# End to end, from an empty checkout. Downloads BEIR, embeds, retrieves,
# fuses, reranks, times, evaluates, and regenerates README.md.
set -euo pipefail
cd "$(dirname "$0")"

[ -d .venv ] || make venv
make all
