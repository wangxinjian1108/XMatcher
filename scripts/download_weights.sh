#!/usr/bin/env bash
set -euo pipefail
TARGET="${XMATCHER_WEIGHTS_DIR:-$HOME/.cache/xmatcher}"
mkdir -p "$TARGET"
METHODS="${1:-efficient_loftr}"
for m in $METHODS; do
    python scripts/_download.py --method "$m" --target "$TARGET"
done
