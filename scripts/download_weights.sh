#!/usr/bin/env bash
set -euo pipefail

TARGET="${XMATCHER_WEIGHTS_DIR:-$HOME/.cache/xmatcher}"
mkdir -p "$TARGET"

EXTRA_ARGS=()
METHODS=()
for arg in "$@"; do
    case "$arg" in
        --bootstrap) EXTRA_ARGS+=(--bootstrap) ;;
        --*) EXTRA_ARGS+=("$arg") ;;
        *) METHODS+=("$arg") ;;
    esac
done

if [ ${#METHODS[@]} -eq 0 ]; then
    METHODS=(efficient_loftr)
fi

for m in "${METHODS[@]}"; do
    python scripts/_download.py --method "$m" --target "$TARGET" "${EXTRA_ARGS[@]}"
done
