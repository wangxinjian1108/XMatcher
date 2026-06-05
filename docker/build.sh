#!/usr/bin/env bash
set -euo pipefail
git submodule update --init --recursive
docker build -t xmatcher:dev -f docker/Dockerfile .
