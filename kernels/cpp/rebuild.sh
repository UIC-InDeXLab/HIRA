#!/usr/bin/env bash
# Rebuild all C++ kernels in this directory from scratch.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

echo "==> Cleaning previous build artefacts..."
make clean

echo "==> Building kernels..."
make -j"$(nproc)"

echo "==> Done."