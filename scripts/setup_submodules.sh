#!/bin/bash
# Copyright 2025 The tt-iree Authors
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"

# Pinned versions
IREE_VERSION="v3.9.0"
TT_METAL_VERSION="v0.65.0"

echo "=== Setting up tt-iree submodules ==="
echo "  IREE version: ${IREE_VERSION}"
echo "  tt-metal version: ${TT_METAL_VERSION}"
echo ""

cd "${ROOT_DIR}"

# Initialize submodules (without recursive for now)
echo "=== Initializing submodules ==="
git submodule init

# Update IREE to specific version
echo ""
echo "=== Fetching IREE ${IREE_VERSION} ==="
git submodule update --init third_party/iree
cd third_party/iree
git fetch --tags
git checkout ${IREE_VERSION}
echo "IREE checked out to ${IREE_VERSION}"

# Initialize IREE's submodules (this takes a while)
echo ""
echo "=== Initializing IREE submodules (this may take a while) ==="
git submodule update --init --recursive

cd "${ROOT_DIR}"

# Update tt-metal to specific version
echo ""
echo "=== Fetching tt-metal ${TT_METAL_VERSION} ==="
git submodule update --init third_party/tt-metal
cd third_party/tt-metal
git fetch --tags
git checkout ${TT_METAL_VERSION}
echo "tt-metal checked out to ${TT_METAL_VERSION}"

# Initialize tt-metal's submodules
echo ""
echo "=== Initializing tt-metal submodules ==="
git submodule update --init --recursive

cd "${ROOT_DIR}"

echo ""
echo "=== Submodules initialized ==="
echo "  - third_party/iree @ ${IREE_VERSION}"
echo "  - third_party/tt-metal @ ${TT_METAL_VERSION}"
echo ""
echo "Next steps:"
echo "  1. Configure: cmake -G Ninja -B build -DTT_IREE_ENABLE_MOCK=ON"
echo "  2. Build: cmake --build build"
