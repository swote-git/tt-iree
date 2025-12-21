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

# Check if this is a git repository
if [ ! -d ".git" ]; then
  echo "Error: Not a git repository. Run 'git init' first."
  exit 1
fi

#-------------------------------------------------------------------------------
# IREE
#-------------------------------------------------------------------------------
echo "=== Setting up IREE ${IREE_VERSION} ==="

if [ ! -d "third_party/iree/.git" ] && [ ! -f "third_party/iree/.git" ]; then
  echo "Adding IREE submodule..."
  rm -rf third_party/iree
  git submodule add https://github.com/iree-org/iree.git third_party/iree
fi

git submodule update --init third_party/iree
cd third_party/iree
git fetch --tags --force
git checkout ${IREE_VERSION}
echo "IREE checked out to ${IREE_VERSION}"

echo ""
echo "Initializing IREE submodules (this may take 10-20 minutes)..."
git submodule update --init --recursive

cd "${ROOT_DIR}"

#-------------------------------------------------------------------------------
# tt-metal
#-------------------------------------------------------------------------------
echo ""
echo "=== Setting up tt-metal ${TT_METAL_VERSION} ==="

if [ ! -d "third_party/tt-metal/.git" ] && [ ! -f "third_party/tt-metal/.git" ]; then
  echo "Adding tt-metal submodule..."
  rm -rf third_party/tt-metal
  git submodule add https://github.com/tenstorrent/tt-metal.git third_party/tt-metal
fi

git submodule update --init third_party/tt-metal
cd third_party/tt-metal
git fetch --tags --force
git checkout ${TT_METAL_VERSION}
echo "tt-metal checked out to ${TT_METAL_VERSION}"

echo ""
echo "Initializing tt-metal submodules..."
git submodule update --init --recursive

cd "${ROOT_DIR}"

#-------------------------------------------------------------------------------
# Done
#-------------------------------------------------------------------------------
echo ""
echo "=== Submodules setup complete ==="
echo "  - third_party/iree @ ${IREE_VERSION}"
echo "  - third_party/tt-metal @ ${TT_METAL_VERSION}"
echo ""
git add .gitmodules third_party/iree third_party/tt-metal
git commit -m "Add submodules: IREE ${IREE_VERSION}, tt-metal ${TT_METAL_VERSION}" || echo "(already committed)"
echo ""
echo "Next steps:"
echo "  1. cmake -G Ninja -B build -DTT_IREE_ENABLE_MOCK=ON"
echo "  2. cmake --build build"
