#!/bin/bash
# Copyright 2025 The tt-iree Authors
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
BUILD_DIR="${ROOT_DIR}/build"

# Default options
BUILD_TYPE="RelWithDebInfo"
ENABLE_MOCK="ON"
ENABLE_TTNN="OFF"
JOBS=$(nproc 2>/dev/null || sysctl -n hw.ncpu 2>/dev/null || echo 4)

# Parse arguments
while [[ $# -gt 0 ]]; do
  case $1 in
    --debug)
      BUILD_TYPE="Debug"
      shift
      ;;
    --release)
      BUILD_TYPE="Release"
      shift
      ;;
    --no-mock)
      ENABLE_MOCK="OFF"
      shift
      ;;
    --ttnn)
      ENABLE_TTNN="ON"
      shift
      ;;
    --jobs|-j)
      JOBS="$2"
      shift 2
      ;;
    --clean)
      echo "Cleaning build directory..."
      rm -rf "${BUILD_DIR}"
      shift
      ;;
    --help|-h)
      echo "Usage: $0 [options]"
      echo "Options:"
      echo "  --debug       Build with debug symbols"
      echo "  --release     Build optimized release"
      echo "  --no-mock     Disable mock mode (requires hardware)"
      echo "  --ttnn        Enable TTNN integration"
      echo "  --jobs N      Number of parallel jobs (default: $(nproc))"
      echo "  --clean       Clean build directory first"
      echo "  --help        Show this help"
      exit 0
      ;;
    *)
      echo "Unknown option: $1"
      exit 1
      ;;
  esac
done

echo "=== tt-iree Build ==="
echo "  Build type: ${BUILD_TYPE}"
echo "  Mock mode: ${ENABLE_MOCK}"
echo "  TTNN: ${ENABLE_TTNN}"
echo "  Jobs: ${JOBS}"
echo ""

# Configure
echo "=== Configuring ==="
cmake -G Ninja -B "${BUILD_DIR}" \
  -DCMAKE_BUILD_TYPE="${BUILD_TYPE}" \
  -DCMAKE_C_COMPILER="${CC:-clang}" \
  -DCMAKE_CXX_COMPILER="${CXX:-clang++}" \
  -DTT_IREE_ENABLE_MOCK="${ENABLE_MOCK}" \
  -DTT_IREE_ENABLE_TTNN="${ENABLE_TTNN}"

# Build
echo ""
echo "=== Building ==="
cmake --build "${BUILD_DIR}" -j "${JOBS}"

echo ""
echo "=== Build complete ==="
echo "Binaries in: ${BUILD_DIR}"
