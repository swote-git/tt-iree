#!/bin/bash
# Copyright 2025 The tt-iree Authors
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#
# tt-iree Development Environment Setup
# Usage: source scripts/env_setup.sh
#
# This script sets up the environment for tt-iree development with tt-metal.

set -e

# Get the root directory of tt-iree
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export TT_IREE_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

#-------------------------------------------------------------------------------
# TT-Metal Configuration
#-------------------------------------------------------------------------------

export TT_METAL_HOME="${TT_IREE_ROOT}/third_party/tt-metal"
export ARCH_NAME=blackhole

# Verify tt-metal exists
if [ ! -d "${TT_METAL_HOME}" ]; then
    echo "ERROR: TT-Metal not found at ${TT_METAL_HOME}"
    echo "Run: git submodule update --init --recursive"
    return 1
fi

# Verify build exists
if [ ! -d "${TT_METAL_HOME}/build_Release" ]; then
    echo "ERROR: TT-Metal not built. Run:"
    echo "  cd ${TT_METAL_HOME}"
    echo "  ./build_metal.sh"
    return 1
fi

#-------------------------------------------------------------------------------
# Python Environment
#-------------------------------------------------------------------------------

VENV_PATH="${TT_METAL_HOME}/build/python_env"

if [ ! -d "${VENV_PATH}" ]; then
    echo "WARNING: Python virtual environment not found at ${VENV_PATH}"
    echo "Creating virtual environment..."
    python3 -m venv "${VENV_PATH}"
fi

# Activate virtual environment
source "${VENV_PATH}/bin/activate"

#-------------------------------------------------------------------------------
# Python Path Configuration
#-------------------------------------------------------------------------------

# ttnn Python package (source tree)
export PYTHONPATH="${TT_METAL_HOME}/ttnn:${PYTHONPATH}"

# tracy tools
export PYTHONPATH="${TT_METAL_HOME}/tools:${PYTHONPATH}"

# Build artifacts
export PYTHONPATH="${TT_METAL_HOME}/build_Release:${PYTHONPATH}"

#-------------------------------------------------------------------------------
# Library Path Configuration
#-------------------------------------------------------------------------------

export LD_LIBRARY_PATH="${TT_METAL_HOME}/build_Release/lib:${LD_LIBRARY_PATH}"

#-------------------------------------------------------------------------------
# Symlink for _ttnn.so (if not exists)
#-------------------------------------------------------------------------------

TTNN_SO_LINK="${TT_METAL_HOME}/ttnn/ttnn/_ttnn.so"
TTNN_SO_TARGET="${TT_METAL_HOME}/build_Release/ttnn/_ttnn.so"

if [ ! -L "${TTNN_SO_LINK}" ] && [ -f "${TTNN_SO_TARGET}" ]; then
    echo "Creating symlink for _ttnn.so..."
    ln -sf "${TTNN_SO_TARGET}" "${TTNN_SO_LINK}"
fi

#-------------------------------------------------------------------------------
# Verification
#-------------------------------------------------------------------------------

echo ""
echo "╔══════════════════════════════════════════════════════════════════╗"
echo "║              tt-iree Development Environment                     ║"
echo "╠══════════════════════════════════════════════════════════════════╣"
echo "║ TT_IREE_ROOT:   ${TT_IREE_ROOT}"
echo "║ TT_METAL_HOME:  ${TT_METAL_HOME}"
echo "║ ARCH_NAME:      ${ARCH_NAME}"
echo "║ Python:         $(which python3)"
echo "║ Virtual Env:    ${VIRTUAL_ENV}"
echo "╚══════════════════════════════════════════════════════════════════╝"
echo ""

# Quick verification
python3 -c "import ttnn; print('✓ TTNN imported successfully')" 2>/dev/null || {
    echo "WARNING: TTNN import failed. Check dependencies."
}

echo ""
echo "Environment ready! You can now:"
echo "  - Run tt-iree tests"
echo "  - Build tt-iree: ./scripts/build.sh"
echo "  - Test device: python3 -c \"import ttnn; d=ttnn.open_device(device_id=0); print(d); ttnn.close_device(d)\""
echo ""