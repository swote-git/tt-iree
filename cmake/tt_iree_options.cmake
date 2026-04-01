# Copyright 2025 The tt-iree Authors
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#
# Shared build options for tt-iree plugins.
# Included by iree_compiler_plugin.cmake and iree_runtime_plugin.cmake.

option(TT_IREE_ENABLE_MOCK "Enable mock mode (no hardware required)" OFF)
option(TT_IREE_ENABLE_TTNN "Enable TT-Metal/TTNN hardware integration" ON)
