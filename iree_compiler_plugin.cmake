# Copyright 2025 The tt-iree Authors
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#
# IREE compiler plugin for Tenstorrent HAL target backend.
# Included by IREE via -DIREE_CMAKE_PLUGIN_PATHS=<tt-iree-root>

set(TT_IREE_SOURCE_DIR "${CMAKE_CURRENT_LIST_DIR}")

include("${TT_IREE_SOURCE_DIR}/cmake/tt_iree_options.cmake")

# Schema targets are needed by the compiler plugin (TTEX FlatBuffer builder).
# Guard against double-inclusion (runtime plugin may also add these).
if(NOT TARGET tt_executable_builder_util)
  add_subdirectory(
    "${TT_IREE_SOURCE_DIR}/runtime/src/iree/schema"
    "${CMAKE_CURRENT_BINARY_DIR}/tt_iree/schema"
  )
endif()

add_subdirectory(
  "${TT_IREE_SOURCE_DIR}/compiler/plugins/target/tenstorrent"
  "${CMAKE_CURRENT_BINARY_DIR}/tt_iree/TenstorrentTarget"
)

iree_compiler_register_plugin(
  PLUGIN_ID "hal_target_tenstorrent"
  TARGET    TenstorrentTarget
)
