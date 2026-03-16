# Copyright 2025 The tt-iree Authors
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#
# Included by IREE's iree_include_cmake_plugin_dirs() during CMake configuration.
# At this point TenstorrentTarget already exists (created by add_subdirectory(compiler)
# in the top-level CMakeLists before IREE was added).

if(TARGET TenstorrentTarget)
  iree_compiler_register_plugin(
    PLUGIN_ID "hal_target_tenstorrent"
    TARGET    TenstorrentTarget
  )
endif()
