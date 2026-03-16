# Copyright 2025 The tt-iree Authors
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#
# Tenstorrent HAL target backend compiler plugin.
# Depends on tt_executable_builder_util from runtime/src/iree/schema/.

message(STATUS "tt-iree: Configuring Tenstorrent compiler target plugin")

add_library(TenstorrentTarget STATIC
  TenstorrentTarget.h
  TenstorrentTarget.cpp
  TenstorrentRegistration.cpp
)

target_include_directories(TenstorrentTarget
  PUBLIC
    # Own directory: makes #include "TenstorrentTarget.h" work in sibling files
    ${CMAKE_CURRENT_SOURCE_DIR}
    # IREE compiler headers: iree/compiler/Dialect/HAL/...
    ${CMAKE_SOURCE_DIR}/third_party/iree/compiler/src
)

target_link_libraries(TenstorrentTarget
  PUBLIC
    iree_compiler_Dialect_HAL_Target_Target
    iree_compiler_PluginAPI_PluginAPI
    MLIRIR
    MLIRSupport
    tt_executable_builder_util
)

target_compile_features(TenstorrentTarget PUBLIC cxx_std_17)
