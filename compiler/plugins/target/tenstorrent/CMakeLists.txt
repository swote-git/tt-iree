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
    # IREE_SOURCE_DIR is set by IREE's own CMakeLists when IREE is the top-level project.
    ${IREE_SOURCE_DIR}/compiler/src
)

target_link_libraries(TenstorrentTarget
  PUBLIC
    iree_compiler_Codegen_Common_Common
    iree_compiler_Dialect_HAL_Target_Target
    iree_compiler_PluginAPI_PluginAPI
    MLIRIR
    MLIRSupport
    tt_executable_builder_util
)

target_compile_features(TenstorrentTarget PUBLIC cxx_std_17)

set_target_properties(TenstorrentTarget PROPERTIES
  POSITION_INDEPENDENT_CODE ON
)

# Match IREE compiler build flags: no RTTI, no exceptions, hidden visibility.
target_compile_options(TenstorrentTarget PRIVATE
  -fno-rtti
  -fno-exceptions
  -fvisibility=hidden
)
