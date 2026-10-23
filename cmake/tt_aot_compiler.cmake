# Copyright 2026 The tt-iree Authors
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

# Offline Metal compilation is shared by the compiler plugin and build-time
# HAL transfer artifact generation. Runtime-only builds need no MLIR libraries.
function(tt_iree_add_aot_compiler)
  if(TARGET TenstorrentAOTCompiler)
    return()
  endif()
  get_filename_component(_root "${CMAKE_CURRENT_FUNCTION_LIST_DIR}/.." ABSOLUTE)
  set(_compiler "${_root}/compiler/plugins/target/tenstorrent")
  add_library(TenstorrentAOTCompiler STATIC
    "${_compiler}/TenstorrentAOTCompiler.cpp"
    "${_compiler}/TenstorrentAOTCompiler.h"
    "${_compiler}/TenstorrentNativeKernels.cpp"
    "${_compiler}/TenstorrentNativeKernels.h")
  target_compile_features(TenstorrentAOTCompiler PRIVATE cxx_std_20)
  target_compile_options(TenstorrentAOTCompiler PRIVATE -fsized-deallocation)
  set_property(TARGET TenstorrentAOTCompiler PROPERTY CXX_SCAN_FOR_MODULES OFF)
  target_include_directories(TenstorrentAOTCompiler BEFORE PRIVATE
    "${_compiler}"
    "${TT_METAL_HOME}/tt_metal/api"
    "${TT_METAL_HOME}/tt_metal/api/tt-metalium"
    "${TT_METAL_HOME}/tt_metal"
    "${TT_METAL_HOME}/tt_metal/llrt"
    "${TT_METAL_HOME}/tt_metal/hostdevcommon/api"
    "${TT_METAL_HOME}/tt_stl"
    ${TT_METAL_INCLUDE_DIRS}
    "${_root}/runtime/src")
  target_link_libraries(TenstorrentAOTCompiler PRIVATE
    ${TT_METAL_LIBRARIES} "${TT_METAL_LIB_DIR}/libfmt.so"
    tt_executable_builder_util)
  target_compile_definitions(TenstorrentAOTCompiler PRIVATE
    SPDLOG_FMT_EXTERNAL=1 TT_IREE_TT_METAL_HOME="${TT_METAL_HOME}")
  set_target_properties(TenstorrentAOTCompiler PROPERTIES
    POSITION_INDEPENDENT_CODE ON BUILD_RPATH "${TT_METAL_LIB_DIR}")
endfunction()
