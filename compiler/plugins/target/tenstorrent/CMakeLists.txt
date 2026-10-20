# Copyright 2025 The tt-iree Authors
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#
# Tenstorrent HAL target backend compiler plugin.
# Depends on tt_executable_builder_util from runtime/src/iree/schema/.

message(STATUS "tt-iree: Configuring Tenstorrent compiler target plugin")

add_library(TenstorrentTarget STATIC
  TenstorrentAOTCompiler.h
  TenstorrentPreprocessingPass.h
  TenstorrentPreprocessingPass.cpp
  TenstorrentTarget.h
  TenstorrentTarget.cpp
  TenstorrentRegistration.cpp
)

if(TT_IREE_ENABLE_TTNN)
  add_library(TenstorrentAOTCompiler STATIC
    TenstorrentAOTCompiler.cpp
    TenstorrentAOTCompiler.h
    TenstorrentNativeKernels.cpp
    TenstorrentNativeKernels.h
  )
  target_compile_features(TenstorrentAOTCompiler PRIVATE cxx_std_20)
  target_compile_options(TenstorrentAOTCompiler PRIVATE -fsized-deallocation)
  set_property(TARGET TenstorrentAOTCompiler PROPERTY CXX_SCAN_FOR_MODULES OFF)
  target_include_directories(TenstorrentAOTCompiler BEFORE
    PRIVATE
      ${CMAKE_CURRENT_SOURCE_DIR}
      "${TT_METAL_HOME}/tt_metal/api"
      "${TT_METAL_HOME}/tt_metal/api/tt-metalium"
      "${TT_METAL_HOME}/tt_metal"
      "${TT_METAL_HOME}/tt_metal/llrt"
      "${TT_METAL_HOME}/tt_metal/hostdevcommon/api"
      "${TT_METAL_HOME}/tt_stl"
      ${TT_METAL_INCLUDE_DIRS}
      "${CMAKE_CURRENT_SOURCE_DIR}/../../../../runtime/src"
  )
  target_link_libraries(TenstorrentAOTCompiler
    PRIVATE
      ${TT_METAL_LIBRARIES}
      "${TT_METAL_LIB_DIR}/libfmt.so"
      tt_executable_builder_util
  )
  target_compile_definitions(TenstorrentAOTCompiler
    PRIVATE
      SPDLOG_FMT_EXTERNAL=1
      TT_IREE_TT_METAL_HOME="${TT_METAL_HOME}"
  )
  set_target_properties(TenstorrentAOTCompiler PROPERTIES
    POSITION_INDEPENDENT_CODE ON
    BUILD_RPATH "${TT_METAL_LIB_DIR}"
  )
  target_compile_definitions(TenstorrentTarget PRIVATE TT_IREE_ENABLE_AOT=1)
  target_link_libraries(TenstorrentTarget PUBLIC TenstorrentAOTCompiler)
endif()

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
    MLIRPass
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
