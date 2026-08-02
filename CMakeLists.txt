# Copyright 2025 The tt-iree Authors
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

cmake_minimum_required(VERSION 3.21)

project(tt-iree
  VERSION 0.1.0
  DESCRIPTION "IREE Backend for Tenstorrent AI Accelerators"
  LANGUAGES C CXX
)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

#-------------------------------------------------------------------------------
# Options
#-------------------------------------------------------------------------------

option(TT_IREE_BUILD_COMPILER "Build compiler plugin" ON)
option(TT_IREE_BUILD_RUNTIME "Build runtime driver" ON)
option(TT_IREE_BUILD_TESTS "Build tests" ON)
option(TT_IREE_ENABLE_MOCK "Enable mock mode (no hardware required)" ON)
option(TT_IREE_ENABLE_TTNN "Enable TTNN integration" OFF)


message(STATUS "tt-iree configuration:")
message(STATUS "  TT_IREE_BUILD_COMPILER: ${TT_IREE_BUILD_COMPILER}")
message(STATUS "  TT_IREE_BUILD_RUNTIME: ${TT_IREE_BUILD_RUNTIME}")
message(STATUS "  TT_IREE_BUILD_TESTS: ${TT_IREE_BUILD_TESTS}")
message(STATUS "  TT_IREE_ENABLE_MOCK: ${TT_IREE_ENABLE_MOCK}")
message(STATUS "  TT_IREE_ENABLE_TTNN: ${TT_IREE_ENABLE_TTNN}")

#-------------------------------------------------------------------------------
# IREE Configuration
#-------------------------------------------------------------------------------

# Configure IREE build options
set(IREE_BUILD_COMPILER ${TT_IREE_BUILD_COMPILER} CACHE BOOL "" FORCE)
set(IREE_BUILD_TESTS ${TT_IREE_BUILD_TESTS} CACHE BOOL "" FORCE)
set(IREE_BUILD_SAMPLES OFF CACHE BOOL "" FORCE)
set(IREE_HAL_DRIVER_DEFAULTS OFF CACHE BOOL "" FORCE)
# Disable IREE install targets — we're embedding IREE, not installing it.
# This avoids export-set errors when external HAL driver targets (iree_hal_tenstorrent)
# are linked into iree_hal_drivers_drivers but not in IREE's IREEExported-Runtime set.
set(IREE_ENABLE_INSTALL OFF CACHE BOOL "" FORCE)

# Enable only necessary HAL drivers for testing
set(IREE_HAL_DRIVER_LOCAL_SYNC ON CACHE BOOL "" FORCE)
set(IREE_HAL_DRIVER_LOCAL_TASK ON CACHE BOOL "" FORCE)

# Disable unused input plugins (avoid large submodules like torch-mlir)
set(IREE_INPUT_TORCH OFF CACHE BOOL "" FORCE)
set(IREE_INPUT_STABLEHLO OFF CACHE BOOL "" FORCE)
set(IREE_INPUT_TOSA OFF CACHE BOOL "" FORCE)

# Register the out-of-tree compiler and runtime plugins before configuring IREE.
# IREE then includes the plugin targets in its compiler and HAL driver
# registries, including command-line tools such as iree-run-module.
if(TT_IREE_BUILD_COMPILER OR TT_IREE_BUILD_RUNTIME)
  list(APPEND IREE_CMAKE_BUILTIN_PLUGIN_PATHS
    "${CMAKE_CURRENT_SOURCE_DIR}")
endif()

if(TT_IREE_BUILD_RUNTIME)
  list(APPEND IREE_EXTERNAL_HAL_DRIVERS tenstorrent)
  list(REMOVE_DUPLICATES IREE_EXTERNAL_HAL_DRIVERS)
  set(IREE_EXTERNAL_HAL_DRIVERS "${IREE_EXTERNAL_HAL_DRIVERS}"
    CACHE STRING "External IREE HAL drivers to build" FORCE)
endif()

# Add IREE
if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/third_party/iree/CMakeLists.txt")
  message(STATUS "Adding IREE from third_party/iree")
  add_subdirectory(third_party/iree EXCLUDE_FROM_ALL)
else()
  message(FATAL_ERROR "IREE not found. Run: git submodule update --init --recursive")
endif()

#-------------------------------------------------------------------------------
# Compile Definitions
#-------------------------------------------------------------------------------

if(TT_IREE_ENABLE_MOCK)
  add_compile_definitions(TT_IREE_ENABLE_MOCK=1)
endif()

if(TT_IREE_ENABLE_TTNN)
  add_compile_definitions(TT_IREE_ENABLE_TTNN=1)
endif()

#-------------------------------------------------------------------------------
# Project Tests and Tools
#-------------------------------------------------------------------------------

if(TT_IREE_BUILD_TESTS)
  enable_testing()
  add_subdirectory(test)
endif()

add_subdirectory(tools)
