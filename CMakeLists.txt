# Copyright 2025 The tt-iree Authors
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

cmake_minimum_required(VERSION 3.21)

project(tt-iree
  VERSION 0.1.0
  DESCRIPTION "IREE Backend for Tenstorrent AI Accelerators"
  LANGUAGES C CXX
)

set(CMAKE_CXX_STANDARD 17)
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
set(IREE_BUILD_TESTS OFF CACHE BOOL "" FORCE)
set(IREE_BUILD_SAMPLES OFF CACHE BOOL "" FORCE)
set(IREE_HAL_DRIVER_DEFAULTS OFF CACHE BOOL "" FORCE)

# Enable only necessary HAL drivers for testing
set(IREE_HAL_DRIVER_LOCAL_SYNC ON CACHE BOOL "" FORCE)
set(IREE_HAL_DRIVER_LOCAL_TASK ON CACHE BOOL "" FORCE)

# Add IREE
if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/third_party/iree/CMakeLists.txt")
  message(STATUS "Adding IREE from third_party/iree")
  add_subdirectory(third_party/iree EXCLUDE_FROM_ALL)
else()
  message(FATAL_ERROR "IREE not found. Run: git submodule update --init --recursive")
endif()

#-------------------------------------------------------------------------------
# TT-Metal Configuration (when not in mock mode)
#-------------------------------------------------------------------------------

if(NOT TT_IREE_ENABLE_MOCK)
  include(cmake/tt_metal.cmake)
  find_tt_metal()
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
# Subdirectories
#-------------------------------------------------------------------------------

if(TT_IREE_BUILD_COMPILER)
  add_subdirectory(compiler)
endif()

if(TT_IREE_BUILD_RUNTIME)
  add_subdirectory(runtime)
endif()

if(TT_IREE_BUILD_TESTS)
  enable_testing()
  add_subdirectory(test)
endif()

add_subdirectory(tools)
