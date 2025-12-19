# Copyright 2025 The tt-iree Authors
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

# Find and configure TT-Metal SDK

function(find_tt_metal)
  # Check for TT_METAL_HOME environment variable
  if(DEFINED ENV{TT_METAL_HOME})
    set(TT_METAL_HOME "$ENV{TT_METAL_HOME}")
  elseif(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/third_party/tt-metal")
    set(TT_METAL_HOME "${CMAKE_CURRENT_SOURCE_DIR}/third_party/tt-metal")
  else()
    message(FATAL_ERROR 
      "TT-Metal not found. Either:\n"
      "  1. Set TT_METAL_HOME environment variable, or\n"
      "  2. Initialize submodule: git submodule update --init third_party/tt-metal")
  endif()

  message(STATUS "Using TT-Metal from: ${TT_METAL_HOME}")

  # Verify TT-Metal installation
  if(NOT EXISTS "${TT_METAL_HOME}/tt_metal")
    message(FATAL_ERROR "Invalid TT_METAL_HOME: ${TT_METAL_HOME}/tt_metal not found")
  endif()

  # Export variables for use in the build
  set(TT_METAL_HOME "${TT_METAL_HOME}" PARENT_SCOPE)
  set(TT_METAL_INCLUDE_DIRS
    "${TT_METAL_HOME}"
    "${TT_METAL_HOME}/tt_metal"
    "${TT_METAL_HOME}/ttnn"
    PARENT_SCOPE)

  # TODO: Find TT-Metal libraries
  # This will need to be adjusted based on how TT-Metal is built
endfunction()
