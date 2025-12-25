# Copyright 2025 The tt-iree Authors
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

# Find and configure TT-Metal SDK v0.65.0 for Blackhole architecture

function(find_tt_metal)
  # Check for TT_METAL_HOME environment variable
  if(DEFINED ENV{TT_METAL_HOME})
    set(TT_METAL_HOME "$ENV{TT_METAL_HOME}")
  elseif(EXISTS "${CMAKE_SOURCE_DIR}/third_party/tt-metal")
    set(TT_METAL_HOME "${CMAKE_SOURCE_DIR}/third_party/tt-metal")
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

  # Export TT_METAL_HOME
  set(TT_METAL_HOME "${TT_METAL_HOME}" PARENT_SCOPE)

  #-----------------------------------------------------------------------------
  # Include directories
  #-----------------------------------------------------------------------------
  set(TT_METAL_INCLUDE_DIRS
    "${TT_METAL_HOME}"
    "${TT_METAL_HOME}/tt_metal"
    "${TT_METAL_HOME}/tt_metal/include"
    "${TT_METAL_HOME}/tt_metal/common"
    "${TT_METAL_HOME}/tt_metal/hw/inc"
    "${TT_METAL_HOME}/tt_metal/hw/inc/blackhole"
    "${TT_METAL_HOME}/tt_metal/third_party/umd"
    "${TT_METAL_HOME}/tt_metal/third_party/fmt"
    "${TT_METAL_HOME}/ttnn"
    "${TT_METAL_HOME}/ttnn/cpp"
    PARENT_SCOPE
  )

  #-----------------------------------------------------------------------------
  # Find TT-Metal libraries
  #-----------------------------------------------------------------------------
  set(TT_METAL_LIB_DIR "${TT_METAL_HOME}/build/lib")
  
  if(NOT EXISTS "${TT_METAL_LIB_DIR}")
    message(FATAL_ERROR 
      "TT-Metal libraries not found at ${TT_METAL_LIB_DIR}\n"
      "Please build TT-Metal first: cd ${TT_METAL_HOME} && ./build_metal.sh")
  endif()

  find_library(TT_METAL_LIB NAMES tt_metal
    PATHS "${TT_METAL_LIB_DIR}" NO_DEFAULT_PATH REQUIRED)
  message(STATUS "Found tt_metal: ${TT_METAL_LIB}")

  find_library(TT_DEVICE_LIB NAMES device
    PATHS "${TT_METAL_LIB_DIR}" NO_DEFAULT_PATH)
  if(TT_DEVICE_LIB)
    message(STATUS "Found device: ${TT_DEVICE_LIB}")
  endif()

  set(TT_METAL_LIBRARIES ${TT_METAL_LIB} PARENT_SCOPE)
  if(TT_DEVICE_LIB)
    set(TT_DEVICE_LIBRARIES ${TT_DEVICE_LIB} PARENT_SCOPE)
  endif()
  set(TT_METAL_LIB_DIR "${TT_METAL_LIB_DIR}" PARENT_SCOPE)

  # Architecture
  if(DEFINED ENV{ARCH_NAME})
    set(TT_ARCH "$ENV{ARCH_NAME}" PARENT_SCOPE)
  else()
    set(TT_ARCH "blackhole" PARENT_SCOPE)
  endif()
  message(STATUS "TT-Metal architecture: ${TT_ARCH}")
endfunction()

#-------------------------------------------------------------------------------
# Helper to link TT-Metal to a target
#-------------------------------------------------------------------------------
function(tt_metal_target_link_libraries target)
  target_include_directories(${target} PRIVATE ${TT_METAL_INCLUDE_DIRS})
  target_link_libraries(${target} PRIVATE ${TT_METAL_LIBRARIES})
  
  if(TT_DEVICE_LIBRARIES)
    target_link_libraries(${target} PRIVATE ${TT_DEVICE_LIBRARIES})
  endif()

  set_target_properties(${target} PROPERTIES
    BUILD_RPATH "${TT_METAL_LIB_DIR}"
    INSTALL_RPATH "${TT_METAL_LIB_DIR}"
  )

  target_compile_definitions(${target} PRIVATE
    ARCH_BLACKHOLE=1
    TT_METAL_VERSIM_DISABLED=1
  )
endfunction()
