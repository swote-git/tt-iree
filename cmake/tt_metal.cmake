# Copyright 2025 The tt-iree Authors
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

# Find and configure TT-Metal SDK v0.65.0 for Blackhole architecture

function(find_tt_metal)
  # Check for TT_METAL_HOME: CMake variable > env var > submodule paths
  if(NOT DEFINED TT_METAL_HOME)
    if(DEFINED ENV{TT_METAL_HOME})
      set(TT_METAL_HOME "$ENV{TT_METAL_HOME}")
    elseif(DEFINED TT_IREE_SOURCE_DIR AND EXISTS "${TT_IREE_SOURCE_DIR}/third_party/tt-metal")
      set(TT_METAL_HOME "${TT_IREE_SOURCE_DIR}/third_party/tt-metal")
    elseif(EXISTS "${CMAKE_SOURCE_DIR}/third_party/tt-metal")
      set(TT_METAL_HOME "${CMAKE_SOURCE_DIR}/third_party/tt-metal")
    else()
      message(FATAL_ERROR
        "TT-Metal not found. Either:\n"
        "  1. Pass -DTT_METAL_HOME=/path/to/tt-metal to cmake, or\n"
        "  2. Set TT_METAL_HOME environment variable, or\n"
        "  3. Initialize submodule: git submodule update --init third_party/tt-metal")
    endif()
  endif()

  message(STATUS "Using TT-Metal from: ${TT_METAL_HOME}")

  # Verify TT-Metal installation
  if(NOT EXISTS "${TT_METAL_HOME}/tt_metal")
    message(FATAL_ERROR "Invalid TT_METAL_HOME: ${TT_METAL_HOME}/tt_metal not found")
  endif()

  # Export TT_METAL_HOME as a cache variable so it's visible in all CMake scopes
  # (including driver CMakeLists added via iree_register_external_hal_driver SOURCE_DIR).
  set(TT_METAL_HOME "${TT_METAL_HOME}" CACHE PATH "TT-Metal root directory" FORCE)

  #-----------------------------------------------------------------------------
  # Include directories
  #-----------------------------------------------------------------------------
  set(_TT_METAL_INCLUDE_DIRS
    "${TT_METAL_HOME}"
    "${TT_METAL_HOME}/tt_metal"
    "${TT_METAL_HOME}/tt_metal/api"
    "${TT_METAL_HOME}/tt_metal/include"
    "${TT_METAL_HOME}/tt_metal/common"
    "${TT_METAL_HOME}/tt_metal/hw/inc"
    "${TT_METAL_HOME}/tt_metal/hw/inc/blackhole"
    "${TT_METAL_HOME}/tt_metal/third_party/umd"
    "${TT_METAL_HOME}/tt_metal/third_party/fmt"
    "${TT_METAL_HOME}/ttnn"
    "${TT_METAL_HOME}/ttnn/cpp"
    "${TT_METAL_HOME}/build_Release/include"
    "${TT_METAL_HOME}/build_Release/include/metalium-thirdparty"
  )
  set(TT_METAL_INCLUDE_DIRS "${_TT_METAL_INCLUDE_DIRS}" CACHE STRING "TT-Metal include directories" FORCE)
  set(TT_METAL_INCLUDE_DIRS "${_TT_METAL_INCLUDE_DIRS}" PARENT_SCOPE)

  #-----------------------------------------------------------------------------
  # Find TT-Metal libraries
  #-----------------------------------------------------------------------------
  if(EXISTS "${TT_METAL_HOME}/build_Release/lib")
    set(TT_METAL_LIB_DIR "${TT_METAL_HOME}/build_Release/lib")
  elseif(EXISTS "${TT_METAL_HOME}/build_Debug/lib")
    set(TT_METAL_LIB_DIR "${TT_METAL_HOME}/build_Debug/lib")
  elseif(EXISTS "${TT_METAL_HOME}/build/lib")
    set(TT_METAL_LIB_DIR "${TT_METAL_HOME}/build/lib")
  else()
    message(FATAL_ERROR "TT-Metal libraries not found")
  endif()

  set(TT_METAL_LIB_DIR "${TT_METAL_LIB_DIR}" CACHE PATH "TT-Metal library directory" FORCE)
  set(TT_METAL_LIB_DIR "${TT_METAL_LIB_DIR}" PARENT_SCOPE)
  message(STATUS "TT-Metal library directory: ${TT_METAL_LIB_DIR}")

  # Find libraries
  find_library(TT_METAL_LIB
    NAMES tt_metal
    PATHS ${TT_METAL_LIB_DIR}
    NO_DEFAULT_PATH
    REQUIRED
  )

  find_library(TT_DEVICE_LIB
    NAMES device
    PATHS ${TT_METAL_LIB_DIR}
    NO_DEFAULT_PATH
    REQUIRED
  )

  message(STATUS "Found TT-Metal: ${TT_METAL_LIB}")
  message(STATUS "Found TT-Device: ${TT_DEVICE_LIB}")

  # Set variables for linking
  set(TT_METAL_LIBRARIES "${TT_METAL_LIB};${TT_DEVICE_LIB}" CACHE STRING "TT-Metal libraries" FORCE)
  set(TT_METAL_LIBRARIES "${TT_METAL_LIB};${TT_DEVICE_LIB}" PARENT_SCOPE)

  # Architecture
  if(DEFINED ENV{ARCH_NAME})
    set(TT_ARCH "$ENV{ARCH_NAME}" CACHE STRING "TT-Metal architecture" FORCE)
  else()
    set(TT_ARCH "blackhole" CACHE STRING "TT-Metal architecture" FORCE)
  endif()
  set(TT_ARCH "${TT_ARCH}" PARENT_SCOPE)
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

  target_link_libraries(${target} PRIVATE "${TT_METAL_LIB_DIR}/libfmt.so")

  target_compile_definitions(${target} PRIVATE
    ARCH_BLACKHOLE=1
    TT_METAL_VERSIM_DISABLED=1
    SPDLOG_FMT_EXTERNAL=1
  )
endfunction()
