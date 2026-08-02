# Copyright 2025 The tt-iree Authors
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

message(STATUS "tt-iree: Configuring Tenstorrent HAL driver")

#-------------------------------------------------------------------------------
# Source files
#-------------------------------------------------------------------------------

set(TT_HAL_DRIVER_SRCS
  tt_driver.cc
  tt_device.cc
  tt_allocator.cc
  tt_buffer.cc
  tt_executable.cc
  tt_semaphore.cc
  tt_command_buffer.cc
  tt_executable_cache.cc
  registration/driver_module.c
)

set(TT_HAL_DRIVER_HDRS
  api.h
  tt_driver.h
  tt_device.h
  tt_allocator.h
  tt_buffer.h
  tt_executable.h
  tt_semaphore.h
  tt_command_buffer.h
  tt_executable_cache.h
  registration/driver_module.h
)

#-------------------------------------------------------------------------------
# Build library
#-------------------------------------------------------------------------------

# Create HAL driver library
add_library(iree_hal_tenstorrent STATIC
  ${TT_HAL_DRIVER_SRCS}
  ${TT_HAL_DRIVER_HDRS}
)

# TT-Metal requires C++20 (CMAKE_CXX_STANDARD 20 in tt-metal/CMakeLists.txt).
target_compile_features(iree_hal_tenstorrent PRIVATE cxx_std_20)

if(TT_IREE_ENABLE_TTNN)
  target_include_directories(iree_hal_tenstorrent
    PRIVATE
      ${TT_METAL_INCLUDE_DIRS}
  )

  target_link_libraries(iree_hal_tenstorrent
    PRIVATE
      ${TT_METAL_LIBRARIES}
      "${TT_METAL_LIB_DIR}/libfmt.so"
  )

  # TTEX FlatBuffer schema support.
  # Expand tt_executable_def_c_fbs's include dirs via generator expression instead of
  # target_link_libraries to avoid creating a CMake target dep that would require
  # tt_executable_def_c_fbs to appear in IREE's install export set.
  if(TARGET tt_executable_def_c_fbs)
    target_include_directories(iree_hal_tenstorrent PRIVATE
      $<TARGET_PROPERTY:tt_executable_def_c_fbs,INTERFACE_INCLUDE_DIRECTORIES>
    )
    target_link_libraries(iree_hal_tenstorrent PRIVATE flatcc::runtime)
    add_dependencies(iree_hal_tenstorrent tt_executable_def_gen)
    target_compile_definitions(iree_hal_tenstorrent PRIVATE TT_IREE_ENABLE_TTEX=1)
    message(STATUS "  TTEX FlatBuffer support: enabled")
  else()
    message(STATUS "  TTEX FlatBuffer support: disabled (schema not built)")
  endif()
  
  target_compile_definitions(iree_hal_tenstorrent
    PRIVATE
      SPDLOG_FMT_EXTERNAL=1
  )
  set_target_properties(iree_hal_tenstorrent PROPERTIES
    BUILD_RPATH "${TT_METAL_LIB_DIR}"
    INSTALL_RPATH "${TT_METAL_LIB_DIR}"
  )
endif()

# Set include directories
# Resolve to absolute path so CMake doesn't warn about relative INTERFACE paths.
get_filename_component(_TT_IREE_RUNTIME_SRC
  "${CMAKE_CURRENT_SOURCE_DIR}/../../../.." ABSOLUTE)
target_include_directories(iree_hal_tenstorrent
  PUBLIC
    # runtime/src — allows: #include "iree/hal/drivers/tenstorrent/tt_buffer.h"
    ${_TT_IREE_RUNTIME_SRC}
)

# Link against IREE HAL
# Note: These targets come from the IREE submodule
target_link_libraries(iree_hal_tenstorrent
  PUBLIC
    iree_base_base
    iree_hal_hal
    iree_hal_utils_semaphore_base
)

# Add math library (for sqrt in tile conversion)
target_link_libraries(iree_hal_tenstorrent
  PRIVATE
    m
)

#-------------------------------------------------------------------------------
# Compile definitions
#-------------------------------------------------------------------------------

# Mock mode flag (already set globally, but repeat for clarity)
if(TT_IREE_ENABLE_MOCK)
  target_compile_definitions(iree_hal_tenstorrent
    PRIVATE
      TT_IREE_ENABLE_MOCK=1
  )
endif()

# TTNN flag (for future hardware mode)
if(TT_IREE_ENABLE_TTNN)
  target_compile_definitions(iree_hal_tenstorrent
    PRIVATE
      TT_IREE_ENABLE_TTNN=1
  )
endif()

#-------------------------------------------------------------------------------
# Installation
#-------------------------------------------------------------------------------

# When included by IREE via iree_register_external_hal_driver(SOURCE_DIR ...),
# the directory inherits IREE_INSTALL_LIBRARY_TARGETS_DEFAULT_COMPONENT and
# IREE_INSTALL_LIBRARY_TARGETS_DEFAULT_EXPORT_SET from the drivers/ scope.
# iree_install_targets() adds iree_hal_tenstorrent to IREE's export set so that
# the CMake generate step doesn't error on missing export set membership.
if(COMMAND iree_install_targets)
  iree_install_targets(
    TARGETS iree_hal_tenstorrent
    HDRS    ${TT_HAL_DRIVER_HDRS}
  )
else()
  install(FILES ${TT_HAL_DRIVER_HDRS}
    DESTINATION include/iree/hal/drivers/tenstorrent
  )
  install(TARGETS iree_hal_tenstorrent
    ARCHIVE DESTINATION lib
    LIBRARY DESTINATION lib
  )
endif()

#-------------------------------------------------------------------------------
# Status
#-------------------------------------------------------------------------------

message(STATUS "  Tenstorrent HAL driver configured")
message(STATUS "    Mock mode: ${TT_IREE_ENABLE_MOCK}")
message(STATUS "    TTNN mode: ${TT_IREE_ENABLE_TTNN}")
message(STATUS "    Source files: ${TT_HAL_DRIVER_SRCS}")
