# Copyright 2025 The tt-iree Authors
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

message(STATUS "tt-iree: Configuring Tenstorrent HAL driver")

#-------------------------------------------------------------------------------
# Source files
#-------------------------------------------------------------------------------

set(TT_HAL_DRIVER_SRCS
  tt_driver.c
  tt_device.c
  tt_allocator.c
  tt_buffer.c
  tt_executable.c
  tt_semaphore.c
  tt_command_buffer.c
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

# Set include directories
# FIXED: Correct path to allow "iree/hal/drivers/tenstorrent/..." includes
target_include_directories(iree_hal_tenstorrent
  PUBLIC
    # Path to runtime/src (4 levels up from current directory)
    # This allows: #include "iree/hal/drivers/tenstorrent/tt_buffer.h"
    ${CMAKE_CURRENT_SOURCE_DIR}/../../../..
)

# Link against IREE HAL
# Note: These targets come from the IREE submodule
target_link_libraries(iree_hal_tenstorrent
  PUBLIC
    iree_base_base
    iree_hal_hal
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
# Installation (optional)
#-------------------------------------------------------------------------------

# Install headers (if needed for external use)
install(FILES ${TT_HAL_DRIVER_HDRS}
  DESTINATION include/iree/hal/drivers/tenstorrent
)

# Install library
install(TARGETS iree_hal_tenstorrent
  ARCHIVE DESTINATION lib
  LIBRARY DESTINATION lib
)

#-------------------------------------------------------------------------------
# Status
#-------------------------------------------------------------------------------

message(STATUS "  Tenstorrent HAL driver configured")
message(STATUS "    Mock mode: ${TT_IREE_ENABLE_MOCK}")
message(STATUS "    TTNN mode: ${TT_IREE_ENABLE_TTNN}")
message(STATUS "    Source files: ${TT_HAL_DRIVER_SRCS}")
