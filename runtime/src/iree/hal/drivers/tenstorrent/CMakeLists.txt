# Copyright 2025 The tt-iree Authors
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

message(STATUS "tt-iree: Configuring Tenstorrent HAL driver")

#-------------------------------------------------------------------------------
# Tenstorrent HAL Driver
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

# For now, create a simple library
# TODO: Integrate properly with IREE's build system
add_library(iree_hal_tenstorrent STATIC
  ${TT_HAL_DRIVER_SRCS}
)

target_include_directories(iree_hal_tenstorrent
  PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${CMAKE_CURRENT_SOURCE_DIR}/../../..
)

# Link against IREE HAL
target_link_libraries(iree_hal_tenstorrent
  PUBLIC
    iree_base_base
    iree_hal_hal
)

#-------------------------------------------------------------------------------
# TTNN Bridge (optional)
#-------------------------------------------------------------------------------

if(TT_IREE_ENABLE_TTNN)
  add_subdirectory(ttnn_bridge)
  target_link_libraries(iree_hal_tenstorrent PRIVATE tt_iree_ttnn_bridge)
endif()
