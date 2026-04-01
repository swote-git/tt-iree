# Copyright 2025 The tt-iree Authors
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#
# IREE runtime plugin for Tenstorrent HAL driver.
# Included by IREE via -DIREE_CMAKE_PLUGIN_PATHS=<tt-iree-root>

if(NOT "tenstorrent" IN_LIST IREE_EXTERNAL_HAL_DRIVERS)
  return()
endif()

set(TT_IREE_SOURCE_DIR "${CMAKE_CURRENT_LIST_DIR}")

include("${TT_IREE_SOURCE_DIR}/cmake/tt_iree_options.cmake")

# Find TT-Metal when hardware integration is enabled.
if(TT_IREE_ENABLE_TTNN)
  include("${TT_IREE_SOURCE_DIR}/cmake/tt_metal.cmake")
  find_tt_metal()
endif()

# Schema targets (may already exist if compiler plugin was processed first).
if(NOT TARGET tt_executable_def_c_fbs)
  add_subdirectory(
    "${TT_IREE_SOURCE_DIR}/runtime/src/iree/schema"
    "${CMAKE_CURRENT_BINARY_DIR}/tt_iree/schema"
  )
endif()

# Register with IREE's external HAL driver mechanism.
# SOURCE_DIR causes IREE to add_subdirectory our driver from within its own
# driver infrastructure (runtime/src/iree/hal/drivers/), which ensures the
# install export sets are handled correctly.
iree_register_external_hal_driver(
  NAME        tenstorrent
  SOURCE_DIR  "${TT_IREE_SOURCE_DIR}/runtime/src/iree/hal/drivers/tenstorrent"
  BINARY_DIR  "tenstorrent"
  DRIVER_TARGET iree_hal_tenstorrent
  REGISTER_FN   iree_hal_tenstorrent_driver_module_register
)
