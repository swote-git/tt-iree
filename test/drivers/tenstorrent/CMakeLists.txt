# Copyright 2025 The tt-iree Authors
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#-------------------------------------------------------------------------------
# Tenstorrent HAL Driver Tests
#-------------------------------------------------------------------------------

# tile_layout_test: standalone, no IREE dependency
add_executable(tile_layout_test
  tile_layout_test.cc
)
target_compile_features(tile_layout_test PRIVATE cxx_std_17)
add_test(NAME tt_tile_layout_test COMMAND tile_layout_test)
set_tests_properties(tt_tile_layout_test PROPERTIES LABELS "tt-iree;unit")

# driver_test: driver registration and enumeration
add_executable(driver_test
  driver_test.cc
)
target_compile_features(driver_test PRIVATE cxx_std_17)
target_link_libraries(driver_test
  PRIVATE
    iree_hal_tenstorrent
    iree_base_base
    iree_hal_hal
)
target_include_directories(driver_test
  PRIVATE
    ${CMAKE_SOURCE_DIR}/runtime/src
)
add_test(NAME tt_driver_test COMMAND driver_test)
set_tests_properties(tt_driver_test PROPERTIES LABELS "tt-iree;hal")

# device_test: device creation and query
add_executable(device_test
  device_test.cc
)
target_compile_features(device_test PRIVATE cxx_std_17)
target_link_libraries(device_test
  PRIVATE
    iree_hal_tenstorrent
    iree_base_base
    iree_hal_hal
)
target_include_directories(device_test
  PRIVATE
    ${CMAKE_SOURCE_DIR}/runtime/src
)
add_test(NAME tt_device_test COMMAND device_test)
set_tests_properties(tt_device_test PROPERTIES LABELS "tt-iree;hal")

# buffer_test: buffer allocation and data transfer
add_executable(buffer_test
  buffer_test.cc
)
target_compile_features(buffer_test PRIVATE cxx_std_17)
target_link_libraries(buffer_test
  PRIVATE
    iree_hal_tenstorrent
    iree_base_base
    iree_hal_hal
)
target_include_directories(buffer_test
  PRIVATE
    ${CMAKE_SOURCE_DIR}/runtime/src
)
add_test(NAME tt_buffer_test COMMAND buffer_test)
set_tests_properties(tt_buffer_test PROPERTIES LABELS "tt-iree;hal")
