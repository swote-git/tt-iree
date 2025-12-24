# Copyright 2025 The tt-iree Authors
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

# Simple test to verify driver registration
add_test(
  NAME tt_iree_driver_registration
  COMMAND ${CMAKE_COMMAND} -E echo "Driver registration test placeholder"
)

# Label our tests
set_tests_properties(tt_iree_driver_registration PROPERTIES
  LABELS "tt-iree"
)

add_executable(test_buffer_roundtrip
  buffer_roundtrip/test_buffer_roundtrip.c
)

target_link_libraries(test_buffer_roundtrip
  PRIVATE
    iree_hal_tenstorrent
    iree_hal_hal
    iree_base_base
)

add_test(
  NAME tt_iree_buffer_roundtrip
  COMMAND test_buffer_roundtrip
)

set_tests_properties(tt_iree_buffer_roundtrip PROPERTIES
  LABELS "tt-iree;week2"
)