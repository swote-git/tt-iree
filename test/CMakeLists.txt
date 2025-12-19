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

# TODO: Add actual tests
# add_subdirectory(e2e)
