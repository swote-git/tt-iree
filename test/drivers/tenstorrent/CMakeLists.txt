# Copyright 2025 The tt-iree Authors
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#-------------------------------------------------------------------------------
# Tenstorrent HAL Driver Tests
#-------------------------------------------------------------------------------

# tile_layout_test: standalone tile conversion (no HAL/TT-Metal dependency)
add_executable(tile_layout_test
  tile_layout_test.cc
)
target_compile_features(tile_layout_test PRIVATE cxx_std_17)
target_link_libraries(tile_layout_test
  PRIVATE
    iree_testing_gtest
    iree_testing_gtest_main
)
add_test(NAME tt_tile_layout_test COMMAND tile_layout_test)
set_tests_properties(tt_tile_layout_test PROPERTIES LABELS "tt-iree;unit")

#-------------------------------------------------------------------------------
# Helper: create a gtest-based HAL test
#-------------------------------------------------------------------------------
function(tt_iree_hal_test)
  cmake_parse_arguments(
    _RULE
    ""              # Options
    "NAME"          # One-value args
    "SRCS;DEPS"     # Multi-value args
    ${ARGN}
  )

  add_executable(${_RULE_NAME} ${_RULE_SRCS})
  target_compile_features(${_RULE_NAME} PRIVATE cxx_std_17)

  target_include_directories(${_RULE_NAME}
    PRIVATE
      ${CMAKE_SOURCE_DIR}/runtime/src
      ${CMAKE_CURRENT_SOURCE_DIR}
  )

  target_link_libraries(${_RULE_NAME}
    PRIVATE
      iree_hal_tenstorrent
      iree_base_base
      iree_hal_hal
      iree_testing_gtest
      iree_testing_gtest_main
      ${_RULE_DEPS}
  )

  if(TT_IREE_ENABLE_TTNN)
    target_link_libraries(${_RULE_NAME} PRIVATE "${TT_METAL_LIB_DIR}/libfmt.so")
    set_target_properties(${_RULE_NAME} PROPERTIES
      BUILD_RPATH "${TT_METAL_LIB_DIR}"
      INSTALL_RPATH "${TT_METAL_LIB_DIR}"
    )
  endif()

  # TT-Metal/OpenMPI may segfault during process exit cleanup (not in our
  # code). Wrap with shell to mask the signal; judge pass/fail by gtest output.
  add_test(NAME tt_${_RULE_NAME}
    COMMAND sh -c "$<TARGET_FILE:${_RULE_NAME}> 2>&1 || true")
  set_tests_properties(tt_${_RULE_NAME} PROPERTIES
    LABELS "tt-iree;hal"
    PASS_REGULAR_EXPRESSION "\\[  PASSED  \\]"
    FAIL_REGULAR_EXPRESSION "\\[  FAILED  \\]"
  )
endfunction()

#-------------------------------------------------------------------------------
# HAL Tests
#-------------------------------------------------------------------------------

tt_iree_hal_test(
  NAME driver_test
  SRCS driver_test.cc
)

tt_iree_hal_test(
  NAME device_test
  SRCS device_test.cc
)

tt_iree_hal_test(
  NAME buffer_test
  SRCS buffer_test.cc
)

tt_iree_hal_test(
  NAME dispatch_test
  SRCS dispatch_test.cc
)

# dispatch_test needs direct TT-Metal access for bfloat16 data transfer.
if(NOT TT_IREE_ENABLE_MOCK)
  tt_metal_target_link_libraries(dispatch_test)
endif()
