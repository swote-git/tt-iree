# Copyright 2025 The tt-iree Authors
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#-------------------------------------------------------------------------------
# tt-iree E2E Tests
#
#   compiler/            Compile smoke tests and compiler ABI/selection contracts
#   native_f32/          F32 device-native matmul/tail kernels on hardware
#   native_bf16/         BF16 device-native kernels (batch, spatial, Tensix)
#   resnet50/            ResNet50 operator fixtures, pipelines, and model tools
#   mnist/               MNIST MLP fixtures and session gates
#-------------------------------------------------------------------------------

if(NOT TARGET iree-compile)
  message(STATUS "tt-iree: Skipping compiler E2E tests (iree-compile unavailable)")
  return()
endif()

find_program(_TT_IREE_PYTHON3_EXECUTABLE NAMES python3)

# Root for tests that reference sources from another group.
set(_TT_IREE_E2E_SOURCE_DIR "${CMAKE_CURRENT_SOURCE_DIR}")

function(tt_iree_add_compile_smoke_test)
  cmake_parse_arguments(
    _RULE
    ""
    "NAME;SRC"
    ""
    ${ARGN}
  )

  get_filename_component(_SRC_NAME "${_RULE_SRC}" NAME_WE)
  set(_OUTPUT_PATH "${CMAKE_CURRENT_BINARY_DIR}/${_SRC_NAME}.vmfb")

  add_test(
    NAME ${_RULE_NAME}
    COMMAND $<TARGET_FILE:iree-compile>
      --iree-hal-target-backends=tenstorrent
      ${CMAKE_CURRENT_SOURCE_DIR}/${_RULE_SRC}
      -o ${_OUTPUT_PATH}
  )
  set_tests_properties(${_RULE_NAME} PROPERTIES LABELS "tt-iree;e2e;compile")
endfunction()

add_subdirectory(compiler)
add_subdirectory(native_f32)
add_subdirectory(native_bf16)
add_subdirectory(resnet50)
add_subdirectory(mnist)
