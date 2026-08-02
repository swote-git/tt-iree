# Copyright 2025 The tt-iree Authors
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

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

tt_iree_add_compile_smoke_test(
  NAME tt_e2e_compile_simple_add
  SRC simple_add.mlir
)

tt_iree_add_compile_smoke_test(
  NAME tt_e2e_compile_simple_matmul
  SRC simple_matmul.mlir
)

tt_iree_add_compile_smoke_test(
  NAME tt_e2e_compile_simple_matmul_64
  SRC simple_matmul_64.mlir
)

tt_iree_add_compile_smoke_test(
  NAME tt_e2e_compile_dynamic_matmul
  SRC dynamic_matmul.mlir
)
