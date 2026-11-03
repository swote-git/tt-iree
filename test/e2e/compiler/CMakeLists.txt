# Copyright 2025 The tt-iree Authors
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

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

add_test(
  NAME tt_e2e_check_dynamic_matmul_abi
  COMMAND ${CMAKE_COMMAND}
    -DIREE_COMPILE=$<TARGET_FILE:iree-compile>
    -DSOURCE=${CMAKE_CURRENT_SOURCE_DIR}/dynamic_matmul.mlir
    -DOUTPUT=${CMAKE_CURRENT_BINARY_DIR}/dynamic_matmul_abi.mlir
    -P ${CMAKE_CURRENT_SOURCE_DIR}/check_dynamic_matmul_abi.cmake
)
set_tests_properties(tt_e2e_check_dynamic_matmul_abi PROPERTIES
  LABELS "tt-iree;e2e;compile;abi"
)

foreach(_TT_MATMUL_CASE IN ITEMS static dynamic)
  if(_TT_MATMUL_CASE STREQUAL "static")
    set(_TT_MATMUL_SOURCE simple_matmul.mlir)
    set(_TT_MATMUL_PROGRAM 2)
    set(_TT_MATMUL_SHAPE_MODE static)
  else()
    set(_TT_MATMUL_SOURCE dynamic_matmul.mlir)
    set(_TT_MATMUL_PROGRAM 3)
    set(_TT_MATMUL_SHAPE_MODE dynamic)
  endif()
  add_test(
    NAME tt_e2e_check_${_TT_MATMUL_CASE}_matmul_selection
    COMMAND ${CMAKE_COMMAND}
      -DIREE_COMPILE=$<TARGET_FILE:iree-compile>
      -DSOURCE=${CMAKE_CURRENT_SOURCE_DIR}/${_TT_MATMUL_SOURCE}
      -DOUTPUT=${CMAKE_CURRENT_BINARY_DIR}/${_TT_MATMUL_CASE}_matmul_selection.mlir
      -DEXPECTED_PROGRAM=${_TT_MATMUL_PROGRAM}
      -DEXPECTED_SHAPE_MODE=${_TT_MATMUL_SHAPE_MODE}
      -P ${CMAKE_CURRENT_SOURCE_DIR}/check_matmul_selection_contract.cmake
  )
  set_tests_properties(
    tt_e2e_check_${_TT_MATMUL_CASE}_matmul_selection PROPERTIES
    LABELS "tt-iree;e2e;compile;abi"
  )
endforeach()

add_test(
  NAME tt_e2e_reject_reduction_as_add
  COMMAND ${CMAKE_COMMAND}
    -DIREE_COMPILE=$<TARGET_FILE:iree-compile>
    -DSOURCE_FILE=${CMAKE_CURRENT_SOURCE_DIR}/reduction_not_add.mlir
    -P ${CMAKE_CURRENT_SOURCE_DIR}/check_reject_reduction_as_add.cmake
)
set_tests_properties(tt_e2e_reject_reduction_as_add PROPERTIES
  LABELS "tt-iree;e2e;compile"
)

# Covers builtin AOT kernels across groups, so sources are e2e-root relative.
if(TT_IREE_ENABLE_TTNN)
  add_test(
    NAME tt_e2e_check_builtin_aot
    COMMAND ${CMAKE_COMMAND}
      -DIREE_COMPILE=$<TARGET_FILE:iree-compile>
      -DSOURCE_DIR=${_TT_IREE_E2E_SOURCE_DIR}
      -DOUTPUT_DIR=${CMAKE_CURRENT_BINARY_DIR}/builtin_aot
      -P ${CMAKE_CURRENT_SOURCE_DIR}/check_builtin_aot.cmake
  )
  set_tests_properties(tt_e2e_check_builtin_aot PROPERTIES
    LABELS "tt-iree;e2e;compile;aot"
    RUN_SERIAL TRUE
    TIMEOUT 300
  )
endif()

add_test(
  NAME tt_e2e_check_target_arch_options
  COMMAND ${CMAKE_COMMAND}
    -DIREE_COMPILE=$<TARGET_FILE:iree-compile>
    -DSOURCE=${CMAKE_CURRENT_SOURCE_DIR}/simple_matmul.mlir
    -DOUTPUT_DIR=${CMAKE_CURRENT_BINARY_DIR}/target_arch_options
    -P ${CMAKE_CURRENT_SOURCE_DIR}/check_target_arch_options.cmake
)
set_tests_properties(tt_e2e_check_target_arch_options PROPERTIES
  LABELS "tt-iree;e2e;compile;aot"
  TIMEOUT 60
)

add_test(
  NAME tt_e2e_check_multicore_workgroup_count
  COMMAND ${CMAKE_COMMAND}
    -DIREE_COMPILE=$<TARGET_FILE:iree-compile>
    -DSOURCE=${CMAKE_CURRENT_SOURCE_DIR}/simple_matmul_64.mlir
    -DOUTPUT=${CMAKE_CURRENT_BINARY_DIR}/simple_matmul_64_multicore_hal.mlir
    -P ${CMAKE_CURRENT_SOURCE_DIR}/check_multicore_workgroup_count.cmake
)
set_tests_properties(tt_e2e_check_multicore_workgroup_count PROPERTIES
  LABELS "tt-iree;e2e;compile"
)
