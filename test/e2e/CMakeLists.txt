# Copyright 2025 The tt-iree Authors
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

if(NOT TARGET iree-compile)
  message(STATUS "tt-iree: Skipping compiler E2E tests (iree-compile unavailable)")
  return()
endif()

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

if(TT_IREE_ENABLE_TTNN)
  add_test(
    NAME tt_e2e_check_builtin_aot
    COMMAND ${CMAKE_COMMAND}
      -DIREE_COMPILE=$<TARGET_FILE:iree-compile>
      -DSOURCE_DIR=${CMAKE_CURRENT_SOURCE_DIR}
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

if(TT_IREE_ENABLE_TTNN)
  add_test(
    NAME tt_e2e_compile_resnet50_projection
    COMMAND ${CMAKE_COMMAND}
      -DIREE_COMPILE=$<TARGET_FILE:iree-compile>
      -DSOURCE=${CMAKE_CURRENT_SOURCE_DIR}/resnet50_projection.mlir
      -DOUTPUT=${CMAKE_CURRENT_BINARY_DIR}/resnet50_projection_aot.vmfb
      -DDEVICE_ID=p100a
      -P ${CMAKE_CURRENT_SOURCE_DIR}/check_resnet50_projection_aot.cmake
  )
  set_tests_properties(tt_e2e_compile_resnet50_projection PROPERTIES
    LABELS "tt-iree;e2e;compile;model;aot"
    RUN_SERIAL TRUE
    TIMEOUT 300
  )
else()
  tt_iree_add_compile_smoke_test(
    NAME tt_e2e_compile_resnet50_projection
    SRC resnet50_projection.mlir
  )
endif()

tt_iree_add_compile_smoke_test(
  NAME tt_e2e_compile_resnet50_conv2d
  SRC resnet50_conv2d.mlir
)

tt_iree_add_compile_smoke_test(
  NAME tt_e2e_compile_resnet50_stem_conv
  SRC resnet50_stem_conv.mlir
)

tt_iree_add_compile_smoke_test(
  NAME tt_e2e_compile_resnet50_input_pad
  SRC resnet50_input_pad.mlir
)

tt_iree_add_compile_smoke_test(
  NAME tt_e2e_compile_resnet50_max_pool
  SRC resnet50_max_pool.mlir
)

tt_iree_add_compile_smoke_test(
  NAME tt_e2e_compile_resnet50_per_channel_projection
  SRC resnet50_per_channel_projection.mlir
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

add_test(
  NAME tt_e2e_check_resnet50_projection_workgroup_count
  COMMAND ${CMAKE_COMMAND}
    -DIREE_COMPILE=$<TARGET_FILE:iree-compile>
    -DSOURCE=${CMAKE_CURRENT_SOURCE_DIR}/resnet50_projection.mlir
    -DOUTPUT=${CMAKE_CURRENT_BINARY_DIR}/resnet50_projection_multicore_hal.mlir
    -P ${CMAKE_CURRENT_SOURCE_DIR}/check_multicore_workgroup_count.cmake
)
set_tests_properties(tt_e2e_check_resnet50_projection_workgroup_count PROPERTIES
  LABELS "tt-iree;e2e;compile;model"
)

add_test(
  NAME tt_e2e_check_resnet50_per_channel_projection_workgroup_count
  COMMAND ${CMAKE_COMMAND}
    -DIREE_COMPILE=$<TARGET_FILE:iree-compile>
    -DSOURCE=${CMAKE_CURRENT_SOURCE_DIR}/resnet50_per_channel_projection.mlir
    -DOUTPUT=${CMAKE_CURRENT_BINARY_DIR}/resnet50_per_channel_projection_multicore_hal.mlir
    -DEXPECTED_X=2
    -DEXPECTED_Y=7
    -P ${CMAKE_CURRENT_SOURCE_DIR}/check_multicore_workgroup_count.cmake
)
set_tests_properties(
  tt_e2e_check_resnet50_per_channel_projection_workgroup_count PROPERTIES
  LABELS "tt-iree;e2e;compile;model"
)

add_test(
  NAME tt_e2e_resnet50_bottleneck_pipeline
  COMMAND ${CMAKE_COMMAND}
    -DIREE_COMPILE=$<TARGET_FILE:iree-compile>
    -DIREE_RUN_MODULE=$<TARGET_FILE:iree-run-module>
    -DSOURCE=${CMAKE_CURRENT_SOURCE_DIR}/resnet50_bottleneck.mlir
    -DOUTPUT_DIR=${CMAKE_CURRENT_BINARY_DIR}/resnet50_bottleneck
    -P ${CMAKE_CURRENT_SOURCE_DIR}/check_resnet50_bottleneck.cmake
)
set_tests_properties(tt_e2e_resnet50_bottleneck_pipeline PROPERTIES
  LABELS "tt-iree;e2e;compile;model"
)

if(TT_IREE_ENABLE_MOCK OR TT_IREE_ENABLE_TTNN)
  add_test(
    NAME tt_e2e_resnet50_projection_runtime
    COMMAND ${CMAKE_COMMAND}
      -DIREE_COMPILE=$<TARGET_FILE:iree-compile>
      -DIREE_RUN_MODULE=$<TARGET_FILE:iree-run-module>
      -DSOURCE=${CMAKE_CURRENT_SOURCE_DIR}/resnet50_projection.mlir
      -DOUTPUT=${CMAKE_CURRENT_BINARY_DIR}/resnet50_projection_runtime.vmfb
      -P ${CMAKE_CURRENT_SOURCE_DIR}/check_resnet50_projection_runtime.cmake
  )
  if(TT_IREE_ENABLE_MOCK)
    set_tests_properties(tt_e2e_resnet50_projection_runtime PROPERTIES
      LABELS "tt-iree;e2e;compile;mock;model"
    )
  else()
    set_tests_properties(tt_e2e_resnet50_projection_runtime PROPERTIES
      ENVIRONMENT "TT_METAL_RUNTIME_ROOT=${CMAKE_SOURCE_DIR}/third_party/tt-metal"
      LABELS "tt-iree;e2e;compile;hardware;model;aot"
      RUN_SERIAL TRUE
      TIMEOUT 300
    )
  endif()
endif()

if(TT_IREE_ENABLE_MOCK)
  add_test(
    NAME tt_e2e_resnet50_conv2d_runtime_boundary
    COMMAND ${CMAKE_COMMAND}
      -DIREE_COMPILE=$<TARGET_FILE:iree-compile>
      -DIREE_RUN_MODULE=$<TARGET_FILE:iree-run-module>
      -DSOURCE=${CMAKE_CURRENT_SOURCE_DIR}/resnet50_conv2d.mlir
      -DOUTPUT=${CMAKE_CURRENT_BINARY_DIR}/resnet50_conv2d_runtime.vmfb
      -P ${CMAKE_CURRENT_SOURCE_DIR}/check_resnet50_conv2d_runtime_boundary.cmake
  )
  set_tests_properties(tt_e2e_resnet50_conv2d_runtime_boundary PROPERTIES
    LABELS "tt-iree;e2e;compile;mock;model"
  )

  add_test(
    NAME tt_e2e_resnet50_stem_conv_runtime
    COMMAND ${CMAKE_COMMAND}
      -DIREE_COMPILE=$<TARGET_FILE:iree-compile>
      -DIREE_RUN_MODULE=$<TARGET_FILE:iree-run-module>
      -DSOURCE=${CMAKE_CURRENT_SOURCE_DIR}/resnet50_stem_conv.mlir
      -DOUTPUT=${CMAKE_CURRENT_BINARY_DIR}/resnet50_stem_conv_runtime.vmfb
      -P ${CMAKE_CURRENT_SOURCE_DIR}/check_resnet50_stem_conv_runtime.cmake
  )
  set_tests_properties(tt_e2e_resnet50_stem_conv_runtime PROPERTIES
    LABELS "tt-iree;e2e;compile;mock;model"
  )

  add_test(
    NAME tt_e2e_resnet50_max_pool_runtime
    COMMAND ${CMAKE_COMMAND}
      -DIREE_COMPILE=$<TARGET_FILE:iree-compile>
      -DIREE_RUN_MODULE=$<TARGET_FILE:iree-run-module>
      -DSOURCE=${CMAKE_CURRENT_SOURCE_DIR}/resnet50_max_pool.mlir
      -DOUTPUT=${CMAKE_CURRENT_BINARY_DIR}/resnet50_max_pool_runtime.vmfb
      -P ${CMAKE_CURRENT_SOURCE_DIR}/check_resnet50_max_pool_runtime.cmake
  )
  set_tests_properties(tt_e2e_resnet50_max_pool_runtime PROPERTIES
    LABELS "tt-iree;e2e;compile;mock;model"
  )

  add_test(
    NAME tt_e2e_resnet50_per_channel_projection_runtime
    COMMAND ${CMAKE_COMMAND}
      -DIREE_COMPILE=$<TARGET_FILE:iree-compile>
      -DIREE_RUN_MODULE=$<TARGET_FILE:iree-run-module>
      -DSOURCE=${CMAKE_CURRENT_SOURCE_DIR}/resnet50_per_channel_projection.mlir
      -DOUTPUT=${CMAKE_CURRENT_BINARY_DIR}/resnet50_per_channel_projection_runtime.vmfb
      -P ${CMAKE_CURRENT_SOURCE_DIR}/check_resnet50_per_channel_projection_runtime.cmake
  )
  set_tests_properties(
    tt_e2e_resnet50_per_channel_projection_runtime PROPERTIES
    LABELS "tt-iree;e2e;compile;mock;model"
  )

  add_test(
    NAME tt_e2e_resnet50_bottleneck_runtime_boundary
    COMMAND ${CMAKE_COMMAND}
      -DIREE_COMPILE=$<TARGET_FILE:iree-compile>
      -DIREE_RUN_MODULE=$<TARGET_FILE:iree-run-module>
      -DSOURCE=${CMAKE_CURRENT_SOURCE_DIR}/resnet50_bottleneck.mlir
      -DOUTPUT=${CMAKE_CURRENT_BINARY_DIR}/resnet50_bottleneck_runtime.vmfb
      -P ${CMAKE_CURRENT_SOURCE_DIR}/check_resnet50_bottleneck_runtime_boundary.cmake
  )
  set_tests_properties(tt_e2e_resnet50_bottleneck_runtime_boundary PROPERTIES
    LABELS "tt-iree;e2e;compile;mock;model"
  )
endif()
