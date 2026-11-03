# Copyright 2025 The tt-iree Authors
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#-------------------------------------------------------------------------------
# Model tooling tests (host-only Python)
#-------------------------------------------------------------------------------

if(_TT_IREE_PYTHON3_EXECUTABLE)
  add_test(NAME tt_e2e_resnet50_bf16_batch_tools
    COMMAND "${_TT_IREE_PYTHON3_EXECUTABLE}"
      "${CMAKE_CURRENT_SOURCE_DIR}/resnet50_bf16_batch_tools_test.py")
  set_tests_properties(tt_e2e_resnet50_bf16_batch_tools PROPERTIES
    LABELS "tt-iree;e2e;unit;bf16;batch")
  add_test(NAME tt_e2e_ttnn_benchmark_metrics
    COMMAND "${_TT_IREE_PYTHON3_EXECUTABLE}"
      "${CMAKE_CURRENT_SOURCE_DIR}/ttnn_benchmark_test.py")
  set_tests_properties(tt_e2e_ttnn_benchmark_metrics PROPERTIES
    LABELS "tt-iree;e2e;profiling;ttnn" TIMEOUT 30)
  add_test(NAME tt_e2e_resnet50_dataset
    COMMAND "${_TT_IREE_PYTHON3_EXECUTABLE}"
      "${CMAKE_CURRENT_SOURCE_DIR}/resnet50_dataset_test.py")
  set_tests_properties(tt_e2e_resnet50_dataset PROPERTIES
    LABELS "tt-iree;e2e;correctness;golden;bf16" TIMEOUT 120)
  if(TARGET tt-iree-resnet-eval AND IREE_TARGET_BACKEND_LLVM_CPU AND IREE_HAL_DRIVER_LOCAL_SYNC)
    add_test(NAME tt_e2e_resnet50_session
      COMMAND "${_TT_IREE_PYTHON3_EXECUTABLE}"
        "${CMAKE_CURRENT_SOURCE_DIR}/resnet50_session_test.py"
        --iree-compile $<TARGET_FILE:iree-compile>
        --session-runner $<TARGET_FILE:tt-iree-resnet-eval>)
    set_tests_properties(tt_e2e_resnet50_session PROPERTIES
      LABELS "tt-iree;e2e;correctness;bf16;session" TIMEOUT 120)
  endif()
  add_test(NAME tt_e2e_resnet50_bf16_analysis
    COMMAND "${_TT_IREE_PYTHON3_EXECUTABLE}"
      "${CMAKE_CURRENT_SOURCE_DIR}/resnet50_bf16_analysis_test.py")
  set_tests_properties(tt_e2e_resnet50_bf16_analysis PROPERTIES
    LABELS "tt-iree;e2e;correctness;golden;bf16" TIMEOUT 120)
  add_test(NAME tt_e2e_resnet50_bf16_tools
    COMMAND "${_TT_IREE_PYTHON3_EXECUTABLE}"
      "${CMAKE_CURRENT_SOURCE_DIR}/resnet50_bf16_tools_test.py")
  set_tests_properties(tt_e2e_resnet50_bf16_tools PROPERTIES
    LABELS "tt-iree;e2e;correctness;golden;bf16")
  add_test(
    NAME tt_e2e_resnet50_precision_golden_tool
    COMMAND "${_TT_IREE_PYTHON3_EXECUTABLE}"
      "${CMAKE_CURRENT_SOURCE_DIR}/resnet50_precision_golden_test.py"
  )
  set_tests_properties(tt_e2e_resnet50_precision_golden_tool PROPERTIES
    LABELS "tt-iree;e2e;correctness;golden"
  )
  add_test(
    NAME tt_e2e_resnet50_correctness_tool
    COMMAND "${_TT_IREE_PYTHON3_EXECUTABLE}"
      "${CMAKE_CURRENT_SOURCE_DIR}/resnet50_correctness_test.py"
      "${CMAKE_SOURCE_DIR}/scripts/check_resnet50_correctness.py"
  )
  set_tests_properties(tt_e2e_resnet50_correctness_tool PROPERTIES
    LABELS "tt-iree;e2e;correctness"
  )
endif()

#-------------------------------------------------------------------------------
# Operator fixture compilation
#-------------------------------------------------------------------------------

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
  NAME tt_e2e_check_resnet50_projection_workgroup_count
  COMMAND ${CMAKE_COMMAND}
    -DIREE_COMPILE=$<TARGET_FILE:iree-compile>
    -DSOURCE=${CMAKE_CURRENT_SOURCE_DIR}/resnet50_projection.mlir
    -DOUTPUT=${CMAKE_CURRENT_BINARY_DIR}/resnet50_projection_multicore_hal.mlir
    -P ${_TT_IREE_E2E_SOURCE_DIR}/compiler/check_multicore_workgroup_count.cmake
)
set_tests_properties(tt_e2e_check_resnet50_projection_workgroup_count PROPERTIES
  LABELS "tt-iree;e2e;compile;model"
)

# Native matrix dispatch: 2 N-tiles x 98 M-tiles clamped to the Blackhole
# 11x10 grid, with unequal row partitions.
add_test(
  NAME tt_e2e_check_resnet50_per_channel_projection_workgroup_count
  COMMAND ${CMAKE_COMMAND}
    -DIREE_COMPILE=$<TARGET_FILE:iree-compile>
    -DSOURCE=${CMAKE_CURRENT_SOURCE_DIR}/resnet50_per_channel_projection.mlir
    -DOUTPUT=${CMAKE_CURRENT_BINARY_DIR}/resnet50_per_channel_projection_multicore_hal.mlir
    -DEXPECTED_X=2
    -DEXPECTED_Y=10
    -P ${_TT_IREE_E2E_SOURCE_DIR}/compiler/check_multicore_workgroup_count.cmake
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

#-------------------------------------------------------------------------------
# Runtime checks (mock or hardware)
#-------------------------------------------------------------------------------

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
