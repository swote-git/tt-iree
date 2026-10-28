# Copyright 2025 The tt-iree Authors
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

if(NOT TARGET iree-compile)
  message(STATUS "tt-iree: Skipping compiler E2E tests (iree-compile unavailable)")
  return()
endif()

find_program(_TT_IREE_PYTHON3_EXECUTABLE NAMES python3)
if(_TT_IREE_PYTHON3_EXECUTABLE)
  add_test(NAME tt_e2e_device_native_bf16_batch_tools
    COMMAND "${_TT_IREE_PYTHON3_EXECUTABLE}"
      "${CMAKE_CURRENT_SOURCE_DIR}/device_native_bf16_batch_tools_test.py")
  set_tests_properties(tt_e2e_device_native_bf16_batch_tools PROPERTIES
    LABELS "tt-iree;e2e;unit;correctness;bf16;batch")
  add_test(NAME tt_e2e_resnet50_bf16_batch_tools
    COMMAND "${_TT_IREE_PYTHON3_EXECUTABLE}"
      "${CMAKE_CURRENT_SOURCE_DIR}/resnet50_bf16_batch_tools_test.py")
  set_tests_properties(tt_e2e_resnet50_bf16_batch_tools PROPERTIES
    LABELS "tt-iree;e2e;unit;bf16;batch")
endif()
if(_TT_IREE_PYTHON3_EXECUTABLE AND TT_IREE_ENABLE_TTNN AND
   NOT TT_IREE_ENABLE_MOCK AND TARGET iree-run-module)
  add_test(NAME tt_e2e_device_native_bf16_batch
    COMMAND "${_TT_IREE_PYTHON3_EXECUTABLE}"
      "${CMAKE_CURRENT_SOURCE_DIR}/device_native_bf16_batch_test.py"
      --iree-compile $<TARGET_FILE:iree-compile>
      --iree-run-module $<TARGET_FILE:iree-run-module>
      --output-dir "${CMAKE_CURRENT_BINARY_DIR}/device_native_bf16_batch")
  set_tests_properties(tt_e2e_device_native_bf16_batch PROPERTIES
    LABELS "tt-iree;e2e;hardware;device-native;correctness;bf16;batch"
    RUN_SERIAL TRUE TIMEOUT 1200)
  add_test(NAME tt_e2e_device_native_bf16_batch_reuse
    COMMAND "${_TT_IREE_PYTHON3_EXECUTABLE}"
      "${CMAKE_CURRENT_SOURCE_DIR}/device_native_bf16_batch_test.py"
      --iree-compile $<TARGET_FILE:iree-compile>
      --iree-run-module $<TARGET_FILE:iree-run-module>
      --reuse-output-tiles
      --output-dir "${CMAKE_CURRENT_BINARY_DIR}/device_native_bf16_batch_reuse")
  set_tests_properties(tt_e2e_device_native_bf16_batch_reuse PROPERTIES
    LABELS "tt-iree;e2e;hardware;device-native;correctness;bf16;batch;reuse"
    RUN_SERIAL TRUE TIMEOUT 1200)
  add_test(NAME tt_e2e_device_native_bf16_spatial
    COMMAND "${_TT_IREE_PYTHON3_EXECUTABLE}"
      "${CMAKE_CURRENT_SOURCE_DIR}/device_native_bf16_spatial_test.py"
      --iree-compile $<TARGET_FILE:iree-compile>
      --iree-run-module $<TARGET_FILE:iree-run-module>
      --output-dir "${CMAKE_CURRENT_BINARY_DIR}/device_native_bf16_spatial")
  set_tests_properties(tt_e2e_device_native_bf16_spatial PROPERTIES
    LABELS "tt-iree;e2e;hardware;device-native;correctness;bf16;spatial"
    RUN_SERIAL TRUE TIMEOUT 600)
  add_test(NAME tt_e2e_mnist_mlp_fixture
    COMMAND "${_TT_IREE_PYTHON3_EXECUTABLE}"
      "${CMAKE_CURRENT_SOURCE_DIR}/mnist_mlp_fixture_test.py"
      --iree-compile $<TARGET_FILE:iree-compile>
      --iree-run-module $<TARGET_FILE:iree-run-module>
      --output-dir "${CMAKE_CURRENT_BINARY_DIR}/mnist_mlp_fixture")
  set_tests_properties(tt_e2e_mnist_mlp_fixture PROPERTIES
    LABELS "tt-iree;e2e;hardware;device-native;correctness;bf16;tensix;mnist"
    RUN_SERIAL TRUE TIMEOUT 600)
  add_test(NAME tt_e2e_device_native_bf16_tensix
    COMMAND "${_TT_IREE_PYTHON3_EXECUTABLE}"
      "${CMAKE_CURRENT_SOURCE_DIR}/device_native_bf16_tensix_test.py"
      --iree-compile $<TARGET_FILE:iree-compile>
      --iree-run-module $<TARGET_FILE:iree-run-module>
      --output-dir "${CMAKE_CURRENT_BINARY_DIR}/device_native_bf16_tensix")
  set_tests_properties(tt_e2e_device_native_bf16_tensix PROPERTIES
    LABELS "tt-iree;e2e;hardware;device-native;correctness;bf16;tensix"
    RUN_SERIAL TRUE TIMEOUT 600)
  add_test(
    NAME tt_e2e_device_native_bf16
    COMMAND "${_TT_IREE_PYTHON3_EXECUTABLE}"
      "${CMAKE_CURRENT_SOURCE_DIR}/device_native_bf16_test.py"
      --iree-compile $<TARGET_FILE:iree-compile>
      --iree-run-module $<TARGET_FILE:iree-run-module>
      --output-dir "${CMAKE_CURRENT_BINARY_DIR}/device_native_bf16"
  )
  set_tests_properties(tt_e2e_device_native_bf16 PROPERTIES
    LABELS "tt-iree;e2e;hardware;device-native;correctness;bf16"
    RUN_SERIAL TRUE TIMEOUT 600
  )
  add_test(
    NAME tt_e2e_device_native_resnet_operations
    COMMAND "${_TT_IREE_PYTHON3_EXECUTABLE}"
      "${CMAKE_CURRENT_SOURCE_DIR}/device_native_resnet_test.py"
      --iree-compile $<TARGET_FILE:iree-compile>
      --iree-run-module $<TARGET_FILE:iree-run-module>
      --output-dir "${CMAKE_CURRENT_BINARY_DIR}/device_native_resnet"
  )
  set_tests_properties(tt_e2e_device_native_resnet_operations PROPERTIES
    LABELS "tt-iree;e2e;hardware;device-native;correctness;aot"
    RUN_SERIAL TRUE TIMEOUT 1800
  )
  add_test(
    NAME tt_e2e_device_native_tensix_operations
    COMMAND "${_TT_IREE_PYTHON3_EXECUTABLE}"
      "${CMAKE_CURRENT_SOURCE_DIR}/device_native_resnet_test.py"
      --iree-compile $<TARGET_FILE:iree-compile>
      --iree-run-module $<TARGET_FILE:iree-run-module>
      --use-tensix
      --output-dir "${CMAKE_CURRENT_BINARY_DIR}/device_native_tensix"
  )
  set_tests_properties(tt_e2e_device_native_tensix_operations PROPERTIES
    LABELS "tt-iree;e2e;hardware;device-native;correctness;aot;tensix"
    RUN_SERIAL TRUE TIMEOUT 1800
  )
endif()

if(_TT_IREE_PYTHON3_EXECUTABLE)
  add_test(NAME tt_e2e_mnist_mlp_contract
    COMMAND "${_TT_IREE_PYTHON3_EXECUTABLE}"
      "${CMAKE_CURRENT_SOURCE_DIR}/mnist_mlp_test.py")
  set_tests_properties(tt_e2e_mnist_mlp_contract PROPERTIES
    LABELS "tt-iree;e2e;correctness;bf16;mnist" TIMEOUT 30)
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
  if(TARGET tt-iree-mnist-eval AND IREE_TARGET_BACKEND_LLVM_CPU AND IREE_HAL_DRIVER_LOCAL_SYNC)
    add_test(NAME tt_e2e_mnist_session
      COMMAND "${_TT_IREE_PYTHON3_EXECUTABLE}"
        "${CMAKE_CURRENT_SOURCE_DIR}/resnet50_session_test.py"
        --iree-compile $<TARGET_FILE:iree-compile>
        --session-runner $<TARGET_FILE:tt-iree-mnist-eval> --mnist)
    set_tests_properties(tt_e2e_mnist_session PROPERTIES
      LABELS "tt-iree;e2e;correctness;bf16;session;mnist" TIMEOUT 120)
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
  NAME tt_e2e_compile_device_native_f32_matmul
  SRC device_native_f32_matmul.mlir
)

tt_iree_add_compile_smoke_test(
  NAME tt_e2e_compile_device_native_f32_matmul_tail
  SRC device_native_f32_matmul_tail.mlir
)

tt_iree_add_compile_smoke_test(
  NAME tt_e2e_compile_device_native_f32_matmul_page_tail
  SRC device_native_f32_matmul_page_tail.mlir
)

tt_iree_add_compile_smoke_test(
  NAME tt_e2e_compile_device_native_f32_runtime_affine_matmul
  SRC device_native_f32_runtime_affine_matmul.mlir
)

tt_iree_add_compile_smoke_test(
  NAME tt_e2e_compile_device_native_f32_runtime_affine_tail
  SRC device_native_f32_runtime_affine_tail.mlir
)

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

  add_test(
    NAME tt_e2e_device_native_f32_runtime_affine_matmul
    COMMAND ${CMAKE_COMMAND}
      -DIREE_COMPILE=$<TARGET_FILE:iree-compile>
      -DIREE_RUN_MODULE=$<TARGET_FILE:iree-run-module>
      -DSOURCE=${CMAKE_CURRENT_SOURCE_DIR}/device_native_f32_runtime_affine_matmul.mlir
      -DOUTPUT=${CMAKE_CURRENT_BINARY_DIR}/device_native_f32_runtime_affine_matmul.vmfb
      -P ${CMAKE_CURRENT_SOURCE_DIR}/check_device_native_f32_runtime_affine_matmul.cmake
  )
  if(TT_IREE_ENABLE_MOCK)
    set_tests_properties(
      tt_e2e_device_native_f32_runtime_affine_matmul PROPERTIES
      LABELS "tt-iree;e2e;compile;mock;model"
    )
  else()
    set_tests_properties(
      tt_e2e_device_native_f32_runtime_affine_matmul PROPERTIES
      ENVIRONMENT
        "TT_METAL_RUNTIME_ROOT=${CMAKE_SOURCE_DIR}/third_party/tt-metal;TT_IREE_REQUIRE_DEVICE_NATIVE=1"
      LABELS "tt-iree;e2e;compile;hardware;model;aot;device-native"
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
