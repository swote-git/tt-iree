# Copyright 2025 The tt-iree Authors
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

# device_native_resnet_test.py also runs the resnet50/ operator fixtures.
if(_TT_IREE_PYTHON3_EXECUTABLE AND TT_IREE_ENABLE_TTNN AND
   NOT TT_IREE_ENABLE_MOCK AND TARGET iree-run-module)
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

if(TT_IREE_ENABLE_MOCK OR TT_IREE_ENABLE_TTNN)
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
