# Copyright 2025 The tt-iree Authors
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

if(_TT_IREE_PYTHON3_EXECUTABLE)
  add_test(NAME tt_e2e_device_native_bf16_batch_tools
    COMMAND "${_TT_IREE_PYTHON3_EXECUTABLE}"
      "${CMAKE_CURRENT_SOURCE_DIR}/device_native_bf16_batch_tools_test.py")
  set_tests_properties(tt_e2e_device_native_bf16_batch_tools PROPERTIES
    LABELS "tt-iree;e2e;unit;correctness;bf16;batch")
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
endif()
