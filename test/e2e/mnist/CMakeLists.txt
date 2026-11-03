# Copyright 2025 The tt-iree Authors
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

if(_TT_IREE_PYTHON3_EXECUTABLE AND TT_IREE_ENABLE_TTNN AND
   NOT TT_IREE_ENABLE_MOCK AND TARGET iree-run-module)
  add_test(NAME tt_e2e_mnist_mlp_fixture
    COMMAND "${_TT_IREE_PYTHON3_EXECUTABLE}"
      "${CMAKE_CURRENT_SOURCE_DIR}/mnist_mlp_fixture_test.py"
      --iree-compile $<TARGET_FILE:iree-compile>
      --iree-run-module $<TARGET_FILE:iree-run-module>
      --output-dir "${CMAKE_CURRENT_BINARY_DIR}/mnist_mlp_fixture")
  set_tests_properties(tt_e2e_mnist_mlp_fixture PROPERTIES
    LABELS "tt-iree;e2e;hardware;device-native;correctness;bf16;tensix;mnist"
    RUN_SERIAL TRUE TIMEOUT 600)
endif()

if(_TT_IREE_PYTHON3_EXECUTABLE)
  add_test(NAME tt_e2e_mnist_mlp_contract
    COMMAND "${_TT_IREE_PYTHON3_EXECUTABLE}"
      "${CMAKE_CURRENT_SOURCE_DIR}/mnist_mlp_test.py")
  set_tests_properties(tt_e2e_mnist_mlp_contract PROPERTIES
    LABELS "tt-iree;e2e;correctness;bf16;mnist" TIMEOUT 30)
  # The session transport gate is shared with resnet50/.
  if(TARGET tt-iree-mnist-eval AND IREE_TARGET_BACKEND_LLVM_CPU AND IREE_HAL_DRIVER_LOCAL_SYNC)
    add_test(NAME tt_e2e_mnist_session
      COMMAND "${_TT_IREE_PYTHON3_EXECUTABLE}"
        "${_TT_IREE_E2E_SOURCE_DIR}/resnet50/resnet50_session_test.py"
        --iree-compile $<TARGET_FILE:iree-compile>
        --session-runner $<TARGET_FILE:tt-iree-mnist-eval> --mnist)
    set_tests_properties(tt_e2e_mnist_session PROPERTIES
      LABELS "tt-iree;e2e;correctness;bf16;session;mnist" TIMEOUT 120)
  endif()
endif()
