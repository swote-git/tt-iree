# Copyright 2026 The tt-iree Authors
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

foreach(_REQUIRED_VAR IREE_COMPILE IREE_RUN_MODULE SOURCE OUTPUT)
  if(NOT DEFINED ${_REQUIRED_VAR})
    message(FATAL_ERROR "${_REQUIRED_VAR} is required")
  endif()
endforeach()

execute_process(
  COMMAND "${IREE_COMPILE}" "${SOURCE}"
    --iree-hal-target-backends=tenstorrent
    --iree-tenstorrent-device-id=p100a
    -o "${OUTPUT}"
  RESULT_VARIABLE _COMPILE_RESULT
  ERROR_VARIABLE _COMPILE_ERROR
)
if(NOT _COMPILE_RESULT EQUAL 0)
  message(FATAL_ERROR
    "runtime-affine matmul compilation failed (${_COMPILE_RESULT}):\n"
    "${_COMPILE_ERROR}")
endif()

execute_process(
  COMMAND "${IREE_RUN_MODULE}"
    --device=tenstorrent
    "--module=${OUTPUT}"
    --function=device_native_f32_runtime_affine_matmul
    --input=64x49xf32=1.0
    --input=64x64xf32=0.015625
    --input=64xf32=1.0
    --input=64xf32=0.5
    --input=64xf32=0.25
    --expected_output=64x49xf32=0.25
    --output=
  RESULT_VARIABLE _RUN_RESULT
  OUTPUT_VARIABLE _RUN_OUTPUT
  ERROR_VARIABLE _RUN_ERROR
)
if(NOT _RUN_RESULT EQUAL 0)
  message(FATAL_ERROR
    "runtime-affine matmul execution failed (${_RUN_RESULT}):\n${_RUN_ERROR}")
endif()

set(_RUN_LOG "${_RUN_OUTPUT}\n${_RUN_ERROR}")
foreach(_EXPECTED_TEXT
    "EXEC @device_native_f32_runtime_affine_matmul"
    "[SUCCESS] all function outputs matched their expected values.")
  string(FIND "${_RUN_LOG}" "${_EXPECTED_TEXT}" _MATCH_POSITION)
  if(_MATCH_POSITION EQUAL -1)
    message(FATAL_ERROR
      "missing expected runtime diagnostic '${_EXPECTED_TEXT}':\n${_RUN_LOG}")
  endif()
endforeach()

message(STATUS "Device-native runtime-affine F32 matmul smoke: PASS")
