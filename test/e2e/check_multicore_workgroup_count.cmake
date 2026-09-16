# Copyright 2026 The tt-iree Authors
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

foreach(_REQUIRED_VAR IREE_COMPILE SOURCE OUTPUT)
  if(NOT DEFINED ${_REQUIRED_VAR})
    message(FATAL_ERROR "Missing required variable: ${_REQUIRED_VAR}")
  endif()
endforeach()

if(NOT DEFINED EXPECTED_X)
  set(EXPECTED_X 2)
endif()
if(NOT DEFINED EXPECTED_Y)
  set(EXPECTED_Y 2)
endif()

execute_process(
  COMMAND "${IREE_COMPILE}"
    --iree-hal-target-backends=tenstorrent
    --compile-to=hal
    --emit-mlir-bytecode=false
    "${SOURCE}"
    -o "${OUTPUT}"
  RESULT_VARIABLE _COMPILE_RESULT
  OUTPUT_VARIABLE _COMPILE_STDOUT
  ERROR_VARIABLE _COMPILE_STDERR
)
if(NOT _COMPILE_RESULT EQUAL 0)
  message(FATAL_ERROR
    "Tenstorrent HAL compilation failed:\n${_COMPILE_STDOUT}${_COMPILE_STDERR}")
endif()

file(READ "${OUTPUT}" _HAL_IR)
string(FIND "${_HAL_IR}"
  "workgroups([%c${EXPECTED_X}, %c${EXPECTED_Y}, %c1])" _WORKGROUP_POS)
if(_WORKGROUP_POS EQUAL -1)
  message(FATAL_ERROR
    "Expected ${SOURCE} to dispatch with workgroups([${EXPECTED_X}, ${EXPECTED_Y}, 1])")
endif()
