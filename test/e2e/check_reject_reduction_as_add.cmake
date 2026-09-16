# Copyright 2026 The tt-iree Authors
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

if(NOT DEFINED IREE_COMPILE OR NOT DEFINED SOURCE_FILE)
  message(FATAL_ERROR "IREE_COMPILE and SOURCE_FILE are required")
endif()

execute_process(
  COMMAND "${IREE_COMPILE}" "${SOURCE_FILE}"
    --iree-hal-target-device=tenstorrent
    -o "${CMAKE_CURRENT_BINARY_DIR}/reduction_not_add.vmfb"
  RESULT_VARIABLE compile_result
  OUTPUT_VARIABLE compile_stdout
  ERROR_VARIABLE compile_stderr
)

if(compile_result EQUAL 0)
  message(FATAL_ERROR "reduction was incorrectly accepted as elementwise add")
endif()
if(NOT compile_stderr MATCHES "unsupported Tenstorrent dispatch")
  message(FATAL_ERROR
    "reduction failed for an unexpected reason:\n${compile_stdout}\n${compile_stderr}")
endif()
