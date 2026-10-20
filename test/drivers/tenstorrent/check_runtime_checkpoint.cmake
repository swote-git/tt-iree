# Copyright 2026 The tt-iree Authors
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

foreach(_REQUIRED_VAR TEST_EXECUTABLE OUTPUT_DIR)
  if(NOT DEFINED ${_REQUIRED_VAR})
    message(FATAL_ERROR "${_REQUIRED_VAR} is required")
  endif()
endforeach()

file(REMOVE_RECURSE "${OUTPUT_DIR}")
execute_process(
  COMMAND "${CMAKE_COMMAND}" -E env
    "TT_IREE_CHECKPOINT_OUTPUT=${OUTPUT_DIR}"
    "${TEST_EXECUTABLE}"
    "--gtest_filter=DispatchTest.ExecutesF32PaddedCopyAndPreservesBorder"
  RESULT_VARIABLE _RESULT
  OUTPUT_VARIABLE _OUTPUT
  ERROR_VARIABLE _ERROR
)
if(NOT _RESULT EQUAL 0)
  message(FATAL_ERROR
    "checkpoint dispatch failed (${_RESULT}):\n${_OUTPUT}\n${_ERROR}")
endif()

set(_MANIFEST "${OUTPUT_DIR}/manifest.json")
if(NOT EXISTS "${_MANIFEST}")
  message(FATAL_ERROR "checkpoint manifest was not created")
endif()
file(READ "${_MANIFEST}" _MANIFEST_TEXT)
foreach(_EXPECTED_TEXT
    "\"checkpoint_count\": 1"
    "\"dispatch_index\": 1"
    "\"op_code\": \"resnet50_input_pad\""
    "\"shape\": [1, 2, 2]"
    "\"file\": \"dispatch_0001_resnet50_input_pad.npy\"")
  string(FIND "${_MANIFEST_TEXT}" "${_EXPECTED_TEXT}" _POSITION)
  if(_POSITION EQUAL -1)
    message(FATAL_ERROR
      "checkpoint manifest is missing '${_EXPECTED_TEXT}':\n${_MANIFEST_TEXT}")
  endif()
endforeach()

if(NOT EXISTS "${OUTPUT_DIR}/dispatch_0001_resnet50_input_pad.npy")
  message(FATAL_ERROR "checkpoint NPY was not created")
endif()

message(STATUS "Runtime checkpoint dispatch integration: PASS")
