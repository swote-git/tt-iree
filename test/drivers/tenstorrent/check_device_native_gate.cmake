# Copyright 2026 The tt-iree Authors
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

if(NOT DEFINED TEST_EXECUTABLE)
  message(FATAL_ERROR "TEST_EXECUTABLE is required")
endif()

execute_process(
  COMMAND "${CMAKE_COMMAND}" -E env
    TT_IREE_REQUIRE_DEVICE_NATIVE=1
    "${TEST_EXECUTABLE}"
    --gtest_filter=DispatchTest.ExecutesF32MaxPoolAndPreservesBorder
  RESULT_VARIABLE result
  OUTPUT_VARIABLE stdout
  ERROR_VARIABLE stderr
)

set(output "${stdout}${stderr}")
if(result EQUAL 0)
  message(FATAL_ERROR
    "device-native gate accepted a host-only dispatch:\n${output}")
endif()

string(FIND "${output}" "requires device-native execution" marker)
if(marker EQUAL -1)
  message(FATAL_ERROR
    "device-native gate failed without the expected diagnostic:\n${output}")
endif()
