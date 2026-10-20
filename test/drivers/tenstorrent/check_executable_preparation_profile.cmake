# Copyright 2026 The tt-iree Authors
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

file(REMOVE_RECURSE "${OUTPUT_DIR}")
execute_process(
  COMMAND "${CMAKE_COMMAND}" -E env "TT_IREE_PROFILE_OUTPUT=${OUTPUT_DIR}"
    "${TEST_EXECUTABLE}" --gtest_filter=ExecutableCacheTest.ReusesIdenticalPreparedExecutable
  RESULT_VARIABLE result OUTPUT_VARIABLE output ERROR_VARIABLE error)
if(NOT result EQUAL 0)
  message(FATAL_ERROR "Preparation profiling test failed: ${output}\n${error}")
endif()
file(READ "${OUTPUT_DIR}/run_results.json" summary)
foreach(pair "dispatch_count;0" "executable_prepare_count;3" "executable_cache_hit_count;1")
  list(GET pair 0 key)
  list(GET pair 1 expected)
  string(JSON actual GET "${summary}" "${key}")
  if(NOT actual EQUAL expected)
    message(FATAL_ERROR "Unexpected ${key}: ${actual}, expected ${expected}")
  endif()
endforeach()
file(READ "${OUTPUT_DIR}/host_setup_results.csv" setup)
if(NOT setup MATCHES "EXECUTABLE_PREPARE,[0-9]+,[0-9]+,[0-9]+,3,0,")
  message(FATAL_ERROR "Missing invalid-argument preparation event: ${setup}")
endif()
if(NOT setup MATCHES "DEVICE_CREATE," OR NOT setup MATCHES "DEVICE_DESTROY,")
  message(FATAL_ERROR "Missing device lifecycle events: ${setup}")
endif()
