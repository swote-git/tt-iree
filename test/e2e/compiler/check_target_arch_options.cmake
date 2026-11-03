# Copyright 2026 The tt-iree Authors
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

if(NOT DEFINED IREE_COMPILE OR NOT DEFINED SOURCE OR NOT DEFINED OUTPUT_DIR)
  message(FATAL_ERROR "IREE_COMPILE, SOURCE, and OUTPUT_DIR are required")
endif()

file(MAKE_DIRECTORY "${OUTPUT_DIR}")

function(check_target_arch device_id expected_arch)
  set(output "${OUTPUT_DIR}/${device_id}_executable_sources.mlir")
  execute_process(
    COMMAND "${IREE_COMPILE}"
      --iree-hal-target-backends=tenstorrent
      "--iree-tenstorrent-device-id=${device_id}"
      --compile-to=executable-sources
      "${SOURCE}"
      -o "${output}"
    RESULT_VARIABLE compile_result
    OUTPUT_VARIABLE compile_stdout
    ERROR_VARIABLE compile_stderr
  )
  if(NOT compile_result EQUAL 0)
    message(FATAL_ERROR
      "Tenstorrent ${device_id} target compilation failed:\n"
      "${compile_stdout}\n${compile_stderr}")
  endif()

  file(READ "${output}" executable_sources)
  string(FIND "${executable_sources}"
    "device_id = \"${device_id}\"" device_id_position)
  string(FIND "${executable_sources}"
    "target_arch = \"${expected_arch}\"" target_arch_position)
  if(device_id_position EQUAL -1 OR target_arch_position EQUAL -1)
    message(FATAL_ERROR
      "Tenstorrent ${device_id} did not select target_arch=${expected_arch}")
  endif()
endfunction()

check_target_arch(p100a blackhole)
check_target_arch(wormhole wormhole_b0)

execute_process(
  COMMAND "${IREE_COMPILE}"
    --iree-hal-target-backends=tenstorrent
    --iree-tenstorrent-device-id=invalid-device
    "${SOURCE}"
    -o "${OUTPUT_DIR}/invalid.vmfb"
  RESULT_VARIABLE invalid_result
  OUTPUT_VARIABLE invalid_stdout
  ERROR_VARIABLE invalid_stderr
)
if(invalid_result EQUAL 0)
  message(FATAL_ERROR "Invalid Tenstorrent target device was accepted")
endif()
if(NOT invalid_stderr MATCHES "unsupported Tenstorrent target architecture")
  message(FATAL_ERROR
    "Invalid target failed without the expected diagnostic:\n"
    "${invalid_stdout}\n${invalid_stderr}")
endif()

message(STATUS "Tenstorrent target architecture options verified")
