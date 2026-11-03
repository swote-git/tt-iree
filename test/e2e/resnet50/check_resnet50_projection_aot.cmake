# Copyright 2026 The tt-iree Authors
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

if(NOT DEFINED IREE_COMPILE OR NOT DEFINED SOURCE OR NOT DEFINED OUTPUT)
  message(FATAL_ERROR "IREE_COMPILE, SOURCE, and OUTPUT are required")
endif()
if(NOT DEFINED DEVICE_ID)
  set(DEVICE_ID p100a)
endif()

execute_process(
  COMMAND "${IREE_COMPILE}"
    --iree-hal-target-backends=tenstorrent
    "--iree-tenstorrent-device-id=${DEVICE_ID}"
    "${SOURCE}"
    -o "${OUTPUT}"
  RESULT_VARIABLE compile_result
  OUTPUT_VARIABLE compile_stdout
  ERROR_VARIABLE compile_stderr
)
if(NOT compile_result EQUAL 0)
  message(FATAL_ERROR
    "Tenstorrent AOT projection compilation failed:\n"
    "${compile_stdout}\n${compile_stderr}")
endif()

file(SIZE "${OUTPUT}" vmfb_size)
if(vmfb_size LESS 1048576)
  message(FATAL_ERROR
    "AOT VMFB is unexpectedly small (${vmfb_size} bytes); ELF artifacts may be missing")
endif()
if(vmfb_size GREATER 52428800)
  message(FATAL_ERROR
    "AOT VMFB is unexpectedly large (${vmfb_size} bytes); target filtering may be broken")
endif()

file(STRINGS "${OUTPUT}" elf_paths
  LIMIT_COUNT 1
  REGEX "Kernel_Source_Code/.+[.]elf"
)
if(NOT elf_paths)
  message(FATAL_ERROR "AOT VMFB contains no embedded TT-Metal ELF path")
endif()

file(STRINGS "${OUTPUT}" generated_source
  LIMIT_COUNT 1
  REGEX "Device RISC-V tiled FP32"
)
if(NOT generated_source)
  message(FATAL_ERROR "AOT VMFB contains no compiler-generated compute source")
endif()

file(STRINGS "${OUTPUT}" legacy_kernel_path
  LIMIT_COUNT 1
  REGEX "tests/tt_metal/tt_metal/test_kernels"
)
if(legacy_kernel_path)
  message(FATAL_ERROR
    "AOT VMFB unexpectedly depends on a tt-metal test kernel source path")
endif()

message(STATUS
  "Tenstorrent ${DEVICE_ID} AOT projection contract verified (${vmfb_size} byte VMFB)")
