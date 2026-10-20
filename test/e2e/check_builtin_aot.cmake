# Copyright 2026 The tt-iree Authors
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

if(NOT DEFINED IREE_COMPILE OR NOT DEFINED SOURCE_DIR OR
   NOT DEFINED OUTPUT_DIR)
  message(FATAL_ERROR "IREE_COMPILE, SOURCE_DIR, and OUTPUT_DIR are required")
endif()

file(MAKE_DIRECTORY "${OUTPUT_DIR}")

function(compile_and_check name source source_marker)
  set(output "${OUTPUT_DIR}/${name}.vmfb")
  execute_process(
    COMMAND "${IREE_COMPILE}"
      --iree-hal-target-backends=tenstorrent
      --iree-tenstorrent-device-id=p100a
      "${SOURCE_DIR}/${source}"
      -o "${output}"
    RESULT_VARIABLE compile_result
    OUTPUT_VARIABLE compile_stdout
    ERROR_VARIABLE compile_stderr
  )
  if(NOT compile_result EQUAL 0)
    message(FATAL_ERROR
      "Tenstorrent ${name} AOT compilation failed:\n"
      "${compile_stdout}\n${compile_stderr}")
  endif()

  file(STRINGS "${output}" elf_path
    LIMIT_COUNT 1
    REGEX "Kernel_Source_Code/.+[.]elf"
  )
  if(NOT elf_path)
    message(FATAL_ERROR "${name} contains no embedded TT-Metal ELF")
  endif()

  file(STRINGS "${output}" generated_source
    LIMIT_COUNT 1
    REGEX "${source_marker}"
  )
  if(NOT generated_source)
    message(FATAL_ERROR "${name} contains no compiler-generated kernel source")
  endif()

  file(STRINGS "${output}" legacy_kernel_path
    LIMIT_COUNT 1
    REGEX "custom_sfpi_add/kernels|tests/tt_metal/tt_metal/test_kernels"
  )
  if(legacy_kernel_path)
    message(FATAL_ERROR "${name} contains a legacy tt-metal kernel path")
  endif()

  set("${name}_ELF_PATH" "${elf_path}" PARENT_SCOPE)
endfunction()

compile_and_check(simple_add simple_add.mlir "binary_op_init_common")
compile_and_check(simple_matmul simple_matmul.mlir
                  "out_subblock_num_tiles = get_arg_val")
compile_and_check(simple_matmul_64 simple_matmul_64.mlir
                  "out_subblock_num_tiles = get_arg_val")
compile_and_check(dynamic_matmul dynamic_matmul.mlir
                  "out_subblock_num_tiles = get_arg_val")
compile_and_check(resnet50_input_pad resnet50_input_pad.mlir
                  "write_scratch_offset = 4096")
compile_and_check(device_native_f32_matmul device_native_f32_matmul.mlir
                  "native-v15 descriptor-specialized")
compile_and_check(device_native_f32_matmul_tail
                  device_native_f32_matmul_tail.mlir
                  "native-v15 descriptor-specialized")
compile_and_check(device_native_f32_runtime_affine_matmul
                  device_native_f32_runtime_affine_matmul.mlir
                  "native-v15 descriptor-specialized")

if(NOT "${simple_matmul_ELF_PATH}" STREQUAL
       "${simple_matmul_64_ELF_PATH}" OR
   NOT "${simple_matmul_ELF_PATH}" STREQUAL
       "${dynamic_matmul_ELF_PATH}")
  message(FATAL_ERROR
    "Static, multicore, and dynamic matmul did not reuse the same AOT artifact")
endif()

# v15 specializes a single FP32 data-movement kernel for each descriptor.
# Its ELF identity intentionally differs across shapes and epilogues.

message(STATUS
  "Tenstorrent add, matmul, native F32 matmul/affine, and padded-copy AOT verified")
