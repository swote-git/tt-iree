# Copyright 2026 The tt-iree Authors
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

foreach(_REQUIRED_VAR IREE_COMPILE IREE_RUN_MODULE SOURCE OUTPUT_DIR)
  if(NOT DEFINED ${_REQUIRED_VAR})
    message(FATAL_ERROR "${_REQUIRED_VAR} is required")
  endif()
endforeach()

file(MAKE_DIRECTORY "${OUTPUT_DIR}")
set(_DISPATCH_IR "${OUTPUT_DIR}/resnet50_bottleneck_dispatch.mlir")
set(_EXECUTABLE_IR "${OUTPUT_DIR}/resnet50_bottleneck_executable.mlir")
set(_CPU_VMFB "${OUTPUT_DIR}/resnet50_bottleneck_cpu.vmfb")
set(_TT_VMFB "${OUTPUT_DIR}/resnet50_bottleneck_tenstorrent.vmfb")

function(_compile_tenstorrent_phase PHASE OUTPUT)
  execute_process(
    COMMAND "${IREE_COMPILE}" "${SOURCE}"
      "--compile-to=${PHASE}"
      --iree-hal-target-device=tenstorrent
      --emit-mlir-bytecode=false
      -o "${OUTPUT}"
    RESULT_VARIABLE _RESULT
    ERROR_VARIABLE _ERROR
  )
  if(NOT _RESULT EQUAL 0)
    message(FATAL_ERROR
      "Tenstorrent ${PHASE} compilation failed (${_RESULT}):\n${_ERROR}")
  endif()
endfunction()

_compile_tenstorrent_phase(dispatch-creation "${_DISPATCH_IR}")
_compile_tenstorrent_phase(executable-sources "${_EXECUTABLE_IR}")

file(READ "${_DISPATCH_IR}" _DISPATCH_TEXT)
string(REGEX MATCHALL "flow\\.dispatch\\.workgroups" _DISPATCH_MATCHES
  "${_DISPATCH_TEXT}")
list(LENGTH _DISPATCH_MATCHES _DISPATCH_COUNT)
if(NOT _DISPATCH_COUNT EQUAL 4)
  message(FATAL_ERROR
    "expected 4 ResNet50 bottleneck dispatches, got ${_DISPATCH_COUNT}")
endif()

file(READ "${_EXECUTABLE_IR}" _EXECUTABLE_TEXT)
string(REGEX MATCHALL "hal\\.executable\\.export public @" _EXPORT_MATCHES
  "${_EXECUTABLE_TEXT}")
list(LENGTH _EXPORT_MATCHES _EXPORT_COUNT)
if(NOT _EXPORT_COUNT EQUAL 4)
  message(FATAL_ERROR
    "expected 4 ResNet50 bottleneck exports, got ${_EXPORT_COUNT}")
endif()

set(_EXPECTED_EXPORTS
  "resnet50_bottleneck_dispatch_0_matmul_like_64x8x8x64_f32"
  "resnet50_bottleneck_dispatch_1_conv_64x8x8x64x3x3_f32"
  "resnet50_bottleneck_dispatch_2_matmul_like_256x64x64_f32"
  "resnet50_bottleneck_dispatch_3_matmul_like_256x64x64_f32"
)
foreach(_EXPECTED_EXPORT IN LISTS _EXPECTED_EXPORTS)
  string(FIND "${_EXECUTABLE_TEXT}" "${_EXPECTED_EXPORT}" _EXPORT_POSITION)
  if(_EXPORT_POSITION EQUAL -1)
    message(FATAL_ERROR "missing expected export: ${_EXPECTED_EXPORT}")
  endif()
endforeach()

# All four exports, including the flattened residual projection, must serialize
# into one complete Tenstorrent VMFB.
execute_process(
  COMMAND "${IREE_COMPILE}" "${SOURCE}"
    --iree-hal-target-backends=tenstorrent
    -o "${_TT_VMFB}"
  RESULT_VARIABLE _TT_COMPILE_RESULT
  ERROR_VARIABLE _TT_COMPILE_ERROR
)
if(NOT _TT_COMPILE_RESULT EQUAL 0)
  message(FATAL_ERROR
    "Tenstorrent VMFB compilation failed (${_TT_COMPILE_RESULT}):\n"
    "${_TT_COMPILE_ERROR}")
endif()

# Conv1, conv2, and the residual output each retain a fused ReLU.
string(REGEX MATCHALL "arith\\.select" _SELECT_MATCHES "${_EXECUTABLE_TEXT}")
list(LENGTH _SELECT_MATCHES _SELECT_COUNT)
if(NOT _SELECT_COUNT EQUAL 3)
  message(FATAL_ERROR
    "expected 3 fused ReLU selects, got ${_SELECT_COUNT}")
endif()

# Execute the same fixture on CPU so the standalone block remains a complete,
# runnable reference while Tenstorrent descriptor support is brought up.
execute_process(
  COMMAND "${IREE_COMPILE}" "${SOURCE}"
    --iree-hal-target-backends=llvm-cpu
    --iree-llvmcpu-target-cpu=host
    -o "${_CPU_VMFB}"
  RESULT_VARIABLE _CPU_COMPILE_RESULT
  ERROR_VARIABLE _CPU_COMPILE_ERROR
)
if(NOT _CPU_COMPILE_RESULT EQUAL 0)
  message(FATAL_ERROR
    "CPU reference compilation failed (${_CPU_COMPILE_RESULT}):\n"
    "${_CPU_COMPILE_ERROR}")
endif()

execute_process(
  COMMAND "${IREE_RUN_MODULE}"
    --device=local-task
    "--module=${_CPU_VMFB}"
    --function=resnet50_bottleneck
    --input=64x8x8xf32=1.0
    --input=64x64xf32=0.015625
    --input=64x64x3x3xf32=0.0
    --input=256x64xf32=0.015625
    --input=256x64xf32=0.015625
    --expected_output=256x8x8xf32=0.96875
  RESULT_VARIABLE _CPU_RUN_RESULT
  ERROR_VARIABLE _CPU_RUN_ERROR
)
if(NOT _CPU_RUN_RESULT EQUAL 0)
  message(FATAL_ERROR
    "CPU reference execution failed (${_CPU_RUN_RESULT}):\n${_CPU_RUN_ERROR}")
endif()

message(STATUS
  "ResNet50 bottleneck: 4-dispatch Tenstorrent VMFB and CPU reference PASS")
