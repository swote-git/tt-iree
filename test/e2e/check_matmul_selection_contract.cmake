if(NOT DEFINED IREE_COMPILE OR NOT DEFINED SOURCE OR NOT DEFINED OUTPUT OR
   NOT DEFINED EXPECTED_PROGRAM OR NOT DEFINED EXPECTED_SHAPE_MODE)
  message(FATAL_ERROR
    "IREE_COMPILE, SOURCE, OUTPUT, EXPECTED_PROGRAM, and EXPECTED_SHAPE_MODE are required")
endif()

execute_process(
  COMMAND "${IREE_COMPILE}"
    --iree-hal-target-backends=tenstorrent
    --compile-to=executable-sources
    "${SOURCE}"
    -o "${OUTPUT}"
  RESULT_VARIABLE compile_result
  ERROR_VARIABLE compile_error
)
if(NOT compile_result EQUAL 0)
  message(FATAL_ERROR "matmul selection compile failed: ${compile_error}")
endif()

file(READ "${OUTPUT}" ir)
if(NOT ir MATCHES
   "linalg.matmul \\{[^\n]*iree.tenstorrent.builtin_program = ${EXPECTED_PROGRAM} : i32")
  message(FATAL_ERROR
    "dispatch matmul is missing builtin program ${EXPECTED_PROGRAM}")
endif()
if(NOT ir MATCHES
   "linalg.matmul \\{[^\n]*iree.tenstorrent.phase = \"selection-v1\"")
  message(FATAL_ERROR "dispatch matmul is missing selection-v1 phase")
endif()
if(NOT ir MATCHES
   "linalg.matmul \\{[^\n]*iree.tenstorrent.shape_mode = \"${EXPECTED_SHAPE_MODE}\"")
  message(FATAL_ERROR
    "dispatch matmul is missing shape mode ${EXPECTED_SHAPE_MODE}")
endif()

# Removing the operation-level handoff must fail instead of falling back to a
# second shape matcher in the serializer.
string(REGEX REPLACE
  "linalg\\.matmul \\{iree\\.tenstorrent[^}]*\\} ins"
  "linalg.matmul ins" missing_selection_ir "${ir}")
if(missing_selection_ir STREQUAL ir)
  message(FATAL_ERROR "failed to remove matmul selection for negative test")
endif()
set(missing_selection_output "${OUTPUT}.missing_selection.mlir")
file(WRITE "${missing_selection_output}" "${missing_selection_ir}")
execute_process(
  COMMAND "${IREE_COMPILE}"
    --iree-hal-target-backends=tenstorrent
    --compile-from=executable-sources
    "${missing_selection_output}"
    -o "${OUTPUT}.unexpected.vmfb"
  RESULT_VARIABLE missing_selection_result
  ERROR_VARIABLE missing_selection_error
)
if(missing_selection_result EQUAL 0)
  message(FATAL_ERROR
    "serializer accepted a BF16 matmul without the selection contract")
endif()
if(NOT missing_selection_error MATCHES
   "missing a valid Tenstorrent selection-v1 annotation")
  message(FATAL_ERROR
    "missing selection failed for an unexpected reason: ${missing_selection_error}")
endif()
