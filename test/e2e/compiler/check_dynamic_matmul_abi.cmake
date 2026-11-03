if(NOT DEFINED IREE_COMPILE OR NOT DEFINED SOURCE OR NOT DEFINED OUTPUT)
  message(FATAL_ERROR "IREE_COMPILE, SOURCE, and OUTPUT are required")
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
  message(FATAL_ERROR "dynamic matmul ABI compile failed: ${compile_error}")
endif()

file(READ "${OUTPUT}" ir)
if(NOT ir MATCHES "layout\\(<constants = 8,")
  message(FATAL_ERROR "dynamic matmul export does not declare 8 constants")
endif()
foreach(ordinal RANGE 0 7)
  if(NOT ir MATCHES "hal.interface.constant.load[^\n]*ordinal\\(${ordinal}\\)")
    message(FATAL_ERROR
      "dynamic matmul is missing interface constant ordinal ${ordinal}")
  endif()
endforeach()
if(ir MATCHES "hal.interface.constant.load[^\n]*ordinal\\(8\\)")
  message(FATAL_ERROR "dynamic matmul unexpectedly uses constant ordinal 8")
endif()
