# Copyright 2025 The tt-iree Authors
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#-------------------------------------------------------------------------------
# Tenstorrent HAL Driver Tests
#-------------------------------------------------------------------------------

# tile_layout_test: standalone tile conversion (no HAL/TT-Metal dependency)
add_executable(tile_layout_test
  tile_layout_test.cc
)
target_compile_features(tile_layout_test PRIVATE cxx_std_17)
target_link_libraries(tile_layout_test
  PRIVATE
    iree_testing_gtest
    iree_testing_gtest_main
)
add_test(NAME tt_tile_layout_test COMMAND tile_layout_test)
set_tests_properties(tt_tile_layout_test PROPERTIES LABELS "tt-iree;unit")

# Native source generation is descriptor-only; test it without Metal or hardware.
add_executable(native_kernel_test
  native_kernel_test.cc
  ../../../compiler/plugins/target/tenstorrent/TenstorrentNativeKernels.cpp
)
target_include_directories(native_kernel_test PRIVATE
  ../../../compiler/plugins/target/tenstorrent
  ../../../runtime/src
)
target_link_libraries(native_kernel_test PRIVATE
  tt_executable_builder_util
  iree_testing_gtest
  iree_testing_gtest_main
)
add_test(NAME tt_native_kernel_test COMMAND native_kernel_test)
set_tests_properties(tt_native_kernel_test PROPERTIES LABELS "tt-iree;unit;codegen")

#-------------------------------------------------------------------------------
# Helper: create a gtest-based HAL test
#-------------------------------------------------------------------------------
function(tt_iree_hal_test)
  cmake_parse_arguments(
    _RULE
    ""              # Options
    "NAME"          # One-value args
    "SRCS;DEPS"     # Multi-value args
    ${ARGN}
  )

  add_executable(${_RULE_NAME} ${_RULE_SRCS})
  target_compile_features(${_RULE_NAME} PRIVATE cxx_std_17)

  target_include_directories(${_RULE_NAME}
    PRIVATE
      ${CMAKE_SOURCE_DIR}/runtime/src
      ${CMAKE_CURRENT_SOURCE_DIR}
  )

  target_link_libraries(${_RULE_NAME}
    PRIVATE
      iree_hal_tenstorrent
      iree_base_base
      iree_hal_hal
      iree_testing_gtest
      iree_testing_gtest_main
      ${_RULE_DEPS}
  )

  if(TT_IREE_ENABLE_TTNN)
    target_link_libraries(${_RULE_NAME} PRIVATE "${TT_METAL_LIB_DIR}/libfmt.so")
    set_target_properties(${_RULE_NAME} PROPERTIES
      BUILD_RPATH "${TT_METAL_LIB_DIR}"
      INSTALL_RPATH "${TT_METAL_LIB_DIR}"
    )
  endif()

  # TT-Metal/OpenMPI may segfault during process exit cleanup (not in our
  # code). Wrap with shell to mask the signal; judge pass/fail by gtest output.
  add_test(NAME tt_${_RULE_NAME}
    COMMAND sh -c "$<TARGET_FILE:${_RULE_NAME}> 2>&1 || true")
  set_tests_properties(tt_${_RULE_NAME} PROPERTIES
    LABELS "tt-iree;hal"
    PASS_REGULAR_EXPRESSION "\\[  PASSED  \\]"
    FAIL_REGULAR_EXPRESSION "\\[  FAILED  \\]"
  )
  if(NOT TT_IREE_ENABLE_MOCK)
    set_tests_properties(tt_${_RULE_NAME} PROPERTIES
      ENVIRONMENT "TT_METAL_RUNTIME_ROOT=${CMAKE_SOURCE_DIR}/third_party/tt-metal"
    )
  endif()
endfunction()

#-------------------------------------------------------------------------------
# HAL Tests
#-------------------------------------------------------------------------------

tt_iree_hal_test(
  NAME driver_test
  SRCS driver_test.cc
)

tt_iree_hal_test(
  NAME device_test
  SRCS device_test.cc
)

tt_iree_hal_test(
  NAME runtime_profiler_test
  SRCS runtime_profiler_test.cc
)

tt_iree_hal_test(NAME buffer_transfer_test SRCS buffer_transfer_test.cc)
set_tests_properties(tt_buffer_transfer_test PROPERTIES LABELS "tt-iree;unit;transfer")

tt_iree_hal_test(
  NAME runtime_checkpoint_test
  SRCS runtime_checkpoint_test.cc
)

tt_iree_hal_test(
  NAME executable_cache_test
  SRCS executable_cache_test.cc
  DEPS tt_executable_builder_util
)

tt_iree_hal_test(
  NAME spatial_plan_test
  SRCS spatial_plan_test.cc
  DEPS tt_executable_builder_util
)
set_tests_properties(tt_spatial_plan_test PROPERTIES LABELS "tt-iree;unit;abi;spatial")

tt_iree_hal_test(
  NAME buffer_test
  SRCS buffer_test.cc
)

tt_iree_hal_test(
  NAME copy_test
  SRCS copy_test.cc
)

find_package(Python3 COMPONENTS Interpreter QUIET)
if(Python3_Interpreter_FOUND)
  set(_TT_TRANSFER_ANALYSIS_ARGS)
  if(TT_IREE_ENABLE_MOCK)
    list(APPEND _TT_TRANSFER_ANALYSIS_ARGS --copy-test $<TARGET_FILE:copy_test>)
  endif()
  add_test(NAME tt_runtime_transfer_analysis_test
    COMMAND "${Python3_EXECUTABLE}"
      "${CMAKE_CURRENT_SOURCE_DIR}/runtime_transfer_analysis_test.py"
      ${_TT_TRANSFER_ANALYSIS_ARGS})
  set_tests_properties(tt_runtime_transfer_analysis_test PROPERTIES
    LABELS "tt-iree;unit;profiling")
  if(NOT TT_IREE_ENABLE_MOCK)
    add_test(NAME tt_device_fill_profile_test
      COMMAND "${Python3_EXECUTABLE}"
        "${CMAKE_CURRENT_SOURCE_DIR}/runtime_transfer_analysis_test.py"
        --copy-test $<TARGET_FILE:copy_test> --expect-device-fill)
    set_tests_properties(tt_device_fill_profile_test PROPERTIES
      LABELS "tt-iree;hal;device-native;profiling"
      RUN_SERIAL TRUE
      ENVIRONMENT "TT_METAL_RUNTIME_ROOT=${CMAKE_SOURCE_DIR}/third_party/tt-metal")
  endif()
endif()

tt_iree_hal_test(
  NAME file_test
  SRCS file_test.cc
)

tt_iree_hal_test(
  NAME dispatch_test
  SRCS dispatch_test.cc
  DEPS tt_executable_builder_util
)

# Physical dispatch tests exercise the same compiler-generated AOT bundles as
# production VMFBs instead of relying on tt-metal example/test kernel paths.
if(TARGET TenstorrentAOTCompiler)
  tt_iree_hal_test(NAME aot_artifact_test SRCS aot_artifact_test.cc
    DEPS TenstorrentAOTCompiler tt_executable_builder_util)
  target_compile_features(aot_artifact_test PRIVATE cxx_std_20)
  target_include_directories(aot_artifact_test PRIVATE
    ${CMAKE_SOURCE_DIR}/compiler/plugins/target/tenstorrent)
  tt_metal_target_link_libraries(aot_artifact_test)
  set_tests_properties(tt_aot_artifact_test PROPERTIES
    LABELS "tt-iree;unit;aot")
  target_link_libraries(dispatch_test PRIVATE TenstorrentAOTCompiler)
  target_include_directories(dispatch_test PRIVATE
    ${CMAKE_SOURCE_DIR}/compiler/plugins/target/tenstorrent
  )
  target_compile_definitions(dispatch_test PRIVATE
    TT_IREE_ENABLE_AOT_TESTS=1
  )
endif()

# dispatch_test needs direct TT-Metal access for bfloat16 data transfer.
if(NOT TT_IREE_ENABLE_MOCK)
  tt_metal_target_link_libraries(dispatch_test)
endif()

if(TT_IREE_ENABLE_MOCK)
  add_test(NAME tt_executable_preparation_profile_test
    COMMAND ${CMAKE_COMMAND}
      -DTEST_EXECUTABLE=$<TARGET_FILE:executable_cache_test>
      -DOUTPUT_DIR=${CMAKE_CURRENT_BINARY_DIR}/executable_preparation_profile
      -P ${CMAKE_CURRENT_SOURCE_DIR}/check_executable_preparation_profile.cmake)
  set_tests_properties(tt_executable_preparation_profile_test PROPERTIES
    LABELS "tt-iree;hal;mock;profiling")
  add_test(
    NAME tt_runtime_checkpoint_dispatch_test
    COMMAND ${CMAKE_COMMAND}
      -DTEST_EXECUTABLE=$<TARGET_FILE:dispatch_test>
      -DOUTPUT_DIR=${CMAKE_CURRENT_BINARY_DIR}/runtime_checkpoint_dispatch
      -P ${CMAKE_CURRENT_SOURCE_DIR}/check_runtime_checkpoint.cmake
  )
  set_tests_properties(tt_runtime_checkpoint_dispatch_test PROPERTIES
    LABELS "tt-iree;hal;mock;correctness"
  )

  add_test(
    NAME tt_device_native_gate_test
    COMMAND ${CMAKE_COMMAND}
      -DTEST_EXECUTABLE=$<TARGET_FILE:dispatch_test>
      -P ${CMAKE_CURRENT_SOURCE_DIR}/check_device_native_gate.cmake
  )
  set_tests_properties(tt_device_native_gate_test PROPERTIES
    LABELS "tt-iree;hal;mock;device-native"
  )
endif()

# Descriptor-only executable creation does not require a physical device.
add_test(
  NAME tt_dispatch_aot_contract_test
  COMMAND dispatch_test
    --gtest_filter=DispatchContractTest.*
)
set_tests_properties(tt_dispatch_aot_contract_test PROPERTIES
  LABELS "tt-iree;unit;aot"
)

#-------------------------------------------------------------------------------
# Schema tests (no hardware, no MPI — run directly)
#-------------------------------------------------------------------------------

add_executable(ttex_test ttex_test.cc)
target_compile_features(ttex_test PRIVATE cxx_std_17)
target_link_libraries(ttex_test
  PRIVATE
    tt_executable_builder_util
    iree_base_base
    iree_testing_gtest
    iree_testing_gtest_main
)
add_test(NAME tt_ttex_test COMMAND ttex_test)
set_tests_properties(tt_ttex_test PROPERTIES LABELS "tt-iree;unit")

# Manual precision diagnostic: deliberately independent of IREE and TTEX.
# Numerical discrepancies are recorded, not silently accepted as CI passes.
if(TT_IREE_ENABLE_TTNN AND NOT TT_IREE_ENABLE_MOCK)
  add_executable(tensix_precision_probe tensix_precision_probe.cc)
  target_compile_features(tensix_precision_probe PRIVATE cxx_std_20)
  target_compile_options(tensix_precision_probe PRIVATE -ffp-contract=off)
  tt_metal_target_link_libraries(tensix_precision_probe)
endif()
