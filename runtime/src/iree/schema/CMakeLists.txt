# Copyright 2025 The tt-iree Authors
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

message(STATUS "tt-iree: Configuring TTEX schema generation")

set(TT_IREE_TTEX_SCHEMA
  "${CMAKE_CURRENT_SOURCE_DIR}/tt_executable_def.fbs"
)

# flatcc generates these headers from the .fbs file:
set(TT_IREE_TTEX_GENERATED_HDRS
  "${CMAKE_CURRENT_BINARY_DIR}/tt_executable_def_reader.h"
  "${CMAKE_CURRENT_BINARY_DIR}/tt_executable_def_builder.h"
  "${CMAKE_CURRENT_BINARY_DIR}/tt_executable_def_verifier.h"
)

# IREE_ROOT_DIR: the root of the IREE source tree.
# IREE_SOURCE_DIR is set by IREE when it is the top-level project.
# Fall back to the submodule path for legacy tt-iree top-level builds.
if(NOT DEFINED IREE_ROOT_DIR)
  if(DEFINED IREE_SOURCE_DIR)
    set(IREE_ROOT_DIR "${IREE_SOURCE_DIR}")
  else()
    set(IREE_ROOT_DIR "${CMAKE_SOURCE_DIR}/third_party/iree")
  endif()
endif()

add_custom_command(
  OUTPUT ${TT_IREE_TTEX_GENERATED_HDRS}
  COMMAND $<TARGET_FILE:iree-flatcc-cli>
    -o "${CMAKE_CURRENT_BINARY_DIR}"
    --common --reader --builder --verifier
    "${TT_IREE_TTEX_SCHEMA}"
  DEPENDS
    iree-flatcc-cli
    "${TT_IREE_TTEX_SCHEMA}"
  COMMENT "Generating TTEX FlatBuffer headers"
  VERBATIM
)

add_custom_target(tt_executable_def_gen
  DEPENDS ${TT_IREE_TTEX_GENERATED_HDRS}
)

# Interface library: link this to get generated headers + flatcc runtime includes.
add_library(tt_executable_def_c_fbs INTERFACE)
add_dependencies(tt_executable_def_c_fbs tt_executable_def_gen)

target_include_directories(tt_executable_def_c_fbs INTERFACE
  # Generated headers directory
  "$<BUILD_INTERFACE:${CMAKE_CURRENT_BINARY_DIR}>"
  # flatcc runtime headers (flatcc/flatbuffers.h etc.)
  "$<BUILD_INTERFACE:${IREE_ROOT_DIR}/third_party/flatcc/include>"
)

# ---------------------------------------------------------------------------
# Builder utility library (for tests and tools)
# ---------------------------------------------------------------------------
# Provides tt_iree_build_ttex_executable_def() to create TTEX blobs
# programmatically, without requiring the compiler backend.

add_library(tt_executable_builder_util STATIC
  tt_executable_builder_util.cc
  tt_executable_builder_util.h
)

target_link_libraries(tt_executable_builder_util
  PUBLIC
    tt_executable_def_c_fbs
    iree_base_base
  PRIVATE
    flatcc::runtime
)

# Expose runtime/src so callers can use:
#   #include "iree/schema/tt_executable_builder_util.h"
target_include_directories(tt_executable_builder_util
  PUBLIC
    "${CMAKE_CURRENT_SOURCE_DIR}/../.."
)

message(STATUS "  TTEX schema: ${TT_IREE_TTEX_SCHEMA}")
message(STATUS "  Generated headers: ${CMAKE_CURRENT_BINARY_DIR}")
