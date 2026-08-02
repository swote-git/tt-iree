// Copyright 2025 The tt-iree Authors
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// Utility to build TTEX FlatBuffer blobs without a compiler.
// Primary use case: unit tests that need to exercise the runtime's
// TTEX parsing/verification path before the compiler backend exists.

#ifndef IREE_SCHEMA_TT_EXECUTABLE_BUILDER_UTIL_H_
#define IREE_SCHEMA_TT_EXECUTABLE_BUILDER_UTIL_H_

#include <stdint.h>

#include "iree/base/api.h"

#ifdef __cplusplus
extern "C" {
#endif

// TTEX wire-format versions. Version 0 represents legacy/unversioned blobs;
// new blobs must explicitly encode the current version.
#define TT_IREE_TTEX_VERSION_LEGACY 0u
#define TT_IREE_TTEX_VERSION_1 1u
#define TT_IREE_TTEX_CURRENT_VERSION TT_IREE_TTEX_VERSION_1

// BuiltinProgram enum values shared by the compiler, runtime, and tests.
// Must stay in sync with BuiltinProgram in tt_executable_def.fbs.
typedef enum tt_iree_ttex_builtin_program_e {
  TT_IREE_TTEX_BUILTIN_PROGRAM_CUSTOM_SFPI_ADD = 0,
  TT_IREE_TTEX_BUILTIN_PROGRAM_TTNN_ELTWISE_ADD = 1,
  TT_IREE_TTEX_BUILTIN_PROGRAM_BF16_MATMUL_32X32X32 = 2,
  TT_IREE_TTEX_BUILTIN_PROGRAM_BF16_MATMUL_TILED = 3,
} tt_iree_ttex_builtin_program_e;

// Describes a single entry point to be serialized into a TTEX blob.
// Field semantics match EntryPointDef in tt_executable_def.fbs.
typedef struct tt_iree_ttex_entry_point_desc_t {
  const char* name;  // null-terminated, may be NULL (defaults to "")
  uint16_t constant_count;
  uint16_t binding_count;
  uint64_t flags;
  uint32_t workgroup_size[3];  // [x, y, z]
  uint32_t builtin_program;    // BuiltinProgram enum value
  uint32_t builtin_m_tiles;    // static matmul M tile count, 0 if dynamic/unused
  uint32_t builtin_n_tiles;    // static matmul N tile count, 0 if dynamic/unused
  uint32_t builtin_k_tiles;    // static matmul K tile count, 0 if dynamic/unused
} tt_iree_ttex_entry_point_desc_t;

// Builds a complete TTEX FlatBuffer from the given entry point descriptors.
//
// The resulting bytes are allocated via |allocator| and returned in |out_data|.
// Caller must free with iree_allocator_free(allocator, out_data->data).
//
// The blob will:
//   - Have file_identifier "TTEX"
//   - Pass flatcc verify_as_root()
//   - Contain |entry_point_count| EntryPointDef entries
iree_status_t tt_iree_build_ttex_executable_def(
    iree_allocator_t allocator,
    iree_host_size_t entry_point_count,
    const tt_iree_ttex_entry_point_desc_t* entry_points,
    iree_byte_span_t* out_data);

// Builds a TTEX blob with an explicit version. Intended for compatibility
// tests; production serializers should use the current-version wrapper above.
iree_status_t tt_iree_build_ttex_executable_def_with_version(
    iree_allocator_t allocator,
    uint32_t version,
    iree_host_size_t entry_point_count,
    const tt_iree_ttex_entry_point_desc_t* entry_points,
    iree_byte_span_t* out_data);

#ifdef __cplusplus
}
#endif

#endif  // IREE_SCHEMA_TT_EXECUTABLE_BUILDER_UTIL_H_
