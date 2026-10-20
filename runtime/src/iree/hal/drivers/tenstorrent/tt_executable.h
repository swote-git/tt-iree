// Copyright 2025 The tt-iree Authors
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// Tenstorrent HAL executable: holds compiled kernel programs and
// per-entry-point metadata for IREE's HAL dispatch protocol.

#ifndef IREE_HAL_DRIVERS_TENSTORRENT_TT_EXECUTABLE_H_
#define IREE_HAL_DRIVERS_TENSTORRENT_TT_EXECUTABLE_H_

#include <stdint.h>

#include "iree/base/api.h"
#include "iree/hal/api.h"
#include "iree/schema/tt_executable_builder_util.h"

#ifdef __cplusplus
namespace tt {
namespace tt_metal {
class Program;
}
} // namespace tt
typedef tt::tt_metal::Program *tt_metal_program_ptr_t;
typedef uint32_t tt_metal_kernel_id_t;
#else
typedef void *tt_metal_program_ptr_t;
typedef uint32_t tt_metal_kernel_id_t;
#endif

#ifdef __cplusplus
extern "C" {
#endif

// ---------------------------------------------------------------------------
// Kernel parameters (TT-Metal program + kernel IDs)
// ---------------------------------------------------------------------------
// Unchanged from existing code. One per entry point.

typedef struct iree_hal_tt_kernel_params_t {
  tt_metal_program_ptr_t program;
  uint32_t builtin_program;
  // Executable wire version selects operation-specific kernel argument ABIs.
  uint32_t ttex_version;

  // Kernel handles (set during program creation)
  tt_metal_kernel_id_t reader_kernel_id;
  tt_metal_kernel_id_t writer_kernel_id;
  tt_metal_kernel_id_t compute_kernel_id;

  // Grid size / core range
  uint32_t core_range_x;
  uint32_t core_range_y;

  // Static matmul shape metadata expressed in 32x32 tile counts.
  // These remain 0 when runtime dispatch constants provide the shape.
  uint32_t builtin_m_tiles;
  uint32_t builtin_n_tiles;
  uint32_t builtin_k_tiles;

  // Descriptor-driven dispatch metadata (TTEX v2).
  uint8_t dispatch_kind;
  tt_iree_ttex_matmul_like_desc_t matmul_like;
  tt_iree_ttex_conv2d_desc_t conv2d;
  tt_iree_ttex_padded_copy_desc_t padded_copy;
  tt_iree_ttex_max_pool_2d_desc_t max_pool;
  tt_iree_ttex_global_sum_desc_t global_sum;
  tt_iree_ttex_scalar_div_desc_t scalar_div;
  tt_iree_ttex_linear_desc_t linear;
  tt_iree_ttex_bf16_native_desc_t bf16_native;

  // Points into the executable-owned TTEX buffer when a v3 AOT bundle exists.
  // The concrete FlatBuffer table type remains private to the runtime.
  const void *aot_kernel_bundle;
  uint8_t aot_target_arch;

  // Executable-owned directory containing the materialized AOT artifacts.
  // Empty for metadata-only entries and valid until executable destruction.
  iree_string_view_t aot_artifact_root;

  // Debug info
  iree_string_view_t kernel_name;
} iree_hal_tt_kernel_params_t;

// ---------------------------------------------------------------------------
// Entry point metadata (HAL export_info + kernel_params)
// ---------------------------------------------------------------------------

typedef struct iree_hal_tt_entry_point_t {
  // HAL export_info fields
  iree_string_view_t name;
  uint16_t constant_count;
  uint16_t binding_count;
  uint64_t flags;
  uint32_t workgroup_size[3];

  // Tenstorrent dispatch routing
  uint32_t builtin_program; // BuiltinProgram enum value
  uint32_t builtin_m_tiles;
  uint32_t builtin_n_tiles;
  uint32_t builtin_k_tiles;
  uint8_t dispatch_kind;

  // TT-Metal program (created at executable_create time)
  iree_hal_tt_kernel_params_t kernel_params;
} iree_hal_tt_entry_point_t;

// ---------------------------------------------------------------------------
// Public API
// ---------------------------------------------------------------------------

// Creates an executable from the given params.
//
// Dispatch by executable_format:
//   - "tenstorrent-ttex-fb": parse TTEX FlatBuffer from executable_data
//   - "TT-METAL": recognized only to report that the legacy path was removed
// Unknown formats and malformed or unsupported TTEX blobs are rejected.
#define IREE_HAL_TT_LEGACY_EXECUTABLE_FORMAT "TT-METAL"

iree_status_t iree_hal_tt_executable_create(
    iree_hal_device_t *device,
    const iree_hal_executable_params_t *executable_params,
    iree_allocator_t host_allocator, iree_hal_executable_t **out_executable);

// Returns true when |executable_data| exactly matches the executable-owned
// TTEX payload. Used by the cache to resolve hash collisions without retaining
// a second copy of large VMFB payloads.
bool iree_hal_tt_executable_matches_data(
    iree_hal_executable_t *executable, iree_const_byte_span_t executable_data);

// Looks up kernel params for a given entry point ordinal.
// Used by queue_execute to get the TT-Metal Program for dispatch.
// This is our internal fast-path; the vtable export_info path is
// also available for IREE-standard queries.
iree_status_t iree_hal_tt_executable_lookup_kernel_params(
    iree_hal_executable_t *executable, int32_t entry_point,
    const iree_hal_tt_kernel_params_t **out_params);

// Mutable variant — used by dispatch to restore moved programs after execute.
iree_status_t iree_hal_tt_executable_lookup_kernel_params_mutable(
    iree_hal_executable_t *executable, int32_t entry_point,
    iree_hal_tt_kernel_params_t **out_params);

#ifdef __cplusplus
}

namespace iree::hal::tenstorrent {
struct RuntimePreparationRecord;
}

// Cache-owned timing avoids double-counting executable creation as a second
// preparation event. Direct descriptor-only C API calls are not profiled.
iree_status_t iree_hal_tt_executable_create_with_profile(
    iree_hal_device_t *device,
    const iree_hal_executable_params_t *executable_params,
    iree_allocator_t host_allocator, iree_hal_executable_t **out_executable,
    iree::hal::tenstorrent::RuntimePreparationRecord *profile);
#endif

#endif // IREE_HAL_DRIVERS_TENSTORRENT_TT_EXECUTABLE_H_
