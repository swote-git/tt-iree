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

#ifdef __cplusplus
namespace tt {
namespace tt_metal {
class Program;
}
}  // namespace tt
typedef tt::tt_metal::Program* tt_metal_program_ptr_t;
typedef uint32_t tt_metal_kernel_id_t;
#else
typedef void* tt_metal_program_ptr_t;
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

  // Kernel handles (set during program creation)
  tt_metal_kernel_id_t reader_kernel_id;
  tt_metal_kernel_id_t writer_kernel_id;
  tt_metal_kernel_id_t compute_kernel_id;

  // Grid size / core range
  uint32_t core_range_x;
  uint32_t core_range_y;

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
  uint32_t builtin_program;  // BuiltinProgram enum value

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
//   - anything else / NULL data: legacy path (hardcoded single entry point)
iree_status_t iree_hal_tt_executable_create(
    iree_hal_device_t* device,
    const iree_hal_executable_params_t* executable_params,
    iree_allocator_t host_allocator,
    iree_hal_executable_t** out_executable);

// Looks up kernel params for a given entry point ordinal.
// Used by queue_execute to get the TT-Metal Program for dispatch.
// This is our internal fast-path; the vtable export_info path is
// also available for IREE-standard queries.
iree_status_t iree_hal_tt_executable_lookup_kernel_params(
    iree_hal_executable_t* executable,
    int32_t entry_point,
    const iree_hal_tt_kernel_params_t** out_params);

// Mutable variant — used by dispatch to restore moved programs after execute.
iree_status_t iree_hal_tt_executable_lookup_kernel_params_mutable(
    iree_hal_executable_t* executable,
    int32_t entry_point,
    iree_hal_tt_kernel_params_t** out_params);

#ifdef __cplusplus
}
#endif

#endif  // IREE_HAL_DRIVERS_TENSTORRENT_TT_EXECUTABLE_H_
