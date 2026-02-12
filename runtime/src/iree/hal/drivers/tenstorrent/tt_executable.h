// runtime/src/iree/hal/drivers/tenstorrent/tt_executable.h

#ifndef IREE_HAL_DRIVERS_TENSTORRENT_TT_EXECUTABLE_H_
#define IREE_HAL_DRIVERS_TENSTORRENT_TT_EXECUTABLE_H_

#include <stdint.h>
#include "iree/base/api.h"
#include "iree/hal/api.h"

#ifdef __cplusplus
namespace tt { namespace tt_metal { class Program; } }
typedef tt::tt_metal::Program* tt_metal_program_ptr_t;
typedef uint32_t tt_metal_kernel_id_t;
#else
typedef void* tt_metal_program_ptr_t;
typedef uint32_t tt_metal_kernel_id_t;
#endif

#ifdef __cplusplus
extern "C" {
#endif

// Kernel parameters for tenstorrent
typedef struct iree_hal_tt_kernel_params_t {
  tt_metal_program_ptr_t program;
  
  // Kernel Handles
  tt_metal_kernel_id_t reader_kernel_id;
  tt_metal_kernel_id_t writer_kernel_id;
  tt_metal_kernel_id_t compute_kernel_id;

  // Grid size / Core range info
  uint32_t core_range_x;
  uint32_t core_range_y;

  // Debug info
  iree_string_view_t kernel_name;
} iree_hal_tt_kernel_params_t;

iree_status_t iree_hal_tt_executable_create(
    iree_hal_device_t* device,
    const iree_hal_executable_params_t* executable_params,
    iree_allocator_t host_allocator,
    iree_hal_executable_t** out_executable);

iree_status_t iree_hal_tt_executable_lookup_kernel_params(
    iree_hal_executable_t* executable,
    int32_t entry_point,
    const iree_hal_tt_kernel_params_t** out_params);

// Mutable variant: needed by queue_execute to restore program after dispatch.
iree_status_t iree_hal_tt_executable_lookup_kernel_params_mutable(
    iree_hal_executable_t* executable,
    int32_t entry_point,
    iree_hal_tt_kernel_params_t** out_params);

#ifdef __cplusplus
}
#endif

#endif // IREE_HAL_DRIVERS_TENSTORRENT_TT_EXECUTABLE_H_
