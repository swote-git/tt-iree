// runtime/src/iree/hal/drivers/tenstorrent/tt_command_buffer.h

#ifndef IREE_HAL_DRIVERS_TENSTORRENT_TT_COMMAND_BUFFER_H_
#define IREE_HAL_DRIVERS_TENSTORRENT_TT_COMMAND_BUFFER_H_

#include "iree/base/api.h"
#include "iree/hal/api.h"

#ifdef __cplusplus
extern "C" {
#endif

// --- Command Types ---
typedef enum iree_hal_tt_command_type_t {
  IREE_HAL_TT_COMMAND_TYPE_DISPATCH,
  IREE_HAL_TT_COMMAND_TYPE_BARRIER, // Placeholder for barriers
} iree_hal_tt_command_type_t;

// --- Constants ---
#define IREE_HAL_TT_MAX_BINDINGS 16
#define IREE_HAL_TT_MAX_CONSTANTS 256

// --- Dispatch Command (v3.9.0 API) ---
typedef struct iree_hal_tt_dispatch_command_t {
  iree_hal_executable_t* executable;
  iree_hal_executable_export_ordinal_t export_ordinal;  // Changed from entry_point
  iree_hal_dispatch_config_t config;  // Contains workgroup counts
  // Snapshot of bindings and constants
  iree_hal_buffer_ref_t bindings[IREE_HAL_TT_MAX_BINDINGS];
  iree_host_size_t binding_count;
  uint8_t constants[IREE_HAL_TT_MAX_CONSTANTS];
  iree_host_size_t constants_length;
  iree_hal_dispatch_flags_t flags;
} iree_hal_tt_dispatch_command_t;

// --- Generic Command Wrapper ---
typedef struct iree_hal_tt_command_t {
  iree_hal_tt_command_type_t type;
  union {
    iree_hal_tt_dispatch_command_t dispatch;
  };
} iree_hal_tt_command_t;

// --- Device Command Buffer Creation ---
iree_status_t iree_hal_tt_device_create_command_buffer(
    iree_hal_device_t* device, iree_hal_command_buffer_mode_t mode,
    iree_hal_command_category_t command_categories,
    iree_hal_queue_affinity_t queue_affinity, iree_host_size_t binding_capacity,
    iree_hal_command_buffer_t** out_command_buffer);

// --- Accessor ---
iree_hal_tt_command_t* iree_hal_tt_command_buffer_get_commands(
    iree_hal_command_buffer_t* command_buffer, iree_host_size_t* out_count);

#ifdef __cplusplus
}
#endif

#endif  // IREE_HAL_DRIVERS_TENSTORRENT_TT_COMMAND_BUFFER_H_
