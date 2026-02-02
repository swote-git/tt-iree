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
#define IREE_HAL_TT_MAX_DESCRIPTOR_SETS 4
#define IREE_HAL_TT_MAX_BINDINGS_PER_SET 8

// --- Descriptor Set Snapshot ---
typedef struct iree_hal_tt_descriptor_set_t {
  iree_hal_buffer_t* bindings[IREE_HAL_TT_MAX_BINDINGS_PER_SET];
  iree_device_size_t offsets[IREE_HAL_TT_MAX_BINDINGS_PER_SET];
} iree_hal_tt_descriptor_set_t;

// --- Dispatch Command ---
typedef struct iree_hal_tt_dispatch_command_t {
  iree_hal_executable_t* executable;
  int32_t entry_point;
  uint32_t workgroup_x;
  uint32_t workgroup_y;
  uint32_t workgroup_z;
  // Snapshot of bindings needed for execution
  iree_hal_tt_descriptor_set_t descriptor_sets[IREE_HAL_TT_MAX_DESCRIPTOR_SETS];
} iree_hal_tt_dispatch_command_t;

// --- Generic Command Wrapper ---
typedef struct iree_hal_tt_command_t {
  iree_hal_tt_command_type_t type;
  union {
    iree_hal_tt_dispatch_command_t dispatch;
  };
} iree_hal_tt_command_t;

// --- Accessor ---
iree_hal_tt_command_t* iree_hal_tt_command_buffer_get_commands(
    iree_hal_command_buffer_t* command_buffer, iree_host_size_t* out_count);

#ifdef __cplusplus
}
#endif

#endif  // IREE_HAL_DRIVERS_TENSTORRENT_TT_COMMAND_BUFFER_H_
