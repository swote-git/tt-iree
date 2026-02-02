// runtime/src/iree/hal/drivers/tenstorrent/tt_command_buffer.cc

#include "iree/hal/drivers/tenstorrent/tt_command_buffer.h"

#include <cstring>
#include <vector>

#include "iree/base/api.h"
#include "iree/hal/utils/resource_set.h"

//===----------------------------------------------------------------------===//
// Data Structures
//===----------------------------------------------------------------------===//

typedef struct iree_hal_tt_command_buffer_t {
  iree_hal_command_buffer_t base;
  iree_allocator_t host_allocator;
  
  // Tracks resources to prevent premature deallocation
  iree_hal_resource_set_t* resource_set;

  // Recorded commands
  std::vector<iree_hal_tt_command_t> commands;

  // Current binding state
  iree_hal_tt_descriptor_set_t current_descriptor_sets[IREE_HAL_TT_MAX_DESCRIPTOR_SETS];
} iree_hal_tt_command_buffer_t;

static const iree_hal_command_buffer_vtable_t iree_hal_tt_command_buffer_vtable;

static iree_hal_tt_command_buffer_t* iree_hal_tt_command_buffer_cast(
    iree_hal_command_buffer_t* base) {
  IREE_HAL_ASSERT_TYPE(base, &iree_hal_tt_command_buffer_vtable);
  return (iree_hal_tt_command_buffer_t*)base;
}

//===----------------------------------------------------------------------===//
// Creation / Destruction
//===----------------------------------------------------------------------===//

iree_status_t iree_hal_tt_device_create_command_buffer(
    iree_hal_device_t* device, iree_hal_command_buffer_mode_t mode,
    iree_hal_command_category_t command_categories,
    iree_hal_queue_affinity_t queue_affinity, iree_host_size_t binding_capacity,
    iree_hal_command_buffer_t** out_command_buffer) {
  
  iree_allocator_t host_allocator = iree_hal_device_host_allocator(device);
  iree_hal_tt_command_buffer_t* command_buffer = nullptr;
  
  IREE_RETURN_IF_ERROR(iree_allocator_malloc(
      host_allocator, sizeof(*command_buffer), (void**)&command_buffer));

  iree_hal_command_buffer_initialize(
      device, mode, command_categories, queue_affinity, binding_capacity,
      &iree_hal_tt_command_buffer_vtable, &command_buffer->base);
  
  command_buffer->host_allocator = host_allocator;
  
  new (&command_buffer->commands) std::vector<iree_hal_tt_command_t>();
  
  std::memset(command_buffer->current_descriptor_sets, 0, 
              sizeof(command_buffer->current_descriptor_sets));
  
  iree_status_t status = iree_hal_resource_set_create(host_allocator, &command_buffer->resource_set);

  if (iree_status_is_ok(status)) {
    *out_command_buffer = &command_buffer->base;
  } else {
    iree_hal_command_buffer_release(&command_buffer->base);
  }
  return status;
}

static void iree_hal_tt_command_buffer_destroy(iree_hal_command_buffer_t* base) {
  iree_hal_tt_command_buffer_t* command_buffer = iree_hal_tt_command_buffer_cast(base);
  
  iree_hal_resource_set_free(command_buffer->resource_set);
  
  command_buffer->commands.~vector();
  
  iree_allocator_free(command_buffer->host_allocator, command_buffer);
}

//===----------------------------------------------------------------------===//
// Recording Implementation
//===----------------------------------------------------------------------===//

static iree_status_t iree_hal_tt_command_buffer_begin(
    iree_hal_command_buffer_t* base) {
  iree_hal_tt_command_buffer_t* command_buffer = iree_hal_tt_command_buffer_cast(base);
  command_buffer->commands.clear();
  return iree_ok_status();
}

static iree_status_t iree_hal_tt_command_buffer_end(
    iree_hal_command_buffer_t* base) {
  return iree_ok_status();
}

// NOTE: Execution barriers are no-ops for this PoC but can be recorded if needed
static iree_status_t iree_hal_tt_command_buffer_execution_barrier(
    iree_hal_command_buffer_t* base, iree_hal_execution_stage_t source_stage_mask,
    iree_hal_execution_stage_t target_stage_mask,
    iree_hal_execution_barrier_flags_t flags,
    iree_host_size_t memory_barrier_count,
    const iree_hal_memory_barrier_t* memory_barriers,
    iree_host_size_t buffer_barrier_count,
    const iree_hal_buffer_barrier_t* buffer_barriers) {
  return iree_ok_status();
}

// Updates current binding state. Does NOT emit a command
static iree_status_t iree_hal_tt_command_buffer_push_descriptor_set(
    iree_hal_command_buffer_t* base, iree_hal_pipeline_layout_t* pipeline_layout,
    uint32_t set, iree_host_size_t binding_count,
    const iree_hal_descriptor_set_binding_t* bindings) {
  
  iree_hal_tt_command_buffer_t* command_buffer = iree_hal_tt_command_buffer_cast(base);
  
  if (set >= IREE_HAL_TT_MAX_DESCRIPTOR_SETS) return iree_ok_status();

  for (iree_host_size_t i = 0; i < binding_count; i++) {
    if (bindings[i].binding >= IREE_HAL_TT_MAX_BINDINGS_PER_SET) continue;
    
    // Update state
    command_buffer->current_descriptor_sets[set].bindings[bindings[i].binding] = bindings[i].buffer;
    command_buffer->current_descriptor_sets[set].offsets[bindings[i].binding] = bindings[i].offset;

    // check buffer
    if (bindings[i].buffer) {
      iree_hal_resource_set_insert(command_buffer->resource_set, 1, &bindings[i].buffer);
    }
  }
  return iree_ok_status();
}

// Records a dispatch command with the current binding state
static iree_status_t iree_hal_tt_command_buffer_dispatch(
    iree_hal_command_buffer_t* base, iree_hal_executable_t* executable,
    int32_t entry_point, uint32_t workgroup_x, uint32_t workgroup_y,
    uint32_t workgroup_z, iree_hal_dispatch_flags_t flags) {
  
  iree_hal_tt_command_buffer_t* command_buffer = iree_hal_tt_command_buffer_cast(base);
  
  iree_hal_tt_command_t cmd;
  cmd.type = IREE_HAL_TT_COMMAND_TYPE_DISPATCH;
  cmd.dispatch.executable = executable;
  cmd.dispatch.entry_point = entry_point;
  cmd.dispatch.workgroup_x = workgroup_x;
  cmd.dispatch.workgroup_y = workgroup_y;
  cmd.dispatch.workgroup_z = workgroup_z;
  
  // Snapshot
  std::memcpy(cmd.dispatch.descriptor_sets, command_buffer->current_descriptor_sets, 
              sizeof(command_buffer->current_descriptor_sets));
  
  command_buffer->commands.push_back(cmd);
  
  iree_hal_resource_set_insert(command_buffer->resource_set, 1, &executable);
  
  return iree_ok_status();
}

//===----------------------------------------------------------------------===//
// Public Accessor
//===----------------------------------------------------------------------===//

iree_hal_tt_command_t* iree_hal_tt_command_buffer_get_commands(
    iree_hal_command_buffer_t* base, iree_host_size_t* out_count) {
  iree_hal_tt_command_buffer_t* command_buffer = iree_hal_tt_command_buffer_cast(base);
  *out_count = command_buffer->commands.size();
  return command_buffer->commands.data();
}

//===----------------------------------------------------------------------===//
// VTable
//===----------------------------------------------------------------------===//

static const iree_hal_command_buffer_vtable_t iree_hal_tt_command_buffer_vtable = {
    .destroy = iree_hal_tt_command_buffer_destroy,
    .begin = iree_hal_tt_command_buffer_begin,
    .end = iree_hal_tt_command_buffer_end,
    .execution_barrier = iree_hal_tt_command_buffer_execution_barrier,
    .signal_event = NULL,
    .reset_event = NULL,
    .wait_events = NULL,
    .discard_buffer = NULL,
    .update_buffer = NULL,
    .copy_buffer = NULL,
    .collective = NULL,
    .push_constants = iree_hal_command_buffer_push_constants,
    .push_descriptor_set = iree_hal_tt_command_buffer_push_descriptor_set,
    .dispatch = iree_hal_tt_command_buffer_dispatch,
    .dispatch_indirect = NULL,
};
