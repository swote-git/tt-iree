// runtime/src/iree/hal/drivers/tenstorrent/tt_command_buffer.cc

#include "iree/hal/drivers/tenstorrent/tt_command_buffer.h"

#include <cinttypes>
#include <cstring>
#include <vector>

#include "iree/base/api.h"
#include "iree/hal/drivers/tenstorrent/tt_executable.h"
#include "iree/schema/tt_executable_builder_util.h"
#include "iree/schema/tt_native_dispatch.h"

//===----------------------------------------------------------------------===//
// Data Structures
//===----------------------------------------------------------------------===//

typedef struct iree_hal_tt_command_buffer_t {
  iree_hal_command_buffer_t base;
  iree_allocator_t host_allocator;

  // Recorded commands
  std::vector<iree_hal_tt_command_t> commands;

  // Host data copied immediately by update commands. Commands store offsets
  // instead of pointers so vector growth cannot invalidate recorded payloads.
  std::vector<uint8_t> update_data;

  // Resources to retain (simplified - just keep a list for PoC)
  std::vector<iree_hal_resource_t *> resources;
} iree_hal_tt_command_buffer_t;

// Forward declare vtable (defined at end of file)
extern const iree_hal_command_buffer_vtable_t iree_hal_tt_command_buffer_vtable;

static iree_hal_tt_command_buffer_t *
iree_hal_tt_command_buffer_cast(iree_hal_command_buffer_t *base) {
  IREE_HAL_ASSERT_TYPE(base, &iree_hal_tt_command_buffer_vtable);
  return (iree_hal_tt_command_buffer_t *)base;
}

//===----------------------------------------------------------------------===//
// Creation / Destruction
//===----------------------------------------------------------------------===//

iree_status_t iree_hal_tt_device_create_command_buffer(
    iree_hal_device_t *device, iree_hal_command_buffer_mode_t mode,
    iree_hal_command_category_t command_categories,
    iree_hal_queue_affinity_t queue_affinity, iree_host_size_t binding_capacity,
    iree_hal_command_buffer_t **out_command_buffer) {

  iree_allocator_t host_allocator = iree_hal_device_host_allocator(device);
  iree_hal_allocator_t *device_allocator = iree_hal_device_allocator(device);
  iree_hal_tt_command_buffer_t *command_buffer = nullptr;

  // Allocate space for command buffer + validation state
  iree_host_size_t total_size =
      sizeof(*command_buffer) +
      iree_hal_command_buffer_validation_state_size(mode, binding_capacity);

  IREE_RETURN_IF_ERROR(iree_allocator_malloc(host_allocator, total_size,
                                             (void **)&command_buffer));

  // Initialize base command buffer with validation_state pointing to memory
  // after struct
  iree_hal_command_buffer_initialize(
      device_allocator, mode, command_categories, queue_affinity,
      binding_capacity, (uint8_t *)command_buffer + sizeof(*command_buffer),
      &iree_hal_tt_command_buffer_vtable, &command_buffer->base);

  command_buffer->host_allocator = host_allocator;

  // Initialize C++ members
  new (&command_buffer->commands) std::vector<iree_hal_tt_command_t>();
  new (&command_buffer->update_data) std::vector<uint8_t>();
  new (&command_buffer->resources) std::vector<iree_hal_resource_t *>();

  *out_command_buffer = &command_buffer->base;
  return iree_ok_status();
}

static void
iree_hal_tt_command_buffer_destroy(iree_hal_command_buffer_t *base) {
  iree_hal_tt_command_buffer_t *command_buffer =
      iree_hal_tt_command_buffer_cast(base);

  // Release all retained resources
  for (auto *resource : command_buffer->resources) {
    iree_hal_resource_release(resource);
  }

  // Destroy C++ members
  command_buffer->commands.~vector();
  command_buffer->update_data.~vector();
  command_buffer->resources.~vector();

  iree_allocator_free(command_buffer->host_allocator, command_buffer);
}

//===----------------------------------------------------------------------===//
// Recording Implementation
//===----------------------------------------------------------------------===//

static iree_status_t
iree_hal_tt_command_buffer_begin(iree_hal_command_buffer_t *base) {
  iree_hal_tt_command_buffer_t *command_buffer =
      iree_hal_tt_command_buffer_cast(base);
  command_buffer->commands.clear();
  command_buffer->update_data.clear();

  // Release previous resources
  for (auto *resource : command_buffer->resources) {
    iree_hal_resource_release(resource);
  }
  command_buffer->resources.clear();

  return iree_ok_status();
}

static iree_status_t
iree_hal_tt_command_buffer_end(iree_hal_command_buffer_t *base) {
  return iree_ok_status();
}

static iree_status_t iree_hal_tt_command_buffer_begin_debug_group(
    iree_hal_command_buffer_t *base, iree_string_view_t label,
    iree_hal_label_color_t label_color,
    const iree_hal_label_location_t *location) {
  return iree_ok_status();
}

static iree_status_t
iree_hal_tt_command_buffer_end_debug_group(iree_hal_command_buffer_t *base) {
  return iree_ok_status();
}

static iree_status_t iree_hal_tt_command_buffer_execution_barrier(
    iree_hal_command_buffer_t *base,
    iree_hal_execution_stage_t source_stage_mask,
    iree_hal_execution_stage_t target_stage_mask,
    iree_hal_execution_barrier_flags_t flags,
    iree_host_size_t memory_barrier_count,
    const iree_hal_memory_barrier_t *memory_barriers,
    iree_host_size_t buffer_barrier_count,
    const iree_hal_buffer_barrier_t *buffer_barriers) {
  return iree_ok_status();
}

static iree_status_t iree_hal_tt_command_buffer_signal_event(
    iree_hal_command_buffer_t *base, iree_hal_event_t *event,
    iree_hal_execution_stage_t source_stage_mask) {
  return iree_make_status(IREE_STATUS_UNIMPLEMENTED,
                          "signal_event not implemented");
}

static iree_status_t iree_hal_tt_command_buffer_reset_event(
    iree_hal_command_buffer_t *base, iree_hal_event_t *event,
    iree_hal_execution_stage_t source_stage_mask) {
  return iree_make_status(IREE_STATUS_UNIMPLEMENTED,
                          "reset_event not implemented");
}

static iree_status_t iree_hal_tt_command_buffer_wait_events(
    iree_hal_command_buffer_t *base, iree_host_size_t event_count,
    const iree_hal_event_t **events,
    iree_hal_execution_stage_t source_stage_mask,
    iree_hal_execution_stage_t target_stage_mask,
    iree_host_size_t memory_barrier_count,
    const iree_hal_memory_barrier_t *memory_barriers,
    iree_host_size_t buffer_barrier_count,
    const iree_hal_buffer_barrier_t *buffer_barriers) {
  return iree_make_status(IREE_STATUS_UNIMPLEMENTED,
                          "wait_events not implemented");
}

static iree_status_t iree_hal_tt_command_buffer_advise_buffer(
    iree_hal_command_buffer_t *base, iree_hal_buffer_ref_t buffer_ref,
    iree_hal_memory_advise_flags_t flags, uint64_t arg0, uint64_t arg1) {
  return iree_ok_status();
}

static iree_status_t iree_hal_tt_command_buffer_fill_buffer(
    iree_hal_command_buffer_t *base, iree_hal_buffer_ref_t target_ref,
    const void *pattern, iree_host_size_t pattern_length,
    iree_hal_fill_flags_t flags) {
  iree_hal_tt_command_buffer_t *command_buffer =
      iree_hal_tt_command_buffer_cast(base);

  if (flags != IREE_HAL_FILL_FLAG_NONE) {
    return iree_make_status(
        IREE_STATUS_UNIMPLEMENTED,
        "Tenstorrent fill does not support flags 0x%" PRIx64, flags);
  }
  if (target_ref.reserved != 0) {
    return iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "fill buffer reference must have zero reserved bits");
  }
  if (pattern_length != 1 && pattern_length != 2 && pattern_length != 4) {
    return iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "fill pattern length must be 1, 2, or 4 bytes (got %" PRIhsz ")",
        pattern_length);
  }

  if (target_ref.buffer) {
    iree_device_size_t target_offset = 0;
    iree_device_size_t target_length = 0;
    IREE_RETURN_IF_ERROR(iree_hal_buffer_calculate_range(
                             /*base_offset=*/0,
                             iree_hal_buffer_byte_length(target_ref.buffer),
                             target_ref.offset, target_ref.length,
                             &target_offset, &target_length),
                         "fill target range");
  } else if (target_ref.buffer_slot >= command_buffer->base.binding_capacity) {
    return iree_make_status(
        IREE_STATUS_OUT_OF_RANGE,
        "fill target slot %u exceeds command buffer capacity %u",
        target_ref.buffer_slot, command_buffer->base.binding_capacity);
  }

  iree_hal_tt_command_t cmd = {};
  cmd.type = IREE_HAL_TT_COMMAND_TYPE_FILL;
  cmd.fill.target_ref = target_ref;
  std::memcpy(cmd.fill.pattern, pattern, pattern_length);
  cmd.fill.pattern_length = pattern_length;
  command_buffer->commands.push_back(cmd);

  if (target_ref.buffer) {
    iree_hal_buffer_retain(target_ref.buffer);
    command_buffer->resources.push_back(
        reinterpret_cast<iree_hal_resource_t *>(target_ref.buffer));
  }
  return iree_ok_status();
}

static iree_status_t iree_hal_tt_command_buffer_update_buffer(
    iree_hal_command_buffer_t *base, const void *source_buffer,
    iree_host_size_t source_offset, iree_hal_buffer_ref_t target_ref,
    iree_hal_update_flags_t flags) {
  iree_hal_tt_command_buffer_t *command_buffer =
      iree_hal_tt_command_buffer_cast(base);

  if (flags != IREE_HAL_UPDATE_FLAG_NONE) {
    return iree_make_status(
        IREE_STATUS_UNIMPLEMENTED,
        "Tenstorrent update does not support flags 0x%" PRIx64, flags);
  }
  if (target_ref.reserved != 0) {
    return iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "update buffer reference must have zero reserved bits");
  }
  if (target_ref.length > IREE_HAL_COMMAND_BUFFER_MAX_UPDATE_SIZE) {
    return iree_make_status(
        IREE_STATUS_RESOURCE_EXHAUSTED,
        "update exceeds the command buffer limit of %" PRIdsz
        " bytes (got %" PRIdsz ")",
        IREE_HAL_COMMAND_BUFFER_MAX_UPDATE_SIZE, target_ref.length);
  }

  if (target_ref.buffer) {
    iree_device_size_t target_offset = 0;
    iree_device_size_t target_length = 0;
    IREE_RETURN_IF_ERROR(iree_hal_buffer_calculate_range(
                             /*base_offset=*/0,
                             iree_hal_buffer_byte_length(target_ref.buffer),
                             target_ref.offset, target_ref.length,
                             &target_offset, &target_length),
                         "update target range");
  } else if (target_ref.buffer_slot >= command_buffer->base.binding_capacity) {
    return iree_make_status(
        IREE_STATUS_OUT_OF_RANGE,
        "update target slot %u exceeds command buffer capacity %u",
        target_ref.buffer_slot, command_buffer->base.binding_capacity);
  }

  iree_hal_tt_command_t cmd = {};
  cmd.type = IREE_HAL_TT_COMMAND_TYPE_UPDATE;
  cmd.update.target_ref = target_ref;
  cmd.update.source_data_offset = command_buffer->update_data.size();
  cmd.update.source_data_length =
      static_cast<iree_host_size_t>(target_ref.length);
  const uint8_t *source_data =
      static_cast<const uint8_t *>(source_buffer) + source_offset;
  command_buffer->update_data.insert(command_buffer->update_data.end(),
                                     source_data,
                                     source_data + target_ref.length);
  command_buffer->commands.push_back(cmd);

  if (target_ref.buffer) {
    iree_hal_buffer_retain(target_ref.buffer);
    command_buffer->resources.push_back(
        reinterpret_cast<iree_hal_resource_t *>(target_ref.buffer));
  }
  return iree_ok_status();
}

static iree_status_t iree_hal_tt_command_buffer_copy_buffer(
    iree_hal_command_buffer_t *base, iree_hal_buffer_ref_t source_ref,
    iree_hal_buffer_ref_t target_ref, iree_hal_copy_flags_t flags) {
  iree_hal_tt_command_buffer_t *command_buffer =
      iree_hal_tt_command_buffer_cast(base);

  if (flags != IREE_HAL_COPY_FLAG_NONE) {
    return iree_make_status(
        IREE_STATUS_UNIMPLEMENTED,
        "Tenstorrent copy does not support flags 0x%" PRIx64, flags);
  }
  if (source_ref.reserved != 0 || target_ref.reserved != 0) {
    return iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "copy buffer references must have zero reserved bits");
  }
  if (source_ref.length != target_ref.length) {
    return iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "copy source and target lengths must match (source=%" PRIdsz
        " target=%" PRIdsz ")",
        source_ref.length, target_ref.length);
  }

  iree_device_size_t source_offset = source_ref.offset;
  iree_device_size_t source_length = source_ref.length;
  if (source_ref.buffer) {
    IREE_RETURN_IF_ERROR(iree_hal_buffer_calculate_range(
                             /*base_offset=*/0,
                             iree_hal_buffer_byte_length(source_ref.buffer),
                             source_ref.offset, source_ref.length,
                             &source_offset, &source_length),
                         "copy source range");
  } else if (source_ref.buffer_slot >= command_buffer->base.binding_capacity) {
    return iree_make_status(
        IREE_STATUS_OUT_OF_RANGE,
        "copy source slot %u exceeds command buffer capacity %u",
        source_ref.buffer_slot, command_buffer->base.binding_capacity);
  }

  iree_device_size_t target_offset = target_ref.offset;
  iree_device_size_t target_length = target_ref.length;
  if (target_ref.buffer) {
    IREE_RETURN_IF_ERROR(iree_hal_buffer_calculate_range(
                             /*base_offset=*/0,
                             iree_hal_buffer_byte_length(target_ref.buffer),
                             target_ref.offset, target_ref.length,
                             &target_offset, &target_length),
                         "copy target range");
  } else if (target_ref.buffer_slot >= command_buffer->base.binding_capacity) {
    return iree_make_status(
        IREE_STATUS_OUT_OF_RANGE,
        "copy target slot %u exceeds command buffer capacity %u",
        target_ref.buffer_slot, command_buffer->base.binding_capacity);
  }

  if (source_ref.buffer && target_ref.buffer &&
      iree_hal_buffer_test_overlap(
          source_ref.buffer, source_offset, source_length, target_ref.buffer,
          target_offset, target_length) != IREE_HAL_BUFFER_OVERLAP_DISJOINT) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "copy source and target ranges overlap");
  }

  iree_hal_tt_command_t cmd = {};
  cmd.type = IREE_HAL_TT_COMMAND_TYPE_COPY;
  cmd.copy.source_ref = source_ref;
  cmd.copy.target_ref = target_ref;
  command_buffer->commands.push_back(cmd);

  // Retain buffers
  if (source_ref.buffer) {
    iree_hal_buffer_retain(source_ref.buffer);
    command_buffer->resources.push_back(
        (iree_hal_resource_t *)source_ref.buffer);
  }
  if (target_ref.buffer) {
    iree_hal_buffer_retain(target_ref.buffer);
    command_buffer->resources.push_back(
        (iree_hal_resource_t *)target_ref.buffer);
  }
  return iree_ok_status();
}

static iree_status_t iree_hal_tt_command_buffer_collective(
    iree_hal_command_buffer_t *base, iree_hal_channel_t *channel,
    iree_hal_collective_op_t op, uint32_t param, iree_hal_buffer_ref_t send_ref,
    iree_hal_buffer_ref_t recv_ref, iree_device_size_t element_count) {
  return iree_make_status(IREE_STATUS_UNIMPLEMENTED,
                          "collective not implemented");
}

static iree_status_t iree_hal_tt_validate_tensor_binding_range(
    const tt_iree_ttex_tensor_desc_t &tensor, const char *label,
    iree_hal_buffer_ref_list_t bindings) {
  if (tensor.binding >= bindings.count) {
    return iree_make_status(
        IREE_STATUS_OUT_OF_RANGE,
        "TTEX %s binding %u is outside dispatch binding count %zu", label,
        tensor.binding, bindings.count);
  }

  uint64_t required_length = tensor.element_type == TT_IREE_TTEX_TENSOR_ELEMENT_TYPE_BF16 ? 2 : sizeof(float);
  for (uint8_t i = 0; i < tensor.rank; ++i) {
    required_length *= tensor.storage_shape[i];
  }
  required_length += tensor.byte_offset;

  const iree_hal_buffer_ref_t &ref = bindings.values[tensor.binding];
  iree_device_size_t available_length = ref.length;
  if (ref.buffer) {
    iree_device_size_t resolved_offset = 0;
    IREE_RETURN_IF_ERROR(
        iree_hal_buffer_calculate_range(
            /*base_offset=*/0, iree_hal_buffer_byte_length(ref.buffer),
            ref.offset, ref.length, &resolved_offset, &available_length),
        "TTEX %s binding range", label);
  } else if (available_length == IREE_HAL_WHOLE_BUFFER) {
    // The final length is not known until the indirect binding table resolves.
    return iree_ok_status();
  }
  if (required_length > available_length) {
    return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                            "TTEX %s requires %" PRIu64
                            " binding bytes but only %" PRIu64 " are available",
                            label, required_length, available_length);
  }
  return iree_ok_status();
}

static iree_status_t iree_hal_tt_command_buffer_validate_dispatch(
    iree_hal_tt_command_buffer_t *command_buffer,
    iree_hal_executable_t *executable,
    iree_hal_executable_export_ordinal_t export_ordinal,
    const iree_hal_dispatch_config_t &config, iree_const_byte_span_t constants,
    iree_hal_buffer_ref_list_t bindings, iree_hal_dispatch_flags_t flags) {
  if (flags != IREE_HAL_DISPATCH_FLAG_NONE) {
    return iree_make_status(
        IREE_STATUS_UNIMPLEMENTED,
        "Tenstorrent M1 only supports direct HAL ABI dispatch arguments "
        "and static workgroup counts (flags=0x%" PRIx64 ")",
        flags);
  }
  if (config.dynamic_workgroup_local_memory != 0) {
    return iree_make_status(
        IREE_STATUS_UNIMPLEMENTED,
        "Tenstorrent M1 does not support dynamic workgroup local memory");
  }
  if (config.workgroup_count[0] == 0 || config.workgroup_count[1] == 0 ||
      config.workgroup_count[2] == 0) {
    return iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "dispatch workgroup_count must be non-zero, got %ux%ux%u",
        config.workgroup_count[0], config.workgroup_count[1],
        config.workgroup_count[2]);
  }

  const iree_hal_tt_kernel_params_t *kernel_params = nullptr;
  IREE_RETURN_IF_ERROR(iree_hal_tt_executable_lookup_kernel_params(
      executable, export_ordinal, &kernel_params));
  const bool native_matrix_grid =
      kernel_params && kernel_params->ttex_version >= TT_IREE_TTEX_VERSION_13 &&
      kernel_params->aot_kernel_bundle &&
      tt_iree_ttex_is_native_matrix_kind(kernel_params->dispatch_kind);
  bool is_multicore_compute =
      kernel_params &&
      ((kernel_params->dispatch_kind ==
            TT_IREE_TTEX_DISPATCH_KIND_BF16_NATIVE &&
        (kernel_params->bf16_native.operation_kind ==
             TT_IREE_TTEX_DISPATCH_KIND_MATMUL_LIKE ||
         kernel_params->bf16_native.operation_kind ==
             TT_IREE_TTEX_DISPATCH_KIND_CONVOLUTION ||
         kernel_params->bf16_native.operation_kind ==
             TT_IREE_TTEX_DISPATCH_KIND_MAX_POOL ||
         kernel_params->bf16_native.operation_kind ==
             TT_IREE_TTEX_DISPATCH_KIND_PADDED_COPY ||
         kernel_params->bf16_native.operation_kind ==
             TT_IREE_TTEX_DISPATCH_KIND_LINEAR)) ||
       (kernel_params->ttex_version >= TT_IREE_TTEX_VERSION_13 &&
        kernel_params->dispatch_kind == TT_IREE_TTEX_DISPATCH_KIND_MAX_POOL) ||
       kernel_params->builtin_program ==
           TT_IREE_TTEX_BUILTIN_PROGRAM_BF16_MATMUL_TILED ||
       kernel_params->builtin_program ==
           TT_IREE_TTEX_BUILTIN_PROGRAM_F32_MATMUL_LIKE ||
       kernel_params->builtin_program ==
           TT_IREE_TTEX_BUILTIN_PROGRAM_F32_CONV2D ||
       kernel_params->builtin_program ==
           TT_IREE_TTEX_BUILTIN_PROGRAM_F32_LINEAR);
  if (is_multicore_compute && config.workgroup_count[2] != 1) {
    return iree_make_status(
        IREE_STATUS_UNIMPLEMENTED,
        "Tenstorrent tiled compute supports a 2D workgroup grid, got %ux%ux%u",
        config.workgroup_count[0], config.workgroup_count[1],
        config.workgroup_count[2]);
  }
  if (!is_multicore_compute &&
      (config.workgroup_count[0] != 1 || config.workgroup_count[1] != 1 ||
       config.workgroup_count[2] != 1)) {
    return iree_make_status(
        IREE_STATUS_UNIMPLEMENTED,
        "Tenstorrent single-core builtin requires workgroup_count=1x1x1, got "
        "%ux%ux%u",
        config.workgroup_count[0], config.workgroup_count[1],
        config.workgroup_count[2]);
  }

  iree_hal_executable_export_info_t export_info = {};
  IREE_RETURN_IF_ERROR(iree_hal_executable_export_info(
      executable, export_ordinal, &export_info));

  const bool has_workgroup_size =
      (config.workgroup_size[0] | config.workgroup_size[1] |
       config.workgroup_size[2]) != 0;
  if (has_workgroup_size &&
      (config.workgroup_size[0] != export_info.workgroup_size[0] ||
       config.workgroup_size[1] != export_info.workgroup_size[1] ||
       config.workgroup_size[2] != export_info.workgroup_size[2])) {
    return iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "dispatch workgroup_size %ux%ux%u does not match export %ux%ux%u",
        config.workgroup_size[0], config.workgroup_size[1],
        config.workgroup_size[2], export_info.workgroup_size[0],
        export_info.workgroup_size[1], export_info.workgroup_size[2]);
  }

  if (export_info.binding_count > IREE_HAL_TT_MAX_BINDINGS ||
      bindings.count > IREE_HAL_TT_MAX_BINDINGS) {
    return iree_make_status(IREE_STATUS_RESOURCE_EXHAUSTED,
                            "too many bindings: export=%u dispatch=%zu max=%d",
                            export_info.binding_count, bindings.count,
                            IREE_HAL_TT_MAX_BINDINGS);
  }
  if (bindings.count != export_info.binding_count) {
    return iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "dispatch binding count %zu does not match export binding count %u",
        bindings.count, export_info.binding_count);
  }
  if (bindings.count > 0 && !bindings.values) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "dispatch bindings are null");
  }

  iree_host_size_t expected_constants_length =
      export_info.constant_count * sizeof(uint32_t);
  if (expected_constants_length > IREE_HAL_TT_MAX_CONSTANTS ||
      constants.data_length > IREE_HAL_TT_MAX_CONSTANTS) {
    return iree_make_status(
        IREE_STATUS_RESOURCE_EXHAUSTED,
        "too many constant bytes: export=%zu dispatch=%zu max=%d",
        expected_constants_length, constants.data_length,
        IREE_HAL_TT_MAX_CONSTANTS);
  }
  if (constants.data_length != expected_constants_length) {
    return iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "dispatch constant byte length %zu does not match export constant "
        "count %u (%zu bytes)",
        constants.data_length, export_info.constant_count,
        expected_constants_length);
  }
  if (constants.data_length > 0 && !constants.data) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "dispatch constants are null");
  }

  for (iree_host_size_t i = 0; i < bindings.count; ++i) {
    const iree_hal_buffer_ref_t &ref = bindings.values[i];
    if (ref.reserved != 0) {
      return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                              "binding[%zu] has non-zero reserved bits", i);
    }
    if (ref.buffer) {
      iree_device_size_t resolved_offset = 0;
      iree_device_size_t resolved_length = 0;
      IREE_RETURN_IF_ERROR(
          iree_hal_buffer_calculate_range(
              /*base_offset=*/0, iree_hal_buffer_byte_length(ref.buffer),
              ref.offset, ref.length, &resolved_offset, &resolved_length),
          "binding[%zu] range", i);
    } else if (ref.buffer_slot >= command_buffer->base.binding_capacity) {
      return iree_make_status(
          IREE_STATUS_OUT_OF_RANGE,
          "binding[%zu] indirect slot %u exceeds command buffer capacity %u", i,
          ref.buffer_slot, command_buffer->base.binding_capacity);
    }
  }

  if (kernel_params && kernel_params->dispatch_kind == TT_IREE_TTEX_DISPATCH_KIND_BF16_NATIVE) {
    for (const auto &tensor : kernel_params->bf16_native.tensors) {
      if (tensor.rank) {
        IREE_RETURN_IF_ERROR(iree_hal_tt_validate_tensor_binding_range(tensor, "BF16 native tensor", bindings));
      }
    }
    if (tt_iree_ttex_is_native_matrix_kind(
            kernel_params->bf16_native.operation_kind)) {
      uint32_t mt = kernel_params->bf16_native.m / 32 + (kernel_params->bf16_native.m % 32 != 0);
      uint32_t nt = kernel_params->bf16_native.n / 32 + (kernel_params->bf16_native.n % 32 != 0);
      if (config.workgroup_count[1] > mt || config.workgroup_count[0] > nt)
        return iree_make_status(IREE_STATUS_INVALID_ARGUMENT, "BF16 native grid cannot partition output tiles");
    }
    return iree_ok_status();
  }
  if (kernel_params && kernel_params->builtin_program ==
                           TT_IREE_TTEX_BUILTIN_PROGRAM_F32_MATMUL_LIKE) {
    const tt_iree_ttex_matmul_like_desc_t &matmul = kernel_params->matmul_like;
    IREE_RETURN_IF_ERROR(iree_hal_tt_validate_tensor_binding_range(
        matmul.lhs, "matmul lhs", bindings));
    IREE_RETURN_IF_ERROR(iree_hal_tt_validate_tensor_binding_range(
        matmul.rhs, "matmul rhs", bindings));
    IREE_RETURN_IF_ERROR(iree_hal_tt_validate_tensor_binding_range(
        matmul.output, "matmul output", bindings));
    if (matmul.epilogue_flags & TT_IREE_TTEX_EPILOGUE_RESIDUAL) {
      IREE_RETURN_IF_ERROR(iree_hal_tt_validate_tensor_binding_range(
          matmul.residual, "matmul residual", bindings));
    }
    if (matmul.epilogue_flags & TT_IREE_TTEX_EPILOGUE_RUNTIME_AFFINE) {
      IREE_RETURN_IF_ERROR(iree_hal_tt_validate_tensor_binding_range(
          matmul.epilogue_mean, "matmul affine mean", bindings));
      IREE_RETURN_IF_ERROR(iree_hal_tt_validate_tensor_binding_range(
          matmul.epilogue_channel_scale, "matmul affine scale", bindings));
      IREE_RETURN_IF_ERROR(iree_hal_tt_validate_tensor_binding_range(
          matmul.epilogue_channel_bias, "matmul affine bias", bindings));
    }
    if (matmul.epilogue_flags &
        TT_IREE_TTEX_EPILOGUE_RESIDUAL_PER_CHANNEL) {
      IREE_RETURN_IF_ERROR(iree_hal_tt_validate_tensor_binding_range(
          matmul.residual_mean, "matmul residual mean", bindings));
      IREE_RETURN_IF_ERROR(iree_hal_tt_validate_tensor_binding_range(
          matmul.residual_channel_scale, "matmul residual scale", bindings));
      IREE_RETURN_IF_ERROR(iree_hal_tt_validate_tensor_binding_range(
          matmul.residual_channel_bias, "matmul residual bias", bindings));
    }
    constexpr uint32_t kTileSize = 32;
    uint32_t m_tiles = (matmul.m + kTileSize - 1) / kTileSize;
    uint32_t n_tiles = (matmul.n + kTileSize - 1) / kTileSize;
    if (native_matrix_grid
            ? (config.workgroup_count[1] > m_tiles ||
               config.workgroup_count[0] > n_tiles)
            : (m_tiles % config.workgroup_count[1] != 0 ||
               n_tiles % config.workgroup_count[0] != 0)) {
      return iree_make_status(
          IREE_STATUS_INVALID_ARGUMENT,
          "workgroup grid %ux%u cannot partition F32 matmul output "
          "tiles %ux%u",
          config.workgroup_count[0], config.workgroup_count[1], n_tiles,
          m_tiles);
    }
    return iree_ok_status();
  }
  if (kernel_params && kernel_params->builtin_program ==
                           TT_IREE_TTEX_BUILTIN_PROGRAM_F32_CONV2D) {
    const tt_iree_ttex_conv2d_desc_t &conv2d = kernel_params->conv2d;
    IREE_RETURN_IF_ERROR(iree_hal_tt_validate_tensor_binding_range(
        conv2d.input, "convolution input", bindings));
    IREE_RETURN_IF_ERROR(iree_hal_tt_validate_tensor_binding_range(
        conv2d.filter, "convolution filter", bindings));
    IREE_RETURN_IF_ERROR(iree_hal_tt_validate_tensor_binding_range(
        conv2d.output, "convolution output", bindings));
    if (conv2d.epilogue_flags & TT_IREE_TTEX_EPILOGUE_RUNTIME_AFFINE) {
      IREE_RETURN_IF_ERROR(iree_hal_tt_validate_tensor_binding_range(
          conv2d.epilogue_mean, "convolution affine mean", bindings));
      IREE_RETURN_IF_ERROR(iree_hal_tt_validate_tensor_binding_range(
          conv2d.epilogue_channel_scale, "convolution affine scale", bindings));
      IREE_RETURN_IF_ERROR(iree_hal_tt_validate_tensor_binding_range(
          conv2d.epilogue_channel_bias, "convolution affine bias", bindings));
    }
    constexpr uint32_t kTileSize = 32;
    uint64_t output_positions =
        static_cast<uint64_t>(conv2d.output.shape[1]) * conv2d.output.shape[2];
    uint64_t m_tiles = (output_positions + kTileSize - 1) / kTileSize;
    uint64_t n_tiles =
        (static_cast<uint64_t>(conv2d.output.shape[0]) + kTileSize - 1) /
        kTileSize;
    if (native_matrix_grid
            ? (config.workgroup_count[1] > m_tiles ||
               config.workgroup_count[0] > n_tiles)
            : (m_tiles % config.workgroup_count[1] != 0 ||
               n_tiles % config.workgroup_count[0] != 0)) {
      return iree_make_status(
          IREE_STATUS_INVALID_ARGUMENT,
          "workgroup grid %ux%u cannot partition F32 convolution "
          "output tiles %" PRIu64 "x%" PRIu64,
          config.workgroup_count[0], config.workgroup_count[1], n_tiles,
          m_tiles);
    }
    return iree_ok_status();
  }
  if (kernel_params && kernel_params->builtin_program ==
                           TT_IREE_TTEX_BUILTIN_PROGRAM_F32_MAX_POOL_2D) {
    const tt_iree_ttex_max_pool_2d_desc_t &pool = kernel_params->max_pool;
    IREE_RETURN_IF_ERROR(iree_hal_tt_validate_tensor_binding_range(
        pool.input, "max-pool input", bindings));
    IREE_RETURN_IF_ERROR(iree_hal_tt_validate_tensor_binding_range(
        pool.output, "max-pool output", bindings));
    return iree_ok_status();
  }
  if (kernel_params && kernel_params->builtin_program ==
                           TT_IREE_TTEX_BUILTIN_PROGRAM_F32_GLOBAL_SUM) {
    const auto &desc = kernel_params->global_sum;
    IREE_RETURN_IF_ERROR(iree_hal_tt_validate_tensor_binding_range(
        desc.input, "global-sum input", bindings));
    IREE_RETURN_IF_ERROR(iree_hal_tt_validate_tensor_binding_range(
        desc.sum, "global-sum output", bindings));
    IREE_RETURN_IF_ERROR(iree_hal_tt_validate_tensor_binding_range(
        desc.divisor, "global-sum divisor", bindings));
    return iree_ok_status();
  }
  if (kernel_params && kernel_params->builtin_program ==
                           TT_IREE_TTEX_BUILTIN_PROGRAM_F32_SCALAR_DIV) {
    const auto &desc = kernel_params->scalar_div;
    IREE_RETURN_IF_ERROR(iree_hal_tt_validate_tensor_binding_range(
        desc.input, "scalar-div input", bindings));
    IREE_RETURN_IF_ERROR(iree_hal_tt_validate_tensor_binding_range(
        desc.divisor, "scalar-div divisor", bindings));
    IREE_RETURN_IF_ERROR(iree_hal_tt_validate_tensor_binding_range(
        desc.output, "scalar-div output", bindings));
    return iree_ok_status();
  }
  if (kernel_params && kernel_params->builtin_program ==
                           TT_IREE_TTEX_BUILTIN_PROGRAM_F32_LINEAR) {
    const auto &desc = kernel_params->linear;
    IREE_RETURN_IF_ERROR(iree_hal_tt_validate_tensor_binding_range(
        desc.input, "linear input", bindings));
    IREE_RETURN_IF_ERROR(iree_hal_tt_validate_tensor_binding_range(
        desc.weights, "linear weights", bindings));
    IREE_RETURN_IF_ERROR(iree_hal_tt_validate_tensor_binding_range(
        desc.bias, "linear bias", bindings));
    IREE_RETURN_IF_ERROR(iree_hal_tt_validate_tensor_binding_range(
        desc.output, "linear output", bindings));
    constexpr uint32_t kTileSize = 32;
    uint32_t m_tiles = (desc.m + kTileSize - 1) / kTileSize;
    uint32_t n_tiles = (desc.n + kTileSize - 1) / kTileSize;
    if (native_matrix_grid
            ? (config.workgroup_count[1] > m_tiles ||
               config.workgroup_count[0] > n_tiles)
            : (m_tiles % config.workgroup_count[1] != 0 ||
               n_tiles % config.workgroup_count[0] != 0)) {
      return iree_make_status(
          IREE_STATUS_INVALID_ARGUMENT,
          "workgroup grid %ux%u cannot partition linear tiles %ux%u",
          config.workgroup_count[0], config.workgroup_count[1], n_tiles,
          m_tiles);
    }
    return iree_ok_status();
  }

  return iree_ok_status();
}

// Records a dispatch command using the current IREE HAL API.
static iree_status_t iree_hal_tt_command_buffer_dispatch(
    iree_hal_command_buffer_t *base, iree_hal_executable_t *executable,
    iree_hal_executable_export_ordinal_t export_ordinal,
    const iree_hal_dispatch_config_t config, iree_const_byte_span_t constants,
    iree_hal_buffer_ref_list_t bindings, iree_hal_dispatch_flags_t flags) {

  iree_hal_tt_command_buffer_t *command_buffer =
      iree_hal_tt_command_buffer_cast(base);

  IREE_RETURN_IF_ERROR(iree_hal_tt_command_buffer_validate_dispatch(
      command_buffer, executable, export_ordinal, config, constants, bindings,
      flags));

  iree_hal_tt_command_t cmd;
  cmd.type = IREE_HAL_TT_COMMAND_TYPE_DISPATCH;
  cmd.dispatch.executable = executable;
  cmd.dispatch.export_ordinal = export_ordinal;
  cmd.dispatch.config = config;
  cmd.dispatch.flags = flags;

  // Copy bindings
  cmd.dispatch.binding_count = bindings.count;
  if (bindings.count > 0) {
    std::memcpy(cmd.dispatch.bindings, bindings.values,
                bindings.count * sizeof(iree_hal_buffer_ref_t));
  }

  // Copy constants
  cmd.dispatch.constants_length = constants.data_length;
  if (constants.data_length > 0) {
    std::memcpy(cmd.dispatch.constants, constants.data, constants.data_length);
  }

  command_buffer->commands.push_back(cmd);

  // Retain executable
  iree_hal_executable_retain(executable);
  command_buffer->resources.push_back((iree_hal_resource_t *)executable);

  // Retain buffers from bindings
  for (iree_host_size_t i = 0; i < bindings.count; i++) {
    if (bindings.values[i].buffer) {
      iree_hal_buffer_retain(bindings.values[i].buffer);
      command_buffer->resources.push_back(
          (iree_hal_resource_t *)bindings.values[i].buffer);
    }
  }

  return iree_ok_status();
}

//===----------------------------------------------------------------------===//
// Public Accessor
//===----------------------------------------------------------------------===//

iree_hal_tt_command_t *
iree_hal_tt_command_buffer_get_commands(iree_hal_command_buffer_t *base,
                                        iree_host_size_t *out_count) {
  iree_hal_tt_command_buffer_t *command_buffer =
      iree_hal_tt_command_buffer_cast(base);
  *out_count = command_buffer->commands.size();
  return command_buffer->commands.data();
}

iree_status_t iree_hal_tt_command_buffer_get_update_data(
    iree_hal_command_buffer_t *base, iree_host_size_t offset,
    iree_host_size_t length, iree_const_byte_span_t *out_data) {
  iree_hal_tt_command_buffer_t *command_buffer =
      iree_hal_tt_command_buffer_cast(base);
  if (!out_data) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "out_data must not be null");
  }
  *out_data = iree_const_byte_span_empty();
  if (offset > command_buffer->update_data.size() ||
      length > command_buffer->update_data.size() - offset) {
    return iree_make_status(
        IREE_STATUS_OUT_OF_RANGE,
        "update data range exceeds command buffer payload (offset=%" PRIhsz
        ", length=%" PRIhsz ", payload=%" PRIhsz ")",
        offset, length, command_buffer->update_data.size());
  }
  *out_data = iree_make_const_byte_span(
      command_buffer->update_data.data() + offset, length);
  return iree_ok_status();
}

//===----------------------------------------------------------------------===//
// VTable
//===----------------------------------------------------------------------===//

const iree_hal_command_buffer_vtable_t iree_hal_tt_command_buffer_vtable = {
    .destroy = iree_hal_tt_command_buffer_destroy,
    .begin = iree_hal_tt_command_buffer_begin,
    .end = iree_hal_tt_command_buffer_end,
    .begin_debug_group = iree_hal_tt_command_buffer_begin_debug_group,
    .end_debug_group = iree_hal_tt_command_buffer_end_debug_group,
    .execution_barrier = iree_hal_tt_command_buffer_execution_barrier,
    .signal_event = iree_hal_tt_command_buffer_signal_event,
    .reset_event = iree_hal_tt_command_buffer_reset_event,
    .wait_events = iree_hal_tt_command_buffer_wait_events,
    .advise_buffer = iree_hal_tt_command_buffer_advise_buffer,
    .fill_buffer = iree_hal_tt_command_buffer_fill_buffer,
    .update_buffer = iree_hal_tt_command_buffer_update_buffer,
    .copy_buffer = iree_hal_tt_command_buffer_copy_buffer,
    .collective = iree_hal_tt_command_buffer_collective,
    .dispatch = iree_hal_tt_command_buffer_dispatch,
};
