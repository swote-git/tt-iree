# HAL Driver Implementation

Detailed design of the Tenstorrent HAL driver.

## Overview

The HAL (Hardware Abstraction Layer) driver implements IREE's device interface
for Tenstorrent hardware. It follows patterns established by existing IREE
drivers (CUDA, HIP, Vulkan) while adapting to Tenstorrent's unique architecture.

## Component Hierarchy

```
iree_hal_driver_t (tt_driver)
    │
    └── iree_hal_device_t (tt_device)
            │
            ├── iree_hal_allocator_t (tt_allocator)
            │       └── iree_hal_buffer_t (tt_buffer)
            │
            ├── iree_hal_command_buffer_t (tt_command_buffer)
            │
            ├── iree_hal_executable_t (tt_executable)
            │
            └── iree_hal_semaphore_t (tt_semaphore)
```

## Driver (tt_driver.c)

The driver is the entry point for device enumeration and creation.

### Registration

```c
// Called at startup to register the driver factory
IREE_API_EXPORT iree_status_t iree_hal_tenstorrent_driver_module_register(
    iree_hal_driver_registry_t* registry);
```

### Device Enumeration

```c
// Mock mode: Returns one simulated device
// Hardware mode: Queries TT-Metal for available devices
static iree_status_t iree_hal_tt_driver_query_available_devices(
    iree_hal_driver_t* base_driver,
    iree_allocator_t host_allocator,
    iree_host_size_t* out_device_info_count,
    iree_hal_device_info_t** out_device_infos);
```

## Device (tt_device.c)

Represents a single Tenstorrent accelerator (e.g., one P100A card).

### Structure

```c
typedef struct iree_hal_tt_device_t {
  iree_hal_resource_t resource;
  iree_allocator_t host_allocator;
  
  iree_string_view_t identifier;
  iree_hal_device_id_t device_id;
  
  // Device allocator
  iree_hal_allocator_t* device_allocator;
  
  // TT-Metal handle (hardware mode only)
#ifndef TT_IREE_ENABLE_MOCK
  tt::tt_metal::Device* tt_device;
  // P100A supports multiple HW command queues
  // Use separate queues for Compute and Data Movement if possible
  tt::tt_metal::CommandQueue* compute_queue;
  tt::tt_metal::CommandQueue* transfer_queue;
#endif
} iree_hal_tt_device_t;
```

### Key Operations

| Method | Description |
|--------|-------------|
| `create` | Open device, initialize TT-Metal |
| `destroy` | Close device, cleanup resources |
| `device_allocator` | Return buffer allocator |
| `create_command_buffer` | Create command recording buffer |
| `create_executable_cache` | Create executable loader |
| `queue_execute` | Submit commands for execution |

## Allocator (tt_allocator.c)

Manages buffer allocation on device memory.

### Memory Types

```c
typedef enum iree_hal_tt_memory_type_t {
  IREE_HAL_TT_MEMORY_DRAM,            // 28GB (P100A)
  IREE_HAL_TT_MEMORY_L1_INTERLEAVED,  // Small buffers distributed across cores
  IREE_HAL_TT_MEMORY_L1_SHARDED,      // Height/Width sharded (Main compute buffers)
} iree_hal_tt_memory_type_t;
```

### Allocation Strategy

```c
iree_status_t iree_hal_tt_allocator_allocate_buffer(
    iree_hal_allocator_t* allocator,
    iree_hal_buffer_params_t params,
    iree_device_size_t allocation_size,
    iree_hal_buffer_t** out_buffer) {
  
  // Determine memory type based on usage hints
  iree_hal_tt_memory_type_t memory_type;
  
  if (params.usage & IREE_HAL_BUFFER_USAGE_DISPATCH_STORAGE) {
    // P100A L1 is 1.5MB per core. Leave room for code/stack (~200KB).
    const iree_device_size_t L1_SAFE_LIMIT = 1.3 * 1024 * 1024; 
    
    if (allocation_size <= L1_SAFE_LIMIT && is_sharding_compatible(params)) {
      memory_type = IREE_HAL_TT_MEMORY_L1_SHARDED;
    } else {
      memory_type = IREE_HAL_TT_MEMORY_DRAM;
    }
  }
  
  return iree_hal_tt_buffer_create(allocator, memory_type, ...);
}
```

## Buffer (tt_buffer.c)

Represents an allocated memory region.

### Structure

```c
typedef struct iree_hal_tt_buffer_t {
  iree_hal_buffer_t base;
  
  // Memory type
  iree_hal_tt_memory_type_t memory_type;
  
  // TT-Metal buffer (hardware mode)
#ifndef TT_IREE_ENABLE_MOCK
  std::shared_ptr<tt::tt_metal::Buffer> tt_buffer;
#else
  // Mock mode: host memory
  void* host_ptr;
#endif
  
} iree_hal_tt_buffer_t;
```

### Tile Layout

Tenstorrent requires 32x32 tile layout for compute operations:

```c
// Convert row-major to tile layout during buffer write
iree_status_t iree_hal_tt_buffer_map_write(
    iree_hal_buffer_t* buffer,
    iree_device_size_t local_byte_offset,
    const void* source,
    iree_device_size_t length) {
  
  // For compute buffers, apply tile transformation
  if (needs_tile_layout(buffer)) {
    transform_to_tile_layout(source, buffer, length);
  } else {
    // Direct copy for non-compute buffers
    memcpy(buffer_ptr + local_byte_offset, source, length);
  }
  
  return iree_ok_status();
}
```

## Command Buffer (tt_command_buffer.c)

Records commands for later execution.

### Structure

```c
typedef struct iree_hal_tt_command_buffer_t {
  iree_hal_command_buffer_t base;
  
  iree_hal_tt_device_t* device;
  iree_hal_command_buffer_mode_t mode;
  
  // Recorded commands
  iree_arena_t arena;
  struct {
    iree_hal_tt_command_type_t type;
    union {
      struct { /* dispatch args */ } dispatch;
      struct { /* copy args */ } copy;
      struct { /* barrier args */ } barrier;
    };
  }* commands;
  iree_host_size_t command_count;
  
} iree_hal_tt_command_buffer_t;
```

### Command Recording

```c
// Record a dispatch command (doesn't execute yet)
iree_status_t iree_hal_tt_command_buffer_dispatch(
    iree_hal_command_buffer_t* base,
    iree_hal_executable_t* executable,
    int32_t entry_point,
    uint32_t workgroup_x,
    uint32_t workgroup_y,
    uint32_t workgroup_z,
    iree_const_byte_span_t push_constants) {
  
  iree_hal_tt_command_buffer_t* command_buffer = ...;
  
  // Store command for later execution
  command_buffer->commands[command_buffer->command_count++] = {
    .type = IREE_HAL_TT_COMMAND_DISPATCH,
    .dispatch = {
      .executable = executable,
      .entry_point = entry_point,
      .workgroup_count = {workgroup_x, workgroup_y, workgroup_z},
      .push_constants = push_constants,
    },
  };
  
  return iree_ok_status();
}
```

## Executable (tt_executable.c)

Contains compiled kernel code.

### Structure

```c
typedef struct iree_hal_tt_executable_t {
  iree_hal_resource_t resource;
  
  // Entry point count
  iree_host_size_t entry_point_count;
  
  // TT-Metal programs (one per entry point)
#ifndef TT_IREE_ENABLE_MOCK
  std::vector<tt::tt_metal::Program> programs;
#else
  // Mock mode: function pointers
  void (*mock_kernels[])(void* args);
#endif
  
} iree_hal_tt_executable_t;
```

### Loading

```c
iree_status_t iree_hal_tt_executable_create(
    iree_hal_device_t* device,
    const iree_hal_executable_params_t* params,
    iree_hal_executable_t** out_executable) {
  
  // Parse executable from FlatBuffer
  // Extract TT-Metal kernel source
  // Create TT-Metal Program objects
  
  return iree_ok_status();
}
```

## Execution Flow

### Queue Execute

```c
// Note: This is a simplified synchronous implementation for PoC.
// Production driver must use tt_metal::EnqueueWaitForEvent() for true async execution.
iree_status_t iree_hal_tt_device_queue_execute(...) {
  
  // 1. Host-side wait (Bottleneck! Should be device-side wait)
  for (auto& sem : wait_semaphore_list) {
    iree_hal_semaphore_wait(sem, ...); 
  }
  
  // 2. Enqueue commands to HW Queue (Non-blocking)
  for (size_t i = 0; i < command_buffer_count; i++) {
     EnqueueProgram(device->compute_queue, ...);
  }
  
  // 3. Finish/Flush (Ideally, use Event for signaling)
  Finish(device->compute_queue); 
  
  // 4. Signal output semaphores
  for (auto& sem : signal_semaphore_list) {
    iree_hal_semaphore_signal(sem, ...);
  }
  
  return iree_ok_status();
}
```

## Mock Mode Implementation

Mock mode enables development without hardware:

```c
#ifdef TT_IREE_ENABLE_MOCK

// Buffer uses host memory
iree_status_t iree_hal_tt_buffer_create_mock(
    iree_device_size_t size,
    iree_hal_tt_buffer_t** out_buffer) {
  
  buffer->host_ptr = malloc(size);
  return iree_ok_status();
}

// Dispatch executes on CPU
iree_status_t execute_dispatch_mock(
    iree_hal_tt_executable_t* executable,
    int32_t entry_point,
    void** buffers) {
  
  // Call mock kernel function
  executable->mock_kernels[entry_point](buffers);
  return iree_ok_status();
}

#endif  // TT_IREE_ENABLE_MOCK
```

## Error Handling

All functions return `iree_status_t`:

```c
// Check TT-Metal errors
if (tt_metal_status != TT_METAL_SUCCESS) {
  return iree_make_status(
      IREE_STATUS_INTERNAL,
      "TT-Metal error: %s",
      tt_metal_status_string(tt_metal_status));
}

// Propagate errors
IREE_RETURN_IF_ERROR(some_operation());
```

## Thread Safety

- Driver and device creation: Not thread-safe (typically done once)
- Buffer allocation: Thread-safe (uses allocator lock)
- Command buffer recording: Not thread-safe (one thread per command buffer)
- Queue execution: Thread-safe (uses queue lock)

## References

- [IREE HAL API](https://github.com/iree-org/iree/tree/main/runtime/src/iree/hal)
- [IREE CUDA Driver](https://github.com/iree-org/iree/tree/main/runtime/src/iree/hal/drivers/cuda)
- [TT-Metal API](https://docs.tenstorrent.com/)
