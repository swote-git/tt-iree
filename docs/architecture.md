# Architecture

Technical architecture of the tt-iree project.

## System Overview

```
┌─────────────────────────────────────────────────────────────┐
│                    ML Frameworks                            │
│              PyTorch │ JAX │ TensorFlow                     │
└────────────────────────┬────────────────────────────────────┘
                         │
                         ▼ (torch-mlir, jax2mlir, etc.)
┌─────────────────────────────────────────────────────────────┐
│                  MLIR Input Dialects                        │
│              StableHLO │ TOSA │ Linalg                      │
└────────────────────────┬────────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────────────┐
│                 IREE Compiler Pipeline                      │
│  ┌───────────────────────────────────────────────────────┐  │
│  │ Flow Dialect    - Dispatch region formation           │  │
│  │ Stream Dialect  - Async execution modeling            │  │
│  │ HAL Dialect     - Hardware abstraction                │  │
│  └───────────────────────────────────────────────────────┘  │
│                         │                                   │
│                         ▼                                  │
│  ┌───────────────────────────────────────────────────────┐  │
│  │ Tenstorrent Target Backend (tt-iree)                  │  │
│  │  - Linalg → TT-Metal kernel conversion                │  │
│  │  - Tile layout transformation (32x32)                 │  │
│  │  - Core grid mapping (8x8 Tensix)                     │  │
│  └───────────────────────────────────────────────────────┘  │
└────────────────────────┬────────────────────────────────────┘
                         │
                         ▼
              ┌─────────────────────┐
              │  .vmfb (FlatBuffer) │
              │  + TT-Metal kernel  │
              └──────────┬──────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────────────┐
│                    IREE Runtime                             │
│  ┌───────────────────────────────────────────────────────┐  │
│  │ Tenstorrent HAL Driver (tt-iree)                      │  │
│  │  - Device management                                  │  │
│  │  - Buffer allocation (DRAM/L1)                        │  │
│  │  - Command buffer execution                           │  │
│  └───────────────────────────────────────────────────────┘  │
└────────────────────────┬────────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────────────┐
│                   TT-Metal Runtime                          │
│              TTNN │ TT-Metal API │ Device Driver            │
└────────────────────────┬────────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────────────┐
│              Tenstorrent Hardware (P100A)                   │
│                  8x8 Tensix Core Grid                       │
│                     24GB GDDR6                              │
└─────────────────────────────────────────────────────────────┘
```

## Key Components

### 1. Compiler Backend

**Location:** `compiler/plugins/target/tenstorrent/`

The compiler backend extends IREE to generate code for Tenstorrent hardware.

**Responsibilities:**
- Register "tenstorrent" target with IREE
- Convert Linalg operations to TT-Metal kernels
- Handle tile layout transformation (row-major → 32x32 tiles)
- Map workloads to Tensix core grid
- Serialize executables to FlatBuffer format

**Key Design Decision:** Generate TT-Metal kernel code directly rather than
going through tt-mlir dialects. This keeps the PoC simple while maintaining
the option to integrate tt-mlir optimizations later.

### 2. Runtime HAL Driver

**Location:** `runtime/src/iree/hal/drivers/tenstorrent/`

The runtime driver implements IREE's Hardware Abstraction Layer (HAL) interface.

| Component | File | Description |
|-----------|------|-------------|
| Driver | `tt_driver.c` | Driver registration, device enumeration |
| Device | `tt_device.c` | Device lifecycle, capability queries |
| Allocator | `tt_allocator.c` | Buffer allocation (DRAM/L1) |
| Buffer | `tt_buffer.c` | Memory management with tile layout |
| Command Buffer | `tt_command_buffer.c` | Command recording and dispatch |
| Executable | `tt_executable.c` | Kernel loading and execution |
| Semaphore | `tt_semaphore.c` | Synchronization primitives |

## Memory Architecture

### Tenstorrent Memory Hierarchy

```
┌─────────────────────────────────────────┐
│           Host Memory (System RAM)       │
└────────────────────┬────────────────────┘
                     │ PCIe
                     ▼
┌─────────────────────────────────────────┐
│         Device DRAM (24GB GDDR6)         │
│         - Large capacity                 │
│         - Higher latency                 │
└────────────────────┬────────────────────┘
                     │ NoC (Network-on-Chip)
                     ▼
┌─────────────────────────────────────────┐
│      L1 SRAM (~1MB per Tensix core)      │
│      - Fast access                       │
│      - Limited capacity                  │
│      - Distributed across 64 cores       │
└─────────────────────────────────────────┘
```

### Memory Layout Types

Tenstorrent uses tile-based memory layouts:

| Layout | Description | Use Case |
|--------|-------------|----------|
| Row-Major | Standard contiguous layout | Host tensors |
| Tile (32x32) | 32x32 element tiles | Device compute |
| Interleaved | Tiles distributed across cores | Default device layout |
| Sharded | Tiles fixed to specific cores | Optimized data locality |

### HAL Buffer Mapping

```c
// IREE buffer types → TT-Metal memory
IREE_HAL_MEMORY_TYPE_DEVICE_LOCAL → DRAM or L1 (based on size)
IREE_HAL_MEMORY_TYPE_HOST_VISIBLE → System memory (pinned)

// Allocation strategy
if (size < L1_THRESHOLD && frequently_accessed) {
    allocate_in_L1();
} else {
    allocate_in_DRAM();
}
```

## Execution Model

### Command Buffer Flow

```
1. Create command buffer
   iree_hal_command_buffer_create()
   
2. Record commands (no execution yet)
   iree_hal_command_buffer_begin()
   iree_hal_command_buffer_dispatch()
   iree_hal_command_buffer_end()
   
3. Submit for execution
   iree_hal_device_queue_execute()
   
4. Wait for completion
   iree_hal_semaphore_wait()
```

### Dispatch Execution

```c
// Each IREE dispatch → One TT-Metal program
iree_hal_command_buffer_dispatch(
    command_buffer,
    executable,           // Contains TT-Metal kernel
    entry_point,          // Kernel function
    workgroup_count[3],   // Maps to core grid
    push_constants,       // Kernel arguments
    binding_table         // Buffer bindings
);
```

## Integration Points

### IREE ↔ TT-Metal Mapping

| IREE Concept | TT-Metal Equivalent |
|--------------|---------------------|
| `iree_hal_driver_t` | System-level initialization |
| `iree_hal_device_t` | `tt::tt_metal::Device` |
| `iree_hal_buffer_t` | `tt::tt_metal::Buffer` |
| `iree_hal_command_buffer_t` | `tt::tt_metal::CommandQueue` |
| `iree_hal_executable_t` | `tt::tt_metal::Program` |
| Dispatch | `EnqueueProgram()` |

### Preserving IREE Optimizations

The HAL driver preserves IREE's optimization decisions:

- **Dispatch formation:** IREE decides what operations to fuse
- **Buffer lifetime:** IREE manages allocation/deallocation timing
- **Async execution:** IREE schedules overlapping operations
- **Memory aliasing:** IREE determines buffer reuse

The Tenstorrent backend handles hardware-specific concerns:
- Tile layout transformation
- Core grid mapping
- L1 vs DRAM placement
- Kernel code generation

## Build Modes

### Mock Mode (No Hardware)

```cmake
-DTT_IREE_ENABLE_MOCK=ON
```

- Uses host memory instead of device memory
- Executes kernels on CPU
- Enables development without P100A hardware
- Useful for CI/CD and initial development

### Hardware Mode

```cmake
-DTT_IREE_ENABLE_MOCK=OFF
-DTT_IREE_ENABLE_TTNN=ON
```

- Requires TT-Metal SDK and P100A hardware
- Full device execution
- Real performance measurement

## References

- [IREE HAL Design](https://iree.dev/developers/design-docs/hal-design/)
- [tt-metal Programming Guide](https://docs.tenstorrent.com/)
- [tt-mlir Dialects](https://docs.tenstorrent.com/tt-mlir/)
