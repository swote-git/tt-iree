# PoC Development Plan

Proof of Concept development plan for tt-iree targeting Tenstorrent P100A.

## Objective

Achieve end-to-end execution of a single operation on real P100A hardware via IREE.

| Attribute | Value |
|-----------|-------|
| Target Hardware | Tenstorrent P100A (Wormhole, 120 Cores) |
| SDK Version | tt-metal v0.65.0 |
| IREE Version | v3.9.0 |
| Target Operation | Elementwise Add (`x + 1.0`) on 32x32 Tile |
| Success Criteria | Correct calculation result on hardware |
| Duration | 6-8 Weeks |

## Why Hardware-First?

With P100A access available through the Tenstorrent Developer Program, we skip 
the Mock implementation phase:

- **No double work**: Avoid writing Mock code that gets replaced
- **Real validation**: Catch hardware-specific issues early
- **Faster iteration**: Direct feedback from actual execution

## Goals

### Primary Goals

1. **Hardware Interaction Verification**
   - Successfully initialize P100A (120 Cores) from IREE Runtime
   - Allocate buffers in P100A DRAM
   - Execute kernel on Tensix core

2. **Valid Kernel Generation**
   - Compiler generates valid TT-Metal C++ Kernels (Reader/Writer/Compute)
   - Compiler handles 32x32 Tiling correctly
   - Proper separation of data movement and compute kernels

3. **Data Movement Verification**
   - Host (Row-Major) ↔ Device (Tile Layout) conversion works correctly
   - Circular buffer synchronization works

### Non-Goals (Deferred to MVP)

- Multi-core distribution (use single core only)
- L1 memory optimization
- Async execution
- Multiple operations
- Performance tuning

## Scope Constraints

To ensure success within 8 weeks, strictly limit scope:

| Constraint | Value | Rationale |
|------------|-------|-----------|
| Core Count | **Single Core (0,0)** | Avoid multi-core complexity |
| Tile Count | **Single 32x32 Tile** | Minimum viable size |
| Memory | **DRAM only** | Let TT-Metal handle L1 via CBs |
| Execution | **Blocking (sync)** | Use `Finish()` after dispatch |
| Data Type | **Float32** | Avoid BFloat16 conversion issues |

## Timeline

### Week 1: Environment & SDK Verification

**Objective:** Verify P100A works with TT-Metal SDK (independent of IREE)

**Tasks:**
- [ ] Install P100A drivers & firmware
- [ ] Build TT-Metal from source (v0.65.0)
- [ ] Run SDK examples: `hello_world`, `eltwise_binary`
- [ ] Verify 120-core grid: `tt-smi` shows full 10x12 grid
- [ ] Set up tt-iree repo with TT-Metal linking

**Verification:**
```bash
# Must pass before any IREE development
cd tt-metal/build
./test/tt_metal/test_eltwise_binary

# Check device
tt-smi
```

**Deliverables:**
- Working development environment
- tt-iree CMake links against `libtt_metal.so`

### Week 2-3: Runtime HAL Driver (Hardware)

**Objective:** Implement IREE HAL driver that talks to real P100A

**Tasks:**
- [ ] `tt_driver.c`: Query available devices via TT-Metal
- [ ] `tt_device.c`: Call `tt::tt_metal::CreateDevice(0)`
- [ ] `tt_allocator.c`: Implement DRAM buffer allocation
  - Use `tt::tt_metal::CreateBuffer()` with `BufferType::DRAM`
- [ ] `tt_buffer.c`: Implement Host ↔ Device transfer
  - **Critical:** Implement tile packing/unpacking on CPU side
- [ ] `tt_command_buffer.c`: Wrap `CommandQueue`

**Tile Layout Conversion:**
```cpp
// Host (row-major) -> Device (32x32 tiles)
void pack_to_tiles(const float* src, float* dst, int rows, int cols) {
  for (int tr = 0; tr < rows/32; tr++) {
    for (int tc = 0; tc < cols/32; tc++) {
      for (int r = 0; r < 32; r++) {
        for (int c = 0; c < 32; c++) {
          int src_idx = (tr*32 + r) * cols + (tc*32 + c);
          int dst_idx = (tr * (cols/32) + tc) * 1024 + r * 32 + c;
          dst[dst_idx] = src[src_idx];
        }
      }
    }
  }
}
```

**Deliverables:**
- Driver initializes P100A successfully
- Buffer round-trip test passes: Host → DRAM → Host

**Test:**
```cpp
TEST(TenstorrentRuntime, BufferRoundTrip) {
  // 1. Create Device
  auto device = tt::tt_metal::CreateDevice(0);
  
  // 2. Allocate Buffer (DRAM, 32x32 floats = 4KB)
  auto buffer = tt::tt_metal::CreateBuffer(...);
  
  // 3. Write test data (Host -> Device)
  std::vector<float> input(1024, 1.0f);
  tt::tt_metal::WriteToBuffer(buffer, input);
  
  // 4. Read back (Device -> Host)
  std::vector<float> output(1024);
  tt::tt_metal::ReadFromBuffer(buffer, output);
  
  // 5. Compare
  EXPECT_EQ(input, output);
}
```

### Week 4-5: Compiler Backend (Real Kernels)

**Objective:** Generate valid TT-Metal C++ kernels for `arith.addf`

**Tasks:**
- [ ] Register "tenstorrent" target backend with IREE
- [ ] Implement 3-file kernel generation structure:
  - `reader_kernel.cpp`: DRAM → L1 (via Circular Buffer)
  - `writer_kernel.cpp`: L1 → DRAM
  - `compute_kernel.cpp`: Element-wise add using tile API
- [ ] Tiling pass: Ensure `tensor<32x32xf32>` maps to single tile
- [ ] FlatBuffer serialization: Embed C++ source strings

**Generated Kernel Structure:**
```cpp
// reader_kernel.cpp
void kernel_main() {
  uint32_t src_addr = get_arg_val<uint32_t>(0);
  constexpr uint32_t cb_id = 0;
  
  cb_reserve_back(cb_id, 1);
  uint32_t l1_addr = get_write_ptr(cb_id);
  noc_async_read_tile(0, src_addr, l1_addr);
  noc_async_read_barrier();
  cb_push_back(cb_id, 1);
}

// compute_kernel.cpp  
void kernel_main() {
  constexpr uint32_t cb_in = 0;
  constexpr uint32_t cb_out = 1;
  
  cb_wait_front(cb_in, 1);
  cb_reserve_back(cb_out, 1);
  
  // Add scalar 1.0 to each element
  add_tiles_init();
  add_tiles(cb_in, cb_scalar, 0, 0, 0);
  
  cb_push_back(cb_out, 1);
  cb_pop_front(cb_in, 1);
}

// writer_kernel.cpp
void kernel_main() {
  uint32_t dst_addr = get_arg_val<uint32_t>(0);
  constexpr uint32_t cb_id = 1;
  
  cb_wait_front(cb_id, 1);
  uint32_t l1_addr = get_read_ptr(cb_id);
  noc_async_write_tile(0, dst_addr, l1_addr);
  noc_async_write_barrier();
  cb_pop_front(cb_id, 1);
}
```

**Deliverables:**
- `iree-compile --iree-hal-target-backends=tenstorrent` produces `.vmfb`
- FlatBuffer contains valid TT-Metal C++ source

### Week 6: JIT Compilation & Kernel Loading

**Objective:** Compile generated C++ kernels at runtime

**Tasks:**
- [ ] `tt_executable.c`: Extract C++ source from FlatBuffer
- [ ] Create `tt::tt_metal::Program` with CircularBuffer config
- [ ] Create kernels via `tt::tt_metal::CreateKernel()` (JIT compilation)
- [ ] Map IREE buffer bindings to kernel runtime arguments

**JIT Flow:**
```cpp
iree_status_t iree_hal_tt_executable_create(...) {
  // 1. Parse FlatBuffer, extract kernel sources
  auto reader_src = extract_kernel(fb, "reader");
  auto compute_src = extract_kernel(fb, "compute");
  auto writer_src = extract_kernel(fb, "writer");
  
  // 2. Create Program
  auto program = tt::tt_metal::CreateProgram();
  
  // 3. Configure Circular Buffers
  auto cb_config = tt::tt_metal::CircularBufferConfig(...)
    .set_page_size(0, 1024 * sizeof(float));
  tt::tt_metal::CreateCircularBuffer(program, core, cb_config);
  
  // 4. Create Kernels (JIT compile happens here)
  tt::tt_metal::CreateKernel(program, reader_src, core, 
    tt::tt_metal::DataMovementConfig{...});
  tt::tt_metal::CreateKernel(program, compute_src, core,
    tt::tt_metal::ComputeConfig{...});
  tt::tt_metal::CreateKernel(program, writer_src, core,
    tt::tt_metal::DataMovementConfig{...});
  
  return iree_ok_status();
}
```

**Note:** First run will be slow (~10-30s) due to JIT compilation.
Enable TT-Metal kernel caching for faster subsequent runs.

**Deliverables:**
- Runtime successfully JIT compiles kernels
- Program executes without hang

### Week 7-8: Integration & Debugging

**Objective:** End-to-end execution and result verification

**Tasks:**
- [ ] Run `simple_add.mlir` (32x32 input)
- [ ] Debug data mismatches:
  - Endianness issues
  - Tile layout correctness
  - Off-by-one in index calculations
- [ ] Debug hangs:
  - Circular buffer overflow/underflow
  - Missing barriers
  - Deadlocks between Reader/Compute/Writer
- [ ] Verify floating point accuracy

**Debugging Tools:**
```bash
# Reset hung device
tt-smi -r 0

# Check device status
tt-smi

# Enable TT-Metal debug logging
export TT_METAL_LOGGER_LEVEL=DEBUG
```

**Deliverables:**
- E2E test passes on P100A hardware
- Documentation complete

## Test Plan

### 1. SDK Sanity Check (Week 1)
```bash
# Run before any IREE work
cd tt-metal/build
./test/tt_metal/test_eltwise_binary
```

### 2. Buffer Round-Trip Test (Week 2-3)
```cpp
TEST(TenstorrentRuntime, BufferRoundTrip) {
  // Create 32x32 float buffer, write, read, compare
  // Validates: Device init, DRAM alloc, data transfer
}
```

### 3. Tile Layout Test (Week 2-3)
```cpp
TEST(TenstorrentRuntime, TilePackUnpack) {
  // Row-major -> Tile -> Row-major
  // Must be bit-exact
}
```

### 4. Compile Test (Week 4-5)
```bash
iree-compile \
  --iree-hal-target-backends=tenstorrent \
  test/simple_add.mlir \
  -o /tmp/simple_add.vmfb

# Inspect FlatBuffer contains kernel source
```

### 5. E2E Test (Week 7-8)
```python
# examples/simple_add.py
import numpy as np

mlir = """
func.func @main(%arg0: tensor<32x32xf32>) -> tensor<32x32xf32> {
  %c = arith.constant dense<1.0> : tensor<32x32xf32>
  %0 = arith.addf %arg0, %c : tensor<32x32xf32>
  return %0 : tensor<32x32xf32>
}
"""

# Compile & run on tenstorrent device
input_data = np.ones((32, 32), dtype=np.float32)
result = run_on_tenstorrent(mlir, input_data)

expected = np.full((32, 32), 2.0, dtype=np.float32)
assert np.allclose(result, expected)
print("PoC Success on P100A!")
```

## Risks & Mitigations

| Risk | Impact | Likelihood | Mitigation |
|------|--------|------------|------------|
| JIT compilation slow | Medium | High | Enable kernel caching |
| Device hangs | High | Medium | Use `tt-smi -r` to reset; keep kernels simple |
| Tile layout bugs | Medium | High | Test pack/unpack in isolation first |
| IREE API complexity | High | Medium | Study HIP driver as reference |
| Build system issues | Medium | High | Start minimal, expand incrementally |

## Success Criteria

| Criterion | Target | Measurement |
|-----------|--------|-------------|
| Device Init | Works | `CreateDevice(0)` succeeds |
| Buffer Transfer | Works | Round-trip test passes |
| Compilation | Works | `.vmfb` contains valid kernels |
| Execution | Works | Kernel runs without hang |
| Accuracy | 100% | Results match expected |

## Definition of Done

PoC is complete when:

- [ ] P100A initializes successfully from IREE HAL
- [ ] `iree-compile --iree-hal-target-backends=tenstorrent` produces valid output
- [ ] `iree-run-module --device=tenstorrent` executes kernel on hardware
- [ ] 32x32 elementwise add produces correct results
- [ ] Documentation is complete

## Transition to MVP

After PoC completion:

1. **Multi-Core:** Distribute work across core grid
2. **More Ops:** matmul, relu, conv2d
3. **L1 Optimization:** Manual sharding for performance
4. **Async Execution:** Pipeline multiple dispatches
5. **Target:** MNIST MLP inference