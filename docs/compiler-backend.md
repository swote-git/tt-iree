# Compiler Backend Design

Design of the Tenstorrent compiler target backend.

## Overview

The compiler backend extends IREE to generate executable code for Tenstorrent
hardware. It plugs into IREE's HAL target infrastructure and produces
TT-Metal kernel code embedded in IREE's FlatBuffer format.

## Plugin Architecture

```
IREE Compiler
    │
    ├── Built-in targets (LLVM-CPU, VMVX, etc.)
    │
    └── Plugins (loaded at compile time)
            │
            └── tt-iree compiler plugin
                    │
                    └── TenstorrentTargetBackend
```

### Registration

```cpp
// iree_compiler_plugin.cmake
iree_compiler_register_plugin(
  NAME hal_target_tenstorrent
  SRCS TenstorrentTarget.cpp
  DEPS ...
)
```

```cpp
// Plugin entry point
extern "C" bool iree_register_compiler_plugin_hal_target_tenstorrent(
    mlir::iree_compiler::PluginRegistrar* registrar) {
  registrar->registerPlugin<TenstorrentSession>("hal_target_tenstorrent");
  return true;
}
```

## Compilation Pipeline

```
Input: IREE HAL Executable
    │
    ▼
┌─────────────────────────────────────┐
│ 1. Extract Dispatch Functions       │
│    - One function per dispatch      │
│    - Contains Linalg/tensor ops     │
└─────────────────────────────────────┘
    │
    ▼
┌─────────────────────────────────────┐
│ 2. Analyze Operations               │
│    - Identify supported ops         │
│    - Determine tile sizes           │
│    - Plan core grid mapping         │
└─────────────────────────────────────┘
    │
    ▼
┌─────────────────────────────────────┐
│ 3. Generate TT-Metal Kernel         │
│    - Template-based generation      │
│    - Insert operation code          │
│    - Configure memory layout        │
└─────────────────────────────────────┘
    │
    ▼
┌─────────────────────────────────────┐
│ 4. Serialize to FlatBuffer          │
│    - Embed kernel source            │
│    - Store metadata                 │
│    - Package entry points           │
└─────────────────────────────────────┘
    │
    ▼
Output: HAL Executable (in .vmfb)
```

## Target Backend Implementation

### Class Structure

```cpp
class TenstorrentTargetBackend final
    : public IREE::HAL::TargetBackend {
public:
  std::string name() const override { 
    return "tenstorrent"; 
  }

  void getDefaultExecutableTargets(
      MLIRContext* context,
      StringRef deviceID,
      DictionaryAttr deviceConfigAttr,
      SmallVectorImpl<IREE::HAL::ExecutableTargetAttr>& executableTargetAttrs
  ) const override;

  void buildTranslationPassPipeline(
      IREE::HAL::ExecutableTargetAttr targetAttr,
      OpPassManager& passManager
  ) const override;

  LogicalResult serializeExecutable(
      const SerializationOptions& options,
      IREE::HAL::ExecutableVariantOp variantOp,
      OpBuilder& executableBuilder
  ) override;
};
```

### Pass Pipeline

```cpp
void TenstorrentTargetBackend::buildTranslationPassPipeline(
    IREE::HAL::ExecutableTargetAttr targetAttr,
    OpPassManager& passManager) const {
  
  // Standard IREE lowering passes
  IREE::HAL::buildHALTransformPassPipeline(passManager, targetAttr);
  
  // Tenstorrent-specific passes
  passManager.addPass(createTTConvertLinalgToLoopsPass());
  passManager.addPass(createTTTileAndDistributePass());
  passManager.addPass(createTTLowerToKernelPass());
}
```

## Code Generation

### Template-Based Approach

For the PoC, we use template-based code generation.

**Critical Design Note:** Decoupling Data Movement from Compute is critical. Without this separation, pipeline parallelism is impossible, and hardware deadlocks may occur due to the multi-processor nature of Tensix cores (NCRISC/BRISC for data, TRISC for compute).

Therefore, the compiler generates separate source files for data movement and computation for each dispatch:

```cpp
struct KernelSource {
  std::string readerSource;  // Runs on RISC-V NCRISC/BRISC
  std::string writerSource;  // Runs on RISC-V NCRISC/BRISC
  std::string computeSource; // Runs on RISC-V TRISC
};

KernelSource generateKernels(Operation* dispatchOp) {
  KernelSource source;

  // 1. Data Movement Kernel (Reader)
  source.readerSource = R"cpp(
    void kernel_main() {
      uint32_t src_addr  = get_arg_val<uint32_t>(0);
      uint32_t num_tiles = get_arg_val<uint32_t>(2);
      
      // DRAM -> L1 (Circular Buffer)
      cb_reserve_back(cb_in0, num_tiles);
      uint32_t l1_write_addr = get_write_ptr(cb_in0);
      noc_async_read_tile(0, src_addr, l1_write_addr);
      noc_async_read_barrier();
      cb_push_back(cb_in0, num_tiles);
    }
  )cpp";

  // 2. Data Movement Kernel (Writer)
  source.writerSource = R"cpp(
    void kernel_main() {
      uint32_t dst_addr  = get_arg_val<uint32_t>(1);
      uint32_t num_tiles = get_arg_val<uint32_t>(2);
      
      // L1 (Circular Buffer) -> DRAM
      cb_wait_front(cb_out0, num_tiles);
      uint32_t l1_read_addr = get_read_ptr(cb_out0);
      noc_async_write_tile(0, dst_addr, l1_read_addr);
      noc_async_write_barrier();
      cb_pop_front(cb_out0, num_tiles);
    }
  )cpp";

  // 3. Compute Kernel
  source.computeSource = R"cpp(
    #include "compute_kernel_api/common.h"
    #include "compute_kernel_api/eltwise_binary.h"
    
    void MAIN {
      uint32_t num_tiles = get_arg_val<uint32_t>(0);
      
      // Wait for input tiles
      cb_wait_front(cb_in0, num_tiles);
      cb_reserve_back(cb_out0, num_tiles);
      
      // Perform Operation (using Tile API)
      ${OPERATION_CODE}
      
      // Push output tiles
      cb_push_back(cb_out0, num_tiles);
      cb_pop_front(cb_in0, num_tiles);
    }
  )cpp";

  // Substitute operation-specific code into compute kernel
  std::string opCode = generateOperationCode(dispatchOp);
  replace(source.computeSource, "${OPERATION_CODE}", opCode);
  
  return source;
}
```

### Supported Operations (PoC)

| MLIR Operation | TT-Metal Implementation |
|----------------|-------------------------|
| `arith.addf` | Element-wise add |
| `arith.mulf` | Element-wise multiply |
| `linalg.matmul` | `ttnn::matmul` or custom kernel |

### Operation Code Generation

```cpp
std::string generateOperationCode(Operation* op) {
  if (auto addOp = dyn_cast<arith::AddFOp>(op)) {
    return R"cpp(
      // Element-wise addition
      for (uint32_t i = 0; i < TILE_SIZE; i++) {
        output[i] = input_a[i] + input_b[i];
      }
    )cpp";
  }
  
  if (auto matmulOp = dyn_cast<linalg::MatmulOp>(op)) {
    return R"cpp(
      // Matrix multiplication using TTNN
      auto result = ttnn::matmul(input_a, input_b);
      ttnn::memcpy(output, result);
    )cpp";
  }
  
  return "// Unsupported operation";
}
```

## FlatBuffer Serialization

### Executable Format

```cpp
LogicalResult TenstorrentTargetBackend::serializeExecutable(
    const SerializationOptions& options,
    IREE::HAL::ExecutableVariantOp variantOp,
    OpBuilder& executableBuilder) {
  
  // Collect all dispatch functions
  SmallVector<std::string> kernelSources;
  for (auto exportOp : variantOp.getExportOps()) {
    std::string kernel = generateKernel(exportOp);
    kernelSources.push_back(kernel);
  }
  
  // Build FlatBuffer
  FlatBufferBuilder builder;
  
  auto kernelsOffset = builder.CreateVectorOfStrings(kernelSources);
  
  auto executableOffset = CreateTenstorrentExecutable(
      builder,
      kernelsOffset,
      /* entry_point_count= */ kernelSources.size());
  
  builder.Finish(executableOffset);
  
  // Embed in IREE executable
  auto dataAttr = DenseIntElementsAttr::get(
      VectorType::get({builder.GetSize()}, IntegerType::get(context, 8)),
      builder.GetBufferPointer());
  
  executableBuilder.create<IREE::HAL::ExecutableConstantBlockOp>(
      variantOp.getLoc(),
      "tenstorrent_executable",
      dataAttr);
  
  return success();
}
```

## Core Grid Mapping

### Workgroup to Core Mapping

```cpp
// Map IREE workgroups to Tensix core grid
CoreRange calculateCoreRange(
    ArrayRef<int64_t> workgroupCount,
    int64_t problemSize) {
  
  // P100A has 120 cores
  constexpr int MAX_GRID_X = 12;
  constexpr int MAX_GRID_Y = 10;
  
  // Simple heuristic: use square grid
  int coresNeeded = std::min(120, (int)workgroupCount[0]);
  int gridSize = (int)std::sqrt(coresNeeded);
  
  return CoreRange({0, 0}, {gridSize - 1, gridSize - 1});
}
```

## Future: tt-mlir Integration

For production, we can integrate tt-mlir's optimized lowering:

```cpp
void buildTranslationPassPipeline(...) {
  // IREE standard passes
  IREE::HAL::buildHALTransformPassPipeline(passManager);
  
  // Convert to tt-mlir dialects
  passManager.addPass(createConvertLinalgToTTIRPass());
  
  // Apply tt-mlir optimizations
  tt::ttir::buildTTIRPipeline(passManager);
  tt::ttnn::addTTNNPasses(passManager);
  
  // Generate final kernel code
  passManager.addPass(createTTNNToKernelPass());
}
```

## Testing

### Lit Tests

```mlir
// test/simple_add.mlir
// RUN: iree-compile --iree-hal-target-backends=tenstorrent %s | FileCheck %s

func.func @add(%a: tensor<32x32xf32>, %b: tensor<32x32xf32>) -> tensor<32x32xf32> {
  %c = arith.addf %a, %b : tensor<32x32xf32>
  return %c : tensor<32x32xf32>
}

// CHECK: hal.executable.variant public @tenstorrent
// CHECK: tenstorrent_executable
```

### Unit Tests

```cpp
TEST(TenstorrentBackend, GeneratesKernelForAdd) {
  // Create test operation
  auto addOp = createTestAddOp();
  
  // Generate kernel
  std::string kernel = generateKernel(addOp);
  
  // Verify kernel contains expected code
  EXPECT_THAT(kernel, HasSubstr("arith.addf"));
}
```

## References

- [IREE Target Backend Guide](https://iree.dev/developers/design-docs/target-backends/)
- [IREE LLVM-CPU Backend](https://github.com/iree-org/iree/tree/main/compiler/plugins/target/LLVMCPU)
- [TT-Metal Programming Guide](https://docs.tenstorrent.com/)
