# tt-iree

IREE plugin for Tenstorrent AI Accelerators

[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](LICENSE)

## Overview

tt-iree enables execution of ML models compiled with [IREE](https://github.com/iree-org/iree) on [Tenstorrent](https://tenstorrent.com/) AI accelerators (P100A/Blackhole architecture).

```python
# Goal: Seamless deployment to Tenstorrent
import iree.compiler as compiler
import iree.runtime as runtime

# Compile for Tenstorrent
compiled = compiler.compile_str(mlir_code, target_backends=["tenstorrent"])

# Run on Tenstorrent hardware
config = runtime.Config("tenstorrent")
result = runtime.invoke(compiled, inputs)
```

## Status

**Alpha Milestone Complete (Functional Bring-Up on P100A/Blackhole)**

Implemented and validated:
- Compiler backend registration, TTEX serialization, and generated AOT kernels
- Custom IREE HAL driver and execution on real P100A hardware
- Elementwise add and static/dynamic tile-aligned BF16 matmul smoke paths
- Multicore tiled dispatch with runtime shape arguments
- [MNIST MLP inference](reports/2026-10-20/mnist-mlp.md): static batch 1, BF16,
  100 held-out test images, 99/100 correct and 100/100 Top-1 agreement with PyTorch
- [ResNet50 E2E inference](reports/2026-10-15/resnet50-dataset-evaluation-ko.md):
  static batch 1, BF16, 112 ImageNet validation images and 112/112 Top-1 agreement
  with PyTorch BF16; all numerical dispatches execute device-native
- [Tensix matrix execution](reports/2026-10-15/resnet50-bf16-tensix-ko.md) for
  ResNet50 contractions, with device RISC-V execution for the remaining operations

Completion refers to functional bring-up of these verified workloads, not
general model support, bitwise numerical equivalence, or production readiness.
Performance optimization, broader model coverage, and dynamic model batching
remain in progress. TTNN is used as a comparison baseline, not as a model fallback.

## Version Compatibility

| Dependency | Version | Notes |
|------------|---------|-------|
| IREE | v3.11.0 | Compiler & runtime infrastructure |
| tt-metal | v0.75.0 | Tenstorrent SDK (TTNN, TT-Metalium) |


## Project Structure

```
tt-iree/
├── compiler/              # IREE compiler plugin
│   └── plugins/target/tenstorrent/
├── runtime/               # IREE HAL driver
│   └── src/iree/hal/drivers/tenstorrent/
├── third_party/
│   ├── iree/              # IREE v3.11.0 (submodule)
│   └── tt-metal/          # tt-metal v0.75.0 (submodule)
├── docs/                  # Documentation
├── reports/               # Dated AI-assisted reports and investigations
├── test/                  # Tests
└── examples/              # Example programs
```

## Building(WIP: Not working now)

### Prerequisites

- CMake 3.21+
- Ninja
- Clang/LLVM 15+
- Python 3.9+

```bash
# Ubuntu 22.04
sudo apt-get install cmake ninja-build clang lld python3-pip
```

### Clone and Setup

```bash
git clone https://github.com/user/tt-iree.git
cd tt-iree

# Setup submodules with pinned versions
./scripts/setup_submodules.sh
```

> **Note**: Submodule initialization takes ~10-20 minutes due to IREE's large dependency tree.

### Build (Mock Mode) **not working in current state**

```bash
cmake -G Ninja -B build \
  -DCMAKE_BUILD_TYPE=RelWithDebInfo \
  -DCMAKE_C_COMPILER=clang \
  -DCMAKE_CXX_COMPILER=clang++ \
  -DTT_IREE_ENABLE_MOCK=ON

cmake --build build
```

### Build (With Hardware)

```bash
# Set up TT-Metal environment
source third_party/tt-metal/build/python_env/bin/activate
export TT_METAL_HOME=$(pwd)/third_party/tt-metal

cmake -G Ninja -B build \
  -DCMAKE_BUILD_TYPE=RelWithDebInfo \
  -DTT_IREE_ENABLE_MOCK=OFF \
  -DTT_IREE_ENABLE_TTNN=ON

cmake --build build
```

## Architecture

Out-of-tree IREE plugin for Tenstorrent hardware, consisting of two main components:

### Compiler Backend

Extends IREE's compiler to generate code for Tenstorrent's tile-based architecture:

- Converts IREE's HAL dialect to Tenstorrent-specific representation
- Handles 32x32 tile layout transformation
- Maps output tiles to compiler-selected multi-core grids
- Generates TT-Metal kernel code

### Runtime Driver

Implements IREE's HAL(Hardware Abstraction Layer) interface:

| Component | Description |
|-----------|-------------|
| `tt_driver` | Driver registration and device enumeration |
| `tt_device` | Device lifecycle and capability management |
| `tt_allocator` | Buffer allocation (DRAM/L1) |
| `tt_buffer` | Memory management with tile layout |
| `tt_command_buffer` | Command recording and dispatch |
| `tt_executable` | Kernel loading and execution |

### Data Flow

```
PyTorch/JAX/TensorFlow
        ↓
   StableHLO/TOSA
        ↓
  IREE Compiler (Flow → Stream → HAL)
        ↓
  Tenstorrent Backend (HAL → TT-Metal)
        ↓
   .vmfb (VM FlatBuffer)
        ↓
  IREE Runtime + HAL Driver
        ↓
  TT-Metal Runtime → P100A Hardware
```

For detailed architecture documentation, see [docs/](docs/).
AI-assisted implementation snapshots and experiment reports are indexed in
[reports/](reports/README.md), organized by date rather than maintained as
project documentation.

## Roadmap

| Phase | Goal | Status |
|-------|------|--------|
| **PoC** | Single-operation compiler/HAL bring-up | Complete |
| **MVP** | MNIST MLP inference on P100A, static B1 BF16 | [Complete: 100-image hardware verification](reports/2026-10-20/mnist-mlp.md) |
| **Alpha** | ResNet50 E2E inference and multi-core device-native dispatch, static B1 BF16 | [Complete: 112-image hardware verification](reports/2026-10-15/resnet50-dataset-evaluation-ko.md) |
| **Beta** | LLM inference (GPT-2 scale) | Planned |
| **v1.0** | Production release | Planned |

Alpha was validated with ResNet50 rather than the originally planned ResNet-18;
ResNet-18 has not been separately validated. The sample counts above describe
the tested scope, not full-dataset accuracy certification.

## Related Projects

- [IREE](https://github.com/iree-org/iree) - ML compiler infrastructure
- [tt-metal](https://github.com/tenstorrent/tt-metal) - Tenstorrent low-level SDK
- [tt-mlir](https://github.com/tenstorrent/tt-mlir) - Tenstorrent MLIR compiler
- [iree-amd-aie](https://github.com/nod-ai/iree-amd-aie) - Reference for out-of-tree IREE backend

## License

Apache 2.0 with LLVM Exceptions. See [LICENSE](LICENSE).

## Acknowledgments

- Developed as part of the **Tenstorrent Korea Open Source Developer Program**.
