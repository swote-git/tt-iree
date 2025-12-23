# tt-iree

IREE Backend for Tenstorrent AI Accelerators

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

**Early Development (PoC Phase)**

- [ ] Compiler backend registration
- [ ] HAL driver implementation
- [ ] Mock mode execution
- [ ] Basic operation support (elementwise add)
- [ ] TTNN integration
- [ ] Hardware execution

## Version Compatibility

| Dependency | Version | Notes |
|------------|---------|-------|
| IREE | v3.9.0 | Compiler & runtime infrastructure |
| tt-metal | v0.65.0 | Tenstorrent SDK (TTNN, TT-Metalium) |

These versions have been tested together. See [VERSIONS.md](VERSIONS.md) for details.

## Project Structure

```
tt-iree/
├── compiler/              # IREE compiler plugin
│   └── plugins/target/tenstorrent/
├── runtime/               # IREE HAL driver
│   └── src/iree/hal/drivers/tenstorrent/
├── third_party/
│   ├── iree/              # IREE v3.9.0 (submodule)
│   └── tt-metal/          # tt-metal v0.65.0 (submodule)
├── docs/                  # Documentation
├── test/                  # Tests
└── examples/              # Example programs
```

## Building

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

This project implements an out-of-tree IREE backend for Tenstorrent hardware, consisting of two main components:

### Compiler Backend

Extends IREE's compiler to generate code for Tenstorrent's tile-based architecture:

- Converts IREE's HAL dialect to Tenstorrent-specific representation
- Handles 32x32 tile layout transformation
- Maps workloads to 8x8 Tensix core grid
- Generates TT-Metal kernel code

### Runtime Driver

Implements IREE's Hardware Abstraction Layer (HAL) interface:

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

## Roadmap

| Phase | Goal | Status |
|-------|------|--------|
| **PoC** | Single operation, mock execution | WIP |
| **MVP** | MNIST inference on P100A | Planned |
| **Alpha** | ResNet-18, multi-core dispatch | Planned |
| **Beta** | LLM inference (GPT-2 scale) | Planned |
| **v1.0** | Production release | Planned |

## Related Projects

- [IREE](https://github.com/iree-org/iree) - ML compiler infrastructure
- [tt-metal](https://github.com/tenstorrent/tt-metal) - Tenstorrent low-level SDK
- [tt-mlir](https://github.com/tenstorrent/tt-mlir) - Tenstorrent MLIR compiler
- [iree-amd-aie](https://github.com/nod-ai/iree-amd-aie) - Reference for out-of-tree IREE backend

## License

Apache 2.0 with LLVM Exceptions. See [LICENSE](LICENSE).

## Acknowledgments

- Developed as part of the **Tenstorrent Korea Open Source Developer Program**.