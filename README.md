# tt-iree

IREE Backend for Tenstorrent AI Accelerators

[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](LICENSE)

## Overview

tt-iree enables execution of ML models compiled with [IREE](https://github.com/iree-org/iree) on [Tenstorrent](https://tenstorrent.com/) AI accelerators (P100A, Wormhole architecture).

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

## Project Structure

```
tt-iree/
├── compiler/              # IREE compiler plugin
│   └── plugins/target/tenstorrent/
├── runtime/               # IREE HAL driver
│   └── src/iree/hal/drivers/tenstorrent/
├── third_party/
│   ├── iree/              # IREE (submodule)
│   └── tt-metal/          # TT-Metal SDK (submodule)
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

### Clone Repository

```bash
git clone https://github.com/user/tt-iree.git
cd tt-iree
git submodule update --init --recursive
```

### Build (Mock Mode - No Hardware)

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
# Set up TT-Metal environment first
source third_party/tt-metal/build/python_env/bin/activate
export TT_METAL_HOME=$(pwd)/third_party/tt-metal

cmake -G Ninja -B build \
  -DCMAKE_BUILD_TYPE=RelWithDebInfo \
  -DTT_IREE_ENABLE_MOCK=OFF \
  -DTT_IREE_ENABLE_TTNN=ON

cmake --build build
```

### Verify Installation

```bash
# Check driver registration
./build/tools/iree-run-module --list_drivers
# Should show: tenstorrent
```

## Quick Start

### Compile and Run (Mock Mode)

```bash
# Compile MLIR to Tenstorrent target
./build/tools/iree-compile \
  --iree-hal-target-backends=tenstorrent \
  examples/simple_add.mlir \
  -o /tmp/simple_add.vmfb

# Run on mock device
./build/tools/iree-run-module \
  --device=tenstorrent \
  --module=/tmp/simple_add.vmfb \
  --function=main \
  --input="4xf32=1,2,3,4"
```

## Development

### Running Tests

```bash
cd build
ctest -L tt-iree --output-on-failure
```

### Code Formatting

```bash
./scripts/format.sh
```

## Architecture

See [docs/architecture.md](docs/architecture.md) for detailed design documentation.

### Key Components

| Component | Description |
|-----------|-------------|
| `TenstorrentTarget` | IREE compiler backend plugin |
| `tt_driver` | HAL driver implementation |
| `tt_device` | Device management |
| `tt_buffer` | Buffer allocation |
| `ttnn_bridge` | TTNN library integration |

## Roadmap

- **PoC (Current)**: Single operation, mock execution
- **MVP**: MNIST inference on P100A
- **Alpha**: ResNet-18, multi-core dispatch
- **Beta**: LLM inference (GPT-2 scale)
- **v1.0**: Production release

## Related Projects

- [IREE](https://github.com/iree-org/iree) - ML compiler infrastructure
- [tt-metal](https://github.com/tenstorrent/tt-metal) - Tenstorrent low-level SDK
- [tt-mlir](https://github.com/tenstorrent/tt-mlir) - Tenstorrent MLIR compiler
- [iree-amd-aie](https://github.com/nod-ai/iree-amd-aie) - Reference for out-of-tree IREE backend

## License

Apache 2.0 with LLVM Exceptions. See [LICENSE](LICENSE).

## Acknowledgments

- IREE Team for the excellent compiler infrastructure
- Tenstorrent for open-sourcing tt-metal and tt-mlir
