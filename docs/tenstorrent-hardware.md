# Tenstorrent Hardware Overview

Overview of Tenstorrent AI accelerator architecture relevant to tt-iree.

## Target Hardware

### P100A (Wormhole)

The primary target for tt-iree is the P100A card based on the Wormhole architecture.

| Specification | Value | Note |
|---------------|-------|------|
| Architecture | Wormhole | |
| Tensix Cores | **120** | Full 10x12 grid active |
| DRAM | **28GB GDDR6** | 448 GB/s bandwidth |
| **SRAM** | **180 MB** | **1.5MB per core × 120 cores** |
| TDP (TBP) | **300W** | Requires external 12V-2x6 power |
| Interface | **PCIe Gen 5.0 x16** | |
| Memory Bandwidth | 448 GB/s | 16 GT/s |

## Tensix Core Architecture

Each Tensix core contains 5 RISC-V processors:

```
┌─────────────────────────────────────────────────────┐
│                    Tensix Core                      │
│  ┌─────────────────────────────────────────────┐    │
│  │              RISC-V Processors              │    │
│  │  ┌───────┐ ┌───────┐ ┌───────┐              │    │
│  │  │ BRISC │ │NCRISC │ │TRISC0 │ ...          │    │
│  │  │(data) │ │(NoC)  │ │(math) │              │    │
│  │  └───────┘ └───────┘ └───────┘              │    │
│  └─────────────────────────────────────────────┘    │
│  ┌─────────────────────────────────────────────┐    │
│  │           Matrix Engine (FPU)               │    │
│  │         32x32 tile operations               │    │
│  └─────────────────────────────────────────────┘    │
│  ┌─────────────────────────────────────────────┐    │
│  │           L1 SRAM (~1.5MB)                  │    │
│  │         Local storage per core              │    │
│  └─────────────────────────────────────────────┘    │
└─────────────────────────────────────────────────────┘
```

### Processor Roles

| Processor | Role |
|-----------|------|
| BRISC | Data movement, buffer management |
| NCRISC | Network-on-Chip (NoC) control |
| TRISC0-2 | Math/compute operations |

## Memory Hierarchy

```text
Host Memory (System RAM)
        │
        │ PCIe Gen 5.0 x16
        ▼
┌─────────────────────────────────────┐
│      Device DRAM (28GB GDDR6)       │
│      Bandwidth: 448 GB/s            │
└─────────────────────────────────────┘
        │
        │ NoC (Network-on-Chip)
        │ Dual loops (NoC0/NoC1)
        ▼
┌─────────────────────────────────────┐
│    L1 SRAM (distributed, 180MB)     │
│    1.5MB per core × 120 cores       │
│    Bandwidth: Ultra-high aggregate  │
└─────────────────────────────────────┘
        │
        ▼
┌─────────────────────────────────────┐
│         Registers (per core)        │
└─────────────────────────────────────┘
```

### Key Difference from GPUs

| Aspect | GPU | Tenstorrent |
|--------|-----|-------------|
| Memory Model | Single large VRAM | Distributed L1 + DRAM |
| Data Layout | Row-major (flexible) | 32x32 tiles (required) |
| Parallelism | SIMT threads | Core grid + SIMD |
| Memory Management | Hardware managed | Explicit software control |

## Tile-Based Architecture

### 32x32 Tile Format

All matrix operations use 32x32 tiles:

```
Row-Major (Host):              Tiled (Device):
┌─────────────────────┐        ┌────────┬────────┬─...
│ a00 a01 a02 ... a63 │        │ Tile   │ Tile   │
│ a64 a65 a66 ...     │   →    │ (0,0)  │ (0,1)  │
│ ...                 │        │ 32x32  │ 32x32  │
└─────────────────────┘        ├────────┼────────┼─...
                               │ Tile   │ Tile   │
                               │ (1,0)  │ (1,1)  │
                               └────────┴────────┴─...
```

### Tile Layout in Memory

```cpp
// Row-major: elements[row][col]
float row_major[64][64];

// Tiled: tiles[tile_row][tile_col][intra_row][intra_col]
float tiled[2][2][32][32];

// Conversion
void to_tile_layout(float* src, float* dst, int rows, int cols) {
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

## Data Movement

### NoC (Network-on-Chip)

The NoC enables communication between cores and DRAM:

```
     DRAM Bank 0   DRAM Bank 1   ...
          │             │
          ▼             ▼
     ┌────────────────────────────┐
     │        NoC Router          │
     └────────────────────────────┘
          │     │     │     │
          ▼     ▼     ▼     ▼
     ┌────┐ ┌────┐ ┌────┐ ┌────┐
     │Core│ │Core│ │Core│ │Core│
     │0,0 │ │0,1 │ │0,2 │ │0,3 │
     └────┘ └────┘ └────┘ └────┘
          │     │     │     │
          ▼     ▼     ▼     ▼
     ┌────┐ ┌────┐ ┌────┐ ┌────┐
     │Core│ │Core│ │Core│ │Core│
     │1,0 │ │1,1 │ │1,2 │ │1,3 │
     └────┘ └────┘ └────┘ └────┘
          ...
```

### Circular Buffers

Data movement uses circular buffers:

```cpp
// Producer side (e.g., reading from DRAM)
cb_reserve_back(cb_id, num_tiles);
uint32_t write_ptr = get_write_ptr(cb_id);
// ... write data to write_ptr ...
cb_push_back(cb_id, num_tiles);

// Consumer side (e.g., compute kernel)
cb_wait_front(cb_id, num_tiles);
uint32_t read_ptr = get_read_ptr(cb_id);
// ... read data from read_ptr ...
cb_pop_front(cb_id, num_tiles);
```

## Memory Layout Strategies

### Interleaved

Tiles distributed across all DRAM banks:

```
Tile 0 → DRAM Bank 0
Tile 1 → DRAM Bank 1
Tile 2 → DRAM Bank 2
...
```

Good for: General purpose, balanced access

### Height Sharded

Each row of tiles assigned to one core:

```
Core (0,0): Tiles [0,0], [0,1], [0,2], ...
Core (1,0): Tiles [1,0], [1,1], [1,2], ...
Core (2,0): Tiles [2,0], [2,1], [2,2], ...
```

Good for: Row-wise operations

### Block Sharded

2D blocks of tiles assigned to cores:

```
Core (0,0): Tiles [0:2, 0:2]
Core (0,1): Tiles [0:2, 2:4]
Core (1,0): Tiles [2:4, 0:2]
```

Good for: Matrix operations

## Implications for tt-iree

### Compiler Must Handle

1. **Tile transformation:** Convert row-major to 32x32 tiles
2. **Core grid mapping:** Assign work to **120 cores (10x12 grid)**
3. **Memory placement:** Decide L1 vs DRAM allocation (fit within **1.5MB L1**)
4. **Sharding strategy:** Choose interleaved/height/block

### Runtime Must Handle

1. **Buffer allocation:** Manage DRAM and L1 memory
2. **Layout conversion:** Transform data on host↔device transfer
3. **Synchronization:** Coordinate multi-core execution
4. **Resource tracking:** Track core and memory utilization

## TT-Metal Software Stack

```
┌─────────────────────────────────────────────┐
│                 Applications                 │
└─────────────────────────────────────────────┘
                     │
                     ▼
┌─────────────────────────────────────────────┐
│   TTNN (High-level PyTorch-like API)        │
│   - ttnn.matmul(), ttnn.conv2d(), etc.      │
└─────────────────────────────────────────────┘
                     │
                     ▼
┌─────────────────────────────────────────────┐
│   TT-Metal (Low-level API)                  │
│   - Device, Buffer, Program, CommandQueue   │
└─────────────────────────────────────────────┘
                     │
                     ▼
┌─────────────────────────────────────────────┐
│   Device Driver (kernel module)             │
└─────────────────────────────────────────────┘
                     │
                     ▼
┌─────────────────────────────────────────────┐
│   Hardware (P100A)                          │
└─────────────────────────────────────────────┘
```

## References

- [Tenstorrent Documentation](https://docs.tenstorrent.com/)
- [TT-Metal GitHub](https://github.com/tenstorrent/tt-metal)
- [Wormhole Architecture](https://tenstorrent.com/technology)
