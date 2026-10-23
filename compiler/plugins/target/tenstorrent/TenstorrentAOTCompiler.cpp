// Copyright 2026 The tt-iree Authors
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "TenstorrentAOTCompiler.h"
#include "TenstorrentNativeKernels.h"
#include "iree/hal/drivers/tenstorrent/tt_fill_kernel.h"
#include "iree/schema/tt_native_dispatch.h"

#include <algorithm>
#include <cerrno>
#include <atomic>
#include <chrono>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <map>
#include <memory>
#include <mutex>
#include <set>
#include <stdexcept>
#include <string_view>
#include <type_traits>
#include <variant>
#include <spawn.h>
#include <sys/wait.h>

extern char **environ;

#include <hostdevcommon/kernel_structs.h>
#include <tt-metalium/core_coord.hpp>
#include <tt-metalium/data_types.hpp>
#include <tt-metalium/experimental/mock_device/mock_device.hpp>
#include <tt-metalium/experimental/offline_kernel_compile.hpp>
#include <tt-metalium/kernel_types.hpp>
#include <tt-metalium/tensor_accessor_args.hpp>

#include "impl/kernels/kernel.hpp"
#include "impl/program/kernel_compile_utils.hpp"
#include "jit_build/build.hpp"
#include "jit_build/build_env_manager.hpp"
#include "jit_build/genfiles.hpp"
#include "jit_build/jit_build_options.hpp"
#include "jit_build/jit_device_config.hpp"
#include "llrt/hal.hpp"
#include "llrt/rtoptions.hpp"

namespace mlir::iree_compiler::IREE::HAL {
namespace {

namespace fs = std::filesystem;
using tt::tt_metal::ComputeConfig;
using tt::tt_metal::CoreCoord;
using tt::tt_metal::CoreRange;
using tt::tt_metal::CoreRangeSet;
using tt::tt_metal::DataMovementConfig;
using tt::tt_metal::DataMovementProcessor;
using tt::tt_metal::HalProcessorClassType;
using tt::tt_metal::JitBuildOptions;
using tt::tt_metal::Kernel;
using tt::tt_metal::KernelSource;
using tt::tt_metal::NOC;
using tt::tt_metal::TensorAccessorArgs;
using CBCompileConfig =
    tt::tt_metal::experimental::OfflineKernelCompileParams::CBCompileConfig;

std::mutex &aotCompileMutex() {
  static std::mutex mutex;
  return mutex;
}

ComputeConfig makeMatmulComputeConfig() {
  ComputeConfig config;
  config.math_fidelity = tt::tt_metal::MathFidelity::HiFi4;
  config.fp32_dest_acc_en = true;
  config.math_approx_mode = false;
  return config;
}

tt::ARCH toTTMetalArch(TenstorrentAOTTargetArch target_arch) {
  return target_arch == TenstorrentAOTTargetArch::Blackhole
             ? tt::ARCH::BLACKHOLE
             : tt::ARCH::WORMHOLE_B0;
}

constexpr char kAddReaderSource[] = R"ttkernel(
#include <cstdint>
#include "api/dataflow/noc.h"
#include "api/dataflow/circular_buffer.h"
#include "api/tensor/noc_traits.h"

void kernel_main() {
  uint32_t in0_addr = get_arg_val<uint32_t>(0);
  uint32_t in1_addr = get_arg_val<uint32_t>(1);
  uint32_t n_tiles = get_arg_val<uint32_t>(2);

  constexpr uint32_t cb_in0 = tt::CBIndex::c_0;
  constexpr uint32_t cb_in1 = tt::CBIndex::c_1;
  uint32_t tile_size_bytes = get_tile_size(cb_in0);
  constexpr auto in0_args = TensorAccessorArgs<0>();
  const auto in0 = TensorAccessor(in0_args, in0_addr);
  constexpr auto in1_args =
      TensorAccessorArgs<in0_args.next_compile_time_args_offset()>();
  const auto in1 = TensorAccessor(in1_args, in1_addr);
  Noc noc;
  CircularBuffer cb_in0_buf(cb_in0);
  CircularBuffer cb_in1_buf(cb_in1);

  for (uint32_t i = 0; i < n_tiles; ++i) {
    cb_in0_buf.reserve_back(1);
    cb_in1_buf.reserve_back(1);
    noc.async_read(in0, cb_in0_buf, tile_size_bytes, {.page_id = i}, {});
    noc.async_read(in1, cb_in1_buf, tile_size_bytes, {.page_id = i}, {});
    noc.async_read_barrier();
    cb_in0_buf.push_back(1);
    cb_in1_buf.push_back(1);
  }
}
)ttkernel";

constexpr char kAddWriterSource[] = R"ttkernel(
#include <cstdint>
#include "api/dataflow/noc.h"
#include "api/dataflow/circular_buffer.h"
#include "api/tensor/noc_traits.h"

void kernel_main() {
  uint32_t out_addr = get_arg_val<uint32_t>(0);
  uint32_t n_tiles = get_arg_val<uint32_t>(1);

  constexpr uint32_t cb_out = tt::CBIndex::c_16;
  uint32_t tile_size_bytes = get_tile_size(cb_out);
  constexpr auto out_args = TensorAccessorArgs<0>();
  const auto out = TensorAccessor(out_args, out_addr);
  Noc noc;
  CircularBuffer cb_out_buf(cb_out);

  for (uint32_t i = 0; i < n_tiles; ++i) {
    cb_out_buf.wait_front(1);
    noc.async_write(cb_out_buf, out, tile_size_bytes, {}, {.page_id = i});
    noc.async_write_barrier();
    cb_out_buf.pop_front(1);
  }
}
)ttkernel";

constexpr char kAddComputeSource[] = R"ttkernel(
#include <cstdint>
#include "api/compute/common.h"
#include "api/compute/tile_move_copy.h"
#include "api/compute/eltwise_binary.h"
#include "api/compute/compute_kernel_api.h"

void kernel_main() {
  uint32_t n_tiles = get_arg_val<uint32_t>(0);
  constexpr auto cb_in0 = tt::CBIndex::c_0;
  constexpr auto cb_in1 = tt::CBIndex::c_1;
  constexpr auto cb_out = tt::CBIndex::c_16;
  constexpr uint32_t dst_reg = 0;

  binary_op_init_common(cb_in0, cb_in1, cb_out);
  add_tiles_init(cb_in0, cb_in1);
  for (uint32_t i = 0; i < n_tiles; ++i) {
    cb_wait_front(cb_in0, 1);
    cb_wait_front(cb_in1, 1);
    tile_regs_acquire();
    add_tiles(cb_in0, cb_in1, 0, 0, dst_reg);
    tile_regs_commit();
    tile_regs_wait();
    cb_reserve_back(cb_out, 1);
    pack_tile(dst_reg, cb_out);
    cb_push_back(cb_out, 1);
    cb_pop_front(cb_in0, 1);
    cb_pop_front(cb_in1, 1);
    tile_regs_release();
  }
}
)ttkernel";

constexpr char kPaddedCopySource[] = R"ttkernel(
#include <cstdint>
#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/circular_buffer.h"

void kernel_main() {
  uint32_t source_addr = get_arg_val<uint32_t>(0);
  uint32_t destination_addr = get_arg_val<uint32_t>(1);
  uint32_t source_base_offset = get_arg_val<uint32_t>(2);
  uint32_t destination_base_offset = get_arg_val<uint32_t>(3);
  uint32_t channels = get_arg_val<uint32_t>(4);
  uint32_t height = get_arg_val<uint32_t>(5);
  uint32_t width = get_arg_val<uint32_t>(6);
  uint32_t source_storage_h = get_arg_val<uint32_t>(7);
  uint32_t source_storage_w = get_arg_val<uint32_t>(8);
  uint32_t source_offset_c = get_arg_val<uint32_t>(9);
  uint32_t source_offset_h = get_arg_val<uint32_t>(10);
  uint32_t source_offset_w = get_arg_val<uint32_t>(11);
  uint32_t source_stride_c = get_arg_val<uint32_t>(12);
  uint32_t source_stride_h = get_arg_val<uint32_t>(13);
  uint32_t destination_storage_h = get_arg_val<uint32_t>(14);
  uint32_t destination_storage_w = get_arg_val<uint32_t>(15);
  uint32_t destination_offset_c = get_arg_val<uint32_t>(16);
  uint32_t destination_offset_h = get_arg_val<uint32_t>(17);
  uint32_t destination_offset_w = get_arg_val<uint32_t>(18);
  uint32_t destination_stride_c = get_arg_val<uint32_t>(19);
  uint32_t destination_stride_h = get_arg_val<uint32_t>(20);

  constexpr uint32_t page_size = 2048;
  constexpr uint32_t read_alignment = 64;
  constexpr uint32_t write_alignment = 16;
  constexpr uint32_t write_scratch_offset = 4096;
  constexpr auto source_args = TensorAccessorArgs<0>();
  constexpr auto destination_args =
      TensorAccessorArgs<source_args.next_compile_time_args_offset()>();
  const auto source = TensorAccessor(source_args, source_addr);
  const auto destination =
      TensorAccessor(destination_args, destination_addr);

  CircularBuffer scratch(tt::CBIndex::c_0);
  scratch.reserve_back(1);
  const uint32_t scratch_base = scratch.get_write_ptr();

  for (uint32_t c = 0; c < channels; ++c) {
    for (uint32_t h = 0; h < height; ++h) {
      uint32_t source_element =
          ((source_offset_c + c * source_stride_c) * source_storage_h +
           source_offset_h + h * source_stride_h) *
              source_storage_w +
          source_offset_w;
      uint32_t destination_element =
          ((destination_offset_c + c * destination_stride_c) *
               destination_storage_h +
           destination_offset_h + h * destination_stride_h) *
              destination_storage_w +
          destination_offset_w;
      uint32_t source_byte = source_base_offset + source_element * 4;
      uint32_t destination_byte =
          destination_base_offset + destination_element * 4;
      uint32_t remaining = width * 4;

      while (remaining != 0) {
        uint32_t source_page_offset = source_byte % page_size;
        uint32_t destination_page_offset = destination_byte % page_size;
        uint32_t chunk = remaining;
        uint32_t source_page_remaining = page_size - source_page_offset;
        uint32_t destination_page_remaining =
            page_size - destination_page_offset;
        if (chunk > source_page_remaining) chunk = source_page_remaining;
        if (chunk > destination_page_remaining)
          chunk = destination_page_remaining;

        uint32_t read_l1 =
            scratch_base + (source_page_offset % read_alignment);
        noc_async_read(
            source.get_noc_addr(source_byte / page_size, source_page_offset),
            read_l1, chunk);
        noc_async_read_barrier();

        uint32_t write_l1 = write_scratch_offset + scratch_base +
                            (destination_page_offset % write_alignment);
        volatile tt_l1_ptr uint32_t* source_words =
            reinterpret_cast<volatile tt_l1_ptr uint32_t*>(read_l1);
        volatile tt_l1_ptr uint32_t* destination_words =
            reinterpret_cast<volatile tt_l1_ptr uint32_t*>(write_l1);
        for (uint32_t byte = 0; byte < chunk; byte += 4) {
          destination_words[byte / 4] = source_words[byte / 4];
        }

        noc_async_write(
            write_l1,
            destination.get_noc_addr(destination_byte / page_size,
                                     destination_page_offset),
            chunk);
        noc_async_write_barrier();
        source_byte += chunk;
        destination_byte += chunk;
        remaining -= chunk;
      }
    }
  }
}
)ttkernel";

constexpr char kMatmulReaderSource[] = R"ttkernel(
#include <stdint.h>
#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/circular_buffer.h"
#include "api/dataflow/endpoints.h"

void kernel_main() {
  uint32_t src0_addr = get_arg_val<uint32_t>(0);
  uint32_t src0_bank_id = get_arg_val<uint32_t>(1);
  uint32_t src1_addr = get_arg_val<uint32_t>(2);
  uint32_t src1_bank_id = get_arg_val<uint32_t>(3);
  uint32_t num_blocks = get_arg_val<uint32_t>(4);
  uint32_t in0_block_tile_cnt = get_arg_val<uint32_t>(5);
  uint32_t in1_block_tile_cnt = get_arg_val<uint32_t>(6);
  uint32_t in0_block_size_bytes = get_arg_val<uint32_t>(7);
  uint32_t in1_block_size_bytes = get_arg_val<uint32_t>(8);

  CircularBuffer cb_in0(0);
  CircularBuffer cb_in1(1);
  Noc noc;
  for (uint32_t i = 0; i < num_blocks; ++i) {
    cb_in0.reserve_back(in0_block_tile_cnt);
    cb_in1.reserve_back(in1_block_tile_cnt);
    noc.async_read(AllocatorBank<AllocatorBankType::DRAM>{}, cb_in0,
                   in0_block_size_bytes,
                   {.bank_id = src0_bank_id, .addr = src0_addr}, {});
    noc.async_read(AllocatorBank<AllocatorBankType::DRAM>{}, cb_in1,
                   in1_block_size_bytes,
                   {.bank_id = src1_bank_id, .addr = src1_addr}, {});
    noc.async_read_barrier();
    cb_in0.push_back(in0_block_tile_cnt);
    cb_in1.push_back(in1_block_tile_cnt);
    src0_addr += in0_block_size_bytes;
    src1_addr += in1_block_size_bytes;
  }
}
)ttkernel";

constexpr char kMatmulWriterSource[] = R"ttkernel(
#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/circular_buffer.h"
#include "api/dataflow/endpoints.h"

void kernel_main() {
  uint32_t dst_addr = get_arg_val<uint32_t>(0);
  uint32_t dst_bank_id = get_arg_val<uint32_t>(1);
  uint32_t inner_r = get_arg_val<uint32_t>(2);
  uint32_t inner_c = get_arg_val<uint32_t>(3);
  uint32_t num_sub_blocks_m = get_arg_val<uint32_t>(4);
  uint32_t num_sub_blocks_n = get_arg_val<uint32_t>(5);
  uint32_t stride_r = get_arg_val<uint32_t>(6);
  uint32_t stride_subblock_r = get_arg_val<uint32_t>(7);
  uint32_t stride_subblock_c = get_arg_val<uint32_t>(8);

  CircularBuffer cb(16);
  Noc noc;
  uint32_t tile_size_bytes = get_tile_size(16);
  uint32_t block_row_addr = dst_addr;
  for (uint32_t sb_m = 0; sb_m < num_sub_blocks_m; ++sb_m) {
    uint32_t block_addr = block_row_addr;
    for (uint32_t sb_n = 0; sb_n < num_sub_blocks_n; ++sb_n) {
      uint32_t row_addr = block_addr;
      for (uint32_t r = 0; r < inner_r; ++r) {
        uint32_t tile_addr = row_addr;
        for (uint32_t c = 0; c < inner_c; ++c) {
          cb.wait_front(1);
          noc.async_write(cb, AllocatorBank<AllocatorBankType::DRAM>{},
                          tile_size_bytes, {},
                          {.bank_id = dst_bank_id, .addr = tile_addr});
          noc.async_write_barrier();
          cb.pop_front(1);
          tile_addr += tile_size_bytes;
        }
        row_addr += stride_r;
      }
      block_addr += stride_subblock_c;
    }
    block_row_addr += stride_subblock_r;
  }
}
)ttkernel";

constexpr char kMatmulComputeSource[] = R"ttkernel(
#include <cstdint>
#include "api/compute/tile_move_copy.h"
#include "api/compute/matmul.h"
#include "api/compute/compute_kernel_hw_startup.h"
#include "api/dataflow/circular_buffer.h"

void kernel_main() {
  uint32_t in0_block_w = get_arg_val<uint32_t>(0);
  uint32_t in0_num_subblocks = get_arg_val<uint32_t>(1);
  uint32_t in0_block_num_tiles = get_arg_val<uint32_t>(2);
  uint32_t in0_subblock_num_tiles = get_arg_val<uint32_t>(3);
  uint32_t in1_num_subblocks = get_arg_val<uint32_t>(4);
  uint32_t in1_block_num_tiles = get_arg_val<uint32_t>(5);
  uint32_t in1_per_core_w = get_arg_val<uint32_t>(6);
  uint32_t num_blocks = get_arg_val<uint32_t>(7);
  uint32_t out_subblock_h = get_arg_val<uint32_t>(8);
  uint32_t out_subblock_w = get_arg_val<uint32_t>(9);
  uint32_t out_subblock_num_tiles = get_arg_val<uint32_t>(10);

  CircularBuffer cb0(tt::CBIndex::c_0);
  CircularBuffer cb1(tt::CBIndex::c_1);
  CircularBuffer cb24(tt::CBIndex::c_24);
  CircularBuffer cb16(tt::CBIndex::c_16);
  compute_kernel_hw_startup<SrcOrder::Reverse>(tt::CBIndex::c_0,
                                                tt::CBIndex::c_1,
                                                tt::CBIndex::c_16);
  matmul_init(tt::CBIndex::c_0, tt::CBIndex::c_1);
  bool enable_reload = false;
  for (uint32_t block = 0; block < num_blocks; ++block) {
    bool last_out = block == num_blocks - 1;
    cb0.wait_front(in0_block_num_tiles);
    cb1.wait_front(in1_block_num_tiles);
    int in0_subblock_offset = 0;
    for (uint32_t in0_subblock = 0; in0_subblock < in0_num_subblocks;
         ++in0_subblock) {
      int in1_subblock_offset = 0;
      for (uint32_t in1_subblock = 0; in1_subblock < in1_num_subblocks;
           ++in1_subblock) {
        tile_regs_acquire();
        if (enable_reload) {
          copy_tile_to_dst_init_short(tt::CBIndex::c_24);
          cb24.wait_front(out_subblock_num_tiles);
          for (uint32_t i = 0; i < out_subblock_num_tiles; ++i)
            copy_tile(tt::CBIndex::c_24, i, i);
          cb24.pop_front(out_subblock_num_tiles);
          matmul_init(tt::CBIndex::c_0, tt::CBIndex::c_1);
        }
        int dst_index = 0;
        int in0_row_offset = 0;
        for (uint32_t h = 0; h < out_subblock_h; ++h) {
          for (uint32_t w = 0; w < out_subblock_w; ++w) {
            int in1_inner_offset = 0;
            for (uint32_t inner = 0; inner < in0_block_w; ++inner) {
              int in0_index = in0_subblock_offset + in0_row_offset + inner;
              int in1_index = in1_subblock_offset + in1_inner_offset + w;
              matmul_tiles(tt::CBIndex::c_0, tt::CBIndex::c_1, in0_index,
                           in1_index, dst_index);
              in1_inner_offset += in1_per_core_w;
            }
            ++dst_index;
          }
          in0_row_offset += in0_block_w;
        }
        tile_regs_commit();
        tile_regs_wait();
        CircularBuffer &out_cb = last_out ? cb16 : cb24;
        uint32_t out_cb_id = last_out ? 16 : 24;
        out_cb.reserve_back(out_subblock_num_tiles);
        for (uint32_t i = 0; i < out_subblock_num_tiles; ++i)
          pack_tile(i, out_cb_id);
        out_cb.push_back(out_subblock_num_tiles);
        tile_regs_release();
        in1_subblock_offset += out_subblock_w;
      }
      in0_subblock_offset += in0_subblock_num_tiles;
    }
    enable_reload = true;
    cb0.pop_front(in0_block_num_tiles);
    cb1.pop_front(in1_block_num_tiles);
  }
}
)ttkernel";

constexpr char kF32MatmulReaderSource[] = R"ttkernel(
#include <cstdint>
#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/circular_buffer.h"

void kernel_main() {
  uint32_t lhs_addr = get_arg_val<uint32_t>(0);
  uint32_t rhs_addr = get_arg_val<uint32_t>(1);
  uint32_t lhs_base_offset = get_arg_val<uint32_t>(2);
  uint32_t rhs_base_offset = get_arg_val<uint32_t>(3);
  uint32_t lhs_layout = get_arg_val<uint32_t>(4);
  uint32_t lhs_storage_1 = get_arg_val<uint32_t>(5);
  uint32_t lhs_storage_2 = get_arg_val<uint32_t>(6);
  uint32_t lhs_logical_width = get_arg_val<uint32_t>(7);
  uint32_t lhs_offset_0 = get_arg_val<uint32_t>(8);
  uint32_t lhs_offset_1 = get_arg_val<uint32_t>(9);
  uint32_t lhs_offset_2 = get_arg_val<uint32_t>(10);
  uint32_t lhs_stride_0 = get_arg_val<uint32_t>(11);
  uint32_t lhs_stride_1 = get_arg_val<uint32_t>(12);
  uint32_t lhs_stride_2 = get_arg_val<uint32_t>(13);
  uint32_t rhs_storage_1 = get_arg_val<uint32_t>(14);
  uint32_t rhs_offset_0 = get_arg_val<uint32_t>(15);
  uint32_t rhs_offset_1 = get_arg_val<uint32_t>(16);
  uint32_t rhs_stride_0 = get_arg_val<uint32_t>(17);
  uint32_t rhs_stride_1 = get_arg_val<uint32_t>(18);
  uint32_t core_m_tile_start = get_arg_val<uint32_t>(19);
  uint32_t core_n_tile_start = get_arg_val<uint32_t>(20);
  uint32_t per_core_m_tiles = get_arg_val<uint32_t>(21);
  uint32_t per_core_n_tiles = get_arg_val<uint32_t>(22);
  uint32_t k_tiles = get_arg_val<uint32_t>(23);
  uint32_t logical_m_size = get_arg_val<uint32_t>(24);

  constexpr uint32_t page_size = 2048;
  constexpr uint32_t read_alignment = 64;
  constexpr uint32_t layout_chw = 2;
  constexpr auto lhs_args = TensorAccessorArgs<0>();
  constexpr auto rhs_args =
      TensorAccessorArgs<lhs_args.next_compile_time_args_offset()>();
  const auto lhs = TensorAccessor(lhs_args, lhs_addr);
  const auto rhs = TensorAccessor(rhs_args, rhs_addr);

  CircularBuffer cb_lhs(tt::CBIndex::c_0);
  CircularBuffer cb_rhs(tt::CBIndex::c_1);
  CircularBuffer scratch(tt::CBIndex::c_2);
  scratch.reserve_back(1);
  uint32_t scratch_base = scratch.get_write_ptr();

  for (uint32_t block = 0; block < k_tiles; ++block) {
    cb_lhs.reserve_back(per_core_m_tiles);
    uint32_t lhs_tiles_base = cb_lhs.get_write_ptr();
    for (uint32_t tile_m = 0; tile_m < per_core_m_tiles; ++tile_m) {
      volatile tt_l1_ptr uint16_t* tile =
          reinterpret_cast<volatile tt_l1_ptr uint16_t*>(
              lhs_tiles_base + tile_m * 2048);
      for (uint32_t row = 0; row < 32; ++row) {
        uint32_t logical_m = (core_m_tile_start + tile_m) * 32 + row;
        for (uint32_t col = 0; col < 32; ++col) {
          uint32_t logical_k = block * 32 + col;
          uint32_t tile_index = (row / 16) * 512 + (col / 16) * 256 +
                                (row % 16) * 16 + col % 16;
          if (logical_m >= logical_m_size) {
            tile[tile_index] = 0;
            continue;
          }
          uint32_t element = 0;
          if (lhs_layout == layout_chw) {
            uint32_t h = logical_m / lhs_logical_width;
            uint32_t w = logical_m - h * lhs_logical_width;
            element =
                ((lhs_offset_0 + logical_k * lhs_stride_0) * lhs_storage_1 +
                 lhs_offset_1 + h * lhs_stride_1) *
                    lhs_storage_2 +
                lhs_offset_2 + w * lhs_stride_2;
          } else {
            element =
                (lhs_offset_0 + logical_k * lhs_stride_0) * lhs_storage_1 +
                lhs_offset_1 + logical_m * lhs_stride_1;
          }
          uint32_t source_byte = lhs_base_offset + element * 4;
          uint32_t page_offset = source_byte % page_size;
          uint32_t read_l1 = scratch_base + page_offset % read_alignment;
          noc_async_read(
              lhs.get_noc_addr(source_byte / page_size, page_offset),
              read_l1, 4);
          noc_async_read_barrier();
          uint32_t bits =
              *reinterpret_cast<volatile tt_l1_ptr uint32_t*>(read_l1);
          uint32_t rounded = bits + 0x7fffu + ((bits >> 16) & 1u);
          tile[tile_index] = static_cast<uint16_t>(rounded >> 16);
        }
      }
    }
    cb_lhs.push_back(per_core_m_tiles);

    cb_rhs.reserve_back(per_core_n_tiles);
    uint32_t rhs_tiles_base = cb_rhs.get_write_ptr();
    for (uint32_t tile_n = 0; tile_n < per_core_n_tiles; ++tile_n) {
      volatile tt_l1_ptr uint16_t* tile =
          reinterpret_cast<volatile tt_l1_ptr uint16_t*>(
              rhs_tiles_base + tile_n * 2048);
      for (uint32_t row = 0; row < 32; ++row) {
        uint32_t logical_k = block * 32 + row;
        for (uint32_t col = 0; col < 32; ++col) {
          uint32_t logical_n = (core_n_tile_start + tile_n) * 32 + col;
          uint32_t element =
              (rhs_offset_0 + logical_n * rhs_stride_0) * rhs_storage_1 +
              rhs_offset_1 + logical_k * rhs_stride_1;
          uint32_t source_byte = rhs_base_offset + element * 4;
          uint32_t page_offset = source_byte % page_size;
          uint32_t read_l1 = scratch_base + page_offset % read_alignment;
          noc_async_read(
              rhs.get_noc_addr(source_byte / page_size, page_offset),
              read_l1, 4);
          noc_async_read_barrier();
          uint32_t bits =
              *reinterpret_cast<volatile tt_l1_ptr uint32_t*>(read_l1);
          uint32_t rounded = bits + 0x7fffu + ((bits >> 16) & 1u);
          uint32_t tile_index = (row / 16) * 512 + (col / 16) * 256 +
                                (row % 16) * 16 + col % 16;
          tile[tile_index] = static_cast<uint16_t>(rounded >> 16);
        }
      }
    }
    cb_rhs.push_back(per_core_n_tiles);
  }
}
)ttkernel";

constexpr char kF32MatmulWriterSource[] = R"ttkernel(
#include <cstdint>
#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/circular_buffer.h"

void kernel_main() {
  uint32_t output_addr = get_arg_val<uint32_t>(0);
  uint32_t output_base_offset = get_arg_val<uint32_t>(1);
  uint32_t output_layout = get_arg_val<uint32_t>(2);
  uint32_t output_storage_1 = get_arg_val<uint32_t>(3);
  uint32_t output_storage_2 = get_arg_val<uint32_t>(4);
  uint32_t output_logical_width = get_arg_val<uint32_t>(5);
  uint32_t output_offset_0 = get_arg_val<uint32_t>(6);
  uint32_t output_offset_1 = get_arg_val<uint32_t>(7);
  uint32_t output_offset_2 = get_arg_val<uint32_t>(8);
  uint32_t output_stride_0 = get_arg_val<uint32_t>(9);
  uint32_t output_stride_1 = get_arg_val<uint32_t>(10);
  uint32_t output_stride_2 = get_arg_val<uint32_t>(11);
  uint32_t core_m_tile_start = get_arg_val<uint32_t>(12);
  uint32_t core_n_tile_start = get_arg_val<uint32_t>(13);
  uint32_t per_core_m_tiles = get_arg_val<uint32_t>(14);
  uint32_t per_core_n_tiles = get_arg_val<uint32_t>(15);
  uint32_t logical_m_size = get_arg_val<uint32_t>(16);
  uint32_t epilogue_mode = get_arg_val<uint32_t>(17);
  uint32_t mean_addr = get_arg_val<uint32_t>(18);
  uint32_t mean_base_offset = get_arg_val<uint32_t>(19);
  uint32_t mean_offset_0 = get_arg_val<uint32_t>(20);
  uint32_t mean_stride_0 = get_arg_val<uint32_t>(21);
  uint32_t channel_scale_addr = get_arg_val<uint32_t>(22);
  uint32_t channel_scale_base_offset = get_arg_val<uint32_t>(23);
  uint32_t channel_scale_offset_0 = get_arg_val<uint32_t>(24);
  uint32_t channel_scale_stride_0 = get_arg_val<uint32_t>(25);
  uint32_t channel_bias_addr = get_arg_val<uint32_t>(26);
  uint32_t channel_bias_base_offset = get_arg_val<uint32_t>(27);
  uint32_t channel_bias_offset_0 = get_arg_val<uint32_t>(28);
  uint32_t channel_bias_stride_0 = get_arg_val<uint32_t>(29);
  uint32_t common_scale_bits = get_arg_val<uint32_t>(30);

  constexpr uint32_t page_size = 2048;
  constexpr uint32_t read_alignment = 64;
  constexpr uint32_t write_alignment = 16;
  constexpr uint32_t layout_chw = 2;
  constexpr uint32_t runtime_per_channel_affine_relu = 1;
  constexpr auto output_args = TensorAccessorArgs<0>();
  constexpr auto mean_args =
      TensorAccessorArgs<output_args.next_compile_time_args_offset()>();
  constexpr auto channel_scale_args =
      TensorAccessorArgs<mean_args.next_compile_time_args_offset()>();
  constexpr auto channel_bias_args =
      TensorAccessorArgs<channel_scale_args.next_compile_time_args_offset()>();
  const auto output = TensorAccessor(output_args, output_addr);
  const auto mean = TensorAccessor(mean_args, mean_addr);
  const auto channel_scale =
      TensorAccessor(channel_scale_args, channel_scale_addr);
  const auto channel_bias =
      TensorAccessor(channel_bias_args, channel_bias_addr);

  CircularBuffer cb_output(tt::CBIndex::c_16);
  CircularBuffer scratch(tt::CBIndex::c_3);
  scratch.reserve_back(1);
  uint32_t scratch_base = scratch.get_write_ptr();

  union FloatBits {
    uint32_t bits;
    float value;
  };
  FloatBits common_scale;
  common_scale.bits = common_scale_bits;

  for (uint32_t tile_m = 0; tile_m < per_core_m_tiles; ++tile_m) {
    for (uint32_t tile_n = 0; tile_n < per_core_n_tiles; ++tile_n) {
      cb_output.wait_front(1);
      volatile tt_l1_ptr uint16_t* tile =
          reinterpret_cast<volatile tt_l1_ptr uint16_t*>(
              cb_output.get_read_ptr());
      for (uint32_t row = 0; row < 32; ++row) {
        uint32_t logical_m = (core_m_tile_start + tile_m) * 32 + row;
        if (logical_m >= logical_m_size)
          continue;
        for (uint32_t col = 0; col < 32; ++col) {
          uint32_t logical_n = (core_n_tile_start + tile_n) * 32 + col;
          uint32_t tile_index = (row / 16) * 512 + (col / 16) * 256 +
                                (row % 16) * 16 + col % 16;
          FloatBits value;
          value.bits = static_cast<uint32_t>(tile[tile_index]) << 16;

          if (epilogue_mode == runtime_per_channel_affine_relu) {
            uint32_t mean_byte =
                mean_base_offset +
                (mean_offset_0 + logical_n * mean_stride_0) * 4;
            uint32_t mean_page_offset = mean_byte % page_size;
            uint32_t mean_l1 =
                scratch_base + mean_page_offset % read_alignment;
            noc_async_read(
                mean.get_noc_addr(mean_byte / page_size, mean_page_offset),
                mean_l1, 4);
            noc_async_read_barrier();
            FloatBits mean_value;
            mean_value.bits =
                *reinterpret_cast<volatile tt_l1_ptr uint32_t*>(mean_l1);

            uint32_t scale_byte =
                channel_scale_base_offset +
                (channel_scale_offset_0 + logical_n * channel_scale_stride_0) *
                    4;
            uint32_t scale_page_offset = scale_byte % page_size;
            uint32_t scale_l1 =
                scratch_base + scale_page_offset % read_alignment;
            noc_async_read(channel_scale.get_noc_addr(scale_byte / page_size,
                                                       scale_page_offset),
                           scale_l1, 4);
            noc_async_read_barrier();
            FloatBits scale_value;
            scale_value.bits =
                *reinterpret_cast<volatile tt_l1_ptr uint32_t*>(scale_l1);

            uint32_t bias_byte =
                channel_bias_base_offset +
                (channel_bias_offset_0 + logical_n * channel_bias_stride_0) *
                    4;
            uint32_t bias_page_offset = bias_byte % page_size;
            uint32_t bias_l1 =
                scratch_base + bias_page_offset % read_alignment;
            noc_async_read(channel_bias.get_noc_addr(bias_byte / page_size,
                                                      bias_page_offset),
                           bias_l1, 4);
            noc_async_read_barrier();
            FloatBits bias_value;
            bias_value.bits =
                *reinterpret_cast<volatile tt_l1_ptr uint32_t*>(bias_l1);

            float scale = common_scale.value * scale_value.value;
            value.value = value.value * scale +
                          (bias_value.value - mean_value.value * scale);
            if (value.value < 0.0f)
              value.value = 0.0f;
          }

          uint32_t element = 0;
          if (output_layout == layout_chw) {
            uint32_t h = logical_m / output_logical_width;
            uint32_t w = logical_m - h * output_logical_width;
            element = ((output_offset_0 + logical_n * output_stride_0) *
                           output_storage_1 +
                       output_offset_1 + h * output_stride_1) *
                          output_storage_2 +
                      output_offset_2 + w * output_stride_2;
          } else {
            element = (output_offset_0 + logical_n * output_stride_0) *
                          output_storage_1 +
                      output_offset_1 + logical_m * output_stride_1;
          }
          uint32_t destination_byte = output_base_offset + element * 4;
          uint32_t page_offset = destination_byte % page_size;
          uint32_t write_l1 =
              scratch_base + page_offset % write_alignment;
          *reinterpret_cast<volatile tt_l1_ptr uint32_t*>(write_l1) =
              value.bits;
          noc_async_write(
              write_l1,
              output.get_noc_addr(destination_byte / page_size, page_offset),
              4);
          noc_async_write_barrier();
        }
      }
      cb_output.pop_front(1);
    }
  }
}
)ttkernel";

class ScopedTempDirectory {
public:
  ScopedTempDirectory() {
    static std::atomic<uint64_t> next_id{0};
    path_ = fs::temp_directory_path() /
            ("tt-iree-aot-" +
             std::to_string(
                 std::chrono::steady_clock::now().time_since_epoch().count()) +
             "-" + std::to_string(next_id.fetch_add(1)));
    fs::remove_all(path_);
    fs::create_directories(path_);
  }
  ~ScopedTempDirectory() {
    std::error_code ec;
    fs::remove_all(path_, ec);
  }
  const fs::path &path() const { return path_; }

private:
  fs::path path_;
};

class ScopedMockMetalContext {
public:
  explicit ScopedMockMetalContext(TenstorrentAOTTargetArch target_arch) {
    tt::tt_metal::experimental::configure_mock_mode(toTTMetalArch(target_arch),
                                                    1);
  }
  ~ScopedMockMetalContext() { tt::tt_metal::experimental::disable_mock_mode(); }
};

class ScopedFallbackEnvironmentVariable {
public:
  ScopedFallbackEnvironmentVariable(const char *name,
                                    const std::string &fallback_value)
      : name_(name) {
    if (std::getenv(name))
      return;
    if (::setenv(name, fallback_value.c_str(), 0) != 0) {
      throw std::runtime_error("failed to set environment variable " + name_);
    }
    changed_ = true;
  }
  ~ScopedFallbackEnvironmentVariable() {
    if (changed_) {
      ::unsetenv(name_.c_str());
    }
  }

private:
  std::string name_;
  bool changed_ = false;
};

std::shared_ptr<Kernel>
makeKernel(const std::string &source,
           const std::variant<DataMovementConfig, ComputeConfig> &config) {
  KernelSource kernel_source(source, KernelSource::SOURCE_CODE);
  CoreRangeSet core_ranges(
      std::set<CoreRange>{CoreRange(CoreCoord{0, 0}, CoreCoord{0, 0})});
  return std::visit(
      [&](const auto &typed_config) -> std::shared_ptr<Kernel> {
        using ConfigT = std::decay_t<decltype(typed_config)>;
        if constexpr (std::is_same_v<ConfigT, DataMovementConfig>) {
          return std::make_shared<tt::tt_metal::DataMovementKernel>(
              kernel_source, core_ranges, typed_config);
        } else {
          return std::make_shared<tt::tt_metal::ComputeKernel>(
              kernel_source, core_ranges, typed_config);
        }
      },
      config);
}

void compileSourceOffline(
    const std::string &source,
    const std::variant<DataMovementConfig, ComputeConfig> &config,
    const std::vector<CBCompileConfig> &cb_configs,
    TenstorrentAOTTargetArch target_arch, const fs::path &output_dir) {
  std::shared_ptr<Kernel> kernel = makeKernel(source, config);
  tt::llrt::RunTimeOptions runtime_options;
  size_t matching_config_count = 0;
  tt::tt_metal::enumerate_offline_compile_device_configs(
      runtime_options, [&](const tt::tt_metal::JitDeviceConfig &device_config) {
        tt::ARCH expected_arch = toTTMetalArch(target_arch);
        if (device_config.arch != expected_arch)
          return;
        ++matching_config_count;
        tt::tt_metal::BuildEnvManager build_env_manager(*device_config.hal);
        build_env_manager.add_build_env(0, device_config, runtime_options);
        const tt::tt_metal::DeviceBuildEnv &device_build_env =
            build_env_manager.get_device_build_env(0);
        JitBuildOptions build_options(device_build_env.build_env);
        kernel->set_build_options(build_options);
        for (const CBCompileConfig &cb_config : cb_configs) {
          build_options.set_cb_data_fmt_and_tile(
              static_cast<tt::CBIndex>(cb_config.cb_index),
              cb_config.data_format, cb_config.tile);
        }

        size_t kernel_hash = tt::tt_metal::detail::KernelCompileHash(
            kernel, build_options, device_build_env.build_key());
        std::string kernel_suffix =
            kernel->name() + "/" + std::to_string(kernel_hash) + "/";
        kernel->set_full_name(kernel_suffix);
        build_options.set_name(kernel_suffix);
        tt::tt_metal::jit_build_genfiles_descriptors(device_build_env.build_env,
                                                     build_options);

        uint32_t core_type =
            device_config.hal->get_programmable_core_type_index(
                kernel->get_kernel_programmable_core_type());
        uint32_t processor_class =
            static_cast<std::underlying_type_t<HalProcessorClassType>>(
                kernel->get_kernel_processor_class());
        if (kernel->get_kernel_processor_class() ==
            HalProcessorClassType::COMPUTE) {
          tt::tt_metal::jit_build_genfiles_triscs_src(
              device_build_env.build_env, *kernel, kernel->kernel_source());
          tt::tt_metal::jit_build_subset(
              build_env_manager.get_kernel_build_states(0, core_type,
                                                        processor_class),
              kernel.get());
        } else {
          tt::tt_metal::jit_build_genfiles_kernel_include(
              device_build_env.build_env, *kernel, kernel->kernel_source());
          uint32_t processor_id = kernel->get_kernel_processor_type(0);
          tt::tt_metal::jit_build(
              build_env_manager.get_kernel_build_state(
                  0, core_type, processor_class, processor_id),
              kernel.get());
        }

        std::string source_root =
            device_build_env.build_env.get_out_kernel_root_path();
        for (uint8_t i = 0; i < kernel->expected_num_binaries(); ++i) {
          uint32_t processor_id = kernel->get_kernel_processor_type(i);
          std::string source_elf = build_env_manager.get_kernel_binary_path(
              0, core_type, processor_class, processor_id, source_root,
              kernel->get_full_kernel_name());
          std::string output_elf = build_env_manager.get_kernel_binary_path(
              0, core_type, processor_class, processor_id, output_dir.string(),
              kernel->get_full_kernel_name());
          fs::create_directories(fs::path(output_elf).parent_path());
          const char *keep_debug = std::getenv("TT_IREE_AOT_KEEP_DEBUG");
          if (keep_debug && std::string_view(keep_debug) == "1") {
            fs::copy_file(source_elf, output_elf,
                          fs::copy_options::overwrite_existing);
          } else {
            std::string error;
            if (!stripTenstorrentAOTArtifactDebugInfo(source_elf, output_elf, error))
              throw std::runtime_error(error);
          }
        }
      });
  if (matching_config_count == 0) {
    throw std::runtime_error(
        "tt-metal has no offline compile config for the requested target");
  }
}

std::vector<TenstorrentAOTArtifact> readArtifacts(const fs::path &root) {
  std::vector<TenstorrentAOTArtifact> artifacts;
  for (const fs::directory_entry &entry :
       fs::recursive_directory_iterator(root)) {
    if (!entry.is_regular_file())
      continue;
    std::ifstream input(entry.path(), std::ios::binary);
    if (!input) {
      throw std::runtime_error("failed to read AOT artifact " +
                               entry.path().string());
    }
    TenstorrentAOTArtifact artifact;
    artifact.path = fs::relative(entry.path(), root).generic_string();
    artifact.data.assign(std::istreambuf_iterator<char>(input),
                         std::istreambuf_iterator<char>());
    if (artifact.data.empty()) {
      throw std::runtime_error("offline compiler emitted an empty artifact " +
                               artifact.path);
    }
    artifacts.push_back(std::move(artifact));
  }
  std::sort(
      artifacts.begin(), artifacts.end(),
      [](const auto &lhs, const auto &rhs) { return lhs.path < rhs.path; });
  return artifacts;
}

} // namespace

bool stripTenstorrentAOTArtifactDebugInfo(const std::string &input_path,
                                        const std::string &output_path,
                                        std::string &error_message) {
  error_message.clear();
  try {
    std::error_code ec;
    if (fs::equivalent(input_path, output_path, ec) ||
        fs::absolute(input_path).lexically_normal() ==
            fs::absolute(output_path).lexically_normal()) {
      throw std::runtime_error("AOT objcopy input and output must be distinct");
    }
    tt::llrt::RunTimeOptions options;
    fs::path tool = fs::path(options.get_root_dir()) /
                    "runtime/sfpi/compiler/bin/riscv-tt-elf-objcopy";
    if (!fs::exists(tool))
      tool = "/opt/tenstorrent/sfpi/compiler/bin/riscv-tt-elf-objcopy";
    std::string tool_path = tool.string();
    char *args[] = {tool_path.data(), const_cast<char *>("--strip-debug"),
                    const_cast<char *>("--"),
                    const_cast<char *>(input_path.c_str()),
                    const_cast<char *>(output_path.c_str()), nullptr};
    pid_t pid;
    int result = posix_spawn(&pid, tool_path.c_str(), nullptr, nullptr, args, environ);
    if (result != 0)
      throw std::runtime_error("failed to start AOT objcopy: " +
                               std::string(std::strerror(result)));
    int status = 0;
    while (waitpid(pid, &status, 0) < 0) {
      if (errno != EINTR)
        throw std::runtime_error("failed to wait for AOT objcopy");
    }
    if (!WIFEXITED(status) || WEXITSTATUS(status) != 0)
      throw std::runtime_error("AOT objcopy --strip-debug failed for " + input_path);
    return true;
  } catch (const std::exception &e) {
    error_message = e.what();
    return false;
  }
}

static bool compileAOTKernelBundle(
    const char *reader_source, const DataMovementConfig &reader_config,
    const char *writer_source, const DataMovementConfig &writer_config,
    const char *compute_source, const ComputeConfig &compute_config,
    const std::vector<CBCompileConfig> &cb_configs,
    TenstorrentAOTTargetArch target_arch,
    TenstorrentAOTKernelBundle &out_bundle, std::string &error_message) {
  // TT-Metal's mock context and cache configuration are process-global.
  std::lock_guard<std::mutex> compile_lock(aotCompileMutex());

  out_bundle = {};
  error_message.clear();

  try {
#ifdef TT_IREE_TT_METAL_HOME
    tt::llrt::RunTimeOptions::set_root_dir(TT_IREE_TT_METAL_HOME);
#endif
    ScopedTempDirectory temp_dir;
    fs::path cache_dir = fs::temp_directory_path() / "tt-iree-metal-aot-cache";
    fs::path artifact_dir = temp_dir.path() / "artifacts";
    fs::create_directories(cache_dir);
    fs::create_directories(artifact_dir);
    ScopedFallbackEnvironmentVariable cache_env("TT_METAL_CACHE",
                                                cache_dir.string());
    ScopedMockMetalContext mock_context(target_arch);

    compileSourceOffline(reader_source, reader_config, cb_configs, target_arch,
                         artifact_dir);
    compileSourceOffline(writer_source, writer_config, cb_configs, target_arch,
                         artifact_dir);
    compileSourceOffline(compute_source, compute_config, cb_configs,
                         target_arch, artifact_dir);

    out_bundle.targetArch = target_arch;
    out_bundle.kernels = {
        {static_cast<uint8_t>(TenstorrentAOTKernelRole::Reader), reader_source},
        {static_cast<uint8_t>(TenstorrentAOTKernelRole::Writer), writer_source},
        {static_cast<uint8_t>(TenstorrentAOTKernelRole::Compute),
         compute_source},
    };
    out_bundle.artifacts = readArtifacts(artifact_dir);
    if (out_bundle.artifacts.empty()) {
      throw std::runtime_error("offline compiler emitted no artifacts");
    }
    return true;
  } catch (const std::exception &e) {
    error_message = e.what();
    return false;
  } catch (...) {
    error_message = "unknown tt-metal offline compiler failure";
    return false;
  }
}

static bool compileDataMovementAOTKernel(
    const char *source, const DataMovementConfig &config,
    const std::vector<CBCompileConfig> &cb_configs,
    TenstorrentAOTTargetArch target_arch,
    TenstorrentAOTKernelBundle &out_bundle, std::string &error_message) {
  std::lock_guard<std::mutex> compile_lock(aotCompileMutex());

  out_bundle = {};
  error_message.clear();
  try {
#ifdef TT_IREE_TT_METAL_HOME
    tt::llrt::RunTimeOptions::set_root_dir(TT_IREE_TT_METAL_HOME);
#endif
    ScopedTempDirectory temp_dir;
    fs::path cache_dir = fs::temp_directory_path() / "tt-iree-metal-aot-cache";
    fs::path artifact_dir = temp_dir.path() / "artifacts";
    fs::create_directories(cache_dir);
    fs::create_directories(artifact_dir);
    ScopedFallbackEnvironmentVariable cache_env("TT_METAL_CACHE",
                                                cache_dir.string());
    ScopedMockMetalContext mock_context(target_arch);

    compileSourceOffline(source, config, cb_configs, target_arch, artifact_dir);
    out_bundle.targetArch = target_arch;
    out_bundle.kernels = {
        {static_cast<uint8_t>(TenstorrentAOTKernelRole::Reader), source}};
    out_bundle.artifacts = readArtifacts(artifact_dir);
    if (out_bundle.artifacts.empty()) {
      throw std::runtime_error("offline compiler emitted no artifacts");
    }
    return true;
  } catch (const std::exception &e) {
    error_message = e.what();
    return false;
  } catch (...) {
    error_message = "unknown tt-metal offline compiler failure";
    return false;
  }
}

bool compileFillAOT(TenstorrentAOTTargetArch target_arch,
                    TenstorrentAOTKernelBundle &out_bundle,
                    std::string &error_message) {
  std::vector<uint32_t> args;
  TensorAccessorArgs::create_dram_interleaved().append_to(args);
  args.back() = 2048;
  DataMovementConfig config{
      .processor = DataMovementProcessor::RISCV_0,
      .noc = NOC::RISCV_0_default,
      .compile_args = std::move(args),
  };
  return compileDataMovementAOTKernel(
      iree::hal::tenstorrent::kDeviceFillSource, config,
      {{.cb_index = 0, .data_format = tt::DataFormat::UInt32}}, target_arch,
      out_bundle, error_message);
}

bool compileAddAOT(TenstorrentAOTTargetArch target_arch,
                   TenstorrentAOTKernelBundle &out_bundle,
                   std::string &error_message) {
  constexpr uint32_t kTileSizeBytes = 32 * 32 * 2;
  std::vector<CBCompileConfig> cb_configs = {
      {.cb_index = 0, .data_format = tt::DataFormat::Float16_b},
      {.cb_index = 1, .data_format = tt::DataFormat::Float16_b},
      {.cb_index = 16, .data_format = tt::DataFormat::Float16_b},
  };
  std::vector<uint32_t> reader_args;
  TensorAccessorArgs::create_dram_interleaved().append_to(reader_args);
  reader_args.back() = kTileSizeBytes;
  TensorAccessorArgs::create_dram_interleaved().append_to(reader_args);
  reader_args.back() = kTileSizeBytes;
  std::vector<uint32_t> writer_args;
  TensorAccessorArgs::create_dram_interleaved().append_to(writer_args);
  writer_args.back() = kTileSizeBytes;

  DataMovementConfig reader_config{
      .processor = DataMovementProcessor::RISCV_0,
      .noc = NOC::RISCV_0_default,
      .compile_args = std::move(reader_args),
  };
  DataMovementConfig writer_config{
      .processor = DataMovementProcessor::RISCV_1,
      .noc = NOC::RISCV_1_default,
      .compile_args = std::move(writer_args),
  };
  return compileAOTKernelBundle(kAddReaderSource, reader_config,
                                kAddWriterSource, writer_config,
                                kAddComputeSource, ComputeConfig{}, cb_configs,
                                target_arch, out_bundle, error_message);
}

bool compileMatmulAOT(TenstorrentAOTTargetArch target_arch,
                      TenstorrentAOTKernelBundle &out_bundle,
                      std::string &error_message) {
  std::vector<CBCompileConfig> cb_configs = {
      {.cb_index = 0, .data_format = tt::DataFormat::Float16_b},
      {.cb_index = 1, .data_format = tt::DataFormat::Float16_b},
      {.cb_index = 16, .data_format = tt::DataFormat::Float16_b},
      {.cb_index = 24, .data_format = tt::DataFormat::Float16_b},
  };
  DataMovementConfig reader_config{
      .processor = DataMovementProcessor::RISCV_1,
      .noc = NOC::RISCV_1_default,
  };
  DataMovementConfig writer_config{
      .processor = DataMovementProcessor::RISCV_0,
      .noc = NOC::RISCV_0_default,
  };
  return compileAOTKernelBundle(
      kMatmulReaderSource, reader_config, kMatmulWriterSource, writer_config,
      kMatmulComputeSource, makeMatmulComputeConfig(), cb_configs, target_arch,
      out_bundle, error_message);
}

bool compileF32MatmulAOT(TenstorrentAOTTargetArch target_arch,
                         TenstorrentAOTKernelBundle &out_bundle,
                         std::string &error_message) {
  constexpr uint32_t kHalBufferPageSize = 32 * 32 * sizeof(uint16_t);
  std::vector<CBCompileConfig> cb_configs = {
      {.cb_index = 0, .data_format = tt::DataFormat::Float16_b},
      {.cb_index = 1, .data_format = tt::DataFormat::Float16_b},
      {.cb_index = 2, .data_format = tt::DataFormat::UInt32},
      {.cb_index = 3, .data_format = tt::DataFormat::UInt32},
      {.cb_index = 16, .data_format = tt::DataFormat::Float16_b},
      {.cb_index = 24, .data_format = tt::DataFormat::Float16_b},
  };
  std::vector<uint32_t> reader_args;
  TensorAccessorArgs::create_dram_interleaved().append_to(reader_args);
  reader_args.back() = kHalBufferPageSize;
  TensorAccessorArgs::create_dram_interleaved().append_to(reader_args);
  reader_args.back() = kHalBufferPageSize;
  std::vector<uint32_t> writer_args;
  for (int i = 0; i < 4; ++i) {
    TensorAccessorArgs::create_dram_interleaved().append_to(writer_args);
    writer_args.back() = kHalBufferPageSize;
  }
  DataMovementConfig reader_config{
      .processor = DataMovementProcessor::RISCV_1,
      .noc = NOC::RISCV_1_default,
      .compile_args = std::move(reader_args),
  };
  DataMovementConfig writer_config{
      .processor = DataMovementProcessor::RISCV_0,
      .noc = NOC::RISCV_0_default,
      .compile_args = std::move(writer_args),
  };
  return compileAOTKernelBundle(
      kF32MatmulReaderSource, reader_config, kF32MatmulWriterSource,
      writer_config, kMatmulComputeSource, makeMatmulComputeConfig(),
      cb_configs, target_arch, out_bundle, error_message);
}
bool compilePaddedCopyAOT(TenstorrentAOTTargetArch target_arch,
                          TenstorrentAOTKernelBundle &out_bundle,
                          std::string &error_message) {
  constexpr uint32_t kHalBufferPageSize = 32 * 32 * sizeof(uint16_t);
  std::vector<uint32_t> compile_args;
  TensorAccessorArgs::create_dram_interleaved().append_to(compile_args);
  compile_args.back() = kHalBufferPageSize;
  TensorAccessorArgs::create_dram_interleaved().append_to(compile_args);
  compile_args.back() = kHalBufferPageSize;
  DataMovementConfig config{
      .processor = DataMovementProcessor::RISCV_0,
      .noc = NOC::RISCV_0_default,
      .compile_args = std::move(compile_args),
  };
  std::vector<CBCompileConfig> cb_configs = {
      {.cb_index = 0, .data_format = tt::DataFormat::UInt32},
  };
  return compileDataMovementAOTKernel(kPaddedCopySource, config, cb_configs,
                                      target_arch, out_bundle, error_message);
}

bool compileBF16NativeAOT(const tt_iree_ttex_entry_point_desc_t &entry,
                          const std::string &source,
                          const TenstorrentNativeSources &sources,
                          TenstorrentAOTTargetArch target_arch,
                          TenstorrentAOTKernelBundle &out_bundle,
                          std::string &error_message) {
  std::vector<uint32_t> args;
  for (unsigned i = 0; i < TT_IREE_BF16_NATIVE_TENSOR_COUNT; ++i) {
    TensorAccessorArgs::create_dram_interleaved().append_to(args);
    args.back() = 2048;
  }
  std::vector<CBCompileConfig> cb = {
      {.cb_index = 2, .data_format = tt::DataFormat::UInt32}};
  if (entry.builtin_program == TT_IREE_TTEX_BUILTIN_PROGRAM_BF16_NATIVE_TENSIX) {
    if (sources.reader.empty() || sources.writer.empty() || sources.compute.empty()) {
      error_message = "BF16 Tensix requires all three generated kernel roles";
      return false;
    }
    for (uint8_t i : {0, 1})
      cb.push_back({.cb_index = i, .data_format = tt::DataFormat::Float16_b});
    for (uint8_t i : {3, 8, 9, 17})
      cb.push_back({.cb_index = i, .data_format = tt::DataFormat::UInt32});
    cb.push_back({.cb_index = 16, .data_format = tt::DataFormat::Float32});
    DataMovementConfig reader{.processor = DataMovementProcessor::RISCV_1,
                              .noc = NOC::RISCV_1_default, .compile_args = args};
    DataMovementConfig writer{.processor = DataMovementProcessor::RISCV_0,
                              .noc = NOC::RISCV_0_default, .compile_args = args};
    return compileAOTKernelBundle(sources.reader.c_str(), reader,
                                  sources.writer.c_str(), writer, sources.compute.c_str(),
                                  makeMatmulComputeConfig(), cb, target_arch,
                                  out_bundle, error_message);
  }
  if (tt_iree_ttex_is_native_matrix_kind(entry.bf16_native.operation_kind))
    for (uint8_t i : {0, 1, 16})
      cb.push_back({.cb_index = i, .data_format = tt::DataFormat::UInt32});
  DataMovementConfig config{.processor = DataMovementProcessor::RISCV_0,
                            .noc = NOC::RISCV_0_default,
                            .compile_args = args};
  return compileDataMovementAOTKernel(source.c_str(), config, cb, target_arch,
                                      out_bundle, error_message);
}

bool compileNativeDispatchAOT(const tt_iree_ttex_entry_point_desc_t &entry,
                              TenstorrentAOTTargetArch target_arch,
                              TenstorrentAOTKernelBundle &out_bundle,
                              std::string &error_message, bool use_tensix) {
  try {
    auto sources = generateNativeSources(entry, use_tensix);
    std::vector<uint32_t> args;
    for (unsigned i = 0; i < TT_IREE_NATIVE_TENSOR_COUNT; ++i) {
      TensorAccessorArgs::create_dram_interleaved().append_to(args);
      args.back() = 2048;
    }
    std::vector<CBCompileConfig> cb = {
        {.cb_index = 2, .data_format = tt::DataFormat::UInt32}};
    if (sources.compute.empty()) {
      if (tt_iree_ttex_is_native_matrix_kind(entry.dispatch_kind)) {
        for (uint8_t i : {0, 1, 16})
          cb.push_back({.cb_index = i, .data_format = tt::DataFormat::UInt32});
      }
      DataMovementConfig config{.processor = DataMovementProcessor::RISCV_0,
                                .noc = NOC::RISCV_0_default,
                                .compile_args = args};
      return compileDataMovementAOTKernel(sources.reader.c_str(), config, cb,
                                          target_arch, out_bundle, error_message);
    }
    for (uint8_t i : {0, 1, 4, 5, 6, 7, 10, 11})
      cb.push_back({.cb_index = i, .data_format = tt::DataFormat::Int8});
    for (uint8_t i : {8, 9, 12, 17, 18})
      cb.push_back({.cb_index = i, .data_format = tt::DataFormat::UInt32});
    cb.push_back({.cb_index = 3, .data_format = tt::DataFormat::UInt32});
    cb.push_back({.cb_index = 16, .data_format = tt::DataFormat::Int32});
    DataMovementConfig reader{.processor = DataMovementProcessor::RISCV_1,
                              .noc = NOC::RISCV_1_default,
                              .compile_args = args};
    DataMovementConfig writer{.processor = DataMovementProcessor::RISCV_0,
                              .noc = NOC::RISCV_0_default,
                              .compile_args = args};
    return compileAOTKernelBundle(sources.reader.c_str(), reader,
                                  sources.writer.c_str(), writer,
                                  sources.compute.c_str(), makeMatmulComputeConfig(),
                                  cb, target_arch, out_bundle, error_message);
  } catch (const std::exception &e) {
    error_message = e.what();
    return false;
  }
}

} // namespace mlir::iree_compiler::IREE::HAL
