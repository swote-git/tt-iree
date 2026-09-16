// Copyright 2025 The tt-iree Authors
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/hal/drivers/tenstorrent/tt_device.h"

#include <algorithm>
#include <array>
#include <cinttypes>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <limits>
#include <map>
#include <set>
#include <string>
#include <vector>

#include "iree/hal/drivers/tenstorrent/tt_allocator.h"
#include "iree/hal/drivers/tenstorrent/tt_buffer.h"
#include "iree/hal/drivers/tenstorrent/tt_command_buffer.h"
#include "iree/hal/drivers/tenstorrent/tt_executable.h"
#include "iree/hal/drivers/tenstorrent/tt_executable_cache.h"
#include "iree/hal/drivers/tenstorrent/tt_semaphore.h"
#include "iree/schema/tt_executable_builder_util.h"

#ifndef TT_IREE_ENABLE_MOCK
#include "impl/kernels/kernel.hpp"
#include "impl/program/program_impl.hpp"
#include "tt-metalium/bfloat16.hpp"
#include "tt-metalium/device.hpp"
#include "tt-metalium/distributed.hpp"
#include "tt-metalium/experimental/offline_kernel_compile.hpp"
#include "tt-metalium/host_api.hpp"
#include "tt-metalium/tilize_utils.hpp"
#include "tt-metalium/tt_metal.hpp"
#include "tt_executable_def_reader.h"
#define tt_device_ttex_ns(x) FLATBUFFERS_WRAP_NAMESPACE(iree_tenstorrent_hal, x)
#endif

//===----------------------------------------------------------------------===//
// iree_hal_tt_device_t
//===----------------------------------------------------------------------===//

struct iree_hal_tt_device_t {
  iree_hal_resource_t resource;
  iree_allocator_t host_allocator;

  iree_string_view_t identifier;
  iree_hal_device_id_t device_id;
  iree_hal_allocator_t *device_allocator;
  iree_hal_device_topology_info_t topology_info;

#ifndef TT_IREE_ENABLE_MOCK
  std::shared_ptr<tt::tt_metal::distributed::MeshDevice> mesh_device;
  tt::tt_metal::distributed::MeshCommandQueue *mesh_cq;
#endif
};

static iree_status_t
iree_hal_tt_copy_resolved_buffer_refs(const iree_hal_buffer_ref_t &source_ref,
                                      const iree_hal_buffer_ref_t &target_ref) {
  if (!source_ref.buffer || !target_ref.buffer) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "resolved copy buffers must not be null");
  }
  if (source_ref.length != target_ref.length) {
    return iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "resolved copy ranges must have equal lengths (source=%" PRIdsz
        " target=%" PRIdsz ")",
        source_ref.length, target_ref.length);
  }
  if (source_ref.length == 0)
    return iree_ok_status();

  if (iree_hal_buffer_test_overlap(source_ref.buffer, source_ref.offset,
                                   source_ref.length, target_ref.buffer,
                                   target_ref.offset, target_ref.length) !=
      IREE_HAL_BUFFER_OVERLAP_DISJOINT) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "resolved copy source and target ranges overlap");
  }

  iree_hal_buffer_mapping_t source_mapping = {};
  IREE_RETURN_IF_ERROR(
      iree_hal_buffer_map_range(source_ref.buffer, IREE_HAL_MAPPING_MODE_SCOPED,
                                IREE_HAL_MEMORY_ACCESS_READ, source_ref.offset,
                                source_ref.length, &source_mapping));

  iree_hal_buffer_mapping_t target_mapping = {};
  iree_status_t status =
      iree_hal_buffer_map_range(target_ref.buffer, IREE_HAL_MAPPING_MODE_SCOPED,
                                IREE_HAL_MEMORY_ACCESS_WRITE, target_ref.offset,
                                target_ref.length, &target_mapping);
  if (iree_status_is_ok(status)) {
    std::memcpy(target_mapping.contents.data, source_mapping.contents.data,
                static_cast<size_t>(source_ref.length));
    status =
        iree_status_join(status, iree_hal_buffer_unmap_range(&target_mapping));
  }
  status =
      iree_status_join(status, iree_hal_buffer_unmap_range(&source_mapping));
  return status;
}

static iree_status_t iree_hal_tt_resolve_transfer_buffer_ref(
    iree_hal_buffer_binding_table_t binding_table,
    const iree_hal_buffer_ref_t &buffer_ref,
    iree_hal_buffer_ref_t *out_resolved_ref) {
  if (!out_resolved_ref) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "out_resolved_ref is null");
  }
  *out_resolved_ref = {};

  if (!buffer_ref.buffer) {
    if (!binding_table.bindings ||
        buffer_ref.buffer_slot >= binding_table.count) {
      return iree_make_status(
          IREE_STATUS_OUT_OF_RANGE,
          "indirect buffer slot %u is not present in binding table of size %zu",
          buffer_ref.buffer_slot, binding_table.count);
    }
    const iree_hal_buffer_binding_t &binding =
        binding_table.bindings[buffer_ref.buffer_slot];
    if (!binding.buffer) {
      return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                              "indirect buffer slot %u is null",
                              buffer_ref.buffer_slot);
    }
    iree_device_size_t binding_offset = 0;
    iree_device_size_t binding_length = 0;
    IREE_RETURN_IF_ERROR(iree_hal_buffer_calculate_range(
        /*base_offset=*/0, iree_hal_buffer_byte_length(binding.buffer),
        binding.offset, binding.length, &binding_offset, &binding_length));
  }

  IREE_RETURN_IF_ERROR(iree_hal_buffer_binding_table_resolve_ref(
      binding_table, buffer_ref, out_resolved_ref));
  if (!out_resolved_ref->buffer) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "resolved buffer reference is null");
  }

  iree_device_size_t resolved_offset = 0;
  iree_device_size_t resolved_length = 0;
  IREE_RETURN_IF_ERROR(iree_hal_buffer_calculate_range(
      /*base_offset=*/0, iree_hal_buffer_byte_length(out_resolved_ref->buffer),
      out_resolved_ref->offset, out_resolved_ref->length, &resolved_offset,
      &resolved_length));
  out_resolved_ref->offset = resolved_offset;
  out_resolved_ref->length = resolved_length;
  return iree_ok_status();
}

static iree_status_t iree_hal_tt_execute_transfer_command(
    iree_hal_command_buffer_t *command_buffer,
    iree_hal_buffer_binding_table_t binding_table,
    const iree_hal_tt_command_t &command, bool *out_handled) {
  *out_handled = true;
  switch (command.type) {
  case IREE_HAL_TT_COMMAND_TYPE_FILL: {
    const auto &cmd = command.fill;
    iree_hal_buffer_ref_t target_ref = {};
    IREE_RETURN_IF_ERROR(iree_hal_tt_resolve_transfer_buffer_ref(
        binding_table, cmd.target_ref, &target_ref));
    return iree_hal_buffer_map_fill(target_ref.buffer, target_ref.offset,
                                    target_ref.length, cmd.pattern,
                                    cmd.pattern_length);
  }
  case IREE_HAL_TT_COMMAND_TYPE_UPDATE: {
    const auto &cmd = command.update;
    iree_hal_buffer_ref_t target_ref = {};
    IREE_RETURN_IF_ERROR(iree_hal_tt_resolve_transfer_buffer_ref(
        binding_table, cmd.target_ref, &target_ref));
    iree_const_byte_span_t source_data = iree_const_byte_span_empty();
    IREE_RETURN_IF_ERROR(iree_hal_tt_command_buffer_get_update_data(
        command_buffer, cmd.source_data_offset, cmd.source_data_length,
        &source_data));
    if (target_ref.length != source_data.data_length) {
      return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                              "resolved update target length %" PRIdsz
                              " does not match recorded source length %" PRIhsz,
                              target_ref.length, source_data.data_length);
    }
    return iree_hal_buffer_map_write(target_ref.buffer, target_ref.offset,
                                     source_data.data, target_ref.length);
  }
  case IREE_HAL_TT_COMMAND_TYPE_COPY: {
    const auto &cmd = command.copy;
    iree_hal_buffer_ref_t source_ref = {};
    iree_hal_buffer_ref_t target_ref = {};
    IREE_RETURN_IF_ERROR(iree_hal_tt_resolve_transfer_buffer_ref(
        binding_table, cmd.source_ref, &source_ref));
    IREE_RETURN_IF_ERROR(iree_hal_tt_resolve_transfer_buffer_ref(
        binding_table, cmd.target_ref, &target_ref));
    return iree_hal_tt_copy_resolved_buffer_refs(source_ref, target_ref);
  }
  default:
    *out_handled = false;
    return iree_ok_status();
  }
}

#ifndef TT_IREE_ENABLE_MOCK
static iree_status_t iree_hal_tt_set_add_runtime_args(
    tt::tt_metal::Program &program, const iree_hal_tt_kernel_params_t *params,
    const tt::tt_metal::CoreCoord &core, uint32_t in0_addr, uint32_t in1_addr,
    uint32_t out_addr, uint32_t in0_byte_length, uint32_t in1_byte_length,
    uint32_t out_byte_length) {
  constexpr uint32_t bf16_tile_bytes = 32 * 32 * 2;
  if (in0_byte_length == 0 || in0_byte_length != in1_byte_length ||
      in0_byte_length != out_byte_length ||
      in0_byte_length % bf16_tile_bytes != 0) {
    return iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "builtin add expects equal, non-empty, tile-aligned bindings "
        "(in0=%u in1=%u out=%u bytes)",
        in0_byte_length, in1_byte_length, out_byte_length);
  }
  uint32_t n_tiles = in0_byte_length / bf16_tile_bytes;

  std::vector<uint32_t> reader_args = {in0_addr, in1_addr, n_tiles};
  tt::tt_metal::SetRuntimeArgs(program, params->reader_kernel_id, core,
                               reader_args);

  std::vector<uint32_t> writer_args = {out_addr, n_tiles};
  tt::tt_metal::SetRuntimeArgs(program, params->writer_kernel_id, core,
                               writer_args);

  std::vector<uint32_t> compute_args = {n_tiles};
  tt::tt_metal::SetRuntimeArgs(program, params->compute_kernel_id, core,
                               compute_args);

  return iree_ok_status();
}

static iree_status_t iree_hal_tt_set_matmul_runtime_args(
    tt::tt_metal::Program &program, const iree_hal_tt_kernel_params_t *params,
    const tt::tt_metal::CoreCoord &core, uint32_t lhs_addr, uint32_t rhs_addr,
    uint32_t out_addr, uint32_t lhs_byte_length, uint32_t rhs_byte_length,
    uint32_t out_byte_length) {
  constexpr uint32_t kTileBytes = 32 * 32 * 2;
  if (lhs_byte_length != kTileBytes || rhs_byte_length != kTileBytes ||
      out_byte_length != kTileBytes) {
    return iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "builtin matmul expects exactly one 32x32 bf16 tile per binding "
        "(lhs=%u rhs=%u out=%u bytes)",
        lhs_byte_length, rhs_byte_length, out_byte_length);
  }

  std::vector<uint32_t> reader_args = {
      lhs_addr,
      0, // src0_bank_id
      rhs_addr,
      0, // src1_bank_id
      1, // num_blocks
      1, // in0_block_tile_cnt
      1, // in1_block_tile_cnt
      kTileBytes, kTileBytes,
  };
  tt::tt_metal::SetRuntimeArgs(program, params->reader_kernel_id, core,
                               reader_args);

  std::array<uint32_t, 9> writer_args = {
      out_addr,
      0,          // dst_bank_id
      1,          // inner_r
      1,          // inner_c
      1,          // num_sub_blocks_m
      1,          // num_sub_blocks_n
      kTileBytes, // stride_r
      kTileBytes, // stride_subblock_r
      kTileBytes, // stride_subblock_c
  };
  std::array<uint32_t, 11> compute_args = {
      1, // in0_block_w
      1, // in0_num_subblocks
      1, // in0_block_num_tiles
      1, // in0_subblock_num_tiles
      1, // in1_num_subblocks
      1, // in1_block_num_tiles
      1, // in1_per_core_w
      1, // num_blocks
      1, // out_subblock_h
      1, // out_subblock_w
      1, // out_subblock_num_tiles
  };
  tt::tt_metal::SetRuntimeArgs(program, params->writer_kernel_id, core,
                               writer_args);
  tt::tt_metal::SetRuntimeArgs(program, params->compute_kernel_id, core,
                               compute_args);

  return iree_ok_status();
}

static uint32_t iree_hal_tt_decode_u32_le(const uint8_t *data) {
  return static_cast<uint32_t>(data[0]) |
         (static_cast<uint32_t>(data[1]) << 8) |
         (static_cast<uint32_t>(data[2]) << 16) |
         (static_cast<uint32_t>(data[3]) << 24);
}

static uint64_t iree_hal_tt_decode_index_pair(const uint8_t *data) {
  uint64_t lo = iree_hal_tt_decode_u32_le(data);
  uint64_t hi = iree_hal_tt_decode_u32_le(data + 4);
  return lo | (hi << 32);
}

static bool iree_hal_tt_is_tile_aligned(uint64_t value) {
  return value != 0 && value % TT_TILE_WIDTH == 0;
}

static iree_status_t
iree_hal_tt_resolve_buffer_ref(iree_hal_buffer_binding_table_t binding_table,
                               const iree_hal_buffer_ref_t &buffer_ref,
                               iree_hal_buffer_ref_t *out_resolved_ref) {
  return iree_hal_tt_resolve_transfer_buffer_ref(binding_table, buffer_ref,
                                                 out_resolved_ref);
}

static iree_status_t
iree_hal_tt_get_binding_runtime_args(const iree_hal_buffer_ref_t &resolved_ref,
                                     uint32_t *out_device_address,
                                     uint32_t *out_byte_length) {
  if (resolved_ref.length > std::numeric_limits<uint32_t>::max()) {
    return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                            "binding length exceeds 32-bit runtime ABI");
  }
  uint32_t base_address =
      iree_hal_tt_buffer_device_address(resolved_ref.buffer);
  if (resolved_ref.offset >
      std::numeric_limits<uint32_t>::max() - base_address) {
    return iree_make_status(
        IREE_STATUS_OUT_OF_RANGE,
        "binding device address exceeds 32-bit runtime ABI");
  }
  *out_device_address =
      base_address + static_cast<uint32_t>(resolved_ref.offset);
  *out_byte_length = static_cast<uint32_t>(resolved_ref.length);
  return iree_ok_status();
}

static std::vector<uint32_t>
iree_hal_tt_transpose_tiles(const std::vector<uint32_t> &data,
                            uint32_t row_tiles, uint32_t col_tiles,
                            uint32_t block_width_tiles) {
  constexpr uint32_t kWordsPerTile =
      (TT_TILE_SIZE * sizeof(uint16_t)) / sizeof(uint32_t);
  std::vector<uint32_t> result;
  result.reserve(data.size());
  for (uint32_t c = 0; c < col_tiles; c += block_width_tiles) {
    for (uint32_t r = 0; r < row_tiles; ++r) {
      for (uint32_t k = 0; k < block_width_tiles; ++k) {
        uint32_t tile_index = r * col_tiles + c + k;
        uint32_t offset = tile_index * kWordsPerTile;
        result.insert(result.end(), data.begin() + offset,
                      data.begin() + offset + kWordsPerTile);
      }
    }
  }
  return result;
}

static iree_status_t iree_hal_tt_read_buffer_bfloat16(
    iree_hal_buffer_t *buffer, iree_device_size_t byte_offset,
    iree_device_size_t byte_length, std::vector<bfloat16> *out_values) {
  if (!out_values) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT, "out_values is null");
  }
  out_values->assign(byte_length / sizeof(bfloat16), bfloat16(0.0f));

  iree_hal_buffer_mapping_t mapping = {};
  IREE_RETURN_IF_ERROR(iree_hal_buffer_map_range(
      buffer, IREE_HAL_MAPPING_MODE_SCOPED, IREE_HAL_MEMORY_ACCESS_READ,
      byte_offset, byte_length, &mapping));
  std::memcpy(out_values->data(), mapping.contents.data, byte_length);
  return iree_hal_buffer_unmap_range(&mapping);
}

static iree_status_t
iree_hal_tt_write_buffer_bfloat16(iree_hal_buffer_t *buffer,
                                  iree_device_size_t byte_offset,
                                  const std::vector<bfloat16> &values) {
  iree_device_size_t byte_length = values.size() * sizeof(bfloat16);
  iree_hal_buffer_mapping_t mapping = {};
  IREE_RETURN_IF_ERROR(iree_hal_buffer_map_range(
      buffer, IREE_HAL_MAPPING_MODE_SCOPED, IREE_HAL_MEMORY_ACCESS_WRITE,
      byte_offset, byte_length, &mapping));
  std::memcpy(mapping.contents.data, values.data(), byte_length);
  return iree_hal_buffer_unmap_range(&mapping);
}

static std::vector<bfloat16>
iree_hal_tt_slice_row_major_matrix(const std::vector<bfloat16> &matrix,
                                   uint32_t rows, uint32_t cols,
                                   uint32_t row_offset, uint32_t col_offset,
                                   uint32_t slice_rows, uint32_t slice_cols) {
  std::vector<bfloat16> slice(slice_rows * slice_cols);
  for (uint32_t r = 0; r < slice_rows; ++r) {
    const bfloat16 *src = matrix.data() + (row_offset + r) * cols + col_offset;
    bfloat16 *dst = slice.data() + r * slice_cols;
    std::copy(src, src + slice_cols, dst);
  }
  return slice;
}

static void iree_hal_tt_scatter_row_major_matrix(
    const std::vector<bfloat16> &slice, uint32_t slice_rows,
    uint32_t slice_cols, uint32_t row_offset, uint32_t col_offset,
    uint32_t full_cols, std::vector<bfloat16> *full_matrix) {
  for (uint32_t r = 0; r < slice_rows; ++r) {
    const bfloat16 *src = slice.data() + r * slice_cols;
    bfloat16 *dst =
        full_matrix->data() + (row_offset + r) * full_cols + col_offset;
    std::copy(src, src + slice_cols, dst);
  }
}

static std::vector<uint32_t>
iree_hal_tt_stage_lhs_tiles(const std::vector<bfloat16> &row_major,
                            uint32_t rows, uint32_t cols, uint32_t row_tiles,
                            uint32_t col_tiles, uint32_t block_width_tiles) {
  auto swizzled = tilize_swizzled(row_major, rows, cols);
  auto nfaces = convert_layout_tile_swizzled_to_tile_nfaces(
      ttsl::make_const_span(swizzled));
  auto packed = pack_bfloat16_vec_into_uint32_vec(nfaces);
  return iree_hal_tt_transpose_tiles(packed, row_tiles, col_tiles,
                                     block_width_tiles);
}

static std::vector<uint32_t>
iree_hal_tt_stage_rhs_tiles(const std::vector<bfloat16> &row_major,
                            uint32_t rows, uint32_t cols) {
  auto swizzled = tilize_swizzled(row_major, rows, cols);
  auto nfaces = convert_layout_tile_swizzled_to_tile_nfaces(
      ttsl::make_const_span(swizzled));
  return pack_bfloat16_vec_into_uint32_vec(nfaces);
}

static std::vector<bfloat16>
iree_hal_tt_collect_output_tiles(const std::vector<uint32_t> &packed_tiles,
                                 uint32_t rows, uint32_t cols) {
  auto bfloat_tiles = unpack_uint32_vec_into_bfloat16_vec(packed_tiles);
  auto swizzled = convert_layout_tile_nfaces_to_tile_swizzled(
      ttsl::make_const_span(bfloat_tiles));
  return untilize_swizzled(swizzled, rows, cols);
}

static iree_status_t iree_hal_tt_decode_tiled_matmul_shape(
    const iree_hal_tt_kernel_params_t *params,
    const iree_hal_tt_dispatch_command_t &cmd, uint32_t *out_m_tiles,
    uint32_t *out_n_tiles, uint32_t *out_k_tiles) {
  if (params->builtin_m_tiles && params->builtin_n_tiles &&
      params->builtin_k_tiles) {
    *out_m_tiles = params->builtin_m_tiles;
    *out_n_tiles = params->builtin_n_tiles;
    *out_k_tiles = params->builtin_k_tiles;
    return iree_ok_status();
  }

  if (cmd.constants_length != 8 * sizeof(uint32_t)) {
    return iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "dynamic tiled matmul expects 8 i32 dispatch constants, got %zu bytes",
        cmd.constants_length);
  }

  uint64_t lhs_k = iree_hal_tt_decode_index_pair(cmd.constants + 0);
  uint64_t rhs_k = iree_hal_tt_decode_index_pair(cmd.constants + 8);
  uint64_t lhs_m = iree_hal_tt_decode_index_pair(cmd.constants + 16);
  uint64_t rhs_n = iree_hal_tt_decode_index_pair(cmd.constants + 24);
  if (lhs_k != rhs_k) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "matmul K dimensions disagree (%" PRIu64
                            " vs %" PRIu64 ")",
                            lhs_k, rhs_k);
  }
  if (!iree_hal_tt_is_tile_aligned(lhs_m) ||
      !iree_hal_tt_is_tile_aligned(rhs_n) ||
      !iree_hal_tt_is_tile_aligned(lhs_k)) {
    return iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "dynamic tiled matmul expects M/N/K to be multiples of 32");
  }

  uint64_t m_tiles = lhs_m / TT_TILE_HEIGHT;
  uint64_t n_tiles = rhs_n / TT_TILE_WIDTH;
  uint64_t k_tiles = lhs_k / TT_TILE_WIDTH;
  if (m_tiles > std::numeric_limits<uint32_t>::max() ||
      n_tiles > std::numeric_limits<uint32_t>::max() ||
      k_tiles > std::numeric_limits<uint32_t>::max()) {
    return iree_make_status(
        IREE_STATUS_OUT_OF_RANGE,
        "dynamic tiled matmul dimensions exceed the 32-bit runtime ABI");
  }

  *out_m_tiles = static_cast<uint32_t>(m_tiles);
  *out_n_tiles = static_cast<uint32_t>(n_tiles);
  *out_k_tiles = static_cast<uint32_t>(k_tiles);
  return iree_ok_status();
}

static std::string iree_hal_tt_find_aot_kernel_source(
    tt_device_ttex_ns(TTMetalKernelBundleDef_table_t) bundle, uint8_t role) {
  tt_device_ttex_ns(KernelSourceDef_vec_t) kernels =
      tt_device_ttex_ns(TTMetalKernelBundleDef_kernels(bundle));
  size_t kernel_count = tt_device_ttex_ns(KernelSourceDef_vec_len(kernels));
  for (size_t i = 0; i < kernel_count; ++i) {
    tt_device_ttex_ns(KernelSourceDef_table_t) kernel =
        tt_device_ttex_ns(KernelSourceDef_vec_at(kernels, i));
    if (tt_device_ttex_ns(KernelSourceDef_role(kernel)) != role)
      continue;
    flatbuffers_string_t source =
        tt_device_ttex_ns(KernelSourceDef_source(kernel));
    return std::string(source, flatbuffers_string_len(source));
  }
  throw std::runtime_error("AOT kernel role is missing");
}

template <typename CoreSpec, typename KernelConfig>
static tt::tt_metal::KernelHandle
iree_hal_tt_create_kernel_from_embedded_precompiled(
    tt::tt_metal::Program &program, const std::string &source,
    const CoreSpec &core_spec, const KernelConfig &config,
    const std::string &artifact_root) {
  tt::tt_metal::KernelHandle kernel =
      tt::tt_metal::CreateKernelFromString(program, source, core_spec, config);
  program.impl().get_kernel(kernel)->set_precompiled_config(
      tt::tt_metal::experimental::PrecompiledKernelConfig{
          .precompiled_dir = artifact_root,
          .fallback_policy = tt::tt_metal::experimental::
              PrecompiledKernelConfig::FallbackPolicy::Error,
      });
  return kernel;
}

static iree_status_t iree_hal_tt_execute_tiled_matmul_host_matrices(
    iree_hal_tt_device_t *device, const iree_hal_tt_kernel_params_t *params,
    const iree_hal_dispatch_config_t &config, uint32_t m_tiles,
    uint32_t n_tiles, uint32_t k_tiles,
    const std::vector<bfloat16> &lhs_row_major,
    const std::vector<bfloat16> &rhs_row_major,
    std::vector<bfloat16> *out_row_major) {
  if (!out_row_major) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "out_row_major is null");
  }
  if (!params || !params->aot_kernel_bundle) {
    return iree_make_status(
        IREE_STATUS_FAILED_PRECONDITION,
        "tiled matmul requires a compiler-generated TTEX AOT kernel bundle");
  }
  constexpr uint32_t kTileBytes = TT_TILE_SIZE * sizeof(uint16_t);
  constexpr uint32_t kBlockWidthTiles = 1;
  constexpr uint32_t kOutSubblockH = 1;
  constexpr uint32_t kOutSubblockW = 1;

  uint32_t lhs_rows = m_tiles * TT_TILE_HEIGHT;
  uint32_t lhs_cols = k_tiles * TT_TILE_WIDTH;
  uint32_t rhs_rows = k_tiles * TT_TILE_HEIGHT;
  uint32_t rhs_cols = n_tiles * TT_TILE_WIDTH;
  uint32_t out_rows = m_tiles * TT_TILE_HEIGHT;
  uint32_t out_cols = n_tiles * TT_TILE_WIDTH;

  if (lhs_row_major.size() != static_cast<size_t>(lhs_rows) * lhs_cols ||
      rhs_row_major.size() != static_cast<size_t>(rhs_rows) * rhs_cols) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "tiled matmul host matrices do not match M/N/K"
                            " (lhs=%zu rhs=%zu)",
                            lhs_row_major.size(), rhs_row_major.size());
  }

  tt::tt_metal::IDevice *idevice = iree_hal_tt_device_idevice(device);
  if (!idevice) {
    return iree_make_status(IREE_STATUS_UNAVAILABLE,
                            "TT-Metal device not initialized");
  }
  tt::ARCH expected_arch = tt::ARCH::Invalid;
  const char *expected_arch_name = "unknown";
  switch (params->aot_target_arch) {
  case TT_IREE_TTEX_TARGET_ARCH_WORMHOLE_B0:
    expected_arch = tt::ARCH::WORMHOLE_B0;
    expected_arch_name = "wormhole_b0";
    break;
  case TT_IREE_TTEX_TARGET_ARCH_BLACKHOLE:
    expected_arch = tt::ARCH::BLACKHOLE;
    expected_arch_name = "blackhole";
    break;
  default:
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "AOT bundle has an invalid target arch %u",
                            params->aot_target_arch);
  }
  if (idevice->arch() != expected_arch) {
    return iree_make_status(
        IREE_STATUS_FAILED_PRECONDITION,
        "AOT bundle targets %s but the device reports architecture %u",
        expected_arch_name, static_cast<unsigned>(idevice->arch()));
  }

  auto grid = idevice->compute_with_storage_grid_size();
  uint32_t num_cores_c = config.workgroup_count[0];
  uint32_t num_cores_r = config.workgroup_count[1];
  if (num_cores_c == 0 || num_cores_r == 0 || config.workgroup_count[2] != 1) {
    return iree_make_status(
        IREE_STATUS_UNIMPLEMENTED,
        "tiled matmul requires a 2D workgroup grid, got %ux%ux%u", num_cores_c,
        num_cores_r, config.workgroup_count[2]);
  }
  if (num_cores_c > grid.x || num_cores_r > grid.y) {
    return iree_make_status(
        IREE_STATUS_RESOURCE_EXHAUSTED,
        "workgroup grid %ux%u exceeds TT compute grid %zux%zu", num_cores_c,
        num_cores_r, grid.x, grid.y);
  }
  if (n_tiles % num_cores_c != 0 || m_tiles % num_cores_r != 0) {
    return iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "workgroup grid %ux%u does not evenly partition matmul output tiles "
        "%ux%u",
        num_cores_c, num_cores_r, n_tiles, m_tiles);
  }
  uint32_t per_core_m_tiles = m_tiles / num_cores_r;
  uint32_t per_core_n_tiles = n_tiles / num_cores_c;
  uint32_t per_core_count = num_cores_r * num_cores_c;

  try {
    std::vector<std::shared_ptr<tt::tt_metal::Buffer>> lhs_scratch(
        per_core_count);
    std::vector<std::shared_ptr<tt::tt_metal::Buffer>> rhs_scratch(
        per_core_count);
    std::vector<std::shared_ptr<tt::tt_metal::Buffer>> out_scratch(
        per_core_count);

    for (uint32_t core_index = 0; core_index < per_core_count; ++core_index) {
      iree_device_size_t lhs_size =
          static_cast<iree_device_size_t>(per_core_m_tiles) * k_tiles *
          kTileBytes;
      iree_device_size_t rhs_size = static_cast<iree_device_size_t>(k_tiles) *
                                    per_core_n_tiles * kTileBytes;
      iree_device_size_t out_size =
          static_cast<iree_device_size_t>(per_core_m_tiles) * per_core_n_tiles *
          kTileBytes;
      lhs_scratch[core_index] =
          tt::tt_metal::CreateBuffer(tt::tt_metal::InterleavedBufferConfig{
              .device = idevice,
              .size = static_cast<uint32_t>(lhs_size),
              .page_size = static_cast<uint32_t>(lhs_size),
              .buffer_type = tt::tt_metal::BufferType::DRAM,
          });
      rhs_scratch[core_index] =
          tt::tt_metal::CreateBuffer(tt::tt_metal::InterleavedBufferConfig{
              .device = idevice,
              .size = static_cast<uint32_t>(rhs_size),
              .page_size = static_cast<uint32_t>(rhs_size),
              .buffer_type = tt::tt_metal::BufferType::DRAM,
          });
      out_scratch[core_index] =
          tt::tt_metal::CreateBuffer(tt::tt_metal::InterleavedBufferConfig{
              .device = idevice,
              .size = static_cast<uint32_t>(out_size),
              .page_size = static_cast<uint32_t>(out_size),
              .buffer_type = tt::tt_metal::BufferType::DRAM,
          });
    }

    for (uint32_t core_r = 0; core_r < num_cores_r; ++core_r) {
      for (uint32_t core_c = 0; core_c < num_cores_c; ++core_c) {
        uint32_t core_index = core_r * num_cores_c + core_c;
        uint32_t lhs_row_offset = core_r * per_core_m_tiles * TT_TILE_HEIGHT;
        uint32_t rhs_col_offset = core_c * per_core_n_tiles * TT_TILE_WIDTH;

        auto lhs_slice = iree_hal_tt_slice_row_major_matrix(
            lhs_row_major, lhs_rows, lhs_cols, lhs_row_offset,
            /*col_offset=*/0, per_core_m_tiles * TT_TILE_HEIGHT, lhs_cols);
        auto rhs_slice = iree_hal_tt_slice_row_major_matrix(
            rhs_row_major, rhs_rows, rhs_cols, /*row_offset=*/0, rhs_col_offset,
            rhs_rows, per_core_n_tiles * TT_TILE_WIDTH);

        auto lhs_packed = iree_hal_tt_stage_lhs_tiles(
            lhs_slice, per_core_m_tiles * TT_TILE_HEIGHT, lhs_cols,
            per_core_m_tiles, k_tiles, kBlockWidthTiles);
        auto rhs_packed = iree_hal_tt_stage_rhs_tiles(
            rhs_slice, rhs_rows, per_core_n_tiles * TT_TILE_WIDTH);

        tt::tt_metal::detail::WriteToBuffer(lhs_scratch[core_index],
                                            lhs_packed);
        tt::tt_metal::detail::WriteToBuffer(rhs_scratch[core_index],
                                            rhs_packed);
      }
    }

    tt::tt_metal::Program program = tt::tt_metal::CreateProgram();
    tt::tt_metal::CoreRange all_cores({0, 0},
                                      {num_cores_c - 1, num_cores_r - 1});
    tt::tt_metal::CoreRangeSet all_core_set(
        std::set<tt::tt_metal::CoreRange>{all_cores});

    for (uint32_t core_r = 0; core_r < num_cores_r; ++core_r) {
      for (uint32_t core_c = 0; core_c < num_cores_c; ++core_c) {
        tt::tt_metal::CoreCoord core = {core_c, core_r};
        uint32_t cb0_tiles = per_core_m_tiles * kBlockWidthTiles * 2;
        uint32_t cb1_tiles = per_core_n_tiles * kBlockWidthTiles * 2;
        tt::tt_metal::CreateCircularBuffer(
            program, core,
            tt::tt_metal::CircularBufferConfig(
                cb0_tiles * kTileBytes,
                {{tt::CBIndex::c_0, tt::DataFormat::Float16_b}})
                .set_page_size(tt::CBIndex::c_0, kTileBytes));
        tt::tt_metal::CreateCircularBuffer(
            program, core,
            tt::tt_metal::CircularBufferConfig(
                cb1_tiles * kTileBytes,
                {{tt::CBIndex::c_1, tt::DataFormat::Float16_b}})
                .set_page_size(tt::CBIndex::c_1, kTileBytes));

        std::map<uint8_t, tt::DataFormat> output_data_format = {
            {tt::CBIndex::c_16, tt::DataFormat::Float16_b},
            {tt::CBIndex::c_24, tt::DataFormat::Float16_b},
        };
        tt::tt_metal::CoreRangeSet one_core_set(
            std::set<tt::tt_metal::CoreRange>{
                tt::tt_metal::CoreRange(core, core)});
        tt::tt_metal::CreateCircularBuffer(
            program, one_core_set,
            tt::tt_metal::CircularBufferConfig(
                per_core_m_tiles * per_core_n_tiles * kTileBytes,
                output_data_format)
                .set_page_size(tt::CBIndex::c_16, kTileBytes)
                .set_page_size(tt::CBIndex::c_24, kTileBytes));
      }
    }

    tt::tt_metal::DataMovementConfig reader_config{
        .processor = tt::tt_metal::DataMovementProcessor::RISCV_1,
        .noc = tt::tt_metal::NOC::RISCV_1_default};
    tt::tt_metal::DataMovementConfig writer_config{
        .processor = tt::tt_metal::DataMovementProcessor::RISCV_0,
        .noc = tt::tt_metal::NOC::RISCV_0_default};

    std::vector<uint32_t> compute_kernel_args = {
        kBlockWidthTiles,
        per_core_m_tiles / kOutSubblockH,
        per_core_m_tiles,
        kOutSubblockH * kBlockWidthTiles,
        per_core_n_tiles / kOutSubblockW,
        per_core_n_tiles,
        per_core_n_tiles,
        k_tiles / kBlockWidthTiles,
        kOutSubblockH,
        kOutSubblockW,
        kOutSubblockH * kOutSubblockW,
    };
    auto aot_bundle =
        static_cast<tt_device_ttex_ns(TTMetalKernelBundleDef_table_t)>(
            params->aot_kernel_bundle);
    if (iree_string_view_is_empty(params->aot_artifact_root)) {
      throw std::runtime_error("AOT artifact cache is missing");
    }
    std::string root(params->aot_artifact_root.data,
                     params->aot_artifact_root.size);
    tt::tt_metal::KernelHandle reader_kernel =
        iree_hal_tt_create_kernel_from_embedded_precompiled(
            program,
            iree_hal_tt_find_aot_kernel_source(aot_bundle,
                                               TT_IREE_TTEX_KERNEL_ROLE_READER),
            all_core_set, reader_config, root);
    tt::tt_metal::KernelHandle writer_kernel =
        iree_hal_tt_create_kernel_from_embedded_precompiled(
            program,
            iree_hal_tt_find_aot_kernel_source(aot_bundle,
                                               TT_IREE_TTEX_KERNEL_ROLE_WRITER),
            all_core_set, writer_config, root);
    tt::tt_metal::KernelHandle compute_kernel =
        iree_hal_tt_create_kernel_from_embedded_precompiled(
            program,
            iree_hal_tt_find_aot_kernel_source(
                aot_bundle, TT_IREE_TTEX_KERNEL_ROLE_COMPUTE),
            all_core_set, tt::tt_metal::ComputeConfig{}, root);

    for (uint32_t core_r = 0; core_r < num_cores_r; ++core_r) {
      for (uint32_t core_c = 0; core_c < num_cores_c; ++core_c) {
        uint32_t core_index = core_r * num_cores_c + core_c;
        tt::tt_metal::CoreCoord core = {core_c, core_r};
        std::array<uint32_t, 9> reader_args = {
            lhs_scratch[core_index]->address(),
            0,
            rhs_scratch[core_index]->address(),
            0,
            k_tiles / kBlockWidthTiles,
            per_core_m_tiles * kBlockWidthTiles,
            per_core_n_tiles * kBlockWidthTiles,
            per_core_m_tiles * kBlockWidthTiles * kTileBytes,
            per_core_n_tiles * kBlockWidthTiles * kTileBytes,
        };
        std::array<uint32_t, 9> writer_args = {
            out_scratch[core_index]->address(),
            0,
            kOutSubblockH,
            kOutSubblockW,
            per_core_m_tiles / kOutSubblockH,
            per_core_n_tiles / kOutSubblockW,
            kOutSubblockW * kTileBytes * (per_core_n_tiles / kOutSubblockW),
            kOutSubblockH * kOutSubblockW * kTileBytes *
                (per_core_n_tiles / kOutSubblockW),
            kOutSubblockW * kTileBytes,
        };
        tt::tt_metal::SetRuntimeArgs(program, reader_kernel, core, reader_args);
        tt::tt_metal::SetRuntimeArgs(program, writer_kernel, core, writer_args);
        tt::tt_metal::SetRuntimeArgs(program, compute_kernel, core,
                                     compute_kernel_args);
      }
    }

    tt::tt_metal::distributed::MeshWorkload workload;
    auto device_range = tt::tt_metal::distributed::MeshCoordinateRange(
        device->mesh_device->shape());
    workload.add_program(device_range, std::move(program));
    tt::tt_metal::distributed::EnqueueMeshWorkload(*device->mesh_cq, workload,
                                                   /*blocking=*/true);

    out_row_major->assign(out_rows * out_cols, bfloat16(0.0f));
    for (uint32_t core_r = 0; core_r < num_cores_r; ++core_r) {
      for (uint32_t core_c = 0; core_c < num_cores_c; ++core_c) {
        uint32_t core_index = core_r * num_cores_c + core_c;
        std::vector<uint32_t> packed_output;
        tt::tt_metal::detail::ReadFromBuffer(out_scratch[core_index],
                                             packed_output);
        auto slice_row_major = iree_hal_tt_collect_output_tiles(
            packed_output, per_core_m_tiles * TT_TILE_HEIGHT,
            per_core_n_tiles * TT_TILE_WIDTH);
        iree_hal_tt_scatter_row_major_matrix(
            slice_row_major, per_core_m_tiles * TT_TILE_HEIGHT,
            per_core_n_tiles * TT_TILE_WIDTH,
            core_r * per_core_m_tiles * TT_TILE_HEIGHT,
            core_c * per_core_n_tiles * TT_TILE_WIDTH, out_cols, out_row_major);
      }
    }

    return iree_ok_status();
  } catch (const std::exception &e) {
    return iree_make_status(IREE_STATUS_INTERNAL,
                            "tiled matmul setup failed: %s", e.what());
  }
}

static iree_status_t iree_hal_tt_execute_tiled_matmul_dispatch(
    iree_hal_tt_device_t *device, const iree_hal_tt_kernel_params_t *params,
    const iree_hal_tt_dispatch_command_t &cmd,
    iree_hal_buffer_binding_table_t binding_table) {
  if (cmd.binding_count != 3) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "tiled matmul expects exactly 3 bindings, got %zu",
                            cmd.binding_count);
  }

  iree_hal_buffer_ref_t lhs_ref = {};
  iree_hal_buffer_ref_t rhs_ref = {};
  iree_hal_buffer_ref_t out_ref = {};
  IREE_RETURN_IF_ERROR(
      iree_hal_tt_resolve_buffer_ref(binding_table, cmd.bindings[0], &lhs_ref));
  IREE_RETURN_IF_ERROR(
      iree_hal_tt_resolve_buffer_ref(binding_table, cmd.bindings[1], &rhs_ref));
  IREE_RETURN_IF_ERROR(
      iree_hal_tt_resolve_buffer_ref(binding_table, cmd.bindings[2], &out_ref));

  uint32_t m_tiles = 0;
  uint32_t n_tiles = 0;
  uint32_t k_tiles = 0;
  IREE_RETURN_IF_ERROR(iree_hal_tt_decode_tiled_matmul_shape(
      params, cmd, &m_tiles, &n_tiles, &k_tiles));

  iree_device_size_t expected_lhs_bytes =
      static_cast<iree_device_size_t>(m_tiles) * k_tiles * TT_TILE_SIZE *
      sizeof(bfloat16);
  iree_device_size_t expected_rhs_bytes =
      static_cast<iree_device_size_t>(k_tiles) * n_tiles * TT_TILE_SIZE *
      sizeof(bfloat16);
  iree_device_size_t expected_out_bytes =
      static_cast<iree_device_size_t>(m_tiles) * n_tiles * TT_TILE_SIZE *
      sizeof(bfloat16);
  if (lhs_ref.length != expected_lhs_bytes ||
      rhs_ref.length != expected_rhs_bytes ||
      out_ref.length != expected_out_bytes) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "tiled matmul binding ranges do not match M/N/K"
                            " (lhs=%" PRIhsz " rhs=%" PRIhsz " out=%" PRIhsz
                            ")",
                            lhs_ref.length, rhs_ref.length, out_ref.length);
  }

  std::vector<bfloat16> lhs_row_major;
  std::vector<bfloat16> rhs_row_major;
  std::vector<bfloat16> out_row_major;
  IREE_RETURN_IF_ERROR(iree_hal_tt_read_buffer_bfloat16(
      lhs_ref.buffer, lhs_ref.offset, lhs_ref.length, &lhs_row_major));
  IREE_RETURN_IF_ERROR(iree_hal_tt_read_buffer_bfloat16(
      rhs_ref.buffer, rhs_ref.offset, rhs_ref.length, &rhs_row_major));
  IREE_RETURN_IF_ERROR(iree_hal_tt_execute_tiled_matmul_host_matrices(
      device, params, cmd.config, m_tiles, n_tiles, k_tiles, lhs_row_major,
      rhs_row_major, &out_row_major));
  return iree_hal_tt_write_buffer_bfloat16(out_ref.buffer, out_ref.offset,
                                           out_row_major);
}
#endif

static iree_status_t
iree_hal_tt_tensor_storage_element_count(const tt_iree_ttex_tensor_desc_t &desc,
                                         uint64_t *out_element_count) {
  if (!out_element_count) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "out_element_count is null");
  }
  if (desc.rank == 0 || desc.rank > TT_IREE_TTEX_MAX_TENSOR_RANK) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "invalid tensor rank %u", desc.rank);
  }

  uint64_t element_count = 1;
  for (uint32_t i = 0; i < desc.rank; ++i) {
    if (desc.storage_shape[i] == 0 ||
        element_count >
            std::numeric_limits<uint64_t>::max() / desc.storage_shape[i]) {
      return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                              "tensor storage shape overflows");
    }
    element_count *= desc.storage_shape[i];
  }
  *out_element_count = element_count;
  return iree_ok_status();
}

static iree_status_t iree_hal_tt_resolve_tensor_buffer_ref(
    const tt_iree_ttex_tensor_desc_t &desc,
    const iree_hal_tt_dispatch_command_t &cmd,
    iree_hal_buffer_binding_table_t binding_table,
    iree_hal_buffer_ref_t *out_ref) {
  if (desc.binding >= cmd.binding_count) {
    return iree_make_status(
        IREE_STATUS_OUT_OF_RANGE,
        "tensor binding %u exceeds dispatch binding count %zu", desc.binding,
        cmd.binding_count);
  }

  iree_hal_buffer_ref_t ref = {};
  IREE_RETURN_IF_ERROR(iree_hal_tt_resolve_transfer_buffer_ref(
      binding_table, cmd.bindings[desc.binding], &ref));
  uint64_t element_count = 0;
  IREE_RETURN_IF_ERROR(
      iree_hal_tt_tensor_storage_element_count(desc, &element_count));
  if (element_count >
      std::numeric_limits<iree_device_size_t>::max() / sizeof(float)) {
    return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                            "F32 tensor storage byte size overflows");
  }
  iree_device_size_t byte_length =
      static_cast<iree_device_size_t>(element_count * sizeof(float));
  uint64_t byte_offset = desc.byte_offset;
  if (desc.byte_offset_constant) {
    size_t constant_offset =
        static_cast<size_t>(desc.byte_offset_constant - 1) * sizeof(uint32_t);
    if (constant_offset + sizeof(uint32_t) > cmd.constants_length) {
      return iree_make_status(
          IREE_STATUS_OUT_OF_RANGE,
          "tensor byte-offset constant ordinal %u exceeds %zu constant bytes",
          desc.byte_offset_constant - 1, cmd.constants_length);
    }
    uint32_t dynamic_offset = 0;
    std::memcpy(&dynamic_offset, cmd.constants + constant_offset,
                sizeof(dynamic_offset));
    if (byte_offset > std::numeric_limits<uint64_t>::max() - dynamic_offset) {
      return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                              "tensor byte offset overflows");
    }
    byte_offset += dynamic_offset;
  }
  if (byte_offset > ref.length || byte_length > ref.length - byte_offset) {
    return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                            "tensor storage range [offset=%" PRIu64
                            ", length=%" PRIdsz
                            "] exceeds resolved binding length %" PRIdsz,
                            byte_offset, byte_length, ref.length);
  }
  ref.offset += static_cast<iree_device_size_t>(byte_offset);
  ref.length = byte_length;
  *out_ref = ref;
  return iree_ok_status();
}

static iree_status_t
iree_hal_tt_read_buffer_f32(const iree_hal_buffer_ref_t &ref,
                            std::vector<float> *out_values) {
  if (!out_values || ref.length % sizeof(float) != 0) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "invalid F32 buffer read");
  }
  out_values->resize(ref.length / sizeof(float));
  iree_hal_buffer_mapping_t mapping = {};
  IREE_RETURN_IF_ERROR(iree_hal_buffer_map_range(
      ref.buffer, IREE_HAL_MAPPING_MODE_SCOPED, IREE_HAL_MEMORY_ACCESS_READ,
      ref.offset, ref.length, &mapping));
  std::memcpy(out_values->data(), mapping.contents.data,
              static_cast<size_t>(ref.length));
  return iree_hal_buffer_unmap_range(&mapping);
}

static iree_status_t
iree_hal_tt_write_buffer_f32(const iree_hal_buffer_ref_t &ref,
                             const std::vector<float> &values) {
  if (values.size() != ref.length / sizeof(float)) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "F32 buffer write size mismatch");
  }
  iree_hal_buffer_mapping_t mapping = {};
  IREE_RETURN_IF_ERROR(iree_hal_buffer_map_range(
      ref.buffer, IREE_HAL_MAPPING_MODE_SCOPED, IREE_HAL_MEMORY_ACCESS_WRITE,
      ref.offset, ref.length, &mapping));
  std::memcpy(mapping.contents.data, values.data(),
              static_cast<size_t>(ref.length));
  return iree_hal_buffer_unmap_range(&mapping);
}

static uint64_t iree_hal_tt_tensor_linear_index(
    const tt_iree_ttex_tensor_desc_t &desc,
    const std::array<uint32_t, TT_IREE_TTEX_MAX_TENSOR_RANK> &indices) {
  uint64_t linear_index = 0;
  for (uint32_t i = 0; i < desc.rank; ++i) {
    linear_index = linear_index * desc.storage_shape[i] + desc.offset[i] +
                   static_cast<uint64_t>(indices[i]) *
                       std::max(1u, desc.stride[i]);
  }
  return linear_index;
}

static iree_status_t iree_hal_tt_execute_f32_padded_copy_dispatch(
    const iree_hal_tt_kernel_params_t *params,
    const iree_hal_tt_dispatch_command_t &cmd,
    iree_hal_buffer_binding_table_t binding_table) {
  if (!params || cmd.binding_count != 2) {
    return iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "F32 padded copy requires kernel params and exactly two bindings");
  }

  const tt_iree_ttex_padded_copy_desc_t &copy = params->padded_copy;
  iree_hal_buffer_ref_t source_ref = {};
  iree_hal_buffer_ref_t destination_ref = {};
  IREE_RETURN_IF_ERROR(iree_hal_tt_resolve_tensor_buffer_ref(
      copy.source, cmd, binding_table, &source_ref));
  IREE_RETURN_IF_ERROR(iree_hal_tt_resolve_tensor_buffer_ref(
      copy.destination, cmd, binding_table, &destination_ref));

  std::vector<float> source_storage;
  std::vector<float> destination_storage;
  IREE_RETURN_IF_ERROR(
      iree_hal_tt_read_buffer_f32(source_ref, &source_storage));
  IREE_RETURN_IF_ERROR(
      iree_hal_tt_read_buffer_f32(destination_ref, &destination_storage));

  for (uint32_t c = 0; c < copy.source.shape[0]; ++c) {
    for (uint32_t h = 0; h < copy.source.shape[1]; ++h) {
      for (uint32_t w = 0; w < copy.source.shape[2]; ++w) {
        std::array<uint32_t, TT_IREE_TTEX_MAX_TENSOR_RANK> indices = {c, h, w,
                                                                      0};
        uint64_t source_index =
            iree_hal_tt_tensor_linear_index(copy.source, indices);
        uint64_t destination_index =
            iree_hal_tt_tensor_linear_index(copy.destination, indices);
        destination_storage[destination_index] = source_storage[source_index];
      }
    }
  }
  return iree_hal_tt_write_buffer_f32(destination_ref, destination_storage);
}

static iree_status_t iree_hal_tt_execute_f32_max_pool_dispatch(
    const iree_hal_tt_kernel_params_t *params,
    const iree_hal_tt_dispatch_command_t &cmd,
    iree_hal_buffer_binding_table_t binding_table) {
  if (!params || cmd.binding_count != 2) {
    return iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "F32 max-pool requires kernel params and exactly two bindings");
  }

  const tt_iree_ttex_max_pool_2d_desc_t &pool = params->max_pool;
  iree_hal_buffer_ref_t input_ref = {};
  iree_hal_buffer_ref_t output_ref = {};
  IREE_RETURN_IF_ERROR(iree_hal_tt_resolve_tensor_buffer_ref(
      pool.input, cmd, binding_table, &input_ref));
  IREE_RETURN_IF_ERROR(iree_hal_tt_resolve_tensor_buffer_ref(
      pool.output, cmd, binding_table, &output_ref));

  std::vector<float> input_storage;
  std::vector<float> output_storage;
  IREE_RETURN_IF_ERROR(iree_hal_tt_read_buffer_f32(input_ref, &input_storage));
  IREE_RETURN_IF_ERROR(
      iree_hal_tt_read_buffer_f32(output_ref, &output_storage));

  for (uint32_t c = 0; c < pool.output.shape[0]; ++c) {
    for (uint32_t oh = 0; oh < pool.output.shape[1]; ++oh) {
      for (uint32_t ow = 0; ow < pool.output.shape[2]; ++ow) {
        float value = -std::numeric_limits<float>::infinity();
        for (uint32_t kh = 0; kh < pool.kernel_h; ++kh) {
          for (uint32_t kw = 0; kw < pool.kernel_w; ++kw) {
            uint32_t ih = oh * pool.stride_h + kh * pool.dilation_h;
            uint32_t iw = ow * pool.stride_w + kw * pool.dilation_w;
            std::array<uint32_t, TT_IREE_TTEX_MAX_TENSOR_RANK> input_indices = {
                c, ih, iw, 0};
            float candidate = input_storage[iree_hal_tt_tensor_linear_index(
                pool.input, input_indices)];
            if (std::isnan(value) || std::isnan(candidate)) {
              value = std::numeric_limits<float>::quiet_NaN();
            } else {
              value = std::fmax(value, candidate);
            }
          }
        }
        std::array<uint32_t, TT_IREE_TTEX_MAX_TENSOR_RANK> output_indices = {
            c, oh, ow, 0};
        output_storage[iree_hal_tt_tensor_linear_index(pool.output,
                                                       output_indices)] = value;
      }
    }
  }
  return iree_hal_tt_write_buffer_f32(output_ref, output_storage);
}

static iree_status_t iree_hal_tt_execute_f32_global_sum_dispatch(
    const iree_hal_tt_kernel_params_t *params,
    const iree_hal_tt_dispatch_command_t &cmd,
    iree_hal_buffer_binding_table_t binding_table) {
  const tt_iree_ttex_global_sum_desc_t &reduction = params->global_sum;
  iree_hal_buffer_ref_t input_ref = {};
  iree_hal_buffer_ref_t sum_ref = {};
  iree_hal_buffer_ref_t divisor_ref = {};
  IREE_RETURN_IF_ERROR(iree_hal_tt_resolve_tensor_buffer_ref(
      reduction.input, cmd, binding_table, &input_ref));
  IREE_RETURN_IF_ERROR(iree_hal_tt_resolve_tensor_buffer_ref(
      reduction.sum, cmd, binding_table, &sum_ref));
  IREE_RETURN_IF_ERROR(iree_hal_tt_resolve_tensor_buffer_ref(
      reduction.divisor, cmd, binding_table, &divisor_ref));
  std::vector<float> input_storage;
  std::vector<float> sum_storage;
  std::vector<float> divisor_storage;
  IREE_RETURN_IF_ERROR(iree_hal_tt_read_buffer_f32(input_ref, &input_storage));
  IREE_RETURN_IF_ERROR(iree_hal_tt_read_buffer_f32(sum_ref, &sum_storage));
  IREE_RETURN_IF_ERROR(
      iree_hal_tt_read_buffer_f32(divisor_ref, &divisor_storage));
  for (uint32_t c = 0; c < reduction.input.shape[0]; ++c) {
    float sum = 0.0f;
    for (uint32_t h = 0; h < reduction.reduce_h; ++h) {
      for (uint32_t w = 0; w < reduction.reduce_w; ++w) {
        std::array<uint32_t, TT_IREE_TTEX_MAX_TENSOR_RANK> indices = {c, h, w,
                                                                      0};
        sum += input_storage[
            iree_hal_tt_tensor_linear_index(reduction.input, indices)];
      }
    }
    std::array<uint32_t, TT_IREE_TTEX_MAX_TENSOR_RANK> index = {c, 0, 0, 0};
    sum_storage[iree_hal_tt_tensor_linear_index(reduction.sum, index)] = sum;
  }
  std::array<uint32_t, TT_IREE_TTEX_MAX_TENSOR_RANK> scalar_index = {};
  divisor_storage[iree_hal_tt_tensor_linear_index(reduction.divisor,
                                                   scalar_index)] =
      reduction.divisor_value;
  IREE_RETURN_IF_ERROR(iree_hal_tt_write_buffer_f32(sum_ref, sum_storage));
  return iree_hal_tt_write_buffer_f32(divisor_ref, divisor_storage);
}

static iree_status_t iree_hal_tt_execute_f32_scalar_div_dispatch(
    const iree_hal_tt_kernel_params_t *params,
    const iree_hal_tt_dispatch_command_t &cmd,
    iree_hal_buffer_binding_table_t binding_table) {
  const tt_iree_ttex_scalar_div_desc_t &divide = params->scalar_div;
  iree_hal_buffer_ref_t input_ref = {};
  iree_hal_buffer_ref_t divisor_ref = {};
  iree_hal_buffer_ref_t output_ref = {};
  IREE_RETURN_IF_ERROR(iree_hal_tt_resolve_tensor_buffer_ref(
      divide.input, cmd, binding_table, &input_ref));
  IREE_RETURN_IF_ERROR(iree_hal_tt_resolve_tensor_buffer_ref(
      divide.divisor, cmd, binding_table, &divisor_ref));
  IREE_RETURN_IF_ERROR(iree_hal_tt_resolve_tensor_buffer_ref(
      divide.output, cmd, binding_table, &output_ref));
  std::vector<float> input_storage;
  std::vector<float> divisor_storage;
  std::vector<float> output_storage;
  IREE_RETURN_IF_ERROR(iree_hal_tt_read_buffer_f32(input_ref, &input_storage));
  IREE_RETURN_IF_ERROR(
      iree_hal_tt_read_buffer_f32(divisor_ref, &divisor_storage));
  IREE_RETURN_IF_ERROR(
      iree_hal_tt_read_buffer_f32(output_ref, &output_storage));
  std::array<uint32_t, TT_IREE_TTEX_MAX_TENSOR_RANK> scalar_index = {};
  float divisor = divisor_storage[
      iree_hal_tt_tensor_linear_index(divide.divisor, scalar_index)];
  for (uint32_t i = 0; i < divide.input.shape[0]; ++i) {
    std::array<uint32_t, TT_IREE_TTEX_MAX_TENSOR_RANK> index = {i, 0, 0, 0};
    output_storage[iree_hal_tt_tensor_linear_index(divide.output, index)] =
        input_storage[iree_hal_tt_tensor_linear_index(divide.input, index)] /
        divisor;
  }
  return iree_hal_tt_write_buffer_f32(output_ref, output_storage);
}

static iree_status_t iree_hal_tt_execute_f32_linear_dispatch(
    iree_hal_tt_device_t *device, const iree_hal_tt_kernel_params_t *params,
    const iree_hal_tt_dispatch_command_t &cmd,
    iree_hal_buffer_binding_table_t binding_table) {
  const tt_iree_ttex_linear_desc_t &linear = params->linear;
  iree_hal_buffer_ref_t input_ref = {};
  iree_hal_buffer_ref_t weights_ref = {};
  iree_hal_buffer_ref_t bias_ref = {};
  iree_hal_buffer_ref_t output_ref = {};
  IREE_RETURN_IF_ERROR(iree_hal_tt_resolve_tensor_buffer_ref(
      linear.input, cmd, binding_table, &input_ref));
  IREE_RETURN_IF_ERROR(iree_hal_tt_resolve_tensor_buffer_ref(
      linear.weights, cmd, binding_table, &weights_ref));
  IREE_RETURN_IF_ERROR(iree_hal_tt_resolve_tensor_buffer_ref(
      linear.bias, cmd, binding_table, &bias_ref));
  IREE_RETURN_IF_ERROR(iree_hal_tt_resolve_tensor_buffer_ref(
      linear.output, cmd, binding_table, &output_ref));
  std::vector<float> input_storage;
  std::vector<float> weights_storage;
  std::vector<float> bias_storage;
  std::vector<float> output_storage;
  IREE_RETURN_IF_ERROR(iree_hal_tt_read_buffer_f32(input_ref, &input_storage));
  IREE_RETURN_IF_ERROR(
      iree_hal_tt_read_buffer_f32(weights_ref, &weights_storage));
  IREE_RETURN_IF_ERROR(iree_hal_tt_read_buffer_f32(bias_ref, &bias_storage));
  IREE_RETURN_IF_ERROR(
      iree_hal_tt_read_buffer_f32(output_ref, &output_storage));

  uint32_t padded_m = (linear.m + TT_TILE_HEIGHT - 1) / TT_TILE_HEIGHT *
                      TT_TILE_HEIGHT;
  uint32_t padded_n =
      (linear.n + TT_TILE_WIDTH - 1) / TT_TILE_WIDTH * TT_TILE_WIDTH;
  uint32_t padded_k =
      (linear.k + TT_TILE_WIDTH - 1) / TT_TILE_WIDTH * TT_TILE_WIDTH;
  std::vector<float> lhs(static_cast<size_t>(padded_m) * padded_k, 0.0f);
  std::vector<float> rhs(static_cast<size_t>(padded_k) * padded_n, 0.0f);
  std::vector<float> result(static_cast<size_t>(padded_m) * padded_n, 0.0f);
  for (uint32_t m = 0; m < linear.m; ++m) {
    for (uint32_t k = 0; k < linear.k; ++k) {
      std::array<uint32_t, TT_IREE_TTEX_MAX_TENSOR_RANK> index = {m, k, 0, 0};
      lhs[static_cast<size_t>(m) * padded_k + k] =
          input_storage[iree_hal_tt_tensor_linear_index(linear.input, index)];
    }
  }
  for (uint32_t n = 0; n < linear.n; ++n) {
    for (uint32_t k = 0; k < linear.k; ++k) {
      std::array<uint32_t, TT_IREE_TTEX_MAX_TENSOR_RANK> index = {n, k, 0, 0};
      rhs[static_cast<size_t>(k) * padded_n + n] =
          weights_storage[
              iree_hal_tt_tensor_linear_index(linear.weights, index)];
    }
  }
#ifdef TT_IREE_ENABLE_MOCK
  (void)device;
  for (uint32_t m = 0; m < linear.m; ++m) {
    for (uint32_t n = 0; n < linear.n; ++n) {
      for (uint32_t k = 0; k < linear.k; ++k) {
        result[static_cast<size_t>(m) * padded_n + n] +=
            lhs[static_cast<size_t>(m) * padded_k + k] *
            rhs[static_cast<size_t>(k) * padded_n + n];
      }
    }
  }
#else
  std::vector<bfloat16> lhs_bfloat16;
  std::vector<bfloat16> rhs_bfloat16;
  lhs_bfloat16.reserve(lhs.size());
  rhs_bfloat16.reserve(rhs.size());
  for (float value : lhs) lhs_bfloat16.emplace_back(value);
  for (float value : rhs) rhs_bfloat16.emplace_back(value);
  std::vector<bfloat16> result_bfloat16;
  IREE_RETURN_IF_ERROR(iree_hal_tt_execute_tiled_matmul_host_matrices(
      device, params, cmd.config, padded_m / TT_TILE_HEIGHT,
      padded_n / TT_TILE_WIDTH, padded_k / TT_TILE_WIDTH, lhs_bfloat16,
      rhs_bfloat16, &result_bfloat16));
  std::transform(result_bfloat16.begin(), result_bfloat16.end(), result.begin(),
                 [](bfloat16 value) { return static_cast<float>(value); });
#endif
  for (uint32_t m = 0; m < linear.m; ++m) {
    for (uint32_t n = 0; n < linear.n; ++n) {
      std::array<uint32_t, TT_IREE_TTEX_MAX_TENSOR_RANK> index = {m, n, 0, 0};
      output_storage[
          iree_hal_tt_tensor_linear_index(linear.output, index)] =
          result[static_cast<size_t>(m) * padded_n + n] +
          bias_storage[iree_hal_tt_tensor_linear_index(linear.bias, index)];
    }
  }
  return iree_hal_tt_write_buffer_f32(output_ref, output_storage);
}

static iree_status_t iree_hal_tt_gather_f32_matmul_inputs(
    const tt_iree_ttex_matmul_like_desc_t &matmul,
    const std::vector<float> &lhs_storage,
    const std::vector<float> &rhs_storage, std::vector<float> *out_lhs,
    std::vector<float> *out_rhs) {
  if (!out_lhs || !out_rhs) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "matmul gather outputs are null");
  }
  const bool chw_lhs = matmul.lhs.rank == 3 &&
                       matmul.lhs.layout == TT_IREE_TTEX_TENSOR_LAYOUT_CHW;
  const bool channel_major_lhs =
      matmul.lhs.rank == 2 &&
      matmul.lhs.layout == TT_IREE_TTEX_TENSOR_LAYOUT_CHANNEL_MAJOR;
  if ((!chw_lhs && !channel_major_lhs) || matmul.rhs.rank != 2 ||
      matmul.rhs.layout != TT_IREE_TTEX_TENSOR_LAYOUT_OI) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "unsupported F32 matmul tensor layout");
  }

  out_lhs->resize(static_cast<size_t>(matmul.m) * matmul.k);
  out_rhs->resize(static_cast<size_t>(matmul.k) * matmul.n);
  for (uint32_t m = 0; m < matmul.m; ++m) {
    for (uint32_t k = 0; k < matmul.k; ++k) {
      std::array<uint32_t, TT_IREE_TTEX_MAX_TENSOR_RANK> indices = {};
      indices[0] = k;
      if (chw_lhs) {
        indices[1] = m / matmul.lhs.shape[2];
        indices[2] = m % matmul.lhs.shape[2];
      } else {
        indices[1] = m;
      }
      (*out_lhs)[static_cast<size_t>(m) * matmul.k + k] =
          lhs_storage[iree_hal_tt_tensor_linear_index(matmul.lhs, indices)];
    }
  }
  for (uint32_t k = 0; k < matmul.k; ++k) {
    for (uint32_t n = 0; n < matmul.n; ++n) {
      std::array<uint32_t, TT_IREE_TTEX_MAX_TENSOR_RANK> indices = {n, k};
      (*out_rhs)[static_cast<size_t>(k) * matmul.n + n] =
          rhs_storage[iree_hal_tt_tensor_linear_index(matmul.rhs, indices)];
    }
  }
  return iree_ok_status();
}

static iree_status_t iree_hal_tt_scatter_f32_matmul_output(
    const tt_iree_ttex_matmul_like_desc_t &matmul,
    const std::vector<float> &matrix, std::vector<float> *output_storage) {
  if (!output_storage ||
      matrix.size() != static_cast<size_t>(matmul.m) * matmul.n) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "invalid F32 matmul output matrix");
  }
  const bool chw_output =
      matmul.output.rank == 3 &&
      matmul.output.layout == TT_IREE_TTEX_TENSOR_LAYOUT_CHW;
  const bool channel_major_output =
      matmul.output.rank == 2 &&
      matmul.output.layout == TT_IREE_TTEX_TENSOR_LAYOUT_CHANNEL_MAJOR;
  if (!chw_output && !channel_major_output) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "unsupported F32 matmul output layout");
  }

  for (uint32_t n = 0; n < matmul.n; ++n) {
    for (uint32_t m = 0; m < matmul.m; ++m) {
      float value = matrix[static_cast<size_t>(m) * matmul.n + n];
      std::array<uint32_t, TT_IREE_TTEX_MAX_TENSOR_RANK> indices = {};
      indices[0] = n;
      if (chw_output) {
        indices[1] = m / matmul.output.shape[2];
        indices[2] = m % matmul.output.shape[2];
      } else {
        indices[1] = m;
      }
      (*output_storage)[iree_hal_tt_tensor_linear_index(matmul.output,
                                                        indices)] = value;
    }
  }
  return iree_ok_status();
}

static iree_status_t iree_hal_tt_read_logical_f32_vector(
    const tt_iree_ttex_tensor_desc_t &desc,
    const iree_hal_tt_dispatch_command_t &cmd,
    iree_hal_buffer_binding_table_t binding_table,
    std::vector<float> *out_values) {
  iree_hal_buffer_ref_t ref = {};
  IREE_RETURN_IF_ERROR(
      iree_hal_tt_resolve_tensor_buffer_ref(desc, cmd, binding_table, &ref));
  std::vector<float> storage;
  IREE_RETURN_IF_ERROR(iree_hal_tt_read_buffer_f32(ref, &storage));
  out_values->resize(desc.shape[0]);
  for (uint32_t i = 0; i < desc.shape[0]; ++i) {
    std::array<uint32_t, TT_IREE_TTEX_MAX_TENSOR_RANK> index = {i, 0, 0, 0};
    (*out_values)[i] =
        storage[iree_hal_tt_tensor_linear_index(desc, index)];
  }
  return iree_ok_status();
}

static iree_status_t iree_hal_tt_execute_f32_matmul_like_dispatch(
    iree_hal_tt_device_t *device, const iree_hal_tt_kernel_params_t *params,
    const iree_hal_tt_dispatch_command_t &cmd,
    iree_hal_buffer_binding_table_t binding_table) {
  const tt_iree_ttex_matmul_like_desc_t &matmul = params->matmul_like;
  iree_hal_buffer_ref_t lhs_ref = {};
  iree_hal_buffer_ref_t rhs_ref = {};
  iree_hal_buffer_ref_t output_ref = {};
  IREE_RETURN_IF_ERROR(iree_hal_tt_resolve_tensor_buffer_ref(
      matmul.lhs, cmd, binding_table, &lhs_ref));
  IREE_RETURN_IF_ERROR(iree_hal_tt_resolve_tensor_buffer_ref(
      matmul.rhs, cmd, binding_table, &rhs_ref));
  IREE_RETURN_IF_ERROR(iree_hal_tt_resolve_tensor_buffer_ref(
      matmul.output, cmd, binding_table, &output_ref));

  std::vector<float> lhs_storage;
  std::vector<float> rhs_storage;
  std::vector<float> output_storage;
  IREE_RETURN_IF_ERROR(iree_hal_tt_read_buffer_f32(lhs_ref, &lhs_storage));
  IREE_RETURN_IF_ERROR(iree_hal_tt_read_buffer_f32(rhs_ref, &rhs_storage));
  IREE_RETURN_IF_ERROR(
      iree_hal_tt_read_buffer_f32(output_ref, &output_storage));

  std::vector<float> lhs_row_major;
  std::vector<float> rhs_row_major;
  std::vector<float> output_row_major(static_cast<size_t>(matmul.m) * matmul.n,
                                      0.0f);
  IREE_RETURN_IF_ERROR(iree_hal_tt_gather_f32_matmul_inputs(
      matmul, lhs_storage, rhs_storage, &lhs_row_major, &rhs_row_major));

  uint32_t padded_m = (matmul.m + TT_TILE_HEIGHT - 1) / TT_TILE_HEIGHT *
                      TT_TILE_HEIGHT;
  uint32_t padded_n =
      (matmul.n + TT_TILE_WIDTH - 1) / TT_TILE_WIDTH * TT_TILE_WIDTH;
  uint32_t padded_k =
      (matmul.k + TT_TILE_WIDTH - 1) / TT_TILE_WIDTH * TT_TILE_WIDTH;
  std::vector<float> padded_lhs(static_cast<size_t>(padded_m) * padded_k, 0.0f);
  std::vector<float> padded_rhs(static_cast<size_t>(padded_k) * padded_n, 0.0f);
  for (uint32_t m = 0; m < matmul.m; ++m) {
    std::copy_n(lhs_row_major.data() + static_cast<size_t>(m) * matmul.k,
                matmul.k,
                padded_lhs.data() + static_cast<size_t>(m) * padded_k);
  }
  for (uint32_t k = 0; k < matmul.k; ++k) {
    std::copy_n(rhs_row_major.data() + static_cast<size_t>(k) * matmul.n,
                matmul.n,
                padded_rhs.data() + static_cast<size_t>(k) * padded_n);
  }

#ifdef TT_IREE_ENABLE_MOCK
  (void)device;
  for (uint32_t m = 0; m < matmul.m; ++m) {
    for (uint32_t n = 0; n < matmul.n; ++n) {
      float value = 0.0f;
      for (uint32_t k = 0; k < matmul.k; ++k) {
        value += lhs_row_major[static_cast<size_t>(m) * matmul.k + k] *
                 rhs_row_major[static_cast<size_t>(k) * matmul.n + n];
      }
      output_row_major[static_cast<size_t>(m) * matmul.n + n] = value;
    }
  }
#else
  std::vector<bfloat16> lhs_bfloat16;
  std::vector<bfloat16> rhs_bfloat16;
  lhs_bfloat16.reserve(padded_lhs.size());
  rhs_bfloat16.reserve(padded_rhs.size());
  for (float value : padded_lhs)
    lhs_bfloat16.emplace_back(value);
  for (float value : padded_rhs)
    rhs_bfloat16.emplace_back(value);

  std::vector<bfloat16> output_bfloat16;
  IREE_RETURN_IF_ERROR(iree_hal_tt_execute_tiled_matmul_host_matrices(
      device, params, cmd.config, padded_m / TT_TILE_HEIGHT,
      padded_n / TT_TILE_WIDTH, padded_k / TT_TILE_WIDTH, lhs_bfloat16,
      rhs_bfloat16, &output_bfloat16));
  for (uint32_t m = 0; m < matmul.m; ++m) {
    for (uint32_t n = 0; n < matmul.n; ++n) {
      output_row_major[static_cast<size_t>(m) * matmul.n + n] =
          static_cast<float>(
              output_bfloat16[static_cast<size_t>(m) * padded_n + n]);
    }
  }
#endif

  std::vector<float> main_mean;
  std::vector<float> main_scale;
  std::vector<float> main_bias;
  bool runtime_affine =
      matmul.epilogue_flags & TT_IREE_TTEX_EPILOGUE_RUNTIME_AFFINE;
  if (runtime_affine) {
    IREE_RETURN_IF_ERROR(iree_hal_tt_read_logical_f32_vector(
        matmul.epilogue_mean, cmd, binding_table, &main_mean));
    IREE_RETURN_IF_ERROR(iree_hal_tt_read_logical_f32_vector(
        matmul.epilogue_channel_scale, cmd, binding_table, &main_scale));
    IREE_RETURN_IF_ERROR(iree_hal_tt_read_logical_f32_vector(
        matmul.epilogue_channel_bias, cmd, binding_table, &main_bias));
  }

  std::vector<float> residual_storage;
  std::vector<float> residual_mean;
  std::vector<float> residual_scale;
  std::vector<float> residual_bias;
  bool has_residual =
      matmul.epilogue_flags & TT_IREE_TTEX_EPILOGUE_RESIDUAL;
  bool residual_per_channel =
      matmul.epilogue_flags & TT_IREE_TTEX_EPILOGUE_RESIDUAL_PER_CHANNEL;
  if (has_residual) {
    iree_hal_buffer_ref_t residual_ref = {};
    IREE_RETURN_IF_ERROR(iree_hal_tt_resolve_tensor_buffer_ref(
        matmul.residual, cmd, binding_table, &residual_ref));
    IREE_RETURN_IF_ERROR(
        iree_hal_tt_read_buffer_f32(residual_ref, &residual_storage));
  }
  if (residual_per_channel) {
    IREE_RETURN_IF_ERROR(iree_hal_tt_read_logical_f32_vector(
        matmul.residual_mean, cmd, binding_table, &residual_mean));
    IREE_RETURN_IF_ERROR(iree_hal_tt_read_logical_f32_vector(
        matmul.residual_channel_scale, cmd, binding_table, &residual_scale));
    IREE_RETURN_IF_ERROR(iree_hal_tt_read_logical_f32_vector(
        matmul.residual_channel_bias, cmd, binding_table, &residual_bias));
  }

  bool chw_output = matmul.output.rank == 3;
  for (uint32_t n = 0; n < matmul.n; ++n) {
    float scale = matmul.epilogue_scale;
    float bias = matmul.epilogue_bias;
    if (runtime_affine) {
      scale = matmul.epilogue_common_scale * main_scale[n];
      bias = main_bias[n] - main_mean[n] * scale;
    } else if (matmul.epilogue_flags & TT_IREE_TTEX_EPILOGUE_PER_CHANNEL) {
      scale = matmul.epilogue_scales[n];
      bias = matmul.epilogue_biases[n];
    }
    float residual_channel_scale = matmul.residual_scale;
    float residual_channel_bias = matmul.residual_bias;
    if (residual_per_channel) {
      residual_channel_scale =
          matmul.residual_common_scale * residual_scale[n];
      residual_channel_bias =
          residual_bias[n] - residual_mean[n] * residual_channel_scale;
    }
    for (uint32_t m = 0; m < matmul.m; ++m) {
      size_t index = static_cast<size_t>(m) * matmul.n + n;
      float value = output_row_major[index];
      if (matmul.epilogue_flags & TT_IREE_TTEX_EPILOGUE_AFFINE)
        value = value * scale + bias;
      if (has_residual) {
        std::array<uint32_t, TT_IREE_TTEX_MAX_TENSOR_RANK> tensor_index = {};
        tensor_index[0] = n;
        if (chw_output) {
          tensor_index[1] = m / matmul.output.shape[2];
          tensor_index[2] = m % matmul.output.shape[2];
        } else {
          tensor_index[1] = m;
        }
        float residual_value = residual_storage[
            iree_hal_tt_tensor_linear_index(matmul.residual, tensor_index)];
        value += residual_value * residual_channel_scale +
                 residual_channel_bias;
      }
      if (matmul.epilogue_flags & TT_IREE_TTEX_EPILOGUE_RELU)
        value = std::max(value, 0.0f);
      output_row_major[index] = value;
    }
  }

  IREE_RETURN_IF_ERROR(iree_hal_tt_scatter_f32_matmul_output(
      matmul, output_row_major, &output_storage));
  return iree_hal_tt_write_buffer_f32(output_ref, output_storage);
}

static iree_status_t iree_hal_tt_execute_f32_conv2d_dispatch(
    iree_hal_tt_device_t *device, const iree_hal_tt_kernel_params_t *params,
    const iree_hal_tt_dispatch_command_t &cmd,
    iree_hal_buffer_binding_table_t binding_table) {
  if (!params || cmd.binding_count != 3) {
    return iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "F32 convolution requires kernel params and exactly three bindings");
  }
  const tt_iree_ttex_conv2d_desc_t &conv = params->conv2d;

  iree_hal_buffer_ref_t input_ref = {};
  iree_hal_buffer_ref_t filter_ref = {};
  iree_hal_buffer_ref_t output_ref = {};
  IREE_RETURN_IF_ERROR(iree_hal_tt_resolve_tensor_buffer_ref(
      conv.input, cmd, binding_table, &input_ref));
  IREE_RETURN_IF_ERROR(iree_hal_tt_resolve_tensor_buffer_ref(
      conv.filter, cmd, binding_table, &filter_ref));
  IREE_RETURN_IF_ERROR(iree_hal_tt_resolve_tensor_buffer_ref(
      conv.output, cmd, binding_table, &output_ref));

  std::vector<float> input_storage;
  std::vector<float> filter_storage;
  std::vector<float> output_storage;
  IREE_RETURN_IF_ERROR(iree_hal_tt_read_buffer_f32(input_ref, &input_storage));
  IREE_RETURN_IF_ERROR(
      iree_hal_tt_read_buffer_f32(filter_ref, &filter_storage));
  IREE_RETURN_IF_ERROR(
      iree_hal_tt_read_buffer_f32(output_ref, &output_storage));

  std::vector<float> affine_mean;
  std::vector<float> affine_scale;
  std::vector<float> affine_bias;
  bool runtime_affine =
      conv.epilogue_flags & TT_IREE_TTEX_EPILOGUE_RUNTIME_AFFINE;
  if (runtime_affine) {
    IREE_RETURN_IF_ERROR(iree_hal_tt_read_logical_f32_vector(
        conv.epilogue_mean, cmd, binding_table, &affine_mean));
    IREE_RETURN_IF_ERROR(iree_hal_tt_read_logical_f32_vector(
        conv.epilogue_channel_scale, cmd, binding_table, &affine_scale));
    IREE_RETURN_IF_ERROR(iree_hal_tt_read_logical_f32_vector(
        conv.epilogue_channel_bias, cmd, binding_table, &affine_bias));
  }

  uint64_t logical_m =
      static_cast<uint64_t>(conv.output.shape[1]) * conv.output.shape[2];
  uint64_t logical_n = conv.output.shape[0];
  uint64_t logical_k = static_cast<uint64_t>(conv.input.shape[0]) *
                       conv.kernel_h * conv.kernel_w;
  uint64_t padded_m =
      (logical_m + TT_TILE_HEIGHT - 1) / TT_TILE_HEIGHT * TT_TILE_HEIGHT;
  uint64_t padded_n =
      (logical_n + TT_TILE_WIDTH - 1) / TT_TILE_WIDTH * TT_TILE_WIDTH;
  uint64_t padded_k =
      (logical_k + TT_TILE_WIDTH - 1) / TT_TILE_WIDTH * TT_TILE_WIDTH;
  if (padded_m > std::numeric_limits<uint32_t>::max() ||
      padded_n > std::numeric_limits<uint32_t>::max() ||
      padded_k > std::numeric_limits<uint32_t>::max() ||
      padded_m > std::numeric_limits<size_t>::max() / padded_k ||
      padded_k > std::numeric_limits<size_t>::max() / padded_n ||
      padded_m > std::numeric_limits<size_t>::max() / padded_n) {
    return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                            "F32 convolution staging matrix size overflows");
  }

  std::vector<float> lhs_row_major(static_cast<size_t>(padded_m * padded_k),
                                   0.0f);
  std::vector<float> rhs_row_major(static_cast<size_t>(padded_k * padded_n),
                                   0.0f);
  std::vector<float> output_row_major(static_cast<size_t>(padded_m * padded_n),
                                      0.0f);

  for (uint32_t oh = 0; oh < conv.output.shape[1]; ++oh) {
    for (uint32_t ow = 0; ow < conv.output.shape[2]; ++ow) {
      uint64_t m = static_cast<uint64_t>(oh) * conv.output.shape[2] + ow;
      for (uint32_t ic = 0; ic < conv.input.shape[0]; ++ic) {
        for (uint32_t kh = 0; kh < conv.kernel_h; ++kh) {
          for (uint32_t kw = 0; kw < conv.kernel_w; ++kw) {
            uint32_t ih = oh * conv.stride_h + kh * conv.dilation_h;
            uint32_t iw = ow * conv.stride_w + kw * conv.dilation_w;
            uint64_t k = (static_cast<uint64_t>(ic) * conv.kernel_h + kh) *
                             conv.kernel_w +
                         kw;
            std::array<uint32_t, TT_IREE_TTEX_MAX_TENSOR_RANK> indices = {
                ic, ih, iw, 0};
            lhs_row_major[m * padded_k + k] =
                input_storage[iree_hal_tt_tensor_linear_index(conv.input,
                                                              indices)];
          }
        }
      }
    }
  }
  for (uint32_t oc = 0; oc < conv.output.shape[0]; ++oc) {
    for (uint32_t ic = 0; ic < conv.input.shape[0]; ++ic) {
      for (uint32_t kh = 0; kh < conv.kernel_h; ++kh) {
        for (uint32_t kw = 0; kw < conv.kernel_w; ++kw) {
          uint64_t k =
              (static_cast<uint64_t>(ic) * conv.kernel_h + kh) * conv.kernel_w +
              kw;
          std::array<uint32_t, TT_IREE_TTEX_MAX_TENSOR_RANK> indices = {oc, ic,
                                                                        kh, kw};
          rhs_row_major[k * padded_n + oc] =
              filter_storage[iree_hal_tt_tensor_linear_index(conv.filter,
                                                             indices)];
        }
      }
    }
  }

#ifdef TT_IREE_ENABLE_MOCK
  (void)device;
  for (uint64_t m = 0; m < logical_m; ++m) {
    for (uint64_t n = 0; n < logical_n; ++n) {
      float value = 0.0f;
      for (uint64_t k = 0; k < logical_k; ++k) {
        value +=
            lhs_row_major[m * padded_k + k] * rhs_row_major[k * padded_n + n];
      }
      output_row_major[m * padded_n + n] = value;
    }
  }
#else
  std::vector<bfloat16> lhs_bfloat16;
  std::vector<bfloat16> rhs_bfloat16;
  lhs_bfloat16.reserve(lhs_row_major.size());
  rhs_bfloat16.reserve(rhs_row_major.size());
  for (float value : lhs_row_major)
    lhs_bfloat16.emplace_back(value);
  for (float value : rhs_row_major)
    rhs_bfloat16.emplace_back(value);

  std::vector<bfloat16> output_bfloat16;
  IREE_RETURN_IF_ERROR(iree_hal_tt_execute_tiled_matmul_host_matrices(
      device, params, cmd.config,
      static_cast<uint32_t>(padded_m / TT_TILE_HEIGHT),
      static_cast<uint32_t>(padded_n / TT_TILE_WIDTH),
      static_cast<uint32_t>(padded_k / TT_TILE_WIDTH), lhs_bfloat16,
      rhs_bfloat16, &output_bfloat16));
  std::transform(output_bfloat16.begin(), output_bfloat16.end(),
                 output_row_major.begin(),
                 [](bfloat16 value) { return static_cast<float>(value); });
#endif

  bool per_channel = conv.epilogue_flags & TT_IREE_TTEX_EPILOGUE_PER_CHANNEL;
  for (uint32_t oc = 0; oc < conv.output.shape[0]; ++oc) {
    float scale = conv.epilogue_scale;
    float bias = conv.epilogue_bias;
    if (runtime_affine) {
      scale = conv.epilogue_common_scale * affine_scale[oc];
      bias = affine_bias[oc] - affine_mean[oc] * scale;
    } else if (per_channel) {
      scale = conv.epilogue_scales[oc];
      bias = conv.epilogue_biases[oc];
    }
    for (uint32_t oh = 0; oh < conv.output.shape[1]; ++oh) {
      for (uint32_t ow = 0; ow < conv.output.shape[2]; ++ow) {
        uint64_t m = static_cast<uint64_t>(oh) * conv.output.shape[2] + ow;
        float value = output_row_major[m * padded_n + oc];
        if (conv.epilogue_flags & TT_IREE_TTEX_EPILOGUE_AFFINE)
          value = value * scale + bias;
        if (conv.epilogue_flags & TT_IREE_TTEX_EPILOGUE_RELU)
          value = std::max(value, 0.0f);
        std::array<uint32_t, TT_IREE_TTEX_MAX_TENSOR_RANK> indices = {oc, oh,
                                                                      ow, 0};
        output_storage[iree_hal_tt_tensor_linear_index(conv.output, indices)] =
            value;
      }
    }
  }
  return iree_hal_tt_write_buffer_f32(output_ref, output_storage);
}

// Forward declarations of vtable functions
static void iree_hal_tt_device_destroy(iree_hal_device_t *base);
static iree_string_view_t iree_hal_tt_device_id(iree_hal_device_t *base);
static iree_allocator_t
iree_hal_tt_device_host_allocator(iree_hal_device_t *base);
static iree_hal_allocator_t *
iree_hal_tt_device_allocator(iree_hal_device_t *base);
static void iree_hal_tt_device_replace_allocator(iree_hal_device_t *,
                                                 iree_hal_allocator_t *);
static void
iree_hal_tt_device_replace_channel_provider(iree_hal_device_t *,
                                            iree_hal_channel_provider_t *);
static iree_status_t iree_hal_tt_device_trim(iree_hal_device_t *);
static iree_status_t iree_hal_tt_device_query_i64(iree_hal_device_t *,
                                                  iree_string_view_t,
                                                  iree_string_view_t,
                                                  int64_t *);
static iree_status_t
iree_hal_tt_device_query_capabilities(iree_hal_device_t *,
                                      iree_hal_device_capabilities_t *);
static const iree_hal_device_topology_info_t *
iree_hal_tt_device_topology_info(iree_hal_device_t *);
static iree_status_t iree_hal_tt_device_refine_topology_edge(
    iree_hal_device_t *, iree_hal_device_t *, iree_hal_topology_edge_t *);
static iree_status_t iree_hal_tt_device_assign_topology_info(
    iree_hal_device_t *, const iree_hal_device_topology_info_t *);
static iree_status_t iree_hal_tt_device_create_channel(
    iree_hal_device_t *, iree_hal_queue_affinity_t, iree_hal_channel_params_t,
    iree_hal_channel_t **);
// Implemented in tt_command_buffer.cc
extern iree_status_t iree_hal_tt_device_create_command_buffer(
    iree_hal_device_t *, iree_hal_command_buffer_mode_t,
    iree_hal_command_category_t, iree_hal_queue_affinity_t, iree_host_size_t,
    iree_hal_command_buffer_t **);
static iree_status_t iree_hal_tt_device_create_event(iree_hal_device_t *,
                                                     iree_hal_queue_affinity_t,
                                                     iree_hal_event_flags_t,
                                                     iree_hal_event_t **);
static iree_status_t
iree_hal_tt_device_create_executable_cache(iree_hal_device_t *,
                                           iree_string_view_t, iree_loop_t,
                                           iree_hal_executable_cache_t **);
static iree_status_t iree_hal_tt_device_import_file(
    iree_hal_device_t *, iree_hal_queue_affinity_t, iree_hal_memory_access_t,
    iree_io_file_handle_t *, iree_hal_external_file_flags_t,
    iree_hal_file_t **);
static iree_status_t iree_hal_tt_device_create_semaphore(
    iree_hal_device_t *, iree_hal_queue_affinity_t, uint64_t,
    iree_hal_semaphore_flags_t, iree_hal_semaphore_t **);
static iree_hal_semaphore_compatibility_t
iree_hal_tt_device_query_semaphore_compatibility(iree_hal_device_t *,
                                                 iree_hal_semaphore_t *);
static iree_status_t iree_hal_tt_device_queue_alloca(
    iree_hal_device_t *, iree_hal_queue_affinity_t,
    const iree_hal_semaphore_list_t, const iree_hal_semaphore_list_t,
    iree_hal_allocator_pool_t, iree_hal_buffer_params_t, iree_device_size_t,
    iree_hal_alloca_flags_t, iree_hal_buffer_t **);
static iree_status_t iree_hal_tt_device_queue_dealloca(
    iree_hal_device_t *, iree_hal_queue_affinity_t,
    const iree_hal_semaphore_list_t, const iree_hal_semaphore_list_t,
    iree_hal_buffer_t *, iree_hal_dealloca_flags_t);
static iree_status_t iree_hal_tt_device_queue_fill(
    iree_hal_device_t *, iree_hal_queue_affinity_t,
    const iree_hal_semaphore_list_t, const iree_hal_semaphore_list_t,
    iree_hal_buffer_t *, iree_device_size_t, iree_device_size_t, const void *,
    iree_host_size_t, iree_hal_fill_flags_t);
static iree_status_t iree_hal_tt_device_queue_update(
    iree_hal_device_t *, iree_hal_queue_affinity_t,
    const iree_hal_semaphore_list_t, const iree_hal_semaphore_list_t,
    const void *, iree_host_size_t, iree_hal_buffer_t *, iree_device_size_t,
    iree_device_size_t, iree_hal_update_flags_t);
static iree_status_t iree_hal_tt_device_queue_copy(
    iree_hal_device_t *, iree_hal_queue_affinity_t,
    const iree_hal_semaphore_list_t, const iree_hal_semaphore_list_t,
    iree_hal_buffer_t *, iree_device_size_t, iree_hal_buffer_t *,
    iree_device_size_t, iree_device_size_t, iree_hal_copy_flags_t);
static iree_status_t iree_hal_tt_device_queue_read(
    iree_hal_device_t *, iree_hal_queue_affinity_t,
    const iree_hal_semaphore_list_t, const iree_hal_semaphore_list_t,
    iree_hal_file_t *, uint64_t, iree_hal_buffer_t *, iree_device_size_t,
    iree_device_size_t, iree_hal_read_flags_t);
static iree_status_t iree_hal_tt_device_queue_write(
    iree_hal_device_t *, iree_hal_queue_affinity_t,
    const iree_hal_semaphore_list_t, const iree_hal_semaphore_list_t,
    iree_hal_buffer_t *, iree_device_size_t, iree_hal_file_t *, uint64_t,
    iree_device_size_t, iree_hal_write_flags_t);
static iree_status_t iree_hal_tt_device_queue_host_call(
    iree_hal_device_t *, iree_hal_queue_affinity_t,
    const iree_hal_semaphore_list_t, const iree_hal_semaphore_list_t,
    iree_hal_host_call_t, const uint64_t[4], iree_hal_host_call_flags_t);
static iree_status_t iree_hal_tt_device_queue_dispatch(
    iree_hal_device_t *, iree_hal_queue_affinity_t,
    const iree_hal_semaphore_list_t, const iree_hal_semaphore_list_t,
    iree_hal_executable_t *, iree_hal_executable_export_ordinal_t,
    const iree_hal_dispatch_config_t, iree_const_byte_span_t,
    const iree_hal_buffer_ref_list_t, iree_hal_dispatch_flags_t);
static iree_status_t iree_hal_tt_device_queue_execute(
    iree_hal_device_t *, iree_hal_queue_affinity_t,
    const iree_hal_semaphore_list_t, const iree_hal_semaphore_list_t,
    iree_hal_command_buffer_t *, iree_hal_buffer_binding_table_t,
    iree_hal_execute_flags_t);
static iree_status_t iree_hal_tt_device_queue_flush(iree_hal_device_t *,
                                                    iree_hal_queue_affinity_t);
static iree_status_t
iree_hal_tt_device_wait_semaphores(iree_hal_device_t *, iree_hal_wait_mode_t,
                                   const iree_hal_semaphore_list_t,
                                   iree_timeout_t, iree_hal_wait_flags_t);
static iree_status_t
iree_hal_tt_device_profiling_begin(iree_hal_device_t *,
                                   const iree_hal_device_profiling_options_t *);
static iree_status_t iree_hal_tt_device_profiling_flush(iree_hal_device_t *);
static iree_status_t iree_hal_tt_device_profiling_end(iree_hal_device_t *);

// Define vtable early so it can be used by cast function
static const iree_hal_device_vtable_t iree_hal_tt_device_vtable = {
    .destroy = iree_hal_tt_device_destroy,
    .id = iree_hal_tt_device_id,
    .host_allocator = iree_hal_tt_device_host_allocator,
    .device_allocator = iree_hal_tt_device_allocator,
    .replace_device_allocator = iree_hal_tt_device_replace_allocator,
    .replace_channel_provider = iree_hal_tt_device_replace_channel_provider,
    .trim = iree_hal_tt_device_trim,
    .query_i64 = iree_hal_tt_device_query_i64,
    .query_capabilities = iree_hal_tt_device_query_capabilities,
    .topology_info = iree_hal_tt_device_topology_info,
    .refine_topology_edge = iree_hal_tt_device_refine_topology_edge,
    .assign_topology_info = iree_hal_tt_device_assign_topology_info,
    .create_channel = iree_hal_tt_device_create_channel,
    .create_command_buffer = iree_hal_tt_device_create_command_buffer,
    .create_event = iree_hal_tt_device_create_event,
    .create_executable_cache = iree_hal_tt_device_create_executable_cache,
    .import_file = iree_hal_tt_device_import_file,
    .create_semaphore = iree_hal_tt_device_create_semaphore,
    .query_semaphore_compatibility =
        iree_hal_tt_device_query_semaphore_compatibility,
    .queue_alloca = iree_hal_tt_device_queue_alloca,
    .queue_dealloca = iree_hal_tt_device_queue_dealloca,
    .queue_fill = iree_hal_tt_device_queue_fill,
    .queue_update = iree_hal_tt_device_queue_update,
    .queue_copy = iree_hal_tt_device_queue_copy,
    .queue_read = iree_hal_tt_device_queue_read,
    .queue_write = iree_hal_tt_device_queue_write,
    .queue_host_call = iree_hal_tt_device_queue_host_call,
    .queue_dispatch = iree_hal_tt_device_queue_dispatch,
    .queue_execute = iree_hal_tt_device_queue_execute,
    .queue_flush = iree_hal_tt_device_queue_flush,
    .wait_semaphores = iree_hal_tt_device_wait_semaphores,
    .profiling_begin = iree_hal_tt_device_profiling_begin,
    .profiling_flush = iree_hal_tt_device_profiling_flush,
    .profiling_end = iree_hal_tt_device_profiling_end,
};

static iree_hal_tt_device_t *iree_hal_tt_device_cast(iree_hal_device_t *base) {
  IREE_HAL_ASSERT_TYPE(base, &iree_hal_tt_device_vtable);
  return (iree_hal_tt_device_t *)base;
}

//===----------------------------------------------------------------------===//
// Internal accessors
//===----------------------------------------------------------------------===//

#ifndef TT_IREE_ENABLE_MOCK

tt::tt_metal::distributed::MeshDevice *
iree_hal_tt_device_mesh_handle(iree_hal_tt_device_t *device) {
  return device ? device->mesh_device.get() : nullptr;
}

tt::tt_metal::distributed::MeshCommandQueue *
iree_hal_tt_device_mesh_queue(iree_hal_tt_device_t *device) {
  return device ? device->mesh_cq : nullptr;
}

tt::tt_metal::IDevice *
iree_hal_tt_device_idevice(iree_hal_tt_device_t *device) {
  if (!device || !device->mesh_device)
    return nullptr;
  // Legacy Buffer requires the local physical device backing the unit mesh.
  auto devices = device->mesh_device->get_devices();
  return devices.empty() ? nullptr : devices.front();
}
#endif

//===----------------------------------------------------------------------===//
// Device creation
//===----------------------------------------------------------------------===//

iree_status_t iree_hal_tt_device_create(iree_hal_tenstorrent_driver_t *driver,
                                        iree_hal_device_id_t device_id,
                                        iree_allocator_t host_allocator,
                                        iree_hal_device_t **out_device) {
  IREE_ASSERT_ARGUMENT(out_device);
  *out_device = nullptr;

  IREE_TRACE_ZONE_BEGIN(z0);

  iree_hal_tt_device_t *device = nullptr;
  iree_status_t status =
      iree_allocator_malloc(host_allocator, sizeof(*device), (void **)&device);

  if (iree_status_is_ok(status)) {
    std::memset(device, 0, sizeof(*device));
    iree_hal_resource_initialize(&iree_hal_tt_device_vtable, &device->resource);
    device->host_allocator = host_allocator;
    device->identifier = iree_make_cstring_view("tenstorrent");
    device->device_id = device_id;
  }

#ifndef TT_IREE_ENABLE_MOCK
  // Create a TT-Metal MeshDevice even for a single physical device.
  if (iree_status_is_ok(status)) {
    try {
      // Create a 1x1 mesh (unit mesh) for single device
      device->mesh_device =
          tt::tt_metal::distributed::MeshDevice::create_unit_mesh(device_id);

      if (!device->mesh_device) {
        status =
            iree_make_status(IREE_STATUS_UNAVAILABLE,
                             "failed to create mesh device %d", (int)device_id);
      }
    } catch (const std::exception &e) {
      status = iree_make_status(IREE_STATUS_INTERNAL, "TT-Metal error: %s",
                                e.what());
    }
  }

  // Get mesh command queue and print device info
  if (iree_status_is_ok(status)) {
    try {
      device->mesh_cq = &device->mesh_device->mesh_command_queue();

      tt::tt_metal::IDevice *idevice = device->mesh_device.get();
      if (idevice) {
        auto grid = idevice->compute_with_storage_grid_size();
        auto arch = idevice->arch();
        const char *arch_name = (arch == tt::ARCH::BLACKHOLE)     ? "Blackhole"
                                : (arch == tt::ARCH::WORMHOLE_B0) ? "Wormhole"
                                                                  : "Unknown";

        fprintf(
            stderr,
            "tt-iree: MeshDevice %d opened (%s, %zux%zu cores, %lu MB DRAM)\n",
            (int)device_id, arch_name, grid.x, grid.y,
            (unsigned long)(idevice->num_dram_channels() *
                            idevice->dram_size_per_channel() / (1024 * 1024)));
      }
    } catch (const std::exception &e) {
      status = iree_make_status(IREE_STATUS_INTERNAL,
                                "failed to get mesh queue: %s", e.what());
    }
  }
#else
  if (iree_status_is_ok(status)) {
    fprintf(stderr, "tt-iree: Device %d opened (MOCK MODE)\n", (int)device_id);
  }
#endif

  // Create allocator
  if (iree_status_is_ok(status)) {
    status = iree_hal_tt_allocator_create(device, host_allocator,
                                          &device->device_allocator);
  }

  if (iree_status_is_ok(status)) {
    *out_device = (iree_hal_device_t *)device;
  } else {
    if (device) {
#ifndef TT_IREE_ENABLE_MOCK
      if (device->mesh_device) {
        try {
          device->mesh_device->close();
        } catch (...) {
        }
        // shared_ptr will handle deletion
      }
#endif
      if (device->device_allocator) {
        iree_hal_allocator_release(device->device_allocator);
      }
      iree_allocator_free(host_allocator, device);
    }
  }

  IREE_TRACE_ZONE_END(z0);
  return status;
}

//===----------------------------------------------------------------------===//
// Device vtable
//===----------------------------------------------------------------------===//

static void iree_hal_tt_device_destroy(iree_hal_device_t *base) {
  auto *device = iree_hal_tt_device_cast(base);
  iree_allocator_t host_allocator = device->host_allocator;

  IREE_TRACE_ZONE_BEGIN(z0);

  fprintf(stderr, "tt-iree: Closing mesh device %d\n", (int)device->device_id);

  if (device->device_allocator) {
    iree_hal_allocator_release(device->device_allocator);
  }

#ifndef TT_IREE_ENABLE_MOCK
  if (device->mesh_device) {
    try {
      device->mesh_device->close();
      // shared_ptr will handle deletion
    } catch (...) {
    }
  }
#endif

  iree_allocator_free(host_allocator, device);
  IREE_TRACE_ZONE_END(z0);
}

static iree_string_view_t iree_hal_tt_device_id(iree_hal_device_t *base) {
  return iree_hal_tt_device_cast(base)->identifier;
}

static iree_allocator_t
iree_hal_tt_device_host_allocator(iree_hal_device_t *base) {
  return iree_hal_tt_device_cast(base)->host_allocator;
}

static iree_hal_allocator_t *
iree_hal_tt_device_allocator(iree_hal_device_t *base) {
  return iree_hal_tt_device_cast(base)->device_allocator;
}

static void
iree_hal_tt_device_replace_allocator(iree_hal_device_t *base,
                                     iree_hal_allocator_t *new_allocator) {
  auto *device = iree_hal_tt_device_cast(base);
  if (device->device_allocator)
    iree_hal_allocator_release(device->device_allocator);
  device->device_allocator = new_allocator;
  iree_hal_allocator_retain(new_allocator);
}

static void
iree_hal_tt_device_replace_channel_provider(iree_hal_device_t *,
                                            iree_hal_channel_provider_t *) {}

static iree_status_t iree_hal_tt_device_trim(iree_hal_device_t *) {
  return iree_ok_status();
}

static iree_status_t iree_hal_tt_device_query_i64(iree_hal_device_t *base,
                                                  iree_string_view_t category,
                                                  iree_string_view_t key,
                                                  int64_t *out_value) {
  auto *device = iree_hal_tt_device_cast(base);
  *out_value = 0;

  if (iree_string_view_equal(category, IREE_SV("hal.device.id"))) {
    // key is a glob pattern; return 1 if our identifier matches it.
    *out_value =
        iree_string_view_match_pattern(device->identifier, key) ? 1 : 0;
    return iree_ok_status();
  }

  if (iree_string_view_equal(category, IREE_SV("hal.executable.format"))) {
    // Report support for the TTEX FlatBuffer format.
    *out_value =
        iree_string_view_equal(key, IREE_SV("tenstorrent-ttex-fb")) ? 1 : 0;
    return iree_ok_status();
  }

#ifndef TT_IREE_ENABLE_MOCK
  if (iree_string_view_equal(category, IREE_SV("hal.device")) &&
      device->mesh_device) {
    tt::tt_metal::IDevice *idevice = device->mesh_device.get();
    if (idevice) {
      auto grid = idevice->compute_with_storage_grid_size();
      if (iree_string_view_equal(key, IREE_SV("core_count_x"))) {
        *out_value = grid.x;
        return iree_ok_status();
      }
      if (iree_string_view_equal(key, IREE_SV("core_count_y"))) {
        *out_value = grid.y;
        return iree_ok_status();
      }
      if (iree_string_view_equal(key, IREE_SV("dram_size"))) {
        *out_value =
            idevice->num_dram_channels() * idevice->dram_size_per_channel();
        return iree_ok_status();
      }
    }
  }
#endif

  return iree_make_status(IREE_STATUS_NOT_FOUND, "unknown key '%.*s::%.*s'",
                          (int)category.size, category.data, (int)key.size,
                          key.data);
}

static iree_status_t iree_hal_tt_device_query_capabilities(
    iree_hal_device_t *, iree_hal_device_capabilities_t *out_capabilities) {
  std::memset(out_capabilities, 0, sizeof(*out_capabilities));
  return iree_ok_status();
}

static const iree_hal_device_topology_info_t *
iree_hal_tt_device_topology_info(iree_hal_device_t *base) {
  return &iree_hal_tt_device_cast(base)->topology_info;
}

static iree_status_t iree_hal_tt_device_refine_topology_edge(
    iree_hal_device_t *, iree_hal_device_t *, iree_hal_topology_edge_t *) {
  return iree_ok_status();
}

static iree_status_t iree_hal_tt_device_assign_topology_info(
    iree_hal_device_t *base,
    const iree_hal_device_topology_info_t *topology_info) {
  iree_hal_tt_device_cast(base)->topology_info = *topology_info;
  return iree_ok_status();
}

// Stub implementations
static iree_status_t iree_hal_tt_device_create_channel(
    iree_hal_device_t *, iree_hal_queue_affinity_t, iree_hal_channel_params_t,
    iree_hal_channel_t **) {
  return iree_make_status(IREE_STATUS_UNIMPLEMENTED, "channel not implemented");
}

static iree_status_t iree_hal_tt_device_create_event(iree_hal_device_t *,
                                                     iree_hal_queue_affinity_t,
                                                     iree_hal_event_flags_t,
                                                     iree_hal_event_t **) {
  return iree_make_status(IREE_STATUS_UNIMPLEMENTED, "event not implemented");
}

static iree_status_t iree_hal_tt_device_create_executable_cache(
    iree_hal_device_t *base_device, iree_string_view_t identifier,
    iree_loop_t loop, iree_hal_executable_cache_t **out_executable_cache) {
  // Passthrough cache: no caching, every prepare_executable creates fresh.
  // The cache recognizes "tenstorrent-ttex-fb" format and delegates to
  // iree_hal_tt_executable_create().
  return iree_hal_tt_executable_cache_create(
      base_device, identifier, iree_hal_device_host_allocator(base_device),
      out_executable_cache);
}

static iree_status_t iree_hal_tt_device_import_file(
    iree_hal_device_t *, iree_hal_queue_affinity_t, iree_hal_memory_access_t,
    iree_io_file_handle_t *, iree_hal_external_file_flags_t,
    iree_hal_file_t **) {
  return iree_make_status(IREE_STATUS_UNIMPLEMENTED,
                          "file import not implemented");
}

static iree_status_t iree_hal_tt_device_create_semaphore(
    iree_hal_device_t *base_device, iree_hal_queue_affinity_t queue_affinity,
    uint64_t initial_value, iree_hal_semaphore_flags_t flags,
    iree_hal_semaphore_t **out_semaphore) {
  return iree_hal_tt_semaphore_create(
      iree_hal_tt_device_cast(base_device)->host_allocator, initial_value,
      out_semaphore);
}

static iree_hal_semaphore_compatibility_t
iree_hal_tt_device_query_semaphore_compatibility(iree_hal_device_t *,
                                                 iree_hal_semaphore_t *) {
  return IREE_HAL_SEMAPHORE_COMPATIBILITY_HOST_ONLY;
}

static iree_status_t iree_hal_tt_device_queue_alloca(
    iree_hal_device_t *base_device, iree_hal_queue_affinity_t queue_affinity,
    const iree_hal_semaphore_list_t wait_semaphore_list,
    const iree_hal_semaphore_list_t signal_semaphore_list,
    iree_hal_allocator_pool_t pool, iree_hal_buffer_params_t params,
    iree_device_size_t allocation_size, iree_hal_alloca_flags_t flags,
    iree_hal_buffer_t **IREE_RESTRICT out_buffer) {
  IREE_RETURN_IF_ERROR(
      iree_hal_semaphore_list_wait(wait_semaphore_list, iree_infinite_timeout(),
                                   IREE_HAL_WAIT_FLAG_DEFAULT));
  IREE_RETURN_IF_ERROR(
      iree_hal_allocator_allocate_buffer(iree_hal_device_allocator(base_device),
                                         params, allocation_size, out_buffer));
  IREE_RETURN_IF_ERROR(iree_hal_semaphore_list_signal(signal_semaphore_list));
  return iree_ok_status();
}

static iree_status_t iree_hal_tt_device_queue_dealloca(
    iree_hal_device_t *base_device, iree_hal_queue_affinity_t queue_affinity,
    const iree_hal_semaphore_list_t wait_semaphore_list,
    const iree_hal_semaphore_list_t signal_semaphore_list,
    iree_hal_buffer_t *buffer, iree_hal_dealloca_flags_t flags) {
  return iree_hal_device_queue_barrier(
      base_device, queue_affinity, wait_semaphore_list, signal_semaphore_list,
      IREE_HAL_EXECUTE_FLAG_NONE);
}

static iree_status_t iree_hal_tt_device_queue_fill(
    iree_hal_device_t *base_device, iree_hal_queue_affinity_t queue_affinity,
    const iree_hal_semaphore_list_t wait_semaphore_list,
    const iree_hal_semaphore_list_t signal_semaphore_list,
    iree_hal_buffer_t *target_buffer, iree_device_size_t target_offset,
    iree_device_size_t length, const void *pattern,
    iree_host_size_t pattern_length, iree_hal_fill_flags_t flags) {
  if (flags != IREE_HAL_FILL_FLAG_NONE) {
    return iree_make_status(
        IREE_STATUS_UNIMPLEMENTED,
        "Tenstorrent queue fill does not support flags 0x%" PRIx64, flags);
  }
  if (!target_buffer || !pattern) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "queue fill arguments must not be null");
  }

  IREE_RETURN_IF_ERROR(
      iree_hal_semaphore_list_wait(wait_semaphore_list, iree_infinite_timeout(),
                                   IREE_HAL_WAIT_FLAG_DEFAULT));

  iree_device_size_t resolved_offset = 0;
  iree_device_size_t resolved_length = 0;
  IREE_RETURN_IF_ERROR(iree_hal_buffer_calculate_range(
      /*base_offset=*/0, iree_hal_buffer_byte_length(target_buffer),
      target_offset, length, &resolved_offset, &resolved_length));
  IREE_RETURN_IF_ERROR(iree_hal_buffer_map_fill(target_buffer, resolved_offset,
                                                resolved_length, pattern,
                                                pattern_length));
  return iree_hal_semaphore_list_signal(signal_semaphore_list);
}

static iree_status_t iree_hal_tt_device_queue_update(
    iree_hal_device_t *base_device, iree_hal_queue_affinity_t queue_affinity,
    const iree_hal_semaphore_list_t wait_semaphore_list,
    const iree_hal_semaphore_list_t signal_semaphore_list,
    const void *source_buffer, iree_host_size_t source_offset,
    iree_hal_buffer_t *target_buffer, iree_device_size_t target_offset,
    iree_device_size_t length, iree_hal_update_flags_t flags) {
  if (flags != IREE_HAL_UPDATE_FLAG_NONE) {
    return iree_make_status(
        IREE_STATUS_UNIMPLEMENTED,
        "Tenstorrent queue update does not support flags 0x%" PRIx64, flags);
  }
  if (!source_buffer || !target_buffer) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "queue update arguments must not be null");
  }

  IREE_RETURN_IF_ERROR(
      iree_hal_semaphore_list_wait(wait_semaphore_list, iree_infinite_timeout(),
                                   IREE_HAL_WAIT_FLAG_DEFAULT));

  iree_device_size_t resolved_offset = 0;
  iree_device_size_t resolved_length = 0;
  IREE_RETURN_IF_ERROR(iree_hal_buffer_calculate_range(
      /*base_offset=*/0, iree_hal_buffer_byte_length(target_buffer),
      target_offset, length, &resolved_offset, &resolved_length));
  const uint8_t *source_data =
      static_cast<const uint8_t *>(source_buffer) + source_offset;
  IREE_RETURN_IF_ERROR(iree_hal_buffer_map_write(target_buffer, resolved_offset,
                                                 source_data, resolved_length));
  return iree_hal_semaphore_list_signal(signal_semaphore_list);
}

// Inline synchronous device-to-device (host-mediated) copy.
static iree_status_t iree_hal_tt_device_queue_copy(
    iree_hal_device_t *base_device, iree_hal_queue_affinity_t queue_affinity,
    const iree_hal_semaphore_list_t wait_semaphore_list,
    const iree_hal_semaphore_list_t signal_semaphore_list,
    iree_hal_buffer_t *source_buffer, iree_device_size_t source_offset,
    iree_hal_buffer_t *target_buffer, iree_device_size_t target_offset,
    iree_device_size_t length, iree_hal_copy_flags_t flags) {
  if (flags != IREE_HAL_COPY_FLAG_NONE) {
    return iree_make_status(
        IREE_STATUS_UNIMPLEMENTED,
        "Tenstorrent queue copy does not support flags 0x%" PRIx64, flags);
  }
  if (!source_buffer || !target_buffer) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "queue copy buffers must not be null");
  }

  IREE_RETURN_IF_ERROR(
      iree_hal_semaphore_list_wait(wait_semaphore_list, iree_infinite_timeout(),
                                   IREE_HAL_WAIT_FLAG_DEFAULT));

  iree_hal_buffer_ref_t source_ref =
      iree_hal_make_buffer_ref(source_buffer, source_offset, length);
  IREE_RETURN_IF_ERROR(iree_hal_buffer_calculate_range(
      /*base_offset=*/0, iree_hal_buffer_byte_length(source_buffer),
      source_offset, length, &source_ref.offset, &source_ref.length));

  iree_hal_buffer_ref_t target_ref =
      iree_hal_make_buffer_ref(target_buffer, target_offset, length);
  IREE_RETURN_IF_ERROR(iree_hal_buffer_calculate_range(
      /*base_offset=*/0, iree_hal_buffer_byte_length(target_buffer),
      target_offset, length, &target_ref.offset, &target_ref.length));

  IREE_RETURN_IF_ERROR(
      iree_hal_tt_copy_resolved_buffer_refs(source_ref, target_ref));

  for (iree_host_size_t i = 0; i < signal_semaphore_list.count; ++i) {
    IREE_RETURN_IF_ERROR(
        iree_hal_semaphore_signal(signal_semaphore_list.semaphores[i],
                                  signal_semaphore_list.payload_values[i]));
  }
  return iree_ok_status();
}

static iree_status_t iree_hal_tt_device_queue_host_call(
    iree_hal_device_t *base_device, iree_hal_queue_affinity_t queue_affinity,
    const iree_hal_semaphore_list_t wait_semaphore_list,
    const iree_hal_semaphore_list_t signal_semaphore_list,
    iree_hal_host_call_t call, const uint64_t args[4],
    iree_hal_host_call_flags_t flags) {
  return iree_make_status(IREE_STATUS_UNIMPLEMENTED,
                          "queue_host_call not implemented");
}

static iree_status_t iree_hal_tt_device_queue_dispatch(
    iree_hal_device_t *base_device, iree_hal_queue_affinity_t queue_affinity,
    const iree_hal_semaphore_list_t wait_semaphore_list,
    const iree_hal_semaphore_list_t signal_semaphore_list,
    iree_hal_executable_t *executable,
    iree_hal_executable_export_ordinal_t export_ordinal,
    const iree_hal_dispatch_config_t config, iree_const_byte_span_t constants,
    const iree_hal_buffer_ref_list_t bindings,
    iree_hal_dispatch_flags_t flags) {
  return iree_make_status(IREE_STATUS_UNIMPLEMENTED,
                          "queue_dispatch not implemented");
}

static iree_status_t iree_hal_tt_device_queue_read(
    iree_hal_device_t *, iree_hal_queue_affinity_t,
    const iree_hal_semaphore_list_t, const iree_hal_semaphore_list_t,
    iree_hal_file_t *, uint64_t, iree_hal_buffer_t *, iree_device_size_t,
    iree_device_size_t, iree_hal_read_flags_t) {
  return iree_make_status(IREE_STATUS_UNIMPLEMENTED,
                          "queue read not implemented");
}

static iree_status_t iree_hal_tt_device_queue_write(
    iree_hal_device_t *, iree_hal_queue_affinity_t,
    const iree_hal_semaphore_list_t, const iree_hal_semaphore_list_t,
    iree_hal_buffer_t *, iree_device_size_t, iree_hal_file_t *, uint64_t,
    iree_device_size_t, iree_hal_write_flags_t) {
  return iree_make_status(IREE_STATUS_UNIMPLEMENTED,
                          "queue write not implemented");
}

static iree_status_t iree_hal_tt_device_queue_execute(
    iree_hal_device_t *base_device, iree_hal_queue_affinity_t queue_affinity,
    const iree_hal_semaphore_list_t wait_semaphores,
    const iree_hal_semaphore_list_t signal_semaphores,
    iree_hal_command_buffer_t *command_buffer,
    iree_hal_buffer_binding_table_t binding_table,
    iree_hal_execute_flags_t flags) {
#ifndef TT_IREE_ENABLE_MOCK
  auto *device = iree_hal_tt_device_cast(base_device);
  IREE_RETURN_IF_ERROR(iree_hal_semaphore_list_wait(
      wait_semaphores, iree_infinite_timeout(), IREE_HAL_WAIT_FLAG_DEFAULT));

  iree_host_size_t command_count = 0;
  iree_hal_tt_command_t *commands =
      iree_hal_tt_command_buffer_get_commands(command_buffer, &command_count);

  if (command_count == 0) {
    for (iree_host_size_t i = 0; i < signal_semaphores.count; ++i) {
      IREE_RETURN_IF_ERROR(
          iree_hal_semaphore_signal(signal_semaphores.semaphores[i],
                                    signal_semaphores.payload_values[i]));
    }
    return iree_ok_status();
  }

  for (iree_host_size_t i = 0; i < command_count; ++i) {
    bool transfer_handled = false;
    IREE_RETURN_IF_ERROR(iree_hal_tt_execute_transfer_command(
        command_buffer, binding_table, commands[i], &transfer_handled));
    if (transfer_handled)
      continue;

    if (commands[i].type == IREE_HAL_TT_COMMAND_TYPE_DISPATCH) {
      auto &cmd = commands[i].dispatch;
      iree_hal_tt_kernel_params_t *params = nullptr;
      IREE_RETURN_IF_ERROR(iree_hal_tt_executable_lookup_kernel_params_mutable(
          cmd.executable, cmd.export_ordinal, &params));

      if (!params) {
        return iree_make_status(IREE_STATUS_FAILED_PRECONDITION,
                                "executable export ordinal %d has no params",
                                cmd.export_ordinal);
      }

      if (params->builtin_program ==
          TT_IREE_TTEX_BUILTIN_PROGRAM_BF16_MATMUL_TILED) {
        IREE_RETURN_IF_ERROR(iree_hal_tt_execute_tiled_matmul_dispatch(
            device, params, cmd, binding_table));
        continue;
      }

      if (params->builtin_program ==
          TT_IREE_TTEX_BUILTIN_PROGRAM_F32_MATMUL_LIKE) {
        IREE_RETURN_IF_ERROR(iree_hal_tt_execute_f32_matmul_like_dispatch(
            device, params, cmd, binding_table));
        continue;
      }

      if (params->builtin_program == TT_IREE_TTEX_BUILTIN_PROGRAM_F32_CONV2D) {
        IREE_RETURN_IF_ERROR(iree_hal_tt_execute_f32_conv2d_dispatch(
            device, params, cmd, binding_table));
        continue;
      }

      if (params->builtin_program ==
          TT_IREE_TTEX_BUILTIN_PROGRAM_F32_PADDED_COPY) {
        IREE_RETURN_IF_ERROR(iree_hal_tt_execute_f32_padded_copy_dispatch(
            params, cmd, binding_table));
        continue;
      }

      if (params->builtin_program ==
          TT_IREE_TTEX_BUILTIN_PROGRAM_F32_MAX_POOL_2D) {
        IREE_RETURN_IF_ERROR(iree_hal_tt_execute_f32_max_pool_dispatch(
            params, cmd, binding_table));
        continue;
      }

      if (params->builtin_program ==
          TT_IREE_TTEX_BUILTIN_PROGRAM_F32_GLOBAL_SUM) {
        IREE_RETURN_IF_ERROR(iree_hal_tt_execute_f32_global_sum_dispatch(
            params, cmd, binding_table));
        continue;
      }

      if (params->builtin_program ==
          TT_IREE_TTEX_BUILTIN_PROGRAM_F32_SCALAR_DIV) {
        IREE_RETURN_IF_ERROR(iree_hal_tt_execute_f32_scalar_div_dispatch(
            params, cmd, binding_table));
        continue;
      }

      if (params->builtin_program == TT_IREE_TTEX_BUILTIN_PROGRAM_F32_LINEAR) {
        IREE_RETURN_IF_ERROR(iree_hal_tt_execute_f32_linear_dispatch(
            device, params, cmd, binding_table));
        continue;
      }

      if (!params->program) {
        return iree_make_status(IREE_STATUS_FAILED_PRECONDITION,
                                "executable export ordinal %d has no program",
                                cmd.export_ordinal);
      }

      if (cmd.binding_count != 3) {
        return iree_make_status(
            IREE_STATUS_INVALID_ARGUMENT,
            "Tenstorrent builtin expects exactly 3 bindings, got %zu",
            cmd.binding_count);
      }

      iree_hal_buffer_ref_t in0_ref = {};
      iree_hal_buffer_ref_t in1_ref = {};
      iree_hal_buffer_ref_t out_ref = {};
      IREE_RETURN_IF_ERROR(iree_hal_tt_resolve_buffer_ref(
          binding_table, cmd.bindings[0], &in0_ref));
      IREE_RETURN_IF_ERROR(iree_hal_tt_resolve_buffer_ref(
          binding_table, cmd.bindings[1], &in1_ref));
      IREE_RETURN_IF_ERROR(iree_hal_tt_resolve_buffer_ref(
          binding_table, cmd.bindings[2], &out_ref));

      uint32_t in0_addr = 0;
      uint32_t in1_addr = 0;
      uint32_t out_addr = 0;
      uint32_t in0_byte_length = 0;
      uint32_t in1_byte_length = 0;
      uint32_t out_byte_length = 0;
      IREE_RETURN_IF_ERROR(iree_hal_tt_get_binding_runtime_args(
          in0_ref, &in0_addr, &in0_byte_length));
      IREE_RETURN_IF_ERROR(iree_hal_tt_get_binding_runtime_args(
          in1_ref, &in1_addr, &in1_byte_length));
      IREE_RETURN_IF_ERROR(iree_hal_tt_get_binding_runtime_args(
          out_ref, &out_addr, &out_byte_length));

      tt::tt_metal::Program *program =
          static_cast<tt::tt_metal::Program *>(params->program);
      tt::tt_metal::CoreCoord core = {0, 0};
      iree_status_t status = iree_ok_status();
      try {
        switch (params->builtin_program) {
        case TT_IREE_TTEX_BUILTIN_PROGRAM_CUSTOM_SFPI_ADD:
          status = iree_hal_tt_set_add_runtime_args(
              *program, params, core, in0_addr, in1_addr, out_addr,
              in0_byte_length, in1_byte_length, out_byte_length);
          break;
        case TT_IREE_TTEX_BUILTIN_PROGRAM_BF16_MATMUL_32X32X32:
          status = iree_hal_tt_set_matmul_runtime_args(
              *program, params, core, in0_addr, in1_addr, out_addr,
              in0_byte_length, in1_byte_length, out_byte_length);
          break;
        default:
          status =
              iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                               "unsupported builtin program %u for dispatch",
                               params->builtin_program);
          break;
        }
      } catch (const std::exception &e) {
        return iree_make_status(IREE_STATUS_INTERNAL,
                                "failed to set runtime args: %s", e.what());
      }
      IREE_RETURN_IF_ERROR(status);

      tt::tt_metal::distributed::MeshWorkload workload;
      auto device_range = tt::tt_metal::distributed::MeshCoordinateRange(
          device->mesh_device->shape());
      try {
        workload.add_program(device_range, std::move(*program));
        tt::tt_metal::distributed::EnqueueMeshWorkload(
            *device->mesh_cq, workload, /*blocking=*/true);
        auto &programs = workload.get_programs();
        if (!programs.empty()) {
          *program = std::move(programs.begin()->second);
        }
      } catch (const std::exception &e) {
        auto &programs = workload.get_programs();
        if (!programs.empty()) {
          *program = std::move(programs.begin()->second);
        }
        return iree_make_status(IREE_STATUS_INTERNAL,
                                "dispatch execution failed: %s", e.what());
      }
      continue;
    }
  }

  for (iree_host_size_t i = 0; i < signal_semaphores.count; ++i) {
    IREE_RETURN_IF_ERROR(iree_hal_semaphore_signal(
        signal_semaphores.semaphores[i], signal_semaphores.payload_values[i]));
  }
#else
  auto *device = iree_hal_tt_device_cast(base_device);
  IREE_RETURN_IF_ERROR(iree_hal_semaphore_list_wait(
      wait_semaphores, iree_infinite_timeout(), IREE_HAL_WAIT_FLAG_DEFAULT));
  iree_host_size_t command_count = 0;
  iree_hal_tt_command_t *commands =
      iree_hal_tt_command_buffer_get_commands(command_buffer, &command_count);
  for (iree_host_size_t i = 0; i < command_count; ++i) {
    bool transfer_handled = false;
    IREE_RETURN_IF_ERROR(iree_hal_tt_execute_transfer_command(
        command_buffer, binding_table, commands[i], &transfer_handled));
    if (transfer_handled)
      continue;

    if (commands[i].type == IREE_HAL_TT_COMMAND_TYPE_DISPATCH) {
      auto &cmd = commands[i].dispatch;
      const iree_hal_tt_kernel_params_t *params = nullptr;
      IREE_RETURN_IF_ERROR(iree_hal_tt_executable_lookup_kernel_params(
          cmd.executable, cmd.export_ordinal, &params));
      if (!params) {
        return iree_make_status(IREE_STATUS_FAILED_PRECONDITION,
                                "executable export ordinal %d has no params",
                                cmd.export_ordinal);
      }
      if (params->builtin_program ==
          TT_IREE_TTEX_BUILTIN_PROGRAM_F32_MATMUL_LIKE) {
        IREE_RETURN_IF_ERROR(iree_hal_tt_execute_f32_matmul_like_dispatch(
            device, params, cmd, binding_table));
      } else if (params->builtin_program ==
                 TT_IREE_TTEX_BUILTIN_PROGRAM_F32_CONV2D) {
        IREE_RETURN_IF_ERROR(iree_hal_tt_execute_f32_conv2d_dispatch(
            device, params, cmd, binding_table));
      } else if (params->builtin_program ==
                 TT_IREE_TTEX_BUILTIN_PROGRAM_F32_PADDED_COPY) {
        IREE_RETURN_IF_ERROR(iree_hal_tt_execute_f32_padded_copy_dispatch(
            params, cmd, binding_table));
      } else if (params->builtin_program ==
                 TT_IREE_TTEX_BUILTIN_PROGRAM_F32_MAX_POOL_2D) {
        IREE_RETURN_IF_ERROR(iree_hal_tt_execute_f32_max_pool_dispatch(
            params, cmd, binding_table));
      } else if (params->builtin_program ==
                 TT_IREE_TTEX_BUILTIN_PROGRAM_F32_GLOBAL_SUM) {
        IREE_RETURN_IF_ERROR(iree_hal_tt_execute_f32_global_sum_dispatch(
            params, cmd, binding_table));
      } else if (params->builtin_program ==
                 TT_IREE_TTEX_BUILTIN_PROGRAM_F32_SCALAR_DIV) {
        IREE_RETURN_IF_ERROR(iree_hal_tt_execute_f32_scalar_div_dispatch(
            params, cmd, binding_table));
      } else if (params->builtin_program ==
                 TT_IREE_TTEX_BUILTIN_PROGRAM_F32_LINEAR) {
        IREE_RETURN_IF_ERROR(iree_hal_tt_execute_f32_linear_dispatch(
            device, params, cmd, binding_table));
      }
    }
  }
  for (iree_host_size_t i = 0; i < signal_semaphores.count; ++i) {
    IREE_RETURN_IF_ERROR(iree_hal_semaphore_signal(
        signal_semaphores.semaphores[i], signal_semaphores.payload_values[i]));
  }
#endif

  return iree_ok_status();
}

static iree_status_t iree_hal_tt_device_queue_flush(iree_hal_device_t *base,
                                                    iree_hal_queue_affinity_t) {
#ifndef TT_IREE_ENABLE_MOCK
  auto *device = iree_hal_tt_device_cast(base);
  if (device->mesh_cq) {
    try {
      // Flush the mesh command queue (wait for all pending operations)
      tt::tt_metal::distributed::Finish(*device->mesh_cq);
    } catch (const std::exception &e) {
      return iree_make_status(IREE_STATUS_INTERNAL, "Failed to flush queue: %s",
                              e.what());
    }
  }
#endif
  return iree_ok_status();
}

static iree_status_t iree_hal_tt_device_wait_semaphores(
    iree_hal_device_t *base_device, iree_hal_wait_mode_t wait_mode,
    const iree_hal_semaphore_list_t semaphore_list, iree_timeout_t timeout,
    iree_hal_wait_flags_t flags) {
  if (semaphore_list.count == 0)
    return iree_ok_status();

  iree_time_t deadline_ns = iree_timeout_as_deadline_ns(timeout);

  // Busy-poll loop over the semaphore list.
  while (true) {
    iree_host_size_t satisfied = 0;
    for (iree_host_size_t i = 0; i < semaphore_list.count; ++i) {
      uint64_t current_value = 0;
      IREE_RETURN_IF_ERROR(iree_hal_semaphore_query(
          semaphore_list.semaphores[i], &current_value));
      if (current_value >= semaphore_list.payload_values[i]) {
        ++satisfied;
        if (wait_mode == IREE_HAL_WAIT_MODE_ANY) {
          return iree_ok_status();
        }
      }
    }
    if (satisfied == semaphore_list.count) {
      return iree_ok_status(); // WAIT_ALL satisfied
    }
    if (iree_time_now() >= deadline_ns) {
      return iree_status_from_code(IREE_STATUS_DEADLINE_EXCEEDED);
    }
  }
}

static iree_status_t iree_hal_tt_device_profiling_begin(
    iree_hal_device_t *, const iree_hal_device_profiling_options_t *) {
  return iree_ok_status();
}

static iree_status_t iree_hal_tt_device_profiling_flush(iree_hal_device_t *) {
  return iree_ok_status();
}

static iree_status_t iree_hal_tt_device_profiling_end(iree_hal_device_t *) {
  return iree_ok_status();
}
