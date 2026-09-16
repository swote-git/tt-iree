// Copyright 2026 The tt-iree Authors
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "TenstorrentAOTCompiler.h"

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <map>
#include <memory>
#include <mutex>
#include <set>
#include <stdexcept>
#include <type_traits>
#include <variant>

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
  const fs::path& path() const { return path_; }

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
  ScopedFallbackEnvironmentVariable(const char* name,
                                    const std::string& fallback_value)
      : name_(name) {
    if (std::getenv(name)) return;
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

std::shared_ptr<Kernel> makeKernel(
    const std::string& source,
    const std::variant<DataMovementConfig, ComputeConfig>& config) {
  KernelSource kernel_source(source, KernelSource::SOURCE_CODE);
  CoreRangeSet core_ranges(
      std::set<CoreRange>{CoreRange(CoreCoord{0, 0}, CoreCoord{0, 0})});
  return std::visit(
      [&](const auto& typed_config) -> std::shared_ptr<Kernel> {
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
    const std::string& source,
    const std::variant<DataMovementConfig, ComputeConfig>& config,
    const std::vector<CBCompileConfig>& cb_configs,
    TenstorrentAOTTargetArch target_arch, const fs::path& output_dir) {
  std::shared_ptr<Kernel> kernel = makeKernel(source, config);
  tt::llrt::RunTimeOptions runtime_options;
  size_t matching_config_count = 0;
  tt::tt_metal::enumerate_offline_compile_device_configs(
      runtime_options, [&](const tt::tt_metal::JitDeviceConfig& device_config) {
        tt::ARCH expected_arch = toTTMetalArch(target_arch);
        if (device_config.arch != expected_arch) return;
        ++matching_config_count;
        tt::tt_metal::BuildEnvManager build_env_manager(*device_config.hal);
        build_env_manager.add_build_env(0, device_config, runtime_options);
        const tt::tt_metal::DeviceBuildEnv& device_build_env =
            build_env_manager.get_device_build_env(0);
        JitBuildOptions build_options(device_build_env.build_env);
        kernel->set_build_options(build_options);
        for (const CBCompileConfig& cb_config : cb_configs) {
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
          fs::copy_file(source_elf, output_elf,
                        fs::copy_options::overwrite_existing);
        }
      });
  if (matching_config_count == 0) {
    throw std::runtime_error(
        "tt-metal has no offline compile config for the requested target");
  }
}

std::vector<TenstorrentAOTArtifact> readArtifacts(const fs::path& root) {
  std::vector<TenstorrentAOTArtifact> artifacts;
  for (const fs::directory_entry& entry :
       fs::recursive_directory_iterator(root)) {
    if (!entry.is_regular_file()) continue;
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
      [](const auto& lhs, const auto& rhs) { return lhs.path < rhs.path; });
  return artifacts;
}

}  // namespace

static bool compileAOTKernelBundle(
    const char* reader_source, const DataMovementConfig& reader_config,
    const char* writer_source, const DataMovementConfig& writer_config,
    const char* compute_source, const ComputeConfig& compute_config,
    const std::vector<CBCompileConfig>& cb_configs,
    TenstorrentAOTTargetArch target_arch,
    TenstorrentAOTKernelBundle& out_bundle, std::string& error_message) {
  // TT-Metal's mock context and cache configuration are process-global.
  static std::mutex compile_mutex;
  std::lock_guard<std::mutex> compile_lock(compile_mutex);

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
  } catch (const std::exception& e) {
    error_message = e.what();
    return false;
  } catch (...) {
    error_message = "unknown tt-metal offline compiler failure";
    return false;
  }
}

bool compileAddAOT(TenstorrentAOTTargetArch target_arch,
                   TenstorrentAOTKernelBundle& out_bundle,
                   std::string& error_message) {
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
                      TenstorrentAOTKernelBundle& out_bundle,
                      std::string& error_message) {
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
      kMatmulComputeSource, ComputeConfig{}, cb_configs, target_arch,
      out_bundle, error_message);
}

}  // namespace mlir::iree_compiler::IREE::HAL
