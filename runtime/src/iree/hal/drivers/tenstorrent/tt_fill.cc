// Copyright 2026 The tt-iree Authors
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/hal/drivers/tenstorrent/tt_fill.h"

#include <cerrno>
#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <stdexcept>
#include <system_error>
#include <vector>

#include "impl/kernels/kernel.hpp"
#include "impl/program/program_impl.hpp"
#include "iree/hal/drivers/tenstorrent/tt_fill_aot.h"
#include "iree/hal/drivers/tenstorrent/tt_runtime_profiler.h"
#include "tt-metalium/distributed.hpp"
#include "tt-metalium/experimental/offline_kernel_compile.hpp"
#include "tt-metalium/host_api.hpp"
#include "tt-metalium/tensor_accessor_args.hpp"

namespace iree::hal::tenstorrent {
namespace metal = tt::tt_metal;
namespace distributed = metal::distributed;
namespace fs = std::filesystem;

namespace {
struct TemporaryArtifacts {
  fs::path root;
  ~TemporaryArtifacts() {
    if (!root.empty()) {
      std::error_code error;
      fs::remove_all(root, error);
    }
  }
  void Materialize(const FillAOTBundle &bundle) {
    std::string name =
        (fs::temp_directory_path() / "tt-iree-fill-aot-XXXXXX").string();
    std::vector<char> writable(name.begin(), name.end());
    writable.push_back(0);
    if (!::mkdtemp(writable.data()))
      throw std::system_error(errno, std::generic_category(),
                              "fill AOT directory");
    root = writable.data();
    for (size_t i = 0; i < bundle.artifact_count; ++i) {
      const auto &artifact = bundle.artifacts[i];
      fs::path relative(artifact.path);
      if (relative.empty() || relative.is_absolute() || !artifact.size)
        throw std::runtime_error("invalid embedded fill AOT artifact");
      for (const auto &part : relative)
        if (part == "..")
          throw std::runtime_error("invalid fill AOT path");
      const auto output_path = root / relative;
      fs::create_directories(output_path.parent_path());
      std::ofstream output(output_path, std::ios::binary);
      output.write(reinterpret_cast<const char *>(artifact.data),
                   artifact.size);
      output.close();
      if (!output)
        throw std::runtime_error("failed to materialize fill AOT artifact");
    }
  }
};
} // namespace

struct DeviceFill::Impl {
  // Destroy Programs before their artifact directory is removed.
  TemporaryArtifacts artifacts;
  distributed::MeshWorkload workload;
  metal::KernelHandle kernel;
};

DeviceFill::DeviceFill() = default;
DeviceFill::~DeviceFill() = default;

iree_status_t DeviceFill::Execute(distributed::MeshDevice *device,
                                  distributed::MeshCommandQueue *queue,
                                  uint32_t address, uint32_t offset,
                                  uint32_t length, uint32_t pattern) {
  if (!length)
    return iree_ok_status();
  try {
    {
      RuntimeTransferStage setup(RuntimeTransferStageKind::DeviceSetup);
      if (!impl_) {
        uint8_t arch = device->arch() == tt::ARCH::BLACKHOLE     ? 2
                       : device->arch() == tt::ARCH::WORMHOLE_B0 ? 1
                                                                 : 0;
        const auto *bundle = GetFillAOTBundle(arch);
        if (!bundle)
          return iree_make_status(
              IREE_STATUS_UNIMPLEMENTED,
              "no embedded fill AOT for device architecture");
        auto candidate = std::make_unique<Impl>();
        candidate->artifacts.Materialize(*bundle);
        auto program = metal::CreateProgram();
        const metal::CoreRange core(metal::CoreCoord{0, 0});
        metal::CreateCircularBuffer(
            program, core,
            metal::CircularBufferConfig(2048, {{0, tt::DataFormat::UInt32}})
                .set_page_size(0, 2048));
        std::vector<uint32_t> args;
        metal::TensorAccessorArgs::create_dram_interleaved().append_to(args);
        args.back() = 2048;
        candidate->kernel = metal::CreateKernelFromString(
            program, bundle->source, core,
            metal::DataMovementConfig{.processor =
                                          metal::DataMovementProcessor::RISCV_0,
                                      .noc = metal::NOC::RISCV_0_default,
                                      .compile_args = std::move(args)});
        program.impl()
            .get_kernel(candidate->kernel)
            ->set_precompiled_config(
                metal::experimental::PrecompiledKernelConfig{
                    .precompiled_dir = candidate->artifacts.root.string(),
                    .fallback_policy = metal::experimental::
                        PrecompiledKernelConfig::FallbackPolicy::Error});
        candidate->workload.add_program(
            distributed::MeshCoordinateRange(device->shape()),
            std::move(program));
        impl_ = std::move(candidate);
        std::fprintf(
            stderr,
            "tt-iree: HAL fill uses cached device-native AOT workload\n");
      }
      auto &program = impl_->workload.get_programs().begin()->second;
      metal::SetRuntimeArgs(program, impl_->kernel, metal::CoreCoord{0, 0},
                            {address, offset, length, pattern});
    }
    {
      RuntimeTransferStage enqueue(RuntimeTransferStageKind::DeviceEnqueueWait);
      distributed::EnqueueMeshWorkload(*queue, impl_->workload, true);
    }
    if (auto *record = RuntimeTransferScope::CurrentRecord())
      record->device_fill_bytes += length;
    return iree_ok_status();
  } catch (const std::exception &error) {
    return iree_make_status(IREE_STATUS_INTERNAL, "device fill failed: %s",
                            error.what());
  } catch (...) {
    return iree_make_status(IREE_STATUS_INTERNAL, "device fill failed");
  }
}

} // namespace iree::hal::tenstorrent
