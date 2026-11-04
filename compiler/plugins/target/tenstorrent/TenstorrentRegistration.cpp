// Copyright 2025 The tt-iree Authors
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// Plugin registration for the Tenstorrent target backend.
//
// Two-phase system:
//   1. Registration: compiler startup calls iree_register_compiler_plugin_*()
//   2. Activation: --iree-hal-target-backends=tenstorrent triggers session
//
#include "TenstorrentTarget.h"
#include "TenstorrentEncoding.h"
#include "TenstorrentPreprocessingPass.h"
#include "iree/compiler/Dialect/HAL/Target/TargetRegistry.h"
#include "iree/compiler/PluginAPI/Client.h"
#include "iree/compiler/Utils/OptionUtils.h"
#include "mlir/Pass/PassManager.h"

namespace mlir::iree_compiler {

namespace {

//===----------------------------------------------------------------------===//
// CLI Options
//===----------------------------------------------------------------------===//

struct TenstorrentOptions {
  void bindOptions(OptionsBinder &binder) {
    static llvm::cl::OptionCategory category("Tenstorrent HAL Target");

    binder.opt<std::string>(
        "iree-tenstorrent-device-id", deviceId,
        llvm::cl::cat(category),
        llvm::cl::desc("Target Tenstorrent device ('p100a', 'blackhole', "
                       "'wormhole', 'n150', or 'n300')"));

    binder.opt<bool>(
        "iree-tenstorrent-dump-ttex", dumpTTEX,
        llvm::cl::cat(category),
        llvm::cl::desc("Dump TTEX FlatBuffer binary for debugging"));
    binder.opt<bool>(
        "iree-tenstorrent-native-bf16-use-tensix", nativeBF16UseTensix,
        llvm::cl::cat(category),
        llvm::cl::desc("Run BF16 arithmetic on Tensix: contractions on the "
                       "FPU (HiFi4, FP32 Dst) with SFPU epilogues, max-pool, "
                       "7x7 sums and scalar division on the SFPU; false "
                       "selects the device-RISC-V FP32 reference. Both "
                       "preserve IR rounding"));
    binder.opt<bool>(
        "iree-tenstorrent-native-bf16-reuse-output-tiles", nativeBF16ReuseOutputTiles,
        llvm::cl::cat(category),
        llvm::cl::desc("Reuse BF16 operands across pairs of output tiles using "
                       "the existing two-page CBs; requires BF16 Tensix"));
    binder.opt<bool>(
        "iree-tenstorrent-bf16-tile-layout", bf16TileLayout,
        llvm::cl::cat(category),
        llvm::cl::desc("Data-tile BF16 matmuls into 32x32 tiles with Tensix "
                       "face order at dispatch boundaries; use with "
                       "--iree-opt-data-tiling"));
    binder.opt<bool>(
        "iree-tenstorrent-native-f32-use-tensix", nativeF32UseTensix,
        llvm::cl::cat(category),
        llvm::cl::desc("Use experimental v16 INT8-digit Tensix math instead "
                       "of the v15 device FP32 correctness path"));
  }

  std::string deviceId = "p100a";
  bool dumpTTEX = false;
  bool nativeF32UseTensix = false;
  bool nativeBF16UseTensix = true;
  bool nativeBF16ReuseOutputTiles = false;
  bool bf16TileLayout = false;
};

//===----------------------------------------------------------------------===//
// Plugin Session
//===----------------------------------------------------------------------===//

struct TenstorrentSession
    : public PluginSession<TenstorrentSession,
                           TenstorrentOptions,
                           PluginActivationPolicy::DefaultActivated> {
  static void registerPasses() {}

  void onRegisterDialects(DialectRegistry &registry) override {
    tenstorrent::registerTenstorrentEncoding(registry);
  }

  void extendPreprocessingPassPipeline(OpPassManager &passManager) override {
    passManager.addPass(
        IREE::HAL::createTenstorrentAnnotateMatmulPreprocessingPass());
  }

  void populateHALTargetDevices(
      IREE::HAL::TargetDeviceList &targets) override {
    targets.add("tenstorrent", [&]() {
      IREE::HAL::TenstorrentTargetOptions opts;
      opts.deviceId = options.deviceId;
      opts.dumpTTEX = options.dumpTTEX;
      opts.nativeF32UseTensix = options.nativeF32UseTensix;
      opts.nativeBF16UseTensix = options.nativeBF16UseTensix;
      opts.nativeBF16ReuseOutputTiles = options.nativeBF16ReuseOutputTiles;
      opts.bf16TileLayout = options.bf16TileLayout;
      return std::make_shared<IREE::HAL::TenstorrentTargetDevice>(opts);
    });
  }

  void populateHALTargetBackends(
      IREE::HAL::TargetBackendList &targets) override {
    targets.add("tenstorrent", [&]() {
      IREE::HAL::TenstorrentTargetOptions opts;
      opts.deviceId = options.deviceId;
      opts.dumpTTEX = options.dumpTTEX;
      opts.nativeF32UseTensix = options.nativeF32UseTensix;
      opts.nativeBF16UseTensix = options.nativeBF16UseTensix;
      opts.nativeBF16ReuseOutputTiles = options.nativeBF16ReuseOutputTiles;
      opts.bf16TileLayout = options.bf16TileLayout;
      return std::make_shared<IREE::HAL::TenstorrentTargetBackend>(opts);
    });
  }
};

}  // namespace

IREE_DEFINE_COMPILER_OPTION_FLAGS(TenstorrentOptions);

}  // namespace mlir::iree_compiler

extern "C" bool iree_register_compiler_plugin_hal_target_tenstorrent(
    mlir::iree_compiler::PluginRegistrar *registrar) {
  registrar->registerPlugin<mlir::iree_compiler::TenstorrentSession>(
      "hal_target_tenstorrent");
  return true;
}
