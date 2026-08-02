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
        llvm::cl::desc("Target Tenstorrent device (e.g., 'p100a')"),
        llvm::cl::init(""));

    binder.opt<bool>(
        "iree-tenstorrent-dump-ttex", dumpTTEX,
        llvm::cl::cat(category),
        llvm::cl::desc("Dump TTEX FlatBuffer binary for debugging"),
        llvm::cl::init(false));
  }

  std::string deviceId;
  bool dumpTTEX = false;
};

//===----------------------------------------------------------------------===//
// Plugin Session
//===----------------------------------------------------------------------===//

struct TenstorrentSession
    : public PluginSession<TenstorrentSession,
                           TenstorrentOptions,
                           PluginActivationPolicy::DefaultActivated> {
  static void registerPasses() {}

  void extendPreprocessingPassPipeline(OpPassManager &passManager) override {
    passManager.addPass(
        IREE::HAL::createTenstorrentAnnotateMatmulPreprocessingPass());
  }

  void populateHALTargetDevices(IREE::HAL::TargetDeviceList &targets) {
    IREE::HAL::TenstorrentTargetOptions opts;
    opts.deviceId = options.deviceId;
    opts.dumpTTEX = options.dumpTTEX;

    targets.add("tenstorrent",
                [=]() {
                  return std::make_shared<IREE::HAL::TenstorrentTargetDevice>(
                      opts);
                });
  }

  void populateHALTargetBackends(IREE::HAL::TargetBackendList &targets) {
    IREE::HAL::TenstorrentTargetOptions opts;
    opts.deviceId = options.deviceId;
    opts.dumpTTEX = options.dumpTTEX;

    targets.add("tenstorrent",
                [=]() {
                  return std::make_shared<IREE::HAL::TenstorrentTargetBackend>(
                      opts);
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
