// Copyright 2025 The tt-iree Authors
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// Tenstorrent HAL target backend for the IREE compiler.
//
// Dispatch semantics are classified from executable-source IR and serialized
// as TTEX. Supported static matmul-like dispatches carry compiler-generated,
// offline-compiled TT-Metal kernels.

#ifndef TT_IREE_COMPILER_PLUGINS_TARGET_TENSTORRENT_TARGET_H_
#define TT_IREE_COMPILER_PLUGINS_TARGET_TENSTORRENT_TARGET_H_

#include <string>

#include "iree/compiler/Dialect/HAL/Target/TargetBackend.h"
#include "iree/compiler/Dialect/HAL/Target/TargetDevice.h"

namespace mlir::iree_compiler::IREE::HAL {

//===----------------------------------------------------------------------===//
// Options — populated from CLI flags via PluginSession
//===----------------------------------------------------------------------===//

struct TenstorrentTargetOptions {
  std::string deviceId = "p100a";
  bool dumpTTEX = false;   // dump TTEX binary for debugging
  bool nativeF32UseTensix = false; // experimental INT8-digit v16 path
  bool nativeBF16UseTensix = false; // v18 HiFi4 contraction, IR epilogue
};

//===----------------------------------------------------------------------===//
// TargetDevice — creates DeviceTargetAttr
//===----------------------------------------------------------------------===//

class TenstorrentTargetDevice final : public TargetDevice {
public:
  TenstorrentTargetDevice(const TenstorrentTargetOptions &options)
      : options_(options) {}

  IREE::HAL::DeviceTargetAttr
  getDefaultDeviceTarget(MLIRContext *context,
                         const TargetRegistry &targetRegistry) const override;

private:
  TenstorrentTargetOptions options_;
};

//===----------------------------------------------------------------------===//
// TargetBackend — compilation + serialization
//===----------------------------------------------------------------------===//

class TenstorrentTargetBackend final : public TargetBackend {
public:
  TenstorrentTargetBackend(const TenstorrentTargetOptions &options)
      : options_(options) {}

  std::string getLegacyDefaultDeviceID() const override {
    return "tenstorrent";
  }

  void getDefaultExecutableTargets(
      MLIRContext *context, StringRef deviceID,
      DictionaryAttr deviceConfigAttr,
      SmallVectorImpl<IREE::HAL::ExecutableTargetAttr> &executableTargetAttrs)
      const override;

  void buildTranslationPassPipeline(
      IREE::HAL::ExecutableTargetAttr targetAttr,
      OpPassManager &passManager) override;

  LogicalResult serializeExecutable(
      const SerializationOptions &options,
      IREE::HAL::ExecutableVariantOp variantOp,
      OpBuilder &executableBuilder) override;

private:
  TenstorrentTargetOptions options_;
};

}  // namespace mlir::iree_compiler::IREE::HAL

#endif  // TT_IREE_COMPILER_PLUGINS_TARGET_TENSTORRENT_TARGET_H_
