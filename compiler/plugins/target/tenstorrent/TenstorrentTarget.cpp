// Copyright 2025 The tt-iree Authors
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "TenstorrentTarget.h"

#include "iree/compiler/Codegen/Common/Passes.h"
#include "iree/compiler/Dialect/HAL/Target/TargetRegistry.h"
#include "llvm/Support/raw_ostream.h"
#include "mlir/IR/Builders.h"

// TTEX builder — shared between compiler and runtime tests.
// This header is from runtime/src/iree/schema/ and is available
// after the schema CMake target is built.
#include "iree/schema/tt_executable_builder_util.h"

namespace mlir::iree_compiler::IREE::HAL {

// Must match IREE_HAL_TT_EXECUTABLE_FORMAT in tt_executable_cache.h.
static constexpr const char *kExecutableFormat = "tenstorrent-ttex-fb";

//===----------------------------------------------------------------------===//
// TenstorrentTargetDevice
//===----------------------------------------------------------------------===//

IREE::HAL::DeviceTargetAttr TenstorrentTargetDevice::getDefaultDeviceTarget(
    MLIRContext *context,
    const TargetRegistry &targetRegistry) const {
  Builder b(context);

  SmallVector<NamedAttribute> deviceConfigAttrs;
  // Future: arch, core_grid, memory config, etc.

  SmallVector<IREE::HAL::ExecutableTargetAttr> executableTargetAttrs;
  targetRegistry.getTargetBackend("tenstorrent")
      ->getDefaultExecutableTargets(context, "tenstorrent",
                                    b.getDictionaryAttr(deviceConfigAttrs),
                                    executableTargetAttrs);

  return IREE::HAL::DeviceTargetAttr::get(
      context, b.getStringAttr("tenstorrent"),
      b.getDictionaryAttr(deviceConfigAttrs), executableTargetAttrs);
}

//===----------------------------------------------------------------------===//
// getDefaultExecutableTargets
//===----------------------------------------------------------------------===//

void TenstorrentTargetBackend::getDefaultExecutableTargets(
    MLIRContext *context, StringRef deviceID,
    DictionaryAttr deviceConfigAttr,
    SmallVectorImpl<IREE::HAL::ExecutableTargetAttr> &executableTargetAttrs)
    const {
  Builder b(context);
  SmallVector<NamedAttribute> backendConfigAttrs;
  // Future: encode target arch, tile size, etc.

  executableTargetAttrs.push_back(IREE::HAL::ExecutableTargetAttr::get(
      context,
      b.getStringAttr("tenstorrent"),
      b.getStringAttr(kExecutableFormat),
      b.getDictionaryAttr(backendConfigAttrs)));
}

//===----------------------------------------------------------------------===//
// buildTranslationPassPipeline
//===----------------------------------------------------------------------===//

void TenstorrentTargetBackend::buildTranslationPassPipeline(
    IREE::HAL::ExecutableTargetAttr targetAttr,
    OpPassManager &passManager) {
  // Lower dispatch.workgroup_count_from_slice → concrete workgroup counts.
  // Without this, index-typed results reach VM conversion and cause a
  // vm.trunc.i64.i32 type error (op expects i64, gets index).
  // When there are no scf.forall distribution loops, ReconcileTranslationInfo
  // replaces the op with {1,1,1} workgroup counts (one tile per dispatch).
  passManager.addPass(createReconcileTranslationInfoPass());
}

//===----------------------------------------------------------------------===//
// serializeExecutable
//===----------------------------------------------------------------------===//
//
// Core of the compiler backend.  Converts ExecutableVariantOp into a
// TTEX FlatBuffer binary embedded in hal.executable.binary.
//
// PoC strategy:
//   - Walk export ops to discover entry points
//   - Map each to CUSTOM_SFPI_ADD (builtin 0)
//   - Build TTEX FlatBuffer via builder_util
//   - Embed binary data as DenseIntElementsAttr
//
// Reference: IREE CUDA backend serializes CUDAExecutableDef FlatBuffer
// with PTX + entry_points + block_sizes.  We do the same with TTEX,
// substituting builtin program IDs for PTX.

LogicalResult TenstorrentTargetBackend::serializeExecutable(
    const SerializationOptions &serOptions,
    IREE::HAL::ExecutableVariantOp variantOp,
    OpBuilder &executableBuilder) {
  auto loc = variantOp.getLoc();

  // ----------------------------------------------------------------
  // 1. Collect export ops (entry points).
  // ----------------------------------------------------------------
  auto exportOps = variantOp.getExportOps();

  if (exportOps.empty()) {
    return variantOp.emitError("no export ops found in variant");
  }

  // ----------------------------------------------------------------
  // 2. Build entry point descriptors.
  // ----------------------------------------------------------------
  // Keep std::string copies alive until FlatBuffer is built.
  std::vector<std::string> nameStorage;
  std::vector<tt_iree_ttex_entry_point_desc_t> entryDescs;

  for (auto exportOp : exportOps) {
    nameStorage.push_back(exportOp.getName().str());

    tt_iree_ttex_entry_point_desc_t desc = {};
    desc.name = nameStorage.back().c_str();

    // PoC: hardcoded for elementwise binary add.
    //   3 bindings: input0, input1, output
    //   builtin_program 0 = CUSTOM_SFPI_ADD
    //
    // Future: analyze inner func's linalg ops to determine:
    //   - binding_count from function signature
    //   - builtin_program from op pattern matching
    //   - workgroup_size from tiling decisions
    desc.constant_count = 0;
    desc.binding_count = 3;
    desc.flags = 0;
    desc.workgroup_size[0] = 1;
    desc.workgroup_size[1] = 1;
    desc.workgroup_size[2] = 1;
    desc.builtin_program = 0;  // CUSTOM_SFPI_ADD

    entryDescs.push_back(desc);
  }

  // ----------------------------------------------------------------
  // 3. Build TTEX FlatBuffer.
  // ----------------------------------------------------------------
  iree_byte_span_t ttexData = {};
  iree_status_t status = tt_iree_build_ttex_executable_def(
      iree_allocator_system(),
      entryDescs.size(), entryDescs.data(), &ttexData);

  if (!iree_status_is_ok(status)) {
    iree_status_ignore(status);
    return variantOp.emitError("failed to build TTEX FlatBuffer");
  }

  // ----------------------------------------------------------------
  // 4. Debug dump (optional).
  // ----------------------------------------------------------------
  if (options_.dumpTTEX) {
    std::string dumpPath =
        (variantOp.getSymName() + ".ttex").str();
    std::error_code ec;
    llvm::raw_fd_ostream out(dumpPath, ec);
    if (!ec) {
      out.write(reinterpret_cast<const char *>(ttexData.data),
                ttexData.data_length);
      llvm::errs() << "tt-iree: TTEX dumped to " << dumpPath
                   << " (" << ttexData.data_length << " bytes)\n";
    }
  }

  // ----------------------------------------------------------------
  // 5. Embed in hal.executable.binary.
  // ----------------------------------------------------------------
  // The binary becomes part of the vmfb.  At runtime:
  //   executable_cache.can_prepare_format("tenstorrent-ttex-fb") → true
  //   executable_cache.prepare_executable() → passes data to
  //     iree_hal_tt_executable_create() → parses TTEX → creates programs
  //
  auto dataRef = llvm::ArrayRef<uint8_t>(ttexData.data, ttexData.data_length);
  auto dataAttr = DenseIntElementsAttr::get(
      VectorType::get({static_cast<int64_t>(ttexData.data_length)},
                      IntegerType::get(variantOp.getContext(), 8)),
      dataRef);

  auto binaryOp = IREE::HAL::ExecutableBinaryOp::create(
      executableBuilder, loc,
      variantOp.getSymName(),
      variantOp.getTarget().getFormat(),
      dataAttr);
  (void)binaryOp;

  // ----------------------------------------------------------------
  // 6. Cleanup.
  // ----------------------------------------------------------------
  iree_allocator_free(iree_allocator_system(), ttexData.data);

  return success();
}

}  // namespace mlir::iree_compiler::IREE::HAL
