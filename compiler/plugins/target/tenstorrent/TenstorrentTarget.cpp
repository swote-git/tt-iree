// Copyright 2025 The tt-iree Authors
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "TenstorrentTarget.h"

#include "iree/compiler/Codegen/Common/Passes.h"
#include "iree/compiler/Dialect/HAL/IR/HALOps.h"
#include "iree/compiler/Dialect/HAL/Target/TargetRegistry.h"
#include "llvm/Support/raw_ostream.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/IR/Builders.h"

// TTEX builder — shared between compiler and runtime tests.
// This header is from runtime/src/iree/schema/ and is available
// after the schema CMake target is built.
#include "iree/schema/tt_executable_builder_util.h"

namespace mlir::iree_compiler::IREE::HAL {

// Must match IREE_HAL_TT_EXECUTABLE_FORMAT in tt_executable_cache.h.
static constexpr const char *kExecutableFormat = "tenstorrent-ttex-fb";

namespace {

static bool isStaticBf16TensorOfShape(Type type, ArrayRef<int64_t> shape) {
  auto rankedType = dyn_cast<RankedTensorType>(type);
  if (!rankedType || !rankedType.hasStaticShape() ||
      rankedType.getElementType().isBF16() == false) {
    return false;
  }
  return rankedType.getShape() == shape;
}

static bool isRank2Bf16Tensor(Type type) {
  auto rankedType = dyn_cast<RankedTensorType>(type);
  return rankedType && rankedType.getRank() == 2 &&
         rankedType.getElementType().isBF16();
}

static bool isDynamicDimOrTileAligned(int64_t dim) {
  return ShapedType::isDynamic(dim) || (dim > 0 && dim % 32 == 0);
}

struct BuiltinSelection {
  uint32_t program = 0;
  uint32_t mTiles = 0;
  uint32_t nTiles = 0;
  uint32_t kTiles = 0;
};

static FailureOr<BuiltinSelection> inferBuiltinProgram(func::FuncOp funcOp) {
  linalg::MatmulOp matchedMatmulOp;
  int matmulCount = 0;
  funcOp.walk([&](linalg::MatmulOp matmulOp) {
    matchedMatmulOp = matmulOp;
    ++matmulCount;
  });

  if (matmulCount > 0) {
    if (matmulCount != 1) {
      return failure();
    }

    auto inputs = matchedMatmulOp.getInputs();
    auto outputs = matchedMatmulOp.getOutputs();
    if (inputs.size() != 2 || outputs.size() != 1) {
      return failure();
    }

    if (!isStaticBf16TensorOfShape(inputs[0].getType(), {32, 32}) ||
        !isStaticBf16TensorOfShape(inputs[1].getType(), {32, 32}) ||
        !isStaticBf16TensorOfShape(outputs[0].getType(), {32, 32})) {
      auto lhsType = dyn_cast<RankedTensorType>(inputs[0].getType());
      auto rhsType = dyn_cast<RankedTensorType>(inputs[1].getType());
      auto outType = dyn_cast<RankedTensorType>(outputs[0].getType());
      if (!lhsType || !rhsType || !outType || lhsType.getRank() != 2 ||
          rhsType.getRank() != 2 || outType.getRank() != 2 ||
          !isRank2Bf16Tensor(lhsType) || !isRank2Bf16Tensor(rhsType) ||
          !isRank2Bf16Tensor(outType)) {
        return failure();
      }

      int64_t lhsM = lhsType.getDimSize(0);
      int64_t lhsK = lhsType.getDimSize(1);
      int64_t rhsK = rhsType.getDimSize(0);
      int64_t rhsN = rhsType.getDimSize(1);
      int64_t outM = outType.getDimSize(0);
      int64_t outN = outType.getDimSize(1);

      if (!isDynamicDimOrTileAligned(lhsM) ||
          !isDynamicDimOrTileAligned(lhsK) ||
          !isDynamicDimOrTileAligned(rhsK) ||
          !isDynamicDimOrTileAligned(rhsN) ||
          !isDynamicDimOrTileAligned(outM) ||
          !isDynamicDimOrTileAligned(outN)) {
        return failure();
      }
      if (!ShapedType::isDynamic(lhsM) && !ShapedType::isDynamic(outM) &&
          lhsM != outM) {
        return failure();
      }
      if (!ShapedType::isDynamic(lhsK) && !ShapedType::isDynamic(rhsK) &&
          lhsK != rhsK) {
        return failure();
      }
      if (!ShapedType::isDynamic(rhsN) && !ShapedType::isDynamic(outN) &&
          rhsN != outN) {
        return failure();
      }

      BuiltinSelection selection;
      selection.program =
          static_cast<uint32_t>(TT_IREE_TTEX_BUILTIN_PROGRAM_BF16_MATMUL_TILED);
      if (!lhsType.isDynamicDim(0) && !lhsType.isDynamicDim(1) &&
          !rhsType.isDynamicDim(1)) {
        selection.mTiles = static_cast<uint32_t>(lhsM / 32);
        selection.kTiles = static_cast<uint32_t>(lhsK / 32);
        selection.nTiles = static_cast<uint32_t>(rhsN / 32);
      }
      return selection;
    }

    return BuiltinSelection{
        static_cast<uint32_t>(
            TT_IREE_TTEX_BUILTIN_PROGRAM_BF16_MATMUL_32X32X32),
        1,
        1,
        1,
    };
  }

  int addCount = 0;
  funcOp.walk([&](arith::AddFOp) { ++addCount; });
  if (addCount == 1) {
    return BuiltinSelection{
        static_cast<uint32_t>(
            TT_IREE_TTEX_BUILTIN_PROGRAM_CUSTOM_SFPI_ADD),
        0,
        0,
        0,
    };
  }

  return failure();
}

static uint16_t inferBindingCount(IREE::HAL::ExecutableExportOp exportOp) {
  return static_cast<uint16_t>(exportOp.getLayout().getBindings().size());
}

static uint16_t inferConstantCount(IREE::HAL::ExecutableExportOp exportOp) {
  return static_cast<uint16_t>(exportOp.getLayout().getConstants());
}

static void inferWorkgroupSize(IREE::HAL::ExecutableExportOp exportOp,
                               tt_iree_ttex_entry_point_desc_t &desc) {
  desc.workgroup_size[0] = 1;
  desc.workgroup_size[1] = 1;
  desc.workgroup_size[2] = 1;

  if (auto workgroupSizeAttr = exportOp.getWorkgroupSize()) {
    auto workgroupSizeValues = workgroupSizeAttr->getValue();
    desc.workgroup_size[0] =
        static_cast<uint32_t>(cast<IntegerAttr>(workgroupSizeValues[0]).getInt());
    desc.workgroup_size[1] =
        static_cast<uint32_t>(cast<IntegerAttr>(workgroupSizeValues[1]).getInt());
    desc.workgroup_size[2] =
        static_cast<uint32_t>(cast<IntegerAttr>(workgroupSizeValues[2]).getInt());
  }
}

}  // namespace

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
  // Lower dispatch.workgroup_count_from_slice to concrete workgroup counts.
  // IREE v3.11 split hint resolution out of ReconcileTranslationInfo, so both
  // passes are required to materialize the default {1, 1, 1} launch shape.
  passManager.addPass(createReconcileTranslationInfoPass());
  passManager.addPass(createResolveWorkgroupCountHintsPass());
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
  ModuleOp innerModuleOp = variantOp.getInnerModule();

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

    auto funcOp = innerModuleOp.lookupSymbol<func::FuncOp>(exportOp.getName());
    if (!funcOp) {
      return exportOp.emitError("missing inner function for executable export");
    }

    FailureOr<BuiltinSelection> builtinSelection = inferBuiltinProgram(funcOp);
    if (failed(builtinSelection)) {
      return exportOp.emitError(
          "unsupported Tenstorrent dispatch; supported builtins are a single "
          "arith.addf path, a single static linalg.matmul on "
          "tensor<32x32xbf16>, and tile-aligned rank-2 bf16 linalg.matmul "
          "(static or dynamic)");
    }

    desc.constant_count = inferConstantCount(exportOp);
    desc.binding_count = inferBindingCount(exportOp);
    desc.flags = 0;
    inferWorkgroupSize(exportOp, desc);
    desc.builtin_program = builtinSelection->program;
    desc.builtin_m_tiles = builtinSelection->mTiles;
    desc.builtin_n_tiles = builtinSelection->nTiles;
    desc.builtin_k_tiles = builtinSelection->kTiles;

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
