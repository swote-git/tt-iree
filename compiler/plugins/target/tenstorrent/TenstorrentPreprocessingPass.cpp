// Copyright 2026 The tt-iree Authors
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "TenstorrentPreprocessingPass.h"

#include "iree/schema/tt_executable_builder_util.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/Interfaces/FunctionInterfaces.h"
#include "mlir/Pass/Pass.h"
#include <cstdint>

namespace mlir::iree_compiler::IREE::HAL {
namespace {

static bool isRank2Bf16Tensor(Type type) {
  auto rankedType = dyn_cast<RankedTensorType>(type);
  return rankedType && rankedType.getRank() == 2 &&
         rankedType.getElementType().isBF16();
}

static bool isDynamicDimOrTileAligned(int64_t dim) {
  return ShapedType::isDynamic(dim) || (dim > 0 && dim % 32 == 0);
}

static void clearTenstorrentAttrs(Operation *op) {
  op->removeAttr(kTenstorrentBuiltinProgramAttr);
  op->removeAttr(kTenstorrentShapeModeAttr);
  op->removeAttr(kTenstorrentMTilesAttr);
  op->removeAttr(kTenstorrentNTilesAttr);
  op->removeAttr(kTenstorrentKTilesAttr);
  op->removeAttr(kTenstorrentPhaseAttr);
}

static void
writeTenstorrentMatmulSelection(Operation *op,
                                const TenstorrentMatmulSelection &selection,
                                Builder &builder) {
  op->setAttr(kTenstorrentBuiltinProgramAttr,
              builder.getI32IntegerAttr(
                  static_cast<int32_t>(selection.builtinProgram)));
  op->setAttr(
      kTenstorrentShapeModeAttr,
      builder.getStringAttr(selection.isDynamic ? "dynamic" : "static"));
  op->setAttr(kTenstorrentPhaseAttr, builder.getStringAttr("selection-v1"));
  if (selection.hasStaticTileCounts) {
    op->setAttr(
        kTenstorrentMTilesAttr,
        builder.getI32IntegerAttr(static_cast<int32_t>(selection.mTiles)));
    op->setAttr(
        kTenstorrentNTilesAttr,
        builder.getI32IntegerAttr(static_cast<int32_t>(selection.nTiles)));
    op->setAttr(
        kTenstorrentKTilesAttr,
        builder.getI32IntegerAttr(static_cast<int32_t>(selection.kTiles)));
  }
}

} // namespace

FailureOr<TenstorrentMatmulSelection>
matchTenstorrentBf16Matmul(linalg::MatmulOp matmulOp) {
  auto inputs = matmulOp.getInputs();
  auto outputs = matmulOp.getOutputs();
  if (inputs.size() != 2 || outputs.size() != 1) {
    return failure();
  }

  auto lhsType = dyn_cast<RankedTensorType>(inputs[0].getType());
  auto rhsType = dyn_cast<RankedTensorType>(inputs[1].getType());
  auto outType = dyn_cast<RankedTensorType>(outputs[0].getType());
  if (!lhsType || !rhsType || !outType || !isRank2Bf16Tensor(lhsType) ||
      !isRank2Bf16Tensor(rhsType) || !isRank2Bf16Tensor(outType)) {
    return failure();
  }

  int64_t lhsM = lhsType.getDimSize(0);
  int64_t lhsK = lhsType.getDimSize(1);
  int64_t rhsK = rhsType.getDimSize(0);
  int64_t rhsN = rhsType.getDimSize(1);
  int64_t outM = outType.getDimSize(0);
  int64_t outN = outType.getDimSize(1);

  if (!isDynamicDimOrTileAligned(lhsM) || !isDynamicDimOrTileAligned(lhsK) ||
      !isDynamicDimOrTileAligned(rhsK) || !isDynamicDimOrTileAligned(rhsN) ||
      !isDynamicDimOrTileAligned(outM) || !isDynamicDimOrTileAligned(outN)) {
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

  TenstorrentMatmulSelection selection;
  bool isStatic32x32x32 = lhsType.hasStaticShape() &&
                          rhsType.hasStaticShape() &&
                          outType.hasStaticShape() &&
                          lhsType.getShape() == ArrayRef<int64_t>({32, 32}) &&
                          rhsType.getShape() == ArrayRef<int64_t>({32, 32}) &&
                          outType.getShape() == ArrayRef<int64_t>({32, 32});
  if (isStatic32x32x32) {
    selection.builtinProgram = static_cast<uint32_t>(
        TT_IREE_TTEX_BUILTIN_PROGRAM_BF16_MATMUL_32X32X32);
    selection.hasStaticTileCounts = true;
    selection.mTiles = 1;
    selection.nTiles = 1;
    selection.kTiles = 1;
    return selection;
  }

  selection.builtinProgram =
      static_cast<uint32_t>(TT_IREE_TTEX_BUILTIN_PROGRAM_BF16_MATMUL_TILED);
  selection.isDynamic = lhsType.isDynamicDim(0) || lhsType.isDynamicDim(1) ||
                        rhsType.isDynamicDim(1);
  if (!selection.isDynamic) {
    selection.hasStaticTileCounts = true;
    selection.mTiles = static_cast<uint32_t>(lhsM / 32);
    selection.kTiles = static_cast<uint32_t>(lhsK / 32);
    selection.nTiles = static_cast<uint32_t>(rhsN / 32);
  }
  return selection;
}

FailureOr<TenstorrentMatmulSelection>
readTenstorrentMatmulSelection(linalg::MatmulOp matmulOp) {
  FailureOr<TenstorrentMatmulSelection> expected =
      matchTenstorrentBf16Matmul(matmulOp);
  if (failed(expected))
    return failure();

  auto programAttr =
      matmulOp->getAttrOfType<IntegerAttr>(kTenstorrentBuiltinProgramAttr);
  auto shapeModeAttr =
      matmulOp->getAttrOfType<StringAttr>(kTenstorrentShapeModeAttr);
  auto phaseAttr = matmulOp->getAttrOfType<StringAttr>(kTenstorrentPhaseAttr);
  if (!programAttr || !shapeModeAttr || !phaseAttr ||
      phaseAttr.getValue() != "selection-v1") {
    return failure();
  }

  const bool annotatedDynamic = shapeModeAttr.getValue() == "dynamic";
  if ((!annotatedDynamic && shapeModeAttr.getValue() != "static") ||
      programAttr.getValue().getZExtValue() != expected->builtinProgram ||
      annotatedDynamic != expected->isDynamic) {
    return failure();
  }

  auto mTilesAttr =
      matmulOp->getAttrOfType<IntegerAttr>(kTenstorrentMTilesAttr);
  auto nTilesAttr =
      matmulOp->getAttrOfType<IntegerAttr>(kTenstorrentNTilesAttr);
  auto kTilesAttr =
      matmulOp->getAttrOfType<IntegerAttr>(kTenstorrentKTilesAttr);
  if (expected->hasStaticTileCounts) {
    if (!mTilesAttr || !nTilesAttr || !kTilesAttr ||
        mTilesAttr.getValue().getZExtValue() != expected->mTiles ||
        nTilesAttr.getValue().getZExtValue() != expected->nTiles ||
        kTilesAttr.getValue().getZExtValue() != expected->kTiles) {
      return failure();
    }
  } else if (mTilesAttr || nTilesAttr || kTilesAttr) {
    return failure();
  }
  return *expected;
}

namespace {

struct TenstorrentAnnotateMatmulPreprocessingPass
    : public PassWrapper<TenstorrentAnnotateMatmulPreprocessingPass,
                         OperationPass<ModuleOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(
      TenstorrentAnnotateMatmulPreprocessingPass)

  void runOnOperation() override {
    ModuleOp moduleOp = getOperation();
    Builder builder(moduleOp.getContext());

    moduleOp.walk([&](FunctionOpInterface functionOp) {
      Operation *op = functionOp.getOperation();
      if (functionOp.isExternal()) {
        return;
      }
      clearTenstorrentAttrs(op);
      linalg::MatmulOp matchedMatmulOp;
      int matmulCount = 0;
      op->walk([&](linalg::MatmulOp matmulOp) {
        clearTenstorrentAttrs(matmulOp);
        matchedMatmulOp = matmulOp;
        ++matmulCount;
      });
      if (matmulCount != 1) {
        return;
      }
      FailureOr<TenstorrentMatmulSelection> selection =
          matchTenstorrentBf16Matmul(matchedMatmulOp);
      if (failed(selection))
        return;

      writeTenstorrentMatmulSelection(op, *selection, builder);
      writeTenstorrentMatmulSelection(matchedMatmulOp, *selection, builder);
    });
  }

  StringRef getArgument() const final {
    return "iree-tenstorrent-annotate-matmul-preprocessing";
  }

  StringRef getDescription() const final {
    return "Annotates Tenstorrent-supported rank-2 bf16 matmul candidates "
           "during preprocessing.";
  }
};

} // namespace

std::unique_ptr<Pass> createTenstorrentAnnotateMatmulPreprocessingPass() {
  return std::make_unique<TenstorrentAnnotateMatmulPreprocessingPass>();
}

} // namespace mlir::iree_compiler::IREE::HAL
