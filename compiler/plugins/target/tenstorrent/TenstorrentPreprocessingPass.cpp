// Copyright 2026 The tt-iree Authors
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "TenstorrentPreprocessingPass.h"

#include <cstdint>
#include <optional>

#include "iree/schema/tt_executable_builder_util.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/Interfaces/FunctionInterfaces.h"
#include "mlir/Pass/Pass.h"

namespace mlir::iree_compiler::IREE::HAL {
namespace {

struct MatmulPreprocessingContract {
  uint32_t builtinProgram = 0;
  bool hasStaticTileCounts = false;
  uint32_t mTiles = 0;
  uint32_t nTiles = 0;
  uint32_t kTiles = 0;
  bool isDynamic = false;
};

static bool isRank2Bf16Tensor(Type type) {
  auto rankedType = dyn_cast<RankedTensorType>(type);
  return rankedType && rankedType.getRank() == 2 &&
         rankedType.getElementType().isBF16();
}

static bool isDynamicDimOrTileAligned(int64_t dim) {
  return ShapedType::isDynamic(dim) || (dim > 0 && dim % 32 == 0);
}

static void clearTenstorrentAttrs(Operation *functionOp) {
  functionOp->removeAttr(kTenstorrentBuiltinProgramAttr);
  functionOp->removeAttr(kTenstorrentShapeModeAttr);
  functionOp->removeAttr(kTenstorrentMTilesAttr);
  functionOp->removeAttr(kTenstorrentNTilesAttr);
  functionOp->removeAttr(kTenstorrentKTilesAttr);
  functionOp->removeAttr(kTenstorrentPhaseAttr);
}

static std::optional<MatmulPreprocessingContract> getMatmulContract(
    Operation *functionOp) {
  linalg::MatmulOp matchedMatmulOp;
  int matmulCount = 0;
  functionOp->walk([&](linalg::MatmulOp matmulOp) {
    matchedMatmulOp = matmulOp;
    ++matmulCount;
  });

  if (matmulCount != 1) {
    return std::nullopt;
  }

  auto inputs = matchedMatmulOp.getInputs();
  auto outputs = matchedMatmulOp.getOutputs();
  if (inputs.size() != 2 || outputs.size() != 1) {
    return std::nullopt;
  }

  auto lhsType = dyn_cast<RankedTensorType>(inputs[0].getType());
  auto rhsType = dyn_cast<RankedTensorType>(inputs[1].getType());
  auto outType = dyn_cast<RankedTensorType>(outputs[0].getType());
  if (!lhsType || !rhsType || !outType || !isRank2Bf16Tensor(lhsType) ||
      !isRank2Bf16Tensor(rhsType) || !isRank2Bf16Tensor(outType)) {
    return std::nullopt;
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
    return std::nullopt;
  }
  if (!ShapedType::isDynamic(lhsM) && !ShapedType::isDynamic(outM) &&
      lhsM != outM) {
    return std::nullopt;
  }
  if (!ShapedType::isDynamic(lhsK) && !ShapedType::isDynamic(rhsK) &&
      lhsK != rhsK) {
    return std::nullopt;
  }
  if (!ShapedType::isDynamic(rhsN) && !ShapedType::isDynamic(outN) &&
      rhsN != outN) {
    return std::nullopt;
  }

  MatmulPreprocessingContract contract;
  bool isStatic32x32x32 = lhsType.hasStaticShape() && rhsType.hasStaticShape() &&
                          outType.hasStaticShape() &&
                          lhsType.getShape() == ArrayRef<int64_t>({32, 32}) &&
                          rhsType.getShape() == ArrayRef<int64_t>({32, 32}) &&
                          outType.getShape() == ArrayRef<int64_t>({32, 32});
  if (isStatic32x32x32) {
    contract.builtinProgram = static_cast<uint32_t>(
        TT_IREE_TTEX_BUILTIN_PROGRAM_BF16_MATMUL_32X32X32);
    contract.hasStaticTileCounts = true;
    contract.mTiles = 1;
    contract.nTiles = 1;
    contract.kTiles = 1;
    return contract;
  }

  contract.builtinProgram = static_cast<uint32_t>(
      TT_IREE_TTEX_BUILTIN_PROGRAM_BF16_MATMUL_TILED);
  contract.isDynamic = lhsType.isDynamicDim(0) || lhsType.isDynamicDim(1) ||
                       rhsType.isDynamicDim(1);
  if (!contract.isDynamic) {
    contract.hasStaticTileCounts = true;
    contract.mTiles = static_cast<uint32_t>(lhsM / 32);
    contract.kTiles = static_cast<uint32_t>(lhsK / 32);
    contract.nTiles = static_cast<uint32_t>(rhsN / 32);
  }
  return contract;
}

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
      auto contract = getMatmulContract(op);
      if (!contract) {
        return;
      }

      op->setAttr(
          kTenstorrentBuiltinProgramAttr,
          builder.getI32IntegerAttr(
              static_cast<int32_t>(contract->builtinProgram)));
      op->setAttr(kTenstorrentShapeModeAttr,
                  builder.getStringAttr(contract->isDynamic ? "dynamic"
                                                            : "static"));
      op->setAttr(kTenstorrentPhaseAttr, builder.getStringAttr("phase1"));
      if (contract->hasStaticTileCounts) {
        op->setAttr(kTenstorrentMTilesAttr,
                    builder.getI32IntegerAttr(
                        static_cast<int32_t>(contract->mTiles)));
        op->setAttr(kTenstorrentNTilesAttr,
                    builder.getI32IntegerAttr(
                        static_cast<int32_t>(contract->nTiles)));
        op->setAttr(kTenstorrentKTilesAttr,
                    builder.getI32IntegerAttr(
                        static_cast<int32_t>(contract->kTiles)));
      }
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

}  // namespace

std::unique_ptr<Pass> createTenstorrentAnnotateMatmulPreprocessingPass() {
  return std::make_unique<TenstorrentAnnotateMatmulPreprocessingPass>();
}

}  // namespace mlir::iree_compiler::IREE::HAL
