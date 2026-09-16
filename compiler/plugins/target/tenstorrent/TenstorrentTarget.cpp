// Copyright 2025 The tt-iree Authors
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "TenstorrentTarget.h"
#ifdef TT_IREE_ENABLE_AOT
#include "TenstorrentAOTCompiler.h"
#endif

#include <algorithm>
#include <cmath>
#include <iterator>
#include <limits>
#include <optional>
#include <vector>

#include "iree/compiler/Codegen/Common/Passes.h"
#include "iree/compiler/Codegen/Dialect/Codegen/IR/IREECodegenOps.h"
#include "iree/compiler/Dialect/HAL/IR/HALOps.h"
#include "iree/compiler/Dialect/HAL/Target/TargetRegistry.h"
#include "iree/compiler/Dialect/TensorExt/IR/TensorExtOps.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/IR/Builders.h"
#include "mlir/Pass/Pass.h"
#include "llvm/Support/raw_ostream.h"

// TTEX builder — shared between compiler and runtime tests.
// This header is from runtime/src/iree/schema/ and is available
// after the schema CMake target is built.
#include "iree/schema/tt_executable_builder_util.h"

namespace mlir::iree_compiler::IREE::HAL {

// Must match IREE_HAL_TT_EXECUTABLE_FORMAT in tt_executable_cache.h.
static constexpr const char *kExecutableFormat = "tenstorrent-ttex-fb";

namespace {

static StringRef getTargetDeviceIdOrDefault(StringRef deviceId) {
  return deviceId.empty() ? StringRef("p100a") : deviceId;
}

static std::optional<uint8_t> inferTargetArch(StringRef deviceId) {
  std::string normalized = getTargetDeviceIdOrDefault(deviceId).lower();
  if (normalized == "p100a" || normalized == "blackhole" ||
      normalized == "bh") {
    return TT_IREE_TTEX_TARGET_ARCH_BLACKHOLE;
  }
  if (normalized == "wormhole" || normalized == "wormhole_b0" ||
      normalized == "n150" || normalized == "n300") {
    return TT_IREE_TTEX_TARGET_ARCH_WORMHOLE_B0;
  }
  return std::nullopt;
}

static StringRef getTargetArchName(StringRef deviceId) {
  std::optional<uint8_t> targetArch = inferTargetArch(deviceId);
  if (!targetArch)
    return "unknown";
  return *targetArch == TT_IREE_TTEX_TARGET_ARCH_BLACKHOLE ? "blackhole"
                                                           : "wormhole_b0";
}

static std::optional<uint8_t>
inferExecutableTargetArch(IREE::HAL::ExecutableTargetAttr targetAttr,
                          StringRef fallbackDeviceId) {
  if (DictionaryAttr config = targetAttr.getConfiguration()) {
    if (auto targetArch = config.getAs<StringAttr>("target_arch")) {
      return inferTargetArch(targetArch.getValue());
    }
  }
  return inferTargetArch(fallbackDeviceId);
}

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
  uint8_t dispatchKind = TT_IREE_TTEX_DISPATCH_KIND_BUILTIN;
  tt_iree_ttex_matmul_like_desc_t matmulLike = {};
  tt_iree_ttex_conv2d_desc_t conv2d = {};
  tt_iree_ttex_padded_copy_desc_t paddedCopy = {};
  tt_iree_ttex_max_pool_2d_desc_t maxPool = {};
  tt_iree_ttex_global_sum_desc_t globalSum = {};
  tt_iree_ttex_scalar_div_desc_t scalarDiv = {};
  tt_iree_ttex_linear_desc_t linear = {};
  std::vector<float> matmulEpilogueScales;
  std::vector<float> matmulEpilogueBiases;
  std::vector<float> convEpilogueScales;
  std::vector<float> convEpilogueBiases;
};

#ifdef TT_IREE_ENABLE_AOT
struct SerializedAOTBundleStorage {
  TenstorrentAOTKernelBundle bundle;
  std::vector<tt_iree_ttex_kernel_source_desc_t> kernelDescs;
  std::vector<tt_iree_ttex_artifact_file_desc_t> artifactDescs;
  tt_iree_ttex_aot_kernel_bundle_desc_t desc = {};

  void finalize() {
    kernelDescs.reserve(bundle.kernels.size());
    for (const TenstorrentAOTKernelSource &kernel : bundle.kernels) {
      kernelDescs.push_back({kernel.role, kernel.source.c_str()});
    }
    artifactDescs.reserve(bundle.artifacts.size());
    for (const TenstorrentAOTArtifact &artifact : bundle.artifacts) {
      artifactDescs.push_back(
          {artifact.path.c_str(), artifact.data.data(), artifact.data.size()});
    }
    desc.target_arch = static_cast<uint8_t>(bundle.targetArch);
    desc.kernel_count = kernelDescs.size();
    desc.kernels = kernelDescs.data();
    desc.artifact_count = artifactDescs.size();
    desc.artifacts = artifactDescs.data();
  }
};
#endif

static std::optional<double> getConstantFloat(Value value) {
  auto constantOp = value.getDefiningOp<arith::ConstantOp>();
  if (!constantOp)
    return std::nullopt;
  auto floatAttr = dyn_cast<FloatAttr>(constantOp.getValue());
  if (!floatAttr)
    return std::nullopt;
  return floatAttr.getValueAsDouble();
}

static bool matchesMultiplyAddBody(linalg::GenericOp genericOp) {
  Block &body = genericOp.getRegion().front();
  if (body.getNumArguments() != 3)
    return false;
  auto yieldOp = dyn_cast<linalg::YieldOp>(body.getTerminator());
  if (!yieldOp || yieldOp.getValues().size() != 1)
    return false;
  auto addOp = yieldOp.getValues()[0].getDefiningOp<arith::AddFOp>();
  if (!addOp)
    return false;

  Value multiplyValue;
  if (addOp.getLhs() == body.getArgument(2)) {
    multiplyValue = addOp.getRhs();
  } else if (addOp.getRhs() == body.getArgument(2)) {
    multiplyValue = addOp.getLhs();
  } else {
    return false;
  }
  auto multiplyOp = multiplyValue.getDefiningOp<arith::MulFOp>();
  if (!multiplyOp)
    return false;
  return (multiplyOp.getLhs() == body.getArgument(0) &&
          multiplyOp.getRhs() == body.getArgument(1)) ||
         (multiplyOp.getLhs() == body.getArgument(1) &&
          multiplyOp.getRhs() == body.getArgument(0));
}

static bool matchesProjectionIndexingMaps(linalg::GenericOp genericOp) {
  MLIRContext *context = genericOp.getContext();
  AffineExpr d0 = getAffineDimExpr(0, context);
  AffineExpr d1 = getAffineDimExpr(1, context);
  AffineExpr d2 = getAffineDimExpr(2, context);
  AffineExpr d3 = getAffineDimExpr(3, context);
  SmallVector<AffineMap, 3> expectedMaps = {
      AffineMap::get(4, 0, {d3, d1, d2}, context),
      AffineMap::get(4, 0, {d0, d3}, context),
      AffineMap::get(4, 0, {d0, d1, d2}, context),
  };
  if (genericOp.getIndexingMapsArray() != expectedMaps)
    return false;
  auto iteratorTypes = genericOp.getIteratorTypesArray();
  return iteratorTypes.size() == 4 &&
         iteratorTypes[0] == utils::IteratorType::parallel &&
         iteratorTypes[1] == utils::IteratorType::parallel &&
         iteratorTypes[2] == utils::IteratorType::parallel &&
         iteratorTypes[3] == utils::IteratorType::reduction;
}

static bool
matchesFlattenedProjectionIndexingMaps(linalg::GenericOp genericOp) {
  MLIRContext *context = genericOp.getContext();
  AffineExpr d0 = getAffineDimExpr(0, context);
  AffineExpr d1 = getAffineDimExpr(1, context);
  AffineExpr d2 = getAffineDimExpr(2, context);
  SmallVector<AffineMap, 3> expectedMaps = {
      AffineMap::get(3, 0, {d2, d1}, context),
      AffineMap::get(3, 0, {d0, d2}, context),
      AffineMap::get(3, 0, {d0, d1}, context),
  };
  if (genericOp.getIndexingMapsArray() != expectedMaps)
    return false;
  auto iteratorTypes = genericOp.getIteratorTypesArray();
  return iteratorTypes.size() == 3 &&
         iteratorTypes[0] == utils::IteratorType::parallel &&
         iteratorTypes[1] == utils::IteratorType::parallel &&
         iteratorTypes[2] == utils::IteratorType::reduction;
}

static std::optional<uint32_t> inferConv2DStride(linalg::GenericOp genericOp) {
  MLIRContext *context = genericOp.getContext();
  AffineExpr d0 = getAffineDimExpr(0, context);
  AffineExpr d1 = getAffineDimExpr(1, context);
  AffineExpr d2 = getAffineDimExpr(2, context);
  AffineExpr d3 = getAffineDimExpr(3, context);
  AffineExpr d4 = getAffineDimExpr(4, context);
  AffineExpr d5 = getAffineDimExpr(5, context);
  auto iteratorTypes = genericOp.getIteratorTypesArray();
  if (iteratorTypes.size() != 6 ||
      iteratorTypes[0] != utils::IteratorType::parallel ||
      iteratorTypes[1] != utils::IteratorType::parallel ||
      iteratorTypes[2] != utils::IteratorType::parallel ||
      iteratorTypes[3] != utils::IteratorType::reduction ||
      iteratorTypes[4] != utils::IteratorType::reduction ||
      iteratorTypes[5] != utils::IteratorType::reduction) {
    return std::nullopt;
  }

  for (uint32_t stride : {1u, 2u}) {
    SmallVector<AffineMap, 3> expectedMaps = {
        AffineMap::get(6, 0, {d3, d1 * stride + d4, d2 * stride + d5}, context),
        AffineMap::get(6, 0, {d0, d3, d4, d5}, context),
        AffineMap::get(6, 0, {d0, d1, d2}, context),
    };
    if (genericOp.getIndexingMapsArray() == expectedMaps)
      return stride;
  }
  return std::nullopt;
}

static FailureOr<uint16_t>
inferLoadBinding(Value value, IREE::TensorExt::DispatchTensorLoadOp &loadOp) {
  loadOp = value.getDefiningOp<IREE::TensorExt::DispatchTensorLoadOp>();
  if (!loadOp)
    return failure();
  auto subspanOp =
      loadOp.getSource().getDefiningOp<IREE::HAL::InterfaceBindingSubspanOp>();
  if (!subspanOp || subspanOp.getBinding().getZExtValue() >
                        std::numeric_limits<uint16_t>::max()) {
    return failure();
  }
  return static_cast<uint16_t>(subspanOp.getBinding().getZExtValue());
}

struct InferredByteOffset {
  uint64_t staticOffset = 0;
  uint16_t constantOrdinalPlusOne = 0;
};

static FailureOr<InferredByteOffset>
inferByteOffset(IREE::HAL::InterfaceBindingSubspanOp subspanOp) {
  if (!subspanOp || !subspanOp.getByteOffset())
    return InferredByteOffset{};

  Value value = subspanOp.getByteOffset();
  for (unsigned depth = 0; depth < 4; ++depth) {
    if (auto constantOp = value.getDefiningOp<arith::ConstantOp>()) {
      auto integerAttr = dyn_cast<IntegerAttr>(constantOp.getValue());
      if (!integerAttr || integerAttr.getValue().isNegative() ||
          integerAttr.getValue().getActiveBits() > 64) {
        return failure();
      }
      return InferredByteOffset{integerAttr.getValue().getZExtValue(), 0};
    }
    if (auto constantLoad =
            value.getDefiningOp<IREE::HAL::InterfaceConstantLoadOp>()) {
      uint64_t ordinal = constantLoad.getOrdinal().getZExtValue();
      if (ordinal >= std::numeric_limits<uint16_t>::max())
        return failure();
      return InferredByteOffset{0, static_cast<uint16_t>(ordinal + 1)};
    }

    Operation *definingOp = value.getDefiningOp();
    auto result = dyn_cast<OpResult>(value);
    if (!definingOp || !result)
      return failure();
    StringRef opName = definingOp->getName().getStringRef();
    if (opName == "util.assume.int") {
      unsigned resultNumber = result.getResultNumber();
      if (resultNumber >= definingOp->getNumOperands())
        return failure();
      value = definingOp->getOperand(resultNumber);
      continue;
    }
    if ((opName == "arith.index_castui" || opName == "arith.index_cast") &&
        definingOp->getNumOperands() == 1) {
      value = definingOp->getOperand(0);
      continue;
    }
    return failure();
  }
  return failure();
}

static LogicalResult
copyStaticOffsets(ArrayRef<OpFoldResult> mixedOffsets, uint8_t rank,
                  uint32_t (&outOffsets)[TT_IREE_TTEX_MAX_TENSOR_RANK]) {
  if (mixedOffsets.empty())
    return success();
  if (mixedOffsets.size() != rank)
    return failure();
  for (auto [index, offset] : llvm::enumerate(mixedOffsets)) {
    auto attr = dyn_cast_if_present<IntegerAttr>(offset.dyn_cast<Attribute>());
    if (!attr || attr.getInt() < 0 ||
        static_cast<uint64_t>(attr.getInt()) >
            std::numeric_limits<uint32_t>::max()) {
      return failure();
    }
    outOffsets[index] = static_cast<uint32_t>(attr.getInt());
  }
  return success();
}

static LogicalResult
copyStaticStrides(ArrayRef<OpFoldResult> mixedStrides, uint8_t rank,
                  uint32_t (&outStrides)[TT_IREE_TTEX_MAX_TENSOR_RANK]) {
  for (uint8_t i = 0; i < TT_IREE_TTEX_MAX_TENSOR_RANK; ++i)
    outStrides[i] = 1;
  if (mixedStrides.empty())
    return success();
  if (mixedStrides.size() != rank)
    return failure();
  for (auto [index, stride] : llvm::enumerate(mixedStrides)) {
    auto attr = dyn_cast_if_present<IntegerAttr>(stride.dyn_cast<Attribute>());
    if (!attr || attr.getInt() <= 0 ||
        static_cast<uint64_t>(attr.getInt()) >
            std::numeric_limits<uint32_t>::max()) {
      return failure();
    }
    outStrides[index] = static_cast<uint32_t>(attr.getInt());
  }
  return success();
}

static LogicalResult populateTensorDesc(RankedTensorType logicalType,
                                        ArrayRef<int64_t> storageShape,
                                        ArrayRef<OpFoldResult> mixedOffsets,
                                        ArrayRef<OpFoldResult> mixedStrides,
                                        uint16_t binding,
                                        InferredByteOffset byteOffset,
                                        uint8_t layout,
                                        tt_iree_ttex_tensor_desc_t &desc) {
  if (!logicalType || !logicalType.hasStaticShape() ||
      !logicalType.getElementType().isF32() || logicalType.getRank() == 0 ||
      logicalType.getRank() > TT_IREE_TTEX_MAX_TENSOR_RANK ||
      storageShape.size() != static_cast<size_t>(logicalType.getRank())) {
    return failure();
  }

  desc.binding = binding;
  desc.element_type = TT_IREE_TTEX_TENSOR_ELEMENT_TYPE_F32;
  desc.layout = layout;
  desc.rank = static_cast<uint8_t>(logicalType.getRank());
  desc.byte_offset = byteOffset.staticOffset;
  desc.byte_offset_constant = byteOffset.constantOrdinalPlusOne;
  for (int64_t i = 0; i < logicalType.getRank(); ++i) {
    int64_t logicalDim = logicalType.getDimSize(i);
    int64_t storageDim = storageShape[i];
    if (logicalDim <= 0 || storageDim <= 0 ||
        static_cast<uint64_t>(logicalDim) >
            std::numeric_limits<uint32_t>::max() ||
        static_cast<uint64_t>(storageDim) >
            std::numeric_limits<uint32_t>::max()) {
      return failure();
    }
    desc.shape[i] = static_cast<uint32_t>(logicalDim);
    desc.storage_shape[i] = static_cast<uint32_t>(storageDim);
  }
  if (failed(copyStaticOffsets(mixedOffsets, desc.rank, desc.offset)))
    return failure();
  return copyStaticStrides(mixedStrides, desc.rank, desc.stride);
}

static void populateScalarTensorDesc(
    uint16_t binding, InferredByteOffset byteOffset,
    tt_iree_ttex_tensor_desc_t &desc) {
  desc.binding = binding;
  desc.element_type = TT_IREE_TTEX_TENSOR_ELEMENT_TYPE_F32;
  desc.layout = TT_IREE_TTEX_TENSOR_LAYOUT_ROW_MAJOR;
  desc.rank = 1;
  desc.shape[0] = 1;
  desc.storage_shape[0] = 1;
  desc.stride[0] = 1;
  desc.byte_offset = byteOffset.staticOffset;
  desc.byte_offset_constant = byteOffset.constantOrdinalPlusOne;
}

static FailureOr<BuiltinSelection>
inferF32PaddedCopyProgram(func::FuncOp funcOp) {
  IREE::TensorExt::DispatchTensorLoadOp sourceLoad;
  IREE::TensorExt::DispatchTensorStoreOp destinationStore;
  int loadCount = 0;
  int storeCount = 0;
  funcOp.walk([&](IREE::TensorExt::DispatchTensorLoadOp loadOp) {
    sourceLoad = loadOp;
    ++loadCount;
  });
  funcOp.walk([&](IREE::TensorExt::DispatchTensorStoreOp storeOp) {
    destinationStore = storeOp;
    ++storeCount;
  });
  if (loadCount != 1 || storeCount != 1 ||
      destinationStore.getValue() != sourceLoad.getResult()) {
    return failure();
  }

  auto logicalType = dyn_cast<RankedTensorType>(sourceLoad.getType());
  if (!logicalType || !logicalType.hasStaticShape() ||
      logicalType.getRank() != 3 || !logicalType.getElementType().isF32()) {
    return failure();
  }

  FailureOr<uint16_t> sourceBinding =
      inferLoadBinding(sourceLoad.getResult(), sourceLoad);
  auto sourceSubspan =
      sourceLoad.getSource()
          .getDefiningOp<IREE::HAL::InterfaceBindingSubspanOp>();
  auto destinationSubspan =
      destinationStore.getTarget()
          .getDefiningOp<IREE::HAL::InterfaceBindingSubspanOp>();
  if (failed(sourceBinding) || !sourceSubspan || !destinationSubspan ||
      destinationSubspan.getBinding().getZExtValue() >
          std::numeric_limits<uint16_t>::max()) {
    return failure();
  }
  uint16_t destinationBinding =
      static_cast<uint16_t>(destinationSubspan.getBinding().getZExtValue());
  if (*sourceBinding == destinationBinding) {
    return failure();
  }

  FailureOr<InferredByteOffset> sourceByteOffset =
      inferByteOffset(sourceSubspan);
  FailureOr<InferredByteOffset> destinationByteOffset =
      inferByteOffset(destinationSubspan);
  if (failed(sourceByteOffset) || failed(destinationByteOffset)) {
    return failure();
  }

  BuiltinSelection selection;
  selection.program = TT_IREE_TTEX_BUILTIN_PROGRAM_F32_PADDED_COPY;
  selection.dispatchKind = TT_IREE_TTEX_DISPATCH_KIND_PADDED_COPY;
  if (failed(populateTensorDesc(
          logicalType, sourceLoad.getSourceType().getShape(),
          sourceLoad.getMixedOffsets(), sourceLoad.getMixedStrides(),
          *sourceBinding, *sourceByteOffset,
          TT_IREE_TTEX_TENSOR_LAYOUT_CHW, selection.paddedCopy.source)) ||
      failed(populateTensorDesc(
          logicalType, destinationStore.getTargetType().getShape(),
          destinationStore.getMixedOffsets(), destinationStore.getMixedStrides(),
          destinationBinding,
          *destinationByteOffset, TT_IREE_TTEX_TENSOR_LAYOUT_CHW,
          selection.paddedCopy.destination))) {
    return failure();
  }

  bool hasPadding = false;
  for (int64_t i = 0; i < logicalType.getRank(); ++i) {
    hasPadding |= selection.paddedCopy.destination.offset[i] != 0 ||
                  selection.paddedCopy.destination.storage_shape[i] !=
                      selection.paddedCopy.destination.shape[i];
  }
  if (!hasPadding)
    return failure();
  return selection;
}

static FailureOr<BuiltinSelection>
inferF32MaxPool2DProgram(func::FuncOp funcOp) {
  linalg::GenericOp poolOp;
  int genericCount = 0;
  funcOp.walk([&](linalg::GenericOp genericOp) {
    poolOp = genericOp;
    ++genericCount;
  });
  if (genericCount != 1 || poolOp.getNumDpsInputs() != 2 ||
      poolOp.getNumDpsInits() != 1) {
    return failure();
  }

  auto inputType =
      dyn_cast<RankedTensorType>(poolOp.getDpsInputs()[0].getType());
  auto windowType =
      dyn_cast<RankedTensorType>(poolOp.getDpsInputs()[1].getType());
  auto outputType = dyn_cast<RankedTensorType>(poolOp.getResult(0).getType());
  if (!inputType || !windowType || !outputType || !inputType.hasStaticShape() ||
      !windowType.hasStaticShape() || !outputType.hasStaticShape() ||
      inputType.getRank() != 3 || outputType.getRank() != 3 ||
      windowType.getRank() != 2 || !inputType.getElementType().isF32() ||
      !windowType.getElementType().isF32() ||
      !outputType.getElementType().isF32() ||
      windowType.getShape() != ArrayRef<int64_t>({3, 3}) ||
      inputType.getDimSize(0) != outputType.getDimSize(0) ||
      outputType.getDimSize(1) != (inputType.getDimSize(1) - 3) / 2 + 1 ||
      outputType.getDimSize(2) != (inputType.getDimSize(2) - 3) / 2 + 1) {
    return failure();
  }

  MLIRContext *context = funcOp.getContext();
  AffineExpr d0 = getAffineDimExpr(0, context);
  AffineExpr d1 = getAffineDimExpr(1, context);
  AffineExpr d2 = getAffineDimExpr(2, context);
  AffineExpr d3 = getAffineDimExpr(3, context);
  AffineExpr d4 = getAffineDimExpr(4, context);
  SmallVector<AffineMap, 3> expectedMaps = {
      AffineMap::get(5, 0, {d0, d1 * 2 + d3, d2 * 2 + d4}, context),
      AffineMap::get(5, 0, {d3, d4}, context),
      AffineMap::get(5, 0, {d0, d1, d2}, context),
  };
  auto iteratorTypes = poolOp.getIteratorTypesArray();
  if (poolOp.getIndexingMapsArray() != expectedMaps ||
      iteratorTypes.size() != 5 ||
      iteratorTypes[0] != utils::IteratorType::parallel ||
      iteratorTypes[1] != utils::IteratorType::parallel ||
      iteratorTypes[2] != utils::IteratorType::parallel ||
      iteratorTypes[3] != utils::IteratorType::reduction ||
      iteratorTypes[4] != utils::IteratorType::reduction ||
      !poolOp.getDpsInputs()[1].getDefiningOp<tensor::EmptyOp>()) {
    return failure();
  }

  Block &body = poolOp.getRegion().front();
  auto yieldOp = dyn_cast<linalg::YieldOp>(body.getTerminator());
  if (body.getNumArguments() != 3 || !yieldOp ||
      yieldOp.getValues().size() != 1) {
    return failure();
  }
  auto maximumOp = yieldOp.getValues()[0].getDefiningOp<arith::MaximumFOp>();
  if (!maximumOp ||
      !((maximumOp.getLhs() == body.getArgument(0) &&
         maximumOp.getRhs() == body.getArgument(2)) ||
        (maximumOp.getLhs() == body.getArgument(2) &&
         maximumOp.getRhs() == body.getArgument(0))) ||
      !body.getArgument(1).use_empty()) {
    return failure();
  }

  auto fillOp = poolOp.getDpsInits()[0].getDefiningOp<linalg::FillOp>();
  if (!fillOp || fillOp.getInputs().size() != 1) {
    return failure();
  }
  std::optional<double> initialValue = getConstantFloat(fillOp.getInputs()[0]);
  if (!initialValue || !std::isinf(*initialValue) || *initialValue >= 0.0) {
    return failure();
  }

  IREE::TensorExt::DispatchTensorLoadOp inputLoad;
  FailureOr<uint16_t> inputBinding =
      inferLoadBinding(poolOp.getDpsInputs()[0], inputLoad);
  IREE::TensorExt::DispatchTensorStoreOp outputStore;
  int storeCount = 0;
  funcOp.walk([&](IREE::TensorExt::DispatchTensorStoreOp storeOp) {
    outputStore = storeOp;
    ++storeCount;
  });
  auto outputSubspan =
      outputStore ? outputStore.getTarget()
                        .getDefiningOp<IREE::HAL::InterfaceBindingSubspanOp>()
                  : IREE::HAL::InterfaceBindingSubspanOp{};
  if (failed(inputBinding) || storeCount != 1 ||
      outputStore.getValue() != poolOp.getResult(0) || !outputSubspan ||
      outputSubspan.getBinding().getZExtValue() >
          std::numeric_limits<uint16_t>::max() ||
      *inputBinding == outputSubspan.getBinding().getZExtValue()) {
    return failure();
  }

  auto inputSubspan =
      inputLoad.getSource()
          .getDefiningOp<IREE::HAL::InterfaceBindingSubspanOp>();
  FailureOr<InferredByteOffset> inputByteOffset = inferByteOffset(inputSubspan);
  FailureOr<InferredByteOffset> outputByteOffset =
      inferByteOffset(outputSubspan);
  if (failed(inputByteOffset) || failed(outputByteOffset))
    return failure();

  BuiltinSelection selection;
  selection.program = TT_IREE_TTEX_BUILTIN_PROGRAM_F32_MAX_POOL_2D;
  selection.dispatchKind = TT_IREE_TTEX_DISPATCH_KIND_MAX_POOL;
  selection.maxPool.kernel_h = 3;
  selection.maxPool.kernel_w = 3;
  selection.maxPool.stride_h = 2;
  selection.maxPool.stride_w = 2;
  selection.maxPool.dilation_h = 1;
  selection.maxPool.dilation_w = 1;
  if (failed(populateTensorDesc(
          inputType, inputLoad.getSourceType().getShape(),
          inputLoad.getMixedOffsets(), inputLoad.getMixedStrides(),
          *inputBinding, *inputByteOffset,
          TT_IREE_TTEX_TENSOR_LAYOUT_CHW, selection.maxPool.input)) ||
      failed(populateTensorDesc(
          outputType, outputStore.getTargetType().getShape(),
          outputStore.getMixedOffsets(), outputStore.getMixedStrides(),
          static_cast<uint16_t>(outputSubspan.getBinding().getZExtValue()),
          *outputByteOffset, TT_IREE_TTEX_TENSOR_LAYOUT_CHW,
          selection.maxPool.output))) {
    return failure();
  }
  return selection;
}

struct ScalarAffine {
  float scale = 1.0f;
  float bias = 0.0f;
};

struct ReluEpilogue {
  linalg::GenericOp op;
  Value input;
};

static bool matchesIdentityElementwise(linalg::GenericOp genericOp) {
  for (AffineMap map : genericOp.getIndexingMapsArray()) {
    if (!map.isIdentity())
      return false;
  }
  return llvm::all_of(genericOp.getIteratorTypesArray(), [](auto iteratorType) {
    return iteratorType == utils::IteratorType::parallel;
  });
}

static bool matchesBF16ElementwiseAdd(linalg::GenericOp genericOp) {
  if (genericOp.getNumDpsInputs() != 2 || genericOp.getNumDpsInits() != 1 ||
      !matchesIdentityElementwise(genericOp)) {
    return false;
  }

  auto lhsType =
      dyn_cast<RankedTensorType>(genericOp.getDpsInputs()[0].getType());
  auto rhsType =
      dyn_cast<RankedTensorType>(genericOp.getDpsInputs()[1].getType());
  auto outType =
      dyn_cast<RankedTensorType>(genericOp.getDpsInits()[0].getType());
  if (!lhsType || lhsType != rhsType || lhsType != outType ||
      !lhsType.hasStaticShape() || !lhsType.getElementType().isBF16() ||
      lhsType.getNumElements() <= 0 ||
      lhsType.getNumElements() % (32 * 32) != 0) {
    return false;
  }

  Block &body = genericOp.getRegion().front();
  if (body.getNumArguments() != 3)
    return false;
  auto yieldOp = dyn_cast<linalg::YieldOp>(body.getTerminator());
  if (!yieldOp || yieldOp.getValues().size() != 1)
    return false;
  auto addOp = yieldOp.getValues()[0].getDefiningOp<arith::AddFOp>();
  if (!addOp)
    return false;
  return (addOp.getLhs() == body.getArgument(0) &&
          addOp.getRhs() == body.getArgument(1)) ||
         (addOp.getLhs() == body.getArgument(1) &&
          addOp.getRhs() == body.getArgument(0));
}

static FailureOr<ReluEpilogue> inferSingleRelu(func::FuncOp funcOp) {
  arith::SelectOp matchedSelect;
  int selectCount = 0;
  funcOp.walk([&](arith::SelectOp selectOp) {
    matchedSelect = selectOp;
    ++selectCount;
  });
  if (selectCount != 1)
    return failure();

  auto epilogueOp = matchedSelect->getParentOfType<linalg::GenericOp>();
  auto compareOp = matchedSelect.getCondition().getDefiningOp<arith::CmpFOp>();
  auto falseValue = getConstantFloat(matchedSelect.getFalseValue());
  if (!epilogueOp || !compareOp || !falseValue || *falseValue != 0.0 ||
      compareOp.getPredicate() != arith::CmpFPredicate::UGT ||
      compareOp.getLhs() != matchedSelect.getTrueValue()) {
    return failure();
  }
  auto compareRhs = getConstantFloat(compareOp.getRhs());
  if (!compareRhs || *compareRhs != 0.0)
    return failure();
  return ReluEpilogue{epilogueOp, matchedSelect.getTrueValue()};
}

static FailureOr<ScalarAffine> inferScalarAffine(Value value, Value input) {
  auto shiftedOp = value.getDefiningOp<arith::AddFOp>();
  if (!shiftedOp)
    return failure();

  Value scaledValue;
  std::optional<double> bias;
  if ((bias = getConstantFloat(shiftedOp.getLhs()))) {
    scaledValue = shiftedOp.getRhs();
  } else if ((bias = getConstantFloat(shiftedOp.getRhs()))) {
    scaledValue = shiftedOp.getLhs();
  } else {
    return failure();
  }

  auto scaledOp = scaledValue.getDefiningOp<arith::MulFOp>();
  if (!scaledOp)
    return failure();
  Value centeredValue;
  std::optional<double> scale;
  if ((scale = getConstantFloat(scaledOp.getLhs()))) {
    centeredValue = scaledOp.getRhs();
  } else if ((scale = getConstantFloat(scaledOp.getRhs()))) {
    centeredValue = scaledOp.getLhs();
  } else {
    return failure();
  }

  auto centeredOp = centeredValue.getDefiningOp<arith::SubFOp>();
  if (!centeredOp || centeredOp.getLhs() != input) {
    return failure();
  }
  std::optional<double> mean = getConstantFloat(centeredOp.getRhs());
  if (!mean)
    return failure();

  double collapsedBias = *bias - *mean * *scale;
  if (!std::isfinite(*scale) || !std::isfinite(collapsedBias))
    return failure();
  return ScalarAffine{static_cast<float>(*scale),
                      static_cast<float>(collapsedBias)};
}

struct ScalarAffineRelu {
  linalg::GenericOp op;
  float scale = 1.0f;
  float bias = 0.0f;
};

static FailureOr<ScalarAffineRelu> inferScalarAffineRelu(func::FuncOp funcOp) {
  FailureOr<ReluEpilogue> relu = inferSingleRelu(funcOp);
  if (failed(relu) || relu->op.getNumDpsInputs() != 1 ||
      relu->op.getNumDpsInits() != 1 || !matchesIdentityElementwise(relu->op)) {
    return failure();
  }
  FailureOr<ScalarAffine> affine = inferScalarAffine(
      relu->input, relu->op.getRegion().front().getArgument(0));
  if (failed(affine))
    return failure();
  return ScalarAffineRelu{relu->op, affine->scale, affine->bias};
}

static FailureOr<std::vector<float>> getConstantF32Vector(Value value,
                                                          size_t expectedSize) {
  auto constantOp = value.getDefiningOp<arith::ConstantOp>();
  if (!constantOp)
    return failure();

  Attribute attr = constantOp.getValue();
  std::vector<float> result;
  result.reserve(expectedSize);
  if (auto dense = dyn_cast<DenseElementsAttr>(attr)) {
    auto type = dyn_cast<RankedTensorType>(dense.getType());
    if (!type || type.getRank() != 1 || !type.getElementType().isF32() ||
        static_cast<size_t>(dense.getNumElements()) != expectedSize) {
      return failure();
    }
    for (APFloat element : dense.getValues<APFloat>())
      result.push_back(element.convertToFloat());
  } else if (auto resource = dyn_cast<DenseF32ResourceElementsAttr>(attr)) {
    auto type = dyn_cast<RankedTensorType>(resource.getType());
    std::optional<ArrayRef<float>> elements = resource.tryGetAsArrayRef();
    if (!type || type.getRank() != 1 || !elements ||
        elements->size() != expectedSize) {
      return failure();
    }
    result.assign(elements->begin(), elements->end());
  } else {
    return failure();
  }

  for (float element : result) {
    if (!std::isfinite(element))
      return failure();
  }
  return result;
}

struct PerChannelAffineRelu {
  linalg::GenericOp op;
  Value mean;
  Value channelScale;
  Value channelBias;
  float commonScale = 1.0f;
  std::vector<float> scales;
  std::vector<float> biases;
};

static FailureOr<PerChannelAffineRelu>
inferPerChannelAffineRelu(func::FuncOp funcOp, size_t channelCount) {
  FailureOr<ReluEpilogue> relu = inferSingleRelu(funcOp);
  if (failed(relu) || relu->op.getNumDpsInputs() != 4 ||
      relu->op.getNumDpsInits() != 1) {
    return failure();
  }

  MLIRContext *context = relu->op.getContext();
  AffineExpr d0 = getAffineDimExpr(0, context);
  AffineExpr d1 = getAffineDimExpr(1, context);
  AffineExpr d2 = getAffineDimExpr(2, context);
  unsigned rank = cast<RankedTensorType>(relu->op.getResult(0).getType()).getRank();
  SmallVector<AffineExpr, 3> identityResults = {d0, d1, d2};
  identityResults.resize(rank);
  SmallVector<AffineMap, 5> expectedMaps;
  expectedMaps.push_back(AffineMap::get(rank, 0, identityResults, context));
  for (unsigned i = 0; i < 3; ++i)
    expectedMaps.push_back(AffineMap::get(rank, 0, {d0}, context));
  expectedMaps.push_back(AffineMap::get(rank, 0, identityResults, context));
  auto iteratorTypes = relu->op.getIteratorTypesArray();
  if (relu->op.getIndexingMapsArray() != expectedMaps ||
      (rank != 2 && rank != 3) || iteratorTypes.size() != rank ||
      !llvm::all_of(iteratorTypes, [](utils::IteratorType iteratorType) {
        return iteratorType == utils::IteratorType::parallel;
      })) {
    return failure();
  }

  Block &body = relu->op.getRegion().front();
  Value activation = body.getArgument(0);
  Value mean = body.getArgument(1);
  Value channelScale = body.getArgument(2);
  Value channelBias = body.getArgument(3);

  auto shiftedOp = relu->input.getDefiningOp<arith::AddFOp>();
  if (!shiftedOp)
    return failure();
  Value scaledValue;
  if (shiftedOp.getLhs() == channelBias) {
    scaledValue = shiftedOp.getRhs();
  } else if (shiftedOp.getRhs() == channelBias) {
    scaledValue = shiftedOp.getLhs();
  } else {
    return failure();
  }

  auto channelScaleOp = scaledValue.getDefiningOp<arith::MulFOp>();
  if (!channelScaleOp)
    return failure();
  Value centeredValue;
  if (channelScaleOp.getLhs() == channelScale) {
    centeredValue = channelScaleOp.getRhs();
  } else if (channelScaleOp.getRhs() == channelScale) {
    centeredValue = channelScaleOp.getLhs();
  } else {
    return failure();
  }

  double commonScale = 1.0;
  if (auto commonScaleOp = centeredValue.getDefiningOp<arith::MulFOp>()) {
    std::optional<double> constant = getConstantFloat(commonScaleOp.getLhs());
    if (constant) {
      centeredValue = commonScaleOp.getRhs();
    } else {
      constant = getConstantFloat(commonScaleOp.getRhs());
      if (!constant)
        return failure();
      centeredValue = commonScaleOp.getLhs();
    }
    commonScale = *constant;
  }

  auto centeredOp = centeredValue.getDefiningOp<arith::SubFOp>();
  if (!centeredOp || centeredOp.getLhs() != activation ||
      centeredOp.getRhs() != mean || !std::isfinite(commonScale)) {
    return failure();
  }

  FailureOr<std::vector<float>> means =
      getConstantF32Vector(relu->op.getDpsInputs()[1], channelCount);
  FailureOr<std::vector<float>> scales =
      getConstantF32Vector(relu->op.getDpsInputs()[2], channelCount);
  FailureOr<std::vector<float>> biases =
      getConstantF32Vector(relu->op.getDpsInputs()[3], channelCount);
  std::vector<float> collapsedScales;
  std::vector<float> collapsedBiases;
  if (succeeded(means) && succeeded(scales) && succeeded(biases)) {
    collapsedScales.resize(channelCount);
    collapsedBiases.resize(channelCount);
    for (size_t i = 0; i < channelCount; ++i) {
      double scale = commonScale * (*scales)[i];
      double bias = (*biases)[i] - (*means)[i] * scale;
      if (!std::isfinite(scale) || !std::isfinite(bias))
        return failure();
      collapsedScales[i] = static_cast<float>(scale);
      collapsedBiases[i] = static_cast<float>(bias);
    }
  }
  return PerChannelAffineRelu{
      relu->op,
      relu->op.getDpsInputs()[1],
      relu->op.getDpsInputs()[2],
      relu->op.getDpsInputs()[3],
      static_cast<float>(commonScale),
      std::move(collapsedScales),
      std::move(collapsedBiases),
  };
}

static LogicalResult populateRuntimeVectorDesc(
    Value value, size_t expectedSize, tt_iree_ttex_tensor_desc_t &desc) {
  IREE::TensorExt::DispatchTensorLoadOp loadOp;
  FailureOr<uint16_t> binding = inferLoadBinding(value, loadOp);
  auto type = dyn_cast<RankedTensorType>(value.getType());
  auto subspan = loadOp ? loadOp.getSource()
                              .getDefiningOp<
                                  IREE::HAL::InterfaceBindingSubspanOp>()
                        : IREE::HAL::InterfaceBindingSubspanOp{};
  FailureOr<InferredByteOffset> byteOffset = inferByteOffset(subspan);
  if (failed(binding) || !type || !type.hasStaticShape() ||
      type.getRank() != 1 || !type.getElementType().isF32() ||
      static_cast<size_t>(type.getDimSize(0)) != expectedSize ||
      failed(byteOffset)) {
    return failure();
  }
  return populateTensorDesc(
      type, loadOp.getSourceType().getShape(), loadOp.getMixedOffsets(),
      loadOp.getMixedStrides(), *binding, *byteOffset,
      TT_IREE_TTEX_TENSOR_LAYOUT_ROW_MAJOR, desc);
}

struct ResidualAffineRelu {
  linalg::GenericOp op;
  Value residual;
  float mainScale = 1.0f;
  float mainBias = 0.0f;
  float residualScale = 1.0f;
  float residualBias = 0.0f;
};

static FailureOr<ResidualAffineRelu>
inferResidualAffineRelu(func::FuncOp funcOp, linalg::GenericOp contractionOp) {
  FailureOr<ReluEpilogue> relu = inferSingleRelu(funcOp);
  if (failed(relu) || relu->op.getNumDpsInputs() != 2 ||
      relu->op.getNumDpsInits() != 1 || !matchesIdentityElementwise(relu->op)) {
    return failure();
  }

  int contractionInput = -1;
  for (auto [index, input] : llvm::enumerate(relu->op.getDpsInputs())) {
    if (input == contractionOp.getResult(0)) {
      if (contractionInput != -1)
        return failure();
      contractionInput = static_cast<int>(index);
    }
  }
  if (contractionInput == -1)
    return failure();
  int residualInput = 1 - contractionInput;

  auto sumOp = relu->input.getDefiningOp<arith::AddFOp>();
  if (!sumOp)
    return failure();
  Value mainArgument =
      relu->op.getRegion().front().getArgument(contractionInput);
  Value residualArgument =
      relu->op.getRegion().front().getArgument(residualInput);

  std::optional<std::pair<ScalarAffine, ScalarAffine>> affinePair;
  auto tryMatchPair = [&](Value mainValue, Value residualValue) {
    FailureOr<ScalarAffine> mainAffine =
        inferScalarAffine(mainValue, mainArgument);
    FailureOr<ScalarAffine> residualAffine =
        inferScalarAffine(residualValue, residualArgument);
    if (succeeded(mainAffine) && succeeded(residualAffine)) {
      affinePair = std::make_pair(*mainAffine, *residualAffine);
    }
  };
  tryMatchPair(sumOp.getLhs(), sumOp.getRhs());
  if (!affinePair)
    tryMatchPair(sumOp.getRhs(), sumOp.getLhs());
  if (!affinePair)
    return failure();

  return ResidualAffineRelu{
      relu->op,
      relu->op.getDpsInputs()[residualInput],
      affinePair->first.scale,
      affinePair->first.bias,
      affinePair->second.scale,
      affinePair->second.bias,
  };
}

struct RuntimePerChannelResidualAffineRelu {
  linalg::GenericOp op;
  Value residual;
  Value mainMean;
  Value mainScale;
  Value mainBias;
  float mainCommonScale = 1.0f;
  Value residualMean;
  Value residualScale;
  Value residualBias;
  float residualCommonScale = 1.0f;
  bool hasResidualAffine = false;
};

static FailureOr<float> matchRuntimeChannelAffine(
    Value value, Value activation, Value mean, Value channelScale,
    Value channelBias) {
  auto shiftedOp = value.getDefiningOp<arith::AddFOp>();
  if (!shiftedOp)
    return failure();
  Value scaledValue;
  if (shiftedOp.getLhs() == channelBias) {
    scaledValue = shiftedOp.getRhs();
  } else if (shiftedOp.getRhs() == channelBias) {
    scaledValue = shiftedOp.getLhs();
  } else {
    return failure();
  }

  auto scaleOp = scaledValue.getDefiningOp<arith::MulFOp>();
  if (!scaleOp)
    return failure();
  Value centeredValue;
  if (scaleOp.getLhs() == channelScale) {
    centeredValue = scaleOp.getRhs();
  } else if (scaleOp.getRhs() == channelScale) {
    centeredValue = scaleOp.getLhs();
  } else {
    return failure();
  }

  double commonScale = 1.0;
  if (auto commonScaleOp = centeredValue.getDefiningOp<arith::MulFOp>()) {
    std::optional<double> constant = getConstantFloat(commonScaleOp.getLhs());
    if (constant) {
      centeredValue = commonScaleOp.getRhs();
    } else {
      constant = getConstantFloat(commonScaleOp.getRhs());
      if (!constant)
        return failure();
      centeredValue = commonScaleOp.getLhs();
    }
    commonScale = *constant;
  }
  auto centeredOp = centeredValue.getDefiningOp<arith::SubFOp>();
  if (!centeredOp || centeredOp.getLhs() != activation ||
      centeredOp.getRhs() != mean || !std::isfinite(commonScale)) {
    return failure();
  }
  return static_cast<float>(commonScale);
}

static FailureOr<RuntimePerChannelResidualAffineRelu>
inferRuntimePerChannelResidualAffineRelu(func::FuncOp funcOp,
                                         linalg::GenericOp contractionOp) {
  FailureOr<ReluEpilogue> relu = inferSingleRelu(funcOp);
  if (failed(relu) || relu->op.getNumDpsInits() != 1 ||
      (relu->op.getNumDpsInputs() != 5 &&
       relu->op.getNumDpsInputs() != 8) ||
      relu->op.getDpsInputs()[0] != contractionOp.getResult(0)) {
    return failure();
  }

  unsigned rank = cast<RankedTensorType>(relu->op.getResult(0).getType()).getRank();
  if (rank != 2 && rank != 3)
    return failure();
  MLIRContext *context = relu->op.getContext();
  SmallVector<AffineExpr, 3> identityResults = {
      getAffineDimExpr(0, context), getAffineDimExpr(1, context),
      getAffineDimExpr(2, context)};
  identityResults.resize(rank);
  AffineMap identityMap =
      AffineMap::get(rank, 0, identityResults, context);
  AffineMap channelMap = AffineMap::get(
      rank, 0, {getAffineDimExpr(0, context)}, context);
  SmallVector<AffineMap> expectedMaps = {identityMap, channelMap, channelMap,
                                         channelMap, identityMap};
  if (relu->op.getNumDpsInputs() == 8) {
    expectedMaps.append({channelMap, channelMap, channelMap});
  }
  expectedMaps.push_back(identityMap);
  auto iteratorTypes = relu->op.getIteratorTypesArray();
  if (relu->op.getIndexingMapsArray() != expectedMaps ||
      iteratorTypes.size() != rank ||
      !llvm::all_of(iteratorTypes, [](utils::IteratorType iteratorType) {
        return iteratorType == utils::IteratorType::parallel;
      })) {
    return failure();
  }

  Block &body = relu->op.getRegion().front();
  auto sumOp = relu->input.getDefiningOp<arith::AddFOp>();
  if (!sumOp)
    return failure();
  auto tryMatch = [&](Value mainValue, Value residualValue)
      -> FailureOr<RuntimePerChannelResidualAffineRelu> {
    FailureOr<float> mainCommonScale = matchRuntimeChannelAffine(
        mainValue, body.getArgument(0), body.getArgument(1),
        body.getArgument(2), body.getArgument(3));
    if (failed(mainCommonScale))
      return failure();

    RuntimePerChannelResidualAffineRelu result;
    result.op = relu->op;
    result.residual = relu->op.getDpsInputs()[4];
    result.mainMean = relu->op.getDpsInputs()[1];
    result.mainScale = relu->op.getDpsInputs()[2];
    result.mainBias = relu->op.getDpsInputs()[3];
    result.mainCommonScale = *mainCommonScale;
    if (relu->op.getNumDpsInputs() == 5) {
      if (residualValue != body.getArgument(4))
        return failure();
      return result;
    }

    FailureOr<float> residualCommonScale = matchRuntimeChannelAffine(
        residualValue, body.getArgument(4), body.getArgument(5),
        body.getArgument(6), body.getArgument(7));
    if (failed(residualCommonScale))
      return failure();
    result.residualMean = relu->op.getDpsInputs()[5];
    result.residualScale = relu->op.getDpsInputs()[6];
    result.residualBias = relu->op.getDpsInputs()[7];
    result.residualCommonScale = *residualCommonScale;
    result.hasResidualAffine = true;
    return result;
  };

  FailureOr<RuntimePerChannelResidualAffineRelu> result =
      tryMatch(sumOp.getLhs(), sumOp.getRhs());
  if (failed(result))
    result = tryMatch(sumOp.getRhs(), sumOp.getLhs());
  return result;
}

static FailureOr<BuiltinSelection>
inferF32MatmulLikeProgram(func::FuncOp funcOp) {
  linalg::GenericOp contractionOp;
  int contractionCount = 0;
  bool isFlattened = false;
  funcOp.walk([&](linalg::GenericOp genericOp) {
    bool matchesSpatial = matchesProjectionIndexingMaps(genericOp);
    bool matchesFlattened = matchesFlattenedProjectionIndexingMaps(genericOp);
    if (genericOp.getNumDpsInputs() != 2 || genericOp.getNumDpsInits() != 1 ||
        (!matchesSpatial && !matchesFlattened) ||
        !matchesMultiplyAddBody(genericOp)) {
      return;
    }
    contractionOp = genericOp;
    isFlattened = matchesFlattened;
    ++contractionCount;
  });
  if (contractionCount != 1)
    return failure();

  auto lhsType =
      dyn_cast<RankedTensorType>(contractionOp.getDpsInputs()[0].getType());
  auto rhsType =
      dyn_cast<RankedTensorType>(contractionOp.getDpsInputs()[1].getType());
  auto outputType =
      dyn_cast<RankedTensorType>(contractionOp.getDpsInits()[0].getType());
  if (!lhsType || !rhsType || !outputType || !lhsType.hasStaticShape() ||
      !rhsType.hasStaticShape() || !outputType.hasStaticShape() ||
      rhsType.getRank() != 2 || !lhsType.getElementType().isF32() ||
      !rhsType.getElementType().isF32() ||
      !outputType.getElementType().isF32()) {
    return failure();
  }

  int64_t m = 0;
  int64_t k = 0;
  int64_t n = rhsType.getDimSize(0);
  uint8_t activationLayout = TT_IREE_TTEX_TENSOR_LAYOUT_CHW;
  if (isFlattened) {
    if (lhsType.getRank() != 2 || outputType.getRank() != 2)
      return failure();
    k = lhsType.getDimSize(0);
    m = lhsType.getDimSize(1);
    activationLayout = TT_IREE_TTEX_TENSOR_LAYOUT_CHANNEL_MAJOR;
    if (rhsType.getDimSize(1) != k || outputType.getDimSize(0) != n ||
        outputType.getDimSize(1) != m) {
      return failure();
    }
  } else {
    if (lhsType.getRank() != 3 || outputType.getRank() != 3)
      return failure();
    k = lhsType.getDimSize(0);
    int64_t h = lhsType.getDimSize(1);
    int64_t w = lhsType.getDimSize(2);
    if (h <= 0 || w <= 0 || h > std::numeric_limits<uint32_t>::max() / w ||
        rhsType.getDimSize(1) != k || outputType.getDimSize(0) != n ||
        outputType.getDimSize(1) != h || outputType.getDimSize(2) != w) {
      return failure();
    }
    m = h * w;
  }
  if (m <= 0 || n <= 0 || k <= 0 ||
      static_cast<uint64_t>(m) > std::numeric_limits<uint32_t>::max() ||
      static_cast<uint64_t>(n) > std::numeric_limits<uint32_t>::max() ||
      static_cast<uint64_t>(k) > std::numeric_limits<uint32_t>::max()) {
    return failure();
  }

  IREE::TensorExt::DispatchTensorStoreOp outputStore;
  int storeCount = 0;
  funcOp.walk([&](IREE::TensorExt::DispatchTensorStoreOp storeOp) {
    outputStore = storeOp;
    ++storeCount;
  });
  if (storeCount != 1)
    return failure();

  BuiltinSelection selection;
  selection.program = TT_IREE_TTEX_BUILTIN_PROGRAM_F32_MATMUL_LIKE;
  selection.dispatchKind = TT_IREE_TTEX_DISPATCH_KIND_MATMUL_LIKE;
  selection.matmulLike.m = static_cast<uint32_t>(m);
  selection.matmulLike.n = static_cast<uint32_t>(n);
  selection.matmulLike.k = static_cast<uint32_t>(k);
  selection.mTiles = static_cast<uint32_t>((m + 31) / 32);
  selection.nTiles = static_cast<uint32_t>((n + 31) / 32);
  selection.kTiles = static_cast<uint32_t>((k + 31) / 32);
  selection.matmulLike.epilogue_scale = 1.0f;
  selection.matmulLike.residual_scale = 1.0f;
  selection.matmulLike.epilogue_common_scale = 1.0f;
  selection.matmulLike.residual_common_scale = 1.0f;

  IREE::TensorExt::DispatchTensorLoadOp residualLoad;
  std::optional<uint16_t> residualBinding;
  if (outputStore.getValue() != contractionOp.getResult(0)) {
    FailureOr<ScalarAffineRelu> scalarEpilogue = inferScalarAffineRelu(funcOp);
    FailureOr<PerChannelAffineRelu> perChannelEpilogue =
        inferPerChannelAffineRelu(funcOp, static_cast<size_t>(n));
    if (succeeded(scalarEpilogue) &&
        scalarEpilogue->op.getDpsInputs()[0] == contractionOp.getResult(0) &&
        outputStore.getValue().getDefiningOp() ==
            scalarEpilogue->op.getOperation()) {
      selection.matmulLike.epilogue_flags =
          TT_IREE_TTEX_EPILOGUE_AFFINE | TT_IREE_TTEX_EPILOGUE_RELU;
      selection.matmulLike.epilogue_scale = scalarEpilogue->scale;
      selection.matmulLike.epilogue_bias = scalarEpilogue->bias;
    } else if (succeeded(perChannelEpilogue) &&
               perChannelEpilogue->op.getDpsInputs()[0] ==
                   contractionOp.getResult(0) &&
               outputStore.getValue().getDefiningOp() ==
                   perChannelEpilogue->op.getOperation()) {
      selection.matmulLike.epilogue_flags = TT_IREE_TTEX_EPILOGUE_AFFINE |
                                            TT_IREE_TTEX_EPILOGUE_RELU |
                                            TT_IREE_TTEX_EPILOGUE_PER_CHANNEL;
      if (!perChannelEpilogue->scales.empty()) {
        selection.matmulEpilogueScales =
            std::move(perChannelEpilogue->scales);
        selection.matmulEpilogueBiases =
            std::move(perChannelEpilogue->biases);
      } else {
        selection.matmulLike.epilogue_flags |=
            TT_IREE_TTEX_EPILOGUE_RUNTIME_AFFINE;
        selection.matmulLike.epilogue_common_scale =
            perChannelEpilogue->commonScale;
        if (failed(populateRuntimeVectorDesc(
                perChannelEpilogue->mean, static_cast<size_t>(n),
                selection.matmulLike.epilogue_mean)) ||
            failed(populateRuntimeVectorDesc(
                perChannelEpilogue->channelScale, static_cast<size_t>(n),
                selection.matmulLike.epilogue_channel_scale)) ||
            failed(populateRuntimeVectorDesc(
                perChannelEpilogue->channelBias, static_cast<size_t>(n),
                selection.matmulLike.epilogue_channel_bias))) {
          return failure();
        }
      }
    } else {
      FailureOr<RuntimePerChannelResidualAffineRelu> runtimeResidualEpilogue =
          inferRuntimePerChannelResidualAffineRelu(funcOp, contractionOp);
      if (succeeded(runtimeResidualEpilogue) &&
          outputStore.getValue().getDefiningOp() ==
              runtimeResidualEpilogue->op.getOperation()) {
        FailureOr<uint16_t> inferredResidualBinding =
            inferLoadBinding(runtimeResidualEpilogue->residual, residualLoad);
        if (failed(inferredResidualBinding))
          return failure();
        residualBinding = *inferredResidualBinding;
        selection.matmulLike.epilogue_flags =
            TT_IREE_TTEX_EPILOGUE_AFFINE |
            TT_IREE_TTEX_EPILOGUE_RESIDUAL |
            TT_IREE_TTEX_EPILOGUE_RELU |
            TT_IREE_TTEX_EPILOGUE_PER_CHANNEL |
            TT_IREE_TTEX_EPILOGUE_RUNTIME_AFFINE;
        selection.matmulLike.epilogue_common_scale =
            runtimeResidualEpilogue->mainCommonScale;
        if (failed(populateRuntimeVectorDesc(
                runtimeResidualEpilogue->mainMean, static_cast<size_t>(n),
                selection.matmulLike.epilogue_mean)) ||
            failed(populateRuntimeVectorDesc(
                runtimeResidualEpilogue->mainScale, static_cast<size_t>(n),
                selection.matmulLike.epilogue_channel_scale)) ||
            failed(populateRuntimeVectorDesc(
                runtimeResidualEpilogue->mainBias, static_cast<size_t>(n),
                selection.matmulLike.epilogue_channel_bias))) {
          return failure();
        }
        if (runtimeResidualEpilogue->hasResidualAffine) {
          selection.matmulLike.epilogue_flags |=
              TT_IREE_TTEX_EPILOGUE_RESIDUAL_PER_CHANNEL;
          selection.matmulLike.residual_common_scale =
              runtimeResidualEpilogue->residualCommonScale;
          if (failed(populateRuntimeVectorDesc(
                  runtimeResidualEpilogue->residualMean,
                  static_cast<size_t>(n),
                  selection.matmulLike.residual_mean)) ||
              failed(populateRuntimeVectorDesc(
                  runtimeResidualEpilogue->residualScale,
                  static_cast<size_t>(n),
                  selection.matmulLike.residual_channel_scale)) ||
              failed(populateRuntimeVectorDesc(
                  runtimeResidualEpilogue->residualBias,
                  static_cast<size_t>(n),
                  selection.matmulLike.residual_channel_bias))) {
            return failure();
          }
        }
      } else {
        FailureOr<ResidualAffineRelu> residualEpilogue =
            inferResidualAffineRelu(funcOp, contractionOp);
        if (failed(residualEpilogue) ||
            outputStore.getValue().getDefiningOp() !=
                residualEpilogue->op.getOperation()) {
          return failure();
        }
        FailureOr<uint16_t> inferredResidualBinding =
            inferLoadBinding(residualEpilogue->residual, residualLoad);
        if (failed(inferredResidualBinding))
          return failure();
        residualBinding = *inferredResidualBinding;
        selection.matmulLike.epilogue_flags =
            TT_IREE_TTEX_EPILOGUE_AFFINE |
            TT_IREE_TTEX_EPILOGUE_RESIDUAL |
            TT_IREE_TTEX_EPILOGUE_RELU;
        selection.matmulLike.epilogue_scale = residualEpilogue->mainScale;
        selection.matmulLike.epilogue_bias = residualEpilogue->mainBias;
        selection.matmulLike.residual_scale = residualEpilogue->residualScale;
        selection.matmulLike.residual_bias = residualEpilogue->residualBias;
      }
    }
  }

  IREE::TensorExt::DispatchTensorLoadOp lhsLoad;
  IREE::TensorExt::DispatchTensorLoadOp rhsLoad;
  FailureOr<uint16_t> lhsBinding =
      inferLoadBinding(contractionOp.getDpsInputs()[0], lhsLoad);
  FailureOr<uint16_t> rhsBinding =
      inferLoadBinding(contractionOp.getDpsInputs()[1], rhsLoad);
  auto outputSubspan =
      outputStore.getTarget()
          .getDefiningOp<IREE::HAL::InterfaceBindingSubspanOp>();
  if (failed(lhsBinding) || failed(rhsBinding) || !outputSubspan ||
      outputSubspan.getBinding().getZExtValue() >
          std::numeric_limits<uint16_t>::max()) {
    return failure();
  }

  auto lhsSubspan =
      lhsLoad.getSource().getDefiningOp<IREE::HAL::InterfaceBindingSubspanOp>();
  auto rhsSubspan =
      rhsLoad.getSource().getDefiningOp<IREE::HAL::InterfaceBindingSubspanOp>();
  FailureOr<InferredByteOffset> lhsByteOffset = inferByteOffset(lhsSubspan);
  FailureOr<InferredByteOffset> rhsByteOffset = inferByteOffset(rhsSubspan);
  FailureOr<InferredByteOffset> outputByteOffset =
      inferByteOffset(outputSubspan);
  if (failed(lhsByteOffset) || failed(rhsByteOffset) ||
      failed(outputByteOffset)) {
    return failure();
  }

  if (failed(populateTensorDesc(lhsType, lhsLoad.getSourceType().getShape(),
                                lhsLoad.getMixedOffsets(),
                                lhsLoad.getMixedStrides(), *lhsBinding,
                                *lhsByteOffset, activationLayout,
                                selection.matmulLike.lhs)) ||
      failed(populateTensorDesc(rhsType, rhsLoad.getSourceType().getShape(),
                                rhsLoad.getMixedOffsets(),
                                rhsLoad.getMixedStrides(), *rhsBinding,
                                *rhsByteOffset, TT_IREE_TTEX_TENSOR_LAYOUT_OI,
                                selection.matmulLike.rhs)) ||
      failed(populateTensorDesc(
          outputType, outputStore.getTargetType().getShape(),
          outputStore.getMixedOffsets(), outputStore.getMixedStrides(),
          static_cast<uint16_t>(outputSubspan.getBinding().getZExtValue()),
          *outputByteOffset, activationLayout, selection.matmulLike.output))) {
    return failure();
  }
  if (residualBinding) {
    auto residualType =
        dyn_cast<RankedTensorType>(residualLoad.getResult().getType());
    auto residualSubspan =
        residualLoad.getSource()
            .getDefiningOp<IREE::HAL::InterfaceBindingSubspanOp>();
    FailureOr<InferredByteOffset> residualByteOffset =
        inferByteOffset(residualSubspan);
    if (!residualType || residualType != outputType ||
        failed(residualByteOffset) ||
        failed(populateTensorDesc(
            residualType, residualLoad.getSourceType().getShape(),
            residualLoad.getMixedOffsets(), residualLoad.getMixedStrides(),
            *residualBinding,
            *residualByteOffset, activationLayout,
            selection.matmulLike.residual))) {
      return failure();
    }
  }
  return selection;
}

static FailureOr<BuiltinSelection> inferF32Conv2DProgram(func::FuncOp funcOp) {
  linalg::GenericOp convolutionOp;
  uint32_t convolutionStride = 0;
  int convolutionCount = 0;
  funcOp.walk([&](linalg::GenericOp genericOp) {
    std::optional<uint32_t> stride = inferConv2DStride(genericOp);
    if (genericOp.getNumDpsInputs() != 2 || genericOp.getNumDpsInits() != 1 ||
        !stride || !matchesMultiplyAddBody(genericOp)) {
      return;
    }
    convolutionOp = genericOp;
    convolutionStride = *stride;
    ++convolutionCount;
  });
  if (convolutionCount != 1)
    return failure();

  auto inputType =
      dyn_cast<RankedTensorType>(convolutionOp.getDpsInputs()[0].getType());
  auto filterType =
      dyn_cast<RankedTensorType>(convolutionOp.getDpsInputs()[1].getType());
  auto outputType =
      dyn_cast<RankedTensorType>(convolutionOp.getDpsInits()[0].getType());
  if (!inputType || !filterType || !outputType || !inputType.hasStaticShape() ||
      !filterType.hasStaticShape() || !outputType.hasStaticShape() ||
      inputType.getRank() != 3 || filterType.getRank() != 4 ||
      outputType.getRank() != 3 || !inputType.getElementType().isF32() ||
      !filterType.getElementType().isF32() ||
      !outputType.getElementType().isF32()) {
    return failure();
  }

  int64_t inputC = inputType.getDimSize(0);
  int64_t inputH = inputType.getDimSize(1);
  int64_t inputW = inputType.getDimSize(2);
  int64_t outputC = filterType.getDimSize(0);
  int64_t filterC = filterType.getDimSize(1);
  int64_t kernelH = filterType.getDimSize(2);
  int64_t kernelW = filterType.getDimSize(3);
  int64_t outputH = outputType.getDimSize(1);
  int64_t outputW = outputType.getDimSize(2);
  bool isSupportedKernel =
      (kernelH == 3 && kernelW == 3 &&
       (convolutionStride == 1 || convolutionStride == 2)) ||
      (kernelH == 7 && kernelW == 7 && convolutionStride == 2);
  if (inputC <= 0 || inputH <= 0 || inputW <= 0 || outputC <= 0 ||
      !isSupportedKernel || filterC != inputC ||
      outputType.getDimSize(0) != outputC || inputH < kernelH ||
      inputW < kernelW ||
      outputH != (inputH - kernelH) / convolutionStride + 1 ||
      outputW != (inputW - kernelW) / convolutionStride + 1 ||
      static_cast<uint64_t>(outputH) * outputW >
          std::numeric_limits<uint32_t>::max() ||
      outputC > std::numeric_limits<uint32_t>::max() ||
      static_cast<uint64_t>(inputC) * kernelH * kernelW >
          std::numeric_limits<uint32_t>::max()) {
    return failure();
  }

  FailureOr<ScalarAffineRelu> scalarEpilogue = inferScalarAffineRelu(funcOp);
  FailureOr<PerChannelAffineRelu> perChannelEpilogue =
      inferPerChannelAffineRelu(funcOp, static_cast<size_t>(outputC));
  linalg::GenericOp epilogueOp;
  bool useScalarEpilogue = false;
  if (succeeded(scalarEpilogue) &&
      scalarEpilogue->op.getDpsInputs()[0] == convolutionOp.getResult(0)) {
    epilogueOp = scalarEpilogue->op;
    useScalarEpilogue = true;
  } else if (succeeded(perChannelEpilogue) &&
             perChannelEpilogue->op.getDpsInputs()[0] ==
                 convolutionOp.getResult(0)) {
    epilogueOp = perChannelEpilogue->op;
  } else {
    return failure();
  }
  IREE::TensorExt::DispatchTensorStoreOp outputStore;
  int storeCount = 0;
  funcOp.walk([&](IREE::TensorExt::DispatchTensorStoreOp storeOp) {
    outputStore = storeOp;
    ++storeCount;
  });
  if (storeCount != 1 ||
      outputStore.getValue().getDefiningOp() != epilogueOp.getOperation()) {
    return failure();
  }

  IREE::TensorExt::DispatchTensorLoadOp inputLoad;
  IREE::TensorExt::DispatchTensorLoadOp filterLoad;
  FailureOr<uint16_t> inputBinding =
      inferLoadBinding(convolutionOp.getDpsInputs()[0], inputLoad);
  FailureOr<uint16_t> filterBinding =
      inferLoadBinding(convolutionOp.getDpsInputs()[1], filterLoad);
  auto outputSubspan =
      outputStore.getTarget()
          .getDefiningOp<IREE::HAL::InterfaceBindingSubspanOp>();
  if (failed(inputBinding) || failed(filterBinding) || !outputSubspan ||
      outputSubspan.getBinding().getZExtValue() >
          std::numeric_limits<uint16_t>::max()) {
    return failure();
  }

  BuiltinSelection selection;
  selection.program = TT_IREE_TTEX_BUILTIN_PROGRAM_F32_CONV2D;
  selection.dispatchKind = TT_IREE_TTEX_DISPATCH_KIND_CONVOLUTION;
  selection.conv2d.kernel_h = static_cast<uint32_t>(kernelH);
  selection.conv2d.kernel_w = static_cast<uint32_t>(kernelW);
  selection.conv2d.stride_h = convolutionStride;
  selection.conv2d.stride_w = convolutionStride;
  selection.conv2d.dilation_h = 1;
  selection.conv2d.dilation_w = 1;
  selection.conv2d.groups = 1;
  selection.conv2d.epilogue_flags =
      TT_IREE_TTEX_EPILOGUE_AFFINE | TT_IREE_TTEX_EPILOGUE_RELU;
  selection.conv2d.epilogue_scale = 1.0f;
  selection.conv2d.epilogue_common_scale = 1.0f;
  if (useScalarEpilogue) {
    selection.conv2d.epilogue_scale = scalarEpilogue->scale;
    selection.conv2d.epilogue_bias = scalarEpilogue->bias;
  } else {
    selection.conv2d.epilogue_flags |= TT_IREE_TTEX_EPILOGUE_PER_CHANNEL;
    if (!perChannelEpilogue->scales.empty()) {
      selection.convEpilogueScales = std::move(perChannelEpilogue->scales);
      selection.convEpilogueBiases = std::move(perChannelEpilogue->biases);
    } else {
      selection.conv2d.epilogue_flags |=
          TT_IREE_TTEX_EPILOGUE_RUNTIME_AFFINE;
      selection.conv2d.epilogue_common_scale =
          perChannelEpilogue->commonScale;
      if (failed(populateRuntimeVectorDesc(
              perChannelEpilogue->mean, static_cast<size_t>(outputC),
              selection.conv2d.epilogue_mean)) ||
          failed(populateRuntimeVectorDesc(
              perChannelEpilogue->channelScale,
              static_cast<size_t>(outputC),
              selection.conv2d.epilogue_channel_scale)) ||
          failed(populateRuntimeVectorDesc(
              perChannelEpilogue->channelBias, static_cast<size_t>(outputC),
              selection.conv2d.epilogue_channel_bias))) {
        return failure();
      }
    }
  }

  uint64_t m = static_cast<uint64_t>(outputH) * outputW;
  uint64_t n = static_cast<uint64_t>(outputC);
  uint64_t k = static_cast<uint64_t>(inputC) * kernelH * kernelW;
  selection.mTiles = static_cast<uint32_t>((m + 31) / 32);
  selection.nTiles = static_cast<uint32_t>((n + 31) / 32);
  selection.kTiles = static_cast<uint32_t>((k + 31) / 32);

  auto inputSubspan =
      inputLoad.getSource()
          .getDefiningOp<IREE::HAL::InterfaceBindingSubspanOp>();
  auto filterSubspan =
      filterLoad.getSource()
          .getDefiningOp<IREE::HAL::InterfaceBindingSubspanOp>();
  FailureOr<InferredByteOffset> inputByteOffset = inferByteOffset(inputSubspan);
  FailureOr<InferredByteOffset> filterByteOffset =
      inferByteOffset(filterSubspan);
  FailureOr<InferredByteOffset> outputByteOffset =
      inferByteOffset(outputSubspan);
  if (failed(inputByteOffset) || failed(filterByteOffset) ||
      failed(outputByteOffset)) {
    return failure();
  }

  if (failed(populateTensorDesc(
          inputType, inputLoad.getSourceType().getShape(),
          inputLoad.getMixedOffsets(), inputLoad.getMixedStrides(),
          *inputBinding, *inputByteOffset,
          TT_IREE_TTEX_TENSOR_LAYOUT_CHW, selection.conv2d.input)) ||
      failed(populateTensorDesc(
          filterType, filterLoad.getSourceType().getShape(),
          filterLoad.getMixedOffsets(), filterLoad.getMixedStrides(),
          *filterBinding, *filterByteOffset,
          TT_IREE_TTEX_TENSOR_LAYOUT_OIHW, selection.conv2d.filter)) ||
      failed(populateTensorDesc(
          outputType, outputStore.getTargetType().getShape(),
          outputStore.getMixedOffsets(), outputStore.getMixedStrides(),
          static_cast<uint16_t>(outputSubspan.getBinding().getZExtValue()),
          *outputByteOffset, TT_IREE_TTEX_TENSOR_LAYOUT_CHW,
          selection.conv2d.output))) {
    return failure();
  }
  return selection;
}

static FailureOr<BuiltinSelection> inferF32GlobalSumProgram(func::FuncOp funcOp) {
  linalg::GenericOp reductionOp;
  int genericCount = 0;
  funcOp.walk([&](linalg::GenericOp op) {
    reductionOp = op;
    ++genericCount;
  });
  if (genericCount != 1 || reductionOp.getNumDpsInputs() != 1 ||
      reductionOp.getNumDpsInits() != 2 || reductionOp->getNumResults() != 2) {
    return failure();
  }

  auto dummyType = dyn_cast<RankedTensorType>(
      reductionOp.getDpsInputs()[0].getType());
  auto sumType = dyn_cast<RankedTensorType>(reductionOp.getResult(0).getType());
  auto divisorType =
      dyn_cast<RankedTensorType>(reductionOp.getResult(1).getType());
  if (!dummyType || !sumType || !divisorType || !dummyType.hasStaticShape() ||
      !sumType.hasStaticShape() || !divisorType.hasStaticShape() ||
      dummyType.getShape() != ArrayRef<int64_t>({49}) ||
      !dummyType.getElementType().isInteger(1) || sumType.getRank() != 1 ||
      !sumType.getElementType().isF32() || divisorType.getRank() != 0 ||
      !divisorType.getElementType().isF32()) {
    return failure();
  }

  MLIRContext *context = funcOp.getContext();
  AffineExpr d0 = getAffineDimExpr(0, context);
  AffineExpr d1 = getAffineDimExpr(1, context);
  SmallVector<AffineMap, 3> expectedMaps = {
      AffineMap::get(2, 0, {d1}, context),
      AffineMap::get(2, 0, {d0}, context),
      AffineMap::get(2, 0, {}, context),
  };
  auto iteratorTypes = reductionOp.getIteratorTypesArray();
  if (reductionOp.getIndexingMapsArray() != expectedMaps ||
      iteratorTypes.size() != 2 ||
      iteratorTypes[0] != utils::IteratorType::parallel ||
      iteratorTypes[1] != utils::IteratorType::reduction) {
    return failure();
  }

  Block &body = reductionOp.getRegion().front();
  auto yieldOp = dyn_cast<linalg::YieldOp>(body.getTerminator());
  if (body.getNumArguments() != 3 || !yieldOp ||
      yieldOp.getValues().size() != 2 ||
      getConstantFloat(yieldOp.getValues()[1]) != std::optional<double>(49.0)) {
    return failure();
  }
  auto addOp = yieldOp.getValues()[0].getDefiningOp<arith::AddFOp>();
  if (!addOp ||
      (addOp.getLhs() != body.getArgument(1) &&
       addOp.getRhs() != body.getArgument(1))) {
    return failure();
  }

  IREE::TensorExt::DispatchTensorLoadOp inputLoad;
  tensor::ExtractOp extractOp;
  int extractCount = 0;
  funcOp.walk([&](tensor::ExtractOp op) {
    extractOp = op;
    ++extractCount;
  });
  if (extractCount != 1) return failure();
  FailureOr<uint16_t> inputBinding =
      inferLoadBinding(extractOp.getTensor(), inputLoad);
  auto inputType = dyn_cast<RankedTensorType>(inputLoad.getType());
  if (failed(inputBinding) || !inputType || !inputType.hasStaticShape() ||
      inputType.getRank() != 3 || !inputType.getElementType().isF32() ||
      inputType.getDimSize(0) != sumType.getDimSize(0) ||
      inputType.getDimSize(1) != 8 || inputType.getDimSize(2) != 8) {
    return failure();
  }

  IREE::TensorExt::DispatchTensorStoreOp sumStore;
  IREE::TensorExt::DispatchTensorStoreOp divisorStore;
  int storeCount = 0;
  funcOp.walk([&](IREE::TensorExt::DispatchTensorStoreOp op) {
    ++storeCount;
    if (op.getValue() == reductionOp.getResult(0)) sumStore = op;
    if (op.getValue() == reductionOp.getResult(1)) divisorStore = op;
  });
  if (storeCount != 2 || !sumStore || !divisorStore) return failure();

  auto inputSubspan =
      inputLoad.getSource().getDefiningOp<IREE::HAL::InterfaceBindingSubspanOp>();
  auto sumSubspan =
      sumStore.getTarget().getDefiningOp<IREE::HAL::InterfaceBindingSubspanOp>();
  auto divisorSubspan = divisorStore.getTarget()
                            .getDefiningOp<IREE::HAL::InterfaceBindingSubspanOp>();
  if (!inputSubspan || !sumSubspan || !divisorSubspan ||
      sumSubspan.getBinding().getZExtValue() >
          std::numeric_limits<uint16_t>::max() ||
      divisorSubspan.getBinding().getZExtValue() >
          std::numeric_limits<uint16_t>::max()) {
    return failure();
  }
  FailureOr<InferredByteOffset> inputByteOffset = inferByteOffset(inputSubspan);
  FailureOr<InferredByteOffset> sumByteOffset = inferByteOffset(sumSubspan);
  FailureOr<InferredByteOffset> divisorByteOffset =
      inferByteOffset(divisorSubspan);
  if (failed(inputByteOffset) || failed(sumByteOffset) ||
      failed(divisorByteOffset)) {
    return failure();
  }

  BuiltinSelection selection;
  selection.program = TT_IREE_TTEX_BUILTIN_PROGRAM_F32_GLOBAL_SUM;
  selection.dispatchKind = TT_IREE_TTEX_DISPATCH_KIND_GLOBAL_SUM;
  selection.globalSum.reduce_h = 7;
  selection.globalSum.reduce_w = 7;
  selection.globalSum.divisor_value = 49.0f;
  if (failed(populateTensorDesc(
          inputType, inputLoad.getSourceType().getShape(),
          inputLoad.getMixedOffsets(), inputLoad.getMixedStrides(),
          *inputBinding, *inputByteOffset, TT_IREE_TTEX_TENSOR_LAYOUT_CHW,
          selection.globalSum.input)) ||
      failed(populateTensorDesc(
          sumType, sumStore.getTargetType().getShape(),
          sumStore.getMixedOffsets(), sumStore.getMixedStrides(),
          static_cast<uint16_t>(sumSubspan.getBinding().getZExtValue()),
          *sumByteOffset, TT_IREE_TTEX_TENSOR_LAYOUT_ROW_MAJOR,
          selection.globalSum.sum))) {
    return failure();
  }
  populateScalarTensorDesc(
      static_cast<uint16_t>(divisorSubspan.getBinding().getZExtValue()),
      *divisorByteOffset, selection.globalSum.divisor);
  return selection;
}

static FailureOr<BuiltinSelection> inferF32ScalarDivProgram(func::FuncOp funcOp) {
  linalg::GenericOp divideOp;
  int genericCount = 0;
  funcOp.walk([&](linalg::GenericOp op) {
    divideOp = op;
    ++genericCount;
  });
  if (genericCount != 1 || divideOp.getNumDpsInputs() != 2 ||
      divideOp.getNumDpsInits() != 1 || divideOp->getNumResults() != 1) {
    return failure();
  }
  auto divisorType =
      dyn_cast<RankedTensorType>(divideOp.getDpsInputs()[0].getType());
  auto inputType =
      dyn_cast<RankedTensorType>(divideOp.getDpsInputs()[1].getType());
  auto outputType = dyn_cast<RankedTensorType>(divideOp.getResult(0).getType());
  if (!divisorType || !inputType || !outputType || divisorType.getRank() != 0 ||
      !divisorType.getElementType().isF32() || !inputType.hasStaticShape() ||
      inputType.getRank() != 1 || !inputType.getElementType().isF32() ||
      inputType != outputType) {
    return failure();
  }
  MLIRContext *context = funcOp.getContext();
  AffineExpr d0 = getAffineDimExpr(0, context);
  SmallVector<AffineMap, 3> expectedMaps = {
      AffineMap::get(1, 0, {}, context),
      AffineMap::get(1, 0, {d0}, context),
      AffineMap::get(1, 0, {d0}, context),
  };
  Block &body = divideOp.getRegion().front();
  auto yieldOp = dyn_cast<linalg::YieldOp>(body.getTerminator());
  auto divOp = yieldOp && yieldOp.getValues().size() == 1
                   ? yieldOp.getValues()[0].getDefiningOp<arith::DivFOp>()
                   : arith::DivFOp{};
  if (divideOp.getIndexingMapsArray() != expectedMaps ||
      divideOp.getIteratorTypesArray() !=
          ArrayRef<utils::IteratorType>({utils::IteratorType::parallel}) ||
      body.getNumArguments() != 3 || !divOp ||
      divOp.getLhs() != body.getArgument(1) ||
      divOp.getRhs() != body.getArgument(0)) {
    return failure();
  }

  IREE::TensorExt::DispatchTensorLoadOp divisorLoad;
  IREE::TensorExt::DispatchTensorLoadOp inputLoad;
  FailureOr<uint16_t> divisorBinding =
      inferLoadBinding(divideOp.getDpsInputs()[0], divisorLoad);
  FailureOr<uint16_t> inputBinding =
      inferLoadBinding(divideOp.getDpsInputs()[1], inputLoad);
  IREE::TensorExt::DispatchTensorStoreOp outputStore;
  int storeCount = 0;
  funcOp.walk([&](IREE::TensorExt::DispatchTensorStoreOp op) {
    outputStore = op;
    ++storeCount;
  });
  auto divisorSubspan = divisorLoad.getSource()
                            .getDefiningOp<IREE::HAL::InterfaceBindingSubspanOp>();
  auto inputSubspan =
      inputLoad.getSource().getDefiningOp<IREE::HAL::InterfaceBindingSubspanOp>();
  auto outputSubspan = outputStore
                           ? outputStore.getTarget().getDefiningOp<
                                 IREE::HAL::InterfaceBindingSubspanOp>()
                           : IREE::HAL::InterfaceBindingSubspanOp{};
  if (failed(divisorBinding) || failed(inputBinding) || storeCount != 1 ||
      outputStore.getValue() != divideOp.getResult(0) || !divisorSubspan ||
      !inputSubspan || !outputSubspan ||
      outputSubspan.getBinding().getZExtValue() >
          std::numeric_limits<uint16_t>::max()) {
    return failure();
  }
  FailureOr<InferredByteOffset> divisorByteOffset =
      inferByteOffset(divisorSubspan);
  FailureOr<InferredByteOffset> inputByteOffset = inferByteOffset(inputSubspan);
  FailureOr<InferredByteOffset> outputByteOffset =
      inferByteOffset(outputSubspan);
  if (failed(divisorByteOffset) || failed(inputByteOffset) ||
      failed(outputByteOffset)) {
    return failure();
  }

  BuiltinSelection selection;
  selection.program = TT_IREE_TTEX_BUILTIN_PROGRAM_F32_SCALAR_DIV;
  selection.dispatchKind = TT_IREE_TTEX_DISPATCH_KIND_SCALAR_DIV;
  populateScalarTensorDesc(*divisorBinding, *divisorByteOffset,
                           selection.scalarDiv.divisor);
  if (failed(populateTensorDesc(
          inputType, inputLoad.getSourceType().getShape(),
          inputLoad.getMixedOffsets(), inputLoad.getMixedStrides(),
          *inputBinding, *inputByteOffset, TT_IREE_TTEX_TENSOR_LAYOUT_ROW_MAJOR,
          selection.scalarDiv.input)) ||
      failed(populateTensorDesc(
          outputType, outputStore.getTargetType().getShape(),
          outputStore.getMixedOffsets(), outputStore.getMixedStrides(),
          static_cast<uint16_t>(outputSubspan.getBinding().getZExtValue()),
          *outputByteOffset, TT_IREE_TTEX_TENSOR_LAYOUT_ROW_MAJOR,
          selection.scalarDiv.output))) {
    return failure();
  }
  return selection;
}

static FailureOr<BuiltinSelection> inferF32LinearProgram(func::FuncOp funcOp) {
  linalg::MatmulOp matmulOp;
  linalg::GenericOp biasOp;
  int matmulCount = 0;
  int genericCount = 0;
  funcOp.walk([&](linalg::MatmulOp op) {
    matmulOp = op;
    ++matmulCount;
  });
  funcOp.walk([&](linalg::GenericOp op) {
    biasOp = op;
    ++genericCount;
  });
  if (matmulCount != 1 || genericCount != 1 ||
      matmulOp.getInputs().size() != 2 || matmulOp.getOutputs().size() != 1 ||
      biasOp.getNumDpsInputs() != 2 || biasOp.getNumDpsInits() != 1 ||
      !matchesIdentityElementwise(biasOp) ||
      biasOp.getDpsInputs()[0] != matmulOp.getResult(0)) {
    return failure();
  }
  auto inputType =
      dyn_cast<RankedTensorType>(matmulOp.getInputs()[0].getType());
  auto weightsType =
      dyn_cast<RankedTensorType>(matmulOp.getInputs()[1].getType());
  auto outputType = dyn_cast<RankedTensorType>(matmulOp.getResult(0).getType());
  auto biasType = dyn_cast<RankedTensorType>(biasOp.getDpsInputs()[1].getType());
  if (!inputType || !weightsType || !outputType || !biasType ||
      !inputType.hasStaticShape() || !weightsType.hasStaticShape() ||
      !outputType.hasStaticShape() || inputType.getRank() != 2 ||
      weightsType.getRank() != 2 || outputType.getRank() != 2 ||
      !inputType.getElementType().isF32() ||
      !weightsType.getElementType().isF32() ||
      !outputType.getElementType().isF32() || biasType != outputType) {
    return failure();
  }
  int64_t m = inputType.getDimSize(0);
  int64_t k = inputType.getDimSize(1);
  int64_t n = weightsType.getDimSize(0);
  if (m <= 0 || n <= 0 || k <= 0 || weightsType.getDimSize(1) != k ||
      outputType.getDimSize(0) != m || outputType.getDimSize(1) != n ||
      static_cast<uint64_t>(m) > std::numeric_limits<uint32_t>::max() ||
      static_cast<uint64_t>(n) > std::numeric_limits<uint32_t>::max() ||
      static_cast<uint64_t>(k) > std::numeric_limits<uint32_t>::max()) {
    return failure();
  }
  Block &body = biasOp.getRegion().front();
  auto yieldOp = dyn_cast<linalg::YieldOp>(body.getTerminator());
  auto addOp = yieldOp && yieldOp.getValues().size() == 1
                   ? yieldOp.getValues()[0].getDefiningOp<arith::AddFOp>()
                   : arith::AddFOp{};
  if (body.getNumArguments() != 3 || !addOp ||
      !((addOp.getLhs() == body.getArgument(0) &&
         addOp.getRhs() == body.getArgument(1)) ||
        (addOp.getLhs() == body.getArgument(1) &&
         addOp.getRhs() == body.getArgument(0)))) {
    return failure();
  }

  IREE::TensorExt::DispatchTensorLoadOp inputLoad;
  IREE::TensorExt::DispatchTensorLoadOp weightsLoad;
  IREE::TensorExt::DispatchTensorLoadOp biasLoad;
  FailureOr<uint16_t> inputBinding =
      inferLoadBinding(matmulOp.getInputs()[0], inputLoad);
  FailureOr<uint16_t> weightsBinding =
      inferLoadBinding(matmulOp.getInputs()[1], weightsLoad);
  FailureOr<uint16_t> biasBinding =
      inferLoadBinding(biasOp.getDpsInputs()[1], biasLoad);
  IREE::TensorExt::DispatchTensorStoreOp outputStore;
  int storeCount = 0;
  funcOp.walk([&](IREE::TensorExt::DispatchTensorStoreOp op) {
    outputStore = op;
    ++storeCount;
  });
  auto inputSubspan =
      inputLoad.getSource().getDefiningOp<IREE::HAL::InterfaceBindingSubspanOp>();
  auto weightsSubspan = weightsLoad.getSource()
                            .getDefiningOp<IREE::HAL::InterfaceBindingSubspanOp>();
  auto biasSubspan =
      biasLoad.getSource().getDefiningOp<IREE::HAL::InterfaceBindingSubspanOp>();
  auto outputSubspan = outputStore
                           ? outputStore.getTarget().getDefiningOp<
                                 IREE::HAL::InterfaceBindingSubspanOp>()
                           : IREE::HAL::InterfaceBindingSubspanOp{};
  if (failed(inputBinding) || failed(weightsBinding) || failed(biasBinding) ||
      storeCount != 1 || outputStore.getValue() != biasOp.getResult(0) ||
      !inputSubspan || !weightsSubspan || !biasSubspan || !outputSubspan ||
      outputSubspan.getBinding().getZExtValue() >
          std::numeric_limits<uint16_t>::max()) {
    return failure();
  }
  FailureOr<InferredByteOffset> inputByteOffset = inferByteOffset(inputSubspan);
  FailureOr<InferredByteOffset> weightsByteOffset =
      inferByteOffset(weightsSubspan);
  FailureOr<InferredByteOffset> biasByteOffset = inferByteOffset(biasSubspan);
  FailureOr<InferredByteOffset> outputByteOffset =
      inferByteOffset(outputSubspan);
  if (failed(inputByteOffset) || failed(weightsByteOffset) ||
      failed(biasByteOffset) || failed(outputByteOffset)) {
    return failure();
  }

  BuiltinSelection selection;
  selection.program = TT_IREE_TTEX_BUILTIN_PROGRAM_F32_LINEAR;
  selection.dispatchKind = TT_IREE_TTEX_DISPATCH_KIND_LINEAR;
  selection.linear.m = static_cast<uint32_t>(m);
  selection.linear.n = static_cast<uint32_t>(n);
  selection.linear.k = static_cast<uint32_t>(k);
  selection.mTiles = static_cast<uint32_t>((m + 31) / 32);
  selection.nTiles = static_cast<uint32_t>((n + 31) / 32);
  selection.kTiles = static_cast<uint32_t>((k + 31) / 32);
  if (failed(populateTensorDesc(
          inputType, inputLoad.getSourceType().getShape(),
          inputLoad.getMixedOffsets(), inputLoad.getMixedStrides(),
          *inputBinding, *inputByteOffset, TT_IREE_TTEX_TENSOR_LAYOUT_ROW_MAJOR,
          selection.linear.input)) ||
      failed(populateTensorDesc(
          weightsType, weightsLoad.getSourceType().getShape(),
          weightsLoad.getMixedOffsets(), weightsLoad.getMixedStrides(),
          *weightsBinding, *weightsByteOffset, TT_IREE_TTEX_TENSOR_LAYOUT_OI,
          selection.linear.weights)) ||
      failed(populateTensorDesc(
          biasType, biasLoad.getSourceType().getShape(),
          biasLoad.getMixedOffsets(), biasLoad.getMixedStrides(), *biasBinding,
          *biasByteOffset, TT_IREE_TTEX_TENSOR_LAYOUT_ROW_MAJOR,
          selection.linear.bias)) ||
      failed(populateTensorDesc(
          outputType, outputStore.getTargetType().getShape(),
          outputStore.getMixedOffsets(), outputStore.getMixedStrides(),
          static_cast<uint16_t>(outputSubspan.getBinding().getZExtValue()),
          *outputByteOffset, TT_IREE_TTEX_TENSOR_LAYOUT_ROW_MAJOR,
          selection.linear.output))) {
    return failure();
  }
  return selection;
}

static FailureOr<BuiltinSelection> inferBuiltinProgram(func::FuncOp funcOp) {
  FailureOr<BuiltinSelection> paddedCopySelection =
      inferF32PaddedCopyProgram(funcOp);
  if (succeeded(paddedCopySelection))
    return paddedCopySelection;

  FailureOr<BuiltinSelection> maxPoolSelection =
      inferF32MaxPool2DProgram(funcOp);
  if (succeeded(maxPoolSelection))
    return maxPoolSelection;

  FailureOr<BuiltinSelection> globalSumSelection =
      inferF32GlobalSumProgram(funcOp);
  if (succeeded(globalSumSelection))
    return globalSumSelection;

  FailureOr<BuiltinSelection> scalarDivSelection =
      inferF32ScalarDivProgram(funcOp);
  if (succeeded(scalarDivSelection))
    return scalarDivSelection;

  FailureOr<BuiltinSelection> linearSelection = inferF32LinearProgram(funcOp);
  if (succeeded(linearSelection))
    return linearSelection;

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

  FailureOr<BuiltinSelection> matmulLikeSelection =
      inferF32MatmulLikeProgram(funcOp);
  if (succeeded(matmulLikeSelection))
    return matmulLikeSelection;

  FailureOr<BuiltinSelection> convolutionSelection =
      inferF32Conv2DProgram(funcOp);
  if (succeeded(convolutionSelection))
    return convolutionSelection;

  int addOpCount = 0;
  funcOp.walk([&](linalg::GenericOp genericOp) {
    if (matchesBF16ElementwiseAdd(genericOp)) {
      ++addOpCount;
    }
  });
  if (addOpCount == 1) {
    return BuiltinSelection{
        static_cast<uint32_t>(TT_IREE_TTEX_BUILTIN_PROGRAM_CUSTOM_SFPI_ADD),
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
    desc.workgroup_size[0] = static_cast<uint32_t>(
        cast<IntegerAttr>(workgroupSizeValues[0]).getInt());
    desc.workgroup_size[1] = static_cast<uint32_t>(
        cast<IntegerAttr>(workgroupSizeValues[1]).getInt());
    desc.workgroup_size[2] = static_cast<uint32_t>(
        cast<IntegerAttr>(workgroupSizeValues[2]).getInt());
  }
}

// Maps one IREE workgroup grid dimension to one TT core grid dimension.
// Static matmuls use compile-time tile counts. Dynamic matmuls use the
// existing dispatch workload ABI: ordinal 2 is M and ordinal 3 is N.
class MaterializeMatmulWorkgroupCountPass final
    : public PassWrapper<MaterializeMatmulWorkgroupCountPass,
                         OperationPass<IREE::HAL::ExecutableVariantOp>> {
public:
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(
      MaterializeMatmulWorkgroupCountPass)

  void runOnOperation() override {
    auto variantOp = getOperation();
    ModuleOp innerModuleOp = variantOp.getInnerModule();

    for (auto exportOp : variantOp.getExportOps()) {
      auto funcOp =
          innerModuleOp.lookupSymbol<func::FuncOp>(exportOp.getName());
      if (!funcOp)
        continue;

      FailureOr<BuiltinSelection> selection = inferBuiltinProgram(funcOp);
      if (failed(selection) ||
          (selection->program !=
               static_cast<uint32_t>(
                   TT_IREE_TTEX_BUILTIN_PROGRAM_BF16_MATMUL_TILED) &&
           selection->program !=
               static_cast<uint32_t>(
                   TT_IREE_TTEX_BUILTIN_PROGRAM_F32_MATMUL_LIKE) &&
           selection->program !=
               static_cast<uint32_t>(TT_IREE_TTEX_BUILTIN_PROGRAM_F32_CONV2D) &&
           selection->program !=
               static_cast<uint32_t>(
                   TT_IREE_TTEX_BUILTIN_PROGRAM_F32_MAX_POOL_2D) &&
           selection->program != static_cast<uint32_t>(
                                     TT_IREE_TTEX_BUILTIN_PROGRAM_F32_GLOBAL_SUM) &&
           selection->program != static_cast<uint32_t>(
                                     TT_IREE_TTEX_BUILTIN_PROGRAM_F32_SCALAR_DIV) &&
           selection->program !=
               static_cast<uint32_t>(TT_IREE_TTEX_BUILTIN_PROGRAM_F32_LINEAR))) {
        continue;
      }

      OpBuilder builder(funcOp.getContext());
      builder.setInsertionPoint(funcOp.getBody().front().getTerminator());
      SmallVector<OpFoldResult, 2> workgroupCount;

      if (selection->dispatchKind == TT_IREE_TTEX_DISPATCH_KIND_MAX_POOL ||
          selection->dispatchKind == TT_IREE_TTEX_DISPATCH_KIND_GLOBAL_SUM ||
          selection->dispatchKind == TT_IREE_TTEX_DISPATCH_KIND_SCALAR_DIV) {
        workgroupCount.push_back(builder.getIndexAttr(1));
        workgroupCount.push_back(builder.getIndexAttr(1));
      } else if (selection->mTiles != 0 && selection->nTiles != 0) {
        uint32_t gridX = selection->nTiles;
        uint32_t gridY = selection->mTiles;
        if (selection->dispatchKind ==
                TT_IREE_TTEX_DISPATCH_KIND_MATMUL_LIKE ||
            selection->dispatchKind == TT_IREE_TTEX_DISPATCH_KIND_CONVOLUTION ||
            selection->dispatchKind == TT_IREE_TTEX_DISPATCH_KIND_LINEAR) {
          constexpr uint32_t kPortableGridLimit = 8;
          gridX = std::min(gridX, kPortableGridLimit);
          while (selection->nTiles % gridX != 0)
            --gridX;
          gridY = std::min(gridY, kPortableGridLimit);
          while (selection->mTiles % gridY != 0)
            --gridY;
        }
        workgroupCount.push_back(builder.getIndexAttr(gridX));
        workgroupCount.push_back(builder.getIndexAttr(gridY));
      } else {
        IREE::TensorExt::DispatchWorkloadOrdinalOp mOrdinal;
        IREE::TensorExt::DispatchWorkloadOrdinalOp nOrdinal;
        funcOp.walk([&](IREE::TensorExt::DispatchWorkloadOrdinalOp ordinalOp) {
          switch (ordinalOp.getOrdinal().getZExtValue()) {
          case 2:
            mOrdinal = ordinalOp;
            break;
          case 3:
            nOrdinal = ordinalOp;
            break;
          default:
            break;
          }
        });
        if (!mOrdinal || !nOrdinal ||
            mOrdinal->getBlock() !=
                funcOp.getBody().front().getTerminator()->getBlock() ||
            nOrdinal->getBlock() !=
                funcOp.getBody().front().getTerminator()->getBlock()) {
          exportOp.emitError("dynamic tiled matmul is missing dominating M/N "
                             "workload ordinals");
          return signalPassFailure();
        }

        Value tileSize =
            arith::ConstantIndexOp::create(builder, funcOp.getLoc(), 32);
        Value nTiles = arith::DivUIOp::create(builder, funcOp.getLoc(),
                                              nOrdinal, tileSize);
        Value mTiles = arith::DivUIOp::create(builder, funcOp.getLoc(),
                                              mOrdinal, tileSize);
        workgroupCount.push_back(nTiles);
        workgroupCount.push_back(mTiles);
      }

      IREE::Codegen::WorkgroupCountHintOp::create(builder, funcOp.getLoc(),
                                                  workgroupCount);
    }
  }

  StringRef getArgument() const final {
    return "iree-tenstorrent-materialize-matmul-workgroup-count";
  }

  StringRef getDescription() const final {
    return "Maps tiled matmul output tiles to the Tenstorrent core grid";
  }
};

} // namespace

//===----------------------------------------------------------------------===//
// TenstorrentTargetDevice
//===----------------------------------------------------------------------===//

IREE::HAL::DeviceTargetAttr TenstorrentTargetDevice::getDefaultDeviceTarget(
    MLIRContext *context, const TargetRegistry &targetRegistry) const {
  Builder b(context);

  SmallVector<NamedAttribute> deviceConfigAttrs;
  StringRef deviceId = getTargetDeviceIdOrDefault(options_.deviceId);
  deviceConfigAttrs.push_back(
      b.getNamedAttr("device_id", b.getStringAttr(deviceId)));
  deviceConfigAttrs.push_back(b.getNamedAttr(
      "target_arch", b.getStringAttr(getTargetArchName(deviceId))));

  SmallVector<IREE::HAL::ExecutableTargetAttr> executableTargetAttrs;
  targetRegistry.getTargetBackend("tenstorrent")
      ->getDefaultExecutableTargets(context, deviceId,
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
    MLIRContext *context, StringRef deviceID, DictionaryAttr deviceConfigAttr,
    SmallVectorImpl<IREE::HAL::ExecutableTargetAttr> &executableTargetAttrs)
    const {
  Builder b(context);
  SmallVector<NamedAttribute> backendConfigAttrs;
  StringRef selectedDeviceId =
      deviceID.empty() || deviceID == "tenstorrent"
          ? getTargetDeviceIdOrDefault(options_.deviceId)
          : deviceID;
  backendConfigAttrs.push_back(
      b.getNamedAttr("device_id", b.getStringAttr(selectedDeviceId)));
  backendConfigAttrs.push_back(b.getNamedAttr(
      "target_arch", b.getStringAttr(getTargetArchName(selectedDeviceId))));

  executableTargetAttrs.push_back(IREE::HAL::ExecutableTargetAttr::get(
      context, b.getStringAttr("tenstorrent"),
      b.getStringAttr(kExecutableFormat),
      b.getDictionaryAttr(backendConfigAttrs)));
}

//===----------------------------------------------------------------------===//
// buildTranslationPassPipeline
//===----------------------------------------------------------------------===//

void TenstorrentTargetBackend::buildTranslationPassPipeline(
    IREE::HAL::ExecutableTargetAttr targetAttr, OpPassManager &passManager) {
  passManager.addPass(std::make_unique<MaterializeMatmulWorkgroupCountPass>());
  // Lower dispatch.workgroup_count_from_slice to concrete workgroup counts.
  // IREE v3.11 split hint resolution out of ReconcileTranslationInfo.
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
// The serializer classifies each executable-source export, builds its TTEX
// dispatch descriptor, and embeds the FlatBuffer as a hal.executable.binary.
// Hardware-enabled builds also offline-compile supported matmul-like exports
// and place their TT-Metal source identity plus ELF artifacts in TTEX v3.

LogicalResult TenstorrentTargetBackend::serializeExecutable(
    const SerializationOptions &serOptions,
    IREE::HAL::ExecutableVariantOp variantOp, OpBuilder &executableBuilder) {
  auto loc = variantOp.getLoc();
  ModuleOp innerModuleOp = variantOp.getInnerModule();
  std::optional<uint8_t> targetArch =
      inferExecutableTargetArch(variantOp.getTarget(), options_.deviceId);
  if (!targetArch) {
    return variantOp.emitError(
        "unsupported Tenstorrent target architecture; expected "
        "p100a/blackhole or wormhole/n150/n300");
  }

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
  std::vector<std::vector<float>> convScaleStorage;
  std::vector<std::vector<float>> convBiasStorage;
  std::vector<std::vector<float>> matmulScaleStorage;
  std::vector<std::vector<float>> matmulBiasStorage;
  size_t exportCount = std::distance(exportOps.begin(), exportOps.end());
  nameStorage.reserve(exportCount);
  entryDescs.reserve(exportCount);
  convScaleStorage.reserve(exportCount);
  convBiasStorage.reserve(exportCount);
  matmulScaleStorage.reserve(exportCount);
  matmulBiasStorage.reserve(exportCount);
#ifdef TT_IREE_ENABLE_AOT
  std::vector<SerializedAOTBundleStorage> aotBundleStorage;
  aotBundleStorage.reserve(exportCount);
#endif

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
          "(static or dynamic), or a static F32 CHW x OI projection with "
          "direct or scalar/per-channel affine, residual, and ReLU "
          "epilogues, or a static F32 CHW x OIHW 3x3 stride-1/stride-2 or "
          "7x7 stride-2 convolution with scalar/per-channel affine and ReLU "
          "epilogues, a static rank-3 F32 padded copy, a static rank-3 F32 "
          "3x3 stride-2 max-pool, or the exact F32 global-sum, scalar-divide, "
          "and biased-linear ResNet50 tail operations");
    }

    desc.constant_count = inferConstantCount(exportOp);
    desc.binding_count = inferBindingCount(exportOp);
    desc.flags = 0;
    inferWorkgroupSize(exportOp, desc);
    desc.builtin_program = builtinSelection->program;
    desc.builtin_m_tiles = builtinSelection->mTiles;
    desc.builtin_n_tiles = builtinSelection->nTiles;
    desc.builtin_k_tiles = builtinSelection->kTiles;
    desc.dispatch_kind = builtinSelection->dispatchKind;
    desc.matmul_like = builtinSelection->matmulLike;
    desc.conv2d = builtinSelection->conv2d;
    desc.padded_copy = builtinSelection->paddedCopy;
    desc.max_pool = builtinSelection->maxPool;
    desc.global_sum = builtinSelection->globalSum;
    desc.scalar_div = builtinSelection->scalarDiv;
    desc.linear = builtinSelection->linear;
    if (builtinSelection->dispatchKind ==
            TT_IREE_TTEX_DISPATCH_KIND_MATMUL_LIKE &&
        !builtinSelection->matmulEpilogueScales.empty()) {
      matmulScaleStorage.push_back(
          std::move(builtinSelection->matmulEpilogueScales));
      matmulBiasStorage.push_back(
          std::move(builtinSelection->matmulEpilogueBiases));
      desc.matmul_like.epilogue_channel_count =
          matmulScaleStorage.back().size();
      desc.matmul_like.epilogue_scales = matmulScaleStorage.back().data();
      desc.matmul_like.epilogue_biases = matmulBiasStorage.back().data();
    }
    if (builtinSelection->dispatchKind ==
            TT_IREE_TTEX_DISPATCH_KIND_CONVOLUTION &&
        !builtinSelection->convEpilogueScales.empty()) {
      convScaleStorage.push_back(
          std::move(builtinSelection->convEpilogueScales));
      convBiasStorage.push_back(
          std::move(builtinSelection->convEpilogueBiases));
      desc.conv2d.epilogue_channel_count = convScaleStorage.back().size();
      desc.conv2d.epilogue_scales = convScaleStorage.back().data();
      desc.conv2d.epilogue_biases = convBiasStorage.back().data();
    }

#ifdef TT_IREE_ENABLE_AOT
    bool isAdd = builtinSelection->program ==
                 TT_IREE_TTEX_BUILTIN_PROGRAM_CUSTOM_SFPI_ADD;
    bool isBF16Matmul = builtinSelection->program ==
                            TT_IREE_TTEX_BUILTIN_PROGRAM_BF16_MATMUL_32X32X32 ||
                        builtinSelection->program ==
                            TT_IREE_TTEX_BUILTIN_PROGRAM_BF16_MATMUL_TILED;
    bool isMatmulLike = builtinSelection->dispatchKind ==
                        TT_IREE_TTEX_DISPATCH_KIND_MATMUL_LIKE;
    bool isConvolution = builtinSelection->dispatchKind ==
                         TT_IREE_TTEX_DISPATCH_KIND_CONVOLUTION;
    bool isLinear =
        builtinSelection->dispatchKind == TT_IREE_TTEX_DISPATCH_KIND_LINEAR;
    if (isAdd || isBF16Matmul || isMatmulLike || isConvolution ||
        isLinear) {
      aotBundleStorage.emplace_back();
      SerializedAOTBundleStorage &aot = aotBundleStorage.back();
      std::string aotError;
      TenstorrentAOTTargetArch aotTargetArch =
          static_cast<TenstorrentAOTTargetArch>(*targetArch);
      bool compiled =
          isAdd ? compileAddAOT(aotTargetArch, aot.bundle, aotError)
                : compileMatmulAOT(aotTargetArch, aot.bundle, aotError);
      if (!compiled) {
        return exportOp.emitError("TT-Metal AOT kernel compilation failed: ")
               << aotError;
      }
      aot.finalize();
      desc.aot_kernel_bundle = &aot.desc;
    }
#endif

    entryDescs.push_back(desc);
  }

  // ----------------------------------------------------------------
  // 3. Build TTEX FlatBuffer.
  // ----------------------------------------------------------------
  iree_byte_span_t ttexData = {};
  iree_status_t status = tt_iree_build_ttex_executable_def(
      iree_allocator_system(), entryDescs.size(), entryDescs.data(), &ttexData);

  if (!iree_status_is_ok(status)) {
    iree_status_ignore(status);
    return variantOp.emitError("failed to build TTEX FlatBuffer");
  }

  // ----------------------------------------------------------------
  // 4. Debug dump (optional).
  // ----------------------------------------------------------------
  if (options_.dumpTTEX) {
    std::string dumpPath = (variantOp.getSymName() + ".ttex").str();
    std::error_code ec;
    llvm::raw_fd_ostream out(dumpPath, ec);
    if (!ec) {
      out.write(reinterpret_cast<const char *>(ttexData.data),
                ttexData.data_length);
      llvm::errs() << "tt-iree: TTEX dumped to " << dumpPath << " ("
                   << ttexData.data_length << " bytes)\n";
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
      executableBuilder, loc, variantOp.getSymName(),
      variantOp.getTarget().getFormat(), dataAttr);
  (void)binaryOp;

  // ----------------------------------------------------------------
  // 6. Cleanup.
  // ----------------------------------------------------------------
  iree_allocator_free(iree_allocator_system(), ttexData.data);

  return success();
}

} // namespace mlir::iree_compiler::IREE::HAL
