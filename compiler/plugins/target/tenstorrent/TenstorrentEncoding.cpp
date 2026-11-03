// Copyright 2026 The tt-iree Authors
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "TenstorrentEncoding.h"

#include "iree/compiler/Codegen/Dialect/Codegen/IR/IREECodegenTypes.h"
#include "iree/compiler/Codegen/Dialect/Codegen/Utils/Utils.h"
#include "iree/compiler/Codegen/ExternalInterfaces/Utils.h"
#include "iree/compiler/Codegen/Utils/Utils.h"
#include "iree/compiler/Dialect/Encoding/IR/EncodingTypes.h"
#include "iree/compiler/Dialect/Encoding/Utils/Utils.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/IR/DialectImplementation.h"
#include "mlir/Interfaces/ValueBoundsOpInterface.h"

MLIR_DEFINE_EXPLICIT_TYPE_ID(
    mlir::iree_compiler::tenstorrent::EncodingResolverAttr)
MLIR_DEFINE_EXPLICIT_TYPE_ID(mlir::iree_compiler::tenstorrent::TTIREEDialect)

namespace mlir::iree_compiler::tenstorrent {

using IREE::Codegen::MaterializeEncodingInfo;
using IREE::Codegen::TileSwizzle;

namespace detail {
struct EncodingResolverAttrStorage : public AttributeStorage {
  using KeyTy = DictionaryAttr;
  explicit EncodingResolverAttrStorage(DictionaryAttr configuration)
      : configuration(configuration) {}
  bool operator==(const KeyTy &key) const { return key == configuration; }
  static llvm::hash_code hashKey(const KeyTy &key) {
    return mlir::hash_value(key);
  }
  static EncodingResolverAttrStorage *
  construct(AttributeStorageAllocator &allocator, const KeyTy &key) {
    return new (allocator.allocate<EncodingResolverAttrStorage>())
        EncodingResolverAttrStorage(key);
  }
  DictionaryAttr configuration;
};
} // namespace detail

EncodingResolverAttr EncodingResolverAttr::get(MLIRContext *context,
                                               DictionaryAttr configuration) {
  if (!configuration)
    configuration = DictionaryAttr::get(context);
  return Base::get(context, configuration);
}

DictionaryAttr EncodingResolverAttr::getConfiguration() const {
  return getImpl()->configuration;
}

TTIREEDialect::TTIREEDialect(MLIRContext *context)
    : Dialect(getDialectNamespace(), context, TypeID::get<TTIREEDialect>()) {
  addAttributes<EncodingResolverAttr>();
}

Attribute TTIREEDialect::parseAttribute(DialectAsmParser &parser,
                                        Type type) const {
  StringRef keyword;
  DictionaryAttr configuration;
  if (parser.parseKeyword(&keyword))
    return {};
  if (keyword != "encoding_resolver") {
    parser.emitError(parser.getNameLoc(), "unknown tt_iree attribute: ")
        << keyword;
    return {};
  }
  if (parser.parseLess())
    return {};
  if (failed(parser.parseOptionalGreater())) {
    if (parser.parseAttribute(configuration) || parser.parseGreater())
      return {};
  }
  return EncodingResolverAttr::get(getContext(), configuration);
}

void TTIREEDialect::printAttribute(Attribute attr,
                                   DialectAsmPrinter &printer) const {
  auto resolver = cast<EncodingResolverAttr>(attr);
  printer << "encoding_resolver<";
  if (!resolver.getConfiguration().empty())
    printer << resolver.getConfiguration();
  printer << ">";
}

namespace {

// Faces of a 32x32 tile in Tensix order: [2,16] x [2,16] -> [2,2,16,16].
TileSwizzle faceSwizzle() {
  TileSwizzle swizzle;
  for (int i = 0; i < 2; ++i)
    swizzle.expandShape().push_back({TileSwizzle::Dim::internal(2),
                                     TileSwizzle::Dim::internal(kFaceSize)});
  swizzle.permutation() = {0, 2, 1, 3};
  return swizzle;
}

// Only plain BF16 matmuls in row-major operand order are tiled. Everything
// else keeps its existing row-major dispatch ABI.
MaterializeEncodingInfo getTileEncodingInfo(RankedTensorType type) {
  auto encoding = dyn_cast_if_present<IREE::Encoding::EncodingAttr>(
      type.getEncoding());
  if (!encoding)
    return {};
  SmallVector<Type> elementTypes = encoding.getElementTypesArray();
  if (elementTypes.size() != 3 || !elementTypes[0].isBF16() ||
      !elementTypes[1].isBF16())
    return {};
  auto dims = IREE::Codegen::getEncodingContractionLikeDims(encoding);
  if (failed(dims) || dims->batchDim.shouldHaveDim || dims->kBDim.shouldHaveDim)
    return {};
  int64_t operand = encoding.getOperandIndex().getInt();
  auto at = [](const auto &dim, unsigned index) {
    return dim.operandIdx && *dim.operandIdx == index;
  };
  bool rowMajor =
      operand == IREE::Encoding::MATMUL_LHS   ? at(dims->mDim, 0) && at(dims->kDim, 1)
      : operand == IREE::Encoding::MATMUL_RHS ? at(dims->kDim, 0) && at(dims->nDim, 1)
      : operand == IREE::Encoding::MATMUL_RESULT
          ? at(dims->mDim, 0) && at(dims->nDim, 1)
          : false;
  if (!rowMajor || type.getRank() != 2)
    return {};
  FailureOr<MaterializeEncodingInfo> info =
      IREE::Codegen::getEncodingInfoForMatmul(
          encoding, IREE::Codegen::TileMxNxK{kTileSize, kTileSize, kTileSize});
  if (failed(info))
    return {};
  // Keep the RHS tile K-major ([K0,N0]) so it matches the Tensix B operand;
  // tiles stay ordered [N1,K1] so one output column streams contiguous tiles.
  if (operand == IREE::Encoding::MATMUL_RHS)
    info->innerDimsPos = {0, 1};
  info->swizzle = faceSwizzle();
  return *info;
}

// Lowers a tiled contraction to a face-level generic:
// (M1,N1,K1, mf,nf,kf, mi,ni,ki), with the original scalar body.
Operation *lowerTileContraction(OpBuilder &builder, linalg::LinalgOp op,
                                ValueRange operands,
                                IREE::Encoding::LayoutMaterializerAttr attr) {
  auto inputs = op.getDpsInputOperands();
  auto layout = cast<IREE::Codegen::PackedLayoutMaterializerAttr>(attr);
  for (Value v : {inputs[0]->get(), inputs[1]->get(), op.getDpsInits()[0]})
    if (IREE::Codegen::isIdentityLayout(
            layout.getEncodingInfo(cast<RankedTensorType>(v.getType()))))
      return dropEncodingAndCloneOp(builder, op,
                                    operands.take_front(inputs.size()),
                                    operands.drop_front(inputs.size()));
  MLIRContext *ctx = builder.getContext();
  auto d = [&](unsigned i) { return getAffineDimExpr(i, ctx); };
  // 0:M1 1:N1 2:K1 3:mf 4:nf 5:kf 6:mi 7:ni 8:ki
  SmallVector<AffineMap> maps = {
      AffineMap::get(9, 0, {d(0), d(2), d(3), d(5), d(6), d(8)}, ctx),
      AffineMap::get(9, 0, {d(1), d(2), d(5), d(4), d(8), d(7)}, ctx),
      AffineMap::get(9, 0, {d(0), d(1), d(3), d(4), d(6), d(7)}, ctx)};
  using utils::IteratorType;
  SmallVector<IteratorType> iterators = {
      IteratorType::parallel,  IteratorType::parallel, IteratorType::reduction,
      IteratorType::parallel,  IteratorType::parallel, IteratorType::reduction,
      IteratorType::parallel,  IteratorType::parallel, IteratorType::reduction};
  Value init = operands[2];
  auto generic = linalg::GenericOp::create(
      builder, op.getLoc(), TypeRange{init.getType()},
      ValueRange{operands[0], operands[1]}, ValueRange{init}, maps, iterators);
  IRMapping mapping;
  op->getRegion(0).cloneInto(&generic.getRegion(), mapping);
  generic->setAttr(kTileMatmulAttrName, UnitAttr::get(ctx));
  return generic;
}

struct PackedLayoutModel
    : IREE::PackedLayoutMaterializerAttrExternalModelBase<PackedLayoutModel,
                                                          EncodingResolverAttr> {
  DictionaryAttr getConfiguration(Attribute attr) const {
    return cast<EncodingResolverAttr>(attr).getConfiguration();
  }
  MaterializeEncodingInfo getEncodingInfoImpl(Attribute,
                                              RankedTensorType type) const {
    return getTileEncodingInfo(type);
  }
};

struct LayoutMaterializerModel
    : IREE::EncodingLayoutMaterializerAttrExternalModelBase<
          LayoutMaterializerModel, EncodingResolverAttr> {
  Operation *lowerOp(Attribute attr, OpBuilder &b, Operation *op,
                     TypeRange convertedResTypes,
                     ValueRange convertedOperands) const {
    auto linalgOp = dyn_cast<linalg::LinalgOp>(op);
    if (!linalgOp)
      return nullptr;
    auto layout = cast<IREE::Encoding::LayoutMaterializerAttr>(attr);
    if (auto fillOp = dyn_cast<linalg::FillOp>(op))
      return IREE::lowerFillOpWithResolvedLayouts(b, fillOp, convertedResTypes,
                                                  convertedOperands);
    if (linalg::isaContractionOpInterface(linalgOp))
      return lowerTileContraction(b, linalgOp, convertedOperands, layout);
    if (auto genericOp = dyn_cast<linalg::GenericOp>(op))
      return IREE::lowerGenericOpWithResolvedLayouts(
          b, genericOp, convertedResTypes, convertedOperands, layout);
    return nullptr;
  }
};

struct LayoutResolverModel
    : IREE::Encoding::LayoutResolverAttr::ExternalModel<LayoutResolverModel,
                                                        EncodingResolverAttr> {
  Attribute cloneWithSimplifiedConfig(Attribute attr, DictionaryAttr) const {
    return EncodingResolverAttr::get(attr.getContext(), {});
  }
  Attribute getLayout(Attribute attr, RankedTensorType type) const {
    return EncodingResolverAttr::get(attr.getContext(),
                                     IREE::getPackedLayoutImpl(attr, type));
  }
};

struct SerializableModel
    : IREE::Encoding::SerializableAttr::ExternalModel<SerializableModel,
                                                      EncodingResolverAttr> {
  bool isSerialized(Attribute attr) const {
    return cast<EncodingResolverAttr>(attr).getConfiguration().contains(
        IREE::kEncodingInfoAttrName);
  }
  Value calculateStorageSizeInBytes(Attribute attr, Location loc,
                                    OpBuilder &builder, RankedTensorType type,
                                    ValueRange dynamicDims) const {
    return IREE::calculatePackedStorageSizeInBytesImpl(attr, loc, builder,
                                                       type, dynamicDims);
  }
};

struct VerifierModel
    : VerifiableTensorEncoding::ExternalModel<VerifierModel,
                                              EncodingResolverAttr> {
  LogicalResult
  verifyEncoding(Attribute attr, ArrayRef<int64_t> shape, Type elementType,
                 function_ref<InFlightDiagnostic()> emitError) const {
    return cast<IREE::Codegen::PackedLayoutMaterializerAttr>(attr)
        .verifyPackedLayoutWithType(shape, elementType, emitError);
  }
};

} // namespace

void registerTenstorrentEncoding(DialectRegistry &registry) {
  registry.insert<TTIREEDialect>();
  registry.addExtension(+[](MLIRContext *context, TTIREEDialect *) {
    EncodingResolverAttr::attachInterface<PackedLayoutModel,
                                          LayoutMaterializerModel,
                                          LayoutResolverModel,
                                          SerializableModel, VerifierModel>(
        *context);
  });
}

} // namespace mlir::iree_compiler::tenstorrent
