// Copyright 2026 The tt-iree Authors
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// Static ResNet BF16 patterns, not a general MLIR interpreter.

static std::string bf16FloatLiteral(float value) {
  uint32_t raw;
  memcpy(&raw, &value, 4);
  return "from_bits(" + std::to_string(raw) + "u)";
}

static bool bf16MatrixProduct(uint32_t &result,
                              std::initializer_list<int64_t> dimensions) {
  uint64_t product = 1;
  for (int64_t dimension : dimensions) {
    if (dimension <= 0 || uint64_t(dimension) > (UINT32_MAX - 31u) / product)
      return false;
    product *= dimension;
  }
  result = product;
  return true;
}

static LogicalResult bf16LoadDesc(Value value, uint8_t layout,
                                  tt_iree_ttex_tensor_desc_t &desc) {
  IREE::TensorExt::DispatchTensorLoadOp load;
  auto binding = inferLoadBinding(value, load);
  if (failed(binding) || !load)
    return failure();
  auto subspan =
      load.getSource().getDefiningOp<IREE::HAL::InterfaceBindingSubspanOp>();
  auto offset = inferByteOffset(subspan);
  auto type = dyn_cast<RankedTensorType>(value.getType());
  if (failed(offset) || !type ||
      (!type.getElementType().isBF16() && !type.getElementType().isF32()))
    return failure();
  if (type.getRank() == 0) {
    populateScalarTensorDesc(*binding, *offset, desc);
    desc.element_type = type.getElementType().isBF16()
                            ? TT_IREE_TTEX_TENSOR_ELEMENT_TYPE_BF16
                            : TT_IREE_TTEX_TENSOR_ELEMENT_TYPE_F32;
    return success();
  }
  return populateTensorDesc(type, load.getSourceType().getShape(),
                            load.getMixedOffsets(), load.getMixedStrides(),
                            *binding, *offset, layout, desc);
}

static LogicalResult bf16StoreDesc(IREE::TensorExt::DispatchTensorStoreOp store,
                                   uint8_t layout,
                                   tt_iree_ttex_tensor_desc_t &desc) {
  auto subspan =
      store.getTarget().getDefiningOp<IREE::HAL::InterfaceBindingSubspanOp>();
  auto offset = inferByteOffset(subspan);
  auto type = dyn_cast<RankedTensorType>(store.getValue().getType());
  if (!subspan || failed(offset) || !type ||
      (!type.getElementType().isBF16() && !type.getElementType().isF32()) ||
      subspan.getBinding().getZExtValue() > UINT16_MAX)
    return failure();
  if (type.getRank() == 0) {
    populateScalarTensorDesc(subspan.getBinding().getZExtValue(), *offset,
                             desc);
    desc.element_type = type.getElementType().isBF16()
                            ? TT_IREE_TTEX_TENSOR_ELEMENT_TYPE_BF16
                            : TT_IREE_TTEX_TENSOR_ELEMENT_TYPE_F32;
    return success();
  }
  return populateTensorDesc(type, store.getTargetType().getShape(),
                            store.getMixedOffsets(), store.getMixedStrides(),
                            subspan.getBinding().getZExtValue(), *offset,
                            layout, desc);
}

// Every explicit BF16 arithmetic result/cast gets its own RNE boundary.
// Keep scalar SSA order: folding BN into a single affine expression loses
// the convolution/output and residual branch BF16 rounding boundaries.
class BF16ScalarEmitter {
public:
  llvm::DenseMap<Value, std::string> values;
  std::ostringstream code;
  unsigned next = 0;

  FailureOr<std::string> get(Value value) {
    auto it = values.find(value);
    if (it != values.end())
      return it->second;
    auto constant = getConstantFloat(value);
    if (constant)
      return bf16FloatLiteral(static_cast<float>(*constant));
    return failure();
  }

  FailureOr<std::string> emit(linalg::GenericOp op) {
    for (Operation &scalar : op.getRegion().front()) {
      if (auto yield = dyn_cast<linalg::YieldOp>(scalar)) {
        if (yield.getValues().size() != 1)
          return failure();
        return get(yield.getValues()[0]);
      }
      if (isa<arith::ConstantOp>(scalar))
        continue;
      if (scalar.getNumResults() != 1)
        return failure();
      SmallVector<std::string> args;
      for (Value operand : scalar.getOperands()) {
        auto expr = get(operand);
        if (failed(expr))
          return failure();
        args.push_back(*expr);
      }
      std::string expr;
      bool comparison = false;
      if (isa<arith::ExtFOp>(scalar) && args.size() == 1 &&
          scalar.getOperand(0).getType().isBF16() &&
          scalar.getResult(0).getType().isF32())
        expr = args[0];
      else if (isa<arith::TruncFOp>(scalar) && args.size() == 1 &&
               scalar.getOperand(0).getType().isF32() &&
               scalar.getResult(0).getType().isBF16())
        expr = args[0];
      else if (isa<arith::AddFOp, arith::SubFOp, arith::MulFOp, arith::DivFOp>(
                   scalar) &&
               args.size() == 2) {
        const char *symbol = isa<arith::AddFOp>(scalar)   ? "+"
                             : isa<arith::SubFOp>(scalar) ? "-"
                             : isa<arith::MulFOp>(scalar) ? "*"
                                                          : "/";
        expr = "(" + args[0] + symbol + args[1] + ")";
      } else if (auto cmp = dyn_cast<arith::CmpFOp>(scalar);
                 cmp && args.size() == 2) {
        if (cmp.getPredicate() != arith::CmpFPredicate::UGT)
          return failure();
        expr = "!(" + args[0] + " <= " + args[1] + ")";
        comparison = true;
      } else if (isa<arith::SelectOp>(scalar) && args.size() == 3) {
        expr = "(" + args[0] + " ? " + args[1] + " : " + args[2] + ")";
      } else
        return failure();
      if (scalar.getResult(0).getType().isBF16())
        expr = "round_bf16(" + expr + ")";
      std::string name = "bfv" + std::to_string(next++);
      code << (comparison ? "bool " : "float ") << name << "=" << expr << ";\n";
      values[scalar.getResult(0)] = name;
    }
    return failure();
  }
};

static bool bf16ContractionBody(linalg::GenericOp op) {
  auto &body = op.getRegion().front();
  if (body.getNumArguments() != 3 || !body.getArgument(2).getType().isF32())
    return false;
  auto yield = dyn_cast<linalg::YieldOp>(body.getTerminator());
  if (!yield || yield.getValues().size() != 1)
    return false;
  auto add = yield.getValues()[0].getDefiningOp<arith::AddFOp>();
  if (!add)
    return false;
  auto product = (add.getLhs() == body.getArgument(2)   ? add.getRhs()
                  : add.getRhs() == body.getArgument(2) ? add.getLhs()
                                                        : Value{});
  auto mul = product ? product.getDefiningOp<arith::MulFOp>() : arith::MulFOp{};
  if (!mul)
    return false;
  auto source = [](Value v) {
    if (auto ext = v.getDefiningOp<arith::ExtFOp>())
      if (ext.getIn().getType().isBF16() && ext.getType().isF32())
        return ext.getIn();
    return v;
  };
  return source(mul.getLhs()) == body.getArgument(0) &&
         source(mul.getRhs()) == body.getArgument(1);
}

static bool bf16BatchedProjection(linalg::GenericOp op, bool spatial) {
  unsigned loops = spatial ? 5 : 4;
  auto *ctx = op.getContext();
  SmallVector<AffineExpr> d;
  for (unsigned i = 0; i < loops; ++i)
    d.push_back(getAffineDimExpr(i, ctx));
  SmallVector<AffineExpr> input = {d[0], d.back(), d[2]};
  SmallVector<AffineExpr> output = {d[0], d[1], d[2]};
  if (spatial) {
    input.push_back(d[3]);
    output.push_back(d[3]);
  }
  SmallVector<utils::IteratorType> iterators(loops,
                                             utils::IteratorType::parallel);
  iterators.back() = utils::IteratorType::reduction;
  return op.getIteratorTypesArray() == iterators &&
         op.getIndexingMapsArray() ==
             SmallVector<AffineMap>{
                 AffineMap::get(loops, 0, input, ctx),
                 AffineMap::get(loops, 0, {d[1], d.back()}, ctx),
                 AffineMap::get(loops, 0, output, ctx)};
}

static std::optional<uint32_t> bf16ConvolutionStride(linalg::GenericOp op) {
  if (auto stride = inferConv2DStride(op))
    return stride;
  auto *ctx = op.getContext();
  auto b = getAffineDimExpr(0, ctx), n = getAffineDimExpr(1, ctx),
       h = getAffineDimExpr(2, ctx), w = getAffineDimExpr(3, ctx),
       k = getAffineDimExpr(4, ctx), kh = getAffineDimExpr(5, ctx),
       kw = getAffineDimExpr(6, ctx);
  if (op.getIteratorTypesArray() !=
      SmallVector<utils::IteratorType>{
          utils::IteratorType::parallel, utils::IteratorType::parallel,
          utils::IteratorType::parallel, utils::IteratorType::parallel,
          utils::IteratorType::reduction, utils::IteratorType::reduction,
          utils::IteratorType::reduction})
    return std::nullopt;
  for (uint32_t stride : {1u, 2u})
    if (op.getIndexingMapsArray() ==
        SmallVector<AffineMap>{
            AffineMap::get(7, 0, {b, k, h * stride + kh, w * stride + kw}, ctx),
            AffineMap::get(7, 0, {n, k, kh, kw}, ctx),
            AffineMap::get(7, 0, {b, n, h, w}, ctx)})
      return stride;
  return std::nullopt;
}

static bool bf16GlobalSumBody(linalg::GenericOp op, tensor::ExtractOp extract) {
  auto &block = op.getRegion().front();
  auto yield = dyn_cast<linalg::YieldOp>(block.getTerminator());
  if (!yield || block.getNumArguments() != 3 || yield.getValues().size() != 2 ||
      !block.getArgument(0).use_empty() ||
      !block.getArgument(1).getType().isBF16() ||
      getConstantFloat(yield.getValues()[1]) != std::optional<double>(49.0) ||
      (extract.getIndices().size() != 3 && extract.getIndices().size() != 4))
    return false;
  auto add = yield.getValues()[0].getDefiningOp<arith::AddFOp>();
  if (!add)
    return false;
  Value selected = add.getLhs() == block.getArgument(1)   ? add.getRhs()
                   : add.getRhs() == block.getArgument(1) ? add.getLhs()
                                                          : Value{};
  if (!selected)
    return false;
  auto isSeven = [](Value value) {
    auto constant = value.getDefiningOp<arith::ConstantOp>();
    auto attr =
        constant ? dyn_cast<IntegerAttr>(constant.getValue()) : IntegerAttr{};
    return attr && attr.getInt() == 7;
  };
  unsigned first = extract.getIndices().size() == 4 ? 1 : 0;
  if (first) {
    auto batch = extract.getIndices()[0].getDefiningOp<arith::DivSIOp>();
    auto channel = extract.getIndices()[1].getDefiningOp<arith::RemSIOp>();
    auto type = cast<RankedTensorType>(extract.getTensor().getType());
    auto constant = batch ? batch.getRhs().getDefiningOp<arith::ConstantOp>()
                          : arith::ConstantOp{};
    auto channels =
        constant ? dyn_cast<IntegerAttr>(constant.getValue()) : IntegerAttr{};
    if (!batch || !channel || batch.getLhs() != channel.getLhs() ||
        batch.getRhs() != channel.getRhs() || !channels ||
        channels.getInt() != type.getDimSize(1))
      return false;
    auto index = batch.getLhs().getDefiningOp<linalg::IndexOp>();
    if (!index || index.getDim() != 0)
      return false;
  } else {
    auto channel = extract.getIndices()[0].getDefiningOp<linalg::IndexOp>();
    if (!channel || channel.getDim() != 0)
      return false;
  }
  auto height = extract.getIndices()[first + 1].getDefiningOp<arith::DivSIOp>();
  auto width = extract.getIndices()[first + 2].getDefiningOp<arith::RemSIOp>();
  if (!height || !width || height.getLhs() != width.getLhs() ||
      !isSeven(height.getRhs()) || !isSeven(width.getRhs()))
    return false;
  auto index = height.getLhs().getDefiningOp<linalg::IndexOp>();
  if (!index || index.getDim() != 1)
    return false;
  for (Value coordinate : {width.getResult(), height.getResult()}) {
    auto select = selected.getDefiningOp<arith::SelectOp>();
    auto cmp = select ? select.getCondition().getDefiningOp<arith::CmpIOp>()
                      : arith::CmpIOp{};
    if (!select || !cmp || cmp.getPredicate() != arith::CmpIPredicate::ult ||
        cmp.getLhs() != coordinate || !isSeven(cmp.getRhs()) ||
        getConstantFloat(select.getFalseValue()) != std::optional<double>(0.0))
      return false;
    selected = select.getTrueValue();
  }
  auto fill = op.getDpsInits()[0].getDefiningOp<linalg::FillOp>();
  return selected == extract.getResult() && fill &&
         getConstantFloat(fill.getInputs()[0]) == std::optional<double>(0.0);
}

static FailureOr<std::string> bf16Epilogue(linalg::GenericOp op,
                                           Value contraction, uint8_t layout,
                                           BuiltinSelection &selection,
                                           unsigned &slot) {
  auto type = dyn_cast<RankedTensorType>(op.getResult(0).getType());
  if (!type || op.getNumDpsInits() != 1 || op.getDpsInputs().empty() ||
      op.getDpsInputs()[0] != contraction ||
      !llvm::all_of(op.getIteratorTypesArray(),
                    [](auto i) { return i == utils::IteratorType::parallel; }))
    return failure();
  auto maps = op.getIndexingMapsArray();
  if (!maps.back().isIdentity() || !maps[0].isIdentity())
    return failure();
  BF16ScalarEmitter emitter;
  const bool nchw = layout == TT_IREE_TTEX_TENSOR_LAYOUT_NCHW;
  const bool ncm = layout == TT_IREE_TTEX_TENSOR_LAYOUT_BATCH_CHANNEL_MAJOR;
  unsigned channelAxis =
      (nchw || ncm || layout == TT_IREE_TTEX_TENSOR_LAYOUT_ROW_MAJOR) ? 1 : 0;
  auto &body = op.getRegion().front();
  emitter.values[body.getArgument(0)] = "value";
  std::ostringstream constants;
  for (unsigned i = 1; i < op.getNumDpsInputs(); ++i) {
    auto input = op.getDpsInputs()[i];
    auto inputType = dyn_cast<RankedTensorType>(input.getType());
    if (!inputType || !inputType.hasStaticShape())
      return failure();
    std::string coords;
    if (maps[i].isIdentity() && inputType.getShape() == type.getShape()) {
      if (nchw) {
        coords = "m/" +
                 std::to_string(type.getDimSize(2) * type.getDimSize(3)) +
                 "u,n,(m/" + std::to_string(type.getDimSize(3)) + "u)%" +
                 std::to_string(type.getDimSize(2)) + "u,m%" +
                 std::to_string(type.getDimSize(3)) + "u";
      } else if (ncm) {
        coords = "m/" + std::to_string(type.getDimSize(2)) + "u,n,m%" +
                 std::to_string(type.getDimSize(2)) + "u";
      } else
        coords = layout == TT_IREE_TTEX_TENSOR_LAYOUT_CHW
                     ? "n,m/" + std::to_string(type.getDimSize(2)) + "u,m%" +
                           std::to_string(type.getDimSize(2)) + "u"
                 : layout == TT_IREE_TTEX_TENSOR_LAYOUT_CHANNEL_MAJOR ? "n,m"
                                                                      : "m,n";
    } else if (inputType.getRank() == 1 && maps[i].getNumResults() == 1 &&
               maps[i].getResult(0) ==
                   getAffineDimExpr(channelAxis, op.getContext()))
      coords = "n";
    else
      return failure();
    if (auto constant = input.getDefiningOp<arith::ConstantOp>()) {
      auto attr = constant.getValue();
      SmallVector<float> elements;
      if (auto dense = dyn_cast<DenseElementsAttr>(attr)) {
        for (APFloat v : dense.getValues<APFloat>())
          elements.push_back(v.convertToFloat());
      } else if (auto resource = dyn_cast<DenseResourceElementsAttr>(attr)) {
        auto blob = resource.getRawHandle().getBlob();
        if (!blob)
          return failure();
        auto data = blob->getData();
        unsigned bytes = inputType.getElementType().isBF16() ? 2 : 4;
        if (data.size() != inputType.getNumElements() * bytes)
          return failure();
        for (unsigned j = 0; j < inputType.getNumElements(); ++j) {
          uint32_t raw = 0;
          memcpy(&raw, data.data() + j * bytes, bytes);
          if (bytes == 2)
            raw <<= 16;
          float v;
          memcpy(&v, &raw, 4);
          elements.push_back(v);
        }
      } else
        return failure();
      if (elements.size() != static_cast<size_t>(inputType.getNumElements()) ||
          inputType.getRank() != 1 || elements.size() > 2048)
        return failure();
      std::string name = "bfc" + std::to_string(i);
      constants << "static const uint32_t " << name << "[]={";
      for (float v : elements) {
        uint32_t raw;
        memcpy(&raw, &v, 4);
        constants << raw << "u,";
      }
      constants << "};\n";
      emitter.values[body.getArgument(i)] = "from_bits(" + name + "[n])";
    } else {
      if (slot >= TT_IREE_BF16_NATIVE_TENSOR_COUNT ||
          failed(bf16LoadDesc(input,
                              inputType.getRank() == 1
                                  ? TT_IREE_TTEX_TENSOR_LAYOUT_ROW_MAJOR
                                  : layout,
                              selection.bf16Native.tensors[slot])))
        return failure();
      emitter.values[body.getArgument(i)] =
          "read_" + std::to_string(slot++) + "(scratch," + coords + ")";
    }
  }
  auto result = emitter.emit(op);
  if (failed(result))
    return failure();
  return constants.str() + emitter.code.str() + "value=" + *result + ";\n";
}

static FailureOr<BuiltinSelection> inferBF16NativeProgram(func::FuncOp func,
                                                          bool useTensix,
                                                          bool reuseOutputTiles) {
  bool hasBF16 = false;
  func.walk([&](IREE::TensorExt::DispatchTensorLoadOp load) {
    auto type = dyn_cast<RankedTensorType>(load.getType());
    hasBF16 |= type && type.getElementType().isBF16();
  });
  if (!hasBF16)
    return failure();
  SmallVector<IREE::TensorExt::DispatchTensorStoreOp> stores;
  SmallVector<linalg::GenericOp> generics;
  linalg::MatmulOp named;
  func.walk(
      [&](IREE::TensorExt::DispatchTensorStoreOp op) { stores.push_back(op); });
  func.walk([&](linalg::GenericOp op) { generics.push_back(op); });
  func.walk([&](linalg::MatmulOp op) { named = op; });
  if (stores.empty() || stores.size() > 2)
    return failure();
  for (auto store : stores)
    if (!cast<RankedTensorType>(store.getValue().getType())
             .getElementType()
             .isBF16())
      return failure();
  BuiltinSelection selection;
  selection.program = TT_IREE_TTEX_BUILTIN_PROGRAM_BF16_NATIVE;
  selection.dispatchKind = TT_IREE_TTEX_DISPATCH_KIND_BF16_NATIVE;
  auto &d = selection.bf16Native;
  linalg::GenericOp contraction;
  for (auto op : generics)
    if (bf16ContractionBody(op)) {
      if (contraction)
        return failure();
      contraction = op;
    }
  tt_iree_ttex_entry_point_desc_t geometry = {};
  if (contraction || named) {
    if (stores.size() != 1 || generics.size() != (named ? 1u : 2u))
      return failure();
    auto lhs = named ? named.getInputs()[0] : contraction.getDpsInputs()[0];
    auto rhs = named ? named.getInputs()[1] : contraction.getDpsInputs()[1];
    auto result = named ? named.getResult(0) : contraction.getResult(0);
    auto lhsType = dyn_cast<RankedTensorType>(lhs.getType());
    auto rhsType = dyn_cast<RankedTensorType>(rhs.getType());
    auto resultType = dyn_cast<RankedTensorType>(result.getType());
    auto init = named ? named.getOutputs()[0] : contraction.getDpsInits()[0];
    auto fill = init.getDefiningOp<linalg::FillOp>();
    if (!lhsType || !rhsType || !resultType || !lhsType.hasStaticShape() ||
        !rhsType.hasStaticShape() || !resultType.hasStaticShape() ||
        !lhsType.getElementType().isBF16() ||
        !rhsType.getElementType().isBF16() ||
        !resultType.getElementType().isF32() || !fill ||
        getConstantFloat(fill.getInputs()[0]) != std::optional<double>(0.0))
      return failure();
    uint8_t layout;
    if (named) {
      auto maps = named.getIndexingMapsArray();
      auto *ctx = func.getContext();
      auto a = getAffineDimExpr(0, ctx), b = getAffineDimExpr(1, ctx),
           k = getAffineDimExpr(2, ctx);
      if (maps != SmallVector<AffineMap>{AffineMap::get(3, 0, {a, k}, ctx),
                                         AffineMap::get(3, 0, {b, k}, ctx),
                                         AffineMap::get(3, 0, {a, b}, ctx)} ||
          lhsType.getRank() != 2 || rhsType.getRank() != 2)
        return failure();
      layout = TT_IREE_TTEX_TENSOR_LAYOUT_ROW_MAJOR;
      d.operation_kind = geometry.dispatch_kind =
          TT_IREE_TTEX_DISPATCH_KIND_LINEAR;
      d.m = lhsType.getDimSize(0);
      d.k = lhsType.getDimSize(1);
      d.n = rhsType.getDimSize(0);
      geometry.linear.m = d.m;
      geometry.linear.n = d.n;
      geometry.linear.k = d.k;
    } else if (auto stride = bf16ConvolutionStride(contraction)) {
      bool batched = lhsType.getRank() == 4;
      unsigned channel = batched ? 1 : 0;
      if (lhsType.getRank() != (batched ? 4 : 3) || rhsType.getRank() != 4 ||
          resultType.getRank() != lhsType.getRank() ||
          (batched && lhsType.getDimSize(0) != resultType.getDimSize(0)))
        return failure();
      layout = batched ? TT_IREE_TTEX_TENSOR_LAYOUT_NCHW
                       : TT_IREE_TTEX_TENSOR_LAYOUT_CHW;
      d.operation_kind = geometry.dispatch_kind =
          TT_IREE_TTEX_DISPATCH_KIND_CONVOLUTION;
      auto &c = geometry.conv2d;
      c.kernel_h = rhsType.getDimSize(2);
      c.kernel_w = rhsType.getDimSize(3);
      c.stride_h = c.stride_w = *stride;
      c.dilation_h = c.dilation_w = 1;
      if (!((c.kernel_h == 3 && c.kernel_w == 3) ||
            (c.kernel_h == 7 && c.kernel_w == 7 && *stride == 2)))
        return failure();
      if (!bf16MatrixProduct(d.m, {resultType.getDimSize(channel + 1),
                                   resultType.getDimSize(channel + 2),
                                   batched ? resultType.getDimSize(0) : 1}) ||
          !bf16MatrixProduct(
              d.k, {lhsType.getDimSize(channel), c.kernel_h, c.kernel_w}))
        return failure();
      d.n = resultType.getDimSize(channel);
      if (rhsType.getDimSize(1) != lhsType.getDimSize(channel) ||
          lhsType.getDimSize(channel + 1) < c.kernel_h ||
          lhsType.getDimSize(channel + 2) < c.kernel_w ||
          (lhsType.getDimSize(channel + 1) - c.kernel_h) / *stride + 1 !=
              resultType.getDimSize(channel + 1) ||
          (lhsType.getDimSize(channel + 2) - c.kernel_w) / *stride + 1 !=
              resultType.getDimSize(channel + 2))
        return failure();
    } else {
      bool batchedSpatial = bf16BatchedProjection(contraction, true);
      bool batchedFlat = bf16BatchedProjection(contraction, false);
      bool batched = batchedSpatial || batchedFlat;
      bool spatial =
          batchedSpatial || matchesProjectionIndexingMaps(contraction);
      if (!batched && !spatial &&
          !matchesFlattenedProjectionIndexingMaps(contraction))
        return failure();
      if (rhsType.getRank() != 2 ||
          lhsType.getRank() != (spatial ? 3 : 2) + (batched ? 1 : 0) ||
          resultType.getRank() != lhsType.getRank() ||
          (batched && lhsType.getDimSize(0) != resultType.getDimSize(0)))
        return failure();
      layout = batched
                   ? (spatial ? TT_IREE_TTEX_TENSOR_LAYOUT_NCHW
                              : TT_IREE_TTEX_TENSOR_LAYOUT_BATCH_CHANNEL_MAJOR)
               : spatial ? TT_IREE_TTEX_TENSOR_LAYOUT_CHW
                         : TT_IREE_TTEX_TENSOR_LAYOUT_CHANNEL_MAJOR;
      d.operation_kind = geometry.dispatch_kind =
          TT_IREE_TTEX_DISPATCH_KIND_MATMUL_LIKE;
      unsigned channel = batched ? 1 : 0;
      d.k = lhsType.getDimSize(channel);
      d.n = rhsType.getDimSize(0);
      if (!bf16MatrixProduct(d.m,
                             {lhsType.getDimSize(channel + 1),
                              spatial ? lhsType.getDimSize(channel + 2) : 1,
                              batched ? lhsType.getDimSize(0) : 1}))
        return failure();
      for (unsigned axis = channel + 1; axis < lhsType.getRank(); ++axis)
        if (lhsType.getDimSize(axis) != resultType.getDimSize(axis))
          return failure();
      geometry.matmul_like.m = d.m;
      geometry.matmul_like.n = d.n;
      geometry.matmul_like.k = d.k;
    }
    unsigned outputChannel =
        named || layout == TT_IREE_TTEX_TENSOR_LAYOUT_NCHW ||
                layout == TT_IREE_TTEX_TENSOR_LAYOUT_BATCH_CHANNEL_MAJOR
            ? 1
            : 0;
    if (!d.m || !d.n || !d.k || d.m > UINT32_MAX - 31 ||
        d.n > UINT32_MAX - 31 || d.k > UINT32_MAX - 31 ||
        resultType.getDimSize(outputChannel) != d.n ||
        rhsType.getDimSize(1) !=
            lhsType.getDimSize(
                named || layout == TT_IREE_TTEX_TENSOR_LAYOUT_NCHW ||
                        layout == TT_IREE_TTEX_TENSOR_LAYOUT_BATCH_CHANNEL_MAJOR
                    ? 1
                    : 0) ||
        failed(bf16LoadDesc(lhs, layout, d.tensors[0])) ||
        failed(bf16LoadDesc(rhs,
                            named ? TT_IREE_TTEX_TENSOR_LAYOUT_OI
                            : geometry.dispatch_kind ==
                                    TT_IREE_TTEX_DISPATCH_KIND_CONVOLUTION
                                ? TT_IREE_TTEX_TENSOR_LAYOUT_OIHW
                                : TT_IREE_TTEX_TENSOR_LAYOUT_OI,
                            d.tensors[1])) ||
        failed(bf16StoreDesc(stores[0], layout, d.tensors[2])))
      return failure();
    linalg::GenericOp epilogueOp;
    for (auto op : generics)
      if (op != contraction)
        epilogueOp = op;
    if (!epilogueOp || stores[0].getValue() != epilogueOp.getResult(0) ||
        cast<RankedTensorType>(epilogueOp.getResult(0).getType()).getShape() !=
            resultType.getShape())
      return failure();
    unsigned slot = 3;
    auto body = bf16Epilogue(epilogueOp, result, layout, selection, slot);
    if (failed(body))
      return failure();
    geometry.matmul_like.lhs = geometry.linear.input = geometry.conv2d.input =
        d.tensors[0];
    geometry.matmul_like.rhs = geometry.linear.weights =
        geometry.conv2d.filter = d.tensors[1];
    geometry.matmul_like.output = geometry.linear.output =
        geometry.conv2d.output = d.tensors[2];
    const tt_iree_ttex_tensor_desc_t *t[TT_IREE_BF16_NATIVE_TENSOR_COUNT];
    for (unsigned i = 0; i < TT_IREE_BF16_NATIVE_TENSOR_COUNT; ++i)
      t[i] = d.tensors[i].rank ? &d.tensors[i] : nullptr;
#ifdef TT_IREE_ENABLE_AOT
    selection.bf16Sources =
        generateBF16NativeMatrix(geometry, t, *body, useTensix, true,
                                reuseOutputTiles);
    selection.bf16Source = selection.bf16Sources.reader;
    if (useTensix)
      selection.program = TT_IREE_TTEX_BUILTIN_PROGRAM_BF16_NATIVE_TENSIX;
#endif
    selection.mTiles = (d.m + 31) / 32;
    selection.nTiles = (d.n + 31) / 32;
    selection.kTiles = (d.k + 31) / 32;
    return selection;
  }
  // Scalar families remain deliberately limited to the exact ResNet patterns.
  std::ostringstream body;
  uint8_t layout =
      cast<RankedTensorType>(stores[0].getValue().getType()).getRank() == 4
          ? TT_IREE_TTEX_TENSOR_LAYOUT_NCHW
          : TT_IREE_TTEX_TENSOR_LAYOUT_CHW;
  if (generics.empty() && stores.size() == 1) {
    auto load = stores[0]
                    .getValue()
                    .getDefiningOp<IREE::TensorExt::DispatchTensorLoadOp>();
    if (!load ||
        (cast<RankedTensorType>(load.getType()).getRank() != 3 &&
         cast<RankedTensorType>(load.getType()).getRank() != 4) ||
        failed(bf16LoadDesc(load.getResult(), layout, d.tensors[0])) ||
        failed(bf16StoreDesc(stores[0], layout, d.tensors[2])))
      return failure();
    d.operation_kind = TT_IREE_TTEX_DISPATCH_KIND_PADDED_COPY;
    auto &o = d.tensors[2];
    bool batched = o.rank == 4;
    unsigned c = batched ? 1 : 0;
    if (batched)
      body << "uint32_t b=i/" << uint64_t(o.shape[1]) * o.shape[2] * o.shape[3]
           << "u;\n";
    body << "uint32_t c=(i/" << uint64_t(o.shape[c + 1]) * o.shape[c + 2]
         << "u)%" << o.shape[c] << "u,h=(i/" << o.shape[c + 2] << "u)%"
         << o.shape[c + 1] << "u,w=i%" << o.shape[c + 2] << "u;\n"
         << "write_2(scratch,read_0(scratch," << (batched ? "b," : "")
         << "c,h,w)," << (batched ? "b," : "") << "c,h,w);\n";
  } else if (generics.size() == 1 && stores.size() == 1 &&
             (generics[0].getNumLoops() == 5 ||
              generics[0].getNumLoops() == 6)) {
    auto op = generics[0];
    if (op.getNumDpsInputs() != 2 || op.getNumDpsInits() != 1 ||
        !op.getDpsInputs()[1].getDefiningOp<tensor::EmptyOp>() ||
        stores[0].getValue() != op.getResult(0) ||
        failed(bf16LoadDesc(op.getDpsInputs()[0], layout, d.tensors[0])) ||
        failed(bf16StoreDesc(stores[0], layout, d.tensors[2])))
      return failure();
    auto &o = d.tensors[2];
    auto &in = d.tensors[0];
    auto *ctx = func.getContext();
    bool batched = o.rank == 4;
    unsigned channel = batched ? 1 : 0;
    unsigned loops = batched ? 6 : 5;
    auto batch = getAffineDimExpr(0, ctx);
    auto a = getAffineDimExpr(channel, ctx),
         b = getAffineDimExpr(channel + 1, ctx),
         c = getAffineDimExpr(channel + 2, ctx),
         h = getAffineDimExpr(channel + 3, ctx),
         w = getAffineDimExpr(channel + 4, ctx);
    SmallVector<AffineExpr> inputMap = {a, b * 2 + h, c * 2 + w};
    SmallVector<AffineExpr> outputMap = {a, b, c};
    if (batched) {
      inputMap.insert(inputMap.begin(), batch);
      outputMap.insert(outputMap.begin(), batch);
    }
    SmallVector<utils::IteratorType> iterators(loops,
                                               utils::IteratorType::parallel);
    iterators[loops - 1] = iterators[loops - 2] =
        utils::IteratorType::reduction;
    if (op.getIndexingMapsArray() !=
            SmallVector<AffineMap>{AffineMap::get(loops, 0, inputMap, ctx),
                                   AffineMap::get(loops, 0, {h, w}, ctx),
                                   AffineMap::get(loops, 0, outputMap, ctx)} ||
        op.getIteratorTypesArray() != iterators ||
        o.rank != (batched ? 4 : 3) || in.rank != o.rank ||
        (batched && in.shape[0] != o.shape[0]) ||
        in.shape[channel] != o.shape[channel] ||
        (in.shape[channel + 1] - 3) / 2 + 1 != o.shape[channel + 1] ||
        (in.shape[channel + 2] - 3) / 2 + 1 != o.shape[channel + 2])
      return failure();
    auto &block = op.getRegion().front();
    auto yield = dyn_cast<linalg::YieldOp>(block.getTerminator());
    auto max = yield.getValues()[0].getDefiningOp<arith::MaximumFOp>();
    auto fill = op.getDpsInits()[0].getDefiningOp<linalg::FillOp>();
    auto initial = fill ? getConstantFloat(fill.getInputs()[0]) : std::nullopt;
    if (!max ||
        !((max.getLhs() == block.getArgument(0) &&
           max.getRhs() == block.getArgument(2)) ||
          (max.getRhs() == block.getArgument(0) &&
           max.getLhs() == block.getArgument(2))) ||
        !initial || !std::isinf(*initial) || *initial >= 0.0 ||
        cast<RankedTensorType>(op.getDpsInputs()[1].getType()).getShape() !=
            ArrayRef<int64_t>({3, 3}) ||
        !block.getArgument(1).use_empty())
      return failure();
    d.operation_kind = TT_IREE_TTEX_DISPATCH_KIND_MAX_POOL;
    if (batched)
      body << "uint32_t b=i/" << uint64_t(o.shape[1]) * o.shape[2] * o.shape[3]
           << "u;\n";
    body << "uint32_t c=(i/"
         << uint64_t(o.shape[channel + 1]) * o.shape[channel + 2] << "u)%"
         << o.shape[channel] << "u,h=(i/" << o.shape[channel + 2] << "u)%"
         << o.shape[channel + 1] << "u,w=i%" << o.shape[channel + 2] << "u;\n"
         << "float value=from_bits(0xff800000u);\nfor(uint32_t kh=0;kh<3;++kh) "
            "for(uint32_t kw=0;kw<3;++kw) {\n"
         << "float v=read_0(scratch," << (batched ? "b," : "")
         << "c,h*2+kh,w*2+kw);\n"
         << "if((to_bits(v)&0x7fffffffu)>0x7f800000u || "
            "(to_bits(value)&0x7fffffffu)>0x7f800000u) "
            "value=from_bits(0x7fc00000u); else if(v>value) value=v; else "
            "if(v==0.0f && value==0.0f) "
            "value=from_bits(to_bits(v)&to_bits(value));\n}\nwrite_2(scratch,"
         << "value," << (batched ? "b," : "") << "c,h,w);\n";
  } else if (generics.size() == 1 && stores.size() == 2) {
    auto op = generics[0];
    if (op.getNumDpsInputs() != 1 || op.getNumDpsInits() != 2 ||
        op.getNumLoops() != 2 ||
        cast<RankedTensorType>(op.getDpsInputs()[0].getType()).getShape() !=
            ArrayRef<int64_t>({49}))
      return failure();
    tensor::ExtractOp extract;
    unsigned extractCount = 0;
    func.walk([&](tensor::ExtractOp v) {
      extract = v;
      ++extractCount;
    });
    if (extract &&
        cast<RankedTensorType>(extract.getTensor().getType()).getRank() == 4)
      layout = TT_IREE_TTEX_TENSOR_LAYOUT_NCHW;
    if (extractCount != 1 || !extract ||
        failed(bf16LoadDesc(extract.getTensor(), layout, d.tensors[0])))
      return failure();
    IREE::TensorExt::DispatchTensorStoreOp sum, divisor;
    for (auto store : stores) {
      if (store.getValue() == op.getResult(0))
        sum = store;
      if (store.getValue() == op.getResult(1))
        divisor = store;
    }
    bool batched = d.tensors[0].rank == 4;
    unsigned channel = batched ? 1 : 0;
    if (!sum || !divisor || d.tensors[0].rank != (batched ? 4 : 3) ||
        d.tensors[0].shape[channel + 1] != 8 ||
        d.tensors[0].shape[channel + 2] != 8 ||
        failed(bf16StoreDesc(sum, TT_IREE_TTEX_TENSOR_LAYOUT_ROW_MAJOR,
                             d.tensors[2])) ||
        failed(bf16StoreDesc(divisor, TT_IREE_TTEX_TENSOR_LAYOUT_ROW_MAJOR,
                             d.tensors[1])))
      return failure();
    auto yield = cast<linalg::YieldOp>(op.getRegion().front().getTerminator());
    auto *ctx = func.getContext();
    auto outputChannel = getAffineDimExpr(0, ctx),
         position = getAffineDimExpr(1, ctx);
    if (!bf16GlobalSumBody(op, extract) ||
        op.getIndexingMapsArray() !=
            SmallVector<AffineMap>{AffineMap::get(2, 0, {position}, ctx),
                                   AffineMap::get(2, 0, {outputChannel}, ctx),
                                   AffineMap::get(2, 0, {}, ctx)} ||
        op.getIteratorTypesArray() !=
            SmallVector<utils::IteratorType>{utils::IteratorType::parallel,
                                             utils::IteratorType::reduction} ||
        d.tensors[2].rank != 1 ||
        d.tensors[2].shape[0] != uint64_t(d.tensors[0].shape[channel]) *
                                     (batched ? d.tensors[0].shape[0] : 1) ||
        d.tensors[1].shape[0] != 1)
      return failure();
    d.operation_kind = TT_IREE_TTEX_DISPATCH_KIND_GLOBAL_SUM;
    body << "float value=0.0f;\nfor(uint32_t h=0;h<7;++h) for(uint32_t "
            "w=0;w<7;++w) value=round_bf16(value+read_0(scratch,";
    if (batched)
      body << "i/" << d.tensors[0].shape[1] << "u,i%" << d.tensors[0].shape[1]
           << "u";
    else
      body << "i";
    body << ",h,w));\nwrite_2(scratch,value,i);\n";
  } else if (generics.size() == 1 && stores.size() == 1 &&
             generics[0].getNumLoops() == 1) {
    auto op = generics[0];
    if (op.getNumDpsInputs() != 2 || op.getNumDpsInits() != 1 ||
        stores[0].getValue() != op.getResult(0) ||
        failed(bf16LoadDesc(op.getDpsInputs()[1],
                            TT_IREE_TTEX_TENSOR_LAYOUT_ROW_MAJOR,
                            d.tensors[0])) ||
        failed(bf16LoadDesc(op.getDpsInputs()[0],
                            TT_IREE_TTEX_TENSOR_LAYOUT_ROW_MAJOR,
                            d.tensors[1])) ||
        failed(bf16StoreDesc(stores[0], TT_IREE_TTEX_TENSOR_LAYOUT_ROW_MAJOR,
                             d.tensors[2])))
      return failure();
    auto &block = op.getRegion().front();
    auto *ctx = func.getContext();
    auto index = getAffineDimExpr(0, ctx);
    if (op.getIndexingMapsArray() !=
            SmallVector<AffineMap>{AffineMap::get(1, 0, {}, ctx),
                                   AffineMap::get(1, 0, {index}, ctx),
                                   AffineMap::get(1, 0, {index}, ctx)} ||
        op.getIteratorTypesArray() !=
            SmallVector<utils::IteratorType>{utils::IteratorType::parallel} ||
        d.tensors[0].shape[0] != d.tensors[2].shape[0])
      return failure();
    BF16ScalarEmitter emit;
    emit.values[block.getArgument(0)] = "read_1(scratch,0)";
    emit.values[block.getArgument(1)] = "read_0(scratch,i)";
    auto result = emit.emit(op);
    if (failed(result) || d.tensors[0].rank != 1 ||
        d.tensors[1].shape[0] != 1 || d.tensors[2].rank != 1)
      return failure();
    d.operation_kind = TT_IREE_TTEX_DISPATCH_KIND_SCALAR_DIV;
    body << emit.code.str() << "write_2(scratch," << *result << ",i);\n";
  } else
    return failure();
  const tt_iree_ttex_tensor_desc_t *t[TT_IREE_BF16_NATIVE_TENSOR_COUNT];
  for (unsigned i = 0; i < TT_IREE_BF16_NATIVE_TENSOR_COUNT; ++i)
    t[i] = d.tensors[i].rank ? &d.tensors[i] : nullptr;
#ifdef TT_IREE_ENABLE_AOT
  selection.bf16Source = generateBF16NativeSpatial(d.operation_kind, t);
  if (selection.bf16Source.empty())
    selection.bf16Source =
        generateBF16NativeCommon(t) +
        "void kernel_main() {\nCircularBuffer cb(tt::CBIndex::c_2); "
        "cb.reserve_back(1); uint32_t scratch=cb.get_write_ptr();\n"
        "uint32_t "
        "start=get_arg_val<uint32_t>(28),count=get_arg_val<uint32_t>(29);\nfor("
        "uint32_t i=start;i<start+count;++i) {\n" +
        body.str() + "}\n" +
        (d.operation_kind == TT_IREE_TTEX_DISPATCH_KIND_GLOBAL_SUM
             ? "if(get_arg_val<uint32_t>(30)==0) write_1(scratch,49.0f,0);\n"
             : "") +
        "}\n";
#endif
  return selection;
}
