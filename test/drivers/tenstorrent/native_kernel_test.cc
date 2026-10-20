// Copyright 2026 The tt-iree Authors
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "TenstorrentNativeKernels.h"
#include "iree/schema/tt_executable_builder_util.h"
#include "iree/testing/gtest.h"

namespace {
using mlir::iree_compiler::IREE::HAL::generateNativeSources;
using mlir::iree_compiler::IREE::HAL::generateBF16NativeMatrix;

tt_iree_ttex_tensor_desc_t Tensor(uint8_t layout, uint8_t rank, uint32_t a,
                                  uint32_t b, uint32_t c = 0, uint32_t d = 0) {
  tt_iree_ttex_tensor_desc_t result = {};
  result.element_type = TT_IREE_TTEX_TENSOR_ELEMENT_TYPE_F32;
  result.layout = layout;
  result.rank = rank;
  uint32_t dims[] = {a, b, c, d};
  for (unsigned i = 0; i < rank; ++i)
    result.shape[i] = result.storage_shape[i] = dims[i];
  return result;
}

tt_iree_ttex_entry_point_desc_t Projection() {
  tt_iree_ttex_entry_point_desc_t ep = {};
  ep.dispatch_kind = TT_IREE_TTEX_DISPATCH_KIND_MATMUL_LIKE;
  auto &m = ep.matmul_like;
  m.m = 49;
  m.n = 33;
  m.k = 35;
  m.lhs = Tensor(TT_IREE_TTEX_TENSOR_LAYOUT_CHANNEL_MAJOR, 2, 35, 49);
  m.rhs = Tensor(TT_IREE_TTEX_TENSOR_LAYOUT_OI, 2, 33, 35);
  m.output = Tensor(TT_IREE_TTEX_TENSOR_LAYOUT_CHANNEL_MAJOR, 2, 33, 49);
  return ep;
}

tt_iree_ttex_entry_point_desc_t Convolution() {
  tt_iree_ttex_entry_point_desc_t ep = {};
  ep.dispatch_kind = TT_IREE_TTEX_DISPATCH_KIND_CONVOLUTION;
  auto &c = ep.conv2d;
  c.input = Tensor(TT_IREE_TTEX_TENSOR_LAYOUT_CHW, 3, 64, 10, 10);
  c.filter = Tensor(TT_IREE_TTEX_TENSOR_LAYOUT_OIHW, 4, 64, 64, 3, 3);
  c.output = Tensor(TT_IREE_TTEX_TENSOR_LAYOUT_CHW, 3, 64, 8, 8);
  c.kernel_h = c.kernel_w = 3;
  c.stride_h = c.stride_w = c.dilation_h = c.dilation_w = 1;
  return ep;
}

tt_iree_ttex_entry_point_desc_t RuntimeAffineProjection() {
  auto ep = Projection();
  auto &m = ep.matmul_like;
  m.epilogue_flags =
      TT_IREE_TTEX_EPILOGUE_AFFINE | TT_IREE_TTEX_EPILOGUE_PER_CHANNEL |
      TT_IREE_TTEX_EPILOGUE_RUNTIME_AFFINE | TT_IREE_TTEX_EPILOGUE_RELU;
  m.epilogue_common_scale = 0.5f;
  m.epilogue_mean = m.epilogue_channel_scale = m.epilogue_channel_bias =
      Tensor(TT_IREE_TTEX_TENSOR_LAYOUT_ROW_MAJOR, 1, m.n, 0);
  return ep;
}

void AddPerChannelResidual(tt_iree_ttex_entry_point_desc_t &ep) {
  auto &m = ep.matmul_like;
  m.epilogue_flags |= TT_IREE_TTEX_EPILOGUE_RESIDUAL |
                      TT_IREE_TTEX_EPILOGUE_RESIDUAL_PER_CHANNEL;
  m.residual = m.output;
  m.residual_mean = m.residual_channel_scale = m.residual_channel_bias =
      Tensor(TT_IREE_TTEX_TENSOR_LAYOUT_ROW_MAJOR, 1, m.n, 0);
  m.residual_common_scale = 0.25f;
}

bool Has(const std::string &source, const char *text) {
  return source.find(text) != std::string::npos;
}

size_t Count(const std::string &source, const std::string &text) {
  size_t count = 0;
  for (size_t pos = 0; (pos = source.find(text, pos)) != std::string::npos;
       pos += text.size())
    ++count;
  return count;
}

TEST(NativeKernelTest, ChannelMajorUsesBulkInputsAndOutput) {
  auto sources = generateNativeSources(Projection());
  EXPECT_TRUE(Has(sources.reader, "byte_0(k,m,0,0),chunk,128u)"));
  EXPECT_TRUE(Has(sources.reader, "byte_1(n,k,0,0),valid_k,128u)"));
  EXPECT_TRUE(Has(sources.reader, "byte_2(n,m,0,0),chunk,128u)"));
  EXPECT_TRUE(sources.writer.empty());
  EXPECT_TRUE(sources.compute.empty());
}

TEST(NativeKernelTest, Bf16StorageUsesTwoByteTransfersAndFp32Scratch) {
  auto ep = Projection();
  auto a = ep.matmul_like.lhs, b = ep.matmul_like.rhs, o = ep.matmul_like.output;
  a.element_type = b.element_type = o.element_type = TT_IREE_TTEX_TENSOR_ELEMENT_TYPE_BF16;
  const tt_iree_ttex_tensor_desc_t *t[TT_IREE_BF16_NATIVE_TENSOR_COUNT] = {&a, &b, &o};
  auto source = generateBF16NativeMatrix(ep, t, "value=round_bf16(value);\n").reader;
  EXPECT_TRUE(Has(source, "size=span_bytes(scratch,byte,count,2u)"));
  EXPECT_TRUE(Has(source, "uint16_t*>(ptr+i*2u)"));
  EXPECT_TRUE(Has(source, "value += a[row*32+inner] * b[inner*32+col]"));
  EXPECT_TRUE(Has(source, "0x7fffu + ((bits >> 16) & 1u)"));
  EXPECT_TRUE(Has(source, "(bits >> 16) | 0x40u"));
}

TEST(NativeKernelTest, Bf16ExtraOperandsDoNotAliasPartitionArguments) {
  auto ep = Projection();
  auto &m = ep.matmul_like;
  auto extra = Tensor(TT_IREE_TTEX_TENSOR_LAYOUT_ROW_MAJOR, 1, 33, 0);
  const tt_iree_ttex_tensor_desc_t *t[TT_IREE_BF16_NATIVE_TENSOR_COUNT] = {&m.lhs, &m.rhs, &m.output};
  t[10] = &extra;
  auto source = generateBF16NativeMatrix(ep, t, "value+=read_10(scratch,n);\n").reader;
  EXPECT_TRUE(Has(source, "m_start=get_arg_val<uint32_t>(28)"));
  EXPECT_TRUE(Has(source, "m_count=get_arg_val<uint32_t>(30)"));
  EXPECT_TRUE(Has(source, "accessor_10, get_arg_val<uint32_t>(20)"));
  EXPECT_TRUE(Has(source, "get_arg_val<uint32_t>(21) +"));
  EXPECT_TRUE(Has(source, "value+=read_10(scratch,n)"));
}

TEST(NativeKernelTest, Bf16TensixUsesThreeRolesAndKeepsKInDst) {
  auto ep = Projection();
  auto a = ep.matmul_like.lhs, b = ep.matmul_like.rhs, o = ep.matmul_like.output;
  a.element_type = b.element_type = o.element_type = TT_IREE_TTEX_TENSOR_ELEMENT_TYPE_BF16;
  const tt_iree_ttex_tensor_desc_t *t[TT_IREE_BF16_NATIVE_TENSOR_COUNT] = {&a, &b, &o};
  auto sources = generateBF16NativeMatrix(ep, t, "value=round_bf16(value);\n", true);
  EXPECT_TRUE(Has(sources.reader, "c_8"));
  EXPECT_TRUE(Has(sources.reader, "ta[face_index(row,col)]=bf16_bits(a[row*32+col])"));
  EXPECT_TRUE(Has(sources.reader, "m_start=get_arg_val<uint32_t>(28)"));
  EXPECT_FALSE(Has(sources.reader, "value += a[row*32+inner]"));
  EXPECT_TRUE(Has(sources.compute, "matmul_tiles(tt::CBIndex::c_0,tt::CBIndex::c_1,0,0,0)"));
  EXPECT_TRUE(Has(sources.compute, "tile_regs_acquire();\n    for(uint32_t block"));
  EXPECT_EQ(Count(sources.compute, "pack_tile("), 1u);
  EXPECT_FALSE(Has(sources.compute, "copy_tile("));
  EXPECT_TRUE(Has(sources.writer, "scratch_cb(tt::CBIndex::c_3)"));
  EXPECT_TRUE(Has(sources.writer, "packed[face_index(row,col)]"));
  EXPECT_TRUE(Has(sources.writer, "value=round_bf16(value)"));
  EXPECT_TRUE(Has(sources.writer, "byte_2(n,m,0,0),chunk,128u)"));
}

TEST(NativeKernelTest, PageAndScratchLimitsAreEmitted) {
  auto source = generateNativeSources(Projection()).reader;
  EXPECT_TRUE(Has(source, "((scratch + 63u) & ~63u) + byte % 64u"));
  EXPECT_TRUE(Has(source, "scratch + 128u - span_scratch(scratch, byte)"));
  EXPECT_TRUE(Has(source, "2048u - byte % 2048u"));
  EXPECT_TRUE(Has(source, "noc_async_read_barrier()"));
  EXPECT_TRUE(Has(source, "noc_async_write_barrier()"));
}

TEST(NativeKernelTest, ArithmeticAndMaskedTailsArePreserved) {
  auto source = generateNativeSources(Projection()).reader;
  EXPECT_TRUE(Has(source, "no-fast-math"));
  EXPECT_TRUE(Has(source, "fp-contract=off"));
  EXPECT_TRUE(Has(source, "uint32_t valid_k=35u-block*32"));
  EXPECT_TRUE(Has(source, "a[i]=0.0f; b[i]=0.0f"));
  EXPECT_TRUE(Has(source, "value += a[row*32+inner] * b[inner*32+col]"));
}

TEST(NativeKernelTest, StridedLhsRetainsScalarGather) {
  auto ep = Projection();
  ep.matmul_like.lhs.stride[1] = 2;
  auto source = generateNativeSources(ep).reader;
  EXPECT_FALSE(Has(source, "read_span_0(scratch,"));
  EXPECT_TRUE(Has(source, "a[row*32+col]=read_0(scratch,k,m)"));
  EXPECT_TRUE(Has(source, "read_span_1(scratch,"));
}

TEST(NativeKernelTest, StridedRhsRetainsScalarGather) {
  auto ep = Projection();
  ep.matmul_like.rhs.stride[1] = 2;
  auto source = generateNativeSources(ep).reader;
  EXPECT_FALSE(Has(source, "read_span_1(scratch,"));
  EXPECT_TRUE(Has(source, "b[row*32+col]=read_1(scratch,n,k)"));
}

TEST(NativeKernelTest, StridedOutputRetainsScalarStores) {
  auto ep = Projection();
  ep.matmul_like.output.stride[1] = 2;
  auto source = generateNativeSources(ep).reader;
  EXPECT_FALSE(Has(source, "write_span_2(scratch,"));
  EXPECT_TRUE(Has(source, "write_2(scratch,acc[row*32+col],n,m)"));
}

TEST(NativeKernelTest, ChwViewsSplitAtLogicalRows) {
  auto ep = Projection();
  auto &m = ep.matmul_like;
  m.m = 64;
  m.lhs = Tensor(TT_IREE_TTEX_TENSOR_LAYOUT_CHW, 3, 35, 8, 8);
  m.output = Tensor(TT_IREE_TTEX_TENSOR_LAYOUT_CHW, 3, 33, 8, 8);
  m.lhs.storage_shape[2] = m.output.storage_shape[2] = 10;
  m.lhs.offset[2] = m.output.offset[2] = 1;
  auto source = generateNativeSources(ep).reader;
  EXPECT_TRUE(Has(source, "row_remaining=8u-m%8u"));
  EXPECT_TRUE(Has(source, "byte_0(k,m/8u,m%8u,0),chunk,128u)"));
  EXPECT_TRUE(Has(source, "byte_2(n,m/8u,m%8u,0),chunk,128u)"));
}

TEST(NativeKernelTest, ContiguousConvUsesSpatialAndFilterSpans) {
  auto source = generateNativeSources(Convolution()).reader;
  EXPECT_TRUE(Has(source, "read_span_0(scratch,"));
  EXPECT_TRUE(Has(source, "row_remaining=8u-m%8u"));
  EXPECT_TRUE(Has(source, "byte_1(n,ic,kh,kw),valid_k,128u)"));
}

TEST(NativeKernelTest, StrideTwoConvRetainsLhsGather) {
  auto ep = Convolution();
  ep.conv2d.stride_h = ep.conv2d.stride_w = 2;
  auto source = generateNativeSources(ep).reader;
  EXPECT_FALSE(Has(source, "read_span_0(scratch,"));
  EXPECT_TRUE(Has(source, "read_0(scratch,ic,"));
  EXPECT_TRUE(Has(source, "read_span_1(scratch,"));
}

TEST(NativeKernelTest, PaddedFilterRetainsScalarGather) {
  auto ep = Convolution();
  ep.conv2d.filter.storage_shape[3] = 4;
  auto source = generateNativeSources(ep).reader;
  EXPECT_FALSE(Has(source, "read_span_1(scratch,"));
  EXPECT_TRUE(Has(source, "b[row*32+col]=read_1(scratch,n,ic,kh,kw)"));
}

TEST(NativeKernelTest, LinearUsesRowMajorSpans) {
  tt_iree_ttex_entry_point_desc_t ep = {};
  ep.dispatch_kind = TT_IREE_TTEX_DISPATCH_KIND_LINEAR;
  auto &l = ep.linear;
  l.m = 1;
  l.n = 17;
  l.k = 33;
  l.input = Tensor(TT_IREE_TTEX_TENSOR_LAYOUT_ROW_MAJOR, 2, 1, 33);
  l.weights = Tensor(TT_IREE_TTEX_TENSOR_LAYOUT_OI, 2, 17, 33);
  l.bias = l.output = Tensor(TT_IREE_TTEX_TENSOR_LAYOUT_ROW_MAJOR, 2, 1, 17);
  auto source = generateNativeSources(ep).reader;
  EXPECT_TRUE(Has(source, "byte_0(m,k,0,0),valid_k,4u)"));
  EXPECT_TRUE(Has(source, "byte_2(m,n_base,0,0),valid_n,4u)"));
  EXPECT_TRUE(Has(source, "value += read_5(scratch, m, n)"));
}

TEST(NativeKernelTest, ExperimentalTensixCodeIsUnchanged) {
  auto sources = generateNativeSources(Projection(), true);
  EXPECT_FALSE(Has(sources.reader, "read_span_0"));
  EXPECT_FALSE(sources.writer.empty());
  EXPECT_FALSE(sources.compute.empty());
}

TEST(NativeKernelTest, RuntimeAffineSetupPrecedesTheSpatialLoop) {
  auto source = generateNativeSources(RuntimeAffineProjection()).reader;
  size_t setup = source.find("float channel_scale =");
  size_t spatial_loop =
      source.find("for(uint32_t row=0;row<valid_m;++row) {", setup);
  size_t apply = source.find("value = value * channel_scale + channel_bias;");
  ASSERT_NE(setup, std::string::npos);
  ASSERT_NE(spatial_loop, std::string::npos);
  ASSERT_NE(apply, std::string::npos);
  EXPECT_LT(setup, spatial_loop);
  EXPECT_LT(spatial_loop, apply);
  EXPECT_EQ(Count(source, "read_3(scratch,n)"), 1u);
  EXPECT_EQ(Count(source, "read_4(scratch,n)"), 1u);
  EXPECT_EQ(Count(source, "read_5(scratch,n)"), 1u);
  EXPECT_TRUE(
      Has(source, "read_5(scratch,n) - read_3(scratch,n) * channel_scale"));
  EXPECT_TRUE(Has(source, "if (value <= 0.0f) value = 0.0f;"));
}

TEST(NativeKernelTest, RuntimeAffineRetainsTailMaskAndVectorAddressing) {
  auto ep = RuntimeAffineProjection();
  ep.matmul_like.epilogue_mean.storage_shape[0] = 68;
  ep.matmul_like.epilogue_mean.offset[0] = 1;
  ep.matmul_like.epilogue_mean.stride[0] = 2;
  auto source = generateNativeSources(ep).reader;
  EXPECT_TRUE(Has(source, "col<valid_n"));
  EXPECT_TRUE(Has(source, "uint32_t valid_n=33u-n_base"));
  EXPECT_TRUE(Has(source, "uint32_t valid_m=49u-m_base"));
  EXPECT_TRUE(Has(source, "68u + 1u + x0 * 2u"));
  EXPECT_FALSE(Has(source, "float channel_scale["));
}

TEST(NativeKernelTest, RuntimeConvolutionCachesAffineCoefficients) {
  auto ep = Convolution();
  auto &c = ep.conv2d;
  c.epilogue_flags = TT_IREE_TTEX_EPILOGUE_AFFINE |
                     TT_IREE_TTEX_EPILOGUE_PER_CHANNEL |
                     TT_IREE_TTEX_EPILOGUE_RUNTIME_AFFINE;
  c.epilogue_common_scale = 0.5f;
  c.epilogue_mean = c.epilogue_channel_scale = c.epilogue_channel_bias =
      Tensor(TT_IREE_TTEX_TENSOR_LAYOUT_ROW_MAJOR, 1, 64, 0);
  auto source = generateNativeSources(ep).reader;
  EXPECT_TRUE(Has(source, "float channel_scale ="));
  EXPECT_TRUE(Has(source, "value = value * channel_scale + channel_bias;"));
  EXPECT_EQ(Count(source, "read_4(scratch,n)"), 1u);
  EXPECT_FALSE(Has(source, "channel_residual_scale"));
}

TEST(NativeKernelTest, ResidualOnlyCachesCoefficientsNotTensorValues) {
  auto ep = Projection();
  AddPerChannelResidual(ep);
  auto source = generateNativeSources(ep).reader;
  size_t setup = source.find("float channel_residual_scale =");
  size_t spatial_loop =
      source.find("for(uint32_t row=0;row<valid_m;++row) {", setup);
  size_t residual_read = source.find("float residual = read_6(scratch,n,m);");
  ASSERT_NE(setup, std::string::npos);
  ASSERT_NE(spatial_loop, std::string::npos);
  ASSERT_NE(residual_read, std::string::npos);
  EXPECT_LT(setup, spatial_loop);
  EXPECT_LT(spatial_loop, residual_read);
  EXPECT_EQ(Count(source, "read_7(scratch,n)"), 1u);
  EXPECT_EQ(Count(source, "read_8(scratch,n)"), 1u);
  EXPECT_EQ(Count(source, "read_9(scratch,n)"), 1u);
  EXPECT_FALSE(Has(source, "float channel_scale ="));
}

TEST(NativeKernelTest, CombinedAffineAndResidualKeepSeparateCoefficients) {
  auto ep = RuntimeAffineProjection();
  AddPerChannelResidual(ep);
  auto source = generateNativeSources(ep).reader;
  EXPECT_TRUE(Has(source, "value = value * channel_scale + channel_bias;"));
  EXPECT_TRUE(Has(
      source,
      "residual = residual * channel_residual_scale + channel_residual_bias;"));
  size_t affine = source.find("value = value * channel_scale");
  size_t residual = source.find("float residual = read_6(scratch,");
  size_t add = source.find("value += residual;");
  size_t relu = source.find("if (value <= 0.0f)");
  EXPECT_LT(affine, residual);
  EXPECT_LT(residual, add);
  EXPECT_LT(add, relu);
}

TEST(NativeKernelTest, ScalarResidualDoesNotEnableChannelReuse) {
  auto ep = Projection();
  ep.matmul_like.epilogue_flags = TT_IREE_TTEX_EPILOGUE_RESIDUAL;
  ep.matmul_like.residual = ep.matmul_like.output;
  ep.matmul_like.residual_scale = 0.5f;
  auto source = generateNativeSources(ep).reader;
  EXPECT_FALSE(Has(source, "channel_residual_scale"));
  EXPECT_TRUE(Has(source, "float residual = read_6(scratch,n,m);"));
}

TEST(NativeKernelTest, ExperimentalTensixRuntimeAffineIsNotHoisted) {
  auto ep = RuntimeAffineProjection();
  AddPerChannelResidual(ep);
  auto sources = generateNativeSources(ep, true);
  EXPECT_FALSE(Has(sources.writer, "channel_scale"));
  EXPECT_FALSE(Has(sources.writer, "channel_residual_scale"));
  EXPECT_TRUE(Has(sources.writer, "float scale ="));
  EXPECT_TRUE(Has(sources.writer, "float residual_scale ="));
}
} // namespace
