// Copyright 2026 The tt-iree Authors
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "TenstorrentNativeKernels.h"
#include "iree/hal/drivers/tenstorrent/tt_native_partition.h"
#include "iree/schema/tt_executable_builder_util.h"
#include "iree/testing/gtest.h"

#include <algorithm>
#include <array>

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
  EXPECT_TRUE(
      Has(sources.reader, "prefetch_bf16_0(lhs.get_write_ptr()+col*128u"));
  EXPECT_TRUE(
      Has(sources.reader, "ta[face_index(first+j,col)]=transport_bf16"));
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

TEST(NativeKernelTest, Bf16PrefetchRetainsArithmeticAndExistingScratchABI) {
  auto ep = Projection();
  auto a = ep.matmul_like.lhs, b = ep.matmul_like.rhs,
       o = ep.matmul_like.output;
  a.element_type = b.element_type = o.element_type =
      TT_IREE_TTEX_TENSOR_ELEMENT_TYPE_BF16;
  const tt_iree_ttex_tensor_desc_t *t[TT_IREE_BF16_NATIVE_TENSOR_COUNT] = {
      &a, &b, &o};
  auto legacy = generateBF16NativeMatrix(ep, t, "value=round_bf16(value);\n",
                                         true, false);
  auto next =
      generateBF16NativeMatrix(ep, t, "value=round_bf16(value);\n", true);
  EXPECT_EQ(next.compute, legacy.compute);
  EXPECT_EQ(next.writer, legacy.writer);
  EXPECT_TRUE(Has(next.reader, "window+((byte-window)&63u)"));
  EXPECT_TRUE(Has(next.reader, "2048u-byte%2048u"));
  EXPECT_TRUE(Has(next.reader, "ptrs[32]"));
  EXPECT_TRUE(Has(next.reader, "ta[i]=0; tb[i]=0"));
  EXPECT_TRUE(Has(next.reader, "(v & 0x7fffu) > 0x7f80u"));
  EXPECT_FALSE(Has(next.reader, "bf16_bits(a[row*32+col])"));
  EXPECT_TRUE(Has(legacy.reader, "bf16_bits(a[row*32+col])"));
  // A window is at most 64B and fits its 128B slot for every base alignment.
  for (uint32_t low = 0; low < 64; ++low)
    for (uint32_t byte = 0; byte < 2048; byte += 2) {
      uint32_t window = 4096 + 31 * 128 + low,
               ptr = window + ((byte - window) & 63u);
      EXPECT_EQ(ptr % 64, byte % 64);
      EXPECT_GE(ptr, window);
      EXPECT_LE(ptr + 64, window + 128);
    }
}

TEST(NativeKernelTest, Bf16OutputPairReusesOperandsWithExistingCBsAndWriter) {
  auto ep = Projection();
  auto a = ep.matmul_like.lhs, b = ep.matmul_like.rhs,
       o = ep.matmul_like.output;
  a.element_type = b.element_type = o.element_type =
      TT_IREE_TTEX_TENSOR_ELEMENT_TYPE_BF16;
  const tt_iree_ttex_tensor_desc_t *t[TT_IREE_BF16_NATIVE_TENSOR_COUNT] = {
      &a, &b, &o};
  auto baseline = generateBF16NativeMatrix(ep, t, "value=round_bf16(value);\n",
                                          true, true, false);
  auto reuse = generateBF16NativeMatrix(ep, t, "value=round_bf16(value);\n",
                                       true, true, true);
  EXPECT_EQ(reuse.writer, baseline.writer);
  EXPECT_NE(reuse.reader, baseline.reader);
  EXPECT_NE(reuse.compute, baseline.compute);
  EXPECT_TRUE(Has(reuse.reader, "group=tiles-first; if(group>2) group=2"));
  EXPECT_TRUE(Has(reuse.reader, "tm=(first+lane)/n_count,tn=(first+lane)%n_count"));
  EXPECT_TRUE(Has(reuse.reader, "bool reuse_a=lane && tm==first/n_count"));
  EXPECT_TRUE(Has(reuse.reader, "bool reuse_b=lane && tn==first%n_count"));
  EXPECT_TRUE(Has(reuse.reader, "copy_operand_tile(saved_tile_a,tile_a_base)"));
  EXPECT_TRUE(Has(reuse.reader, "copy_operand_tile(saved_tile_b,tile_b_base)"));
  EXPECT_TRUE(Has(reuse.reader, "tile_a.reserve_back(1)"));
  EXPECT_TRUE(Has(reuse.reader, "tile_a.push_back(1)"));
  EXPECT_FALSE(Has(reuse.reader, "tile_a.reserve_back(group)"));
  EXPECT_FALSE(Has(reuse.reader, "tile_a.push_back(group)"));
  EXPECT_TRUE(Has(reuse.reader, "first+=group"));
  EXPECT_TRUE(Has(reuse.compute, "matmul_tiles(tt::CBIndex::c_0,tt::CBIndex::c_1,0,0,lane)"));
  EXPECT_TRUE(Has(reuse.compute, "lhs.pop_front(1); rhs.pop_front(1)"));
  EXPECT_FALSE(Has(reuse.compute, "lhs.wait_front(group)"));
  EXPECT_TRUE(Has(reuse.compute, "pack_tile(lane,tt::CBIndex::c_16)"));
  EXPECT_LT(reuse.compute.find("for(uint32_t block"),
            reuse.compute.find("tile_regs_commit()"));
  EXPECT_LT(reuse.compute.find("tile_regs_wait()"),
            reuse.compute.find("pack_tile("));
  EXPECT_EQ(Count(reuse.compute, "pack_tile("), 1u);
  EXPECT_FALSE(Has(reuse.reader, "c_4"));
  EXPECT_FALSE(Has(reuse.compute, "copy_tile("));
}

TEST(NativeKernelTest, Bf16OutputPairRequiresPrefetchAndMultipleOutputs) {
  auto ep = Projection();
  auto a = ep.matmul_like.lhs, b = ep.matmul_like.rhs,
       o = ep.matmul_like.output;
  a.element_type = b.element_type = o.element_type =
      TT_IREE_TTEX_TENSOR_ELEMENT_TYPE_BF16;
  const tt_iree_ttex_tensor_desc_t *t[TT_IREE_BF16_NATIVE_TENSOR_COUNT] = {
      &a, &b, &o};
  for (bool tensix : {false, true}) {
    auto baseline = generateBF16NativeMatrix(ep, t, "", tensix, false, false);
    auto disabled = generateBF16NativeMatrix(ep, t, "", tensix, false, true);
    EXPECT_EQ(disabled.reader, baseline.reader);
    EXPECT_EQ(disabled.compute, baseline.compute);
    EXPECT_EQ(disabled.writer, baseline.writer);
  }
  a.stride[1] = 33;
  EXPECT_EQ(generateBF16NativeMatrix(ep, t, "", true, true, true).reader,
            generateBF16NativeMatrix(ep, t, "", true, true, false).reader);
  a.stride[1] = 0;
  ep.matmul_like.m = ep.matmul_like.n = 16;
  EXPECT_EQ(generateBF16NativeMatrix(ep, t, "", true, true, true).reader,
            generateBF16NativeMatrix(ep, t, "", true, true, false).reader);
  EXPECT_EQ(generateBF16NativeMatrix(ep, t, "", true, true, true).compute,
            generateBF16NativeMatrix(ep, t, "", true, true, false).compute);
}

TEST(NativeKernelTest, Bf16OutputPairCopiesBeforeProducerReusesTheSavedPage) {
  for (uint32_t mc = 1; mc <= 7; ++mc)
    for (uint32_t nc = 1; nc <= 7; ++nc)
      for (uint32_t blocks = 1; blocks <= 9; ++blocks) {
        std::array<uint32_t, 2> a{}, b{};
        uint32_t page = 0, tiles = mc * nc;
        for (uint32_t first = 0; first < tiles;) {
          uint32_t group = std::min(2u, tiles - first);
          for (uint32_t k = 0; k < blocks; ++k) {
            uint32_t saved = page;
            for (uint32_t lane = 0; lane < group; ++lane) {
              uint32_t m = (first + lane) / nc, n = (first + lane) % nc;
              bool reuse_a = lane && m == first / nc;
              bool reuse_b = lane && n == first % nc;
              // Consumer pops never write these pages; only this producer can
              // overwrite the saved source, after copying to the other page.
              if (lane)
                ASSERT_NE(page, saved);
              uint32_t want_a = (m + 3) * blocks + k;
              uint32_t want_b = (n + 5) * blocks + k;
              a[page] = reuse_a ? a[saved] : want_a;
              b[page] = reuse_b ? b[saved] : want_b;
              EXPECT_EQ(a[page], want_a);
              EXPECT_EQ(b[page], want_b);
              page = 1 - page;
            }
          }
          first += group;
        }
      }
}

TEST(NativeKernelTest, Bf16PrefetchSupportsSpatialRowsAndRowMajorLinear) {
  auto ep = Convolution();
  auto a = ep.conv2d.input, b = ep.conv2d.filter, o = ep.conv2d.output;
  a.element_type = b.element_type = o.element_type =
      TT_IREE_TTEX_TENSOR_ELEMENT_TYPE_BF16;
  const tt_iree_ttex_tensor_desc_t *t[TT_IREE_BF16_NATIVE_TENSOR_COUNT] = {
      &a, &b, &o};
  auto source = generateBF16NativeMatrix(ep, t, "", true).reader;
  EXPECT_TRUE(Has(source, "row_remaining=8u-m%8u"));
  EXPECT_TRUE(
      Has(source, "byte_0(ic,(m/8u)*1u+kh*1u,(m%8u)*1u+kw*1u,0),chunk"));
  EXPECT_TRUE(Has(source, "byte_1(n,ic,kh,kw),valid_k"));
  ep = {};
  ep.dispatch_kind = TT_IREE_TTEX_DISPATCH_KIND_LINEAR;
  ep.linear.m = 3;
  ep.linear.n = 17;
  ep.linear.k = 35;
  a = Tensor(TT_IREE_TTEX_TENSOR_LAYOUT_ROW_MAJOR, 2, 3, 35);
  b = Tensor(TT_IREE_TTEX_TENSOR_LAYOUT_OI, 2, 17, 35);
  o = Tensor(TT_IREE_TTEX_TENSOR_LAYOUT_ROW_MAJOR, 2, 3, 17);
  a.element_type = b.element_type = o.element_type =
      TT_IREE_TTEX_TENSOR_ELEMENT_TYPE_BF16;
  source = generateBF16NativeMatrix(ep, t, "", true).reader;
  EXPECT_TRUE(Has(
      source,
      "prefetch_bf16_0(lhs.get_write_ptr()+row*128u,byte_0(m,k,0,0),valid_k)"));
}

TEST(NativeKernelTest, Bf16PrefetchLeavesUnsupportedStorageOnOriginalReader) {
  auto ep = Projection();
  auto a = ep.matmul_like.lhs, b = ep.matmul_like.rhs,
       o = ep.matmul_like.output;
  a.element_type = b.element_type = o.element_type =
      TT_IREE_TTEX_TENSOR_ELEMENT_TYPE_BF16;
  const tt_iree_ttex_tensor_desc_t *t[TT_IREE_BF16_NATIVE_TENSOR_COUNT] = {
      &a, &b, &o};
  a.stride[1] = 33;
  EXPECT_EQ(generateBF16NativeMatrix(ep, t, "", true).reader,
            generateBF16NativeMatrix(ep, t, "", true, false).reader);
  a.stride[1] = 1;
  a.element_type = TT_IREE_TTEX_TENSOR_ELEMENT_TYPE_F32;
  EXPECT_EQ(generateBF16NativeMatrix(ep, t, "", true).reader,
            generateBF16NativeMatrix(ep, t, "", true, false).reader);
  ep = Convolution();
  a = ep.conv2d.input;
  b = ep.conv2d.filter;
  o = ep.conv2d.output;
  a.element_type = b.element_type = o.element_type =
      TT_IREE_TTEX_TENSOR_ELEMENT_TYPE_BF16;
  ep.conv2d.stride_w = 33;
  EXPECT_EQ(generateBF16NativeMatrix(ep, t, "", true).reader,
            generateBF16NativeMatrix(ep, t, "", true, false).reader);
}

TEST(NativeKernelTest, Bf16PrefetchBoundsStridedAndComposedWindows) {
  auto ep = Projection();
  auto a = ep.matmul_like.lhs, b = ep.matmul_like.rhs,
       o = ep.matmul_like.output;
  a.element_type = b.element_type = o.element_type =
      TT_IREE_TTEX_TENSOR_ELEMENT_TYPE_BF16;
  const tt_iree_ttex_tensor_desc_t *t[TT_IREE_BF16_NATIVE_TENSOR_COUNT] = {
      &a, &b, &o};
  a.stride[1] = 2;
  auto source = generateBF16NativeMatrix(ep, t, "", true).reader;
  EXPECT_TRUE(Has(source, "if(chunk>16u) chunk=16u"));
  EXPECT_TRUE(Has(source, "(chunk-1u)*2u+1u"));
  EXPECT_TRUE(Has(source, "ptrs[col])[j*2u]"));
  ep = Convolution();
  a = ep.conv2d.input;
  b = ep.conv2d.filter;
  o = ep.conv2d.output;
  a.element_type = b.element_type = o.element_type =
      TT_IREE_TTEX_TENSOR_ELEMENT_TYPE_BF16;
  ep.conv2d.stride_w = 2;
  a.stride[2] = 2;
  source = generateBF16NativeMatrix(ep, t, "", true).reader;
  EXPECT_TRUE(Has(source, "if(chunk>8u) chunk=8u"));
  EXPECT_TRUE(Has(source, "(chunk-1u)*4u+1u"));
  EXPECT_TRUE(Has(source, "ptrs[col])[j*4u]"));
  a.stride[2] = UINT32_MAX;
  EXPECT_EQ(generateBF16NativeMatrix(ep, t, "", true).reader,
            generateBF16NativeMatrix(ep, t, "", true, false).reader);
  for (unsigned step = 1; step <= 32; ++step) {
    unsigned chunk = 1 + 31 / step, span = (chunk - 1) * step + 1;
    EXPECT_LE(span, 32u);
    EXPECT_LE(span * 2 + 63, 128u);
    EXPECT_LT((chunk - 1) * step, span);
  }
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

TEST(NativeKernelTest, ScalarGridIsBoundedAndDoesNotRequireEvenDivisibility) {
  using mlir::iree_compiler::IREE::HAL::selectNativeScalarGrid;
  for (uint64_t rows : {0, 1, 3, 9, 15, 64, 672, 3584}) {
    auto grid = selectNativeScalarGrid(rows);
    EXPECT_GE(grid.x, 1u);
    EXPECT_GE(grid.y, 1u);
    EXPECT_LE(grid.x, 8u);
    EXPECT_LE(grid.y, 8u);
    if (rows) {
      EXPECT_LE(grid.x * grid.y, rows);
    }
    if (rows >= 64) {
      EXPECT_EQ(grid.x * grid.y, 64u);
    }
  }
}

TEST(NativeKernelTest, ScalarGridUsesTargetLimitsWithoutIdleCores) {
  using mlir::iree_compiler::IREE::HAL::selectNativeScalarGrid;
  for (uint64_t rows : {0ull, 1ull, 3ull, 9ull, 15ull, 64ull, 109ull, 110ull,
                        111ull, 672ull, 3584ull, 16ull * 64 * 56,
                        16ull * 2048, ~0ull}) {
    auto blackhole = selectNativeScalarGrid(rows, 11, 10);
    auto wormhole = selectNativeScalarGrid(rows, 8, 8);
    EXPECT_GE(blackhole.x, 1u);
    EXPECT_GE(blackhole.y, 1u);
    EXPECT_LE(blackhole.x, 11u);
    EXPECT_LE(blackhole.y, 10u);
    EXPECT_LE(wormhole.x, 8u);
    EXPECT_LE(wormhole.y, 8u);
    if (rows) {
      EXPECT_LE(blackhole.x * blackhole.y, rows);
      EXPECT_LE(wormhole.x * wormhole.y, rows);
    }
    if (rows >= 110) {
      EXPECT_EQ(blackhole.x, 11u);
      EXPECT_EQ(blackhole.y, 10u);
      EXPECT_EQ(wormhole.x * wormhole.y, 64u);
    }
    EXPECT_EQ(wormhole.x, selectNativeScalarGrid(rows).x);
    EXPECT_EQ(wormhole.y, selectNativeScalarGrid(rows).y);
  }
  EXPECT_EQ(selectNativeScalarGrid(100, 0, 10).x, 1u);
  EXPECT_EQ(selectNativeScalarGrid(100, 11, 0).y, 1u);
}

TEST(NativeKernelTest, MatrixTileRectanglesCoverUnevenGridsWithoutIdleCores) {
  using namespace iree::hal::tenstorrent;
  for (uint32_t mt : {1, 2, 7, 10, 11, 17, 32, 49, 128})
    for (uint32_t nt : {1, 2, 7, 10, 11, 17, 32, 49, 128})
      for (uint32_t rows : {1, 2, 8, 10})
        for (uint32_t cols : {1, 2, 8, 11}) {
          NativeTileRectangle rectangle;
          if (rows > mt || cols > nt) {
            EXPECT_FALSE(PartitionNativeMatrixTiles(mt, nt, 0, 0, rows, cols,
                                                    &rectangle));
            continue;
          }
          uint32_t m_end = 0;
          uint64_t covered = 0;
          for (uint32_t r = 0; r < rows; ++r) {
            uint32_t n_end = 0;
            for (uint32_t c = 0; c < cols; ++c) {
              ASSERT_TRUE(PartitionNativeMatrixTiles(mt, nt, r, c, rows, cols,
                                                     &rectangle));
              EXPECT_EQ(rectangle.m_begin, m_end);
              EXPECT_EQ(rectangle.n_begin, n_end);
              EXPECT_GE(rectangle.m_count, mt / rows);
              EXPECT_LE(rectangle.m_count, mt / rows + (mt % rows != 0));
              EXPECT_GE(rectangle.n_count, nt / cols);
              EXPECT_LE(rectangle.n_count, nt / cols + (nt % cols != 0));
              EXPECT_GT(rectangle.m_count, 0u);
              EXPECT_GT(rectangle.n_count, 0u);
              covered += uint64_t(rectangle.m_count) * rectangle.n_count;
              n_end += rectangle.n_count;
            }
            EXPECT_EQ(n_end, nt);
            m_end += rectangle.m_count;
          }
          EXPECT_EQ(m_end, mt);
          EXPECT_EQ(covered, uint64_t(mt) * nt);
        }
}

TEST(NativeKernelTest, BlackholeGridUsesAllElevenColumnsForThirtyTwoNTiles) {
  using namespace iree::hal::tenstorrent;
  uint64_t covered = 0;
  uint32_t cores = 0, two_wide = 0, three_wide = 0;
  for (uint32_t r = 0; r < 10; ++r)
    for (uint32_t c = 0; c < 11; ++c) {
      NativeTileRectangle rectangle;
      ASSERT_TRUE(PartitionNativeMatrixTiles(33, 32, r, c, 10, 11, &rectangle));
      EXPECT_TRUE(rectangle.m_count == 3 || rectangle.m_count == 4);
      EXPECT_TRUE(rectangle.n_count == 2 || rectangle.n_count == 3);
      two_wide += rectangle.n_count == 2;
      three_wide += rectangle.n_count == 3;
      covered += rectangle.m_count * rectangle.n_count;
      ++cores;
    }
  EXPECT_EQ(cores, 110u);
  EXPECT_EQ(two_wide, 10u);
  EXPECT_EQ(three_wide, 100u);
  EXPECT_EQ(covered, 33u * 32u);
}

TEST(NativeKernelTest, MatrixTileRectanglesPreserveEvenPartitionAbi) {
  using namespace iree::hal::tenstorrent;
  for (uint32_t r = 0; r < 8; ++r)
    for (uint32_t c = 0; c < 8; ++c) {
      NativeTileRectangle rectangle;
      ASSERT_TRUE(PartitionNativeMatrixTiles(32, 64, r, c, 8, 8, &rectangle));
      EXPECT_EQ(rectangle.m_begin, r * 4);
      EXPECT_EQ(rectangle.n_begin, c * 8);
      EXPECT_EQ(rectangle.m_count, 4u);
      EXPECT_EQ(rectangle.n_count, 8u);
    }
}

TEST(NativeKernelTest, MatrixTileRectanglesRejectInvalidOrIdleCoreGrids) {
  using namespace iree::hal::tenstorrent;
  NativeTileRectangle rectangle;
  EXPECT_FALSE(PartitionNativeMatrixTiles(0, 32, 0, 0, 1, 11, &rectangle));
  EXPECT_FALSE(PartitionNativeMatrixTiles(32, 0, 0, 0, 10, 1, &rectangle));
  EXPECT_FALSE(PartitionNativeMatrixTiles(32, 32, 0, 0, 0, 11, &rectangle));
  EXPECT_FALSE(PartitionNativeMatrixTiles(32, 32, 0, 0, 10, 0, &rectangle));
  EXPECT_FALSE(PartitionNativeMatrixTiles(32, 32, 10, 0, 10, 11, &rectangle));
  EXPECT_FALSE(PartitionNativeMatrixTiles(32, 32, 0, 11, 10, 11, &rectangle));
  EXPECT_FALSE(PartitionNativeMatrixTiles(9, 32, 0, 0, 10, 11, &rectangle));
  EXPECT_FALSE(PartitionNativeMatrixTiles(32, 10, 0, 0, 10, 11, &rectangle));
  EXPECT_FALSE(PartitionNativeMatrixTiles(32, 32, 0, 0, 10, 11, nullptr));
  ASSERT_TRUE(PartitionNativeMatrixTiles(UINT32_MAX, UINT32_MAX, 9, 10, 10, 11,
                                         &rectangle));
  EXPECT_EQ(uint64_t(rectangle.m_begin) + rectangle.m_count, UINT32_MAX);
  EXPECT_EQ(uint64_t(rectangle.n_begin) + rectangle.n_count, UINT32_MAX);
}

TEST(NativeKernelTest, RowPartitionsCoverEveryElementExactlyOnce) {
  using namespace iree::hal::tenstorrent;
  for (uint32_t rows : {0, 1, 3, 15, 64, 672, 3584})
    for (uint32_t width : {1, 19, 56, 224})
      for (uint32_t cores : {1, 2, 4, 8, 16, 32, 64, 110}) {
        uint32_t previous_end = 0;
        for (uint32_t core = 0; core < cores; ++core) {
          NativeElementRange range;
          ASSERT_TRUE(PartitionNativeScalarRange(rows * width, width, core,
                                                 cores, &range));
          EXPECT_EQ(range.begin, previous_end);
          EXPECT_EQ(range.begin % width, 0u);
          EXPECT_EQ(range.count % width, 0u);
          previous_end += range.count;
        }
        EXPECT_EQ(previous_end, rows * width);
      }
  NativeElementRange invalid;
  EXPECT_FALSE(PartitionNativeScalarRange(10, 0, 0, 1, &invalid));
  EXPECT_FALSE(PartitionNativeScalarRange(10, 3, 0, 1, &invalid));
  EXPECT_FALSE(PartitionNativeScalarRange(10, 1, 0, 0, &invalid));
  EXPECT_FALSE(PartitionNativeScalarRange(10, 1, 2, 2, &invalid));
  NativeElementRange last;
  ASSERT_TRUE(PartitionNativeScalarRange(UINT32_MAX, 1, 63, 64, &last));
  EXPECT_EQ(uint64_t(last.begin) + last.count, UINT32_MAX);
}

TEST(NativeKernelTest, SpatialPoolUsesBoundedWindowsAndTheOriginalScratchAbi) {
  auto input = Tensor(TT_IREE_TTEX_TENSOR_LAYOUT_CHW, 3, 5, 7, 39);
  auto output = Tensor(TT_IREE_TTEX_TENSOR_LAYOUT_CHW, 3, 5, 3, 19);
  input.element_type = output.element_type =
      TT_IREE_TTEX_TENSOR_ELEMENT_TYPE_BF16;
  const tt_iree_ttex_tensor_desc_t *tensors[TT_IREE_BF16_NATIVE_TENSOR_COUNT] =
      {};
  tensors[0] = &input;
  tensors[2] = &output;
  using mlir::iree_compiler::IREE::HAL::generateBF16NativeSpatial;
  auto source =
      generateBF16NativeSpatial(TT_IREE_TTEX_DISPATCH_KIND_MAX_POOL, tensors);
  EXPECT_TRUE(Has(source, "float value[8], row[17]"));
  EXPECT_TRUE(Has(source, "scratch + 128u"));
  EXPECT_TRUE(Has(source, "2048u - byte % 2048u"));
  EXPECT_TRUE(Has(source, "get_arg_val<uint32_t>(28)"));
  EXPECT_TRUE(Has(source, "value[j]=maximum_exact"));
  input.stride[2] = 2;
  EXPECT_TRUE(
      generateBF16NativeSpatial(TT_IREE_TTEX_DISPATCH_KIND_MAX_POOL, tensors)
          .empty());
  input.stride[2] = 0;
  output.shape[2] = 18;
  EXPECT_TRUE(
      generateBF16NativeSpatial(TT_IREE_TTEX_DISPATCH_KIND_MAX_POOL, tensors)
          .empty());
}

TEST(NativeKernelTest, SpatialCopyRepackagesRawBitsRatherThanRoundingFloats) {
  auto input = Tensor(TT_IREE_TTEX_TENSOR_LAYOUT_CHW, 3, 3, 7, 39);
  auto output = input;
  input.element_type = output.element_type =
      TT_IREE_TTEX_TENSOR_ELEMENT_TYPE_BF16;
  output.storage_shape[1] = 9;
  output.storage_shape[2] = 43;
  output.offset[1] = output.offset[2] = 1;
  const tt_iree_ttex_tensor_desc_t *tensors[TT_IREE_BF16_NATIVE_TENSOR_COUNT] =
      {};
  tensors[0] = &input;
  tensors[2] = &output;
  using mlir::iree_compiler::IREE::HAL::generateBF16NativeSpatial;
  auto source = generateBF16NativeSpatial(
      TT_IREE_TTEX_DISPATCH_KIND_PADDED_COPY, tensors);
  ASSERT_FALSE(source.empty());
  auto body = source.substr(source.find("void kernel_main()"));
  EXPECT_TRUE(Has(body, "uint16_t values[16]"));
  EXPECT_FALSE(Has(body, "bf16_bits"));
  EXPECT_FALSE(Has(body, "write_2(scratch"));
  input.element_type = TT_IREE_TTEX_TENSOR_ELEMENT_TYPE_F32;
  EXPECT_TRUE(
      generateBF16NativeSpatial(TT_IREE_TTEX_DISPATCH_KIND_PADDED_COPY, tensors)
          .empty());
}
TEST(NativeKernelTest, BatchedProjectionSplitsWindowsAtEveryImageBoundary) {
  auto ep = Projection();
  ep.matmul_like.m = 98;
  auto a = Tensor(TT_IREE_TTEX_TENSOR_LAYOUT_BATCH_CHANNEL_MAJOR, 3, 2, 35, 49);
  auto b = Tensor(TT_IREE_TTEX_TENSOR_LAYOUT_OI, 2, 33, 35);
  auto o = Tensor(TT_IREE_TTEX_TENSOR_LAYOUT_BATCH_CHANNEL_MAJOR, 3, 2, 33, 49);
  a.element_type = b.element_type = o.element_type = TT_IREE_TTEX_TENSOR_ELEMENT_TYPE_BF16;
  const tt_iree_ttex_tensor_desc_t *t[TT_IREE_BF16_NATIVE_TENSOR_COUNT] = {&a, &b, &o};
  auto sources = generateBF16NativeMatrix(ep, t, "value=round_bf16(value);\n", true);
  EXPECT_TRUE(Has(sources.reader, "byte_0(m/49u,k,m%49u,0),chunk)"));
  EXPECT_TRUE(Has(sources.reader, "row_remaining=49u-m%49u"));
  EXPECT_TRUE(Has(sources.reader, "valid_m=98u-m_base"));
  EXPECT_TRUE(Has(sources.writer, "byte_2(m/49u,n,m%49u,0),chunk,128u)"));
  auto reference = generateBF16NativeMatrix(ep, t, "", false).reader;
  EXPECT_TRUE(Has(reference, "byte_0(m/49u,k,m%49u,0),chunk,128u)"));
}

TEST(NativeKernelTest, NchwConvolutionKeepsBatchOutOfTheChannelReduction) {
  auto ep = Convolution();
  ep.conv2d.input = Tensor(TT_IREE_TTEX_TENSOR_LAYOUT_NCHW, 4, 2, 3, 7, 23);
  ep.conv2d.filter = Tensor(TT_IREE_TTEX_TENSOR_LAYOUT_OIHW, 4, 5, 3, 3, 3);
  ep.conv2d.output = Tensor(TT_IREE_TTEX_TENSOR_LAYOUT_NCHW, 4, 2, 5, 3, 11);
  ep.conv2d.stride_h = ep.conv2d.stride_w = 2;
  auto a = ep.conv2d.input, b = ep.conv2d.filter, o = ep.conv2d.output;
  a.element_type = b.element_type = o.element_type = TT_IREE_TTEX_TENSOR_ELEMENT_TYPE_BF16;
  const tt_iree_ttex_tensor_desc_t *t[TT_IREE_BF16_NATIVE_TENSOR_COUNT] = {&a, &b, &o};
  auto sources = generateBF16NativeMatrix(ep, t, "", true);
  EXPECT_TRUE(Has(sources.reader, "valid_m=66u-m_base"));
  EXPECT_TRUE(Has(sources.reader, "valid_n=5u-n_base"));
  EXPECT_TRUE(Has(sources.reader, "valid_k=27u-block*32"));
  EXPECT_TRUE(Has(sources.reader, "byte_0(m/33u,ic,(m/11u)%3u*2u+kh*1u,(m%11u)*2u+kw*1u)"));
  EXPECT_TRUE(Has(sources.reader, "if(chunk>16u) chunk=16u"));
  EXPECT_TRUE(Has(sources.writer, "byte_2(m/33u,n,(m/11u)%3u,m%11u),chunk,128u)"));
}

TEST(NativeKernelTest, BatchedSpatialKernelsDecodeImageAndChannelSeparately) {
  using mlir::iree_compiler::IREE::HAL::generateBF16NativeSpatial;
  auto a = Tensor(TT_IREE_TTEX_TENSOR_LAYOUT_NCHW, 4, 2, 3, 7, 39);
  auto o = Tensor(TT_IREE_TTEX_TENSOR_LAYOUT_NCHW, 4, 2, 3, 3, 19);
  a.element_type = o.element_type = TT_IREE_TTEX_TENSOR_ELEMENT_TYPE_BF16;
  const tt_iree_ttex_tensor_desc_t *t[TT_IREE_BF16_NATIVE_TENSOR_COUNT] = {};
  t[0] = &a;
  t[2] = &o;
  auto pool = generateBF16NativeSpatial(TT_IREE_TTEX_DISPATCH_KIND_MAX_POOL, t);
  EXPECT_TRUE(Has(pool, "batch=i/171u"));
  EXPECT_TRUE(Has(pool, "c=(i/57u)%3u"));
  EXPECT_TRUE(Has(pool, "byte_0(batch,c,h*2u+kh,w*2u)"));
  EXPECT_TRUE(Has(pool, "byte_2(batch,c,h,w)"));
  o = a;
  auto copy = generateBF16NativeSpatial(TT_IREE_TTEX_DISPATCH_KIND_PADDED_COPY, t);
  EXPECT_TRUE(Has(copy, "batch=i/819u"));
  EXPECT_TRUE(Has(copy, "source_byte=byte_0(batch,c,h,w)"));
  EXPECT_TRUE(Has(copy, "target_byte=byte_2(batch,c,h,w)"));
  o.shape[0] = 3;
  EXPECT_TRUE(generateBF16NativeSpatial(TT_IREE_TTEX_DISPATCH_KIND_PADDED_COPY, t).empty());
}
} // namespace
