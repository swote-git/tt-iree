// Copyright 2025 The tt-iree Authors
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// Unit tests for TTEX FlatBuffer schema: build → verify → parse round-trip.
// These tests run without hardware (mock mode OK) and without the compiler.

#include <algorithm>
#include <array>
#include <cstring>
#include <functional>
#include <initializer_list>

#include "iree/base/api.h"
#include "iree/schema/tt_bf16_native_validation.h"
#include "iree/schema/tt_executable_builder_util.h"
#include "iree/schema/tt_native_dispatch.h"
#include "iree/testing/gtest.h"
#include "iree/testing/status_matchers.h"

// Generated flatcc headers
#include "tt_executable_def_reader.h"
#include "tt_executable_def_verifier.h"

#define ttex_ns(x) FLATBUFFERS_WRAP_NAMESPACE(iree_tenstorrent_hal, x)

namespace iree {
namespace hal {
namespace tenstorrent {
namespace {

static tt_iree_ttex_entry_point_desc_t MakeMatmulLikeEntry() {
  tt_iree_ttex_entry_point_desc_t ep = {};
  ep.name = "resnet50_projection";
  ep.binding_count = 3;
  ep.workgroup_size[0] = 1;
  ep.workgroup_size[1] = 1;
  ep.workgroup_size[2] = 1;
  ep.builtin_program = TT_IREE_TTEX_BUILTIN_PROGRAM_F32_MATMUL_LIKE;
  ep.dispatch_kind = TT_IREE_TTEX_DISPATCH_KIND_MATMUL_LIKE;

  ep.matmul_like.lhs = {
      .binding = 0,
      .element_type = TT_IREE_TTEX_TENSOR_ELEMENT_TYPE_F32,
      .layout = TT_IREE_TTEX_TENSOR_LAYOUT_CHW,
      .rank = 3,
      .shape = {64, 8, 8, 0},
      .storage_shape = {64, 8, 8, 0},
  };
  ep.matmul_like.rhs = {
      .binding = 1,
      .element_type = TT_IREE_TTEX_TENSOR_ELEMENT_TYPE_F32,
      .layout = TT_IREE_TTEX_TENSOR_LAYOUT_OI,
      .rank = 2,
      .shape = {64, 64, 0, 0},
      .storage_shape = {64, 64, 0, 0},
  };
  ep.matmul_like.output = {
      .binding = 2,
      .element_type = TT_IREE_TTEX_TENSOR_ELEMENT_TYPE_F32,
      .layout = TT_IREE_TTEX_TENSOR_LAYOUT_CHW,
      .rank = 3,
      .shape = {64, 8, 8, 0},
      .storage_shape = {64, 10, 10, 0},
      .offset = {0, 1, 1, 0},
  };
  ep.matmul_like.m = 64;
  ep.matmul_like.n = 64;
  ep.matmul_like.k = 64;
  ep.matmul_like.epilogue_flags =
      TT_IREE_TTEX_EPILOGUE_AFFINE | TT_IREE_TTEX_EPILOGUE_RELU;
  ep.matmul_like.epilogue_scale = 0.5f;
  ep.matmul_like.epilogue_bias = 0.1875f;
  return ep;
}

static tt_iree_ttex_entry_point_desc_t MakePerChannelMatmulLikeEntry() {
  static const std::array<float, 64> kScales = [] {
    std::array<float, 64> values = {};
    for (size_t i = 0; i < values.size(); ++i)
      values[i] = i % 2 == 0 ? 0.5f : 0.25f;
    return values;
  }();
  static const std::array<float, 64> kBiases = [] {
    std::array<float, 64> values = {};
    for (size_t i = 0; i < values.size(); ++i)
      values[i] = static_cast<float>(i);
    return values;
  }();

  tt_iree_ttex_entry_point_desc_t ep = MakeMatmulLikeEntry();
  ep.name = "resnet50_per_channel_projection";
  ep.matmul_like.epilogue_flags |= TT_IREE_TTEX_EPILOGUE_PER_CHANNEL;
  ep.matmul_like.epilogue_channel_count = kScales.size();
  ep.matmul_like.epilogue_scales = kScales.data();
  ep.matmul_like.epilogue_biases = kBiases.data();
  return ep;
}

static tt_iree_ttex_entry_point_desc_t MakeConv2DEntry() {
  tt_iree_ttex_entry_point_desc_t ep = {};
  ep.name = "resnet50_conv2d";
  ep.binding_count = 3;
  ep.workgroup_size[0] = 1;
  ep.workgroup_size[1] = 1;
  ep.workgroup_size[2] = 1;
  ep.builtin_program = TT_IREE_TTEX_BUILTIN_PROGRAM_F32_CONV2D;
  ep.dispatch_kind = TT_IREE_TTEX_DISPATCH_KIND_CONVOLUTION;
  ep.conv2d.input = {
      .binding = 0,
      .element_type = TT_IREE_TTEX_TENSOR_ELEMENT_TYPE_F32,
      .layout = TT_IREE_TTEX_TENSOR_LAYOUT_CHW,
      .rank = 3,
      .shape = {64, 10, 10, 0},
      .storage_shape = {64, 10, 10, 0},
  };
  ep.conv2d.filter = {
      .binding = 1,
      .element_type = TT_IREE_TTEX_TENSOR_ELEMENT_TYPE_F32,
      .layout = TT_IREE_TTEX_TENSOR_LAYOUT_OIHW,
      .rank = 4,
      .shape = {64, 64, 3, 3},
      .storage_shape = {64, 64, 3, 3},
  };
  ep.conv2d.output = {
      .binding = 2,
      .element_type = TT_IREE_TTEX_TENSOR_ELEMENT_TYPE_F32,
      .layout = TT_IREE_TTEX_TENSOR_LAYOUT_CHW,
      .rank = 3,
      .shape = {64, 8, 8, 0},
      .storage_shape = {64, 8, 8, 0},
  };
  ep.conv2d.kernel_h = 3;
  ep.conv2d.kernel_w = 3;
  ep.conv2d.stride_h = 1;
  ep.conv2d.stride_w = 1;
  ep.conv2d.dilation_h = 1;
  ep.conv2d.dilation_w = 1;
  ep.conv2d.groups = 1;
  ep.conv2d.epilogue_flags =
      TT_IREE_TTEX_EPILOGUE_AFFINE | TT_IREE_TTEX_EPILOGUE_RELU;
  ep.conv2d.epilogue_scale = 0.25f;
  ep.conv2d.epilogue_bias = 0.125f;
  return ep;
}

static tt_iree_ttex_entry_point_desc_t MakePaddedCopyEntry() {
  tt_iree_ttex_entry_point_desc_t ep = {};
  ep.name = "resnet50_input_pad";
  ep.binding_count = 2;
  ep.workgroup_size[0] = 1;
  ep.workgroup_size[1] = 1;
  ep.workgroup_size[2] = 1;
  ep.builtin_program = TT_IREE_TTEX_BUILTIN_PROGRAM_F32_PADDED_COPY;
  ep.dispatch_kind = TT_IREE_TTEX_DISPATCH_KIND_PADDED_COPY;
  ep.padded_copy.source = {
      .binding = 0,
      .element_type = TT_IREE_TTEX_TENSOR_ELEMENT_TYPE_F32,
      .layout = TT_IREE_TTEX_TENSOR_LAYOUT_CHW,
      .rank = 3,
      .shape = {3, 224, 224, 0},
      .storage_shape = {3, 224, 224, 0},
  };
  ep.padded_copy.destination = {
      .binding = 1,
      .element_type = TT_IREE_TTEX_TENSOR_ELEMENT_TYPE_F32,
      .layout = TT_IREE_TTEX_TENSOR_LAYOUT_CHW,
      .rank = 3,
      .shape = {3, 224, 224, 0},
      .storage_shape = {3, 230, 230, 0},
      .offset = {0, 3, 3, 0},
  };
  return ep;
}

static tt_iree_ttex_entry_point_desc_t MakeMaxPoolEntry() {
  tt_iree_ttex_entry_point_desc_t ep = {};
  ep.name = "resnet50_stem_max_pool";
  ep.binding_count = 2;
  ep.workgroup_size[0] = 1;
  ep.workgroup_size[1] = 1;
  ep.workgroup_size[2] = 1;
  ep.builtin_program = TT_IREE_TTEX_BUILTIN_PROGRAM_F32_MAX_POOL_2D;
  ep.dispatch_kind = TT_IREE_TTEX_DISPATCH_KIND_MAX_POOL;
  ep.max_pool.input = {
      .binding = 0,
      .element_type = TT_IREE_TTEX_TENSOR_ELEMENT_TYPE_F32,
      .layout = TT_IREE_TTEX_TENSOR_LAYOUT_CHW,
      .rank = 3,
      .shape = {1, 6, 6, 0},
      .storage_shape = {1, 6, 6, 0},
  };
  ep.max_pool.output = {
      .binding = 1,
      .element_type = TT_IREE_TTEX_TENSOR_ELEMENT_TYPE_F32,
      .layout = TT_IREE_TTEX_TENSOR_LAYOUT_CHW,
      .rank = 3,
      .shape = {1, 2, 2, 0},
      .storage_shape = {1, 2, 2, 0},
  };
  ep.max_pool.kernel_h = 3;
  ep.max_pool.kernel_w = 3;
  ep.max_pool.stride_h = 2;
  ep.max_pool.stride_w = 2;
  ep.max_pool.dilation_h = 1;
  ep.max_pool.dilation_w = 1;
  return ep;
}

static tt_iree_ttex_entry_point_desc_t MakeResidualMatmulLikeEntry() {
  tt_iree_ttex_entry_point_desc_t ep = {};
  ep.name = "resnet50_residual_projection";
  ep.binding_count = 3;
  ep.workgroup_size[0] = 1;
  ep.workgroup_size[1] = 1;
  ep.workgroup_size[2] = 1;
  ep.builtin_program = TT_IREE_TTEX_BUILTIN_PROGRAM_F32_MATMUL_LIKE;
  ep.dispatch_kind = TT_IREE_TTEX_DISPATCH_KIND_MATMUL_LIKE;
  ep.matmul_like.lhs = {
      .binding = 0,
      .element_type = TT_IREE_TTEX_TENSOR_ELEMENT_TYPE_F32,
      .layout = TT_IREE_TTEX_TENSOR_LAYOUT_CHANNEL_MAJOR,
      .rank = 2,
      .shape = {64, 64, 0, 0},
      .storage_shape = {64, 64, 0, 0},
      .byte_offset = 25600,
  };
  ep.matmul_like.rhs = {
      .binding = 1,
      .element_type = TT_IREE_TTEX_TENSOR_ELEMENT_TYPE_F32,
      .layout = TT_IREE_TTEX_TENSOR_LAYOUT_OI,
      .rank = 2,
      .shape = {256, 64, 0, 0},
      .storage_shape = {256, 64, 0, 0},
  };
  ep.matmul_like.output = {
      .binding = 2,
      .element_type = TT_IREE_TTEX_TENSOR_ELEMENT_TYPE_F32,
      .layout = TT_IREE_TTEX_TENSOR_LAYOUT_CHANNEL_MAJOR,
      .rank = 2,
      .shape = {256, 64, 0, 0},
      .storage_shape = {256, 64, 0, 0},
  };
  ep.matmul_like.residual = {
      .binding = 0,
      .element_type = TT_IREE_TTEX_TENSOR_ELEMENT_TYPE_F32,
      .layout = TT_IREE_TTEX_TENSOR_LAYOUT_CHANNEL_MAJOR,
      .rank = 2,
      .shape = {256, 64, 0, 0},
      .storage_shape = {256, 64, 0, 0},
      .byte_offset = 41984,
  };
  ep.matmul_like.m = 64;
  ep.matmul_like.n = 256;
  ep.matmul_like.k = 64;
  ep.matmul_like.epilogue_flags = TT_IREE_TTEX_EPILOGUE_AFFINE |
                                  TT_IREE_TTEX_EPILOGUE_RESIDUAL |
                                  TT_IREE_TTEX_EPILOGUE_RELU;
  ep.matmul_like.epilogue_scale = 0.5f;
  ep.matmul_like.epilogue_bias = 0.1875f;
  ep.matmul_like.residual_scale = 0.25f;
  ep.matmul_like.residual_bias = 0.125f;
  return ep;
}

TEST(TtexSchemaTest, BuildVerifySingleEntry) {
  iree_allocator_t alloc = iree_allocator_system();

  tt_iree_ttex_entry_point_desc_t ep = {};
  ep.name = "test_add";
  ep.constant_count = 0;
  ep.binding_count = 3;
  ep.flags = 0;
  ep.workgroup_size[0] = 1;
  ep.workgroup_size[1] = 1;
  ep.workgroup_size[2] = 1;
  ep.builtin_program = TT_IREE_TTEX_BUILTIN_PROGRAM_CUSTOM_SFPI_ADD;

  iree_byte_span_t data = {0};
  IREE_ASSERT_OK(tt_iree_build_ttex_executable_def(alloc, 1, &ep, &data));
  ASSERT_NE(data.data, nullptr);
  ASSERT_GT(data.data_length, 0u);

  int ret = ttex_ns(ExecutableDef_verify_as_root(data.data, data.data_length));
  ASSERT_EQ(ret, 0);

  ttex_ns(ExecutableDef_table_t) def =
      ttex_ns(ExecutableDef_as_root(data.data));
  ASSERT_EQ(ttex_ns(ExecutableDef_version(def)), TT_IREE_TTEX_CURRENT_VERSION);

  ttex_ns(EntryPointDef_vec_t) eps = ttex_ns(ExecutableDef_entry_points(def));
  ASSERT_EQ(ttex_ns(EntryPointDef_vec_len(eps)), 1u);

  ttex_ns(EntryPointDef_table_t) ep0 = ttex_ns(EntryPointDef_vec_at(eps, 0));
  ASSERT_STREQ(ttex_ns(EntryPointDef_name(ep0)), "test_add");
  ASSERT_EQ(ttex_ns(EntryPointDef_binding_count(ep0)), 3);
  ASSERT_EQ(ttex_ns(EntryPointDef_constant_count(ep0)), 0);
  ASSERT_EQ(ttex_ns(EntryPointDef_workgroup_size_x(ep0)), 1u);
  ASSERT_EQ(ttex_ns(EntryPointDef_workgroup_size_y(ep0)), 1u);
  ASSERT_EQ(ttex_ns(EntryPointDef_workgroup_size_z(ep0)), 1u);
  ASSERT_EQ(ttex_ns(EntryPointDef_builtin(ep0)), 0u);

  iree_allocator_free(alloc, data.data);
}

TEST(TtexSchemaTest, Bf16NativeRoundTripPreservesMixedPrecisionSlots) {
  auto ep = MakeMatmulLikeEntry();
  ep.builtin_program = TT_IREE_TTEX_BUILTIN_PROGRAM_BF16_NATIVE;
  ep.dispatch_kind = TT_IREE_TTEX_DISPATCH_KIND_BF16_NATIVE;
  auto &d = ep.bf16_native;
  d.operation_kind = TT_IREE_TTEX_DISPATCH_KIND_MATMUL_LIKE;
  d.m = d.n = d.k = 64;
  d.tensors[0] = ep.matmul_like.lhs;
  d.tensors[1] = ep.matmul_like.rhs;
  d.tensors[2] = ep.matmul_like.output;
  d.tensors[0].element_type = d.tensors[1].element_type =
      d.tensors[2].element_type = TT_IREE_TTEX_TENSOR_ELEMENT_TYPE_BF16;
  d.tensors[10] = d.tensors[0];
  d.tensors[10].element_type = TT_IREE_TTEX_TENSOR_ELEMENT_TYPE_F32;
  auto alloc = iree_allocator_system();
  iree_byte_span_t data = {};
  IREE_ASSERT_OK(tt_iree_build_ttex_executable_def_with_version(
      alloc, TT_IREE_TTEX_VERSION_17, 1, &ep, &data));
  ASSERT_EQ(ttex_ns(ExecutableDef_verify_as_root(data.data, data.data_length)), 0);
  auto root = ttex_ns(ExecutableDef_as_root(data.data));
  auto fb_ep = ttex_ns(EntryPointDef_vec_at(ttex_ns(ExecutableDef_entry_points(root)), 0));
  auto native = ttex_ns(EntryPointDef_bf16_native(fb_ep));
  ASSERT_NE(native, nullptr);
  auto tensors = ttex_ns(BF16NativeDef_tensors(native));
  EXPECT_EQ(ttex_ns(TensorDesc_vec_len(tensors)), TT_IREE_BF16_NATIVE_TENSOR_COUNT);
  EXPECT_EQ(ttex_ns(TensorDesc_element_type(ttex_ns(TensorDesc_vec_at(tensors, 0)))), TT_IREE_TTEX_TENSOR_ELEMENT_TYPE_BF16);
  EXPECT_EQ(ttex_ns(TensorDesc_element_type(ttex_ns(TensorDesc_vec_at(tensors, 10)))), TT_IREE_TTEX_TENSOR_ELEMENT_TYPE_F32);
  EXPECT_EQ(ttex_ns(TensorDesc_rank(ttex_ns(TensorDesc_vec_at(tensors, 13)))), 0);
  iree_allocator_free(alloc, data.data);
  auto status = tt_iree_build_ttex_executable_def_with_version(
      alloc, TT_IREE_TTEX_VERSION_15, 1, &ep, &data);
  EXPECT_EQ(iree_status_code(status), IREE_STATUS_INVALID_ARGUMENT);
  iree_status_ignore(status);
}

TEST(TtexSchemaTest, Bf16TensixRequiresV18MatrixAndThreeRoles) {
  auto ep = MakeMatmulLikeEntry();
  ep.builtin_program = TT_IREE_TTEX_BUILTIN_PROGRAM_BF16_NATIVE_TENSIX;
  ep.dispatch_kind = TT_IREE_TTEX_DISPATCH_KIND_BF16_NATIVE;
  ep.bf16_native.operation_kind = TT_IREE_TTEX_DISPATCH_KIND_MATMUL_LIKE;
  ep.bf16_native.m = ep.bf16_native.n = ep.bf16_native.k = 64;
  auto allocator = iree_allocator_system();
  iree_byte_span_t data = {};
  IREE_ASSERT_OK(tt_iree_build_ttex_executable_def_with_version(
      allocator, TT_IREE_TTEX_VERSION_18, 1, &ep, &data));
  ASSERT_EQ(ttex_ns(ExecutableDef_verify_as_root(data.data, data.data_length)), 0);
  auto root = ttex_ns(ExecutableDef_as_root(data.data));
  EXPECT_EQ(ttex_ns(ExecutableDef_version(root)), TT_IREE_TTEX_VERSION_18);
  auto entry = ttex_ns(EntryPointDef_vec_at(ttex_ns(ExecutableDef_entry_points(root)), 0));
  EXPECT_EQ(ttex_ns(EntryPointDef_builtin(entry)), TT_IREE_TTEX_BUILTIN_PROGRAM_BF16_NATIVE_TENSIX);
  iree_allocator_free(allocator, data.data);
  auto status = tt_iree_build_ttex_executable_def_with_version(
      allocator, TT_IREE_TTEX_VERSION_17, 1, &ep, &data);
  EXPECT_EQ(iree_status_code(status), IREE_STATUS_INVALID_ARGUMENT);
  iree_status_ignore(status);
  ep.bf16_native.operation_kind = TT_IREE_TTEX_DISPATCH_KIND_MAX_POOL;
  status = tt_iree_build_ttex_executable_def_with_version(
      allocator, TT_IREE_TTEX_VERSION_18, 1, &ep, &data);
  EXPECT_EQ(iree_status_code(status), IREE_STATUS_INVALID_ARGUMENT);
  iree_status_ignore(status);
  EXPECT_EQ(__builtin_popcount(tt_iree_ttex_required_aot_kernel_roles_for_version(
      TT_IREE_TTEX_VERSION_18, TT_IREE_TTEX_BUILTIN_PROGRAM_BF16_NATIVE_TENSIX,
      TT_IREE_TTEX_DISPATCH_KIND_BF16_NATIVE)), 3);
  EXPECT_EQ(__builtin_popcount(tt_iree_ttex_required_aot_kernel_roles_for_version(
      TT_IREE_TTEX_VERSION_18, TT_IREE_TTEX_BUILTIN_PROGRAM_BF16_NATIVE,
      TT_IREE_TTEX_DISPATCH_KIND_BF16_NATIVE)), 1);
}

static tt_iree_ttex_tensor_desc_t
BatchTensor(uint16_t binding, uint8_t layout,
            std::initializer_list<uint32_t> shape) {
  tt_iree_ttex_tensor_desc_t t = {};
  t.binding = binding;
  t.element_type = TT_IREE_TTEX_TENSOR_ELEMENT_TYPE_BF16;
  t.layout = layout;
  t.rank = shape.size();
  std::copy(shape.begin(), shape.end(), t.shape);
  std::copy(shape.begin(), shape.end(), t.storage_shape);
  return t;
}

static tt_iree_ttex_entry_point_desc_t
MakeBatchMatrix(uint8_t layout = TT_IREE_TTEX_TENSOR_LAYOUT_NCHW,
                bool conv = false) {
  tt_iree_ttex_entry_point_desc_t ep = {};
  ep.name = "bf16_batch16";
  ep.binding_count = 14;
  ep.workgroup_size[0] = ep.workgroup_size[1] = ep.workgroup_size[2] = 1;
  ep.builtin_program = TT_IREE_TTEX_BUILTIN_PROGRAM_BF16_NATIVE;
  ep.dispatch_kind = TT_IREE_TTEX_DISPATCH_KIND_BF16_NATIVE;
  auto &d = ep.bf16_native;
  d.operation_kind = conv ? TT_IREE_TTEX_DISPATCH_KIND_CONVOLUTION
                          : TT_IREE_TTEX_DISPATCH_KIND_MATMUL_LIKE;
  d.m = 16 * 5 * 7;
  d.n = 32;
  d.k = conv ? 64 * 3 * 3 : 64;
  if (layout == TT_IREE_TTEX_TENSOR_LAYOUT_NCHW) {
    d.tensors[0] =
        BatchTensor(0, layout, {16, 64, conv ? 7u : 5u, conv ? 9u : 7u});
    d.tensors[2] = BatchTensor(2, layout, {16, 32, 5, 7});
  } else {
    d.tensors[0] = BatchTensor(0, layout, {16, 64, 35});
    d.tensors[2] = BatchTensor(2, layout, {16, 32, 35});
  }
  d.tensors[1] =
      conv ? BatchTensor(1, TT_IREE_TTEX_TENSOR_LAYOUT_OIHW, {32, 64, 3, 3})
           : BatchTensor(1, TT_IREE_TTEX_TENSOR_LAYOUT_OI, {32, 64});
  return ep;
}

static tt_iree_ttex_entry_point_desc_t MakeBatchScalar(uint8_t kind,
                                                       bool flat = true) {
  auto ep = MakeBatchMatrix();
  auto &d = ep.bf16_native;
  d = {};
  d.operation_kind = kind;
  if (kind == TT_IREE_TTEX_DISPATCH_KIND_SCALAR_DIV) {
    d.tensors[0] =
        flat ? BatchTensor(0, TT_IREE_TTEX_TENSOR_LAYOUT_ROW_MAJOR, {32768})
             : BatchTensor(0, TT_IREE_TTEX_TENSOR_LAYOUT_ROW_MAJOR, {16, 2048});
    d.tensors[2] = d.tensors[0];
    d.tensors[2].binding = 2;
  } else if (kind == TT_IREE_TTEX_DISPATCH_KIND_GLOBAL_SUM) {
    d.tensors[0] =
        BatchTensor(0, TT_IREE_TTEX_TENSOR_LAYOUT_NCHW, {16, 2048, 8, 8});
    d.tensors[2] =
        flat ? BatchTensor(2, TT_IREE_TTEX_TENSOR_LAYOUT_ROW_MAJOR, {32768})
             : BatchTensor(2, TT_IREE_TTEX_TENSOR_LAYOUT_ROW_MAJOR, {16, 2048});
  } else {
    d.tensors[0] =
        BatchTensor(0, TT_IREE_TTEX_TENSOR_LAYOUT_NCHW, {16, 3, 5, 7});
    d.tensors[2] =
        BatchTensor(2, TT_IREE_TTEX_TENSOR_LAYOUT_NCHW,
                    kind == TT_IREE_TTEX_DISPATCH_KIND_MAX_POOL
                        ? std::initializer_list<uint32_t>{16, 3, 2, 3}
                        : std::initializer_list<uint32_t>{16, 3, 5, 7});
  }
  if (kind == TT_IREE_TTEX_DISPATCH_KIND_GLOBAL_SUM ||
      kind == TT_IREE_TTEX_DISPATCH_KIND_SCALAR_DIV)
    d.tensors[1] = BatchTensor(1, TT_IREE_TTEX_TENSOR_LAYOUT_ROW_MAJOR, {1});
  return ep;
}

static void ExpectBatchBuild(const tt_iree_ttex_entry_point_desc_t &ep,
                             bool valid,
                             uint32_t version = TT_IREE_TTEX_VERSION_20) {
  iree_byte_span_t data = {};
  auto status = tt_iree_build_ttex_executable_def_with_version(
      iree_allocator_system(), version, 1, &ep, &data);
  EXPECT_EQ(iree_status_code(status),
            valid ? IREE_STATUS_OK : IREE_STATUS_INVALID_ARGUMENT);
  iree_status_ignore(status);
  if (valid && data.data) {
    EXPECT_EQ(
        ttex_ns(ExecutableDef_verify_as_root(data.data, data.data_length)), 0);
    auto root = ttex_ns(ExecutableDef_as_root(data.data));
    EXPECT_EQ(ttex_ns(ExecutableDef_version(root)), version);
    auto entry = ttex_ns(
        EntryPointDef_vec_at(ttex_ns(ExecutableDef_entry_points(root)), 0));
    EXPECT_EQ(ttex_ns(EntryPointDef_spatial_plan(entry)), nullptr);
    auto tensors = ttex_ns(
        BF16NativeDef_tensors(ttex_ns(EntryPointDef_bf16_native(entry))));
    ASSERT_EQ(ttex_ns(TensorDesc_vec_len(tensors)), 14u);
    for (unsigned i = 0; i < 14; ++i) {
      auto t = ttex_ns(TensorDesc_vec_at(tensors, i));
      EXPECT_EQ(ttex_ns(TensorDesc_layout(t)),
                ep.bf16_native.tensors[i].layout);
      EXPECT_EQ(ttex_ns(TensorDesc_rank(t)), ep.bf16_native.tensors[i].rank);
    }
  } else if (!valid) {
    EXPECT_EQ(data.data, nullptr);
    EXPECT_EQ(data.data_length, 0u);
  }
  iree_allocator_free(iree_allocator_system(), data.data);
}

TEST(TtexSchemaTest, Bf16BatchV20OptInKeepsV18DefaultAndRoleSets) {
  EXPECT_EQ(TT_IREE_TTEX_CURRENT_VERSION, 18u);
  EXPECT_EQ(ttex_ns(TensorLayout_NCHW), 6);
  EXPECT_EQ(ttex_ns(TensorLayout_BATCH_CHANNEL_MAJOR), 7);
  EXPECT_EQ(ttex_ns(TensorLayout_CHW), 2);
  for (auto builtin : {TT_IREE_TTEX_BUILTIN_PROGRAM_BF16_NATIVE,
                       TT_IREE_TTEX_BUILTIN_PROGRAM_BF16_NATIVE_TENSIX}) {
    for (auto layout : {TT_IREE_TTEX_TENSOR_LAYOUT_NCHW,
                        TT_IREE_TTEX_TENSOR_LAYOUT_BATCH_CHANNEL_MAJOR}) {
      auto ep = MakeBatchMatrix(layout);
      ep.builtin_program = builtin;
      ExpectBatchBuild(ep, true);
    }
    auto ep = MakeBatchMatrix(TT_IREE_TTEX_TENSOR_LAYOUT_NCHW, true);
    ep.builtin_program = builtin;
    ExpectBatchBuild(ep, true);
    EXPECT_EQ(
        __builtin_popcount(tt_iree_ttex_required_aot_kernel_roles_for_version(
            20, builtin, TT_IREE_TTEX_DISPATCH_KIND_BF16_NATIVE)),
        builtin == TT_IREE_TTEX_BUILTIN_PROGRAM_BF16_NATIVE ? 1 : 3);
  }
}

TEST(TtexSchemaTest, Bf16BatchLayoutsRejectedBeforeV20EvenForGenericScalar) {
  for (uint32_t version : {17u, 18u}) {
    for (auto layout : {TT_IREE_TTEX_TENSOR_LAYOUT_NCHW,
                        TT_IREE_TTEX_TENSOR_LAYOUT_BATCH_CHANNEL_MAJOR}) {
      ExpectBatchBuild(MakeBatchMatrix(layout), false, version);
    }
    ExpectBatchBuild(MakeBatchScalar(TT_IREE_TTEX_DISPATCH_KIND_PADDED_COPY),
                     false, version);
    ExpectBatchBuild(
        MakeBatchScalar(TT_IREE_TTEX_DISPATCH_KIND_GLOBAL_SUM, true), false,
        version);
    auto ep = MakeBatchScalar(TT_IREE_TTEX_DISPATCH_KIND_SCALAR_DIV, true);
    // The legacy flat vector/scalar ABI remains valid without new layouts.
    ExpectBatchBuild(ep, true, version);
    ep.bf16_native.tensors[13].layout = TT_IREE_TTEX_TENSOR_LAYOUT_NCHW;
    ExpectBatchBuild(ep, false, version);
    auto status =
        tt_iree_validate_bf16_layout_version(&ep.bf16_native, version);
    EXPECT_EQ(iree_status_code(status), IREE_STATUS_INVALID_ARGUMENT);
    iree_status_ignore(status);
  }
  auto ep = MakeBatchMatrix();
  iree_byte_span_t data = {};
  auto status =
      tt_iree_build_ttex_executable_def(iree_allocator_system(), 1, &ep, &data);
  EXPECT_EQ(iree_status_code(status), IREE_STATUS_INVALID_ARGUMENT);
  iree_status_ignore(status);
}

TEST(TtexSchemaTest, Bf16BatchMatrixRejectsGeometryAndDtypeMismatches) {
  const std::function<void(tt_iree_ttex_bf16_native_desc_t &)> mutations[] = {
      [](auto &d) { --d.m; },
      [](auto &d) { ++d.n; },
      [](auto &d) { ++d.k; },
      [](auto &d) { d.m = 0; },
      [](auto &d) { d.m = UINT32_MAX; },
      [](auto &d) { d.tensors[0].rank = 0; },
      [](auto &d) { d.tensors[1].rank = 0; },
      [](auto &d) { d.tensors[2].rank = 0; },
      [](auto &d) { d.tensors[0].shape[0] = 8; },
      [](auto &d) { d.tensors[2].shape[0] = 8; },
      [](auto &d) { d.tensors[2].shape[1] = 16; },
      [](auto &d) { d.tensors[1].shape[1] = 32; },
      [](auto &d) { d.tensors[0].rank = 3; },
      [](auto &d) {
        d.tensors[1].layout = TT_IREE_TTEX_TENSOR_LAYOUT_ROW_MAJOR;
      },
      [](auto &d) { d.tensors[2].layout = TT_IREE_TTEX_TENSOR_LAYOUT_CHW; },
      [](auto &d) {
        d.tensors[0].element_type = TT_IREE_TTEX_TENSOR_ELEMENT_TYPE_F32;
      },
      [](auto &d) {
        d.tensors[1].element_type = TT_IREE_TTEX_TENSOR_ELEMENT_TYPE_F32;
      },
      [](auto &d) {
        d.tensors[2].element_type = TT_IREE_TTEX_TENSOR_ELEMENT_TYPE_F32;
      },
  };
  for (bool conv : {false, true}) {
    for (const auto &mutate : mutations) {
      auto ep = MakeBatchMatrix(TT_IREE_TTEX_TENSOR_LAYOUT_NCHW, conv);
      mutate(ep.bf16_native);
      ExpectBatchBuild(ep, false);
    }
  }
  auto ep = MakeBatchMatrix(TT_IREE_TTEX_TENSOR_LAYOUT_BATCH_CHANNEL_MAJOR);
  ep.bf16_native.tensors[2].shape[2] = 34;
  ExpectBatchBuild(ep, false);
  ep = MakeBatchMatrix(TT_IREE_TTEX_TENSOR_LAYOUT_BATCH_CHANNEL_MAJOR);
  ep.bf16_native.operation_kind = TT_IREE_TTEX_DISPATCH_KIND_CONVOLUTION;
  ExpectBatchBuild(ep, false);
}

TEST(TtexSchemaTest, Bf16BatchScalarCopyPoolGapAndDivision) {
  for (auto kind : {TT_IREE_TTEX_DISPATCH_KIND_PADDED_COPY,
                    TT_IREE_TTEX_DISPATCH_KIND_MAX_POOL,
                    TT_IREE_TTEX_DISPATCH_KIND_GLOBAL_SUM,
                    TT_IREE_TTEX_DISPATCH_KIND_SCALAR_DIV}) {
    auto ep = MakeBatchScalar(kind);
    ExpectBatchBuild(ep, true);
    ep.builtin_program = TT_IREE_TTEX_BUILTIN_PROGRAM_BF16_NATIVE_TENSIX;
    ExpectBatchBuild(ep, false);
    ep.builtin_program = TT_IREE_TTEX_BUILTIN_PROGRAM_BF16_NATIVE;
    ep.bf16_native.m = 1;
    ExpectBatchBuild(ep, false);
    ep = MakeBatchScalar(kind);
    ep.bf16_native.tensors[2].shape[0] = 15;
    ExpectBatchBuild(ep, false);
    ep = MakeBatchScalar(kind);
    ep.bf16_native.tensors[0].rank = 0;
    ExpectBatchBuild(ep, false);
    ep = MakeBatchScalar(kind);
    ep.bf16_native.tensors[2].element_type =
        TT_IREE_TTEX_TENSOR_ELEMENT_TYPE_F32;
    ExpectBatchBuild(ep, false);
  }
  for (auto kind : {TT_IREE_TTEX_DISPATCH_KIND_GLOBAL_SUM,
                    TT_IREE_TTEX_DISPATCH_KIND_SCALAR_DIV}) {
    ExpectBatchBuild(MakeBatchScalar(kind, false), false);
    ExpectBatchBuild(MakeBatchScalar(kind, true), true);
    for (const auto &mutate : std::initializer_list<
             std::function<void(tt_iree_ttex_tensor_desc_t &)>>{
             [](auto &t) { t.rank = 0; },
             [](auto &t) { t.shape[0] = t.storage_shape[0] = 2; },
             [](auto &t) {
               t.rank = 2;
               t.shape[1] = t.storage_shape[1] = 1;
             },
             [](auto &t) {
               t.element_type = TT_IREE_TTEX_TENSOR_ELEMENT_TYPE_F32;
             }}) {
      auto ep = MakeBatchScalar(kind);
      mutate(ep.bf16_native.tensors[1]);
      ExpectBatchBuild(ep, false);
    }
  }
  auto ep = MakeBatchScalar(TT_IREE_TTEX_DISPATCH_KIND_GLOBAL_SUM, true);
  ep.bf16_native.tensors[2].shape[0] = 32767;
  ExpectBatchBuild(ep, false);
  ep = MakeBatchScalar(TT_IREE_TTEX_DISPATCH_KIND_GLOBAL_SUM);
  ep.bf16_native.tensors[0].shape[2] = 7;
  ExpectBatchBuild(ep, false);
  ep = MakeBatchScalar(TT_IREE_TTEX_DISPATCH_KIND_MAX_POOL);
  ep.bf16_native.tensors[2].shape[2] = 1;
  ExpectBatchBuild(ep, false);
  ep = MakeBatchScalar(TT_IREE_TTEX_DISPATCH_KIND_PADDED_COPY);
  ep.bf16_native.tensors[1] =
      BatchTensor(1, TT_IREE_TTEX_TENSOR_LAYOUT_ROW_MAJOR, {1});
  ExpectBatchBuild(ep, false);
}

TEST(TtexSchemaTest, Bf16BatchEpilogueSlotsRetainGenericDescriptorRules) {
  auto ep = MakeBatchMatrix();
  for (unsigned i = 3; i < 14; ++i) {
    ep.bf16_native.tensors[i] =
        BatchTensor(i, TT_IREE_TTEX_TENSOR_LAYOUT_ROW_MAJOR, {5});
    ep.bf16_native.tensors[i].element_type =
        i % 2 ? TT_IREE_TTEX_TENSOR_ELEMENT_TYPE_F32
              : TT_IREE_TTEX_TENSOR_ELEMENT_TYPE_BF16;
  }
  ExpectBatchBuild(ep, true);
  const std::function<void(tt_iree_ttex_tensor_desc_t &)> mutations[] = {
      [](auto &t) { t.binding = 14; },
      [](auto &t) { t.byte_offset_constant = 1; },
      [](auto &t) { t.shape[0] = 0; },
      [](auto &t) { t.offset[0] = 5; },
      [](auto &t) { t.stride[0] = 2; },
      [](auto &t) {
        t.element_type = TT_IREE_TTEX_TENSOR_ELEMENT_TYPE_UNKNOWN;
      },
      [](auto &t) { t.rank = 5; },
      [](auto &t) { t.byte_offset = 1; },
      [](auto &t) { t.byte_offset = UINT64_MAX - 3; },
      [](auto &t) { t.layout = TT_IREE_TTEX_TENSOR_LAYOUT_NCHW; },
      [](auto &t) { t.layout = 0; },
      [](auto &t) { t.layout = 99; },
      [](auto &t) { t.layout = TT_IREE_TTEX_TENSOR_LAYOUT_OI; },
  };
  for (const auto &mutate : mutations) {
    auto invalid = ep;
    mutate(invalid.bf16_native.tensors[13]);
    ExpectBatchBuild(invalid, false);
  }
  ep.bf16_native.tensors[13] =
      BatchTensor(13, TT_IREE_TTEX_TENSOR_LAYOUT_NCHW,
                  {UINT32_MAX, UINT32_MAX, UINT32_MAX, UINT32_MAX});
  ExpectBatchBuild(ep, false);
  auto &o = ep.bf16_native.tensors[2];
  o.shape[0] = o.storage_shape[0] = UINT32_MAX;
  o.shape[2] = o.storage_shape[2] = UINT32_MAX;
  o.shape[3] = o.storage_shape[3] = UINT32_MAX;
  ExpectBatchBuild(ep, false);
}

TEST(TtexSchemaTest, Bf16BatchV20AcceptsLegacyChwAndRank2FcAnyM) {
  auto ep = MakeBatchMatrix();
  auto &d = ep.bf16_native;
  d.tensors[0] = BatchTensor(0, TT_IREE_TTEX_TENSOR_LAYOUT_CHW, {64, 5, 7});
  d.tensors[2] = BatchTensor(2, TT_IREE_TTEX_TENSOR_LAYOUT_CHW, {32, 5, 7});
  d.m = 35;
  ExpectBatchBuild(ep, true);
  for (uint32_t m : {1u, 16u, 33u}) {
    d.m = m;
    d.tensors[0] =
        BatchTensor(0, TT_IREE_TTEX_TENSOR_LAYOUT_ROW_MAJOR, {m, 64});
    d.tensors[2] =
        BatchTensor(2, TT_IREE_TTEX_TENSOR_LAYOUT_ROW_MAJOR, {m, 32});
    for (auto kind : {TT_IREE_TTEX_DISPATCH_KIND_LINEAR,
                      TT_IREE_TTEX_DISPATCH_KIND_MATMUL_LIKE}) {
      d.operation_kind = kind;
      ExpectBatchBuild(ep, true, TT_IREE_TTEX_VERSION_18);
      ExpectBatchBuild(ep, true);
      EXPECT_TRUE(tt_iree_bf16_matrix_geometry_is_valid(&d));
    }
  }
}

TEST(TtexSchemaTest, Bf16BatchV20NeverAcceptsSpatialPlan) {
  auto ep = MakeBatchMatrix();
  tt_iree_spatial_plan_t plan = {};
  ep.spatial_plan = &plan;
  ExpectBatchBuild(ep, false);
  ep.spatial_plan = nullptr;
  ExpectBatchBuild(ep, false, TT_IREE_TTEX_VERSION_19);
  ExpectBatchBuild(ep, false, 99);
}

TEST(TtexSchemaTest, Bf16BatchV20RejectsNonBf16Capability) {
  auto ep = MakeBatchMatrix();
  ep.dispatch_kind = TT_IREE_TTEX_DISPATCH_KIND_BUILTIN;
  ep.builtin_program = TT_IREE_TTEX_BUILTIN_PROGRAM_CUSTOM_SFPI_ADD;
  ExpectBatchBuild(ep, false);
  ep = MakeMatmulLikeEntry();
  ExpectBatchBuild(ep, false);
}

TEST(TtexSchemaTest, Bf16BatchV20ConvolutionRequiresSupportedSpatialGeometry) {
  auto ep = MakeBatchMatrix(TT_IREE_TTEX_TENSOR_LAYOUT_NCHW, true);
  auto &d = ep.bf16_native;
  // 3x3 stride 2 with an even-sized input, retaining the floor in valid
  // pooling.
  d.tensors[0] =
      BatchTensor(0, TT_IREE_TTEX_TENSOR_LAYOUT_NCHW, {16, 64, 12, 16});
  ExpectBatchBuild(ep, true);
  d.tensors[0] =
      BatchTensor(0, TT_IREE_TTEX_TENSOR_LAYOUT_NCHW, {16, 64, 15, 19});
  d.tensors[1] =
      BatchTensor(1, TT_IREE_TTEX_TENSOR_LAYOUT_OIHW, {32, 64, 7, 7});
  d.k = 64 * 7 * 7;
  ExpectBatchBuild(ep, true);
  const std::function<void(tt_iree_ttex_bf16_native_desc_t &)> mutations[] = {
      [](auto &d) { d.tensors[0].shape[2] = 6; },
      [](auto &d) { d.tensors[0].shape[3] = 7; },
      [](auto &d) {
        d.tensors[1].shape[2] = 5;
        d.k = 64 * 5 * 7;
      },
      [](auto &d) { d.tensors[1].shape[3] = 0; },
      [](auto &d) { d.tensors[0].shape[2] = 14; },
  };
  for (const auto &mutate : mutations) {
    auto invalid = ep;
    mutate(invalid.bf16_native);
    ExpectBatchBuild(invalid, false);
  }
}

TEST(TtexSchemaTest, BuiltinDispatchPairContract) {
  EXPECT_TRUE(tt_iree_ttex_builtin_matches_dispatch_kind(
      TT_IREE_TTEX_BUILTIN_PROGRAM_CUSTOM_SFPI_ADD,
      TT_IREE_TTEX_DISPATCH_KIND_BUILTIN));
  EXPECT_TRUE(tt_iree_ttex_builtin_matches_dispatch_kind(
      TT_IREE_TTEX_BUILTIN_PROGRAM_BF16_MATMUL_TILED,
      TT_IREE_TTEX_DISPATCH_KIND_BUILTIN));
  EXPECT_TRUE(tt_iree_ttex_builtin_matches_dispatch_kind(
      TT_IREE_TTEX_BUILTIN_PROGRAM_F32_MATMUL_LIKE,
      TT_IREE_TTEX_DISPATCH_KIND_MATMUL_LIKE));
  EXPECT_TRUE(tt_iree_ttex_builtin_matches_dispatch_kind(
      TT_IREE_TTEX_BUILTIN_PROGRAM_F32_CONV2D,
      TT_IREE_TTEX_DISPATCH_KIND_CONVOLUTION));
  EXPECT_TRUE(tt_iree_ttex_builtin_matches_dispatch_kind(
      TT_IREE_TTEX_BUILTIN_PROGRAM_F32_LINEAR,
      TT_IREE_TTEX_DISPATCH_KIND_LINEAR));
  EXPECT_FALSE(tt_iree_ttex_builtin_matches_dispatch_kind(
      TT_IREE_TTEX_BUILTIN_PROGRAM_F32_CONV2D,
      TT_IREE_TTEX_DISPATCH_KIND_MATMUL_LIKE));
  EXPECT_FALSE(tt_iree_ttex_builtin_matches_dispatch_kind(
      /*builtin_program=*/999, TT_IREE_TTEX_DISPATCH_KIND_BUILTIN));
  EXPECT_FALSE(tt_iree_ttex_builtin_matches_dispatch_kind(
      TT_IREE_TTEX_BUILTIN_PROGRAM_CUSTOM_SFPI_ADD,
      /*dispatch_kind=*/255));
}

TEST(TtexSchemaTest, RejectsMismatchedBuiltinDispatchKind) {
  iree_allocator_t alloc = iree_allocator_system();
  tt_iree_ttex_entry_point_desc_t ep = MakeMatmulLikeEntry();
  ep.builtin_program = TT_IREE_TTEX_BUILTIN_PROGRAM_F32_CONV2D;

  iree_byte_span_t data = {};
  iree_status_t status =
      tt_iree_build_ttex_executable_def(alloc, 1, &ep, &data);
  EXPECT_EQ(iree_status_code(status), IREE_STATUS_INVALID_ARGUMENT);
  iree_status_ignore(status);
  EXPECT_EQ(data.data, nullptr);
}

TEST(TtexSchemaTest, BuildExplicitVersion) {
  iree_allocator_t alloc = iree_allocator_system();

  tt_iree_ttex_entry_point_desc_t ep = {};
  ep.name = "future_version";
  ep.binding_count = 3;
  ep.workgroup_size[0] = 1;
  ep.workgroup_size[1] = 1;
  ep.workgroup_size[2] = 1;
  ep.builtin_program = TT_IREE_TTEX_BUILTIN_PROGRAM_BF16_MATMUL_TILED;

  constexpr uint32_t kFutureVersion = 99;
  iree_byte_span_t data = {0};
  IREE_ASSERT_OK(tt_iree_build_ttex_executable_def_with_version(
      alloc, kFutureVersion, 1, &ep, &data));

  ASSERT_EQ(ttex_ns(ExecutableDef_verify_as_root(data.data, data.data_length)),
            0);
  ttex_ns(ExecutableDef_table_t) def =
      ttex_ns(ExecutableDef_as_root(data.data));
  ASSERT_EQ(ttex_ns(ExecutableDef_version(def)), kFutureVersion);

  iree_allocator_free(alloc, data.data);
}

TEST(TtexSchemaTest, BuildVerifyMatmulLikeV2) {
  iree_allocator_t alloc = iree_allocator_system();
  tt_iree_ttex_entry_point_desc_t ep = MakeMatmulLikeEntry();

  iree_byte_span_t data = {0};
  IREE_ASSERT_OK(tt_iree_build_ttex_executable_def_with_version(
      alloc, TT_IREE_TTEX_VERSION_2, 1, &ep, &data));
  ASSERT_EQ(ttex_ns(ExecutableDef_verify_as_root(data.data, data.data_length)),
            0);

  ttex_ns(ExecutableDef_table_t) def =
      ttex_ns(ExecutableDef_as_root(data.data));
  ASSERT_EQ(ttex_ns(ExecutableDef_version(def)), TT_IREE_TTEX_VERSION_2);
  ttex_ns(EntryPointDef_table_t) parsed_ep = ttex_ns(
      EntryPointDef_vec_at(ttex_ns(ExecutableDef_entry_points(def)), 0));
  ASSERT_EQ(ttex_ns(EntryPointDef_dispatch_kind(parsed_ep)),
            TT_IREE_TTEX_DISPATCH_KIND_MATMUL_LIKE);
  ttex_ns(MatmulLikeDef_table_t) matmul =
      ttex_ns(EntryPointDef_matmul_like(parsed_ep));
  ASSERT_NE(matmul, nullptr);
  EXPECT_EQ(ttex_ns(MatmulLikeDef_m(matmul)), 64u);
  EXPECT_EQ(ttex_ns(MatmulLikeDef_n(matmul)), 64u);
  EXPECT_EQ(ttex_ns(MatmulLikeDef_k(matmul)), 64u);
  EXPECT_FLOAT_EQ(ttex_ns(MatmulLikeDef_epilogue_scale(matmul)), 0.5f);
  EXPECT_FLOAT_EQ(ttex_ns(MatmulLikeDef_epilogue_bias(matmul)), 0.1875f);

  ttex_ns(TensorDesc_table_t) output = ttex_ns(MatmulLikeDef_output(matmul));
  ASSERT_NE(output, nullptr);
  EXPECT_EQ(ttex_ns(TensorDesc_binding(output)), 2u);
  EXPECT_EQ(ttex_ns(TensorDesc_rank(output)), 3u);
  EXPECT_EQ(ttex_ns(TensorDesc_storage_shape_1(output)), 10u);
  EXPECT_EQ(ttex_ns(TensorDesc_offset_1(output)), 1u);
  EXPECT_EQ(ttex_ns(TensorDesc_offset_2(output)), 1u);

  iree_allocator_free(alloc, data.data);
}

TEST(TtexSchemaTest, BuildVerifyConv2DV2) {
  iree_allocator_t alloc = iree_allocator_system();
  tt_iree_ttex_entry_point_desc_t ep = MakeConv2DEntry();

  iree_byte_span_t data = {0};
  IREE_ASSERT_OK(tt_iree_build_ttex_executable_def_with_version(
      alloc, TT_IREE_TTEX_VERSION_2, 1, &ep, &data));
  ASSERT_EQ(ttex_ns(ExecutableDef_verify_as_root(data.data, data.data_length)),
            0);

  ttex_ns(ExecutableDef_table_t) def =
      ttex_ns(ExecutableDef_as_root(data.data));
  ttex_ns(EntryPointDef_table_t) parsed_ep = ttex_ns(
      EntryPointDef_vec_at(ttex_ns(ExecutableDef_entry_points(def)), 0));
  ASSERT_EQ(ttex_ns(EntryPointDef_dispatch_kind(parsed_ep)),
            TT_IREE_TTEX_DISPATCH_KIND_CONVOLUTION);
  ttex_ns(Conv2DDef_table_t) conv2d = ttex_ns(EntryPointDef_conv2d(parsed_ep));
  ASSERT_NE(conv2d, nullptr);
  EXPECT_EQ(ttex_ns(Conv2DDef_kernel_h(conv2d)), 3u);
  EXPECT_EQ(ttex_ns(Conv2DDef_kernel_w(conv2d)), 3u);
  EXPECT_EQ(ttex_ns(Conv2DDef_groups(conv2d)), 1u);
  EXPECT_FLOAT_EQ(ttex_ns(Conv2DDef_epilogue_scale(conv2d)), 0.25f);
  EXPECT_FLOAT_EQ(ttex_ns(Conv2DDef_epilogue_bias(conv2d)), 0.125f);

  ttex_ns(TensorDesc_table_t) filter = ttex_ns(Conv2DDef_filter(conv2d));
  ttex_ns(TensorDesc_table_t) output = ttex_ns(Conv2DDef_output(conv2d));
  ASSERT_NE(filter, nullptr);
  ASSERT_NE(output, nullptr);
  EXPECT_EQ(ttex_ns(TensorDesc_layout(filter)),
            TT_IREE_TTEX_TENSOR_LAYOUT_OIHW);
  EXPECT_EQ(ttex_ns(TensorDesc_rank(filter)), 4u);
  EXPECT_EQ(ttex_ns(TensorDesc_shape_2(filter)), 3u);
  EXPECT_EQ(ttex_ns(TensorDesc_shape_1(output)), 8u);

  iree_allocator_free(alloc, data.data);
}

TEST(TtexSchemaTest, BuildVerifyPerChannelMatmulLikeV7) {
  iree_allocator_t alloc = iree_allocator_system();
  tt_iree_ttex_entry_point_desc_t ep = MakePerChannelMatmulLikeEntry();

  iree_byte_span_t data = {0};
  IREE_ASSERT_OK(tt_iree_build_ttex_executable_def_with_version(
      alloc, TT_IREE_TTEX_VERSION_7, 1, &ep, &data));
  ASSERT_EQ(ttex_ns(ExecutableDef_verify_as_root(data.data, data.data_length)),
            0);

  ttex_ns(ExecutableDef_table_t) def =
      ttex_ns(ExecutableDef_as_root(data.data));
  ttex_ns(EntryPointDef_table_t) parsed_ep = ttex_ns(
      EntryPointDef_vec_at(ttex_ns(ExecutableDef_entry_points(def)), 0));
  ttex_ns(MatmulLikeDef_table_t) matmul =
      ttex_ns(EntryPointDef_matmul_like(parsed_ep));
  ASSERT_NE(matmul, nullptr);
  flatbuffers_float_vec_t scales =
      ttex_ns(MatmulLikeDef_epilogue_scales(matmul));
  flatbuffers_float_vec_t biases =
      ttex_ns(MatmulLikeDef_epilogue_biases(matmul));
  ASSERT_EQ(flatbuffers_float_vec_len(scales), 64u);
  ASSERT_EQ(flatbuffers_float_vec_len(biases), 64u);
  EXPECT_FLOAT_EQ(flatbuffers_float_vec_at(scales, 1), 0.25f);
  EXPECT_FLOAT_EQ(flatbuffers_float_vec_at(biases, 63), 63.0f);

  iree_allocator_free(alloc, data.data);
}

TEST(TtexSchemaTest, RejectsPerChannelMatmulLikeBeforeV7) {
  iree_allocator_t alloc = iree_allocator_system();
  tt_iree_ttex_entry_point_desc_t ep = MakePerChannelMatmulLikeEntry();
  iree_byte_span_t data = {0};
  iree_status_t status = tt_iree_build_ttex_executable_def_with_version(
      alloc, TT_IREE_TTEX_VERSION_6, 1, &ep, &data);
  EXPECT_EQ(iree_status_code(status), IREE_STATUS_INVALID_ARGUMENT);
  iree_status_ignore(status);
  EXPECT_EQ(data.data, nullptr);
}

TEST(TtexSchemaTest, BuildVerifyPerChannelConv2DV5) {
  iree_allocator_t alloc = iree_allocator_system();
  tt_iree_ttex_entry_point_desc_t ep = MakeConv2DEntry();
  std::array<float, 64> scales = {};
  std::array<float, 64> biases = {};
  for (size_t i = 0; i < scales.size(); ++i) {
    scales[i] = 0.25f;
    biases[i] = static_cast<float>(i);
  }
  ep.conv2d.epilogue_flags |= TT_IREE_TTEX_EPILOGUE_PER_CHANNEL;
  ep.conv2d.epilogue_channel_count = scales.size();
  ep.conv2d.epilogue_scales = scales.data();
  ep.conv2d.epilogue_biases = biases.data();

  iree_byte_span_t data = {0};
  IREE_ASSERT_OK(tt_iree_build_ttex_executable_def_with_version(
      alloc, TT_IREE_TTEX_VERSION_5, 1, &ep, &data));
  ASSERT_EQ(ttex_ns(ExecutableDef_verify_as_root(data.data, data.data_length)),
            0);

  ttex_ns(ExecutableDef_table_t) def =
      ttex_ns(ExecutableDef_as_root(data.data));
  ttex_ns(EntryPointDef_table_t) parsed_ep = ttex_ns(
      EntryPointDef_vec_at(ttex_ns(ExecutableDef_entry_points(def)), 0));
  ttex_ns(Conv2DDef_table_t) conv2d = ttex_ns(EntryPointDef_conv2d(parsed_ep));
  ASSERT_NE(conv2d, nullptr);
  flatbuffers_float_vec_t parsed_scales =
      ttex_ns(Conv2DDef_epilogue_scales(conv2d));
  flatbuffers_float_vec_t parsed_biases =
      ttex_ns(Conv2DDef_epilogue_biases(conv2d));
  ASSERT_EQ(flatbuffers_float_vec_len(parsed_scales), 64u);
  ASSERT_EQ(flatbuffers_float_vec_len(parsed_biases), 64u);
  EXPECT_FLOAT_EQ(flatbuffers_float_vec_at(parsed_scales, 63), 0.25f);
  EXPECT_FLOAT_EQ(flatbuffers_float_vec_at(parsed_biases, 63), 63.0f);

  iree_allocator_free(alloc, data.data);
}

TEST(TtexSchemaTest, RejectsPerChannelConvBeforeV5) {
  iree_allocator_t alloc = iree_allocator_system();
  tt_iree_ttex_entry_point_desc_t ep = MakeConv2DEntry();
  std::array<float, 64> scales = {};
  std::array<float, 64> biases = {};
  ep.conv2d.epilogue_flags |= TT_IREE_TTEX_EPILOGUE_PER_CHANNEL;
  ep.conv2d.epilogue_channel_count = scales.size();
  ep.conv2d.epilogue_scales = scales.data();
  ep.conv2d.epilogue_biases = biases.data();

  iree_byte_span_t data = {0};
  iree_status_t status = tt_iree_build_ttex_executable_def_with_version(
      alloc, TT_IREE_TTEX_VERSION_4, 1, &ep, &data);
  EXPECT_EQ(iree_status_code(status), IREE_STATUS_INVALID_ARGUMENT);
  iree_status_ignore(status);
  EXPECT_EQ(data.data, nullptr);
}

TEST(TtexSchemaTest, BuildVerifyPaddedCopyV4) {
  iree_allocator_t alloc = iree_allocator_system();
  tt_iree_ttex_entry_point_desc_t ep = MakePaddedCopyEntry();

  iree_byte_span_t data = {0};
  IREE_ASSERT_OK(tt_iree_build_ttex_executable_def_with_version(
      alloc, TT_IREE_TTEX_VERSION_4, 1, &ep, &data));
  ASSERT_EQ(ttex_ns(ExecutableDef_verify_as_root(data.data, data.data_length)),
            0);

  ttex_ns(ExecutableDef_table_t) def =
      ttex_ns(ExecutableDef_as_root(data.data));
  ttex_ns(EntryPointDef_table_t) parsed_ep = ttex_ns(
      EntryPointDef_vec_at(ttex_ns(ExecutableDef_entry_points(def)), 0));
  EXPECT_EQ(ttex_ns(EntryPointDef_dispatch_kind(parsed_ep)),
            TT_IREE_TTEX_DISPATCH_KIND_PADDED_COPY);
  ttex_ns(PaddedCopyDef_table_t) copy =
      ttex_ns(EntryPointDef_padded_copy(parsed_ep));
  ASSERT_NE(copy, nullptr);
  ttex_ns(TensorDesc_table_t) source = ttex_ns(PaddedCopyDef_source(copy));
  ttex_ns(TensorDesc_table_t) destination =
      ttex_ns(PaddedCopyDef_destination(copy));
  ASSERT_NE(source, nullptr);
  ASSERT_NE(destination, nullptr);
  EXPECT_EQ(ttex_ns(TensorDesc_shape_1(source)), 224u);
  EXPECT_EQ(ttex_ns(TensorDesc_storage_shape_1(destination)), 230u);
  EXPECT_EQ(ttex_ns(TensorDesc_offset_1(destination)), 3u);
  EXPECT_EQ(ttex_ns(TensorDesc_offset_2(destination)), 3u);

  iree_allocator_free(alloc, data.data);
}

TEST(TtexSchemaTest, RejectsPaddedCopyBeforeV4) {
  iree_allocator_t alloc = iree_allocator_system();
  tt_iree_ttex_entry_point_desc_t ep = MakePaddedCopyEntry();
  iree_byte_span_t data = {0};
  iree_status_t status = tt_iree_build_ttex_executable_def_with_version(
      alloc, TT_IREE_TTEX_VERSION_3, 1, &ep, &data);
  EXPECT_EQ(iree_status_code(status), IREE_STATUS_INVALID_ARGUMENT);
  iree_status_ignore(status);
}

TEST(TtexSchemaTest, BuildVerifyMaxPoolV6) {
  iree_allocator_t alloc = iree_allocator_system();
  tt_iree_ttex_entry_point_desc_t ep = MakeMaxPoolEntry();

  iree_byte_span_t data = {0};
  IREE_ASSERT_OK(tt_iree_build_ttex_executable_def_with_version(
      alloc, TT_IREE_TTEX_VERSION_6, 1, &ep, &data));
  ASSERT_EQ(ttex_ns(ExecutableDef_verify_as_root(data.data, data.data_length)),
            0);

  ttex_ns(ExecutableDef_table_t) def =
      ttex_ns(ExecutableDef_as_root(data.data));
  ttex_ns(EntryPointDef_table_t) parsed_ep = ttex_ns(
      EntryPointDef_vec_at(ttex_ns(ExecutableDef_entry_points(def)), 0));
  EXPECT_EQ(ttex_ns(EntryPointDef_dispatch_kind(parsed_ep)),
            TT_IREE_TTEX_DISPATCH_KIND_MAX_POOL);
  ttex_ns(MaxPool2DDef_table_t) pool =
      ttex_ns(EntryPointDef_max_pool(parsed_ep));
  ASSERT_NE(pool, nullptr);
  EXPECT_EQ(ttex_ns(MaxPool2DDef_kernel_h(pool)), 3u);
  EXPECT_EQ(ttex_ns(MaxPool2DDef_stride_h(pool)), 2u);
  EXPECT_EQ(ttex_ns(TensorDesc_shape_1(ttex_ns(MaxPool2DDef_input(pool)))), 6u);
  EXPECT_EQ(ttex_ns(TensorDesc_shape_1(ttex_ns(MaxPool2DDef_output(pool)))),
            2u);

  iree_allocator_free(alloc, data.data);
}

TEST(TtexSchemaTest, RejectsMaxPoolBeforeV6) {
  iree_allocator_t alloc = iree_allocator_system();
  tt_iree_ttex_entry_point_desc_t ep = MakeMaxPoolEntry();
  iree_byte_span_t data = {0};
  iree_status_t status = tt_iree_build_ttex_executable_def_with_version(
      alloc, TT_IREE_TTEX_VERSION_5, 1, &ep, &data);
  EXPECT_EQ(iree_status_code(status), IREE_STATUS_INVALID_ARGUMENT);
  iree_status_ignore(status);
  EXPECT_EQ(data.data, nullptr);
}

TEST(TtexSchemaTest, BuildVerifyResidualMatmulLikeV2) {
  iree_allocator_t alloc = iree_allocator_system();
  tt_iree_ttex_entry_point_desc_t ep = MakeResidualMatmulLikeEntry();

  iree_byte_span_t data = {0};
  IREE_ASSERT_OK(tt_iree_build_ttex_executable_def_with_version(
      alloc, TT_IREE_TTEX_VERSION_2, 1, &ep, &data));
  ASSERT_EQ(ttex_ns(ExecutableDef_verify_as_root(data.data, data.data_length)),
            0);

  ttex_ns(ExecutableDef_table_t) def =
      ttex_ns(ExecutableDef_as_root(data.data));
  ttex_ns(EntryPointDef_table_t) parsed_ep = ttex_ns(
      EntryPointDef_vec_at(ttex_ns(ExecutableDef_entry_points(def)), 0));
  ttex_ns(MatmulLikeDef_table_t) matmul =
      ttex_ns(EntryPointDef_matmul_like(parsed_ep));
  ASSERT_NE(matmul, nullptr);
  EXPECT_EQ(ttex_ns(MatmulLikeDef_epilogue_flags(matmul)),
            TT_IREE_TTEX_EPILOGUE_AFFINE | TT_IREE_TTEX_EPILOGUE_RESIDUAL |
                TT_IREE_TTEX_EPILOGUE_RELU);
  EXPECT_FLOAT_EQ(ttex_ns(MatmulLikeDef_residual_scale(matmul)), 0.25f);
  EXPECT_FLOAT_EQ(ttex_ns(MatmulLikeDef_residual_bias(matmul)), 0.125f);

  ttex_ns(TensorDesc_table_t) lhs = ttex_ns(MatmulLikeDef_lhs(matmul));
  ttex_ns(TensorDesc_table_t) residual =
      ttex_ns(MatmulLikeDef_residual(matmul));
  ASSERT_NE(lhs, nullptr);
  ASSERT_NE(residual, nullptr);
  EXPECT_EQ(ttex_ns(TensorDesc_layout(lhs)),
            TT_IREE_TTEX_TENSOR_LAYOUT_CHANNEL_MAJOR);
  EXPECT_EQ(ttex_ns(TensorDesc_byte_offset(lhs)), 25600u);
  EXPECT_EQ(ttex_ns(TensorDesc_binding(residual)), 0u);
  EXPECT_EQ(ttex_ns(TensorDesc_byte_offset(residual)), 41984u);

  iree_allocator_free(alloc, data.data);
}

TEST(TtexSchemaTest, BuildVerifyAotKernelBundleV3) {
  iree_allocator_t alloc = iree_allocator_system();
  tt_iree_ttex_entry_point_desc_t ep = MakeMatmulLikeEntry();

  tt_iree_ttex_kernel_source_desc_t kernels[] = {
      {TT_IREE_TTEX_KERNEL_ROLE_READER, "void kernel_main() {}"},
      {TT_IREE_TTEX_KERNEL_ROLE_WRITER, "void kernel_main() { int x = 1; }"},
      {TT_IREE_TTEX_KERNEL_ROLE_COMPUTE, "void kernel_main() { int x = 2; }"},
  };
  const uint8_t elf_data[] = {0x7f, 'E', 'L', 'F'};
  tt_iree_ttex_artifact_file_desc_t artifacts[] = {
      {"Kernel_Source_Code/123456/brisc/kernel.elf", elf_data,
       sizeof(elf_data)}};
  tt_iree_ttex_aot_kernel_bundle_desc_t bundle = {
      .target_arch = TT_IREE_TTEX_TARGET_ARCH_BLACKHOLE,
      .kernel_count = IREE_ARRAYSIZE(kernels),
      .kernels = kernels,
      .artifact_count = IREE_ARRAYSIZE(artifacts),
      .artifacts = artifacts,
  };
  ep.aot_kernel_bundle = &bundle;

  iree_byte_span_t data = {0};
  IREE_ASSERT_OK(tt_iree_build_ttex_executable_def_with_version(
      alloc, TT_IREE_TTEX_VERSION_3, 1, &ep, &data));
  ASSERT_EQ(ttex_ns(ExecutableDef_verify_as_root(data.data, data.data_length)),
            0);

  ttex_ns(ExecutableDef_table_t) def =
      ttex_ns(ExecutableDef_as_root(data.data));
  EXPECT_EQ(ttex_ns(ExecutableDef_version(def)), TT_IREE_TTEX_VERSION_3);
  ttex_ns(EntryPointDef_table_t) parsed_ep = ttex_ns(
      EntryPointDef_vec_at(ttex_ns(ExecutableDef_entry_points(def)), 0));
  ttex_ns(TTMetalKernelBundleDef_table_t) parsed_bundle =
      ttex_ns(EntryPointDef_aot_kernel_bundle(parsed_ep));
  ASSERT_NE(parsed_bundle, nullptr);
  EXPECT_EQ(ttex_ns(TTMetalKernelBundleDef_target_arch(parsed_bundle)),
            TT_IREE_TTEX_TARGET_ARCH_BLACKHOLE);
  EXPECT_EQ(ttex_ns(KernelSourceDef_vec_len(
                ttex_ns(TTMetalKernelBundleDef_kernels(parsed_bundle)))),
            TT_IREE_TTEX_AOT_KERNEL_COUNT);
  ttex_ns(ArtifactFileDef_vec_t) parsed_artifacts =
      ttex_ns(TTMetalKernelBundleDef_artifacts(parsed_bundle));
  ASSERT_EQ(ttex_ns(ArtifactFileDef_vec_len(parsed_artifacts)), 1u);
  ttex_ns(ArtifactFileDef_table_t) parsed_artifact =
      ttex_ns(ArtifactFileDef_vec_at(parsed_artifacts, 0));
  EXPECT_STREQ(ttex_ns(ArtifactFileDef_path(parsed_artifact)),
               "Kernel_Source_Code/123456/brisc/kernel.elf");
  EXPECT_EQ(
      flatbuffers_uint8_vec_len(ttex_ns(ArtifactFileDef_data(parsed_artifact))),
      sizeof(elf_data));

  iree_allocator_free(alloc, data.data);
}

TEST(TtexSchemaTest, SharesExactArtifactPayloadsAcrossPathsAndExports) {
  auto ep = MakeMatmulLikeEntry();
  tt_iree_ttex_kernel_source_desc_t kernels[] = {
      {TT_IREE_TTEX_KERNEL_ROLE_READER, "reader"},
      {TT_IREE_TTEX_KERNEL_ROLE_WRITER, "writer"},
      {TT_IREE_TTEX_KERNEL_ROLE_COMPUTE, "compute"}};
  const uint8_t first[] = {0x7f, 'E', 'L', 'F', 0, 1};
  const uint8_t equal[] = {0x7f, 'E', 'L', 'F', 0, 1};
  const uint8_t different[] = {0x7f, 'E', 'L', 'F', 0, 2};
  tt_iree_ttex_artifact_file_desc_t artifacts[] = {
      {"kernel/one/brisc.elf", first, sizeof(first)},
      {"kernel/two/brisc.elf", equal, sizeof(equal)},
      {"kernel/three/brisc.elf", different, sizeof(different)},
      {"kernel/four/brisc.elf", first, sizeof(first) - 1}};
  tt_iree_ttex_aot_kernel_bundle_desc_t bundle = {
      TT_IREE_TTEX_TARGET_ARCH_BLACKHOLE, IREE_ARRAYSIZE(kernels), kernels,
      IREE_ARRAYSIZE(artifacts), artifacts};
  ep.aot_kernel_bundle = &bundle;
  tt_iree_ttex_entry_point_desc_t entries[] = {ep, ep};
  entries[1].name = "second_export";
  iree_byte_span_t data = {};
  auto alloc = iree_allocator_system();
  IREE_ASSERT_OK(tt_iree_build_ttex_executable_def_with_version(
      alloc, TT_IREE_TTEX_VERSION_3, IREE_ARRAYSIZE(entries), entries, &data));
  ASSERT_EQ(ttex_ns(ExecutableDef_verify_as_root(data.data, data.data_length)),
            0);
  auto parsed_entries = ttex_ns(
      ExecutableDef_entry_points(ttex_ns(ExecutableDef_as_root(data.data))));
  auto parsed_artifacts = [&](size_t i) {
    return ttex_ns(TTMetalKernelBundleDef_artifacts(
        ttex_ns(EntryPointDef_aot_kernel_bundle(
            ttex_ns(EntryPointDef_vec_at(parsed_entries, i))))));
  };
  auto payload = [&](size_t ep_index, size_t artifact_index) {
    return ttex_ns(ArtifactFileDef_data(ttex_ns(
        ArtifactFileDef_vec_at(parsed_artifacts(ep_index), artifact_index))));
  };
  EXPECT_EQ(payload(0, 0), payload(0, 1));
  EXPECT_EQ(payload(0, 0), payload(1, 0));
  EXPECT_NE(payload(0, 0), payload(0, 2));
  EXPECT_NE(payload(0, 0), payload(0, 3));
  EXPECT_EQ(std::memcmp(payload(0, 2), different, sizeof(different)), 0);
  EXPECT_EQ(flatbuffers_uint8_vec_len(payload(0, 3)), sizeof(first) - 1);
  EXPECT_STREQ(ttex_ns(ArtifactFileDef_path(
                   ttex_ns(ArtifactFileDef_vec_at(parsed_artifacts(0), 1)))),
               "kernel/two/brisc.elf");
  iree_allocator_free(alloc, data.data);
}

TEST(TtexSchemaTest, BuildVerifySingleKernelPaddedCopyAotV9) {
  iree_allocator_t alloc = iree_allocator_system();
  tt_iree_ttex_entry_point_desc_t ep = MakePaddedCopyEntry();
  tt_iree_ttex_kernel_source_desc_t kernel = {
      TT_IREE_TTEX_KERNEL_ROLE_READER, "void kernel_main() {}"};
  const uint8_t elf_data[] = {0x7f, 'E', 'L', 'F'};
  tt_iree_ttex_artifact_file_desc_t artifact = {
      "Kernel_Source_Code/padded_copy/brisc/kernel.elf", elf_data,
      sizeof(elf_data)};
  tt_iree_ttex_aot_kernel_bundle_desc_t bundle = {
      .target_arch = TT_IREE_TTEX_TARGET_ARCH_BLACKHOLE,
      .kernel_count = 1,
      .kernels = &kernel,
      .artifact_count = 1,
      .artifacts = &artifact,
  };
  ep.aot_kernel_bundle = &bundle;

  iree_byte_span_t data = {};
  IREE_ASSERT_OK(tt_iree_build_ttex_executable_def_with_version(
      alloc, TT_IREE_TTEX_VERSION_9, 1, &ep, &data));
  ASSERT_EQ(ttex_ns(ExecutableDef_verify_as_root(data.data, data.data_length)),
            0);
  auto parsed_ep = ttex_ns(EntryPointDef_vec_at(
      ttex_ns(ExecutableDef_entry_points(
          ttex_ns(ExecutableDef_as_root(data.data)))),
      0));
  auto parsed_bundle = ttex_ns(EntryPointDef_aot_kernel_bundle(parsed_ep));
  ASSERT_NE(parsed_bundle, nullptr);
  EXPECT_EQ(ttex_ns(KernelSourceDef_vec_len(
                ttex_ns(TTMetalKernelBundleDef_kernels(parsed_bundle)))),
            1u);
  iree_allocator_free(alloc, data.data);
}

TEST(TtexSchemaTest, RejectsSingleKernelPaddedCopyAotBeforeV9) {
  iree_allocator_t alloc = iree_allocator_system();
  tt_iree_ttex_entry_point_desc_t ep = MakePaddedCopyEntry();
  tt_iree_ttex_kernel_source_desc_t kernel = {
      TT_IREE_TTEX_KERNEL_ROLE_READER, "void kernel_main() {}"};
  const uint8_t artifact_data[] = {1};
  tt_iree_ttex_artifact_file_desc_t artifact = {
      "Kernel_Source_Code/padded_copy/brisc/kernel.elf", artifact_data,
      sizeof(artifact_data)};
  tt_iree_ttex_aot_kernel_bundle_desc_t bundle = {
      .target_arch = TT_IREE_TTEX_TARGET_ARCH_BLACKHOLE,
      .kernel_count = 1,
      .kernels = &kernel,
      .artifact_count = 1,
      .artifacts = &artifact,
  };
  ep.aot_kernel_bundle = &bundle;

  iree_byte_span_t data = {};
  iree_status_t status = tt_iree_build_ttex_executable_def_with_version(
      alloc, TT_IREE_TTEX_VERSION_8, 1, &ep, &data);
  EXPECT_EQ(iree_status_code(status), IREE_STATUS_INVALID_ARGUMENT);
  iree_status_ignore(status);
}

TEST(TtexSchemaTest, RejectsAotBundleBeforeV3) {
  iree_allocator_t alloc = iree_allocator_system();
  tt_iree_ttex_entry_point_desc_t ep = MakeMatmulLikeEntry();
  tt_iree_ttex_kernel_source_desc_t kernels[] = {
      {TT_IREE_TTEX_KERNEL_ROLE_READER, "reader"},
      {TT_IREE_TTEX_KERNEL_ROLE_WRITER, "writer"},
      {TT_IREE_TTEX_KERNEL_ROLE_COMPUTE, "compute"},
  };
  const uint8_t artifact_data[] = {1};
  tt_iree_ttex_artifact_file_desc_t artifacts[] = {
      {"kernel/hash/brisc/kernel.elf", artifact_data, sizeof(artifact_data)}};
  tt_iree_ttex_aot_kernel_bundle_desc_t bundle = {
      TT_IREE_TTEX_TARGET_ARCH_BLACKHOLE, IREE_ARRAYSIZE(kernels), kernels,
      IREE_ARRAYSIZE(artifacts), artifacts};
  ep.aot_kernel_bundle = &bundle;

  iree_byte_span_t data = {0};
  iree_status_t status = tt_iree_build_ttex_executable_def_with_version(
      alloc, TT_IREE_TTEX_VERSION_2, 1, &ep, &data);
  EXPECT_EQ(iree_status_code(status), IREE_STATUS_INVALID_ARGUMENT);
  iree_status_ignore(status);
}

TEST(TtexSchemaTest, RejectsUnknownAotTargetArch) {
  iree_allocator_t alloc = iree_allocator_system();
  tt_iree_ttex_entry_point_desc_t ep = MakeMatmulLikeEntry();
  tt_iree_ttex_kernel_source_desc_t kernels[] = {
      {TT_IREE_TTEX_KERNEL_ROLE_READER, "reader"},
      {TT_IREE_TTEX_KERNEL_ROLE_WRITER, "writer"},
      {TT_IREE_TTEX_KERNEL_ROLE_COMPUTE, "compute"},
  };
  const uint8_t artifact_data[] = {1};
  tt_iree_ttex_artifact_file_desc_t artifacts[] = {
      {"kernel/hash/brisc/kernel.elf", artifact_data, sizeof(artifact_data)}};
  tt_iree_ttex_aot_kernel_bundle_desc_t bundle = {
      TT_IREE_TTEX_TARGET_ARCH_UNKNOWN, IREE_ARRAYSIZE(kernels), kernels,
      IREE_ARRAYSIZE(artifacts), artifacts};
  ep.aot_kernel_bundle = &bundle;

  iree_byte_span_t data = {0};
  iree_status_t status =
      tt_iree_build_ttex_executable_def(alloc, 1, &ep, &data);
  EXPECT_EQ(iree_status_code(status), IREE_STATUS_INVALID_ARGUMENT);
  iree_status_ignore(status);
}

TEST(TtexSchemaTest, RejectsUnsafeAotArtifactPath) {
  iree_allocator_t alloc = iree_allocator_system();
  tt_iree_ttex_entry_point_desc_t ep = MakeMatmulLikeEntry();
  tt_iree_ttex_kernel_source_desc_t kernels[] = {
      {TT_IREE_TTEX_KERNEL_ROLE_READER, "reader"},
      {TT_IREE_TTEX_KERNEL_ROLE_WRITER, "writer"},
      {TT_IREE_TTEX_KERNEL_ROLE_COMPUTE, "compute"},
  };
  const uint8_t artifact_data[] = {1};
  tt_iree_ttex_artifact_file_desc_t artifacts[] = {
      {"../escape.elf", artifact_data, sizeof(artifact_data)}};
  tt_iree_ttex_aot_kernel_bundle_desc_t bundle = {
      TT_IREE_TTEX_TARGET_ARCH_BLACKHOLE, IREE_ARRAYSIZE(kernels), kernels,
      IREE_ARRAYSIZE(artifacts), artifacts};
  ep.aot_kernel_bundle = &bundle;

  iree_byte_span_t data = {0};
  iree_status_t status =
      tt_iree_build_ttex_executable_def(alloc, 1, &ep, &data);
  EXPECT_EQ(iree_status_code(status), IREE_STATUS_INVALID_ARGUMENT);
  iree_status_ignore(status);
}

TEST(TtexSchemaTest, VersionOneBuiltinStillVerifies) {
  iree_allocator_t alloc = iree_allocator_system();
  tt_iree_ttex_entry_point_desc_t ep = {};
  ep.name = "v1_matmul";
  ep.binding_count = 3;
  ep.workgroup_size[0] = 1;
  ep.workgroup_size[1] = 1;
  ep.workgroup_size[2] = 1;
  ep.builtin_program = TT_IREE_TTEX_BUILTIN_PROGRAM_BF16_MATMUL_TILED;

  iree_byte_span_t data = {0};
  IREE_ASSERT_OK(tt_iree_build_ttex_executable_def_with_version(
      alloc, TT_IREE_TTEX_VERSION_1, 1, &ep, &data));
  ASSERT_EQ(ttex_ns(ExecutableDef_verify_as_root(data.data, data.data_length)),
            0);
  ttex_ns(ExecutableDef_table_t) def =
      ttex_ns(ExecutableDef_as_root(data.data));
  EXPECT_EQ(ttex_ns(ExecutableDef_version(def)), TT_IREE_TTEX_VERSION_1);

  iree_allocator_free(alloc, data.data);
}

TEST(TtexSchemaTest, BuildVerifyResNet50TailV8) {
  iree_allocator_t alloc = iree_allocator_system();
  auto tensor = [](uint16_t binding, uint8_t layout, uint8_t rank,
                   std::array<uint32_t, 4> shape, uint64_t byte_offset = 0) {
    tt_iree_ttex_tensor_desc_t desc = {};
    desc.binding = binding;
    desc.element_type = TT_IREE_TTEX_TENSOR_ELEMENT_TYPE_F32;
    desc.layout = layout;
    desc.rank = rank;
    std::copy(shape.begin(), shape.end(), desc.shape);
    std::copy(shape.begin(), shape.end(), desc.storage_shape);
    desc.byte_offset = byte_offset;
    return desc;
  };

  tt_iree_ttex_entry_point_desc_t eps[3] = {};
  eps[0].name = "global_sum";
  eps[0].binding_count = 3;
  eps[0].builtin_program = TT_IREE_TTEX_BUILTIN_PROGRAM_F32_GLOBAL_SUM;
  eps[0].dispatch_kind = TT_IREE_TTEX_DISPATCH_KIND_GLOBAL_SUM;
  eps[0].global_sum.input =
      tensor(0, TT_IREE_TTEX_TENSOR_LAYOUT_CHW, 3, {2048, 8, 8, 0}, 1069056);
  eps[0].global_sum.sum =
      tensor(1, TT_IREE_TTEX_TENSOR_LAYOUT_ROW_MAJOR, 1, {2048, 0, 0, 0});
  eps[0].global_sum.divisor =
      tensor(2, TT_IREE_TTEX_TENSOR_LAYOUT_ROW_MAJOR, 1, {1, 0, 0, 0}, 8192);
  eps[0].global_sum.reduce_h = 7;
  eps[0].global_sum.reduce_w = 7;
  eps[0].global_sum.divisor_value = 49.0f;

  eps[1].name = "scalar_div";
  eps[1].binding_count = 2;
  eps[1].builtin_program = TT_IREE_TTEX_BUILTIN_PROGRAM_F32_SCALAR_DIV;
  eps[1].dispatch_kind = TT_IREE_TTEX_DISPATCH_KIND_SCALAR_DIV;
  eps[1].scalar_div.input =
      tensor(0, TT_IREE_TTEX_TENSOR_LAYOUT_ROW_MAJOR, 1, {2048, 0, 0, 0});
  eps[1].scalar_div.divisor =
      tensor(0, TT_IREE_TTEX_TENSOR_LAYOUT_ROW_MAJOR, 1, {1, 0, 0, 0}, 8192);
  eps[1].scalar_div.output =
      tensor(1, TT_IREE_TTEX_TENSOR_LAYOUT_ROW_MAJOR, 1, {2048, 0, 0, 0}, 8256);

  eps[2].name = "linear";
  eps[2].binding_count = 3;
  eps[2].builtin_program = TT_IREE_TTEX_BUILTIN_PROGRAM_F32_LINEAR;
  eps[2].dispatch_kind = TT_IREE_TTEX_DISPATCH_KIND_LINEAR;
  eps[2].linear.input =
      tensor(0, TT_IREE_TTEX_TENSOR_LAYOUT_ROW_MAJOR, 2, {1, 2048, 0, 0}, 8256);
  eps[2].linear.weights =
      tensor(1, TT_IREE_TTEX_TENSOR_LAYOUT_OI, 2, {1000, 2048, 0, 0});
  eps[2].linear.bias = tensor(1, TT_IREE_TTEX_TENSOR_LAYOUT_ROW_MAJOR, 2,
                              {1, 1000, 0, 0}, 102109952);
  eps[2].linear.output =
      tensor(2, TT_IREE_TTEX_TENSOR_LAYOUT_ROW_MAJOR, 2, {1, 1000, 0, 0});
  eps[2].linear.m = 1;
  eps[2].linear.n = 1000;
  eps[2].linear.k = 2048;
  for (auto &ep : eps) {
    ep.workgroup_size[0] = 1;
    ep.workgroup_size[1] = 1;
    ep.workgroup_size[2] = 1;
  }

  iree_byte_span_t data = {};
  IREE_ASSERT_OK(tt_iree_build_ttex_executable_def_with_version(
      alloc, TT_IREE_TTEX_VERSION_8, IREE_ARRAYSIZE(eps), eps, &data));
  ASSERT_EQ(ttex_ns(ExecutableDef_verify_as_root(data.data, data.data_length)),
            0);
  auto parsed = ttex_ns(
      ExecutableDef_entry_points(ttex_ns(ExecutableDef_as_root(data.data))));
  ASSERT_EQ(ttex_ns(EntryPointDef_vec_len(parsed)), 3u);
  auto global_sum = ttex_ns(
      EntryPointDef_global_sum(ttex_ns(EntryPointDef_vec_at(parsed, 0))));
  ASSERT_NE(global_sum, nullptr);
  EXPECT_EQ(ttex_ns(GlobalSumDef_reduce_h(global_sum)), 7u);
  EXPECT_FLOAT_EQ(ttex_ns(GlobalSumDef_divisor_value(global_sum)), 49.0f);
  auto scalar_div = ttex_ns(
      EntryPointDef_scalar_div(ttex_ns(EntryPointDef_vec_at(parsed, 1))));
  ASSERT_NE(scalar_div, nullptr);
  EXPECT_EQ(ttex_ns(TensorDesc_byte_offset(
                ttex_ns(ScalarDivDef_divisor(scalar_div)))),
            8192u);
  auto linear =
      ttex_ns(EntryPointDef_linear(ttex_ns(EntryPointDef_vec_at(parsed, 2))));
  ASSERT_NE(linear, nullptr);
  EXPECT_EQ(ttex_ns(LinearDef_m(linear)), 1u);
  EXPECT_EQ(ttex_ns(LinearDef_n(linear)), 1000u);
  EXPECT_EQ(ttex_ns(LinearDef_k(linear)), 2048u);
  iree_allocator_free(alloc, data.data);
}

TEST(TtexSchemaTest, RejectsResNet50TailBeforeV8) {
  iree_allocator_t alloc = iree_allocator_system();
  tt_iree_ttex_entry_point_desc_t ep = {};
  ep.name = "global_sum";
  ep.binding_count = 3;
  ep.builtin_program = TT_IREE_TTEX_BUILTIN_PROGRAM_F32_GLOBAL_SUM;
  ep.dispatch_kind = TT_IREE_TTEX_DISPATCH_KIND_GLOBAL_SUM;
  ep.global_sum.input.rank = 3;
  ep.global_sum.sum.rank = 1;
  ep.global_sum.divisor.rank = 1;
  iree_byte_span_t data = {};
  iree_status_t status = tt_iree_build_ttex_executable_def_with_version(
      alloc, TT_IREE_TTEX_VERSION_7, 1, &ep, &data);
  EXPECT_EQ(iree_status_code(status), IREE_STATUS_INVALID_ARGUMENT);
  iree_status_ignore(status);
}

TEST(TtexSchemaTest, BuildVerifyMultiEntry) {
  iree_allocator_t alloc = iree_allocator_system();

  tt_iree_ttex_entry_point_desc_t eps[2] = {};
  eps[0].name = "dispatch_0_add";
  eps[0].binding_count = 3;
  eps[0].builtin_program = TT_IREE_TTEX_BUILTIN_PROGRAM_CUSTOM_SFPI_ADD;
  eps[0].workgroup_size[0] = 1;
  eps[0].workgroup_size[1] = 1;
  eps[0].workgroup_size[2] = 1;

  eps[1].name = "dispatch_1_matmul";
  eps[1].binding_count = 3;
  eps[1].builtin_program = TT_IREE_TTEX_BUILTIN_PROGRAM_BF16_MATMUL_TILED;
  eps[1].workgroup_size[0] = 4;
  eps[1].workgroup_size[1] = 2;
  eps[1].workgroup_size[2] = 1;
  eps[1].builtin_m_tiles = 2;
  eps[1].builtin_n_tiles = 3;
  eps[1].builtin_k_tiles = 4;

  iree_byte_span_t data = {0};
  IREE_ASSERT_OK(tt_iree_build_ttex_executable_def(alloc, 2, eps, &data));

  int ret = ttex_ns(ExecutableDef_verify_as_root(data.data, data.data_length));
  ASSERT_EQ(ret, 0);

  ttex_ns(ExecutableDef_table_t) def =
      ttex_ns(ExecutableDef_as_root(data.data));
  ttex_ns(EntryPointDef_vec_t) parsed_eps =
      ttex_ns(ExecutableDef_entry_points(def));
  ASSERT_EQ(ttex_ns(EntryPointDef_vec_len(parsed_eps)), 2u);

  ttex_ns(EntryPointDef_table_t) ep1 =
      ttex_ns(EntryPointDef_vec_at(parsed_eps, 1));
  ASSERT_STREQ(ttex_ns(EntryPointDef_name(ep1)), "dispatch_1_matmul");
  ASSERT_EQ(ttex_ns(EntryPointDef_builtin(ep1)),
            TT_IREE_TTEX_BUILTIN_PROGRAM_BF16_MATMUL_TILED);
  ASSERT_EQ(ttex_ns(EntryPointDef_workgroup_size_x(ep1)), 4u);
  ASSERT_EQ(ttex_ns(EntryPointDef_workgroup_size_y(ep1)), 2u);
  ASSERT_EQ(ttex_ns(EntryPointDef_builtin_m_tiles(ep1)), 2u);
  ASSERT_EQ(ttex_ns(EntryPointDef_builtin_n_tiles(ep1)), 3u);
  ASSERT_EQ(ttex_ns(EntryPointDef_builtin_k_tiles(ep1)), 4u);

  iree_allocator_free(alloc, data.data);
}

TEST(TtexSchemaTest, VerifyRejectsGarbage) {
  uint8_t garbage[] = {0xDE, 0xAD, 0xBE, 0xEF, 0x00, 0x01, 0x02, 0x03};
  int ret = ttex_ns(ExecutableDef_verify_as_root(garbage, sizeof(garbage)));
  ASSERT_NE(ret, 0);
}

TEST(TtexSchemaTest, NativeRoleContractIsVersioned) {
  auto builtin = TT_IREE_TTEX_BUILTIN_PROGRAM_F32_MAX_POOL_2D;
  auto kind = TT_IREE_TTEX_DISPATCH_KIND_MAX_POOL;
  EXPECT_EQ(tt_iree_ttex_required_aot_kernel_roles_for_version(12, builtin, kind),
            tt_iree_ttex_required_aot_kernel_roles(builtin, kind));
  EXPECT_EQ(tt_iree_ttex_required_aot_kernel_roles_for_version(13, builtin, kind),
            TT_IREE_TTEX_AOT_KERNEL_ROLE_BIT(TT_IREE_TTEX_KERNEL_ROLE_READER));
}

TEST(TtexSchemaTest, NativeAffineAndResidualTensorArgumentOrder) {
  tt_iree_ttex_entry_point_desc_t ep = {};
  ep.dispatch_kind = TT_IREE_TTEX_DISPATCH_KIND_MATMUL_LIKE;
  ep.matmul_like.epilogue_flags = TT_IREE_TTEX_EPILOGUE_RUNTIME_AFFINE |
                                TT_IREE_TTEX_EPILOGUE_RESIDUAL |
                                TT_IREE_TTEX_EPILOGUE_RESIDUAL_PER_CHANNEL;
  const tt_iree_ttex_tensor_desc_t *t[TT_IREE_NATIVE_TENSOR_COUNT];
  tt_iree_ttex_native_tensors(&ep, t);
  EXPECT_EQ(t[0], &ep.matmul_like.lhs);
  EXPECT_EQ(t[1], &ep.matmul_like.rhs);
  EXPECT_EQ(t[2], &ep.matmul_like.output);
  EXPECT_EQ(t[3], &ep.matmul_like.epilogue_mean);
  EXPECT_EQ(t[4], &ep.matmul_like.epilogue_channel_scale);
  EXPECT_EQ(t[5], &ep.matmul_like.epilogue_channel_bias);
  EXPECT_EQ(t[6], &ep.matmul_like.residual);
  EXPECT_EQ(t[7], &ep.matmul_like.residual_mean);
  EXPECT_EQ(t[8], &ep.matmul_like.residual_channel_scale);
  EXPECT_EQ(t[9], &ep.matmul_like.residual_channel_bias);
  ep.matmul_like.epilogue_flags = 0;
  tt_iree_ttex_native_tensors(&ep, t);
  for (unsigned i = 3; i < TT_IREE_NATIVE_TENSOR_COUNT; ++i)
    EXPECT_EQ(t[i], nullptr);
}

TEST(TtexSchemaTest, NativeFp32MatrixRoleContractIsVersioned) {
  auto builtin = TT_IREE_TTEX_BUILTIN_PROGRAM_F32_MATMUL_LIKE;
  auto kind = TT_IREE_TTEX_DISPATCH_KIND_MATMUL_LIKE;
  EXPECT_EQ(tt_iree_ttex_required_aot_kernel_roles_for_version(14, builtin, kind),
            tt_iree_ttex_required_aot_kernel_roles(builtin, kind));
  EXPECT_EQ(tt_iree_ttex_required_aot_kernel_roles_for_version(15, builtin, kind),
            TT_IREE_TTEX_AOT_KERNEL_ROLE_BIT(TT_IREE_TTEX_KERNEL_ROLE_READER));
}

TEST(TtexSchemaTest, NativeReductionUsesSeparateDivisorView) {
  tt_iree_ttex_entry_point_desc_t ep = {};
  ep.dispatch_kind = TT_IREE_TTEX_DISPATCH_KIND_GLOBAL_SUM;
  const tt_iree_ttex_tensor_desc_t *t[TT_IREE_NATIVE_TENSOR_COUNT];
  tt_iree_ttex_native_tensors(&ep, t);
  EXPECT_EQ(t[0], &ep.global_sum.input);
  EXPECT_EQ(t[1], &ep.global_sum.divisor);
  EXPECT_EQ(t[2], &ep.global_sum.sum);
}

TEST(TtexSchemaTest, NativeBundleRolesEnforcedByBuilder) {
  const tt_iree_ttex_kernel_source_desc_t kernels[] = {
      {TT_IREE_TTEX_KERNEL_ROLE_READER, "void kernel_main() {}"},
      {TT_IREE_TTEX_KERNEL_ROLE_WRITER, "void kernel_main() {}"},
      {TT_IREE_TTEX_KERNEL_ROLE_COMPUTE, "void kernel_main() {}"},
  };
  const uint8_t elf[] = {0x7f, 'E', 'L', 'F'};
  const tt_iree_ttex_artifact_file_desc_t artifact = {
      "native/brisc/kernel.elf", elf, sizeof(elf)};
  tt_iree_ttex_aot_kernel_bundle_desc_t bundle = {
      .target_arch = TT_IREE_TTEX_TARGET_ARCH_BLACKHOLE,
      .kernel_count = 1,
      .kernels = kernels,
      .artifact_count = 1,
      .artifacts = &artifact,
  };
  auto check = [&](tt_iree_ttex_entry_point_desc_t ep, uint32_t version,
                   iree_host_size_t roles, bool accepted) {
    bundle.kernel_count = roles;
    ep.aot_kernel_bundle = &bundle;
    iree_byte_span_t data = {};
    iree_status_t status = tt_iree_build_ttex_executable_def_with_version(
        iree_allocator_system(), version, 1, &ep, &data);
    EXPECT_EQ(iree_status_is_ok(status), accepted);
    iree_status_ignore(status);
    if (data.data)
      iree_allocator_free(iree_allocator_system(), data.data);
  };
  check(MakeMatmulLikeEntry(), TT_IREE_TTEX_VERSION_14, 3, true);
  check(MakeMatmulLikeEntry(), TT_IREE_TTEX_VERSION_14, 1, false);
  check(MakeMatmulLikeEntry(), TT_IREE_TTEX_VERSION_15, 1, true);
  check(MakeMatmulLikeEntry(), TT_IREE_TTEX_VERSION_15, 3, false);
  check(MakeMatmulLikeEntry(), TT_IREE_TTEX_VERSION_16, 3, true);
  check(MakeMatmulLikeEntry(), TT_IREE_TTEX_VERSION_16, 1, false);
  check(MakeConv2DEntry(), TT_IREE_TTEX_VERSION_16, 3, true);
  check(MakeConv2DEntry(), TT_IREE_TTEX_VERSION_16, 1, false);
  check(MakeMaxPoolEntry(), TT_IREE_TTEX_VERSION_12, 3, true);
  check(MakeMaxPoolEntry(), TT_IREE_TTEX_VERSION_12, 1, false);
  check(MakeMaxPoolEntry(), TT_IREE_TTEX_VERSION_13, 1, true);
  check(MakeMaxPoolEntry(), TT_IREE_TTEX_VERSION_13, 3, false);
  check(MakeMaxPoolEntry(), TT_IREE_TTEX_VERSION_16, 1, true);
  check(MakeMaxPoolEntry(), TT_IREE_TTEX_VERSION_16, 3, false);
}

TEST(TtexSchemaTest, BuildEmpty) {
  iree_allocator_t alloc = iree_allocator_system();

  iree_byte_span_t data = {0};
  IREE_ASSERT_OK(tt_iree_build_ttex_executable_def(alloc, 0, NULL, &data));
  ASSERT_NE(data.data, nullptr);

  int ret = ttex_ns(ExecutableDef_verify_as_root(data.data, data.data_length));
  ASSERT_EQ(ret, 0);

  ttex_ns(ExecutableDef_table_t) def =
      ttex_ns(ExecutableDef_as_root(data.data));
  ttex_ns(EntryPointDef_vec_t) parsed_eps =
      ttex_ns(ExecutableDef_entry_points(def));
  ASSERT_EQ(ttex_ns(EntryPointDef_vec_len(parsed_eps)), 0u);

  iree_allocator_free(alloc, data.data);
}

} // namespace
} // namespace tenstorrent
} // namespace hal
} // namespace iree
