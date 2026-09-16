// Copyright 2025 The tt-iree Authors
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// Unit tests for TTEX FlatBuffer schema: build → verify → parse round-trip.
// These tests run without hardware (mock mode OK) and without the compiler.

#include <algorithm>
#include <array>
#include <cstring>

#include "iree/base/api.h"
#include "iree/schema/tt_executable_builder_util.h"
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

TEST(TtexSchemaTest, BuildExplicitVersion) {
  iree_allocator_t alloc = iree_allocator_system();

  tt_iree_ttex_entry_point_desc_t ep = {};
  ep.name = "future_version";
  ep.binding_count = 3;
  ep.workgroup_size[0] = 1;
  ep.workgroup_size[1] = 1;
  ep.workgroup_size[2] = 1;
  ep.builtin_program = TT_IREE_TTEX_BUILTIN_PROGRAM_BF16_MATMUL_TILED;

  constexpr uint32_t kFutureVersion = TT_IREE_TTEX_CURRENT_VERSION + 1;
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
