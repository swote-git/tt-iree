// Copyright 2025 The tt-iree Authors
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// End-to-end dispatch test: bfloat16 tile addition on P100A hardware.

#include <algorithm>
#include <array>
#include <cmath>
#include <filesystem>
#include <string>
#include <vector>

#ifdef TT_IREE_ENABLE_AOT_TESTS
#include "TenstorrentAOTCompiler.h"
#endif

#include "iree/hal/drivers/tenstorrent/tt_buffer.h"
#include "iree/hal/drivers/tenstorrent/tt_command_buffer.h"
#include "iree/hal/drivers/tenstorrent/tt_executable.h"
#include "iree/hal/drivers/tenstorrent/tt_executable_cache.h"
#include "iree/schema/tt_executable_builder_util.h"
#include "utils.h"

#ifndef TT_IREE_ENABLE_MOCK
#include <tt-metalium/bfloat16.hpp>
#include <tt-metalium/host_api.hpp>
#include <tt-metalium/tt_metal.hpp>
#endif

namespace iree {
namespace hal {
namespace tenstorrent {
namespace {

using DispatchTest = TenstorrentTestBase;

#ifdef TT_IREE_ENABLE_AOT_TESTS
struct SerializedAOTTestBundle {
  explicit SerializedAOTTestBundle(uint32_t builtin_program,
                                   bool device_native_f32_matmul = false) {
    using namespace mlir::iree_compiler::IREE::HAL;
    bool compiled = false;
    if (builtin_program == TT_IREE_TTEX_BUILTIN_PROGRAM_CUSTOM_SFPI_ADD) {
      compiled = compileAddAOT(TenstorrentAOTTargetArch::Blackhole, bundle,
                               error_message);
    } else if (builtin_program ==
               TT_IREE_TTEX_BUILTIN_PROGRAM_F32_PADDED_COPY) {
      compiled = compilePaddedCopyAOT(TenstorrentAOTTargetArch::Blackhole,
                                      bundle, error_message);
    } else if (device_native_f32_matmul) {
      compiled = compileF32MatmulAOT(TenstorrentAOTTargetArch::Blackhole,
                                     bundle, error_message);
    } else {
      compiled = compileMatmulAOT(TenstorrentAOTTargetArch::Blackhole, bundle,
                                  error_message);
    }
    if (!compiled)
      return;

    kernel_descs.reserve(bundle.kernels.size());
    for (const TenstorrentAOTKernelSource &kernel : bundle.kernels) {
      kernel_descs.push_back({kernel.role, kernel.source.c_str()});
    }
    artifact_descs.reserve(bundle.artifacts.size());
    for (const TenstorrentAOTArtifact &artifact : bundle.artifacts) {
      artifact_descs.push_back(
          {artifact.path.c_str(), artifact.data.data(), artifact.data.size()});
    }
    desc.target_arch = static_cast<uint8_t>(bundle.targetArch);
    desc.kernel_count = kernel_descs.size();
    desc.kernels = kernel_descs.data();
    desc.artifact_count = artifact_descs.size();
    desc.artifacts = artifact_descs.data();
  }

  mlir::iree_compiler::IREE::HAL::TenstorrentAOTKernelBundle bundle;
  std::vector<tt_iree_ttex_kernel_source_desc_t> kernel_descs;
  std::vector<tt_iree_ttex_artifact_file_desc_t> artifact_descs;
  tt_iree_ttex_aot_kernel_bundle_desc_t desc = {};
  std::string error_message;
};

// TT-Metal's offline compiler uses a mock MetalContext. Build both fixtures
// before the test suite opens a physical device; mock and physical contexts
// cannot coexist in one process.
static SerializedAOTTestBundle
    g_add_aot_bundle(TT_IREE_TTEX_BUILTIN_PROGRAM_CUSTOM_SFPI_ADD);
static SerializedAOTTestBundle
    g_matmul_aot_bundle(TT_IREE_TTEX_BUILTIN_PROGRAM_BF16_MATMUL_TILED);
static SerializedAOTTestBundle
    g_native_f32_matmul_aot_bundle(TT_IREE_TTEX_BUILTIN_PROGRAM_F32_MATMUL_LIKE,
                                   /*device_native_f32_matmul=*/true);
static SerializedAOTTestBundle
    g_padded_copy_aot_bundle(TT_IREE_TTEX_BUILTIN_PROGRAM_F32_PADDED_COPY);

static iree_status_t
GetAOTTestBundle(uint32_t builtin_program,
                 const tt_iree_ttex_aot_kernel_bundle_desc_t **out_bundle,
                 bool device_native_f32_matmul = false) {
  IREE_ASSERT_ARGUMENT(out_bundle);
  *out_bundle = nullptr;
  SerializedAOTTestBundle *bundle = &g_matmul_aot_bundle;
  if (builtin_program == TT_IREE_TTEX_BUILTIN_PROGRAM_CUSTOM_SFPI_ADD) {
    bundle = &g_add_aot_bundle;
  } else if (device_native_f32_matmul) {
    bundle = &g_native_f32_matmul_aot_bundle;
  } else if (builtin_program == TT_IREE_TTEX_BUILTIN_PROGRAM_F32_PADDED_COPY) {
    bundle = &g_padded_copy_aot_bundle;
  }
  if (!bundle->error_message.empty()) {
    return iree_make_status(IREE_STATUS_UNAVAILABLE,
                            "AOT test fixture compilation failed: %s",
                            bundle->error_message.c_str());
  }
  *out_bundle = &bundle->desc;
  return iree_ok_status();
}
#endif

static iree_status_t CreateBuiltinTtexExecutable(
    iree_hal_device_t *device, iree_allocator_t host_allocator,
    uint32_t builtin_program, uint16_t constant_count, uint32_t builtin_m_tiles,
    uint32_t builtin_n_tiles, uint32_t builtin_k_tiles,
    iree_hal_executable_t **out_executable, bool include_aot = false) {
  tt_iree_ttex_entry_point_desc_t ep = {};
  ep.name = "builtin_dispatch_0";
  ep.constant_count = constant_count;
  ep.binding_count = 3;
  ep.flags = 0;
  ep.workgroup_size[0] = 1;
  ep.workgroup_size[1] = 1;
  ep.workgroup_size[2] = 1;
  ep.builtin_program = builtin_program;
  ep.builtin_m_tiles = builtin_m_tiles;
  ep.builtin_n_tiles = builtin_n_tiles;
  ep.builtin_k_tiles = builtin_k_tiles;

  if (include_aot) {
#ifdef TT_IREE_ENABLE_AOT_TESTS
    IREE_RETURN_IF_ERROR(
        GetAOTTestBundle(builtin_program, &ep.aot_kernel_bundle));
#else
    return iree_make_status(IREE_STATUS_UNAVAILABLE,
                            "AOT test fixture is unavailable in this build");
#endif
  }

  iree_byte_span_t data = {0};
  IREE_RETURN_IF_ERROR(
      tt_iree_build_ttex_executable_def_with_version(
          host_allocator, TT_IREE_TTEX_VERSION_12, 1, &ep, &data));

  iree_hal_executable_params_t exec_params;
  iree_hal_executable_params_initialize(&exec_params);
  exec_params.executable_format =
      iree_make_cstring_view(IREE_HAL_TT_EXECUTABLE_FORMAT);
  exec_params.executable_data =
      iree_make_const_byte_span(data.data, data.data_length);

  iree_status_t status = iree_hal_tt_executable_create(
      device, &exec_params, host_allocator, out_executable);
  iree_allocator_free(host_allocator, data.data);
  return status;
}

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

static tt_iree_ttex_entry_point_desc_t MakeDeviceNativeMatmulLikeEntry() {
  tt_iree_ttex_entry_point_desc_t ep = MakeMatmulLikeEntry();
  ep.name = "device_native_f32_projection";
  ep.matmul_like.epilogue_flags = 0;
  ep.matmul_like.epilogue_scale = 1.0f;
  ep.matmul_like.epilogue_bias = 0.0f;
  return ep;
}

static tt_iree_ttex_entry_point_desc_t MakeDeviceNativeMatmulLikeMTailEntry() {
  tt_iree_ttex_entry_point_desc_t ep = MakeDeviceNativeMatmulLikeEntry();
  ep.name = "device_native_f32_projection_m_tail";
  ep.matmul_like.lhs.layout = TT_IREE_TTEX_TENSOR_LAYOUT_CHANNEL_MAJOR;
  ep.matmul_like.lhs.rank = 2;
  ep.matmul_like.lhs.shape[0] = 64;
  ep.matmul_like.lhs.shape[1] = 49;
  ep.matmul_like.lhs.shape[2] = 0;
  ep.matmul_like.lhs.storage_shape[0] = 64;
  ep.matmul_like.lhs.storage_shape[1] = 49;
  ep.matmul_like.lhs.storage_shape[2] = 0;
  ep.matmul_like.output.layout = TT_IREE_TTEX_TENSOR_LAYOUT_CHANNEL_MAJOR;
  ep.matmul_like.output.rank = 2;
  ep.matmul_like.output.shape[0] = 64;
  ep.matmul_like.output.shape[1] = 49;
  ep.matmul_like.output.shape[2] = 0;
  ep.matmul_like.output.storage_shape[0] = 64;
  ep.matmul_like.output.storage_shape[1] = 53;
  ep.matmul_like.output.storage_shape[2] = 0;
  ep.matmul_like.output.offset[0] = 0;
  ep.matmul_like.output.offset[1] = 2;
  ep.matmul_like.output.offset[2] = 0;
  ep.matmul_like.m = 49;
  return ep;
}

static tt_iree_ttex_entry_point_desc_t
MakeRuntimePerChannelDeviceNativeMatmulLikeEntry() {
  tt_iree_ttex_entry_point_desc_t ep = MakeDeviceNativeMatmulLikeMTailEntry();
  ep.name = "device_native_f32_runtime_per_channel_projection";
  ep.binding_count = 6;
  ep.matmul_like.epilogue_flags =
      TT_IREE_TTEX_EPILOGUE_AFFINE | TT_IREE_TTEX_EPILOGUE_RELU |
      TT_IREE_TTEX_EPILOGUE_PER_CHANNEL | TT_IREE_TTEX_EPILOGUE_RUNTIME_AFFINE;
  ep.matmul_like.epilogue_mean = {
      .binding = 3,
      .element_type = TT_IREE_TTEX_TENSOR_ELEMENT_TYPE_F32,
      .layout = TT_IREE_TTEX_TENSOR_LAYOUT_ROW_MAJOR,
      .rank = 1,
      .shape = {64, 0, 0, 0},
      .storage_shape = {64, 0, 0, 0},
  };
  ep.matmul_like.epilogue_channel_scale = {
      .binding = 4,
      .element_type = TT_IREE_TTEX_TENSOR_ELEMENT_TYPE_F32,
      .layout = TT_IREE_TTEX_TENSOR_LAYOUT_ROW_MAJOR,
      .rank = 1,
      .shape = {64, 0, 0, 0},
      .storage_shape = {64, 0, 0, 0},
  };
  ep.matmul_like.epilogue_channel_bias = {
      .binding = 5,
      .element_type = TT_IREE_TTEX_TENSOR_ELEMENT_TYPE_F32,
      .layout = TT_IREE_TTEX_TENSOR_LAYOUT_ROW_MAJOR,
      .rank = 1,
      .shape = {64, 0, 0, 0},
      .storage_shape = {64, 0, 0, 0},
  };
  ep.matmul_like.epilogue_common_scale = 0.5f;
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
  ep.matmul_like.epilogue_scale = -1000.0f;
  ep.matmul_like.epilogue_bias = -1000.0f;
  ep.matmul_like.epilogue_channel_count = kScales.size();
  ep.matmul_like.epilogue_scales = kScales.data();
  ep.matmul_like.epilogue_biases = kBiases.data();
  return ep;
}

static tt_iree_ttex_entry_point_desc_t MakeResNet50LinearEntry() {
  tt_iree_ttex_entry_point_desc_t ep = {};
  ep.name = "resnet50_classifier";
  ep.binding_count = 4;
  ep.workgroup_size[0] = 1;
  ep.workgroup_size[1] = 1;
  ep.workgroup_size[2] = 1;
  ep.builtin_program = TT_IREE_TTEX_BUILTIN_PROGRAM_F32_LINEAR;
  ep.dispatch_kind = TT_IREE_TTEX_DISPATCH_KIND_LINEAR;
  ep.linear.input = {
      .binding = 0,
      .element_type = TT_IREE_TTEX_TENSOR_ELEMENT_TYPE_F32,
      .layout = TT_IREE_TTEX_TENSOR_LAYOUT_ROW_MAJOR,
      .rank = 2,
      .shape = {1, 2048, 0, 0},
      .storage_shape = {1, 2048, 0, 0},
  };
  ep.linear.weights = {
      .binding = 1,
      .element_type = TT_IREE_TTEX_TENSOR_ELEMENT_TYPE_F32,
      .layout = TT_IREE_TTEX_TENSOR_LAYOUT_OI,
      .rank = 2,
      .shape = {1000, 2048, 0, 0},
      .storage_shape = {1000, 2048, 0, 0},
  };
  ep.linear.bias = {
      .binding = 2,
      .element_type = TT_IREE_TTEX_TENSOR_ELEMENT_TYPE_F32,
      .layout = TT_IREE_TTEX_TENSOR_LAYOUT_ROW_MAJOR,
      .rank = 2,
      .shape = {1, 1000, 0, 0},
      .storage_shape = {1, 1000, 0, 0},
  };
  ep.linear.output = {
      .binding = 3,
      .element_type = TT_IREE_TTEX_TENSOR_ELEMENT_TYPE_F32,
      .layout = TT_IREE_TTEX_TENSOR_LAYOUT_ROW_MAJOR,
      .rank = 2,
      .shape = {1, 1000, 0, 0},
      .storage_shape = {1, 1000, 0, 0},
  };
  ep.linear.m = 1;
  ep.linear.n = 1000;
  ep.linear.k = 2048;
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

static tt_iree_ttex_entry_point_desc_t MakePerChannelConv2DEntry() {
  static const std::array<float, 64> kScales = [] {
    std::array<float, 64> values = {};
    values.fill(0.25f);
    return values;
  }();
  static const std::array<float, 64> kBiases = [] {
    std::array<float, 64> values = {};
    for (size_t i = 0; i < values.size(); ++i)
      values[i] = static_cast<float>(i);
    return values;
  }();

  tt_iree_ttex_entry_point_desc_t ep = MakeConv2DEntry();
  ep.name = "resnet50_per_channel_conv2d";
  ep.conv2d.output.storage_shape[1] = 10;
  ep.conv2d.output.storage_shape[2] = 10;
  ep.conv2d.output.offset[1] = 1;
  ep.conv2d.output.offset[2] = 1;
  ep.conv2d.epilogue_flags |= TT_IREE_TTEX_EPILOGUE_PER_CHANNEL;
  ep.conv2d.epilogue_scale = 0.0f;
  ep.conv2d.epilogue_bias = -1000.0f;
  ep.conv2d.epilogue_channel_count = kScales.size();
  ep.conv2d.epilogue_scales = kScales.data();
  ep.conv2d.epilogue_biases = kBiases.data();
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
      .shape = {1, 2, 2, 0},
      .storage_shape = {1, 2, 2, 0},
  };
  ep.padded_copy.destination = {
      .binding = 1,
      .element_type = TT_IREE_TTEX_TENSOR_ELEMENT_TYPE_F32,
      .layout = TT_IREE_TTEX_TENSOR_LAYOUT_CHW,
      .rank = 3,
      .shape = {1, 2, 2, 0},
      .storage_shape = {1, 4, 4, 0},
      .offset = {0, 1, 1, 0},
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
      .storage_shape = {1, 4, 4, 0},
      .offset = {0, 1, 1, 0},
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

static iree_status_t CreateDescriptorTtexExecutable(
    iree_hal_device_t *device, iree_allocator_t host_allocator,
    tt_iree_ttex_entry_point_desc_t ep, iree_hal_executable_t **out_executable,
    bool include_aot = false) {
  if (include_aot) {
#ifdef TT_IREE_ENABLE_AOT_TESTS
    constexpr uint32_t kRuntimePerChannelAffineRelu =
        TT_IREE_TTEX_EPILOGUE_AFFINE | TT_IREE_TTEX_EPILOGUE_RELU |
        TT_IREE_TTEX_EPILOGUE_PER_CHANNEL |
        TT_IREE_TTEX_EPILOGUE_RUNTIME_AFFINE;
    const bool device_native_f32_matmul =
        ep.builtin_program == TT_IREE_TTEX_BUILTIN_PROGRAM_F32_MATMUL_LIKE &&
        (ep.matmul_like.epilogue_flags == 0 ||
         ep.matmul_like.epilogue_flags == kRuntimePerChannelAffineRelu) &&
        ep.matmul_like.m != 0 && ep.matmul_like.n % 32 == 0 &&
        ep.matmul_like.k % 32 == 0;
    IREE_RETURN_IF_ERROR(GetAOTTestBundle(
        ep.builtin_program, &ep.aot_kernel_bundle, device_native_f32_matmul));
#else
    return iree_make_status(IREE_STATUS_UNAVAILABLE,
                            "AOT test fixture is unavailable in this build");
#endif
  }

  iree_byte_span_t data = {0};
  IREE_RETURN_IF_ERROR(
      tt_iree_build_ttex_executable_def_with_version(
          host_allocator, TT_IREE_TTEX_VERSION_12, 1, &ep, &data));

  iree_hal_executable_params_t exec_params;
  iree_hal_executable_params_initialize(&exec_params);
  exec_params.executable_format =
      iree_make_cstring_view(IREE_HAL_TT_EXECUTABLE_FORMAT);
  exec_params.executable_data =
      iree_make_const_byte_span(data.data, data.data_length);
  iree_status_t status = iree_hal_tt_executable_create(
      device, &exec_params, host_allocator, out_executable);
  iree_allocator_free(host_allocator, data.data);
  return status;
}

TEST_F(DispatchTest, RejectsEmptyTtexData) {
  iree_hal_executable_params_t exec_params;
  iree_hal_executable_params_initialize(&exec_params);
  exec_params.executable_format =
      iree_make_cstring_view(IREE_HAL_TT_EXECUTABLE_FORMAT);

  iree_hal_executable_t *executable = nullptr;
  iree_status_t status = iree_hal_tt_executable_create(
      device_, &exec_params, host_allocator(), &executable);
  EXPECT_EQ(iree_status_code(status), IREE_STATUS_INVALID_ARGUMENT);
  iree_status_ignore(status);
  EXPECT_EQ(executable, nullptr);
}

TEST_F(DispatchTest, RejectsMalformedTtexData) {
  const uint8_t garbage[] = {0xDE, 0xAD, 0xBE, 0xEF};
  iree_hal_executable_params_t exec_params;
  iree_hal_executable_params_initialize(&exec_params);
  exec_params.executable_format =
      iree_make_cstring_view(IREE_HAL_TT_EXECUTABLE_FORMAT);
  exec_params.executable_data =
      iree_make_const_byte_span(garbage, sizeof(garbage));

  iree_hal_executable_t *executable = nullptr;
  iree_status_t status = iree_hal_tt_executable_create(
      device_, &exec_params, host_allocator(), &executable);
  EXPECT_EQ(iree_status_code(status), IREE_STATUS_INVALID_ARGUMENT);
  iree_status_ignore(status);
  EXPECT_EQ(executable, nullptr);
}

TEST_F(DispatchTest, RejectsLegacyExecutableFormat) {
  iree_hal_executable_params_t exec_params;
  iree_hal_executable_params_initialize(&exec_params);
  exec_params.executable_format =
      iree_make_cstring_view(IREE_HAL_TT_LEGACY_EXECUTABLE_FORMAT);

  iree_hal_executable_t *executable = nullptr;
  iree_status_t status = iree_hal_tt_executable_create(
      device_, &exec_params, host_allocator(), &executable);
  EXPECT_EQ(iree_status_code(status), IREE_STATUS_UNIMPLEMENTED);
  iree_status_ignore(status);
  EXPECT_EQ(executable, nullptr);
}

TEST_F(DispatchTest, RejectsUnsupportedTtexVersion) {
  tt_iree_ttex_entry_point_desc_t ep = {};
  ep.name = "future_version";
  ep.binding_count = 3;
  ep.workgroup_size[0] = 1;
  ep.workgroup_size[1] = 1;
  ep.workgroup_size[2] = 1;
  ep.builtin_program = TT_IREE_TTEX_BUILTIN_PROGRAM_BF16_MATMUL_TILED;

  iree_byte_span_t data = {0};
  IREE_ASSERT_OK(tt_iree_build_ttex_executable_def_with_version(
      host_allocator(), TT_IREE_TTEX_CURRENT_VERSION + 1, 1, &ep, &data));

  iree_hal_executable_params_t exec_params;
  iree_hal_executable_params_initialize(&exec_params);
  exec_params.executable_format =
      iree_make_cstring_view(IREE_HAL_TT_EXECUTABLE_FORMAT);
  exec_params.executable_data =
      iree_make_const_byte_span(data.data, data.data_length);

  iree_hal_executable_t *executable = nullptr;
  iree_status_t status = iree_hal_tt_executable_create(
      device_, &exec_params, host_allocator(), &executable);
  EXPECT_EQ(iree_status_code(status), IREE_STATUS_UNIMPLEMENTED);
  iree_status_ignore(status);
  EXPECT_EQ(executable, nullptr);
  iree_allocator_free(host_allocator(), data.data);
}

TEST_F(DispatchTest, AcceptsVersionOneBuiltin) {
  tt_iree_ttex_entry_point_desc_t ep = {};
  ep.name = "v1_tiled_matmul";
  ep.binding_count = 3;
  ep.workgroup_size[0] = 1;
  ep.workgroup_size[1] = 1;
  ep.workgroup_size[2] = 1;
  ep.builtin_program = TT_IREE_TTEX_BUILTIN_PROGRAM_BF16_MATMUL_TILED;

  iree_byte_span_t data = {0};
  IREE_ASSERT_OK(tt_iree_build_ttex_executable_def_with_version(
      host_allocator(), TT_IREE_TTEX_VERSION_1, 1, &ep, &data));
  iree_hal_executable_params_t exec_params;
  iree_hal_executable_params_initialize(&exec_params);
  exec_params.executable_format =
      iree_make_cstring_view(IREE_HAL_TT_EXECUTABLE_FORMAT);
  exec_params.executable_data =
      iree_make_const_byte_span(data.data, data.data_length);

  iree::vm::ref<iree_hal_executable_t> executable;
  IREE_ASSERT_OK(iree_hal_tt_executable_create(device_, &exec_params,
                                               host_allocator(), &executable));
  iree_allocator_free(host_allocator(), data.data);
}

TEST_F(DispatchTest, MetadataOnlyAddDoesNotCreateProgram) {
  iree::vm::ref<iree_hal_executable_t> executable;
  iree_hal_executable_t *executable_ptr = nullptr;
  IREE_ASSERT_OK(CreateBuiltinTtexExecutable(
      device_, host_allocator(), TT_IREE_TTEX_BUILTIN_PROGRAM_CUSTOM_SFPI_ADD,
      /*constant_count=*/0, /*builtin_m_tiles=*/0, /*builtin_n_tiles=*/0,
      /*builtin_k_tiles=*/0, &executable_ptr));
  executable.assign(executable_ptr);

  const iree_hal_tt_kernel_params_t *params = nullptr;
  IREE_ASSERT_OK(iree_hal_tt_executable_lookup_kernel_params(
      executable.get(), /*entry_point=*/0, &params));
  ASSERT_NE(params, nullptr);
  EXPECT_EQ(params->aot_kernel_bundle, nullptr);
  EXPECT_EQ(params->program, nullptr);
}

TEST_F(DispatchTest, PreservesMatmulLikeDescriptorMetadata) {
  tt_iree_ttex_entry_point_desc_t ep = MakeMatmulLikeEntry();
  iree::vm::ref<iree_hal_executable_t> executable;
  iree_hal_executable_t *executable_ptr = nullptr;
  IREE_ASSERT_OK(CreateDescriptorTtexExecutable(device_, host_allocator(), ep,
                                                &executable_ptr));
  executable.assign(executable_ptr);

  const iree_hal_tt_kernel_params_t *params = nullptr;
  IREE_ASSERT_OK(iree_hal_tt_executable_lookup_kernel_params(
      executable.get(), /*entry_point=*/0, &params));
  ASSERT_NE(params, nullptr);
  EXPECT_EQ(std::string(params->kernel_name.data, params->kernel_name.size),
            "resnet50_projection");
  EXPECT_EQ(params->dispatch_kind, TT_IREE_TTEX_DISPATCH_KIND_MATMUL_LIKE);
  EXPECT_EQ(params->matmul_like.m, 64u);
  EXPECT_EQ(params->matmul_like.n, 64u);
  EXPECT_EQ(params->matmul_like.k, 64u);
  EXPECT_EQ(params->matmul_like.output.storage_shape[1], 10u);
  EXPECT_EQ(params->matmul_like.output.offset[1], 1u);
  EXPECT_FLOAT_EQ(params->matmul_like.epilogue_scale, 0.5f);
  EXPECT_FLOAT_EQ(params->matmul_like.epilogue_bias, 0.1875f);
}

TEST(DispatchContractTest, Bf16NativeDescriptorsFailClosed) {
  auto ep = MakeMatmulLikeEntry();
  ep.builtin_program = TT_IREE_TTEX_BUILTIN_PROGRAM_BF16_NATIVE;
  ep.dispatch_kind = TT_IREE_TTEX_DISPATCH_KIND_BF16_NATIVE;
  auto &d = ep.bf16_native;
  d.operation_kind = TT_IREE_TTEX_DISPATCH_KIND_MATMUL_LIKE;
  d.m = d.n = d.k = 64;
  d.tensors[0] = ep.matmul_like.lhs;
  d.tensors[1] = ep.matmul_like.rhs;
  d.tensors[2] = ep.matmul_like.output;
  for (unsigned i = 0; i < 3; ++i)
    d.tensors[i].element_type = TT_IREE_TTEX_TENSOR_ELEMENT_TYPE_BF16;
  tt_iree_ttex_kernel_source_desc_t source = {TT_IREE_TTEX_KERNEL_ROLE_READER, "void kernel_main() {}"};
  const uint8_t elf[] = {0x7f, 'E', 'L', 'F'};
  tt_iree_ttex_artifact_file_desc_t artifact = {"Kernel_Source_Code/bf16/brisc/brisc.elf", elf, sizeof(elf)};
  tt_iree_ttex_aot_kernel_bundle_desc_t bundle = {
      .target_arch = TT_IREE_TTEX_TARGET_ARCH_BLACKHOLE,
      .kernel_count = 1, .kernels = &source, .artifact_count = 1, .artifacts = &artifact};
  ep.aot_kernel_bundle = &bundle;
  auto check = [&](const tt_iree_ttex_entry_point_desc_t &entry, bool valid,
                   uint32_t version = TT_IREE_TTEX_VERSION_17) {
    iree_byte_span_t data = {};
    IREE_ASSERT_OK(tt_iree_build_ttex_executable_def_with_version(
        iree_allocator_system(), version, 1, &entry, &data));
    iree_hal_executable_params_t params;
    iree_hal_executable_params_initialize(&params);
    params.executable_format = iree_make_cstring_view(IREE_HAL_TT_EXECUTABLE_FORMAT);
    params.executable_data = iree_make_const_byte_span(data.data, data.data_length);
    iree_hal_executable_t *executable = nullptr;
    auto status = iree_hal_tt_executable_create(nullptr, &params, iree_allocator_system(), &executable);
    if (valid) {
      IREE_EXPECT_OK(status);
      const iree_hal_tt_kernel_params_t *kernel = nullptr;
      IREE_ASSERT_OK(iree_hal_tt_executable_lookup_kernel_params(executable, 0, &kernel));
      EXPECT_EQ(kernel->bf16_native.m, 64u);
      EXPECT_EQ(kernel->bf16_native.tensors[0].element_type, TT_IREE_TTEX_TENSOR_ELEMENT_TYPE_BF16);
      EXPECT_EQ(kernel->bf16_native.tensors[13].rank, 0);
    } else {
      EXPECT_EQ(iree_status_code(status), IREE_STATUS_INVALID_ARGUMENT);
      iree_status_ignore(status);
    }
    iree_hal_executable_release(executable);
    iree_allocator_free(iree_allocator_system(), data.data);
  };
  check(ep, true);
  auto invalid = ep;
  invalid.aot_kernel_bundle = nullptr;
  check(invalid, false);
  invalid = ep;
  invalid.bf16_native.tensors[0].rank = 0;
  check(invalid, false);
  invalid = ep;
  invalid.bf16_native.tensors[0].byte_offset = 1;
  check(invalid, false);
  invalid = ep;
  invalid.bf16_native.tensors[1].element_type = TT_IREE_TTEX_TENSOR_ELEMENT_TYPE_UNKNOWN;
  check(invalid, false);
  invalid = ep;
  invalid.bf16_native.m = 63;
  check(invalid, false);
  invalid = ep;
  invalid.bf16_native.k = 63;
  check(invalid, false);
  invalid = ep;
  invalid.bf16_native.operation_kind = TT_IREE_TTEX_DISPATCH_KIND_MAX_POOL;
  invalid.bf16_native.m = invalid.bf16_native.n = invalid.bf16_native.k = 0;
  invalid.bf16_native.tensors[0].rank = 0;
  check(invalid, false);
  invalid = ep;
  invalid.bf16_native.operation_kind = 99;
  check(invalid, false);
  tt_iree_ttex_kernel_source_desc_t tensix_sources[] = {
      {TT_IREE_TTEX_KERNEL_ROLE_READER, "void kernel_main() {}"},
      {TT_IREE_TTEX_KERNEL_ROLE_WRITER, "void kernel_main() {}"},
      {TT_IREE_TTEX_KERNEL_ROLE_COMPUTE, "void kernel_main() {}"}};
  auto tensix_bundle = bundle;
  tensix_bundle.kernel_count = 3;
  tensix_bundle.kernels = tensix_sources;
  auto tensix = ep;
  tensix.builtin_program = TT_IREE_TTEX_BUILTIN_PROGRAM_BF16_NATIVE_TENSIX;
  tensix.aot_kernel_bundle = &tensix_bundle;
  check(tensix, true, TT_IREE_TTEX_VERSION_18);
  invalid = tensix;
  invalid.aot_kernel_bundle = nullptr;
  check(invalid, false, TT_IREE_TTEX_VERSION_18);
  invalid = tensix;
  invalid.bf16_native.tensors[1].element_type = TT_IREE_TTEX_TENSOR_ELEMENT_TYPE_F32;
  check(invalid, false, TT_IREE_TTEX_VERSION_18);
}

TEST(DispatchContractTest, PreservesAotKernelBundleMetadata) {
  tt_iree_ttex_entry_point_desc_t ep = MakeMatmulLikeEntry();
  tt_iree_ttex_kernel_source_desc_t kernels[] = {
      {TT_IREE_TTEX_KERNEL_ROLE_READER, "void kernel_main() {}"},
      {TT_IREE_TTEX_KERNEL_ROLE_WRITER, "void kernel_main() {}"},
      {TT_IREE_TTEX_KERNEL_ROLE_COMPUTE, "void kernel_main() {}"},
  };
  const uint8_t elf_data[] = {0x7f, 'E', 'L', 'F'};
  tt_iree_ttex_artifact_file_desc_t artifacts[] = {
      {"Kernel_Source_Code/hash/brisc/brisc.elf", elf_data, sizeof(elf_data)}};
  tt_iree_ttex_aot_kernel_bundle_desc_t bundle = {
      .target_arch = TT_IREE_TTEX_TARGET_ARCH_BLACKHOLE,
      .kernel_count = IREE_ARRAYSIZE(kernels),
      .kernels = kernels,
      .artifact_count = IREE_ARRAYSIZE(artifacts),
      .artifacts = artifacts,
  };
  ep.aot_kernel_bundle = &bundle;

  iree::vm::ref<iree_hal_executable_t> executable;
  iree_hal_executable_t *executable_ptr = nullptr;
  IREE_ASSERT_OK(CreateDescriptorTtexExecutable(
      /*device=*/nullptr, iree_allocator_system(), ep, &executable_ptr));
  executable.assign(executable_ptr);

  const iree_hal_tt_kernel_params_t *params = nullptr;
  IREE_ASSERT_OK(iree_hal_tt_executable_lookup_kernel_params(
      executable.get(), /*entry_point=*/0, &params));
  ASSERT_NE(params, nullptr);
  EXPECT_NE(params->aot_kernel_bundle, nullptr);
  EXPECT_EQ(params->aot_target_arch, TT_IREE_TTEX_TARGET_ARCH_BLACKHOLE);
  ASSERT_FALSE(iree_string_view_is_empty(params->aot_artifact_root));

  std::filesystem::path artifact_root(std::string(
      params->aot_artifact_root.data, params->aot_artifact_root.size));
  std::filesystem::path artifact_path =
      artifact_root / "Kernel_Source_Code/hash/brisc/brisc.elf";
  EXPECT_TRUE(std::filesystem::is_regular_file(artifact_path));
  EXPECT_EQ(std::filesystem::file_size(artifact_path), sizeof(elf_data));

  executable.reset();
  EXPECT_FALSE(std::filesystem::exists(artifact_root));
}

TEST(DispatchContractTest, AcceptsAotBundleForBuiltinTiledMatmul) {
  tt_iree_ttex_entry_point_desc_t ep = {};
  ep.name = "bf16_tiled_matmul";
  ep.binding_count = 3;
  ep.workgroup_size[0] = 1;
  ep.workgroup_size[1] = 1;
  ep.workgroup_size[2] = 1;
  ep.builtin_program = TT_IREE_TTEX_BUILTIN_PROGRAM_BF16_MATMUL_TILED;
  ep.builtin_m_tiles = 2;
  ep.builtin_n_tiles = 2;
  ep.builtin_k_tiles = 2;

  tt_iree_ttex_kernel_source_desc_t kernels[] = {
      {TT_IREE_TTEX_KERNEL_ROLE_READER, "void kernel_main() {}"},
      {TT_IREE_TTEX_KERNEL_ROLE_WRITER, "void kernel_main() {}"},
      {TT_IREE_TTEX_KERNEL_ROLE_COMPUTE, "void kernel_main() {}"},
  };
  const uint8_t elf_data[] = {0x7f, 'E', 'L', 'F'};
  tt_iree_ttex_artifact_file_desc_t artifacts[] = {
      {"Kernel_Source_Code/hash/brisc/brisc.elf", elf_data, sizeof(elf_data)}};
  tt_iree_ttex_aot_kernel_bundle_desc_t bundle = {
      .target_arch = TT_IREE_TTEX_TARGET_ARCH_BLACKHOLE,
      .kernel_count = IREE_ARRAYSIZE(kernels),
      .kernels = kernels,
      .artifact_count = IREE_ARRAYSIZE(artifacts),
      .artifacts = artifacts,
  };
  ep.aot_kernel_bundle = &bundle;

  iree::vm::ref<iree_hal_executable_t> executable;
  iree_hal_executable_t *executable_ptr = nullptr;
  IREE_ASSERT_OK(CreateDescriptorTtexExecutable(
      /*device=*/nullptr, iree_allocator_system(), ep, &executable_ptr));
  executable.assign(executable_ptr);

  const iree_hal_tt_kernel_params_t *params = nullptr;
  IREE_ASSERT_OK(iree_hal_tt_executable_lookup_kernel_params(
      executable.get(), /*entry_point=*/0, &params));
  ASSERT_NE(params, nullptr);
  EXPECT_EQ(params->builtin_program,
            TT_IREE_TTEX_BUILTIN_PROGRAM_BF16_MATMUL_TILED);
  EXPECT_NE(params->aot_kernel_bundle, nullptr);
  EXPECT_EQ(params->aot_target_arch, TT_IREE_TTEX_TARGET_ARCH_BLACKHOLE);
}

TEST(DispatchContractTest, PreservesResNet50TailDescriptors) {
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

  tt_iree_ttex_entry_point_desc_t entries[3] = {};
  entries[0].name = "global_sum";
  entries[0].binding_count = 3;
  entries[0].builtin_program = TT_IREE_TTEX_BUILTIN_PROGRAM_F32_GLOBAL_SUM;
  entries[0].dispatch_kind = TT_IREE_TTEX_DISPATCH_KIND_GLOBAL_SUM;
  entries[0].global_sum.input =
      tensor(0, TT_IREE_TTEX_TENSOR_LAYOUT_CHW, 3, {2048, 8, 8, 0}, 1069056);
  entries[0].global_sum.sum =
      tensor(1, TT_IREE_TTEX_TENSOR_LAYOUT_ROW_MAJOR, 1, {2048, 0, 0, 0});
  entries[0].global_sum.divisor =
      tensor(2, TT_IREE_TTEX_TENSOR_LAYOUT_ROW_MAJOR, 1, {1, 0, 0, 0}, 8192);
  entries[0].global_sum.reduce_h = 7;
  entries[0].global_sum.reduce_w = 7;
  entries[0].global_sum.divisor_value = 49.0f;

  entries[1].name = "scalar_div";
  entries[1].binding_count = 2;
  entries[1].builtin_program = TT_IREE_TTEX_BUILTIN_PROGRAM_F32_SCALAR_DIV;
  entries[1].dispatch_kind = TT_IREE_TTEX_DISPATCH_KIND_SCALAR_DIV;
  entries[1].scalar_div.input =
      tensor(0, TT_IREE_TTEX_TENSOR_LAYOUT_ROW_MAJOR, 1, {2048, 0, 0, 0});
  entries[1].scalar_div.divisor =
      tensor(0, TT_IREE_TTEX_TENSOR_LAYOUT_ROW_MAJOR, 1, {1, 0, 0, 0}, 8192);
  entries[1].scalar_div.output =
      tensor(1, TT_IREE_TTEX_TENSOR_LAYOUT_ROW_MAJOR, 1, {2048, 0, 0, 0}, 8256);

  entries[2].name = "linear";
  entries[2].binding_count = 3;
  entries[2].builtin_program = TT_IREE_TTEX_BUILTIN_PROGRAM_F32_LINEAR;
  entries[2].dispatch_kind = TT_IREE_TTEX_DISPATCH_KIND_LINEAR;
  entries[2].linear.input =
      tensor(0, TT_IREE_TTEX_TENSOR_LAYOUT_ROW_MAJOR, 2, {1, 2048, 0, 0}, 8256);
  entries[2].linear.weights =
      tensor(1, TT_IREE_TTEX_TENSOR_LAYOUT_OI, 2, {1000, 2048, 0, 0});
  entries[2].linear.bias = tensor(1, TT_IREE_TTEX_TENSOR_LAYOUT_ROW_MAJOR, 2,
                                  {1, 1000, 0, 0}, 102109952);
  entries[2].linear.output =
      tensor(2, TT_IREE_TTEX_TENSOR_LAYOUT_ROW_MAJOR, 2, {1, 1000, 0, 0});
  entries[2].linear.m = 1;
  entries[2].linear.n = 1000;
  entries[2].linear.k = 2048;

  for (auto &entry : entries) {
    entry.workgroup_size[0] = 1;
    entry.workgroup_size[1] = 1;
    entry.workgroup_size[2] = 1;
    iree::vm::ref<iree_hal_executable_t> executable;
    iree_hal_executable_t *executable_ptr = nullptr;
    IREE_ASSERT_OK(CreateDescriptorTtexExecutable(
        /*device=*/nullptr, iree_allocator_system(), entry, &executable_ptr));
    executable.assign(executable_ptr);

    const iree_hal_tt_kernel_params_t *params = nullptr;
    IREE_ASSERT_OK(iree_hal_tt_executable_lookup_kernel_params(
        executable.get(), /*entry_point=*/0, &params));
    ASSERT_NE(params, nullptr);
    EXPECT_EQ(params->dispatch_kind, entry.dispatch_kind);
    EXPECT_EQ(params->builtin_program, entry.builtin_program);
    if (entry.dispatch_kind == TT_IREE_TTEX_DISPATCH_KIND_GLOBAL_SUM) {
      EXPECT_EQ(params->global_sum.reduce_h, 7u);
      EXPECT_EQ(params->global_sum.reduce_w, 7u);
      EXPECT_FLOAT_EQ(params->global_sum.divisor_value, 49.0f);
    } else if (entry.dispatch_kind == TT_IREE_TTEX_DISPATCH_KIND_SCALAR_DIV) {
      EXPECT_EQ(params->scalar_div.divisor.byte_offset, 8192u);
    } else {
      EXPECT_EQ(params->linear.m, 1u);
      EXPECT_EQ(params->linear.n, 1000u);
      EXPECT_EQ(params->linear.k, 2048u);
      EXPECT_EQ(params->linear.bias.byte_offset, 102109952u);
    }
  }
}

TEST_F(DispatchTest, ExecutesResNet50F32LinearClassifier) {
  tt_iree_ttex_entry_point_desc_t ep = MakeResNet50LinearEntry();
  iree::vm::ref<iree_hal_executable_t> executable;
  iree_hal_executable_t *executable_ptr = nullptr;
  iree_status_t exec_status = CreateDescriptorTtexExecutable(
      device_, host_allocator(), ep, &executable_ptr, /*include_aot=*/true);
  if (!iree_status_is_ok(exec_status)) {
    iree_status_ignore(exec_status);
    GTEST_SKIP() << "linear AOT test fixture is unavailable";
  }
  executable.assign(executable_ptr);

  constexpr uint32_t kInputElements = 2048;
  constexpr uint32_t kOutputElements = 1000;
  constexpr iree_device_size_t kInputBytes = kInputElements * sizeof(float);
  constexpr iree_device_size_t kWeightBytes =
      kOutputElements * kInputElements * sizeof(float);
  constexpr iree_device_size_t kBiasBytes = kOutputElements * sizeof(float);
  constexpr iree_device_size_t kOutputBytes = kOutputElements * sizeof(float);
  iree_hal_buffer_params_t buffer_params = {
      .usage = IREE_HAL_BUFFER_USAGE_DISPATCH_STORAGE |
               IREE_HAL_BUFFER_USAGE_TRANSFER,
      .access = IREE_HAL_MEMORY_ACCESS_ALL,
      .type = IREE_HAL_MEMORY_TYPE_DEVICE_LOCAL,
  };
  iree::vm::ref<iree_hal_buffer_t> input_buffer;
  iree::vm::ref<iree_hal_buffer_t> weight_buffer;
  iree::vm::ref<iree_hal_buffer_t> bias_buffer;
  iree::vm::ref<iree_hal_buffer_t> output_buffer;
  IREE_ASSERT_OK(iree_hal_allocator_allocate_buffer(
      device_allocator(), buffer_params, kInputBytes, &input_buffer));
  IREE_ASSERT_OK(iree_hal_allocator_allocate_buffer(
      device_allocator(), buffer_params, kWeightBytes, &weight_buffer));
  IREE_ASSERT_OK(iree_hal_allocator_allocate_buffer(
      device_allocator(), buffer_params, kBiasBytes, &bias_buffer));
  IREE_ASSERT_OK(iree_hal_allocator_allocate_buffer(
      device_allocator(), buffer_params, kOutputBytes, &output_buffer));

  std::vector<float> input(kInputElements, 1.0f);
  std::vector<float> weights(kOutputElements * kInputElements,
                             1.0f / kInputElements);
  std::vector<float> bias(kOutputElements, 0.25f);
  std::vector<float> output(kOutputElements, -1.0f);
  IREE_ASSERT_OK(iree_hal_buffer_map_write(input_buffer.get(), 0, input.data(),
                                           kInputBytes));
  IREE_ASSERT_OK(iree_hal_buffer_map_write(weight_buffer.get(), 0,
                                           weights.data(), kWeightBytes));
  IREE_ASSERT_OK(
      iree_hal_buffer_map_write(bias_buffer.get(), 0, bias.data(), kBiasBytes));
  IREE_ASSERT_OK(iree_hal_buffer_map_write(output_buffer.get(), 0,
                                           output.data(), kOutputBytes));

  iree::vm::ref<iree_hal_command_buffer_t> cmd_buffer;
  IREE_ASSERT_OK(iree_hal_tt_device_create_command_buffer(
      device_, IREE_HAL_COMMAND_BUFFER_MODE_ONE_SHOT,
      IREE_HAL_COMMAND_CATEGORY_DISPATCH, /*queue_affinity=*/0,
      /*binding_capacity=*/4, &cmd_buffer));
  IREE_ASSERT_OK(iree_hal_command_buffer_begin(cmd_buffer.get()));
  iree_hal_buffer_ref_t refs[4] = {
      iree_hal_make_buffer_ref(input_buffer.get(), 0, kInputBytes),
      iree_hal_make_buffer_ref(weight_buffer.get(), 0, kWeightBytes),
      iree_hal_make_buffer_ref(bias_buffer.get(), 0, kBiasBytes),
      iree_hal_make_buffer_ref(output_buffer.get(), 0, kOutputBytes),
  };
  iree_hal_dispatch_config_t config = {{0, 0, 0}, {8, 1, 1}};
  IREE_ASSERT_OK(iree_hal_command_buffer_dispatch(
      cmd_buffer.get(), executable.get(), /*export_ordinal=*/0, config,
      iree_const_byte_span_empty(), {4, refs}, IREE_HAL_DISPATCH_FLAG_NONE));
  IREE_ASSERT_OK(iree_hal_command_buffer_end(cmd_buffer.get()));

  iree_hal_semaphore_list_t no_semaphores = {0};
  iree_hal_buffer_binding_table_t no_binding_table = {0};
  IREE_ASSERT_OK(iree_hal_device_queue_execute(
      device_, /*queue_affinity=*/0, no_semaphores, no_semaphores,
      cmd_buffer.get(), no_binding_table, IREE_HAL_EXECUTE_FLAG_NONE));

  IREE_ASSERT_OK(iree_hal_buffer_map_read(output_buffer.get(), 0, output.data(),
                                          kOutputBytes));
  for (uint32_t i = 0; i < kOutputElements; ++i) {
    ASSERT_NEAR(output[i], 1.25f, 0.05f) << "classifier mismatch at " << i;
  }
}

TEST_F(DispatchTest, RejectsOutOfBoundsMatmulLikeOutput) {
  tt_iree_ttex_entry_point_desc_t ep = MakeMatmulLikeEntry();
  ep.matmul_like.output.offset[1] = 9;

  iree_hal_executable_t *executable = nullptr;
  iree_status_t status = CreateDescriptorTtexExecutable(
      device_, host_allocator(), ep, &executable);
  EXPECT_EQ(iree_status_code(status), IREE_STATUS_INVALID_ARGUMENT);
  iree_status_ignore(status);
  EXPECT_EQ(executable, nullptr);
}

TEST_F(DispatchTest, RecordsExecutableMatmulLikeDispatch) {
  tt_iree_ttex_entry_point_desc_t ep = MakeMatmulLikeEntry();
  iree::vm::ref<iree_hal_executable_t> executable;
  iree_hal_executable_t *executable_ptr = nullptr;
  IREE_ASSERT_OK(CreateDescriptorTtexExecutable(device_, host_allocator(), ep,
                                                &executable_ptr));
  executable.assign(executable_ptr);

  iree::vm::ref<iree_hal_command_buffer_t> cmd_buffer;
  IREE_ASSERT_OK(iree_hal_tt_device_create_command_buffer(
      device_, IREE_HAL_COMMAND_BUFFER_MODE_ONE_SHOT,
      IREE_HAL_COMMAND_CATEGORY_DISPATCH, /*queue_affinity=*/0,
      /*binding_capacity=*/3, &cmd_buffer));
  IREE_ASSERT_OK(iree_hal_command_buffer_begin(cmd_buffer.get()));

  constexpr iree_device_size_t kActivationBytes = 64 * 8 * 8 * sizeof(float);
  constexpr iree_device_size_t kWeightBytes = 64 * 64 * sizeof(float);
  constexpr iree_device_size_t kOutputBytes = 64 * 10 * 10 * sizeof(float);
  iree_hal_buffer_ref_t bindings[3] = {
      iree_hal_make_indirect_buffer_ref(0, 0, kActivationBytes),
      iree_hal_make_indirect_buffer_ref(1, 0, kWeightBytes),
      iree_hal_make_indirect_buffer_ref(2, 0, kOutputBytes),
  };
  iree_hal_dispatch_config_t config = {{0, 0, 0}, {1, 1, 1}};
  IREE_ASSERT_OK(iree_hal_command_buffer_dispatch(
      cmd_buffer.get(), executable.get(), /*export_ordinal=*/0, config,
      iree_const_byte_span_empty(), {3, bindings},
      IREE_HAL_DISPATCH_FLAG_NONE));
  IREE_ASSERT_OK(iree_hal_command_buffer_end(cmd_buffer.get()));
}

TEST_F(DispatchTest, ExecutesF32MatmulLikeProjection) {
  tt_iree_ttex_entry_point_desc_t ep = MakeMatmulLikeEntry();
  iree::vm::ref<iree_hal_executable_t> executable;
  iree_hal_executable_t *executable_ptr = nullptr;
  iree_status_t exec_status = CreateDescriptorTtexExecutable(
      device_, host_allocator(), ep, &executable_ptr, /*include_aot=*/true);
  if (!iree_status_is_ok(exec_status)) {
    iree_status_ignore(exec_status);
    GTEST_SKIP() << "matmul-like AOT test fixture is unavailable";
  }
  executable.assign(executable_ptr);

  constexpr uint32_t kChannels = 64;
  constexpr uint32_t kHeight = 8;
  constexpr uint32_t kWidth = 8;
  constexpr uint32_t kStorageHeight = 10;
  constexpr uint32_t kStorageWidth = 10;
  constexpr iree_device_size_t kActivationBytes =
      kChannels * kHeight * kWidth * sizeof(float);
  constexpr iree_device_size_t kWeightBytes =
      kChannels * kChannels * sizeof(float);
  constexpr iree_device_size_t kOutputBytes =
      kChannels * kStorageHeight * kStorageWidth * sizeof(float);

  iree_hal_buffer_params_t buffer_params = {
      .usage = IREE_HAL_BUFFER_USAGE_DISPATCH_STORAGE |
               IREE_HAL_BUFFER_USAGE_TRANSFER,
      .access = IREE_HAL_MEMORY_ACCESS_ALL,
      .type = IREE_HAL_MEMORY_TYPE_DEVICE_LOCAL,
  };
  iree::vm::ref<iree_hal_buffer_t> activation_buffer;
  iree::vm::ref<iree_hal_buffer_t> weight_buffer;
  iree::vm::ref<iree_hal_buffer_t> output_buffer;
  IREE_ASSERT_OK(iree_hal_allocator_allocate_buffer(
      device_allocator(), buffer_params, kActivationBytes, &activation_buffer));
  IREE_ASSERT_OK(iree_hal_allocator_allocate_buffer(
      device_allocator(), buffer_params, kWeightBytes, &weight_buffer));
  IREE_ASSERT_OK(iree_hal_allocator_allocate_buffer(
      device_allocator(), buffer_params, kOutputBytes, &output_buffer));

  std::vector<float> activation(kChannels * kHeight * kWidth, 1.0f);
  std::vector<float> weight(kChannels * kChannels, 0.015625f);
  std::vector<float> output(kChannels * kStorageHeight * kStorageWidth, -3.0f);
  IREE_ASSERT_OK(iree_hal_buffer_map_write(
      activation_buffer.get(), 0, activation.data(), kActivationBytes));
  IREE_ASSERT_OK(iree_hal_buffer_map_write(weight_buffer.get(), 0,
                                           weight.data(), kWeightBytes));
  IREE_ASSERT_OK(iree_hal_buffer_map_write(output_buffer.get(), 0,
                                           output.data(), kOutputBytes));

  iree::vm::ref<iree_hal_command_buffer_t> cmd_buffer;
  IREE_ASSERT_OK(iree_hal_tt_device_create_command_buffer(
      device_, IREE_HAL_COMMAND_BUFFER_MODE_ONE_SHOT,
      IREE_HAL_COMMAND_CATEGORY_DISPATCH, /*queue_affinity=*/0,
      /*binding_capacity=*/3, &cmd_buffer));
  IREE_ASSERT_OK(iree_hal_command_buffer_begin(cmd_buffer.get()));
  iree_hal_buffer_ref_t refs[3] = {
      iree_hal_make_buffer_ref(activation_buffer.get(), 0, kActivationBytes),
      iree_hal_make_buffer_ref(weight_buffer.get(), 0, kWeightBytes),
      iree_hal_make_buffer_ref(output_buffer.get(), 0, kOutputBytes),
  };
  iree_hal_dispatch_config_t config = {{0, 0, 0}, {2, 2, 1}};
  IREE_ASSERT_OK(iree_hal_command_buffer_dispatch(
      cmd_buffer.get(), executable.get(), /*export_ordinal=*/0, config,
      iree_const_byte_span_empty(), {3, refs}, IREE_HAL_DISPATCH_FLAG_NONE));
  IREE_ASSERT_OK(iree_hal_command_buffer_end(cmd_buffer.get()));

  iree_hal_semaphore_list_t no_semaphores = {0};
  iree_hal_buffer_binding_table_t no_binding_table = {0};
  IREE_ASSERT_OK(iree_hal_device_queue_execute(
      device_, /*queue_affinity=*/0, no_semaphores, no_semaphores,
      cmd_buffer.get(), no_binding_table, IREE_HAL_EXECUTE_FLAG_NONE));

  IREE_ASSERT_OK(iree_hal_buffer_map_read(output_buffer.get(), 0, output.data(),
                                          kOutputBytes));
  for (uint32_t c = 0; c < kChannels; ++c) {
    for (uint32_t h = 0; h < kStorageHeight; ++h) {
      for (uint32_t w = 0; w < kStorageWidth; ++w) {
        bool is_logical_output = h >= 1 && h <= 8 && w >= 1 && w <= 8;
        float expected = is_logical_output ? 0.6875f : -3.0f;
        uint32_t index = (c * kStorageHeight + h) * kStorageWidth + w;
        ASSERT_NEAR(output[index], expected, 0.01f)
            << "projection mismatch at [" << c << ", " << h << ", " << w << "]";
      }
    }
  }
}

TEST_F(DispatchTest, ExecutesDeviceNativeF32MatmulLikeProjection) {
  tt_iree_ttex_entry_point_desc_t ep = MakeDeviceNativeMatmulLikeEntry();
  iree::vm::ref<iree_hal_executable_t> executable;
  iree_hal_executable_t *executable_ptr = nullptr;
#ifdef TT_IREE_ENABLE_MOCK
  constexpr bool kIncludeAot = false;
#else
  constexpr bool kIncludeAot = true;
#endif
  iree_status_t exec_status = CreateDescriptorTtexExecutable(
      device_, host_allocator(), ep, &executable_ptr, kIncludeAot);
  if (!iree_status_is_ok(exec_status)) {
    iree_status_ignore(exec_status);
    GTEST_SKIP() << "device-native F32 matmul AOT fixture is unavailable";
  }
  executable.assign(executable_ptr);

  constexpr uint32_t kChannels = 64;
  constexpr uint32_t kHeight = 8;
  constexpr uint32_t kWidth = 8;
  constexpr uint32_t kStorageHeight = 10;
  constexpr uint32_t kStorageWidth = 10;
  constexpr iree_device_size_t kActivationBytes =
      kChannels * kHeight * kWidth * sizeof(float);
  constexpr iree_device_size_t kWeightBytes =
      kChannels * kChannels * sizeof(float);
  constexpr iree_device_size_t kOutputBytes =
      kChannels * kStorageHeight * kStorageWidth * sizeof(float);
  iree_hal_buffer_params_t buffer_params = {
      .usage = IREE_HAL_BUFFER_USAGE_DISPATCH_STORAGE |
               IREE_HAL_BUFFER_USAGE_TRANSFER,
      .access = IREE_HAL_MEMORY_ACCESS_ALL,
      .type = IREE_HAL_MEMORY_TYPE_DEVICE_LOCAL,
  };
  iree::vm::ref<iree_hal_buffer_t> activation_buffer;
  iree::vm::ref<iree_hal_buffer_t> weight_buffer;
  iree::vm::ref<iree_hal_buffer_t> output_buffer;
  IREE_ASSERT_OK(iree_hal_allocator_allocate_buffer(
      device_allocator(), buffer_params, kActivationBytes, &activation_buffer));
  IREE_ASSERT_OK(iree_hal_allocator_allocate_buffer(
      device_allocator(), buffer_params, kWeightBytes, &weight_buffer));
  IREE_ASSERT_OK(iree_hal_allocator_allocate_buffer(
      device_allocator(), buffer_params, kOutputBytes, &output_buffer));

  std::vector<float> activation(kChannels * kHeight * kWidth, 1.0f);
  std::vector<float> weight(kChannels * kChannels, 0.015625f);
  std::vector<float> output(kChannels * kStorageHeight * kStorageWidth, -3.0f);
  IREE_ASSERT_OK(iree_hal_buffer_map_write(
      activation_buffer.get(), 0, activation.data(), kActivationBytes));
  IREE_ASSERT_OK(iree_hal_buffer_map_write(weight_buffer.get(), 0,
                                           weight.data(), kWeightBytes));
  IREE_ASSERT_OK(iree_hal_buffer_map_write(output_buffer.get(), 0,
                                           output.data(), kOutputBytes));

  iree::vm::ref<iree_hal_command_buffer_t> cmd_buffer;
  IREE_ASSERT_OK(iree_hal_tt_device_create_command_buffer(
      device_, IREE_HAL_COMMAND_BUFFER_MODE_ONE_SHOT,
      IREE_HAL_COMMAND_CATEGORY_DISPATCH, /*queue_affinity=*/0,
      /*binding_capacity=*/3, &cmd_buffer));
  IREE_ASSERT_OK(iree_hal_command_buffer_begin(cmd_buffer.get()));
  iree_hal_buffer_ref_t refs[3] = {
      iree_hal_make_buffer_ref(activation_buffer.get(), 0, kActivationBytes),
      iree_hal_make_buffer_ref(weight_buffer.get(), 0, kWeightBytes),
      iree_hal_make_buffer_ref(output_buffer.get(), 0, kOutputBytes),
  };
  iree_hal_dispatch_config_t config = {{0, 0, 0}, {2, 2, 1}};
  IREE_ASSERT_OK(iree_hal_command_buffer_dispatch(
      cmd_buffer.get(), executable.get(), /*export_ordinal=*/0, config,
      iree_const_byte_span_empty(), {3, refs}, IREE_HAL_DISPATCH_FLAG_NONE));
  IREE_ASSERT_OK(iree_hal_command_buffer_end(cmd_buffer.get()));

  iree_hal_semaphore_list_t no_semaphores = {0};
  iree_hal_buffer_binding_table_t no_binding_table = {0};
  IREE_ASSERT_OK(iree_hal_device_queue_execute(
      device_, /*queue_affinity=*/0, no_semaphores, no_semaphores,
      cmd_buffer.get(), no_binding_table, IREE_HAL_EXECUTE_FLAG_NONE));

  IREE_ASSERT_OK(iree_hal_buffer_map_read(output_buffer.get(), 0, output.data(),
                                          kOutputBytes));
  for (uint32_t c = 0; c < kChannels; ++c) {
    for (uint32_t h = 0; h < kStorageHeight; ++h) {
      for (uint32_t w = 0; w < kStorageWidth; ++w) {
        const bool is_logical_output = h >= 1 && h <= 8 && w >= 1 && w <= 8;
        const float expected = is_logical_output ? 1.0f : -3.0f;
        const uint32_t index = (c * kStorageHeight + h) * kStorageWidth + w;
        ASSERT_NEAR(output[index], expected, 0.01f)
            << "native projection mismatch at [" << c << ", " << h << ", " << w
            << "]";
      }
    }
  }
}

TEST_F(DispatchTest, ExecutesDeviceNativeF32MatmulLikeMTail) {
  tt_iree_ttex_entry_point_desc_t ep = MakeDeviceNativeMatmulLikeMTailEntry();
  iree::vm::ref<iree_hal_executable_t> executable;
  iree_hal_executable_t *executable_ptr = nullptr;
#ifdef TT_IREE_ENABLE_MOCK
  constexpr bool kIncludeAot = false;
#else
  constexpr bool kIncludeAot = true;
#endif
  iree_status_t exec_status = CreateDescriptorTtexExecutable(
      device_, host_allocator(), ep, &executable_ptr, kIncludeAot);
  if (!iree_status_is_ok(exec_status)) {
    iree_status_ignore(exec_status);
    GTEST_SKIP() << "device-native F32 matmul AOT fixture is unavailable";
  }
  executable.assign(executable_ptr);

  constexpr uint32_t kM = 49;
  constexpr uint32_t kN = 64;
  constexpr uint32_t kK = 64;
  constexpr uint32_t kOutputStorageM = 53;
  constexpr uint32_t kOutputMOffset = 2;
  constexpr iree_device_size_t kActivationBytes = kK * kM * sizeof(float);
  constexpr iree_device_size_t kWeightBytes = kN * kK * sizeof(float);
  constexpr iree_device_size_t kOutputBytes =
      kN * kOutputStorageM * sizeof(float);
  iree_hal_buffer_params_t buffer_params = {
      .usage = IREE_HAL_BUFFER_USAGE_DISPATCH_STORAGE |
               IREE_HAL_BUFFER_USAGE_TRANSFER,
      .access = IREE_HAL_MEMORY_ACCESS_ALL,
      .type = IREE_HAL_MEMORY_TYPE_DEVICE_LOCAL,
  };
  iree::vm::ref<iree_hal_buffer_t> activation_buffer;
  iree::vm::ref<iree_hal_buffer_t> weight_buffer;
  iree::vm::ref<iree_hal_buffer_t> output_buffer;
  IREE_ASSERT_OK(iree_hal_allocator_allocate_buffer(
      device_allocator(), buffer_params, kActivationBytes, &activation_buffer));
  IREE_ASSERT_OK(iree_hal_allocator_allocate_buffer(
      device_allocator(), buffer_params, kWeightBytes, &weight_buffer));
  IREE_ASSERT_OK(iree_hal_allocator_allocate_buffer(
      device_allocator(), buffer_params, kOutputBytes, &output_buffer));

  std::vector<float> activation(kK * kM, 1.0f);
  std::vector<float> weight(kN * kK, 0.015625f);
  std::vector<float> output(kN * kOutputStorageM, -3.0f);
  IREE_ASSERT_OK(iree_hal_buffer_map_write(
      activation_buffer.get(), 0, activation.data(), kActivationBytes));
  IREE_ASSERT_OK(iree_hal_buffer_map_write(weight_buffer.get(), 0,
                                           weight.data(), kWeightBytes));
  IREE_ASSERT_OK(iree_hal_buffer_map_write(output_buffer.get(), 0,
                                           output.data(), kOutputBytes));

  iree::vm::ref<iree_hal_command_buffer_t> cmd_buffer;
  IREE_ASSERT_OK(iree_hal_tt_device_create_command_buffer(
      device_, IREE_HAL_COMMAND_BUFFER_MODE_ONE_SHOT,
      IREE_HAL_COMMAND_CATEGORY_DISPATCH, /*queue_affinity=*/0,
      /*binding_capacity=*/3, &cmd_buffer));
  IREE_ASSERT_OK(iree_hal_command_buffer_begin(cmd_buffer.get()));
  iree_hal_buffer_ref_t refs[3] = {
      iree_hal_make_buffer_ref(activation_buffer.get(), 0, kActivationBytes),
      iree_hal_make_buffer_ref(weight_buffer.get(), 0, kWeightBytes),
      iree_hal_make_buffer_ref(output_buffer.get(), 0, kOutputBytes),
  };
  iree_hal_dispatch_config_t config = {{0, 0, 0}, {2, 2, 1}};
  IREE_ASSERT_OK(iree_hal_command_buffer_dispatch(
      cmd_buffer.get(), executable.get(), /*export_ordinal=*/0, config,
      iree_const_byte_span_empty(), {3, refs}, IREE_HAL_DISPATCH_FLAG_NONE));
  IREE_ASSERT_OK(iree_hal_command_buffer_end(cmd_buffer.get()));

  iree_hal_semaphore_list_t no_semaphores = {0};
  iree_hal_buffer_binding_table_t no_binding_table = {0};
  IREE_ASSERT_OK(iree_hal_device_queue_execute(
      device_, /*queue_affinity=*/0, no_semaphores, no_semaphores,
      cmd_buffer.get(), no_binding_table, IREE_HAL_EXECUTE_FLAG_NONE));

  IREE_ASSERT_OK(iree_hal_buffer_map_read(output_buffer.get(), 0, output.data(),
                                          kOutputBytes));
  for (uint32_t n = 0; n < kN; ++n) {
    for (uint32_t m = 0; m < kOutputStorageM; ++m) {
      const bool is_logical_output =
          m >= kOutputMOffset && m < kOutputMOffset + kM;
      const float expected = is_logical_output ? 1.0f : -3.0f;
      const uint32_t index = n * kOutputStorageM + m;
      ASSERT_NEAR(output[index], expected, 0.01f)
          << "native M-tail mismatch at [" << n << ", " << m << "]";
    }
  }
}

TEST_F(DispatchTest,
       ExecutesDeviceNativeF32RuntimePerChannelMatmulLikeProjection) {
  tt_iree_ttex_entry_point_desc_t ep =
      MakeRuntimePerChannelDeviceNativeMatmulLikeEntry();
  iree::vm::ref<iree_hal_executable_t> executable;
  iree_hal_executable_t *executable_ptr = nullptr;
#ifdef TT_IREE_ENABLE_MOCK
  constexpr bool kIncludeAot = false;
#else
  constexpr bool kIncludeAot = true;
#endif
  iree_status_t exec_status = CreateDescriptorTtexExecutable(
      device_, host_allocator(), ep, &executable_ptr, kIncludeAot);
  if (!iree_status_is_ok(exec_status)) {
    iree_status_ignore(exec_status);
    GTEST_SKIP() << "device-native F32 matmul AOT fixture is unavailable";
  }
  executable.assign(executable_ptr);

  constexpr uint32_t kM = 49;
  constexpr uint32_t kN = 64;
  constexpr uint32_t kK = 64;
  constexpr uint32_t kOutputStorageM = 53;
  constexpr uint32_t kOutputMOffset = 2;
  constexpr iree_device_size_t kActivationBytes = kK * kM * sizeof(float);
  constexpr iree_device_size_t kWeightBytes = kN * kK * sizeof(float);
  constexpr iree_device_size_t kOutputBytes =
      kN * kOutputStorageM * sizeof(float);
  constexpr iree_device_size_t kChannelBytes = kN * sizeof(float);
  iree_hal_buffer_params_t buffer_params = {
      .usage = IREE_HAL_BUFFER_USAGE_DISPATCH_STORAGE |
               IREE_HAL_BUFFER_USAGE_TRANSFER,
      .access = IREE_HAL_MEMORY_ACCESS_ALL,
      .type = IREE_HAL_MEMORY_TYPE_DEVICE_LOCAL,
  };
  std::array<iree::vm::ref<iree_hal_buffer_t>, 6> buffers;
  const std::array<iree_device_size_t, 6> buffer_sizes = {
      kActivationBytes, kWeightBytes,  kOutputBytes,
      kChannelBytes,    kChannelBytes, kChannelBytes,
  };
  for (size_t i = 0; i < buffers.size(); ++i) {
    IREE_ASSERT_OK(iree_hal_allocator_allocate_buffer(
        device_allocator(), buffer_params, buffer_sizes[i], &buffers[i]));
  }

  std::vector<float> activation(kK * kM, 1.0f);
  std::vector<float> weight(kN * kK, 0.015625f);
  std::vector<float> output(kN * kOutputStorageM, -3.0f);
  std::vector<float> mean(kN);
  std::vector<float> channel_scale(kN);
  std::vector<float> channel_bias(kN);
  for (uint32_t n = 0; n < kN; ++n) {
    const bool even = n % 2 == 0;
    mean[n] = even ? 1.0f : 2.0f;
    channel_scale[n] = even ? 0.5f : 0.25f;
    channel_bias[n] = even ? 0.25f : -0.25f;
  }
  const std::array<const void *, 6> data = {
      activation.data(), weight.data(),        output.data(),
      mean.data(),       channel_scale.data(), channel_bias.data(),
  };
  for (size_t i = 0; i < buffers.size(); ++i) {
    IREE_ASSERT_OK(iree_hal_buffer_map_write(buffers[i].get(), 0, data[i],
                                             buffer_sizes[i]));
  }

  iree::vm::ref<iree_hal_command_buffer_t> cmd_buffer;
  IREE_ASSERT_OK(iree_hal_tt_device_create_command_buffer(
      device_, IREE_HAL_COMMAND_BUFFER_MODE_ONE_SHOT,
      IREE_HAL_COMMAND_CATEGORY_DISPATCH, /*queue_affinity=*/0,
      /*binding_capacity=*/6, &cmd_buffer));
  IREE_ASSERT_OK(iree_hal_command_buffer_begin(cmd_buffer.get()));
  std::array<iree_hal_buffer_ref_t, 6> refs;
  for (size_t i = 0; i < refs.size(); ++i) {
    refs[i] = iree_hal_make_buffer_ref(buffers[i].get(), 0, buffer_sizes[i]);
  }
  iree_hal_dispatch_config_t config = {{0, 0, 0}, {2, 2, 1}};
  IREE_ASSERT_OK(iree_hal_command_buffer_dispatch(
      cmd_buffer.get(), executable.get(), /*export_ordinal=*/0, config,
      iree_const_byte_span_empty(), {refs.size(), refs.data()},
      IREE_HAL_DISPATCH_FLAG_NONE));
  IREE_ASSERT_OK(iree_hal_command_buffer_end(cmd_buffer.get()));

  iree_hal_semaphore_list_t no_semaphores = {0};
  iree_hal_buffer_binding_table_t no_binding_table = {0};
  IREE_ASSERT_OK(iree_hal_device_queue_execute(
      device_, /*queue_affinity=*/0, no_semaphores, no_semaphores,
      cmd_buffer.get(), no_binding_table, IREE_HAL_EXECUTE_FLAG_NONE));

  IREE_ASSERT_OK(iree_hal_buffer_map_read(buffers[2].get(), 0, output.data(),
                                          kOutputBytes));
  for (uint32_t n = 0; n < kN; ++n) {
    for (uint32_t m = 0; m < kOutputStorageM; ++m) {
      const bool is_logical_output =
          m >= kOutputMOffset && m < kOutputMOffset + kM;
      const float expected =
          is_logical_output ? (n % 2 == 0 ? 0.25f : 0.0f) : -3.0f;
      const uint32_t index = n * kOutputStorageM + m;
      ASSERT_NEAR(output[index], expected, 0.01f)
          << "runtime per-channel native mismatch at [" << n << ", " << m
          << "]";
    }
  }
}

TEST_F(DispatchTest, ExecutesPerChannelF32MatmulLikeProjection) {
  tt_iree_ttex_entry_point_desc_t ep = MakePerChannelMatmulLikeEntry();
  iree::vm::ref<iree_hal_executable_t> executable;
  iree_hal_executable_t *executable_ptr = nullptr;
#ifdef TT_IREE_ENABLE_MOCK
  constexpr bool kIncludeAot = false;
#else
  constexpr bool kIncludeAot = true;
#endif
  IREE_ASSERT_OK(CreateDescriptorTtexExecutable(device_, host_allocator(), ep,
                                                &executable_ptr, kIncludeAot));
  executable.assign(executable_ptr);

  const iree_hal_tt_kernel_params_t *params = nullptr;
  IREE_ASSERT_OK(iree_hal_tt_executable_lookup_kernel_params(
      executable.get(), /*entry_point=*/0, &params));
  ASSERT_NE(params, nullptr);
  EXPECT_EQ(params->matmul_like.epilogue_channel_count, 64u);
  EXPECT_FLOAT_EQ(params->matmul_like.epilogue_scales[1], 0.25f);
  EXPECT_FLOAT_EQ(params->matmul_like.epilogue_biases[63], 63.0f);

  constexpr uint32_t kChannels = 64;
  constexpr uint32_t kHeight = 8;
  constexpr uint32_t kWidth = 8;
  constexpr uint32_t kStorageHeight = 10;
  constexpr uint32_t kStorageWidth = 10;
  constexpr iree_device_size_t kActivationBytes =
      kChannels * kHeight * kWidth * sizeof(float);
  constexpr iree_device_size_t kWeightBytes =
      kChannels * kChannels * sizeof(float);
  constexpr iree_device_size_t kOutputBytes =
      kChannels * kStorageHeight * kStorageWidth * sizeof(float);
  iree_hal_buffer_params_t buffer_params = {
      .usage = IREE_HAL_BUFFER_USAGE_DISPATCH_STORAGE |
               IREE_HAL_BUFFER_USAGE_TRANSFER,
      .access = IREE_HAL_MEMORY_ACCESS_ALL,
      .type = IREE_HAL_MEMORY_TYPE_DEVICE_LOCAL,
  };
  iree::vm::ref<iree_hal_buffer_t> activation_buffer;
  iree::vm::ref<iree_hal_buffer_t> weight_buffer;
  iree::vm::ref<iree_hal_buffer_t> output_buffer;
  IREE_ASSERT_OK(iree_hal_allocator_allocate_buffer(
      device_allocator(), buffer_params, kActivationBytes, &activation_buffer));
  IREE_ASSERT_OK(iree_hal_allocator_allocate_buffer(
      device_allocator(), buffer_params, kWeightBytes, &weight_buffer));
  IREE_ASSERT_OK(iree_hal_allocator_allocate_buffer(
      device_allocator(), buffer_params, kOutputBytes, &output_buffer));

  std::vector<float> activation(kChannels * kHeight * kWidth, 1.0f);
  std::vector<float> weight(kChannels * kChannels, 0.015625f);
  std::vector<float> output(kChannels * kStorageHeight * kStorageWidth, -3.0f);
  IREE_ASSERT_OK(iree_hal_buffer_map_write(
      activation_buffer.get(), 0, activation.data(), kActivationBytes));
  IREE_ASSERT_OK(iree_hal_buffer_map_write(weight_buffer.get(), 0,
                                           weight.data(), kWeightBytes));
  IREE_ASSERT_OK(iree_hal_buffer_map_write(output_buffer.get(), 0,
                                           output.data(), kOutputBytes));

  iree::vm::ref<iree_hal_command_buffer_t> cmd_buffer;
  IREE_ASSERT_OK(iree_hal_tt_device_create_command_buffer(
      device_, IREE_HAL_COMMAND_BUFFER_MODE_ONE_SHOT,
      IREE_HAL_COMMAND_CATEGORY_DISPATCH, /*queue_affinity=*/0,
      /*binding_capacity=*/3, &cmd_buffer));
  IREE_ASSERT_OK(iree_hal_command_buffer_begin(cmd_buffer.get()));
  iree_hal_buffer_ref_t refs[3] = {
      iree_hal_make_buffer_ref(activation_buffer.get(), 0, kActivationBytes),
      iree_hal_make_buffer_ref(weight_buffer.get(), 0, kWeightBytes),
      iree_hal_make_buffer_ref(output_buffer.get(), 0, kOutputBytes),
  };
  iree_hal_dispatch_config_t config = {{0, 0, 0}, {2, 2, 1}};
  IREE_ASSERT_OK(iree_hal_command_buffer_dispatch(
      cmd_buffer.get(), executable.get(), /*export_ordinal=*/0, config,
      iree_const_byte_span_empty(), {3, refs}, IREE_HAL_DISPATCH_FLAG_NONE));
  IREE_ASSERT_OK(iree_hal_command_buffer_end(cmd_buffer.get()));

  iree_hal_semaphore_list_t no_semaphores = {0};
  iree_hal_buffer_binding_table_t no_binding_table = {0};
  IREE_ASSERT_OK(iree_hal_device_queue_execute(
      device_, /*queue_affinity=*/0, no_semaphores, no_semaphores,
      cmd_buffer.get(), no_binding_table, IREE_HAL_EXECUTE_FLAG_NONE));

  IREE_ASSERT_OK(iree_hal_buffer_map_read(output_buffer.get(), 0, output.data(),
                                          kOutputBytes));
  for (uint32_t c = 0; c < kChannels; ++c) {
    for (uint32_t h = 0; h < kStorageHeight; ++h) {
      for (uint32_t w = 0; w < kStorageWidth; ++w) {
        bool is_logical_output = h >= 1 && h <= 8 && w >= 1 && w <= 8;
        float expected = -3.0f;
        if (is_logical_output)
          expected = static_cast<float>(c) + (c % 2 == 0 ? 0.5f : 0.25f);
        uint32_t index = (c * kStorageHeight + h) * kStorageWidth + w;
        ASSERT_NEAR(output[index], expected, 0.01f)
            << "per-channel projection mismatch at [" << c << ", " << h << ", "
            << w << "]";
      }
    }
  }
}

TEST_F(DispatchTest, PreservesResidualMatmulLikeDescriptorMetadata) {
  tt_iree_ttex_entry_point_desc_t ep = MakeResidualMatmulLikeEntry();
  iree::vm::ref<iree_hal_executable_t> executable;
  iree_hal_executable_t *executable_ptr = nullptr;
  IREE_ASSERT_OK(CreateDescriptorTtexExecutable(device_, host_allocator(), ep,
                                                &executable_ptr));
  executable.assign(executable_ptr);

  const iree_hal_tt_kernel_params_t *params = nullptr;
  IREE_ASSERT_OK(iree_hal_tt_executable_lookup_kernel_params(
      executable.get(), /*entry_point=*/0, &params));
  ASSERT_NE(params, nullptr);
  EXPECT_EQ(params->matmul_like.m, 64u);
  EXPECT_EQ(params->matmul_like.n, 256u);
  EXPECT_EQ(params->matmul_like.lhs.layout,
            TT_IREE_TTEX_TENSOR_LAYOUT_CHANNEL_MAJOR);
  EXPECT_EQ(params->matmul_like.lhs.byte_offset, 25600u);
  EXPECT_EQ(params->matmul_like.residual.binding, 0u);
  EXPECT_EQ(params->matmul_like.residual.byte_offset, 41984u);
  EXPECT_FLOAT_EQ(params->matmul_like.residual_scale, 0.25f);
  EXPECT_FLOAT_EQ(params->matmul_like.residual_bias, 0.125f);
}

TEST_F(DispatchTest, AcceptsAliasedResidualMatmulLikeStorage) {
  tt_iree_ttex_entry_point_desc_t ep = MakeResidualMatmulLikeEntry();
  ep.matmul_like.residual.byte_offset = ep.matmul_like.lhs.byte_offset;

  iree::vm::ref<iree_hal_executable_t> executable;
  iree_hal_executable_t *executable_ptr = nullptr;
  IREE_ASSERT_OK(CreateDescriptorTtexExecutable(device_, host_allocator(), ep,
                                                &executable_ptr));
  executable.assign(executable_ptr);

  const iree_hal_tt_kernel_params_t *params = nullptr;
  IREE_ASSERT_OK(iree_hal_tt_executable_lookup_kernel_params(
      executable.get(), /*entry_point=*/0, &params));
  ASSERT_NE(params, nullptr);
  EXPECT_EQ(params->matmul_like.residual.binding,
            params->matmul_like.lhs.binding);
  EXPECT_EQ(params->matmul_like.residual.byte_offset,
            params->matmul_like.lhs.byte_offset);
}

TEST_F(DispatchTest, AcceptsResidualMatmulLikeDispatch) {
  tt_iree_ttex_entry_point_desc_t ep = MakeResidualMatmulLikeEntry();
  iree::vm::ref<iree_hal_executable_t> executable;
  iree_hal_executable_t *executable_ptr = nullptr;
  IREE_ASSERT_OK(CreateDescriptorTtexExecutable(device_, host_allocator(), ep,
                                                &executable_ptr));
  executable.assign(executable_ptr);

  iree::vm::ref<iree_hal_command_buffer_t> cmd_buffer;
  IREE_ASSERT_OK(iree_hal_tt_device_create_command_buffer(
      device_, IREE_HAL_COMMAND_BUFFER_MODE_ONE_SHOT,
      IREE_HAL_COMMAND_CATEGORY_DISPATCH, /*queue_affinity=*/0,
      /*binding_capacity=*/3, &cmd_buffer));
  IREE_ASSERT_OK(iree_hal_command_buffer_begin(cmd_buffer.get()));

  iree_hal_buffer_ref_t bindings[3] = {
      iree_hal_make_indirect_buffer_ref(0, 0, 107520),
      iree_hal_make_indirect_buffer_ref(1, 0, 65536),
      iree_hal_make_indirect_buffer_ref(2, 0, 65536),
  };
  iree_hal_dispatch_config_t config = {{0, 0, 0}, {1, 1, 1}};
  IREE_ASSERT_OK(iree_hal_command_buffer_dispatch(
      cmd_buffer.get(), executable.get(), /*export_ordinal=*/0, config,
      iree_const_byte_span_empty(), {3, bindings},
      IREE_HAL_DISPATCH_FLAG_NONE));
  IREE_ASSERT_OK(iree_hal_command_buffer_end(cmd_buffer.get()));
}

TEST_F(DispatchTest, RejectsUndersizedResidualMatmulLikeBinding) {
  tt_iree_ttex_entry_point_desc_t ep = MakeResidualMatmulLikeEntry();
  iree::vm::ref<iree_hal_executable_t> executable;
  iree_hal_executable_t *executable_ptr = nullptr;
  IREE_ASSERT_OK(CreateDescriptorTtexExecutable(device_, host_allocator(), ep,
                                                &executable_ptr));
  executable.assign(executable_ptr);

  iree::vm::ref<iree_hal_command_buffer_t> cmd_buffer;
  IREE_ASSERT_OK(iree_hal_tt_device_create_command_buffer(
      device_, IREE_HAL_COMMAND_BUFFER_MODE_ONE_SHOT,
      IREE_HAL_COMMAND_CATEGORY_DISPATCH, /*queue_affinity=*/0,
      /*binding_capacity=*/3, &cmd_buffer));
  IREE_ASSERT_OK(iree_hal_command_buffer_begin(cmd_buffer.get()));

  iree_hal_buffer_ref_t bindings[3] = {
      iree_hal_make_indirect_buffer_ref(0, 0, 107519),
      iree_hal_make_indirect_buffer_ref(1, 0, 65536),
      iree_hal_make_indirect_buffer_ref(2, 0, 65536),
  };
  iree_hal_dispatch_config_t config = {{0, 0, 0}, {1, 1, 1}};
  iree_status_t status = iree_hal_command_buffer_dispatch(
      cmd_buffer.get(), executable.get(), /*export_ordinal=*/0, config,
      iree_const_byte_span_empty(), {3, bindings}, IREE_HAL_DISPATCH_FLAG_NONE);
  EXPECT_EQ(iree_status_code(status), IREE_STATUS_OUT_OF_RANGE);
  iree_status_ignore(status);
}

TEST_F(DispatchTest, PreservesConv2DDescriptorMetadata) {
  tt_iree_ttex_entry_point_desc_t ep = MakeConv2DEntry();
  iree::vm::ref<iree_hal_executable_t> executable;
  iree_hal_executable_t *executable_ptr = nullptr;
  IREE_ASSERT_OK(CreateDescriptorTtexExecutable(device_, host_allocator(), ep,
                                                &executable_ptr));
  executable.assign(executable_ptr);

  const iree_hal_tt_kernel_params_t *params = nullptr;
  IREE_ASSERT_OK(iree_hal_tt_executable_lookup_kernel_params(
      executable.get(), /*entry_point=*/0, &params));
  ASSERT_NE(params, nullptr);
  EXPECT_EQ(params->dispatch_kind, TT_IREE_TTEX_DISPATCH_KIND_CONVOLUTION);
  EXPECT_EQ(params->conv2d.input.shape[1], 10u);
  EXPECT_EQ(params->conv2d.filter.layout, TT_IREE_TTEX_TENSOR_LAYOUT_OIHW);
  EXPECT_EQ(params->conv2d.filter.shape[3], 3u);
  EXPECT_EQ(params->conv2d.output.shape[1], 8u);
  EXPECT_EQ(params->conv2d.groups, 1u);
  EXPECT_FLOAT_EQ(params->conv2d.epilogue_scale, 0.25f);
  EXPECT_FLOAT_EQ(params->conv2d.epilogue_bias, 0.125f);
}

TEST_F(DispatchTest, ExecutesF32PaddedCopyAndPreservesBorder) {
  tt_iree_ttex_entry_point_desc_t ep = MakePaddedCopyEntry();
  iree::vm::ref<iree_hal_executable_t> executable;
  iree_hal_executable_t *executable_ptr = nullptr;
  IREE_ASSERT_OK(CreateDescriptorTtexExecutable(device_, host_allocator(), ep,
                                                &executable_ptr,
#ifdef TT_IREE_ENABLE_AOT_TESTS
                                                /*include_aot=*/true
#else
                                                /*include_aot=*/false
#endif
                                                ));
  executable.assign(executable_ptr);

  const iree_hal_tt_kernel_params_t *params = nullptr;
  IREE_ASSERT_OK(iree_hal_tt_executable_lookup_kernel_params(
      executable.get(), /*entry_point=*/0, &params));
  ASSERT_NE(params, nullptr);
  EXPECT_EQ(params->dispatch_kind, TT_IREE_TTEX_DISPATCH_KIND_PADDED_COPY);
  EXPECT_EQ(params->padded_copy.destination.storage_shape[1], 4u);
  EXPECT_EQ(params->padded_copy.destination.offset[1], 1u);

  constexpr iree_device_size_t kSourceBytes = 4 * sizeof(float);
  constexpr iree_device_size_t kDestinationBytes = 16 * sizeof(float);
  iree_hal_buffer_params_t buffer_params = {
      .usage = IREE_HAL_BUFFER_USAGE_DISPATCH_STORAGE |
               IREE_HAL_BUFFER_USAGE_TRANSFER,
      .access = IREE_HAL_MEMORY_ACCESS_ALL,
      .type = IREE_HAL_MEMORY_TYPE_DEVICE_LOCAL,
  };
  iree::vm::ref<iree_hal_buffer_t> source_buffer;
  iree::vm::ref<iree_hal_buffer_t> destination_buffer;
  IREE_ASSERT_OK(iree_hal_allocator_allocate_buffer(
      device_allocator(), buffer_params, kSourceBytes, &source_buffer));
  IREE_ASSERT_OK(iree_hal_allocator_allocate_buffer(
      device_allocator(), buffer_params, kDestinationBytes,
      &destination_buffer));

  std::vector<float> source = {1.0f, 2.0f, 3.0f, 4.0f};
  std::vector<float> destination(16, -7.0f);
  IREE_ASSERT_OK(iree_hal_buffer_map_write(source_buffer.get(), 0,
                                           source.data(), kSourceBytes));
  IREE_ASSERT_OK(iree_hal_buffer_map_write(
      destination_buffer.get(), 0, destination.data(), kDestinationBytes));

  iree::vm::ref<iree_hal_command_buffer_t> cmd_buffer;
  IREE_ASSERT_OK(iree_hal_tt_device_create_command_buffer(
      device_, IREE_HAL_COMMAND_BUFFER_MODE_ONE_SHOT,
      IREE_HAL_COMMAND_CATEGORY_DISPATCH, /*queue_affinity=*/0,
      /*binding_capacity=*/2, &cmd_buffer));
  IREE_ASSERT_OK(iree_hal_command_buffer_begin(cmd_buffer.get()));
  iree_hal_buffer_ref_t refs[2] = {
      iree_hal_make_buffer_ref(source_buffer.get(), 0, kSourceBytes),
      iree_hal_make_buffer_ref(destination_buffer.get(), 0, kDestinationBytes),
  };
  iree_hal_dispatch_config_t config = {{0, 0, 0}, {1, 1, 1}};
  IREE_ASSERT_OK(iree_hal_command_buffer_dispatch(
      cmd_buffer.get(), executable.get(), /*export_ordinal=*/0, config,
      iree_const_byte_span_empty(), {2, refs}, IREE_HAL_DISPATCH_FLAG_NONE));
  IREE_ASSERT_OK(iree_hal_command_buffer_end(cmd_buffer.get()));

  iree_hal_semaphore_list_t no_semaphores = {0};
  iree_hal_buffer_binding_table_t no_binding_table = {0};
  IREE_ASSERT_OK(iree_hal_device_queue_execute(
      device_, /*queue_affinity=*/0, no_semaphores, no_semaphores,
      cmd_buffer.get(), no_binding_table, IREE_HAL_EXECUTE_FLAG_NONE));

  IREE_ASSERT_OK(iree_hal_buffer_map_read(
      destination_buffer.get(), 0, destination.data(), kDestinationBytes));
  const std::vector<float> expected = {
      -7.0f, -7.0f, -7.0f, -7.0f, -7.0f, 1.0f,  2.0f,  -7.0f,
      -7.0f, 3.0f,  4.0f,  -7.0f, -7.0f, -7.0f, -7.0f, -7.0f,
  };
  EXPECT_EQ(destination, expected);
}

TEST_F(DispatchTest, ExecutesF32MaxPoolAndPreservesBorder) {
  tt_iree_ttex_entry_point_desc_t ep = MakeMaxPoolEntry();
  iree::vm::ref<iree_hal_executable_t> executable;
  iree_hal_executable_t *executable_ptr = nullptr;
  IREE_ASSERT_OK(CreateDescriptorTtexExecutable(device_, host_allocator(), ep,
                                                &executable_ptr));
  executable.assign(executable_ptr);

  const iree_hal_tt_kernel_params_t *params = nullptr;
  IREE_ASSERT_OK(iree_hal_tt_executable_lookup_kernel_params(
      executable.get(), /*entry_point=*/0, &params));
  ASSERT_NE(params, nullptr);
  EXPECT_EQ(params->dispatch_kind, TT_IREE_TTEX_DISPATCH_KIND_MAX_POOL);
  EXPECT_EQ(params->max_pool.kernel_h, 3u);
  EXPECT_EQ(params->max_pool.stride_h, 2u);

  constexpr iree_device_size_t kInputBytes = 36 * sizeof(float);
  constexpr iree_device_size_t kOutputBytes = 16 * sizeof(float);
  iree_hal_buffer_params_t buffer_params = {
      .usage = IREE_HAL_BUFFER_USAGE_DISPATCH_STORAGE |
               IREE_HAL_BUFFER_USAGE_TRANSFER,
      .access = IREE_HAL_MEMORY_ACCESS_ALL,
      .type = IREE_HAL_MEMORY_TYPE_DEVICE_LOCAL,
  };
  iree::vm::ref<iree_hal_buffer_t> input_buffer;
  iree::vm::ref<iree_hal_buffer_t> output_buffer;
  IREE_ASSERT_OK(iree_hal_allocator_allocate_buffer(
      device_allocator(), buffer_params, kInputBytes, &input_buffer));
  IREE_ASSERT_OK(iree_hal_allocator_allocate_buffer(
      device_allocator(), buffer_params, kOutputBytes, &output_buffer));

  std::vector<float> input(36);
  for (size_t i = 0; i < input.size(); ++i)
    input[i] = static_cast<float>(i);
  std::vector<float> output(16, -7.0f);
  IREE_ASSERT_OK(iree_hal_buffer_map_write(input_buffer.get(), 0, input.data(),
                                           kInputBytes));
  IREE_ASSERT_OK(iree_hal_buffer_map_write(output_buffer.get(), 0,
                                           output.data(), kOutputBytes));

  iree::vm::ref<iree_hal_command_buffer_t> cmd_buffer;
  IREE_ASSERT_OK(iree_hal_tt_device_create_command_buffer(
      device_, IREE_HAL_COMMAND_BUFFER_MODE_ONE_SHOT,
      IREE_HAL_COMMAND_CATEGORY_DISPATCH, /*queue_affinity=*/0,
      /*binding_capacity=*/2, &cmd_buffer));
  IREE_ASSERT_OK(iree_hal_command_buffer_begin(cmd_buffer.get()));
  iree_hal_buffer_ref_t refs[2] = {
      iree_hal_make_buffer_ref(input_buffer.get(), 0, kInputBytes),
      iree_hal_make_buffer_ref(output_buffer.get(), 0, kOutputBytes),
  };
  iree_hal_dispatch_config_t config = {{0, 0, 0}, {1, 1, 1}};
  IREE_ASSERT_OK(iree_hal_command_buffer_dispatch(
      cmd_buffer.get(), executable.get(), /*export_ordinal=*/0, config,
      iree_const_byte_span_empty(), {2, refs}, IREE_HAL_DISPATCH_FLAG_NONE));
  IREE_ASSERT_OK(iree_hal_command_buffer_end(cmd_buffer.get()));

  iree_hal_semaphore_list_t no_semaphores = {0};
  iree_hal_buffer_binding_table_t no_binding_table = {0};
  IREE_ASSERT_OK(iree_hal_device_queue_execute(
      device_, /*queue_affinity=*/0, no_semaphores, no_semaphores,
      cmd_buffer.get(), no_binding_table, IREE_HAL_EXECUTE_FLAG_NONE));

  IREE_ASSERT_OK(iree_hal_buffer_map_read(output_buffer.get(), 0, output.data(),
                                          kOutputBytes));
  const std::vector<float> expected = {
      -7.0f, -7.0f, -7.0f, -7.0f, -7.0f, 14.0f, 16.0f, -7.0f,
      -7.0f, 26.0f, 28.0f, -7.0f, -7.0f, -7.0f, -7.0f, -7.0f,
  };
  EXPECT_EQ(output, expected);
}

TEST_F(DispatchTest, RejectsMismatchedConv2DOutputShape) {
  tt_iree_ttex_entry_point_desc_t ep = MakeConv2DEntry();
  ep.conv2d.output.shape[1] = 9;
  ep.conv2d.output.storage_shape[1] = 9;

  iree_hal_executable_t *executable = nullptr;
  iree_status_t status = CreateDescriptorTtexExecutable(
      device_, host_allocator(), ep, &executable);
  EXPECT_EQ(iree_status_code(status), IREE_STATUS_INVALID_ARGUMENT);
  iree_status_ignore(status);
  EXPECT_EQ(executable, nullptr);
}

TEST_F(DispatchTest, ExecutesPerChannelF32Conv2DAndPreservesBorder) {
  tt_iree_ttex_entry_point_desc_t ep = MakePerChannelConv2DEntry();
  iree::vm::ref<iree_hal_executable_t> executable;
  iree_hal_executable_t *executable_ptr = nullptr;
#ifdef TT_IREE_ENABLE_MOCK
  constexpr bool kIncludeAot = false;
#else
  constexpr bool kIncludeAot = true;
#endif
  IREE_ASSERT_OK(CreateDescriptorTtexExecutable(device_, host_allocator(), ep,
                                                &executable_ptr, kIncludeAot));
  executable.assign(executable_ptr);

  constexpr iree_device_size_t kActivationBytes = 64 * 10 * 10 * sizeof(float);
  constexpr iree_device_size_t kWeightBytes = 64 * 64 * 3 * 3 * sizeof(float);
  constexpr iree_device_size_t kOutputBytes = 64 * 10 * 10 * sizeof(float);
  iree_hal_buffer_params_t buffer_params = {
      .usage = IREE_HAL_BUFFER_USAGE_DISPATCH_STORAGE |
               IREE_HAL_BUFFER_USAGE_TRANSFER,
      .access = IREE_HAL_MEMORY_ACCESS_ALL,
      .type = IREE_HAL_MEMORY_TYPE_DEVICE_LOCAL,
  };
  iree::vm::ref<iree_hal_buffer_t> activation_buffer;
  iree::vm::ref<iree_hal_buffer_t> weight_buffer;
  iree::vm::ref<iree_hal_buffer_t> output_buffer;
  IREE_ASSERT_OK(iree_hal_allocator_allocate_buffer(
      device_allocator(), buffer_params, kActivationBytes, &activation_buffer));
  IREE_ASSERT_OK(iree_hal_allocator_allocate_buffer(
      device_allocator(), buffer_params, kWeightBytes, &weight_buffer));
  IREE_ASSERT_OK(iree_hal_allocator_allocate_buffer(
      device_allocator(), buffer_params, kOutputBytes, &output_buffer));

  std::vector<float> activations(64 * 10 * 10, 1.0f);
  std::vector<float> weights(64 * 64 * 3 * 3, 1.0f);
  std::vector<float> output(64 * 10 * 10, -7.0f);
  IREE_ASSERT_OK(iree_hal_buffer_map_write(
      activation_buffer.get(), 0, activations.data(), kActivationBytes));
  IREE_ASSERT_OK(iree_hal_buffer_map_write(weight_buffer.get(), 0,
                                           weights.data(), kWeightBytes));
  IREE_ASSERT_OK(iree_hal_buffer_map_write(output_buffer.get(), 0,
                                           output.data(), kOutputBytes));

  iree::vm::ref<iree_hal_command_buffer_t> cmd_buffer;
  IREE_ASSERT_OK(iree_hal_tt_device_create_command_buffer(
      device_, IREE_HAL_COMMAND_BUFFER_MODE_ONE_SHOT,
      IREE_HAL_COMMAND_CATEGORY_DISPATCH, /*queue_affinity=*/0,
      /*binding_capacity=*/3, &cmd_buffer));
  IREE_ASSERT_OK(iree_hal_command_buffer_begin(cmd_buffer.get()));

  iree_hal_buffer_ref_t bindings[3] = {
      iree_hal_make_buffer_ref(activation_buffer.get(), 0, kActivationBytes),
      iree_hal_make_buffer_ref(weight_buffer.get(), 0, kWeightBytes),
      iree_hal_make_buffer_ref(output_buffer.get(), 0, kOutputBytes),
  };
  iree_hal_dispatch_config_t config = {{0, 0, 0}, {2, 2, 1}};
  IREE_ASSERT_OK(iree_hal_command_buffer_dispatch(
      cmd_buffer.get(), executable.get(), /*export_ordinal=*/0, config,
      iree_const_byte_span_empty(), {3, bindings},
      IREE_HAL_DISPATCH_FLAG_NONE));
  IREE_ASSERT_OK(iree_hal_command_buffer_end(cmd_buffer.get()));

  iree_hal_semaphore_list_t no_semaphores = {0};
  iree_hal_buffer_binding_table_t no_binding_table = {0};
  IREE_ASSERT_OK(iree_hal_device_queue_execute(
      device_, /*queue_affinity=*/0, no_semaphores, no_semaphores,
      cmd_buffer.get(), no_binding_table, IREE_HAL_EXECUTE_FLAG_NONE));

  IREE_ASSERT_OK(iree_hal_buffer_map_read(output_buffer.get(), 0, output.data(),
                                          kOutputBytes));
  for (uint32_t c = 0; c < 64; ++c) {
    for (uint32_t h = 0; h < 10; ++h) {
      for (uint32_t w = 0; w < 10; ++w) {
        float expected = (h == 0 || h == 9 || w == 0 || w == 9)
                             ? -7.0f
                             : 144.0f + static_cast<float>(c);
        EXPECT_FLOAT_EQ(output[(c * 10 + h) * 10 + w], expected);
      }
    }
  }
}

TEST_F(DispatchTest, RejectsUnsupportedTtexBuiltin) {
  tt_iree_ttex_entry_point_desc_t ep = {};
  ep.name = "reserved_builtin";
  ep.binding_count = 3;
  ep.workgroup_size[0] = 1;
  ep.workgroup_size[1] = 1;
  ep.workgroup_size[2] = 1;
  ep.builtin_program = TT_IREE_TTEX_BUILTIN_PROGRAM_TTNN_ELTWISE_ADD;

  iree_byte_span_t data = {0};
  IREE_ASSERT_OK(
      tt_iree_build_ttex_executable_def(host_allocator(), 1, &ep, &data));

  iree_hal_executable_params_t exec_params;
  iree_hal_executable_params_initialize(&exec_params);
  exec_params.executable_format =
      iree_make_cstring_view(IREE_HAL_TT_EXECUTABLE_FORMAT);
  exec_params.executable_data =
      iree_make_const_byte_span(data.data, data.data_length);

  iree_hal_executable_t *executable = nullptr;
  iree_status_t status = iree_hal_tt_executable_create(
      device_, &exec_params, host_allocator(), &executable);
  EXPECT_EQ(iree_status_code(status), IREE_STATUS_INVALID_ARGUMENT);
  iree_status_ignore(status);
  EXPECT_EQ(executable, nullptr);
  iree_allocator_free(host_allocator(), data.data);
}

TEST_F(DispatchTest, RejectsUnknownExecutableFormat) {
  iree_hal_executable_params_t exec_params;
  iree_hal_executable_params_initialize(&exec_params);
  exec_params.executable_format = iree_make_cstring_view("unknown-format");

  iree_hal_executable_t *executable = nullptr;
  iree_status_t status = iree_hal_tt_executable_create(
      device_, &exec_params, host_allocator(), &executable);
  EXPECT_EQ(iree_status_code(status), IREE_STATUS_INVALID_ARGUMENT);
  iree_status_ignore(status);
  EXPECT_EQ(executable, nullptr);
}

TEST_F(DispatchTest, RejectsEmptyExecutableFormat) {
  iree_hal_executable_params_t exec_params;
  iree_hal_executable_params_initialize(&exec_params);

  iree_hal_executable_t *executable = nullptr;
  iree_status_t status = iree_hal_tt_executable_create(
      device_, &exec_params, host_allocator(), &executable);
  EXPECT_EQ(iree_status_code(status), IREE_STATUS_INVALID_ARGUMENT);
  iree_status_ignore(status);
  EXPECT_EQ(executable, nullptr);
}

TEST_F(DispatchTest, RejectsDispatchBindingCountMismatch) {
  iree::vm::ref<iree_hal_executable_t> executable;
  iree_hal_executable_t *executable_ptr = nullptr;
  IREE_ASSERT_OK(CreateBuiltinTtexExecutable(
      device_, host_allocator(), TT_IREE_TTEX_BUILTIN_PROGRAM_BF16_MATMUL_TILED,
      /*constant_count=*/0, /*builtin_m_tiles=*/1, /*builtin_n_tiles=*/1,
      /*builtin_k_tiles=*/1, &executable_ptr));
  executable.assign(executable_ptr);

  iree::vm::ref<iree_hal_command_buffer_t> cmd_buffer;
  IREE_ASSERT_OK(iree_hal_tt_device_create_command_buffer(
      device_, IREE_HAL_COMMAND_BUFFER_MODE_ONE_SHOT,
      IREE_HAL_COMMAND_CATEGORY_DISPATCH, /*queue_affinity=*/0,
      /*binding_capacity=*/3, &cmd_buffer));
  IREE_ASSERT_OK(iree_hal_command_buffer_begin(cmd_buffer.get()));

  constexpr iree_device_size_t kTileSizeBytes = 32 * 32 * 2;
  iree_hal_buffer_ref_t bindings[2] = {
      iree_hal_make_indirect_buffer_ref(0, 0, kTileSizeBytes),
      iree_hal_make_indirect_buffer_ref(1, 0, kTileSizeBytes),
  };
  iree_hal_dispatch_config_t config = {{0, 0, 0}, {1, 1, 1}};
  iree_status_t status = iree_hal_command_buffer_dispatch(
      cmd_buffer.get(), executable.get(), /*export_ordinal=*/0, config,
      iree_const_byte_span_empty(), {2, bindings}, IREE_HAL_DISPATCH_FLAG_NONE);
  EXPECT_EQ(iree_status_code(status), IREE_STATUS_INVALID_ARGUMENT);
  iree_status_ignore(status);
}

TEST_F(DispatchTest, RejectsDispatchConstantCountMismatch) {
  iree::vm::ref<iree_hal_executable_t> executable;
  iree_hal_executable_t *executable_ptr = nullptr;
  IREE_ASSERT_OK(CreateBuiltinTtexExecutable(
      device_, host_allocator(), TT_IREE_TTEX_BUILTIN_PROGRAM_BF16_MATMUL_TILED,
      /*constant_count=*/8, /*builtin_m_tiles=*/0, /*builtin_n_tiles=*/0,
      /*builtin_k_tiles=*/0, &executable_ptr));
  executable.assign(executable_ptr);

  iree::vm::ref<iree_hal_command_buffer_t> cmd_buffer;
  IREE_ASSERT_OK(iree_hal_tt_device_create_command_buffer(
      device_, IREE_HAL_COMMAND_BUFFER_MODE_ONE_SHOT,
      IREE_HAL_COMMAND_CATEGORY_DISPATCH, /*queue_affinity=*/0,
      /*binding_capacity=*/3, &cmd_buffer));
  IREE_ASSERT_OK(iree_hal_command_buffer_begin(cmd_buffer.get()));

  constexpr iree_device_size_t kTileSizeBytes = 32 * 32 * 2;
  iree_hal_buffer_ref_t bindings[3] = {
      iree_hal_make_indirect_buffer_ref(0, 0, kTileSizeBytes),
      iree_hal_make_indirect_buffer_ref(1, 0, kTileSizeBytes),
      iree_hal_make_indirect_buffer_ref(2, 0, kTileSizeBytes),
  };
  std::array<uint32_t, 7> too_few_constants = {};
  iree_const_byte_span_t constants = iree_make_const_byte_span(
      reinterpret_cast<const uint8_t *>(too_few_constants.data()),
      sizeof(too_few_constants));
  iree_hal_dispatch_config_t config = {{0, 0, 0}, {1, 1, 1}};
  iree_status_t status = iree_hal_command_buffer_dispatch(
      cmd_buffer.get(), executable.get(), /*export_ordinal=*/0, config,
      constants, {3, bindings}, IREE_HAL_DISPATCH_FLAG_NONE);
  EXPECT_EQ(iree_status_code(status), IREE_STATUS_INVALID_ARGUMENT);
  iree_status_ignore(status);
}

TEST_F(DispatchTest, AcceptsTiledMatmulMulticoreWorkgroupCount) {
  iree::vm::ref<iree_hal_executable_t> executable;
  iree_hal_executable_t *executable_ptr = nullptr;
  IREE_ASSERT_OK(CreateBuiltinTtexExecutable(
      device_, host_allocator(), TT_IREE_TTEX_BUILTIN_PROGRAM_BF16_MATMUL_TILED,
      /*constant_count=*/0, /*builtin_m_tiles=*/2, /*builtin_n_tiles=*/2,
      /*builtin_k_tiles=*/2, &executable_ptr));
  executable.assign(executable_ptr);

  iree::vm::ref<iree_hal_command_buffer_t> cmd_buffer;
  IREE_ASSERT_OK(iree_hal_tt_device_create_command_buffer(
      device_, IREE_HAL_COMMAND_BUFFER_MODE_ONE_SHOT,
      IREE_HAL_COMMAND_CATEGORY_DISPATCH, /*queue_affinity=*/0,
      /*binding_capacity=*/3, &cmd_buffer));
  IREE_ASSERT_OK(iree_hal_command_buffer_begin(cmd_buffer.get()));

  constexpr iree_device_size_t kMatrixSizeBytes = 64 * 64 * 2;
  iree_hal_buffer_ref_t bindings[3] = {
      iree_hal_make_indirect_buffer_ref(0, 0, kMatrixSizeBytes),
      iree_hal_make_indirect_buffer_ref(1, 0, kMatrixSizeBytes),
      iree_hal_make_indirect_buffer_ref(2, 0, kMatrixSizeBytes),
  };
  iree_hal_dispatch_config_t config = {{0, 0, 0}, {2, 2, 1}};
  IREE_ASSERT_OK(iree_hal_command_buffer_dispatch(
      cmd_buffer.get(), executable.get(), /*export_ordinal=*/0, config,
      iree_const_byte_span_empty(), {3, bindings},
      IREE_HAL_DISPATCH_FLAG_NONE));
}

TEST_F(DispatchTest, RejectsUnsupportedDispatchFlags) {
  iree::vm::ref<iree_hal_executable_t> executable;
  iree_hal_executable_t *executable_ptr = nullptr;
  IREE_ASSERT_OK(CreateBuiltinTtexExecutable(
      device_, host_allocator(), TT_IREE_TTEX_BUILTIN_PROGRAM_BF16_MATMUL_TILED,
      /*constant_count=*/0, /*builtin_m_tiles=*/1, /*builtin_n_tiles=*/1,
      /*builtin_k_tiles=*/1, &executable_ptr));
  executable.assign(executable_ptr);

  iree::vm::ref<iree_hal_command_buffer_t> cmd_buffer;
  IREE_ASSERT_OK(iree_hal_tt_device_create_command_buffer(
      device_, IREE_HAL_COMMAND_BUFFER_MODE_ONE_SHOT,
      IREE_HAL_COMMAND_CATEGORY_DISPATCH, /*queue_affinity=*/0,
      /*binding_capacity=*/3, &cmd_buffer));
  IREE_ASSERT_OK(iree_hal_command_buffer_begin(cmd_buffer.get()));

  constexpr iree_device_size_t kTileSizeBytes = 32 * 32 * 2;
  iree_hal_buffer_ref_t bindings[3] = {
      iree_hal_make_indirect_buffer_ref(0, 0, kTileSizeBytes),
      iree_hal_make_indirect_buffer_ref(1, 0, kTileSizeBytes),
      iree_hal_make_indirect_buffer_ref(2, 0, kTileSizeBytes),
  };
  iree_hal_dispatch_config_t config = {{0, 0, 0}, {1, 1, 1}};
  iree_status_t status = iree_hal_command_buffer_dispatch(
      cmd_buffer.get(), executable.get(), /*export_ordinal=*/0, config,
      iree_const_byte_span_empty(), {3, bindings},
      IREE_HAL_DISPATCH_FLAG_CUSTOM_DIRECT_ARGUMENTS);
  EXPECT_EQ(iree_status_code(status), IREE_STATUS_UNIMPLEMENTED);
  iree_status_ignore(status);
}

TEST_F(DispatchTest, ElementwiseAddBfloat16) {
  // 1. Create an executable with compiler-generated AOT kernels.
  iree::vm::ref<iree_hal_executable_t> executable;
  iree_hal_executable_t *executable_ptr = nullptr;
  iree_status_t exec_status = CreateBuiltinTtexExecutable(
      device_, host_allocator(), TT_IREE_TTEX_BUILTIN_PROGRAM_CUSTOM_SFPI_ADD,
      /*constant_count=*/0, /*builtin_m_tiles=*/0, /*builtin_n_tiles=*/0,
      /*builtin_k_tiles=*/0, &executable_ptr, /*include_aot=*/true);
  if (!iree_status_is_ok(exec_status)) {
    iree_status_ignore(exec_status);
    GTEST_SKIP() << "AOT add test fixture is unavailable";
  }
  executable.assign(executable_ptr);

#ifndef TT_IREE_ENABLE_MOCK
  // 2. Prepare bfloat16 test data (1 tile = 32x32 elements).
  constexpr uint32_t kNTiles = 1;
  constexpr uint32_t kElementsPerTile = 32 * 32;
  constexpr uint32_t kTileSizeBytes = kElementsPerTile * sizeof(bfloat16);
  constexpr uint32_t kDramBufferSize = kTileSizeBytes * kNTiles;

  std::vector<bfloat16> host_in0(kElementsPerTile * kNTiles);
  std::vector<bfloat16> host_in1(kElementsPerTile * kNTiles);

  for (uint32_t i = 0; i < kElementsPerTile * kNTiles; i++) {
    host_in0[i] = bfloat16(static_cast<float>(i % 64));
    host_in1[i] = bfloat16(-0.5f);
  }

  // 3. Allocate device buffers via IREE HAL (RAII: auto-released on exit).
  iree_hal_buffer_params_t buffer_params = {
      .usage = IREE_HAL_BUFFER_USAGE_DISPATCH_STORAGE |
               IREE_HAL_BUFFER_USAGE_TRANSFER,
      .access = IREE_HAL_MEMORY_ACCESS_ALL,
      .type = IREE_HAL_MEMORY_TYPE_DEVICE_LOCAL,
  };

  iree::vm::ref<iree_hal_buffer_t> in0_buffer;
  iree::vm::ref<iree_hal_buffer_t> in1_buffer;
  iree::vm::ref<iree_hal_buffer_t> out_buffer;
  IREE_ASSERT_OK(iree_hal_allocator_allocate_buffer(
      device_allocator(), buffer_params, kDramBufferSize, &in0_buffer));
  IREE_ASSERT_OK(iree_hal_allocator_allocate_buffer(
      device_allocator(), buffer_params, kDramBufferSize, &in1_buffer));
  IREE_ASSERT_OK(iree_hal_allocator_allocate_buffer(
      device_allocator(), buffer_params, kDramBufferSize, &out_buffer));

  // 4. Upload test data to device DRAM.
  tt::tt_metal::Buffer *tt_in0 = iree_hal_tt_buffer_handle(in0_buffer.get());
  tt::tt_metal::Buffer *tt_in1 = iree_hal_tt_buffer_handle(in1_buffer.get());
  ASSERT_NE(tt_in0, nullptr);
  ASSERT_NE(tt_in1, nullptr);
  tt::tt_metal::detail::WriteToBuffer(*tt_in0, host_in0);
  tt::tt_metal::detail::WriteToBuffer(*tt_in1, host_in1);

  // 5. Record command buffer.
  iree::vm::ref<iree_hal_command_buffer_t> cmd_buffer;
  IREE_ASSERT_OK(iree_hal_tt_device_create_command_buffer(
      device_, IREE_HAL_COMMAND_BUFFER_MODE_ONE_SHOT,
      IREE_HAL_COMMAND_CATEGORY_DISPATCH,
      /*queue_affinity=*/0, /*binding_capacity=*/16, &cmd_buffer));

  IREE_ASSERT_OK(iree_hal_command_buffer_begin(cmd_buffer.get()));

  iree_hal_buffer_ref_t bindings[3] = {
      iree_hal_make_buffer_ref(in0_buffer.get(), 0, kDramBufferSize),
      iree_hal_make_buffer_ref(in1_buffer.get(), 0, kDramBufferSize),
      iree_hal_make_buffer_ref(out_buffer.get(), 0, kDramBufferSize),
  };
  iree_hal_dispatch_config_t config = {{0, 0, 0}, {1, 1, 1}};
  iree_const_byte_span_t constants = {0};
  iree_hal_buffer_ref_list_t binding_list = {3, bindings};

  IREE_ASSERT_OK(iree_hal_command_buffer_dispatch(
      cmd_buffer.get(), executable.get(), /*export_ordinal=*/0, config,
      constants, binding_list, /*flags=*/0));
  IREE_ASSERT_OK(iree_hal_command_buffer_end(cmd_buffer.get()));

  // 6. Submit and execute on device.
  iree_hal_semaphore_list_t wait_semaphores = {0};
  iree_hal_semaphore_list_t signal_semaphores = {0};
  iree_hal_buffer_binding_table_t binding_table = {0};

  IREE_ASSERT_OK(iree_hal_device_queue_execute(
      device_, /*queue_affinity=*/0, wait_semaphores, signal_semaphores,
      cmd_buffer.get(), binding_table, /*flags=*/0));

  // 7. Read back results.
  std::vector<bfloat16> host_out(kElementsPerTile * kNTiles);
  tt::tt_metal::Buffer *tt_out = iree_hal_tt_buffer_handle(out_buffer.get());
  ASSERT_NE(tt_out, nullptr);
  tt::tt_metal::detail::ReadFromBuffer(*tt_out, host_out);

  // 8. Verify element-wise addition: out[i] == in0[i] + in1[i].
  constexpr float kEps = 1e-1f; // bfloat16 tolerance
  for (uint32_t i = 0; i < kElementsPerTile * kNTiles; i++) {
    float expected =
        static_cast<float>(host_in0[i]) + static_cast<float>(host_in1[i]);
    float actual = static_cast<float>(host_out[i]);
    ASSERT_NEAR(actual, expected, kEps)
        << "mismatch at index " << i << ": expected " << expected << ", got "
        << actual;
  }

  // Log a few sample values for diagnostics.
  fprintf(stderr, "  Sample: %.1f + %.1f = %.1f, %.1f + %.1f = %.1f\n",
          static_cast<float>(host_in0[0]), static_cast<float>(host_in1[0]),
          static_cast<float>(host_out[0]), static_cast<float>(host_in0[10]),
          static_cast<float>(host_in1[10]), static_cast<float>(host_out[10]));
#endif // !TT_IREE_ENABLE_MOCK
  // All resources (executable, buffers, cmd_buffer) auto-released by vm::ref.
}

TEST_F(DispatchTest, ReDispatchSameExecutable) {
  // Verifies that a program survives dispatch and can be re-used.
  iree::vm::ref<iree_hal_executable_t> executable;
  iree_hal_executable_t *executable_ptr = nullptr;
  iree_status_t exec_status = CreateBuiltinTtexExecutable(
      device_, host_allocator(), TT_IREE_TTEX_BUILTIN_PROGRAM_CUSTOM_SFPI_ADD,
      /*constant_count=*/0, /*builtin_m_tiles=*/0, /*builtin_n_tiles=*/0,
      /*builtin_k_tiles=*/0, &executable_ptr, /*include_aot=*/true);
  if (!iree_status_is_ok(exec_status)) {
    iree_status_ignore(exec_status);
    GTEST_SKIP() << "AOT add test fixture is unavailable";
  }
  executable.assign(executable_ptr);

#ifndef TT_IREE_ENABLE_MOCK
  constexpr uint32_t kNTiles = 1;
  constexpr uint32_t kElementsPerTile = 32 * 32;
  constexpr uint32_t kTileSizeBytes = kElementsPerTile * sizeof(bfloat16);
  constexpr uint32_t kDramBufferSize = kTileSizeBytes * kNTiles;

  iree_hal_buffer_params_t buffer_params = {
      .usage = IREE_HAL_BUFFER_USAGE_DISPATCH_STORAGE |
               IREE_HAL_BUFFER_USAGE_TRANSFER,
      .access = IREE_HAL_MEMORY_ACCESS_ALL,
      .type = IREE_HAL_MEMORY_TYPE_DEVICE_LOCAL,
  };

  iree::vm::ref<iree_hal_buffer_t> in0_buffer;
  iree::vm::ref<iree_hal_buffer_t> in1_buffer;
  iree::vm::ref<iree_hal_buffer_t> out_buffer;
  IREE_ASSERT_OK(iree_hal_allocator_allocate_buffer(
      device_allocator(), buffer_params, kDramBufferSize, &in0_buffer));
  IREE_ASSERT_OK(iree_hal_allocator_allocate_buffer(
      device_allocator(), buffer_params, kDramBufferSize, &in1_buffer));
  IREE_ASSERT_OK(iree_hal_allocator_allocate_buffer(
      device_allocator(), buffer_params, kDramBufferSize, &out_buffer));

  constexpr float kEps = 1e-1f;

  // --- First dispatch: 1.0 + 2.0 = 3.0 ---
  {
    std::vector<bfloat16> h_in0(kElementsPerTile, bfloat16(1.0f));
    std::vector<bfloat16> h_in1(kElementsPerTile, bfloat16(2.0f));

    tt::tt_metal::detail::WriteToBuffer(
        *iree_hal_tt_buffer_handle(in0_buffer.get()), h_in0);
    tt::tt_metal::detail::WriteToBuffer(
        *iree_hal_tt_buffer_handle(in1_buffer.get()), h_in1);

    iree::vm::ref<iree_hal_command_buffer_t> cmd;
    IREE_ASSERT_OK(iree_hal_tt_device_create_command_buffer(
        device_, IREE_HAL_COMMAND_BUFFER_MODE_ONE_SHOT,
        IREE_HAL_COMMAND_CATEGORY_DISPATCH, 0, 16, &cmd));
    IREE_ASSERT_OK(iree_hal_command_buffer_begin(cmd.get()));

    iree_hal_buffer_ref_t bindings[3] = {
        iree_hal_make_buffer_ref(in0_buffer.get(), 0, kDramBufferSize),
        iree_hal_make_buffer_ref(in1_buffer.get(), 0, kDramBufferSize),
        iree_hal_make_buffer_ref(out_buffer.get(), 0, kDramBufferSize),
    };
    iree_hal_dispatch_config_t config = {{0, 0, 0}, {1, 1, 1}};
    iree_const_byte_span_t constants = {0};
    iree_hal_buffer_ref_list_t binding_list = {3, bindings};

    IREE_ASSERT_OK(iree_hal_command_buffer_dispatch(
        cmd.get(), executable.get(), 0, config, constants, binding_list, 0));
    IREE_ASSERT_OK(iree_hal_command_buffer_end(cmd.get()));

    iree_hal_semaphore_list_t no_sema = {0};
    iree_hal_buffer_binding_table_t no_bt = {0};
    IREE_ASSERT_OK(iree_hal_device_queue_execute(device_, 0, no_sema, no_sema,
                                                 cmd.get(), no_bt, 0));

    std::vector<bfloat16> h_out(kElementsPerTile);
    tt::tt_metal::detail::ReadFromBuffer(
        *iree_hal_tt_buffer_handle(out_buffer.get()), h_out);

    for (uint32_t i = 0; i < kElementsPerTile; i++) {
      ASSERT_NEAR(static_cast<float>(h_out[i]), 3.0f, kEps)
          << "dispatch 1 mismatch at " << i;
    }
    fprintf(stderr, "  Dispatch 1: 1.0 + 2.0 = %.1f (OK)\n",
            static_cast<float>(h_out[0]));
  }

  // --- Second dispatch (same executable): 10.0 + 5.0 = 15.0 ---
  {
    std::vector<bfloat16> h_in0(kElementsPerTile, bfloat16(10.0f));
    std::vector<bfloat16> h_in1(kElementsPerTile, bfloat16(5.0f));

    tt::tt_metal::detail::WriteToBuffer(
        *iree_hal_tt_buffer_handle(in0_buffer.get()), h_in0);
    tt::tt_metal::detail::WriteToBuffer(
        *iree_hal_tt_buffer_handle(in1_buffer.get()), h_in1);

    iree::vm::ref<iree_hal_command_buffer_t> cmd;
    IREE_ASSERT_OK(iree_hal_tt_device_create_command_buffer(
        device_, IREE_HAL_COMMAND_BUFFER_MODE_ONE_SHOT,
        IREE_HAL_COMMAND_CATEGORY_DISPATCH, 0, 16, &cmd));
    IREE_ASSERT_OK(iree_hal_command_buffer_begin(cmd.get()));

    iree_hal_buffer_ref_t bindings[3] = {
        iree_hal_make_buffer_ref(in0_buffer.get(), 0, kDramBufferSize),
        iree_hal_make_buffer_ref(in1_buffer.get(), 0, kDramBufferSize),
        iree_hal_make_buffer_ref(out_buffer.get(), 0, kDramBufferSize),
    };
    iree_hal_dispatch_config_t config = {{0, 0, 0}, {1, 1, 1}};
    iree_const_byte_span_t constants = {0};
    iree_hal_buffer_ref_list_t binding_list = {3, bindings};

    IREE_ASSERT_OK(iree_hal_command_buffer_dispatch(
        cmd.get(), executable.get(), 0, config, constants, binding_list, 0));
    IREE_ASSERT_OK(iree_hal_command_buffer_end(cmd.get()));

    iree_hal_semaphore_list_t no_sema = {0};
    iree_hal_buffer_binding_table_t no_bt = {0};
    IREE_ASSERT_OK(iree_hal_device_queue_execute(device_, 0, no_sema, no_sema,
                                                 cmd.get(), no_bt, 0));

    std::vector<bfloat16> h_out(kElementsPerTile);
    tt::tt_metal::detail::ReadFromBuffer(
        *iree_hal_tt_buffer_handle(out_buffer.get()), h_out);

    for (uint32_t i = 0; i < kElementsPerTile; i++) {
      ASSERT_NEAR(static_cast<float>(h_out[i]), 15.0f, kEps)
          << "dispatch 2 mismatch at " << i;
    }
    fprintf(stderr, "  Dispatch 2: 10.0 + 5.0 = %.1f (OK)\n",
            static_cast<float>(h_out[0]));
  }
#endif // !TT_IREE_ENABLE_MOCK
}

TEST_F(DispatchTest, BuiltinMatmulBfloat16ViaTTEX) {
  iree::vm::ref<iree_hal_executable_t> executable;
  iree_hal_executable_t *executable_ptr = nullptr;
  iree_status_t exec_status = CreateBuiltinTtexExecutable(
      device_, host_allocator(),
      TT_IREE_TTEX_BUILTIN_PROGRAM_BF16_MATMUL_32X32X32,
      /*constant_count=*/0, /*builtin_m_tiles=*/1, /*builtin_n_tiles=*/1,
      /*builtin_k_tiles=*/1, &executable_ptr, /*include_aot=*/true);
  if (!iree_status_is_ok(exec_status)) {
    iree_status_ignore(exec_status);
    GTEST_SKIP() << "matmul builtin kernel compilation not available";
  }
  executable.assign(executable_ptr);

#ifndef TT_IREE_ENABLE_MOCK
  constexpr uint32_t kElementsPerTile = 32 * 32;
  constexpr uint32_t kTileSizeBytes = kElementsPerTile * sizeof(bfloat16);

  iree_hal_buffer_params_t buffer_params = {
      .usage = IREE_HAL_BUFFER_USAGE_DISPATCH_STORAGE |
               IREE_HAL_BUFFER_USAGE_TRANSFER,
      .access = IREE_HAL_MEMORY_ACCESS_ALL,
      .type = IREE_HAL_MEMORY_TYPE_DEVICE_LOCAL,
  };

  iree::vm::ref<iree_hal_buffer_t> lhs_buffer;
  iree::vm::ref<iree_hal_buffer_t> rhs_buffer;
  iree::vm::ref<iree_hal_buffer_t> out_buffer;
  IREE_ASSERT_OK(iree_hal_allocator_allocate_buffer(
      device_allocator(), buffer_params, kTileSizeBytes, &lhs_buffer));
  IREE_ASSERT_OK(iree_hal_allocator_allocate_buffer(
      device_allocator(), buffer_params, kTileSizeBytes, &rhs_buffer));
  IREE_ASSERT_OK(iree_hal_allocator_allocate_buffer(
      device_allocator(), buffer_params, kTileSizeBytes, &out_buffer));

  std::vector<bfloat16> lhs(kElementsPerTile, bfloat16(1.0f));
  std::vector<bfloat16> rhs(kElementsPerTile, bfloat16(2.0f));
  tt::tt_metal::detail::WriteToBuffer(
      *iree_hal_tt_buffer_handle(lhs_buffer.get()), lhs);
  tt::tt_metal::detail::WriteToBuffer(
      *iree_hal_tt_buffer_handle(rhs_buffer.get()), rhs);

  iree::vm::ref<iree_hal_command_buffer_t> cmd_buffer;
  IREE_ASSERT_OK(iree_hal_tt_device_create_command_buffer(
      device_, IREE_HAL_COMMAND_BUFFER_MODE_ONE_SHOT,
      IREE_HAL_COMMAND_CATEGORY_DISPATCH,
      /*queue_affinity=*/0, /*binding_capacity=*/16, &cmd_buffer));

  IREE_ASSERT_OK(iree_hal_command_buffer_begin(cmd_buffer.get()));
  iree_hal_buffer_ref_t bindings[3] = {
      iree_hal_make_buffer_ref(lhs_buffer.get(), 0, kTileSizeBytes),
      iree_hal_make_buffer_ref(rhs_buffer.get(), 0, kTileSizeBytes),
      iree_hal_make_buffer_ref(out_buffer.get(), 0, kTileSizeBytes),
  };
  iree_hal_dispatch_config_t config = {{0, 0, 0}, {1, 1, 1}};
  iree_const_byte_span_t constants = {0};
  iree_hal_buffer_ref_list_t binding_list = {3, bindings};
  IREE_ASSERT_OK(iree_hal_command_buffer_dispatch(
      cmd_buffer.get(), executable.get(), /*export_ordinal=*/0, config,
      constants, binding_list, /*flags=*/0));
  IREE_ASSERT_OK(iree_hal_command_buffer_end(cmd_buffer.get()));

  iree_hal_semaphore_list_t no_sema = {0};
  iree_hal_buffer_binding_table_t no_bt = {0};
  IREE_ASSERT_OK(iree_hal_device_queue_execute(device_, 0, no_sema, no_sema,
                                               cmd_buffer.get(), no_bt, 0));

  std::vector<bfloat16> host_out(kElementsPerTile);
  tt::tt_metal::detail::ReadFromBuffer(
      *iree_hal_tt_buffer_handle(out_buffer.get()), host_out);

  constexpr float kExpected = 64.0f;
  constexpr float kEps = 0.25f;
  for (uint32_t i = 0; i < kElementsPerTile; ++i) {
    ASSERT_NEAR(static_cast<float>(host_out[i]), kExpected, kEps)
        << "matmul mismatch at " << i;
  }
#endif // !TT_IREE_ENABLE_MOCK
}

TEST_F(DispatchTest, BuiltinTiledMatmulBfloat16ViaTTEX) {
  iree::vm::ref<iree_hal_executable_t> executable;
  iree_hal_executable_t *executable_ptr = nullptr;
  iree_status_t exec_status = CreateBuiltinTtexExecutable(
      device_, host_allocator(), TT_IREE_TTEX_BUILTIN_PROGRAM_BF16_MATMUL_TILED,
      /*constant_count=*/0, /*builtin_m_tiles=*/2, /*builtin_n_tiles=*/2,
      /*builtin_k_tiles=*/2, &executable_ptr, /*include_aot=*/true);
  if (!iree_status_is_ok(exec_status)) {
    iree_status_ignore(exec_status);
    GTEST_SKIP() << "tiled matmul builtin kernel compilation not available";
  }
  executable.assign(executable_ptr);

#ifndef TT_IREE_ENABLE_MOCK
  constexpr uint32_t kRows = 64;
  constexpr uint32_t kCols = 64;
  constexpr uint32_t kElementCount = kRows * kCols;
  constexpr uint32_t kBufferSizeBytes = kElementCount * sizeof(bfloat16);

  iree_hal_buffer_params_t buffer_params = {
      .usage = IREE_HAL_BUFFER_USAGE_DISPATCH_STORAGE |
               IREE_HAL_BUFFER_USAGE_TRANSFER,
      .access = IREE_HAL_MEMORY_ACCESS_ALL,
      .type = IREE_HAL_MEMORY_TYPE_DEVICE_LOCAL,
  };

  iree::vm::ref<iree_hal_buffer_t> lhs_buffer;
  iree::vm::ref<iree_hal_buffer_t> rhs_buffer;
  iree::vm::ref<iree_hal_buffer_t> out_buffer;
  IREE_ASSERT_OK(iree_hal_allocator_allocate_buffer(
      device_allocator(), buffer_params, kBufferSizeBytes, &lhs_buffer));
  IREE_ASSERT_OK(iree_hal_allocator_allocate_buffer(
      device_allocator(), buffer_params, kBufferSizeBytes, &rhs_buffer));
  IREE_ASSERT_OK(iree_hal_allocator_allocate_buffer(
      device_allocator(), buffer_params, kBufferSizeBytes, &out_buffer));

  std::vector<bfloat16> lhs(kElementCount, bfloat16(1.0f));
  std::vector<bfloat16> rhs(kElementCount, bfloat16(2.0f));
  tt::tt_metal::detail::WriteToBuffer(
      *iree_hal_tt_buffer_handle(lhs_buffer.get()), lhs);
  tt::tt_metal::detail::WriteToBuffer(
      *iree_hal_tt_buffer_handle(rhs_buffer.get()), rhs);

  iree::vm::ref<iree_hal_command_buffer_t> cmd_buffer;
  IREE_ASSERT_OK(iree_hal_tt_device_create_command_buffer(
      device_, IREE_HAL_COMMAND_BUFFER_MODE_ONE_SHOT,
      IREE_HAL_COMMAND_CATEGORY_DISPATCH, 0, 16, &cmd_buffer));

  IREE_ASSERT_OK(iree_hal_command_buffer_begin(cmd_buffer.get()));
  iree_hal_buffer_ref_t bindings[3] = {
      iree_hal_make_buffer_ref(lhs_buffer.get(), 0, kBufferSizeBytes),
      iree_hal_make_buffer_ref(rhs_buffer.get(), 0, kBufferSizeBytes),
      iree_hal_make_buffer_ref(out_buffer.get(), 0, kBufferSizeBytes),
  };
  iree_hal_dispatch_config_t config = {{0, 0, 0}, {2, 2, 1}};
  iree_const_byte_span_t constants = {0};
  iree_hal_buffer_ref_list_t binding_list = {3, bindings};
  IREE_ASSERT_OK(iree_hal_command_buffer_dispatch(
      cmd_buffer.get(), executable.get(), /*export_ordinal=*/0, config,
      constants, binding_list, /*flags=*/0));
  IREE_ASSERT_OK(iree_hal_command_buffer_end(cmd_buffer.get()));

  iree_hal_semaphore_list_t no_sema = {0};
  iree_hal_buffer_binding_table_t no_bt = {0};
  IREE_ASSERT_OK(iree_hal_device_queue_execute(device_, 0, no_sema, no_sema,
                                               cmd_buffer.get(), no_bt, 0));

  std::vector<bfloat16> host_out(kElementCount);
  tt::tt_metal::detail::ReadFromBuffer(
      *iree_hal_tt_buffer_handle(out_buffer.get()), host_out);

  constexpr float kExpected = 128.0f;
  constexpr float kEps = 0.5f;
  for (uint32_t i = 0; i < kElementCount; ++i) {
    ASSERT_NEAR(static_cast<float>(host_out[i]), kExpected, kEps)
        << "tiled matmul mismatch at " << i;
  }
#endif // !TT_IREE_ENABLE_MOCK
}

TEST_F(DispatchTest, BuiltinTiledMatmulHonorsIndirectSubranges) {
  iree::vm::ref<iree_hal_executable_t> executable;
  iree_hal_executable_t *executable_ptr = nullptr;
  iree_status_t exec_status = CreateBuiltinTtexExecutable(
      device_, host_allocator(), TT_IREE_TTEX_BUILTIN_PROGRAM_BF16_MATMUL_TILED,
      /*constant_count=*/0, /*builtin_m_tiles=*/1, /*builtin_n_tiles=*/1,
      /*builtin_k_tiles=*/1, &executable_ptr, /*include_aot=*/true);
  if (!iree_status_is_ok(exec_status)) {
    iree_status_ignore(exec_status);
    GTEST_SKIP() << "tiled matmul executable creation not available";
  }
  executable.assign(executable_ptr);

#ifndef TT_IREE_ENABLE_MOCK
  constexpr uint32_t kRows = 32;
  constexpr uint32_t kElementCount = kRows * kRows;
  constexpr uint32_t kMatrixSizeBytes = kElementCount * sizeof(bfloat16);
  constexpr uint32_t kStorageElementCount = kElementCount * 2;
  constexpr uint32_t kStorageSizeBytes = kMatrixSizeBytes * 2;

  iree_hal_buffer_params_t buffer_params = {
      .usage = IREE_HAL_BUFFER_USAGE_DISPATCH_STORAGE |
               IREE_HAL_BUFFER_USAGE_TRANSFER,
      .access = IREE_HAL_MEMORY_ACCESS_ALL,
      .type = IREE_HAL_MEMORY_TYPE_DEVICE_LOCAL,
  };

  iree::vm::ref<iree_hal_buffer_t> lhs_buffer;
  iree::vm::ref<iree_hal_buffer_t> rhs_buffer;
  iree::vm::ref<iree_hal_buffer_t> out_buffer;
  IREE_ASSERT_OK(iree_hal_allocator_allocate_buffer(
      device_allocator(), buffer_params, kStorageSizeBytes, &lhs_buffer));
  IREE_ASSERT_OK(iree_hal_allocator_allocate_buffer(
      device_allocator(), buffer_params, kStorageSizeBytes, &rhs_buffer));
  IREE_ASSERT_OK(iree_hal_allocator_allocate_buffer(
      device_allocator(), buffer_params, kStorageSizeBytes, &out_buffer));

  std::vector<bfloat16> lhs(kStorageElementCount, bfloat16(7.0f));
  std::vector<bfloat16> rhs(kStorageElementCount, bfloat16(9.0f));
  std::vector<bfloat16> out(kStorageElementCount, bfloat16(-3.0f));
  std::fill(lhs.begin() + kElementCount, lhs.end(), bfloat16(1.0f));
  std::fill(rhs.begin() + kElementCount, rhs.end(), bfloat16(2.0f));
  tt::tt_metal::detail::WriteToBuffer(
      *iree_hal_tt_buffer_handle(lhs_buffer.get()), lhs);
  tt::tt_metal::detail::WriteToBuffer(
      *iree_hal_tt_buffer_handle(rhs_buffer.get()), rhs);
  tt::tt_metal::detail::WriteToBuffer(
      *iree_hal_tt_buffer_handle(out_buffer.get()), out);

  iree::vm::ref<iree_hal_command_buffer_t> cmd_buffer;
  IREE_ASSERT_OK(iree_hal_tt_device_create_command_buffer(
      device_, IREE_HAL_COMMAND_BUFFER_MODE_ONE_SHOT,
      IREE_HAL_COMMAND_CATEGORY_DISPATCH, /*queue_affinity=*/0,
      /*binding_capacity=*/3, &cmd_buffer));
  IREE_ASSERT_OK(iree_hal_command_buffer_begin(cmd_buffer.get()));

  iree_hal_buffer_ref_t refs[3] = {
      iree_hal_make_indirect_buffer_ref(
          /*buffer_slot=*/0, kMatrixSizeBytes, kMatrixSizeBytes),
      iree_hal_make_indirect_buffer_ref(
          /*buffer_slot=*/1, kMatrixSizeBytes, kMatrixSizeBytes),
      iree_hal_make_indirect_buffer_ref(
          /*buffer_slot=*/2, kMatrixSizeBytes, kMatrixSizeBytes),
  };
  iree_hal_dispatch_config_t config = {{0, 0, 0}, {1, 1, 1}};
  IREE_ASSERT_OK(iree_hal_command_buffer_dispatch(
      cmd_buffer.get(), executable.get(), /*export_ordinal=*/0, config,
      iree_const_byte_span_empty(), {3, refs}, IREE_HAL_DISPATCH_FLAG_NONE));
  IREE_ASSERT_OK(iree_hal_command_buffer_end(cmd_buffer.get()));

  iree_hal_buffer_binding_t bindings[3] = {
      {lhs_buffer.get(), /*offset=*/0, kStorageSizeBytes},
      {rhs_buffer.get(), /*offset=*/0, kStorageSizeBytes},
      {out_buffer.get(), /*offset=*/0, kStorageSizeBytes},
  };
  iree_hal_buffer_binding_table_t binding_table = {3, bindings};
  iree_hal_semaphore_list_t no_sema = {0};
  IREE_ASSERT_OK(iree_hal_device_queue_execute(
      device_, /*queue_affinity=*/0, no_sema, no_sema, cmd_buffer.get(),
      binding_table, IREE_HAL_EXECUTE_FLAG_NONE));

  std::vector<bfloat16> host_out(kStorageElementCount);
  tt::tt_metal::detail::ReadFromBuffer(
      *iree_hal_tt_buffer_handle(out_buffer.get()), host_out);

  constexpr float kEps = 0.25f;
  for (uint32_t i = 0; i < kElementCount; ++i) {
    ASSERT_NEAR(static_cast<float>(host_out[i]), -3.0f, kEps)
        << "prefix sentinel was modified at " << i;
  }
  for (uint32_t i = kElementCount; i < kStorageElementCount; ++i) {
    ASSERT_NEAR(static_cast<float>(host_out[i]), 64.0f, kEps)
        << "subrange matmul mismatch at " << i;
  }
#endif // !TT_IREE_ENABLE_MOCK
}

TEST_F(DispatchTest, BuiltinDynamicTiledMatmulBfloat16ViaTTEX) {
  iree::vm::ref<iree_hal_executable_t> executable;
  iree_hal_executable_t *executable_ptr = nullptr;
  iree_status_t exec_status = CreateBuiltinTtexExecutable(
      device_, host_allocator(), TT_IREE_TTEX_BUILTIN_PROGRAM_BF16_MATMUL_TILED,
      /*constant_count=*/8, /*builtin_m_tiles=*/0, /*builtin_n_tiles=*/0,
      /*builtin_k_tiles=*/0, &executable_ptr, /*include_aot=*/true);
  if (!iree_status_is_ok(exec_status)) {
    iree_status_ignore(exec_status);
    GTEST_SKIP() << "dynamic tiled matmul builtin kernel compilation not "
                    "available";
  }
  executable.assign(executable_ptr);

#ifndef TT_IREE_ENABLE_MOCK
  constexpr uint32_t kRows = 64;
  constexpr uint32_t kCols = 64;
  constexpr uint32_t kElementCount = kRows * kCols;
  constexpr uint32_t kBufferSizeBytes = kElementCount * sizeof(bfloat16);

  iree_hal_buffer_params_t buffer_params = {
      .usage = IREE_HAL_BUFFER_USAGE_DISPATCH_STORAGE |
               IREE_HAL_BUFFER_USAGE_TRANSFER,
      .access = IREE_HAL_MEMORY_ACCESS_ALL,
      .type = IREE_HAL_MEMORY_TYPE_DEVICE_LOCAL,
  };

  iree::vm::ref<iree_hal_buffer_t> lhs_buffer;
  iree::vm::ref<iree_hal_buffer_t> rhs_buffer;
  iree::vm::ref<iree_hal_buffer_t> out_buffer;
  IREE_ASSERT_OK(iree_hal_allocator_allocate_buffer(
      device_allocator(), buffer_params, kBufferSizeBytes, &lhs_buffer));
  IREE_ASSERT_OK(iree_hal_allocator_allocate_buffer(
      device_allocator(), buffer_params, kBufferSizeBytes, &rhs_buffer));
  IREE_ASSERT_OK(iree_hal_allocator_allocate_buffer(
      device_allocator(), buffer_params, kBufferSizeBytes, &out_buffer));

  std::vector<bfloat16> lhs(kElementCount, bfloat16(1.0f));
  std::vector<bfloat16> rhs(kElementCount, bfloat16(2.0f));
  tt::tt_metal::detail::WriteToBuffer(
      *iree_hal_tt_buffer_handle(lhs_buffer.get()), lhs);
  tt::tt_metal::detail::WriteToBuffer(
      *iree_hal_tt_buffer_handle(rhs_buffer.get()), rhs);

  std::array<uint32_t, 8> shape_constants = {
      64u, 0u, // lhs_k
      64u, 0u, // rhs_k
      64u, 0u, // lhs_m
      64u, 0u, // rhs_n
  };

  iree::vm::ref<iree_hal_command_buffer_t> cmd_buffer;
  IREE_ASSERT_OK(iree_hal_tt_device_create_command_buffer(
      device_, IREE_HAL_COMMAND_BUFFER_MODE_ONE_SHOT,
      IREE_HAL_COMMAND_CATEGORY_DISPATCH, 0, 16, &cmd_buffer));

  IREE_ASSERT_OK(iree_hal_command_buffer_begin(cmd_buffer.get()));
  iree_hal_buffer_ref_t bindings[3] = {
      iree_hal_make_buffer_ref(lhs_buffer.get(), 0, kBufferSizeBytes),
      iree_hal_make_buffer_ref(rhs_buffer.get(), 0, kBufferSizeBytes),
      iree_hal_make_buffer_ref(out_buffer.get(), 0, kBufferSizeBytes),
  };
  iree_hal_dispatch_config_t config = {{0, 0, 0}, {2, 2, 1}};
  iree_const_byte_span_t constants = iree_make_const_byte_span(
      reinterpret_cast<const uint8_t *>(shape_constants.data()),
      sizeof(shape_constants));
  iree_hal_buffer_ref_list_t binding_list = {3, bindings};
  IREE_ASSERT_OK(iree_hal_command_buffer_dispatch(
      cmd_buffer.get(), executable.get(), /*export_ordinal=*/0, config,
      constants, binding_list, /*flags=*/0));
  IREE_ASSERT_OK(iree_hal_command_buffer_end(cmd_buffer.get()));

  iree_hal_semaphore_list_t no_sema = {0};
  iree_hal_buffer_binding_table_t no_bt = {0};
  IREE_ASSERT_OK(iree_hal_device_queue_execute(device_, 0, no_sema, no_sema,
                                               cmd_buffer.get(), no_bt, 0));

  std::vector<bfloat16> host_out(kElementCount);
  tt::tt_metal::detail::ReadFromBuffer(
      *iree_hal_tt_buffer_handle(out_buffer.get()), host_out);

  constexpr float kExpected = 128.0f;
  constexpr float kEps = 0.5f;
  for (uint32_t i = 0; i < kElementCount; ++i) {
    ASSERT_NEAR(static_cast<float>(host_out[i]), kExpected, kEps)
        << "dynamic tiled matmul mismatch at " << i;
  }
#endif // !TT_IREE_ENABLE_MOCK
}

} // namespace
} // namespace tenstorrent
} // namespace hal
} // namespace iree
