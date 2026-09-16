// Copyright 2025 The tt-iree Authors
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// Utility to build TTEX FlatBuffer blobs without a compiler.
// Primary use case: unit tests that need to exercise the runtime's
// TTEX parsing/verification path before the compiler backend exists.

#ifndef IREE_SCHEMA_TT_EXECUTABLE_BUILDER_UTIL_H_
#define IREE_SCHEMA_TT_EXECUTABLE_BUILDER_UTIL_H_

#include <stdint.h>

#include "iree/base/api.h"

#ifdef __cplusplus
extern "C" {
#endif

// TTEX wire-format versions. Version 0 represents legacy/unversioned blobs;
// new blobs must explicitly encode the current version.
#define TT_IREE_TTEX_VERSION_LEGACY 0u
#define TT_IREE_TTEX_VERSION_1 1u
#define TT_IREE_TTEX_VERSION_2 2u
#define TT_IREE_TTEX_VERSION_3 3u
#define TT_IREE_TTEX_VERSION_4 4u
#define TT_IREE_TTEX_VERSION_5 5u
#define TT_IREE_TTEX_VERSION_6 6u
#define TT_IREE_TTEX_VERSION_7 7u
#define TT_IREE_TTEX_VERSION_8 8u
#define TT_IREE_TTEX_CURRENT_VERSION TT_IREE_TTEX_VERSION_8

// BuiltinProgram enum values shared by the compiler, runtime, and tests.
// Must stay in sync with BuiltinProgram in tt_executable_def.fbs.
typedef enum tt_iree_ttex_builtin_program_e {
  TT_IREE_TTEX_BUILTIN_PROGRAM_CUSTOM_SFPI_ADD = 0,
  TT_IREE_TTEX_BUILTIN_PROGRAM_TTNN_ELTWISE_ADD = 1,
  TT_IREE_TTEX_BUILTIN_PROGRAM_BF16_MATMUL_32X32X32 = 2,
  TT_IREE_TTEX_BUILTIN_PROGRAM_BF16_MATMUL_TILED = 3,
  TT_IREE_TTEX_BUILTIN_PROGRAM_F32_MATMUL_LIKE = 4,
  TT_IREE_TTEX_BUILTIN_PROGRAM_F32_CONV2D = 5,
  TT_IREE_TTEX_BUILTIN_PROGRAM_F32_PADDED_COPY = 6,
  TT_IREE_TTEX_BUILTIN_PROGRAM_F32_MAX_POOL_2D = 7,
  TT_IREE_TTEX_BUILTIN_PROGRAM_F32_GLOBAL_SUM = 8,
  TT_IREE_TTEX_BUILTIN_PROGRAM_F32_SCALAR_DIV = 9,
  TT_IREE_TTEX_BUILTIN_PROGRAM_F32_LINEAR = 10,
} tt_iree_ttex_builtin_program_e;

typedef enum tt_iree_ttex_dispatch_kind_e {
  TT_IREE_TTEX_DISPATCH_KIND_BUILTIN = 0,
  TT_IREE_TTEX_DISPATCH_KIND_MATMUL_LIKE = 1,
  TT_IREE_TTEX_DISPATCH_KIND_CONVOLUTION = 2,
  TT_IREE_TTEX_DISPATCH_KIND_PADDED_COPY = 3,
  TT_IREE_TTEX_DISPATCH_KIND_MAX_POOL = 4,
  TT_IREE_TTEX_DISPATCH_KIND_GLOBAL_SUM = 5,
  TT_IREE_TTEX_DISPATCH_KIND_SCALAR_DIV = 6,
  TT_IREE_TTEX_DISPATCH_KIND_LINEAR = 7,
} tt_iree_ttex_dispatch_kind_e;

typedef enum tt_iree_ttex_tensor_element_type_e {
  TT_IREE_TTEX_TENSOR_ELEMENT_TYPE_UNKNOWN = 0,
  TT_IREE_TTEX_TENSOR_ELEMENT_TYPE_BF16 = 1,
  TT_IREE_TTEX_TENSOR_ELEMENT_TYPE_F32 = 2,
} tt_iree_ttex_tensor_element_type_e;

typedef enum tt_iree_ttex_tensor_layout_e {
  TT_IREE_TTEX_TENSOR_LAYOUT_UNKNOWN = 0,
  TT_IREE_TTEX_TENSOR_LAYOUT_ROW_MAJOR = 1,
  TT_IREE_TTEX_TENSOR_LAYOUT_CHW = 2,
  TT_IREE_TTEX_TENSOR_LAYOUT_OI = 3,
  TT_IREE_TTEX_TENSOR_LAYOUT_OIHW = 4,
  TT_IREE_TTEX_TENSOR_LAYOUT_CHANNEL_MAJOR = 5,
} tt_iree_ttex_tensor_layout_e;

typedef enum tt_iree_ttex_kernel_role_e {
  TT_IREE_TTEX_KERNEL_ROLE_UNKNOWN = 0,
  TT_IREE_TTEX_KERNEL_ROLE_READER = 1,
  TT_IREE_TTEX_KERNEL_ROLE_WRITER = 2,
  TT_IREE_TTEX_KERNEL_ROLE_COMPUTE = 3,
} tt_iree_ttex_kernel_role_e;

typedef enum tt_iree_ttex_target_arch_e {
  TT_IREE_TTEX_TARGET_ARCH_UNKNOWN = 0,
  TT_IREE_TTEX_TARGET_ARCH_WORMHOLE_B0 = 1,
  TT_IREE_TTEX_TARGET_ARCH_BLACKHOLE = 2,
} tt_iree_ttex_target_arch_e;

#define TT_IREE_TTEX_AOT_KERNEL_COUNT 3u

typedef struct tt_iree_ttex_kernel_source_desc_t {
  uint8_t role;       // tt_iree_ttex_kernel_role_e
  const char *source; // null-terminated generated source
} tt_iree_ttex_kernel_source_desc_t;

typedef struct tt_iree_ttex_artifact_file_desc_t {
  const char *path; // null-terminated path relative to the artifact root
  const uint8_t *data;
  iree_host_size_t data_length;
} tt_iree_ttex_artifact_file_desc_t;

typedef struct tt_iree_ttex_aot_kernel_bundle_desc_t {
  uint8_t target_arch; // tt_iree_ttex_target_arch_e
  iree_host_size_t kernel_count;
  const tt_iree_ttex_kernel_source_desc_t *kernels;
  iree_host_size_t artifact_count;
  const tt_iree_ttex_artifact_file_desc_t *artifacts;
} tt_iree_ttex_aot_kernel_bundle_desc_t;

#define TT_IREE_TTEX_MAX_TENSOR_RANK 4u
#define TT_IREE_TTEX_EPILOGUE_AFFINE (1u << 0)
#define TT_IREE_TTEX_EPILOGUE_RELU (1u << 1)
#define TT_IREE_TTEX_EPILOGUE_RESIDUAL (1u << 2)
#define TT_IREE_TTEX_EPILOGUE_PER_CHANNEL (1u << 3)
#define TT_IREE_TTEX_EPILOGUE_RUNTIME_AFFINE (1u << 4)
#define TT_IREE_TTEX_EPILOGUE_RESIDUAL_PER_CHANNEL (1u << 5)

typedef struct tt_iree_ttex_tensor_desc_t {
  uint16_t binding;
  uint8_t element_type;
  uint8_t layout;
  uint8_t rank;
  uint32_t shape[TT_IREE_TTEX_MAX_TENSOR_RANK];
  uint32_t storage_shape[TT_IREE_TTEX_MAX_TENSOR_RANK];
  uint32_t offset[TT_IREE_TTEX_MAX_TENSOR_RANK];
  uint64_t byte_offset;
  uint32_t stride[TT_IREE_TTEX_MAX_TENSOR_RANK];
  // One plus the dispatch-constant ordinal, or zero for a static byte offset.
  uint16_t byte_offset_constant;
} tt_iree_ttex_tensor_desc_t;

typedef struct tt_iree_ttex_matmul_like_desc_t {
  tt_iree_ttex_tensor_desc_t lhs;
  tt_iree_ttex_tensor_desc_t rhs;
  tt_iree_ttex_tensor_desc_t output;
  uint32_t m;
  uint32_t n;
  uint32_t k;
  uint32_t epilogue_flags;
  float epilogue_scale;
  float epilogue_bias;
  tt_iree_ttex_tensor_desc_t residual;
  float residual_scale;
  float residual_bias;
  iree_host_size_t epilogue_channel_count;
  const float *epilogue_scales;
  const float *epilogue_biases;
  tt_iree_ttex_tensor_desc_t epilogue_mean;
  tt_iree_ttex_tensor_desc_t epilogue_channel_scale;
  tt_iree_ttex_tensor_desc_t epilogue_channel_bias;
  float epilogue_common_scale;
  tt_iree_ttex_tensor_desc_t residual_mean;
  tt_iree_ttex_tensor_desc_t residual_channel_scale;
  tt_iree_ttex_tensor_desc_t residual_channel_bias;
  float residual_common_scale;
} tt_iree_ttex_matmul_like_desc_t;

typedef struct tt_iree_ttex_conv2d_desc_t {
  tt_iree_ttex_tensor_desc_t input;
  tt_iree_ttex_tensor_desc_t filter;
  tt_iree_ttex_tensor_desc_t output;
  uint32_t kernel_h;
  uint32_t kernel_w;
  uint32_t stride_h;
  uint32_t stride_w;
  uint32_t dilation_h;
  uint32_t dilation_w;
  uint32_t padding_top;
  uint32_t padding_bottom;
  uint32_t padding_left;
  uint32_t padding_right;
  uint32_t groups;
  uint32_t epilogue_flags;
  float epilogue_scale;
  float epilogue_bias;
  iree_host_size_t epilogue_channel_count;
  const float *epilogue_scales;
  const float *epilogue_biases;
  tt_iree_ttex_tensor_desc_t epilogue_mean;
  tt_iree_ttex_tensor_desc_t epilogue_channel_scale;
  tt_iree_ttex_tensor_desc_t epilogue_channel_bias;
  float epilogue_common_scale;
} tt_iree_ttex_conv2d_desc_t;

typedef struct tt_iree_ttex_padded_copy_desc_t {
  tt_iree_ttex_tensor_desc_t source;
  tt_iree_ttex_tensor_desc_t destination;
} tt_iree_ttex_padded_copy_desc_t;

typedef struct tt_iree_ttex_max_pool_2d_desc_t {
  tt_iree_ttex_tensor_desc_t input;
  tt_iree_ttex_tensor_desc_t output;
  uint32_t kernel_h;
  uint32_t kernel_w;
  uint32_t stride_h;
  uint32_t stride_w;
  uint32_t dilation_h;
  uint32_t dilation_w;
  uint32_t padding_top;
  uint32_t padding_bottom;
  uint32_t padding_left;
  uint32_t padding_right;
} tt_iree_ttex_max_pool_2d_desc_t;

typedef struct tt_iree_ttex_global_sum_desc_t {
  tt_iree_ttex_tensor_desc_t input;
  tt_iree_ttex_tensor_desc_t sum;
  tt_iree_ttex_tensor_desc_t divisor;
  uint32_t reduce_h;
  uint32_t reduce_w;
  float divisor_value;
} tt_iree_ttex_global_sum_desc_t;

typedef struct tt_iree_ttex_scalar_div_desc_t {
  tt_iree_ttex_tensor_desc_t input;
  tt_iree_ttex_tensor_desc_t divisor;
  tt_iree_ttex_tensor_desc_t output;
} tt_iree_ttex_scalar_div_desc_t;

typedef struct tt_iree_ttex_linear_desc_t {
  tt_iree_ttex_tensor_desc_t input;
  tt_iree_ttex_tensor_desc_t weights;
  tt_iree_ttex_tensor_desc_t bias;
  tt_iree_ttex_tensor_desc_t output;
  uint32_t m;
  uint32_t n;
  uint32_t k;
} tt_iree_ttex_linear_desc_t;

// Describes a single entry point to be serialized into a TTEX blob.
// Field semantics match EntryPointDef in tt_executable_def.fbs.
typedef struct tt_iree_ttex_entry_point_desc_t {
  const char *name; // null-terminated, may be NULL (defaults to "")
  uint16_t constant_count;
  uint16_t binding_count;
  uint64_t flags;
  uint32_t workgroup_size[3]; // [x, y, z]
  uint32_t builtin_program;   // BuiltinProgram enum value
  uint32_t builtin_m_tiles;   // static matmul M tile count, 0 if dynamic/unused
  uint32_t builtin_n_tiles;   // static matmul N tile count, 0 if dynamic/unused
  uint32_t builtin_k_tiles;   // static matmul K tile count, 0 if dynamic/unused
  uint8_t dispatch_kind;      // tt_iree_ttex_dispatch_kind_e
  tt_iree_ttex_matmul_like_desc_t matmul_like;
  tt_iree_ttex_conv2d_desc_t conv2d;
  tt_iree_ttex_padded_copy_desc_t padded_copy;
  tt_iree_ttex_max_pool_2d_desc_t max_pool;
  tt_iree_ttex_global_sum_desc_t global_sum;
  tt_iree_ttex_scalar_div_desc_t scalar_div;
  tt_iree_ttex_linear_desc_t linear;
  const tt_iree_ttex_aot_kernel_bundle_desc_t *aot_kernel_bundle;
} tt_iree_ttex_entry_point_desc_t;

// Builds a complete TTEX FlatBuffer from the given entry point descriptors.
//
// The resulting bytes are allocated via |allocator| and returned in |out_data|.
// Caller must free with iree_allocator_free(allocator, out_data->data).
//
// The blob will:
//   - Have file_identifier "TTEX"
//   - Pass flatcc verify_as_root()
//   - Contain |entry_point_count| EntryPointDef entries
iree_status_t tt_iree_build_ttex_executable_def(
    iree_allocator_t allocator, iree_host_size_t entry_point_count,
    const tt_iree_ttex_entry_point_desc_t *entry_points,
    iree_byte_span_t *out_data);

// Builds a TTEX blob with an explicit version. Intended for compatibility
// tests; production serializers should use the current-version wrapper above.
iree_status_t tt_iree_build_ttex_executable_def_with_version(
    iree_allocator_t allocator, uint32_t version,
    iree_host_size_t entry_point_count,
    const tt_iree_ttex_entry_point_desc_t *entry_points,
    iree_byte_span_t *out_data);

#ifdef __cplusplus
}
#endif

#endif // IREE_SCHEMA_TT_EXECUTABLE_BUILDER_UTIL_H_
