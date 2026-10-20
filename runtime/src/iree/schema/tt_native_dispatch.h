// Copyright 2026 The tt-iree Authors
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#ifndef IREE_SCHEMA_TT_NATIVE_DISPATCH_H_
#define IREE_SCHEMA_TT_NATIVE_DISPATCH_H_

#include "iree/schema/tt_executable_builder_util.h"

// v13+ native AOT argument ABI: ten (DRAM address, byte base) pairs followed
// by a tile rectangle (M start, N start, M count, N count), or element range.
// Absent operands use the output accessor but are never dereferenced.
#define TT_IREE_NATIVE_TENSOR_COUNT 10u
#define TT_IREE_NATIVE_PARTITION_ARG 20u
#define TT_IREE_NATIVE_MAX_TENSOR_COUNT TT_IREE_BF16_NATIVE_TENSOR_COUNT

static inline bool tt_iree_ttex_is_native_matrix_kind(uint8_t kind) {
  return kind == TT_IREE_TTEX_DISPATCH_KIND_MATMUL_LIKE ||
         kind == TT_IREE_TTEX_DISPATCH_KIND_CONVOLUTION ||
         kind == TT_IREE_TTEX_DISPATCH_KIND_LINEAR;
}

static inline void
tt_iree_ttex_native_tensors(const tt_iree_ttex_entry_point_desc_t *ep,
                            const tt_iree_ttex_tensor_desc_t **t) {
  // Allocate MAX_TENSOR_COUNT slots when accepting v17; legacy-only callers
  // can retain the ten-slot array and ABI.
  for (unsigned i = 0; i < TT_IREE_NATIVE_TENSOR_COUNT; ++i)
    t[i] = NULL;
  switch (ep->dispatch_kind) {
  case TT_IREE_TTEX_DISPATCH_KIND_MATMUL_LIKE: {
    const tt_iree_ttex_matmul_like_desc_t *m = &ep->matmul_like;
    t[0] = &m->lhs;
    t[1] = &m->rhs;
    t[2] = &m->output;
    if (m->epilogue_flags & TT_IREE_TTEX_EPILOGUE_RUNTIME_AFFINE) {
      t[3] = &m->epilogue_mean;
      t[4] = &m->epilogue_channel_scale;
      t[5] = &m->epilogue_channel_bias;
    }
    if (m->epilogue_flags & TT_IREE_TTEX_EPILOGUE_RESIDUAL)
      t[6] = &m->residual;
    if (m->epilogue_flags & TT_IREE_TTEX_EPILOGUE_RESIDUAL_PER_CHANNEL) {
      t[7] = &m->residual_mean;
      t[8] = &m->residual_channel_scale;
      t[9] = &m->residual_channel_bias;
    }
    break;
  }
  case TT_IREE_TTEX_DISPATCH_KIND_CONVOLUTION: {
    const tt_iree_ttex_conv2d_desc_t *c = &ep->conv2d;
    t[0] = &c->input;
    t[1] = &c->filter;
    t[2] = &c->output;
    if (c->epilogue_flags & TT_IREE_TTEX_EPILOGUE_RUNTIME_AFFINE) {
      t[3] = &c->epilogue_mean;
      t[4] = &c->epilogue_channel_scale;
      t[5] = &c->epilogue_channel_bias;
    }
    break;
  }
  case TT_IREE_TTEX_DISPATCH_KIND_LINEAR:
    t[0] = &ep->linear.input;
    t[1] = &ep->linear.weights;
    t[2] = &ep->linear.output;
    t[5] = &ep->linear.bias;
    break;
  case TT_IREE_TTEX_DISPATCH_KIND_MAX_POOL:
    t[0] = &ep->max_pool.input;
    t[2] = &ep->max_pool.output;
    break;
  case TT_IREE_TTEX_DISPATCH_KIND_GLOBAL_SUM:
    t[0] = &ep->global_sum.input;
    t[1] = &ep->global_sum.divisor;
    t[2] = &ep->global_sum.sum;
    break;
  case TT_IREE_TTEX_DISPATCH_KIND_SCALAR_DIV:
    t[0] = &ep->scalar_div.input;
    t[1] = &ep->scalar_div.divisor;
    t[2] = &ep->scalar_div.output;
    break;
  case TT_IREE_TTEX_DISPATCH_KIND_BF16_NATIVE:
    for (unsigned i = 0; i < TT_IREE_BF16_NATIVE_TENSOR_COUNT; ++i)
      t[i] = ep->bf16_native.tensors[i].rank ? &ep->bf16_native.tensors[i] : NULL;
    break;
  default:
    break;
  }
}
#endif
