// Copyright 2026 The tt-iree Authors
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#ifndef IREE_SCHEMA_TT_BF16_NATIVE_VALIDATION_H_
#define IREE_SCHEMA_TT_BF16_NATIVE_VALIDATION_H_

#include "iree/schema/tt_native_dispatch.h"

static inline bool tt_iree_ttex_is_batch_layout(uint8_t layout) {
  return layout == TT_IREE_TTEX_TENSOR_LAYOUT_NCHW ||
         layout == TT_IREE_TTEX_TENSOR_LAYOUT_BATCH_CHANNEL_MAJOR;
}

static inline bool tt_iree_bf16_layout_rank_is_valid(uint8_t layout,
                                                     uint8_t rank) {
  switch (layout) {
  case TT_IREE_TTEX_TENSOR_LAYOUT_ROW_MAJOR:
    return rank == 1 || rank == 2;
  case TT_IREE_TTEX_TENSOR_LAYOUT_OI:
  case TT_IREE_TTEX_TENSOR_LAYOUT_CHANNEL_MAJOR:
    return rank == 2;
  case TT_IREE_TTEX_TENSOR_LAYOUT_CHW:
  case TT_IREE_TTEX_TENSOR_LAYOUT_BATCH_CHANNEL_MAJOR:
    return rank == 3;
  case TT_IREE_TTEX_TENSOR_LAYOUT_OIHW:
  case TT_IREE_TTEX_TENSOR_LAYOUT_NCHW:
    return rank == 4;
  default:
    return false;
  }
}

static inline iree_status_t
tt_iree_validate_bf16_layout_version(const tt_iree_ttex_bf16_native_desc_t *d,
                                     uint32_t version) {
  for (unsigned i = 0; i < TT_IREE_BF16_NATIVE_TENSOR_COUNT; ++i) {
    if (tt_iree_ttex_is_batch_layout(d->tensors[i].layout) &&
        version != TT_IREE_TTEX_VERSION_20)
      return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                              "BF16 batch layouts require opt-in TTEX v20");
  }
  return iree_ok_status();
}

// Division before multiplication keeps even malformed rank-4 shapes safe.
static inline bool tt_iree_bf16_product_matches(const uint32_t *shape,
                                                unsigned begin, unsigned end,
                                                uint32_t expected) {
  uint64_t product = 1;
  for (unsigned i = begin; i < end; ++i) {
    if (!shape[i] || product > UINT32_MAX / shape[i])
      return false;
    product *= shape[i];
  }
  return product == expected;
}

static inline bool
tt_iree_bf16_same_shape(const tt_iree_ttex_tensor_desc_t *a,
                        const tt_iree_ttex_tensor_desc_t *b) {
  if (a->rank != b->rank || a->layout != b->layout)
    return false;
  for (unsigned i = 0; i < a->rank; ++i)
    if (a->shape[i] != b->shape[i])
      return false;
  return true;
}

static inline bool tt_iree_bf16_matrix_geometry_is_valid(
    const tt_iree_ttex_bf16_native_desc_t *d) {
  const auto &a = d->tensors[0], &b = d->tensors[1], &o = d->tensors[2];
  if (!d->m || !d->n || !d->k || d->m > UINT32_MAX - 31 ||
      d->n > UINT32_MAX - 31 || d->k > UINT32_MAX - 31 || b.shape[0] != d->n ||
      a.layout != o.layout || a.rank != o.rank || a.rank > 4 ||
      (d->operation_kind == TT_IREE_TTEX_DISPATCH_KIND_LINEAR &&
       a.layout != TT_IREE_TTEX_TENSOR_LAYOUT_ROW_MAJOR))
    return false;
  const bool conv = d->operation_kind == TT_IREE_TTEX_DISPATCH_KIND_CONVOLUTION;
  if ((conv && (b.layout != TT_IREE_TTEX_TENSOR_LAYOUT_OIHW || b.rank != 4)) ||
      (!conv && (b.layout != TT_IREE_TTEX_TENSOR_LAYOUT_OI || b.rank != 2)))
    return false;
  if (a.layout == TT_IREE_TTEX_TENSOR_LAYOUT_ROW_MAJOR)
    return !conv && a.rank == 2 && a.shape[0] == d->m && a.shape[1] == d->k &&
           b.shape[1] == d->k && o.shape[0] == d->m && o.shape[1] == d->n;
  const bool batch = tt_iree_ttex_is_batch_layout(a.layout);
  const unsigned c = batch ? 1 : 0;
  const bool spatial = a.layout == TT_IREE_TTEX_TENSOR_LAYOUT_CHW ||
                       a.layout == TT_IREE_TTEX_TENSOR_LAYOUT_NCHW;
  if (a.rank != (spatial ? c + 3 : c + 2) ||
      (!spatial && a.layout != TT_IREE_TTEX_TENSOR_LAYOUT_CHANNEL_MAJOR &&
       a.layout != TT_IREE_TTEX_TENSOR_LAYOUT_BATCH_CHANNEL_MAJOR) ||
      (conv && !spatial) || (batch && a.shape[0] != o.shape[0]) ||
      a.shape[c] != b.shape[1] || o.shape[c] != d->n)
    return false;
  if (!tt_iree_bf16_product_matches(b.shape, 1, b.rank, d->k))
    return false;
  uint64_t m = batch ? o.shape[0] : 1;
  for (unsigned i = c + 1; i < o.rank; ++i) {
    if (!o.shape[i] || m > UINT32_MAX / o.shape[i])
      return false;
    m *= o.shape[i];
    if (!conv && a.shape[i] != o.shape[i])
      return false;
  }
  return m == d->m;
}

static inline bool
tt_iree_bf16_conv_spatial_is_valid(const tt_iree_ttex_bf16_native_desc_t *d) {
  const auto &a = d->tensors[0], &b = d->tensors[1], &o = d->tensors[2];
  const unsigned h = a.layout == TT_IREE_TTEX_TENSOR_LAYOUT_NCHW ? 2 : 1;
  const uint32_t kh = b.shape[2], kw = b.shape[3];
  if (!((kh == 3 && kw == 3) || (kh == 7 && kw == 7)) || a.shape[h] < kh ||
      a.shape[h + 1] < kw)
    return false;
  for (uint32_t stride = kh == 7 ? 2 : 1; stride <= 2; ++stride) {
    if ((a.shape[h] - kh) / stride + 1 == o.shape[h] &&
        (a.shape[h + 1] - kw) / stride + 1 == o.shape[h + 1])
      return true;
  }
  return false;
}

// v20 retains the v18 tensor descriptor rules, including arbitrary epilogue
// slots. Only slots 0/1/2 acquire operation-specific static batch geometry.
static inline iree_status_t
tt_iree_validate_bf16_native_v20(const tt_iree_ttex_bf16_native_desc_t *d,
                                 uint32_t builtin, uint16_t binding_count,
                                 uint16_t constant_count) {
  if (!tt_iree_ttex_builtin_matches_dispatch_kind(
          builtin, TT_IREE_TTEX_DISPATCH_KIND_BF16_NATIVE) ||
      d->operation_kind < TT_IREE_TTEX_DISPATCH_KIND_MATMUL_LIKE ||
      d->operation_kind > TT_IREE_TTEX_DISPATCH_KIND_LINEAR)
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "invalid v20 BF16 operation/builtin");
  for (unsigned i = 0; i < TT_IREE_BF16_NATIVE_TENSOR_COUNT; ++i) {
    const auto &t = d->tensors[i];
    if (!t.rank)
      continue;
    if (t.rank > TT_IREE_TTEX_MAX_TENSOR_RANK || t.binding >= binding_count ||
        t.byte_offset_constant > constant_count ||
        (t.element_type != TT_IREE_TTEX_TENSOR_ELEMENT_TYPE_BF16 &&
         t.element_type != TT_IREE_TTEX_TENSOR_ELEMENT_TYPE_F32) ||
        !tt_iree_bf16_layout_rank_is_valid(t.layout, t.rank))
      return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                              "invalid v20 BF16 tensor slot %u", i);
    uint64_t count = 1;
    for (unsigned j = 0; j < t.rank; ++j) {
      uint64_t last =
          uint64_t(t.offset[j]) + uint64_t(t.shape[j] ? t.shape[j] - 1 : 0) *
                                      (t.stride[j] ? t.stride[j] : 1);
      if (!t.shape[j] || !t.storage_shape[j] || last >= t.storage_shape[j] ||
          count > UINT64_MAX / t.storage_shape[j])
        return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                                "invalid v20 BF16 shape/storage in slot %u", i);
      count *= t.storage_shape[j];
    }
    const uint64_t bytes =
        t.element_type == TT_IREE_TTEX_TENSOR_ELEMENT_TYPE_BF16 ? 2 : 4;
    if (t.byte_offset % bytes || count > UINT64_MAX / bytes ||
        t.byte_offset > UINT64_MAX - count * bytes)
      return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                              "v20 BF16 tensor byte range overflows");
  }
  const auto &a = d->tensors[0], &b = d->tensors[1], &o = d->tensors[2];
  if (!a.rank || !o.rank ||
      a.element_type != TT_IREE_TTEX_TENSOR_ELEMENT_TYPE_BF16 ||
      o.element_type != TT_IREE_TTEX_TENSOR_ELEMENT_TYPE_BF16)
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "v20 BF16 input/output must be present and BF16");
  if (tt_iree_ttex_is_native_matrix_kind(d->operation_kind)) {
    if (!b.rank || b.element_type != TT_IREE_TTEX_TENSOR_ELEMENT_TYPE_BF16 ||
        !tt_iree_bf16_matrix_geometry_is_valid(d) ||
        (d->operation_kind == TT_IREE_TTEX_DISPATCH_KIND_CONVOLUTION &&
         !tt_iree_bf16_conv_spatial_is_valid(d)))
      return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                              "v20 BF16 matrix shape/layout/dtype mismatch");
    return iree_ok_status();
  }
  if (builtin != TT_IREE_TTEX_BUILTIN_PROGRAM_BF16_NATIVE || d->m || d->n ||
      d->k)
    return iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "v20 BF16 scalar requires builtin 11 and zero M/N/K");
  bool valid = false;
  switch (d->operation_kind) {
  case TT_IREE_TTEX_DISPATCH_KIND_PADDED_COPY:
    valid = !b.rank && tt_iree_bf16_same_shape(&a, &o) &&
            ((a.layout == TT_IREE_TTEX_TENSOR_LAYOUT_NCHW && a.rank == 4) ||
             (a.layout == TT_IREE_TTEX_TENSOR_LAYOUT_CHW && a.rank == 3));
    break;
  case TT_IREE_TTEX_DISPATCH_KIND_MAX_POOL: {
    unsigned c = a.layout == TT_IREE_TTEX_TENSOR_LAYOUT_NCHW ? 1 : 0;
    valid = !b.rank && a.layout == o.layout && a.rank == o.rank &&
            a.rank == c + 3 &&
            (a.layout == TT_IREE_TTEX_TENSOR_LAYOUT_NCHW ||
             a.layout == TT_IREE_TTEX_TENSOR_LAYOUT_CHW) &&
            (!c || a.shape[0] == o.shape[0]) && a.shape[c] == o.shape[c] &&
            a.shape[c + 1] >= 3 && a.shape[c + 2] >= 3 &&
            (a.shape[c + 1] - 3) / 2 + 1 == o.shape[c + 1] &&
            (a.shape[c + 2] - 3) / 2 + 1 == o.shape[c + 2];
    break;
  }
  case TT_IREE_TTEX_DISPATCH_KIND_GLOBAL_SUM: {
    unsigned c = a.layout == TT_IREE_TTEX_TENSOR_LAYOUT_NCHW ? 1 : 0;
    uint64_t channels = uint64_t(a.shape[c]) * (c ? a.shape[0] : 1);
    valid = a.rank == c + 3 &&
            (a.layout == TT_IREE_TTEX_TENSOR_LAYOUT_NCHW ||
             a.layout == TT_IREE_TTEX_TENSOR_LAYOUT_CHW) &&
            a.shape[c + 1] == 8 && a.shape[c + 2] == 8 &&
            channels <= UINT32_MAX &&
            o.layout == TT_IREE_TTEX_TENSOR_LAYOUT_ROW_MAJOR && o.rank == 1 &&
            o.shape[0] == channels;
    break;
  }
  case TT_IREE_TTEX_DISPATCH_KIND_SCALAR_DIV:
    valid = a.layout == TT_IREE_TTEX_TENSOR_LAYOUT_ROW_MAJOR && a.rank == 1 &&
            tt_iree_bf16_same_shape(&a, &o);
    break;
  }
  if (d->operation_kind == TT_IREE_TTEX_DISPATCH_KIND_GLOBAL_SUM ||
      d->operation_kind == TT_IREE_TTEX_DISPATCH_KIND_SCALAR_DIV)
    valid = valid && b.rank == 1 && b.shape[0] == 1 &&
            b.layout == TT_IREE_TTEX_TENSOR_LAYOUT_ROW_MAJOR &&
            b.element_type == TT_IREE_TTEX_TENSOR_ELEMENT_TYPE_BF16;
  uint64_t output_count = 1;
  for (unsigned i = 0; i < o.rank; ++i) {
    if (output_count > UINT32_MAX / o.shape[i]) {
      valid = false;
      break;
    }
    output_count *= o.shape[i];
  }
  if (!valid)
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "v20 BF16 scalar shape/layout/dtype mismatch");
  return iree_ok_status();
}

#endif // IREE_SCHEMA_TT_BF16_NATIVE_VALIDATION_H_
