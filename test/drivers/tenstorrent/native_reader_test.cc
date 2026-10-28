// Copyright 2026 The tt-iree Authors
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include <array>
#include <cmath>
#include <cstring>
#include <vector>

#include "TenstorrentAOTCompiler.h"
#include "TenstorrentNativeKernels.h"
#include "iree/hal/drivers/tenstorrent/tt_command_buffer.h"
#include "iree/hal/drivers/tenstorrent/tt_executable.h"
#include "iree/hal/drivers/tenstorrent/tt_executable_cache.h"
#include "iree/schema/tt_executable_builder_util.h"
#include "utils.h"

namespace iree::hal::tenstorrent {
namespace {
using namespace mlir::iree_compiler::IREE::HAL;
using Tensor = tt_iree_ttex_tensor_desc_t;

Tensor MakeTensor(uint16_t binding, uint8_t layout,
                  std::initializer_list<uint32_t> dims) {
  Tensor t = {};
  t.binding = binding;
  t.layout = layout;
  t.element_type = TT_IREE_TTEX_TENSOR_ELEMENT_TYPE_BF16;
  t.rank = dims.size();
  unsigned i = 0;
  for (auto d : dims) {
    t.shape[i] = t.storage_shape[i] = d;
    ++i;
  }
  return t;
}

enum ReaderCase {
  Conv,
  Linear,
  StridedConv,
  StridedProjection,
  ComposedConv,
  Stem,
  OddProjection,
  CaseCount
};
bool IsConv(ReaderCase kind) {
  return kind != Linear && kind != StridedProjection && kind != OddProjection;
}

tt_iree_ttex_entry_point_desc_t Entry(ReaderCase kind, bool prefetch) {
  bool conv = IsConv(kind);
  tt_iree_ttex_entry_point_desc_t ep = {};
  ep.name = conv ? (prefetch ? "prefetch_conv" : "legacy_conv")
                 : (prefetch ? "prefetch_linear" : "legacy_linear");
  ep.binding_count = 3;
  ep.workgroup_size[0] = ep.workgroup_size[1] = ep.workgroup_size[2] = 1;
  ep.builtin_program = TT_IREE_TTEX_BUILTIN_PROGRAM_BF16_NATIVE_TENSIX;
  ep.dispatch_kind = TT_IREE_TTEX_DISPATCH_KIND_BF16_NATIVE;
  auto &d = ep.bf16_native;
  d.operation_kind = conv ? TT_IREE_TTEX_DISPATCH_KIND_CONVOLUTION
                          : TT_IREE_TTEX_DISPATCH_KIND_LINEAR;
  d.m = conv ? 77 : 3;
  d.n = conv ? 5 : 33;
  d.k = conv ? 27 : 35;
  if (conv) {
    d.tensors[0] = MakeTensor(0, TT_IREE_TTEX_TENSOR_LAYOUT_CHW, {3, 9, 13});
    d.tensors[1] = MakeTensor(1, TT_IREE_TTEX_TENSOR_LAYOUT_OIHW, {5, 3, 3, 3});
    d.tensors[2] = MakeTensor(2, TT_IREE_TTEX_TENSOR_LAYOUT_CHW, {5, 7, 11});
    d.tensors[0].storage_shape[1] = 11;
    d.tensors[0].storage_shape[2] = 17;
    d.tensors[0].offset[1] = 1;
    d.tensors[0].offset[2] = 2;
    d.tensors[1].storage_shape[0] = 7;
    d.tensors[1].offset[0] = 1;
    d.tensors[2].storage_shape[1] = 9;
    d.tensors[2].storage_shape[2] = 15;
    d.tensors[2].offset[1] = 1;
    d.tensors[2].offset[2] = 2;
  } else {
    d.tensors[0] = MakeTensor(0, TT_IREE_TTEX_TENSOR_LAYOUT_ROW_MAJOR, {3, 35});
    d.tensors[1] = MakeTensor(1, TT_IREE_TTEX_TENSOR_LAYOUT_OI, {33, 35});
    d.tensors[2] = MakeTensor(2, TT_IREE_TTEX_TENSOR_LAYOUT_ROW_MAJOR, {3, 33});
    for (unsigned i : {0, 1, 2}) {
      d.tensors[i].storage_shape[0] += 2;
      d.tensors[i].storage_shape[1] += 6;
      d.tensors[i].offset[0] = 1;
      d.tensors[i].offset[1] = 2;
    }
  }
  if (kind == StridedConv || kind == ComposedConv || kind == Stem) {
    uint32_t kernel = kind == Stem ? 7 : 3;
    uint32_t height = (11 - kernel) / 2 + 1, width = (43 - kernel) / 2 + 1;
    d.m = height * width;
    d.k = 3 * kernel * kernel;
    d.tensors[0] = MakeTensor(0, TT_IREE_TTEX_TENSOR_LAYOUT_CHW, {3, 11, 43});
    d.tensors[1] =
        MakeTensor(1, TT_IREE_TTEX_TENSOR_LAYOUT_OIHW, {5, 3, kernel, kernel});
    d.tensors[2] =
        MakeTensor(2, TT_IREE_TTEX_TENSOR_LAYOUT_CHW, {5, height, width});
    d.tensors[0].storage_shape[1] = 13;
    d.tensors[0].storage_shape[2] = 47;
    d.tensors[1].storage_shape[0] = 7;
    d.tensors[1].offset[0] = 1;
    d.tensors[2].storage_shape[1] += 2;
    d.tensors[2].storage_shape[2] += 4;
    for (unsigned slot : {0, 2}) {
      d.tensors[slot].offset[1] = 1;
      d.tensors[slot].offset[2] = 2;
    }
    if (kind == ComposedConv) {
      d.tensors[0].storage_shape[1] = 25;
      d.tensors[0].storage_shape[2] = 91;
      d.tensors[0].stride[1] = d.tensors[0].stride[2] = 2;
    }
  } else if (kind == StridedProjection || kind == OddProjection) {
    const uint32_t height = kind == OddProjection ? 7 : 5;
    const uint32_t width = kind == OddProjection ? 11 : 21;
    d.operation_kind = TT_IREE_TTEX_DISPATCH_KIND_MATMUL_LIKE;
    d.m = height * width;
    d.n = kind == OddProjection ? 65 : 33;
    d.k = 35;
    d.tensors[0] = MakeTensor(0, TT_IREE_TTEX_TENSOR_LAYOUT_CHW, {35, height, width});
    d.tensors[1] = MakeTensor(1, TT_IREE_TTEX_TENSOR_LAYOUT_OI, {d.n, 35});
    d.tensors[2] = MakeTensor(2, TT_IREE_TTEX_TENSOR_LAYOUT_CHW, {d.n, height, width});
    d.tensors[0].storage_shape[1] = height * 2 + 1;
    d.tensors[0].storage_shape[2] = width * 2 + 5;
    d.tensors[0].stride[1] = d.tensors[0].stride[2] = 2;
    d.tensors[1].storage_shape[0] = d.n + 2;
    d.tensors[1].storage_shape[1] = 41;
    d.tensors[1].offset[0] = 1;
    d.tensors[1].offset[1] = 2;
    d.tensors[2].storage_shape[1] = height + 2;
    d.tensors[2].storage_shape[2] = width + 4;
    for (unsigned slot : {0, 2}) {
      d.tensors[slot].offset[1] = 1;
      d.tensors[slot].offset[2] = 2;
    }
  }
  // First logical LHS/output elements sit at byte 2046; reads/writes must split
  // pages.
  d.tensors[0].byte_offset =
      2046 - 2 * (d.tensors[0].storage_shape[d.tensors[0].rank - 1] + 2);
  d.tensors[1].byte_offset = conv ? 2014 - 2 * d.k : 1960;
  d.tensors[2].byte_offset =
      2046 - 2 * (d.tensors[2].storage_shape[d.tensors[2].rank - 1] + 2);
  return ep;
}

size_t Index(const Tensor &t, std::initializer_list<uint32_t> coords) {
  size_t i = 0, axis = 0;
  for (auto x : coords) {
    i = i * t.storage_shape[axis] + t.offset[axis] +
        x * (t.stride[axis] ? t.stride[axis] : 1);
    ++axis;
  }
  return t.byte_offset / 2 + i;
}
size_t Storage(const Tensor &t) {
  size_t n = 1;
  for (unsigned i = 0; i < t.rank; ++i)
    n *= t.storage_shape[i];
  return t.byte_offset / 2 + n + 32;
}
float Float(uint16_t bits) {
  uint32_t word = uint32_t(bits) << 16;
  float value;
  std::memcpy(&value, &word, 4);
  return value;
}
uint16_t Bits(float value) {
  uint32_t word;
  std::memcpy(&word, &value, 4);
  return (word + 0x7fff + ((word >> 16) & 1)) >> 16;
}
uint16_t Input(size_t i, unsigned seed) {
  if (!seed)
    return Bits(float(int(i * 13 % 5) - 2));
  if (i % 17 == 0)
    return i % 2 ? 0x8000 : 0;
  return uint16_t(((i + seed) % 2) << 15 | (125 + (i + seed) % 5) << 7 |
                  (i * 29 + seed * 31) % 128);
}

bool Build(ReaderCase kind, bool prefetch, std::vector<uint8_t> &wire,
           std::string &error, bool reuse = false) {
  bool conv = IsConv(kind);
  auto ep = Entry(kind, prefetch);
  auto &d = ep.bf16_native;
  tt_iree_ttex_entry_point_desc_t geometry = {};
  geometry.dispatch_kind = d.operation_kind;
  if (conv) {
    geometry.conv2d.input = d.tensors[0];
    geometry.conv2d.filter = d.tensors[1];
    geometry.conv2d.output = d.tensors[2];
    geometry.conv2d.kernel_h = d.tensors[1].shape[2];
    geometry.conv2d.kernel_w = d.tensors[1].shape[3];
    geometry.conv2d.stride_h = geometry.conv2d.stride_w = kind == Conv ? 1 : 2;
    geometry.conv2d.dilation_h = geometry.conv2d.dilation_w = 1;
  } else if (kind == Linear) {
    geometry.linear.m = d.m;
    geometry.linear.n = d.n;
    geometry.linear.k = d.k;
  } else {
    geometry.matmul_like.lhs = d.tensors[0];
    geometry.matmul_like.rhs = d.tensors[1];
    geometry.matmul_like.output = d.tensors[2];
    geometry.matmul_like.m = d.m;
    geometry.matmul_like.n = d.n;
    geometry.matmul_like.k = d.k;
  }
  const Tensor *t[TT_IREE_BF16_NATIVE_TENSOR_COUNT] = {
      &d.tensors[0], &d.tensors[1], &d.tensors[2]};
  // Zero-bias contractions isolate reader transport from unchanged epilogues.
  auto source = generateBF16NativeMatrix(
      geometry, t, "value=round_bf16(value);\n", true, prefetch, reuse);
  TenstorrentAOTKernelBundle bundle;
  if (!compileBF16NativeAOT(ep, source.reader, source,
                            TenstorrentAOTTargetArch::Blackhole, bundle, error))
    return false;
  std::vector<tt_iree_ttex_kernel_source_desc_t> sources;
  std::vector<tt_iree_ttex_artifact_file_desc_t> artifacts;
  for (auto &k : bundle.kernels)
    sources.push_back({k.role, k.source.c_str()});
  for (auto &a : bundle.artifacts)
    artifacts.push_back({a.path.c_str(), a.data.data(), a.data.size()});
  tt_iree_ttex_aot_kernel_bundle_desc_t aot = {
      static_cast<uint8_t>(bundle.targetArch), sources.size(), sources.data(),
      artifacts.size(), artifacts.data()};
  ep.aot_kernel_bundle = &aot;
  iree_byte_span_t data = {};
  auto result = tt_iree_build_ttex_executable_def_with_version(
      iree_allocator_system(), TT_IREE_TTEX_VERSION_18, 1, &ep, &data);
  if (!iree_status_is_ok(result)) {
    iree_status_fprint(stderr, result);
    iree_status_ignore(result);
    return false;
  }
  wire.assign(data.data, data.data + data.data_length);
  iree_allocator_free(iree_allocator_system(), data.data);
  return true;
}

class NativeReaderTest : public TenstorrentTestBase {
protected:
  static void SetUpTestSuite() {
    for (unsigned c = 0; c < CaseCount; ++c)
      for (unsigned r = 0; r < 3; ++r) {
        std::string error;
        ASSERT_TRUE(
            Build(static_cast<ReaderCase>(c), r != 0, wires_[c][r], error,
                  r == 2))
            << error;
      }
    TenstorrentTestBase::SetUpTestSuite();
  }
  void Check(ReaderCase kind, bool single_core = false) {
    bool conv = IsConv(kind), spatial = kind != Linear;
    auto ep = Entry(kind, true);
    auto &d = ep.bf16_native;
    std::array<iree::vm::ref<iree_hal_buffer_t>, 3> buffers;
    iree_hal_buffer_params_t params = {};
    params.type = IREE_HAL_MEMORY_TYPE_DEVICE_LOCAL;
    params.access = IREE_HAL_MEMORY_ACCESS_ALL;
    params.usage =
        IREE_HAL_BUFFER_USAGE_DEFAULT | IREE_HAL_BUFFER_USAGE_MAPPING;
    for (unsigned i = 0; i < 3; ++i)
      IREE_ASSERT_OK(iree_hal_allocator_allocate_buffer(
          device_allocator(), params, Storage(d.tensors[i]) * 2, &buffers[i]));
    std::array<iree::vm::ref<iree_hal_executable_t>, 3> executables;
    for (unsigned r = 0; r < 3; ++r) {
      iree_hal_executable_params_t p;
      iree_hal_executable_params_initialize(&p);
      p.executable_format =
          iree_make_cstring_view(IREE_HAL_TT_EXECUTABLE_FORMAT);
      auto &wire = wires_[kind][r];
      p.executable_data = iree_make_const_byte_span(wire.data(), wire.size());
      IREE_ASSERT_OK(iree_hal_tt_executable_create(
          device_, &p, host_allocator(), &executables[r]));
    }
    std::vector<uint16_t> repeat;
    for (unsigned seed : {0, 1, 2, 1}) {
      SCOPED_TRACE(seed);
      std::vector<uint16_t> a(Storage(d.tensors[0]), 0x55aa),
          b(Storage(d.tensors[1]), 0x55aa);
      uint32_t width = spatial ? d.tensors[2].shape[2] : 0;
      uint32_t kh = conv ? d.tensors[1].shape[2] : 0,
               kw = conv ? d.tensors[1].shape[3] : 0;
      uint32_t stride = kind == Conv ? 1 : 2;
      auto ai = [&](uint32_t m, uint32_t k) {
        return conv      ? Index(d.tensors[0], {k / (kh * kw),
                                                (m / width) * stride + (k / kw) % kh,
                                                (m % width) * stride + k % kw})
               : spatial ? Index(d.tensors[0], {k, m / width, m % width})
                         : Index(d.tensors[0], {m, k});
      };
      auto bi = [&](uint32_t n, uint32_t k) {
        return conv ? Index(d.tensors[1],
                            {n, k / (kh * kw), (k / kw) % kh, k % kw})
                    : Index(d.tensors[1], {n, k});
      };
      auto oi = [&](uint32_t m, uint32_t n) {
        return spatial ? Index(d.tensors[2], {n, m / width, m % width})
                       : Index(d.tensors[2], {m, n});
      };
      for (uint32_t m = 0; m < d.m; ++m)
        for (uint32_t k = 0; k < d.k; ++k)
          a[ai(m, k)] = Input(ai(m, k), seed);
      for (uint32_t n = 0; n < d.n; ++n)
        for (uint32_t k = 0; k < d.k; ++k)
          b[bi(n, k)] = Input(bi(n, k) + 7, seed);
      IREE_ASSERT_OK(iree_hal_buffer_map_write(buffers[0].get(), 0, a.data(),
                                               a.size() * 2));
      IREE_ASSERT_OK(iree_hal_buffer_map_write(buffers[1].get(), 0, b.data(),
                                               b.size() * 2));
      std::vector<uint16_t> reference;
      for (unsigned r = 0; r < 3; ++r) {
        SCOPED_TRACE(r);
        std::vector<uint16_t> actual(Storage(d.tensors[2]), 0x55aa);
        IREE_ASSERT_OK(iree_hal_buffer_map_write(
            buffers[2].get(), 0, actual.data(), actual.size() * 2));
        iree::vm::ref<iree_hal_command_buffer_t> command;
        IREE_ASSERT_OK(iree_hal_tt_device_create_command_buffer(
            device_, IREE_HAL_COMMAND_BUFFER_MODE_ONE_SHOT,
            IREE_HAL_COMMAND_CATEGORY_DISPATCH, IREE_HAL_QUEUE_AFFINITY_ANY, 0,
            &command));
        IREE_ASSERT_OK(iree_hal_command_buffer_begin(command.get()));
        iree_hal_buffer_ref_t refs[3];
        for (unsigned i = 0; i < 3; ++i)
          refs[i] = iree_hal_make_buffer_ref(buffers[i].get(), 0,
                                             Storage(d.tensors[i]) * 2);
        uint32_t mt = (d.m + 31) / 32,
                 rows = mt % 3 == 0 ? 3 : (mt % 2 == 0 ? 2 : 1);
        iree_hal_dispatch_config_t grid = {{0, 0, 0},
                                           {(d.n + 31) / 32, rows, 1}};
        if (kind == OddProjection)
          grid = {{0, 0, 0}, {2, 2, 1}};
        if (single_core)
          grid = {{0, 0, 0}, {1, 1, 1}};
        IREE_ASSERT_OK(iree_hal_command_buffer_dispatch(
            command.get(), executables[r].get(), 0, grid,
            iree_const_byte_span_empty(), {3, refs},
            IREE_HAL_DISPATCH_FLAG_NONE));
        IREE_ASSERT_OK(iree_hal_command_buffer_end(command.get()));
        const iree_hal_semaphore_list_t none = {};
        IREE_ASSERT_OK(iree_hal_device_queue_execute(
            device_, IREE_HAL_QUEUE_AFFINITY_ANY, none, none, command.get(),
            iree_hal_buffer_binding_table_empty(), IREE_HAL_EXECUTE_FLAG_NONE));
        IREE_ASSERT_OK(iree_hal_buffer_map_read(
            buffers[2].get(), 0, actual.data(), actual.size() * 2));
        std::vector<uint16_t> expected(actual.size(), 0x55aa);
        for (uint32_t m = 0; m < d.m; ++m)
          for (uint32_t n = 0; n < d.n; ++n) {
            auto index = oi(m, n);
            EXPECT_TRUE(std::isfinite(Float(actual[index])));
            if (!seed) {
              float sum = 0;
              for (uint32_t k = 0; k < d.k; ++k)
                sum += Float(a[ai(m, k)]) * Float(b[bi(n, k)]);
              expected[index] = Bits(sum);
            } else
              expected[index] = actual[index];
          }
        EXPECT_EQ(
            actual,
            expected); // Includes every prefix, padding and suffix sentinel.
        if (!r)
          reference = actual;
        else
          EXPECT_EQ(actual, reference);
      }
      if (seed == 1) {
        if (repeat.empty())
          repeat = reference;
        else
          EXPECT_EQ(repeat, reference);
      } else if (seed == 2) {
        EXPECT_NE(repeat, reference);
      }
    }
  }
  inline static std::array<std::array<std::vector<uint8_t>, 3>, CaseCount>
      wires_;
};

TEST_F(NativeReaderTest,
       ConvolutionPreservesBitsAcrossRowsPagesAndRepeatedInputs) {
  Check(Conv);
}
TEST_F(NativeReaderTest, RowMajorLinearPreservesBitsWithKTailAndPaddedStorage) {
  Check(Linear);
}
TEST_F(NativeReaderTest, StrideTwoConvolutionPreservesChunksAndPadding) {
  Check(StridedConv);
}
TEST_F(NativeReaderTest, StrideTwoProjectionPreservesAllThreeTileTails) {
  Check(StridedProjection);
}
TEST_F(NativeReaderTest, ConvolutionAndViewStridesComposeWithoutOverflow) {
  Check(ComposedConv);
}
TEST_F(NativeReaderTest, SevenBySevenStrideTwoStemPreservesKTail) {
  Check(Stem);
}
TEST_F(NativeReaderTest, OperandReusePreservesBitsAcrossAllSingleCoreShapes) {
  for (unsigned c = 0; c < CaseCount; ++c) {
    SCOPED_TRACE(c);
    Check(static_cast<ReaderCase>(c), true);
  }
}
TEST_F(NativeReaderTest, OperandReuseHandlesOddRectanglesAndUnevenCorePartitions) {
  Check(OddProjection);
}
} // namespace
} // namespace iree::hal::tenstorrent
