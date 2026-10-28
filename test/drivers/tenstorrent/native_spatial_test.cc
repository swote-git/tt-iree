// Copyright 2026 The tt-iree Authors
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include <algorithm>
#include <array>
#include <cstring>
#include <sstream>
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

tt_iree_ttex_entry_point_desc_t Entry(bool pool) {
  tt_iree_ttex_entry_point_desc_t ep = {};
  ep.name = pool ? "bf16_spatial_pool" : "bf16_spatial_copy";
  ep.binding_count = 2;
  ep.workgroup_size[0] = ep.workgroup_size[1] = ep.workgroup_size[2] = 1;
  ep.builtin_program = TT_IREE_TTEX_BUILTIN_PROGRAM_BF16_NATIVE;
  ep.dispatch_kind = TT_IREE_TTEX_DISPATCH_KIND_BF16_NATIVE;
  ep.bf16_native.operation_kind = pool ? TT_IREE_TTEX_DISPATCH_KIND_MAX_POOL
                                       : TT_IREE_TTEX_DISPATCH_KIND_PADDED_COPY;
  auto &input = ep.bf16_native.tensors[0];
  input.binding = 0;
  input.element_type = TT_IREE_TTEX_TENSOR_ELEMENT_TYPE_BF16;
  input.layout = TT_IREE_TTEX_TENSOR_LAYOUT_CHW;
  input.rank = 3;
  input.shape[0] = input.storage_shape[0] = 5;
  input.shape[1] = 7;
  input.shape[2] = 39;
  input.storage_shape[1] = 9;
  input.storage_shape[2] = 43;
  input.offset[1] = 1;
  input.offset[2] = 2;
  input.byte_offset = 2046;
  auto &output = ep.bf16_native.tensors[2];
  output = input;
  output.binding = 1;
  output.byte_offset = 2014;
  if (pool) {
    output.shape[1] = 3;
    output.shape[2] = 19;
    output.storage_shape[1] = 5;
    output.storage_shape[2] = 23;
  }
  return ep;
}

size_t StorageElements(const tt_iree_ttex_tensor_desc_t &t) {
  return t.byte_offset / 2 +
         t.storage_shape[0] * t.storage_shape[1] * t.storage_shape[2] + 32;
}
size_t Index(const tt_iree_ttex_tensor_desc_t &t, uint32_t c, uint32_t h,
             uint32_t w) {
  return t.byte_offset / 2 +
         ((c + t.offset[0]) * t.storage_shape[1] + h + t.offset[1]) *
             t.storage_shape[2] +
         w + t.offset[2];
}
float Float(uint16_t bits) {
  uint32_t word = uint32_t(bits) << 16;
  float value;
  std::memcpy(&value, &word, 4);
  return value;
}
uint16_t Maximum(uint16_t a, uint16_t b) {
  if ((a & 0x7fff) > 0x7f80 || (b & 0x7fff) > 0x7f80)
    return 0x7fc0;
  if (Float(b) > Float(a))
    return b;
  if (!(a & 0x7fff) && !(b & 0x7fff))
    return a & b;
  return a;
}

bool Build(tt_iree_ttex_entry_point_desc_t ep, std::vector<uint8_t> &wire,
           std::string &error, const std::string &source_override = {}) {
  const tt_iree_ttex_tensor_desc_t *tensors[TT_IREE_BF16_NATIVE_TENSOR_COUNT] =
      {};
  tensors[0] = &ep.bf16_native.tensors[0];
  tensors[2] = &ep.bf16_native.tensors[2];
  auto source =
      source_override.empty()
          ? generateBF16NativeSpatial(ep.bf16_native.operation_kind, tensors)
          : source_override;
  TenstorrentAOTKernelBundle bundle;
  if (source.empty() ||
      !compileBF16NativeAOT(ep, source, {}, TenstorrentAOTTargetArch::Blackhole,
                            bundle, error))
    return false;
  std::vector<tt_iree_ttex_kernel_source_desc_t> sources;
  std::vector<tt_iree_ttex_artifact_file_desc_t> artifacts;
  for (auto &kernel : bundle.kernels)
    sources.push_back({kernel.role, kernel.source.c_str()});
  for (auto &artifact : bundle.artifacts)
    artifacts.push_back(
        {artifact.path.c_str(), artifact.data.data(), artifact.data.size()});
  tt_iree_ttex_aot_kernel_bundle_desc_t aot = {
      static_cast<uint8_t>(bundle.targetArch), sources.size(), sources.data(),
      artifacts.size(), artifacts.data()};
  ep.aot_kernel_bundle = &aot;
  iree_byte_span_t data = {};
  auto result = tt_iree_build_ttex_executable_def_with_version(
      iree_allocator_system(), TT_IREE_TTEX_VERSION_17, 1, &ep, &data);
  if (!iree_status_is_ok(result)) {
    iree_status_fprint(stderr, result);
    iree_status_ignore(result);
    error = "failed to serialize spatial TTEX fixture";
    return false;
  }
  wire.assign(data.data, data.data + data.data_length);
  iree_allocator_free(iree_allocator_system(), data.data);
  return true;
}

class NativeSpatialTest : public TenstorrentTestBase {
protected:
  static void SetUpTestSuite() {
    // Offline mock context and a physical MeshDevice cannot coexist.
    std::string error;
    ASSERT_TRUE(Build(Entry(true), wires_[0], error)) << error;
    ASSERT_TRUE(Build(Entry(false), wires_[1], error)) << error;
    TenstorrentTestBase::SetUpTestSuite();
  }
  void Check(bool pool) {
    auto ep = Entry(pool);
    const auto &in = ep.bf16_native.tensors[0];
    const auto &out = ep.bf16_native.tensors[2];
    std::vector<uint16_t> input(StorageElements(in), 0x5aa5);
    const uint16_t bits[] = {0,      0x8000, 0x3f80, 0xbf80,
                             0x7f80, 0xff80, 0x7fc5, 0x7f81};
    for (uint32_t c = 0; c < in.shape[0]; ++c)
      for (uint32_t h = 0; h < in.shape[1]; ++h)
        for (uint32_t w = 0; w < in.shape[2]; ++w) {
          uint16_t value = bits[(h * in.shape[2] + w) % 8];
          if (pool && c == 0)
            value = 0xbf80 + (h + w) % 8;
          if (pool && c == 1)
            value = h < 3 && w < 3 ? 0x8000 : bits[(h + w) % 2];
          if (pool && c == 2)
            value = h == 1 && w == 1 ? 0x7f80 : 0xff80;
          input[Index(in, c, h, w)] = value;
        }
    std::vector<uint16_t> expected(StorageElements(out), 0x55aa);
    for (uint32_t c = 0; c < out.shape[0]; ++c)
      for (uint32_t h = 0; h < out.shape[1]; ++h)
        for (uint32_t w = 0; w < out.shape[2]; ++w) {
          uint16_t value = input[Index(in, c, h, w)];
          if (pool) {
            value = 0xff80;
            for (uint32_t kh = 0; kh < 3; ++kh)
              for (uint32_t kw = 0; kw < 3; ++kw)
                value =
                    Maximum(value, input[Index(in, c, h * 2 + kh, w * 2 + kw)]);
          }
          expected[Index(out, c, h, w)] = value;
        }
    iree_hal_buffer_params_t params = {};
    params.type = IREE_HAL_MEMORY_TYPE_DEVICE_LOCAL;
    params.access = IREE_HAL_MEMORY_ACCESS_ALL;
    params.usage = IREE_HAL_BUFFER_USAGE_DISPATCH_STORAGE |
                   IREE_HAL_BUFFER_USAGE_TRANSFER |
                   IREE_HAL_BUFFER_USAGE_MAPPING;
    iree::vm::ref<iree_hal_buffer_t> input_buffer, output_buffer;
    IREE_ASSERT_OK(iree_hal_allocator_allocate_buffer(
        device_allocator(), params, input.size() * 2, &input_buffer));
    IREE_ASSERT_OK(iree_hal_allocator_allocate_buffer(
        device_allocator(), params, expected.size() * 2, &output_buffer));
    IREE_ASSERT_OK(iree_hal_buffer_map_write(input_buffer.get(), 0,
                                             input.data(), input.size() * 2));
    iree_hal_executable_params_t exec;
    iree_hal_executable_params_initialize(&exec);
    exec.executable_format =
        iree_make_cstring_view(IREE_HAL_TT_EXECUTABLE_FORMAT);
    auto &wire = wires_[pool ? 0 : 1];
    exec.executable_data = iree_make_const_byte_span(wire.data(), wire.size());
    iree::vm::ref<iree_hal_executable_t> executable;
    IREE_ASSERT_OK(iree_hal_tt_executable_create(
        device_, &exec, host_allocator(), &executable));
    for (uint32_t cores : {1, 2, 4, 8, 16, 32, 64}) {
      SCOPED_TRACE(cores);
      std::vector<uint16_t> actual(expected.size(), 0x55aa);
      IREE_ASSERT_OK(iree_hal_buffer_map_write(
          output_buffer.get(), 0, actual.data(), actual.size() * 2));
      iree::vm::ref<iree_hal_command_buffer_t> command;
      IREE_ASSERT_OK(iree_hal_tt_device_create_command_buffer(
          device_, IREE_HAL_COMMAND_BUFFER_MODE_ONE_SHOT,
          IREE_HAL_COMMAND_CATEGORY_DISPATCH, IREE_HAL_QUEUE_AFFINITY_ANY, 0,
          &command));
      IREE_ASSERT_OK(iree_hal_command_buffer_begin(command.get()));
      iree_hal_buffer_ref_t refs[] = {
          iree_hal_make_buffer_ref(input_buffer.get(), 0, input.size() * 2),
          iree_hal_make_buffer_ref(output_buffer.get(), 0, actual.size() * 2)};
      iree_hal_dispatch_config_t grid = {
          {0, 0, 0}, {std::min(cores, 8u), (cores + 7) / 8, 1}};
      IREE_ASSERT_OK(iree_hal_command_buffer_dispatch(
          command.get(), executable.get(), 0, grid,
          iree_const_byte_span_empty(), {2, refs},
          IREE_HAL_DISPATCH_FLAG_NONE));
      IREE_ASSERT_OK(iree_hal_command_buffer_end(command.get()));
      const iree_hal_semaphore_list_t none = {};
      IREE_ASSERT_OK(iree_hal_device_queue_execute(
          device_, IREE_HAL_QUEUE_AFFINITY_ANY, none, none, command.get(),
          iree_hal_buffer_binding_table_empty(), IREE_HAL_EXECUTE_FLAG_NONE));
      IREE_ASSERT_OK(iree_hal_buffer_map_read(
          output_buffer.get(), 0, actual.data(), actual.size() * 2));
      EXPECT_EQ(actual, expected);
    }
  }
  inline static std::array<std::vector<uint8_t>, 2> wires_;
};

TEST_F(NativeSpatialTest, PoolMatchesExactReferenceAtOneThrough64Cores) {
  Check(true);
}
TEST_F(NativeSpatialTest, CopyPreservesBitsAndHaloAtOneThrough64Cores) {
  Check(false);
}

// Opt-in timing fixture: compare the old per-element transfers and new window
// transfers independently of core count. Disabled in routine correctness CI.
class SpatialBenchmarkTest : public TenstorrentTestBase {
protected:
  static tt_iree_ttex_entry_point_desc_t BenchmarkEntry(bool pool, bool bulk) {
    auto ep = Entry(pool);
    ep.name = pool ? (bulk ? "window_pool" : "scalar_pool")
                   : (bulk ? "window_copy" : "scalar_copy");
    auto &in = ep.bf16_native.tensors[0];
    in.shape[0] = in.storage_shape[0] = pool ? 64 : 3;
    in.shape[1] = in.storage_shape[1] = pool ? 114 : 224;
    in.shape[2] = in.storage_shape[2] = pool ? 114 : 224;
    in.byte_offset = in.offset[1] = in.offset[2] = 0;
    auto &out = ep.bf16_native.tensors[2];
    out = in;
    out.binding = 1;
    if (pool) {
      out.shape[1] = out.storage_shape[1] = 56;
      out.shape[2] = out.storage_shape[2] = 56;
    } else {
      out.storage_shape[1] = out.storage_shape[2] = 230;
      out.offset[1] = out.offset[2] = 3;
    }
    return ep;
  }
  static std::string ScalarSource(const tt_iree_ttex_entry_point_desc_t &ep,
                                  bool pool) {
    const tt_iree_ttex_tensor_desc_t *t[TT_IREE_BF16_NATIVE_TENSOR_COUNT] = {};
    t[0] = &ep.bf16_native.tensors[0];
    t[2] = &ep.bf16_native.tensors[2];
    const auto &out = *t[2];
    std::ostringstream s;
    s << generateBF16NativeCommon(t)
      << "void kernel_main(){ CircularBuffer cb(tt::CBIndex::c_2); "
         "cb.reserve_back(1);"
         "uint32_t scratch=cb.get_write_ptr(),start=get_arg_val<uint32_t>(28),"
         "count=get_arg_val<uint32_t>(29); for(uint32_t "
         "i=start;i<start+count;++i){"
      << "uint32_t c=i/" << out.shape[1] * out.shape[2] << "u,h=(i/"
      << out.shape[2] << "u)%" << out.shape[1] << "u,w=i%" << out.shape[2]
      << "u;";
    if (pool)
      s << "float value=from_bits(0xff800000u);for(uint32_t kh=0;kh<3;++kh)"
           "for(uint32_t kw=0;kw<3;++kw){float "
           "v=read_0(scratch,c,h*2+kh,w*2+kw);"
           "if((to_bits(v)&0x7fffffffu)>0x7f800000u || "
           "(to_bits(value)&0x7fffffffu)>0x7f800000u)value=from_bits("
           "0x7fc00000u);"
           "else if(v>value)value=v;else if(v==0.0f && value==0.0f)"
           "value=from_bits(to_bits(v)&to_bits(value));}write_2(scratch,value,"
           "c,h,w);";
    else
      s << "write_2(scratch,read_0(scratch,c,h,w),c,h,w);";
    return s.str() + "}}\n";
  }
  static void SetUpTestSuite() {
    for (unsigned variant = 0; variant < 4; ++variant) {
      bool pool = variant < 2, bulk = variant % 2;
      auto ep = BenchmarkEntry(pool, bulk);
      std::string error;
      ASSERT_TRUE(Build(ep, wires_[variant], error,
                        bulk ? std::string() : ScalarSource(ep, pool)))
          << error;
    }
    TenstorrentTestBase::SetUpTestSuite();
  }
  inline static std::array<std::vector<uint8_t>, 4> wires_;
};

TEST_F(SpatialBenchmarkTest, DISABLED_ResNetShapesScalarVersusWindowCoreSweep) {
  for (unsigned variant = 0; variant < 4; ++variant) {
    auto ep = BenchmarkEntry(variant < 2, variant % 2);
    const auto &in = ep.bf16_native.tensors[0];
    const auto &out = ep.bf16_native.tensors[2];
    std::vector<uint16_t> input(StorageElements(in), 0x3f80);
    std::vector<uint16_t> expected(StorageElements(out), 0x55aa);
    for (uint32_t c = 0; c < out.shape[0]; ++c)
      for (uint32_t h = 0; h < out.shape[1]; ++h)
        for (uint32_t w = 0; w < out.shape[2]; ++w)
          expected[Index(out, c, h, w)] = 0x3f80;
    iree_hal_buffer_params_t params = {};
    params.type = IREE_HAL_MEMORY_TYPE_DEVICE_LOCAL;
    params.access = IREE_HAL_MEMORY_ACCESS_ALL;
    params.usage = IREE_HAL_BUFFER_USAGE_DEFAULT | IREE_HAL_BUFFER_USAGE_MAPPING;
    iree::vm::ref<iree_hal_buffer_t> input_buffer, output_buffer;
    IREE_ASSERT_OK(iree_hal_allocator_allocate_buffer(
        device_allocator(), params, input.size() * 2, &input_buffer));
    IREE_ASSERT_OK(iree_hal_allocator_allocate_buffer(
        device_allocator(), params, expected.size() * 2, &output_buffer));
    IREE_ASSERT_OK(iree_hal_buffer_map_write(input_buffer.get(), 0,
                                             input.data(), input.size() * 2));
    iree_hal_executable_params_t exec;
    iree_hal_executable_params_initialize(&exec);
    exec.executable_format =
        iree_make_cstring_view(IREE_HAL_TT_EXECUTABLE_FORMAT);
    exec.executable_data = iree_make_const_byte_span(wires_[variant].data(),
                                                     wires_[variant].size());
    iree::vm::ref<iree_hal_executable_t> executable;
    IREE_ASSERT_OK(iree_hal_tt_executable_create(
        device_, &exec, host_allocator(), &executable));
    for (uint32_t cores : {1, 2, 4, 8, 16, 32, 64}) {
      SCOPED_TRACE(ep.name);
      SCOPED_TRACE(cores);
      std::vector<uint16_t> actual(expected.size(), 0x55aa);
      IREE_ASSERT_OK(iree_hal_buffer_map_write(
          output_buffer.get(), 0, actual.data(), actual.size() * 2));
      for (unsigned iteration = 0; iteration < 12; ++iteration) {
        iree::vm::ref<iree_hal_command_buffer_t> command;
        IREE_ASSERT_OK(iree_hal_tt_device_create_command_buffer(
            device_, IREE_HAL_COMMAND_BUFFER_MODE_ONE_SHOT,
            IREE_HAL_COMMAND_CATEGORY_DISPATCH, IREE_HAL_QUEUE_AFFINITY_ANY, 0,
            &command));
        IREE_ASSERT_OK(iree_hal_command_buffer_begin(command.get()));
        iree_hal_buffer_ref_t refs[] = {
            iree_hal_make_buffer_ref(input_buffer.get(), 0, input.size() * 2),
            iree_hal_make_buffer_ref(output_buffer.get(), 0,
                                     actual.size() * 2)};
        iree_hal_dispatch_config_t grid = {
            {0, 0, 0}, {std::min(cores, 8u), (cores + 7) / 8, 1}};
        IREE_ASSERT_OK(iree_hal_command_buffer_dispatch(
            command.get(), executable.get(), 0, grid,
            iree_const_byte_span_empty(), {2, refs},
            IREE_HAL_DISPATCH_FLAG_NONE));
        IREE_ASSERT_OK(iree_hal_command_buffer_end(command.get()));
        const iree_hal_semaphore_list_t none = {};
        IREE_ASSERT_OK(iree_hal_device_queue_execute(
            device_, IREE_HAL_QUEUE_AFFINITY_ANY, none, none, command.get(),
            iree_hal_buffer_binding_table_empty(), IREE_HAL_EXECUTE_FLAG_NONE));
      }
      IREE_ASSERT_OK(iree_hal_buffer_map_read(
          output_buffer.get(), 0, actual.data(), actual.size() * 2));
      EXPECT_EQ(actual, expected);
    }
  }
}
} // namespace
} // namespace iree::hal::tenstorrent
