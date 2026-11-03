// Copyright 2026 The tt-iree Authors
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include <algorithm>
#include <bit>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

#include "tt-metalium/distributed.hpp"
#include "tt-metalium/hal.hpp"
#include "tt-metalium/host_api.hpp"
#include "tt-metalium/tensor_accessor_args.hpp"
#include "tt-metalium/tt_metal.hpp"

namespace metal = tt::tt_metal;
namespace distributed = tt::tt_metal::distributed;
namespace fs = std::filesystem;

namespace {
// These kernels are test-only JIT sources, not a production runtime fallback.
constexpr char kReader[] = R"cpp(
#include "api/dataflow/dataflow_api.h"
void kernel_main() {
  constexpr auto a_args = TensorAccessorArgs<0>();
  constexpr auto b_args = TensorAccessorArgs<a_args.next_compile_time_args_offset()>();
  const auto a = TensorAccessor(a_args, get_arg_val<uint32_t>(0), get_tile_size(0));
  const auto b = TensorAccessor(b_args, get_arg_val<uint32_t>(1), get_tile_size(1));
  for (uint32_t i = 0; i < get_arg_val<uint32_t>(2); ++i) {
    cb_reserve_back(0, 1);
    noc_async_read(a.get_noc_addr(i), get_write_ptr(0), get_tile_size(0));
#ifndef BYPASS
    cb_reserve_back(1, 1);
    noc_async_read(b.get_noc_addr(i), get_write_ptr(1), get_tile_size(1));
#endif
    noc_async_read_barrier();
    cb_push_back(0, 1);
#ifndef BYPASS
    cb_push_back(1, 1);
#endif
  }
}
)cpp";

constexpr char kWriter[] = R"cpp(
#include "api/dataflow/dataflow_api.h"
void kernel_main() {
  constexpr auto args = TensorAccessorArgs<0>();
  const auto out = TensorAccessor(args, get_arg_val<uint32_t>(0), get_tile_size(OUTPUT_CB));
  for (uint32_t i = 0; i < get_arg_val<uint32_t>(1); ++i) {
    cb_wait_front(OUTPUT_CB, 1);
    noc_async_write(get_read_ptr(OUTPUT_CB), out.get_noc_addr(i), get_tile_size(OUTPUT_CB));
    noc_async_write_barrier();
    cb_pop_front(OUTPUT_CB, 1);
  }
}
)cpp";

constexpr char kCompute[] = R"cpp(
#include "api/compute/compute_kernel_hw_startup.h"
#include "api/compute/compute_kernel_api.h"
#include "api/compute/matmul.h"
#include "api/compute/tile_move_copy.h"
#include "api/compute/eltwise_unary/fill.h"
#ifdef TRISC_MATH
namespace ckernel::sfpu {
template <bool APPROX, int ITERATIONS>
inline void _calculate_probe_dst_compare_(uint32_t bits) {
  for (int i = 0; i < ITERATIONS; ++i) {
    sfpi::vUInt value = sfpi::as<sfpi::vUInt>(sfpi::vFloat(sfpi::dst_reg[0]));
    sfpi::vFloat flag = 0.0f;
    v_if(value == bits) { flag = 1.0f; } v_endif;
    sfpi::dst_reg[0] = flag;
    sfpi::dst_reg++;
  }
}
}
#endif
void kernel_main() {
  constexpr uint32_t op = get_compile_time_arg_val(0);
  constexpr bool check_dst = get_compile_time_arg_val(4) != 0;
  constexpr bool prefill_dst = get_compile_time_arg_val(6) != 0;
  if constexpr (op == 0) {
    compute_kernel_hw_startup<SrcOrder::Reverse>(0, 1, 16);
    if constexpr (check_dst || prefill_dst) fill_tile_init();
    matmul_init(0, 1);
  } else {
    compute_kernel_hw_startup<SrcOrder::Regular>(0, 1, 16);
    if constexpr (op == 1) copy_tile_init(0);
    else fill_tile_init();
  }
  for (uint32_t i = 0; i < get_compile_time_arg_val(1); ++i) {
    cb_reserve_back(16, 1);
    tile_regs_acquire();
    if constexpr (prefill_dst) fill_tile_bitcast(0, get_compile_time_arg_val(7));
    for (uint32_t slice = 0; slice < get_compile_time_arg_val(3); ++slice) {
      cb_wait_front(0, 1);
      cb_wait_front(1, 1);
      if constexpr (op == 0) matmul_tiles(0, 1, 0, 0, 0);
      else if constexpr (op == 1) copy_tile(0, 0, 0);
      else fill_tile_bitcast(0, get_compile_time_arg_val(2));
      cb_pop_front(0, 1);
      cb_pop_front(1, 1);
    }
    if constexpr (check_dst) {
      MATH(SFPU_UNARY_CALL(DST_SYNC_MODE, DST_ACCUM_MODE,
          _calculate_probe_dst_compare_, (false, 8), 0, VectorMode::RC,
          get_compile_time_arg_val(5)));
    }
    tile_regs_commit();
    tile_regs_wait();
    pack_tile(0, 16);
    tile_regs_release();
    cb_push_back(16, 1);
  }
}
)cpp";

struct Options {
  fs::path output;
  std::string format = "bf16";
  std::string op = "matmul";
  std::string pattern = "integer";
  std::string rhs_component = "full";
  fs::path lhs_file;
  fs::path rhs_file;
  uint32_t repeats = 4;
  uint32_t fidelity = 4;
  uint32_t fill_bits = 0x3f812345;
  uint32_t k = 32;
  uint32_t k_slice_width = 32;
  int magnitude = 127;
  bool unpack_to_dst = false;
  bool require_exact = false;
  bool check_dst = false;
  uint32_t dst_check_bits = 0;
  bool prefill_dst = false;
  uint32_t dst_prefill_bits = 0;
};

Options Parse(int argc, char **argv) {
  Options o;
  for (int i = 1; i < argc; ++i) {
    std::string key = argv[i];
    if (key == "--require-exact") {
      o.require_exact = true;
      continue;
    }
    if (key == "--f32-unpack-to-dst") {
      o.unpack_to_dst = true;
      continue;
    }
    if (key == "--help") {
      std::cout << "--output-dir PATH [--format bf16|f32|int8] "
                   "[--op matmul|copy|fill|bypass] "
                   "[--case "
                   "integer|snapshot|basis|product|constant-product|wave|two-"
                   "term|custom] "
                   "[--lhs-file RAW_F32 --rhs-file RAW_F32] [--k-slice-width "
                   "1|2|4|8|16|32] "
                   "[--rhs-component full|high5|low5] "
                   "[--fidelity 0|2|3|4] [--magnitude 1..127] [--k 1..32] "
                   "[--repeats 1..64] [--fill-bits HEX] "
                   "[--f32-unpack-to-dst] [--dst-check-bits HEX] "
                   "[--dst-prefill-bits HEX] [--require-exact]\n";
      std::exit(0);
    }
    if (++i == argc)
      throw std::runtime_error("Missing value for " + key);
    std::string value = argv[i];
    if (key == "--output-dir")
      o.output = value;
    else if (key == "--format")
      o.format = value;
    else if (key == "--op")
      o.op = value;
    else if (key == "--case")
      o.pattern = value;
    else if (key == "--rhs-component")
      o.rhs_component = value;
    else if (key == "--lhs-file")
      o.lhs_file = value;
    else if (key == "--rhs-file")
      o.rhs_file = value;
    else if (key == "--fidelity")
      o.fidelity = std::stoul(value);
    else if (key == "--magnitude")
      o.magnitude = std::stoi(value);
    else if (key == "--k")
      o.k = std::stoul(value);
    else if (key == "--k-slice-width")
      o.k_slice_width = std::stoul(value);
    else if (key == "--repeats")
      o.repeats = std::stoul(value);
    else if (key == "--fill-bits")
      o.fill_bits = std::stoul(value, nullptr, 16);
    else if (key == "--dst-check-bits") {
      o.check_dst = true;
      o.dst_check_bits = std::stoul(value, nullptr, 16);
    } else if (key == "--dst-prefill-bits") {
      o.prefill_dst = true;
      o.dst_prefill_bits = std::stoul(value, nullptr, 16);
    } else
      throw std::runtime_error("Unknown option " + key);
  }
  auto member = [](const std::string &s,
                   std::initializer_list<const char *> list) {
    return std::any_of(list.begin(), list.end(),
                       [&](const char *v) { return s == v; });
  };
  if (o.output.empty() || !member(o.format, {"bf16", "f32", "int8"}) ||
      !member(o.op, {"matmul", "copy", "fill", "bypass"}) ||
      !member(o.pattern, {"integer", "snapshot", "basis", "product",
                          "constant-product", "wave", "two-term", "custom"}) ||
      ((o.pattern == "custom") !=
       (!o.lhs_file.empty() && !o.rhs_file.empty())) ||
      (o.lhs_file.empty() != o.rhs_file.empty()) || !o.k_slice_width ||
      o.k_slice_width > 32 || 32 % o.k_slice_width ||
      (o.k_slice_width != 32 && o.op != "matmul") ||
      !member(o.rhs_component, {"full", "high5", "low5"}) ||
      (o.rhs_component != "full" && (o.format == "int8" || o.op != "matmul")) ||
      (o.fidelity != 0 && o.fidelity != 2 && o.fidelity != 3 &&
       o.fidelity != 4) ||
      o.magnitude < 1 || o.magnitude > 127 || !o.k || o.k > 32 || !o.repeats ||
      o.repeats > 64 || (o.op != "matmul" && o.format == "int8") ||
      (o.op == "bypass" && o.format != "f32") ||
      (o.unpack_to_dst && (o.op != "copy" || o.format != "f32")) ||
      (o.check_dst &&
       (o.format == "int8" || o.op == "bypass" ||
        !std::isfinite(std::bit_cast<float>(o.dst_check_bits)))) ||
      (o.prefill_dst &&
       (o.format == "int8" || o.op != "matmul" ||
        !std::isfinite(std::bit_cast<float>(o.dst_prefill_bits)))) ||
      !std::isfinite(std::bit_cast<float>(o.fill_bits))) {
    throw std::runtime_error("Invalid options; see --help");
  }
  return o;
}

uint32_t FaceIndex(uint32_t r, uint32_t c) {
  return (r / 16) * 512 + (c / 16) * 256 + (r % 16) * 16 + c % 16;
}

uint16_t BF16(float value) {
  uint32_t bits = std::bit_cast<uint32_t>(value);
  return (bits + 0x7fff + ((bits >> 16) & 1)) >> 16;
}

void Quantize(std::vector<float> &v, const Options &o) {
  for (float &x : v) {
    if (o.format == "bf16")
      x = std::bit_cast<float>(uint32_t(BF16(x)) << 16);
    else if (o.format == "int8")
      x = std::clamp(std::round(x), -127.0f, 127.0f);
  }
}

std::vector<uint32_t> Pack(const std::vector<float> &v, const Options &o,
                           bool lhs) {
  uint32_t bytes = o.format == "bf16" ? 2 : o.format == "int8" ? 1 : 4;
  uint32_t slices = 32 / o.k_slice_width;
  std::vector<uint32_t> words(1024 * bytes / 4 * o.repeats * slices);
  for (uint32_t r = 0; r < 32; ++r)
    for (uint32_t c = 0; c < 32; ++c) {
      float x = v[r * 32 + c];
      uint32_t bits = std::bit_cast<uint32_t>(x);
      if (bytes == 2)
        bits = BF16(x);
      // Tensix integer formats are sign-magnitude, not host two's complement.
      if (bytes == 1)
        bits = uint32_t(std::abs(int(x))) | (x < 0 ? 0x80u : 0u);
      for (uint32_t i = 0; i < o.repeats; ++i) {
        for (uint32_t s = 0; s < slices; ++s) {
          uint32_t offset = ((i * slices + s) * 1024 + FaceIndex(r, c)) * bytes;
          uint32_t k = lhs ? c : r;
          if (k / o.k_slice_width == s)
            words[offset / 4] |= bits << ((offset % 4) * 8);
        }
      }
    }
  return words;
}

template <typename T>
void Save(const fs::path &path, const std::vector<T> &values) {
  std::ofstream out(path, std::ios::binary);
  out.write(reinterpret_cast<const char *>(values.data()),
            values.size() * sizeof(T));
  if (!out)
    throw std::runtime_error("Could not write " + path.string());
}

void SaveText(const fs::path &path, const std::string &value) {
  std::ofstream out(path);
  out << value;
  if (!out)
    throw std::runtime_error("Could not write " + path.string());
}

std::vector<float> ReadTile(const fs::path &path) {
  if (fs::file_size(path) != 4096)
    throw std::runtime_error("Expected one 32x32 raw F32 tile: " +
                             path.string());
  std::vector<float> values(1024);
  std::ifstream in(path, std::ios::binary);
  in.read(reinterpret_cast<char *>(values.data()), 4096);
  if (!in || !std::all_of(values.begin(), values.end(),
                          [](float x) { return std::isfinite(x); }))
    throw std::runtime_error("Invalid raw F32 tile: " + path.string());
  return values;
}

bool Run(const Options &o,
         const std::shared_ptr<distributed::MeshDevice> &mesh) {
  std::vector<float> a(1024), b(1024), expected(1024);
  int range = o.magnitude * 2 + 1;
  for (uint32_t r = 0; r < 32; ++r)
    for (uint32_t c = 0; c < 32; ++c) {
      uint32_t i = r * 32 + c;
      a[i] = int((i * 37) % range) - o.magnitude;
      b[i] = int((i * 59) % range) - o.magnitude;
      if (o.pattern == "snapshot") {
        a[i] = int(((c * 49 + r) * 37) % 255) - 127;
        b[i] = int(((c * 64 + r) * 59) % 255) - 127;
      } else if (o.pattern == "wave") {
        a[i] = std::sin(double(i) * 0.13) * 0.73 + 0.14;
        b[i] = std::cos(double(i) * 0.07) * 0.021 + 0.003;
      } else if (o.pattern == "basis" || o.pattern == "product" ||
                 o.pattern == "constant-product") {
        a[i] = r == c ? (o.pattern == "basis" ? 1 : o.magnitude) : 0;
        if (o.pattern == "constant-product")
          b[i] = o.magnitude;
      } else if (o.pattern == "two-term") {
        a[i] = c == 0 ? -127 : c == 1 ? -90 : 0;
        b[i] = r == 0 ? -9 : r == 1 ? 94 : 0;
      }
      if (c >= o.k)
        a[i] = 0;
      if (r >= o.k)
        b[i] = 0;
    }
  if (o.pattern == "custom") {
    a = ReadTile(o.lhs_file);
    b = ReadTile(o.rhs_file);
    for (uint32_t r = 0; r < 32; ++r)
      for (uint32_t c = o.k; c < 32; ++c) {
        a[r * 32 + c] = 0;
        b[c * 32 + r] = 0;
      }
  }
  Quantize(a, o);
  Quantize(b, o);
  if (o.rhs_component != "full")
    for (float &x : b) {
      float high =
          std::bit_cast<float>(std::bit_cast<uint32_t>(x) & 0xfff80000u);
      x = o.rhs_component == "high5" ? high : x - high;
    }
  for (uint32_t r = 0; r < 32; ++r)
    for (uint32_t c = 0; c < 32; ++c) {
      uint32_t i = r * 32 + c;
      double sum = o.prefill_dst ? std::bit_cast<float>(o.dst_prefill_bits) : 0;
      for (uint32_t k = 0; k < 32; ++k)
        sum += double(a[r * 32 + k]) * b[k * 32 + c];
      expected[i] = o.op == "matmul" ? float(sum)
                    : o.op == "fill" ? std::bit_cast<float>(o.fill_bits)
                                     : a[i];
    }
  auto packed_a = Pack(a, o, true), packed_b = Pack(b, o, false);
  if (o.check_dst) {
    Save(o.output / "math-expected.f32", expected);
    for (float &value : expected)
      value = std::bit_cast<uint32_t>(value) == o.dst_check_bits ? 1.0f : 0.0f;
  }
  Save(o.output / "lhs.f32", a);
  Save(o.output / "rhs.f32", b);
  Save(o.output / "expected.f32", expected);
  Save(o.output / "lhs-packed.bin", packed_a);
  Save(o.output / "rhs-packed.bin", packed_b);
  SaveText(o.output / "reader.cpp", kReader);
  SaveText(o.output / "writer.cpp", kWriter);
  SaveText(o.output / "compute.cpp", kCompute);

  auto *device = mesh->get_devices().at(0);
  uint32_t slices = 32 / o.k_slice_width;
  uint32_t page = packed_a.size() * 4 / (o.repeats * slices);
  auto buffer = [&](uint32_t page_size, uint32_t tiles) {
    return metal::CreateBuffer(
        metal::InterleavedBufferConfig{.device = device,
                                       .size = uint64_t(page_size) * tiles,
                                       .page_size = page_size,
                                       .buffer_type = metal::BufferType::DRAM});
  };
  auto lhs = buffer(page, o.repeats * slices);
  auto rhs = buffer(page, o.repeats * slices);
  auto out = buffer(4096, o.repeats);
  metal::detail::WriteToBuffer(*lhs, packed_a);
  metal::detail::WriteToBuffer(*rhs, packed_b);
  auto program = metal::CreateProgram();
  metal::CoreCoord core{0, 0};
  auto cb = [&](uint32_t id, tt::DataFormat format, uint32_t bytes) {
    metal::CreateCircularBuffer(
        program, core,
        metal::CircularBufferConfig(2 * bytes, {{id, format}})
            .set_page_size(id, bytes));
  };
  tt::DataFormat input_format = o.format == "bf16"   ? tt::DataFormat::Float16_b
                                : o.format == "int8" ? tt::DataFormat::Int8
                                                     : tt::DataFormat::Float32;
  cb(0, input_format, page);
  cb(1, input_format, page);
  cb(16, o.format == "int8" ? tt::DataFormat::Int32 : tt::DataFormat::Float32,
     4096);
  auto append_accessor = [](std::vector<uint32_t> &args, uint32_t size) {
    metal::TensorAccessorArgs::create_dram_interleaved().append_to(args);
    args.back() = size;
  };
  std::vector<uint32_t> reader_args, writer_args;
  append_accessor(reader_args, page);
  append_accessor(reader_args, page);
  append_accessor(writer_args, 4096);
  metal::DataMovementConfig reader_config{
      .processor = metal::DataMovementProcessor::RISCV_1,
      .noc = metal::NOC::RISCV_1_default,
      .compile_args = reader_args};
  if (o.op == "bypass")
    reader_config.defines["BYPASS"] = "1";
  auto reader =
      metal::CreateKernelFromString(program, kReader, core, reader_config);
  auto writer = metal::CreateKernelFromString(
      program, kWriter, core,
      metal::DataMovementConfig{
          .processor = metal::DataMovementProcessor::RISCV_0,
          .noc = metal::NOC::RISCV_0_default,
          .compile_args = writer_args,
          .defines = {{"OUTPUT_CB", o.op == "bypass" ? "0" : "16"}}});
  if (o.op != "bypass") {
    metal::ComputeConfig config{
        .math_fidelity = static_cast<metal::MathFidelity>(o.fidelity),
        .fp32_dest_acc_en = true,
        .math_approx_mode = false,
        .compile_args = {o.op == "matmul" ? 0u
                         : o.op == "copy" ? 1u
                                          : 2u,
                         o.repeats, o.fill_bits, slices, o.check_dst ? 1u : 0u,
                         o.dst_check_bits, o.prefill_dst ? 1u : 0u,
                         o.dst_prefill_bits}};
    if (o.unpack_to_dst) {
      config.unpack_to_dest_mode.assign(
          metal::hal::get_arch_num_circular_buffers(),
          metal::UnpackToDestMode::Default);
      config.unpack_to_dest_mode[0] = metal::UnpackToDestMode::UnpackToDestFp32;
    }
    metal::CreateKernelFromString(program, kCompute, core, config);
  }
  metal::SetRuntimeArgs(program, reader, core,
                        {lhs->address(), rhs->address(), o.repeats * slices});
  metal::SetRuntimeArgs(program, writer, core, {out->address(), o.repeats});
  distributed::MeshWorkload workload;
  workload.add_program(distributed::MeshCoordinateRange(mesh->shape()),
                       std::move(program));
  distributed::EnqueueMeshWorkload(mesh->mesh_command_queue(), workload, true);
  std::vector<uint32_t> words;
  metal::detail::ReadFromBuffer(*out, words);
  if (words.size() != 1024 * o.repeats)
    throw std::runtime_error("Output size mismatch");
  Save(o.output / "output-packed.bin", words);
  std::vector<float> actual(words.size());
  uint32_t mismatches = 0, outside = 0, nonfinite = 0, unstable = 0;
  double max_error = 0, total_error = 0, squares = 0;
  for (uint32_t t = 0; t < o.repeats; ++t)
    for (uint32_t r = 0; r < 32; ++r)
      for (uint32_t c = 0; c < 32; ++c) {
        uint32_t i = r * 32 + c, offset = t * 1024 + FaceIndex(r, c);
        uint32_t bits = words[offset];
        float value = o.format == "int8" ? float(bits & 0x7fffffffu) *
                                               (bits & 0x80000000u ? -1 : 1)
                                         : std::bit_cast<float>(bits);
        actual[t * 1024 + i] = value;
        if (!std::isfinite(value)) {
          ++nonfinite;
          ++outside;
        } else {
          double error = std::abs(double(value) - expected[i]);
          max_error = std::max(max_error, error);
          total_error += error;
          squares += error * error;
          outside += error > 1e-5 + 1e-5 * std::abs(double(expected[i]));
        }
        mismatches += std::bit_cast<uint32_t>(value) !=
                      std::bit_cast<uint32_t>(expected[i]);
        if (t)
          unstable += bits != words[FaceIndex(r, c)];
      }
  Save(o.output / "actual.f32", actual);
  std::ofstream json(o.output / "summary.json");
  json << std::setprecision(17) << "{\n  \"format\": \"" << o.format
       << "\",\n  \"operation\": \"" << o.op << "\",\n  \"case\": \""
       << o.pattern << "\",\n  \"math_fidelity\": " << o.fidelity
       << ",\n  \"rhs_component\": \"" << o.rhs_component << "\""
       << ",\n  \"architecture\": \"" << metal::hal::get_arch_name() << "\""
       << ",\n  \"core\": [0, 0],\n  \"tile_shape\": [32, 32],\n  "
          "\"compute_kernel_present\": "
       << (o.op == "bypass" ? "false" : "true") << ",\n  \"output_format\": \""
       << (o.format == "int8" ? "Int32" : "Float32") << "\""
       << ",\n  \"fp32_dest_acc_en\": " << (o.op == "bypass" ? "null" : "true")
       << ",\n  \"math_approx_mode\": false,\n  \"f32_unpack_to_dst\": "
       << (o.unpack_to_dst ? "true" : "false")
       << ",\n  \"magnitude\": " << o.magnitude << ",\n  \"logical_k\": " << o.k
       << ",\n  \"k_slice_width\": " << o.k_slice_width
       << ",\n  \"input_tiles_per_output\": " << slices
       << ",\n  \"dst_check_enabled\": " << (o.check_dst ? "true" : "false")
       << ",\n  \"dst_check_bits\": " << o.dst_check_bits
       << ",\n  \"dst_prefill_enabled\": " << (o.prefill_dst ? "true" : "false")
       << ",\n  \"dst_prefill_bits\": " << o.dst_prefill_bits
       << ",\n  \"lhs_file\": " << std::quoted(o.lhs_file.string())
       << ",\n  \"rhs_file\": " << std::quoted(o.rhs_file.string())
       << ",\n  \"fill_bits\": " << o.fill_bits
       << ",\n  \"repeats\": " << o.repeats
       << ",\n  \"elements\": " << actual.size()
       << ",\n  \"bitwise_mismatches\": " << mismatches
       << ",\n  \"outside_atol_rtol_1e_5\": " << outside
       << ",\n  \"nonfinite\": " << nonfinite
       << ",\n  \"different_from_first_repeat\": " << unstable
       << ",\n  \"max_abs_error_finite\": " << max_error
       << ",\n  \"mean_abs_error_finite_over_all_elements\": "
       << total_error / actual.size()
       << ",\n  \"rmse_finite_over_all_elements\": "
       << std::sqrt(squares / actual.size())
       << ",\n  \"reference\": \"stored quantized inputs, double dot then "
          "float32 (including initial Dst when prefilled); non-matmul uses "
          "exact source/fill; dst check compares "
          "reference bits and outputs flags instead of math values\"\n}\n";
  if (!json)
    throw std::runtime_error("Could not write summary");
  std::cout << "Recorded " << o.format << " " << o.op << " " << o.pattern
            << ": max_abs=" << max_error
            << ", bitwise_mismatches=" << mismatches
            << ", outside_1e-5=" << outside << ", unstable=" << unstable
            << '\n';
  return mismatches == 0 && unstable == 0 && nonfinite == 0;
}
} // namespace

int main(int argc, char **argv) {
  try {
    Options options = Parse(argc, argv);
    if (fs::exists(options.output))
      throw std::runtime_error("Output directory already exists");
    fs::create_directories(options.output);
    auto mesh = distributed::MeshDevice::create_unit_mesh(0);
    bool exact = Run(options, mesh);
    mesh->close();
    return options.require_exact && !exact ? 2 : 0;
  } catch (const std::exception &e) {
    std::cerr << "Precision probe failed: " << e.what() << '\n';
    return 1;
  }
}
