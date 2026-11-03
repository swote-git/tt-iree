// Copyright 2026 The tt-iree Authors
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// Hardware characterization of SFPU FP32 arithmetic used by the v22 BF16
// epilogue: FMA, mul, add/sub, statement-level contraction and the tt_div
// helper, compared bit-for-bit with IEEE binary32 round-to-nearest-even, and
// the raw INT32 Dst transport of paired BF16 operands.
// NaN results follow the RISC-V soft-float reference (canonical 0x7fc00000).

#include <gtest/gtest.h>

#include <cmath>
#include <cstdlib>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <random>
#include <string>
#include <vector>

#include <tt-metalium/circular_buffer_constants.h>
#include <tt-metalium/distributed.hpp>
#include <tt-metalium/host_api.hpp>
#include <tt-metalium/tensor_accessor_args.hpp>

#include "TenstorrentNativeKernels.h"

#pragma STDC FP_CONTRACT OFF

namespace {
using namespace tt::tt_metal;

constexpr uint32_t kTiles = 64;
constexpr uint32_t kTileBytes = 4096;
constexpr uint32_t kCount = kTiles * 1024;

float f(uint32_t bits) {
  float v;
  std::memcpy(&v, &bits, 4);
  return v;
}
uint32_t u(float v) {
  uint32_t bits;
  std::memcpy(&bits, &v, 4);
  return bits;
}
bool isNan(uint32_t b) { return (b & 0x7fffffffu) > 0x7f800000u; }
bool isSub(uint32_t b) { return (b & 0x7f800000u) == 0 && (b & 0x7fffffu); }
bool isSpecial(uint32_t b) { return (b & 0x7f800000u) == 0x7f800000u; }

enum Op { kMad, kMul, kAdd, kSub, kSeparate, kDiv, kCopy, kCopyAfterBf16, kCopyAfterTranspose,
          kRawCopy, kHi16, kLo16, kOpCount };
const char *kOpNames[] = {"sfpmad a*b+c", "tt_mul a*b", "add a+c", "sub a-c",
                          "tt_mul+add", "tt_div a/b", "copy c", "copy c (bf16 init)",
                          "copy c (after transpose)", "int32 copy c", "tt_hi16 c",
                          "tt_lo16 c"};
const char *kOpExpr[] = {"a * b + c", "tt_mul(a, b)", "a + c", "a - c",
                         "", "tt_div(a, b)", "c", "c", "c", "", "", ""};
// Integer ops on raw Dst bits: every input pattern must survive, subnormal
// and NaN patterns included (two BF16 epilogue operands share one FP32 tile).
bool isRaw(int op) { return op == kRawCopy || op == kHi16 || op == kLo16; }

uint32_t reference(Op op, uint32_t a, uint32_t b, uint32_t c) {
  volatile float x = f(a), y = f(b), z = f(c), r;
  switch (op) {
  case kMad: r = std::fma(x, y, z); break;
  case kMul: r = x * y; break;
  case kAdd: r = x + z; break;
  case kSub: r = x - z; break;
  case kSeparate: { volatile float t = x * y; r = t + z; break; }
  case kCopy: case kCopyAfterBf16: case kCopyAfterTranspose: case kRawCopy: return c;
  case kHi16: return c & 0xffff0000u;
  case kLo16: return c << 16;
  default: r = x / y; break;
  }
  uint32_t bits = u(r);
  return isNan(bits) ? 0x7fc00000u : bits;
}

std::string computeSource(Op op) {
  std::string body = op == kSeparate
      ? "sfpi::vFloat t = tt_mul(a, b);\n    sfpi::vFloat r = t + c;\n"
      : op == kRawCopy ? "sfpi::vInt r = sfpi::dst_reg[64u + i];\n"
      : op == kHi16 ? "sfpi::vInt r = sfpi::as<sfpi::vInt>(tt_hi16(64u + i));\n"
      : op == kLo16 ? "sfpi::vInt r = sfpi::as<sfpi::vInt>(tt_lo16(64u + i));\n"
      : std::string("sfpi::vFloat r = ") + kOpExpr[op] + ";\n";
  return R"kernel(
#include <cstdint>
#include "api/compute/common.h"
#include "api/compute/tile_move_copy.h"
#include "api/compute/eltwise_binary_sfpu.h"
#include "api/compute/eltwise_unary/eltwise_unary.h"
#include "api/compute/compute_kernel_hw_startup.h"
#include "api/compute/transpose_dest.h"
#ifdef TRISC_MATH
#include "sfpi.h"
namespace ckernel::sfpu {
)kernel" + mlir::iree_compiler::IREE::HAL::tenstorrentSfpuHelperSource() +
         R"kernel(
inline void tt_test_face(const uint32_t, const uint32_t, const uint32_t) {
  for (uint32_t i = 0; i < 8u; ++i) {
    sfpi::vFloat a = sfpi::dst_reg[i], b = sfpi::dst_reg[32u + i], c = sfpi::dst_reg[64u + i];
    )kernel" + body + R"kernel(
    sfpi::dst_reg[96u + i] = r;
  }
}
} // namespace ckernel::sfpu
#endif
void kernel_main() {
  uint32_t n = get_arg_val<uint32_t>(0);
  )kernel" + std::string(op == kCopyAfterBf16 || op == kCopyAfterTranspose
      // As the v22 matmul kernel: hardware set up for BF16 Src operands first.
      ? "compute_kernel_hw_startup<SrcOrder::Reverse>(tt::CBIndex::c_3,tt::CBIndex::c_3,tt::CBIndex::c_16);"
      : "init_sfpu(tt::CBIndex::c_0, tt::CBIndex::c_16);") + R"kernel(
  for (uint32_t t = 0; t < n; ++t) {
    cb_wait_front(tt::CBIndex::c_0, 1); cb_wait_front(tt::CBIndex::c_1, 1); cb_wait_front(tt::CBIndex::c_2, 1);
    tile_regs_acquire();
    )kernel" + std::string(op == kCopyAfterTranspose
        // As v22 channel outputs: 32-bit Dst transpose before the operand copies.
        ? "transpose_dest_init<true,true>(); transpose_dest<true,true>(3);"
          "copy_tile_to_dst_init_short_with_dt(tt::CBIndex::c_3, tt::CBIndex::c_0);"
        : op == kCopyAfterBf16
        ? "copy_tile_to_dst_init_short_with_dt(tt::CBIndex::c_3, tt::CBIndex::c_0);"
        : "copy_tile_to_dst_init_short(tt::CBIndex::c_0);") + R"kernel(
    copy_tile(tt::CBIndex::c_0, 0, 0);
    copy_tile(tt::CBIndex::c_1, 0, 1);
    copy_tile(tt::CBIndex::c_2, 0, 2);
    MATH((SFPU_BINARY_INIT(unused)));
    MATH((SFPU_BINARY_CALL_NO_TEMPLATE_ARGS(DST_SYNC_MODE, DST_ACCUM_MODE,
        tt_test_face, 0, 1, 3, VectorMode::RC)));
    tile_regs_commit(); tile_regs_wait();
    cb_reserve_back(tt::CBIndex::c_16, 1);
    pack_tile(3, tt::CBIndex::c_16);
    cb_push_back(tt::CBIndex::c_16, 1);
    tile_regs_release();
    cb_pop_front(tt::CBIndex::c_0, 1); cb_pop_front(tt::CBIndex::c_1, 1); cb_pop_front(tt::CBIndex::c_2, 1);
  }
}
)kernel";
}

constexpr char kReader[] = R"kernel(
#include <cstdint>
void kernel_main() {
  uint32_t n = get_arg_val<uint32_t>(3);
  constexpr auto a0 = TensorAccessorArgs<0>();
  constexpr auto a1 = TensorAccessorArgs<a0.next_compile_time_args_offset()>();
  constexpr auto a2 = TensorAccessorArgs<a1.next_compile_time_args_offset()>();
  const auto s0 = TensorAccessor(a0, get_arg_val<uint32_t>(0));
  const auto s1 = TensorAccessor(a1, get_arg_val<uint32_t>(1));
  const auto s2 = TensorAccessor(a2, get_arg_val<uint32_t>(2));
  for (uint32_t i = 0; i < n; ++i) {
    cb_reserve_back(tt::CBIndex::c_0, 1); cb_reserve_back(tt::CBIndex::c_1, 1); cb_reserve_back(tt::CBIndex::c_2, 1);
    noc_async_read_page(i, s0, get_write_ptr(tt::CBIndex::c_0));
    noc_async_read_page(i, s1, get_write_ptr(tt::CBIndex::c_1));
    noc_async_read_page(i, s2, get_write_ptr(tt::CBIndex::c_2));
    noc_async_read_barrier();
    cb_push_back(tt::CBIndex::c_0, 1); cb_push_back(tt::CBIndex::c_1, 1); cb_push_back(tt::CBIndex::c_2, 1);
  }
}
)kernel";

constexpr char kWriter[] = R"kernel(
#include <cstdint>
void kernel_main() {
  uint32_t n = get_arg_val<uint32_t>(1);
  constexpr auto args = TensorAccessorArgs<0>();
  const auto out = TensorAccessor(args, get_arg_val<uint32_t>(0));
  for (uint32_t i = 0; i < n; ++i) {
    cb_wait_front(tt::CBIndex::c_16, 1);
    noc_async_write_page(i, out, get_read_ptr(tt::CBIndex::c_16));
    noc_async_write_barrier();
    cb_pop_front(tt::CBIndex::c_16, 1);
  }
}
)kernel";

uint32_t randomFloat(std::mt19937 &rng, int low, int high) {
  uint32_t e = std::uniform_int_distribution<int>(low, high)(rng) + 127;
  return (rng() & 0x80000000u) | (e << 23) | (rng() & 0x7fffffu);
}

void makeInputs(std::vector<uint32_t> &a, std::vector<uint32_t> &b,
                std::vector<uint32_t> &c) {
  std::mt19937 rng(20261104);
  const uint32_t specials[] = {0x00000000u, 0x80000000u, 0x7f800000u, 0xff800000u,
                               0x7fc00000u, 0x7fa00001u, 0xffc12345u, 0x00000001u,
                               0x807fffffu, 0x00400000u, 0x00800000u, 0x80800000u,
                               0x7f7fffffu, 0xff7fffffu, 0x3f800000u, 0x3fffffffu};
  for (uint32_t i = 0; i < kCount; ++i) {
    uint32_t kind = i % 8;
    if (kind < 3) {  // moderate normals
      a[i] = randomFloat(rng, -20, 20);
      b[i] = randomFloat(rng, -20, 20);
      c[i] = randomFloat(rng, -20, 20);
    } else if (kind == 3) {  // full exponent range
      a[i] = randomFloat(rng, -126, 127);
      b[i] = randomFloat(rng, -126, 127);
      c[i] = randomFloat(rng, -126, 127);
    } else if (kind == 4) {  // c = -RN(a*b): the result is the exact FMA residual
      a[i] = randomFloat(rng, -10, 10);
      b[i] = randomFloat(rng, -10, 10);
      volatile float p = f(a[i]) * f(b[i]);
      c[i] = u(-p);
    } else if (kind == 5) {  // near cancellation for add/sub
      a[i] = randomFloat(rng, -10, 10);
      b[i] = randomFloat(rng, -10, 10);
      c[i] = (a[i] ^ 0x80000000u) + (rng() % 5) - 2;
    } else if (kind == 6) {  // subnormal operands / results
      a[i] = (rng() & 0x80000000u) | (rng() & 0x7fffffu);
      b[i] = randomFloat(rng, -126, -100);
      c[i] = (rng() & 1) ? (rng() & 0x80000000u) | (rng() & 0x7fffffu)
                         : randomFloat(rng, -126, -110);
    } else {  // special operands
      a[i] = specials[rng() % 16];
      b[i] = specials[rng() % 16];
      c[i] = specials[rng() % 16];
    }
  }
}

class SfpuFp32Test : public ::testing::Test {
protected:
  static void SetUpTestSuite() {
    mesh_ = distributed::MeshDevice::create_unit_mesh(0);
  }
  static void TearDownTestSuite() {
    if (mesh_) mesh_->close();
    mesh_.reset();
  }
  static std::shared_ptr<distributed::MeshDevice> mesh_;
};
std::shared_ptr<distributed::MeshDevice> SfpuFp32Test::mesh_;

TEST_F(SfpuFp32Test, MatchesIeeeBinary32) {
  auto &cq = mesh_->mesh_command_queue();
  distributed::DeviceLocalBufferConfig local{.page_size = kTileBytes,
                                             .buffer_type = BufferType::DRAM};
  distributed::ReplicatedBufferConfig size{.size = kTileBytes * kTiles};
  std::shared_ptr<distributed::MeshBuffer> in[3], out;
  for (auto &b : in) b = distributed::MeshBuffer::create(size, local, mesh_.get());
  out = distributed::MeshBuffer::create(size, local, mesh_.get());
  std::vector<uint32_t> a(kCount), b(kCount), c(kCount);
  makeInputs(a, b, c);
  distributed::EnqueueWriteMeshBuffer(cq, in[0], a, false);
  distributed::EnqueueWriteMeshBuffer(cq, in[1], b, false);
  distributed::EnqueueWriteMeshBuffer(cq, in[2], c, true);

  for (int op = 0; op < kOpCount; ++op) {
    Program program = CreateProgram();
    CoreCoord core{0, 0};
    for (uint8_t index : {0, 1, 2, 16})
      CreateCircularBuffer(program, core,
                           CircularBufferConfig(2 * kTileBytes,
                                                {{index, tt::DataFormat::Float32}})
                               .set_page_size(index, kTileBytes));
    CreateCircularBuffer(program, core,
                         CircularBufferConfig(2048, {{3, tt::DataFormat::Float16_b}})
                             .set_page_size(3, 2048));
    std::vector<uint32_t> readerArgs, writerArgs;
    for (auto &buffer : in)
      TensorAccessorArgs(*buffer->get_backing_buffer()).append_to(readerArgs);
    TensorAccessorArgs(*out->get_backing_buffer()).append_to(writerArgs);
    auto reader = CreateKernelFromString(
        program, kReader, core,
        DataMovementConfig{.processor = DataMovementProcessor::RISCV_1,
                           .noc = NOC::RISCV_1_default, .compile_args = readerArgs});
    auto writer = CreateKernelFromString(
        program, kWriter, core,
        DataMovementConfig{.processor = DataMovementProcessor::RISCV_0,
                           .noc = NOC::RISCV_0_default, .compile_args = writerArgs});
    ComputeConfig config;
    config.fp32_dest_acc_en = true;
    config.unpack_to_dest_mode.assign(NUM_CIRCULAR_BUFFERS, UnpackToDestMode::Default);
    for (int i = 0; i < 3; ++i)
      config.unpack_to_dest_mode[i] = UnpackToDestMode::UnpackToDestFp32;
    auto compute = CreateKernelFromString(program, computeSource(Op(op)), core, config);
    auto address = [](const auto &buffer) {
      return static_cast<uint32_t>(buffer->address());
    };
    SetRuntimeArgs(program, reader, core,
                   {address(in[0]), address(in[1]), address(in[2]), kTiles});
    SetRuntimeArgs(program, writer, core, {address(out), kTiles});
    SetRuntimeArgs(program, compute, core, {kTiles});
    distributed::MeshWorkload workload;
    workload.add_program(distributed::MeshCoordinateRange(mesh_->shape()),
                         std::move(program));
    distributed::EnqueueMeshWorkload(cq, workload, true);
    std::vector<uint32_t> result;
    distributed::EnqueueReadMeshBuffer(cq, result, out, true);
    ASSERT_EQ(result.size(), kCount);

    // normal: no subnormal/inf/NaN operand and a normal or zero result.
    uint32_t total[3] = {}, wrong[3] = {}, nanPayload = 0, fused = 0;
    uint32_t example[3] = {UINT32_MAX, UINT32_MAX, UINT32_MAX};
    for (uint32_t i = 0; i < kCount; ++i) {
      uint32_t expect = reference(Op(op), a[i], b[i], c[i]);
      bool usesB = op != kAdd && op != kSub;
      bool usesC = op != kMul && op != kDiv;
      // Subnormal anywhere, including the separately rounded product, takes
      // precedence: SFPU float ops flush subnormals.
      bool product = (op == kSeparate || op == kMad) && isSub(reference(kMul, a[i], b[i], 0));
      bool subnormal = isSub(a[i]) || (usesB && isSub(b[i])) || (usesC && isSub(c[i])) ||
                       isSub(expect) || product;
      bool special = isSpecial(a[i]) || (usesB && isSpecial(b[i])) ||
                     (usesC && isSpecial(c[i])) || isSpecial(expect);
      int category = subnormal ? 1 : special ? 2 : 0;
      ++total[category];
      uint32_t actual = result[i];
      if (isNan(expect) && isNan(actual) && actual != expect) ++nanPayload;
      if (actual != expect && category != 1 && std::getenv("SFPU_DETAIL")) {
        int64_t ulps = int64_t(int32_t(actual)) - int64_t(int32_t(expect));
        std::printf("    kind=%u ulps=%lld a=%08x b=%08x c=%08x expect=%08x got=%08x\n",
                    i % 8, (long long)ulps, a[i], b[i], c[i], expect, actual);
      }
      if (actual != expect) {
        if (op == kSeparate && actual == u(std::fma(f(a[i]), f(b[i]), f(c[i]))))
          ++fused;
        if (example[category] == UINT32_MAX) example[category] = i;
        ++wrong[category];
      }
    }
    std::printf("%-12s normal %6u/%6u  subnormal %5u/%5u  special %5u/%5u  "
                "nan-payload %u  matches-fma %u\n",
                kOpNames[op], wrong[0], total[0], wrong[1], total[1], wrong[2],
                total[2], nanPayload, fused);
    const char *names[] = {"normal", "subnormal", "special"};
    for (int k = 0; k < 3; ++k)
      if (example[k] != UINT32_MAX) {
        uint32_t i = example[k];
        std::printf("  first %s mismatch: a=%08x b=%08x c=%08x expect=%08x got=%08x\n",
                    names[k], a[i], b[i], c[i], reference(Op(op), a[i], b[i], c[i]),
                    result[i]);
      }
    if (isRaw(op))
      EXPECT_EQ(wrong[0] + wrong[1] + wrong[2], 0u) << kOpNames[op] << " must keep raw bits";
    // SFPMAD is recorded only: it is not a correctly rounded fused multiply-add.
    else if (op != kMad)
      EXPECT_EQ(wrong[0], 0u) << kOpNames[op] << " must be IEEE RNE on normal values";
  }
}
} // namespace
