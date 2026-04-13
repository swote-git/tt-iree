// Copyright 2025 The tt-iree Authors
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// End-to-end dispatch test: bfloat16 tile addition on P100A hardware.

#include <cmath>
#include <vector>

#include "iree/hal/drivers/tenstorrent/tt_buffer.h"
#include "iree/hal/drivers/tenstorrent/tt_command_buffer.h"
#include "iree/hal/drivers/tenstorrent/tt_executable.h"
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

static iree_status_t CreateBuiltinTtexExecutable(
    iree_hal_device_t* device, iree_allocator_t host_allocator,
    uint32_t builtin_program, iree_hal_executable_t** out_executable) {
  tt_iree_ttex_entry_point_desc_t ep = {};
  ep.name = "builtin_dispatch_0";
  ep.constant_count = 0;
  ep.binding_count = 3;
  ep.flags = 0;
  ep.workgroup_size[0] = 1;
  ep.workgroup_size[1] = 1;
  ep.workgroup_size[2] = 1;
  ep.builtin_program = builtin_program;

  iree_byte_span_t data = {0};
  IREE_RETURN_IF_ERROR(
      tt_iree_build_ttex_executable_def(host_allocator, 1, &ep, &data));

  iree_hal_executable_params_t exec_params;
  iree_hal_executable_params_initialize(&exec_params);
  exec_params.executable_format =
      iree_make_cstring_view("tenstorrent-ttex-fb");
  exec_params.executable_data =
      iree_make_const_byte_span(data.data, data.data_length);

  iree_status_t status = iree_hal_tt_executable_create(
      device, &exec_params, host_allocator, out_executable);
  iree_allocator_free(host_allocator, data.data);
  return status;
}

TEST_F(DispatchTest, ElementwiseAddBfloat16) {
  // 1. Create executable (compiles reader/compute/writer kernels).
  iree_hal_executable_params_t exec_params;
  iree_hal_executable_params_initialize(&exec_params);
  exec_params.executable_format = iree_make_cstring_view("TT-METAL");
  exec_params.executable_data = iree_make_const_byte_span(NULL, 0);

  iree::vm::ref<iree_hal_executable_t> executable;
  iree_status_t exec_status = iree_hal_tt_executable_create(
      device_, &exec_params, host_allocator(), &executable);
  if (!iree_status_is_ok(exec_status)) {
    iree_status_ignore(exec_status);
    GTEST_SKIP() << "kernel compilation not available (mock mode or missing "
                    "kernel sources)";
  }

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
  tt::tt_metal::Buffer* tt_in0 = iree_hal_tt_buffer_handle(in0_buffer.get());
  tt::tt_metal::Buffer* tt_in1 = iree_hal_tt_buffer_handle(in1_buffer.get());
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
  tt::tt_metal::Buffer* tt_out = iree_hal_tt_buffer_handle(out_buffer.get());
  ASSERT_NE(tt_out, nullptr);
  tt::tt_metal::detail::ReadFromBuffer(*tt_out, host_out);

  // 8. Verify element-wise addition: out[i] == in0[i] + in1[i].
  constexpr float kEps = 1e-1f;  // bfloat16 tolerance
  for (uint32_t i = 0; i < kElementsPerTile * kNTiles; i++) {
    float expected =
        static_cast<float>(host_in0[i]) + static_cast<float>(host_in1[i]);
    float actual = static_cast<float>(host_out[i]);
    ASSERT_NEAR(actual, expected, kEps)
        << "mismatch at index " << i << ": expected " << expected << ", got "
        << actual;
  }

  // Log a few sample values for diagnostics.
  fprintf(stderr,
          "  Sample: %.1f + %.1f = %.1f, %.1f + %.1f = %.1f\n",
          static_cast<float>(host_in0[0]), static_cast<float>(host_in1[0]),
          static_cast<float>(host_out[0]), static_cast<float>(host_in0[10]),
          static_cast<float>(host_in1[10]),
          static_cast<float>(host_out[10]));
#endif  // !TT_IREE_ENABLE_MOCK
  // All resources (executable, buffers, cmd_buffer) auto-released by vm::ref.
}

TEST_F(DispatchTest, ReDispatchSameExecutable) {
  // Verifies that a program survives dispatch and can be re-used.
  iree_hal_executable_params_t exec_params;
  iree_hal_executable_params_initialize(&exec_params);
  exec_params.executable_format = iree_make_cstring_view("TT-METAL");
  exec_params.executable_data = iree_make_const_byte_span(NULL, 0);

  iree::vm::ref<iree_hal_executable_t> executable;
  iree_status_t exec_status = iree_hal_tt_executable_create(
      device_, &exec_params, host_allocator(), &executable);
  if (!iree_status_is_ok(exec_status)) {
    iree_status_ignore(exec_status);
    GTEST_SKIP() << "kernel compilation not available";
  }

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
    IREE_ASSERT_OK(iree_hal_device_queue_execute(
        device_, 0, no_sema, no_sema, cmd.get(), no_bt, 0));

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
    IREE_ASSERT_OK(iree_hal_device_queue_execute(
        device_, 0, no_sema, no_sema, cmd.get(), no_bt, 0));

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
#endif  // !TT_IREE_ENABLE_MOCK
}

TEST_F(DispatchTest, BuiltinMatmulBfloat16ViaTTEX) {
  iree::vm::ref<iree_hal_executable_t> executable;
  iree_hal_executable_t* executable_ptr = nullptr;
  iree_status_t exec_status = CreateBuiltinTtexExecutable(
      device_, host_allocator(),
      TT_IREE_TTEX_BUILTIN_PROGRAM_BF16_MATMUL_32X32X32, &executable_ptr);
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
  IREE_ASSERT_OK(iree_hal_device_queue_execute(
      device_, 0, no_sema, no_sema, cmd_buffer.get(), no_bt, 0));

  std::vector<bfloat16> host_out(kElementsPerTile);
  tt::tt_metal::detail::ReadFromBuffer(
      *iree_hal_tt_buffer_handle(out_buffer.get()), host_out);

  constexpr float kExpected = 64.0f;
  constexpr float kEps = 0.25f;
  for (uint32_t i = 0; i < kElementsPerTile; ++i) {
    ASSERT_NEAR(static_cast<float>(host_out[i]), kExpected, kEps)
        << "matmul mismatch at " << i;
  }
#endif  // !TT_IREE_ENABLE_MOCK
}

}  // namespace
}  // namespace tenstorrent
}  // namespace hal
}  // namespace iree
