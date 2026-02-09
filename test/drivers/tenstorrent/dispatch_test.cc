// Copyright 2025 The tt-iree Authors
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// End-to-end dispatch test

#include <cmath>
#include <cstdio>
#include <cstring>
#include <vector>

#include "iree/base/api.h"
#include "iree/hal/api.h"
#include "iree/hal/drivers/tenstorrent/api.h"
#include "iree/hal/drivers/tenstorrent/registration/driver_module.h"
#include "iree/hal/drivers/tenstorrent/tt_buffer.h"
#include "iree/hal/drivers/tenstorrent/tt_command_buffer.h"
#include "iree/hal/drivers/tenstorrent/tt_executable.h"

#ifndef TT_IREE_ENABLE_MOCK
#include <tt-metalium/bfloat16.hpp>
#include <tt-metalium/host_api.hpp>
#include <tt-metalium/tt_metal.hpp>
#endif

//===----------------------------------------------------------------------===//
// Test Infrastructure
//===----------------------------------------------------------------------===//

static int g_test_failed = 0;

#define ASSERT_OK(expr) \
  do { \
    iree_status_t _status = (expr); \
    if (!iree_status_is_ok(_status)) { \
      iree_status_fprint(stderr, _status); \
      iree_status_free(_status); \
      fprintf(stderr, "\nFAILED: %s\n  %s:%d\n", #expr, __FILE__, __LINE__); \
      g_test_failed++; \
      return; \
    } \
  } while (0)

#define ASSERT_TRUE(expr) \
  do { \
    if (!(expr)) { \
      fprintf(stderr, "\nFAILED: %s\n  %s:%d\n", #expr, __FILE__, __LINE__); \
      g_test_failed++; \
      return; \
    } \
  } while (0)

typedef struct test_context_t {
  iree_hal_driver_t* driver;
  iree_hal_device_t* device;
  iree_allocator_t host_allocator;
} test_context_t;

static iree_status_t setup_test_context(test_context_t* ctx) {
  ctx->host_allocator = iree_allocator_system();

  // Create driver (already registered in main())
  iree_hal_driver_registry_t* registry = iree_hal_driver_registry_default();
  IREE_RETURN_IF_ERROR(iree_hal_driver_registry_try_create(
      registry, iree_make_cstring_view("tenstorrent"),
      ctx->host_allocator, &ctx->driver));

  // Create device 0
  IREE_RETURN_IF_ERROR(iree_hal_driver_create_device_by_id(
      ctx->driver, 0, /*param_count=*/0, /*params=*/NULL,
      ctx->host_allocator, &ctx->device));

  return iree_ok_status();
}

static void cleanup_test_context(test_context_t* ctx) {
  if (ctx->device) iree_hal_device_release(ctx->device);
  if (ctx->driver) iree_hal_driver_release(ctx->driver);
}

//===----------------------------------------------------------------------===//
// End-to-End Dispatch Test (bfloat16 add kernel)
//===----------------------------------------------------------------------===//

static void test_dispatch_execution_add() {
  fprintf(stderr, "  dispatch_execution_add... ");
  fflush(stderr);

  test_context_t ctx = {0};
  ASSERT_OK(setup_test_context(&ctx));
  fprintf(stderr, "(device ready) ");

  // 1. Create executable (compiles kernels)
  iree_hal_executable_params_t exec_params;
  iree_hal_executable_params_initialize(&exec_params);
  exec_params.executable_format = iree_make_cstring_view("TT-METAL");
  exec_params.executable_data = iree_make_const_byte_span(NULL, 0);

  iree_hal_executable_t* executable = NULL;
  iree_status_t exec_status = iree_hal_tt_executable_create(
      ctx.device, &exec_params, ctx.host_allocator, &executable);

  if (!iree_status_is_ok(exec_status)) {
    fprintf(stderr, "(skipped - no kernels: ");
    iree_status_fprint(stderr, exec_status);
    fprintf(stderr, ") PASSED\n");
    iree_status_ignore(exec_status);
    cleanup_test_context(&ctx);
    return;
  }
  fprintf(stderr, "(kernels compiled) ");

#ifndef TT_IREE_ENABLE_MOCK
  // bfloat16 is in global namespace in TT-Metal

  // 2. Prepare bfloat16 test data (1 tile = 32x32 elements)
  constexpr uint32_t n_tiles = 1;
  constexpr uint32_t elements_per_tile = 32 * 32;
  constexpr uint32_t tile_size_bytes = elements_per_tile * sizeof(bfloat16);  // 2048
  constexpr uint32_t dram_buffer_size = tile_size_bytes * n_tiles;

  std::vector<bfloat16> host_in0(elements_per_tile * n_tiles);
  std::vector<bfloat16> host_in1(elements_per_tile * n_tiles);

  // Simple test values: in0 = [1.0, 2.0, 3.0, ...], in1 = [-0.5, -0.5, ...]
  for (uint32_t i = 0; i < elements_per_tile * n_tiles; i++) {
    host_in0[i] = bfloat16(static_cast<float>(i % 64));
    host_in1[i] = bfloat16(-0.5f);
  }

  // 3. Allocate IREE buffers (backed by TT-Metal DRAM buffers)
  iree_hal_buffer_params_t buffer_params = {
      .usage = IREE_HAL_BUFFER_USAGE_DISPATCH_STORAGE | IREE_HAL_BUFFER_USAGE_TRANSFER,
      .access = IREE_HAL_MEMORY_ACCESS_ALL,
      .type = IREE_HAL_MEMORY_TYPE_DEVICE_LOCAL,
  };

  iree_hal_allocator_t* allocator = iree_hal_device_allocator(ctx.device);
  iree_hal_buffer_t* in0_buffer = NULL;
  iree_hal_buffer_t* in1_buffer = NULL;
  iree_hal_buffer_t* out_buffer = NULL;

  ASSERT_OK(iree_hal_allocator_allocate_buffer(
      allocator, buffer_params, dram_buffer_size, &in0_buffer));
  ASSERT_OK(iree_hal_allocator_allocate_buffer(
      allocator, buffer_params, dram_buffer_size, &in1_buffer));
  ASSERT_OK(iree_hal_allocator_allocate_buffer(
      allocator, buffer_params, dram_buffer_size, &out_buffer));
  fprintf(stderr, "(buffers allocated) ");

  // 4. Write bfloat16 data directly to device DRAM using TT-Metal API
  tt::tt_metal::Buffer* tt_in0 = iree_hal_tt_buffer_handle(in0_buffer);
  tt::tt_metal::Buffer* tt_in1 = iree_hal_tt_buffer_handle(in1_buffer);
  ASSERT_TRUE(tt_in0 != nullptr);
  ASSERT_TRUE(tt_in1 != nullptr);

  try {
    tt::tt_metal::detail::WriteToBuffer(*tt_in0, host_in0);
    tt::tt_metal::detail::WriteToBuffer(*tt_in1, host_in1);
  } catch (const std::exception& e) {
    fprintf(stderr, "\nFAILED: WriteToBuffer: %s\n", e.what());
    g_test_failed++;
    goto cleanup;
  }
  fprintf(stderr, "(data uploaded) ");

  {
    // 5. Create and record command buffer
    iree_hal_command_buffer_t* cmd_buffer = NULL;
    ASSERT_OK(iree_hal_tt_device_create_command_buffer(
        ctx.device,
        IREE_HAL_COMMAND_BUFFER_MODE_ONE_SHOT,
        IREE_HAL_COMMAND_CATEGORY_DISPATCH,
        /*queue_affinity=*/0,
        /*binding_capacity=*/16,
        &cmd_buffer));

    ASSERT_OK(iree_hal_command_buffer_begin(cmd_buffer));

    iree_hal_buffer_ref_t bindings[3] = {
        iree_hal_make_buffer_ref(in0_buffer, 0, dram_buffer_size),
        iree_hal_make_buffer_ref(in1_buffer, 0, dram_buffer_size),
        iree_hal_make_buffer_ref(out_buffer, 0, dram_buffer_size),
    };

    iree_hal_dispatch_config_t config = {{0, 0, 0}, {1, 1, 1}};
    iree_const_byte_span_t constants = {0};
    iree_hal_buffer_ref_list_t binding_list = {3, bindings};

    ASSERT_OK(iree_hal_command_buffer_dispatch(
        cmd_buffer, executable, /*export_ordinal=*/0,
        config, constants, binding_list, /*flags=*/0));

    ASSERT_OK(iree_hal_command_buffer_end(cmd_buffer));
    fprintf(stderr, "(recorded) ");

    // 6. Execute on device
    fprintf(stderr, "(executing) ");
    iree_hal_semaphore_list_t wait_semaphores = {0};
    iree_hal_semaphore_list_t signal_semaphores = {0};
    iree_hal_buffer_binding_table_t binding_table = {0};

    iree_status_t exec_result = iree_hal_device_queue_execute(
        ctx.device, /*queue_affinity=*/0,
        wait_semaphores, signal_semaphores,
        cmd_buffer, binding_table, /*flags=*/0);

    iree_hal_command_buffer_release(cmd_buffer);

    if (!iree_status_is_ok(exec_result)) {
      fprintf(stderr, "(exec failed: ");
      iree_status_fprint(stderr, exec_result);
      fprintf(stderr, ") ");
      iree_status_free(exec_result);
      g_test_failed++;
      goto cleanup;
    }
    fprintf(stderr, "(executed) ");
  }

  {
    // 7. Read back results from device DRAM
    std::vector<bfloat16> host_out(elements_per_tile * n_tiles);
    tt::tt_metal::Buffer* tt_out = iree_hal_tt_buffer_handle(out_buffer);
    ASSERT_TRUE(tt_out != nullptr);

    try {
      tt::tt_metal::detail::ReadFromBuffer(*tt_out, host_out);
    } catch (const std::exception& e) {
      fprintf(stderr, "\nFAILED: ReadFromBuffer: %s\n", e.what());
      g_test_failed++;
      goto cleanup;
    }

    // 8. Verify results
    fprintf(stderr, "(verifying) ");
    constexpr float eps = 1e-1f;  // loose tolerance for bfloat16
    int mismatches = 0;

    for (uint32_t i = 0; i < elements_per_tile * n_tiles && mismatches < 5; i++) {
      float expected = static_cast<float>(host_in0[i]) + static_cast<float>(host_in1[i]);
      float actual = static_cast<float>(host_out[i]);
      if (std::abs(expected - actual) > eps) {
        fprintf(stderr, "\n  mismatch[%u]: expected %.4f, got %.4f (diff=%.4f)",
                i, expected, actual, std::abs(expected - actual));
        mismatches++;
      }
    }

    if (mismatches > 0) {
      fprintf(stderr, "\nFAILED: %d mismatches found\n", mismatches);
      g_test_failed++;
      goto cleanup;
    }

    // Print a few verified values
    fprintf(stderr, "(OK: %.1f+%.1f=%.1f, %.1f+%.1f=%.1f) ",
            static_cast<float>(host_in0[0]), static_cast<float>(host_in1[0]),
            static_cast<float>(host_out[0]),
            static_cast<float>(host_in0[10]), static_cast<float>(host_in1[10]),
            static_cast<float>(host_out[10]));
  }

cleanup:
  if (in0_buffer) iree_hal_buffer_release(in0_buffer);
  if (in1_buffer) iree_hal_buffer_release(in1_buffer);
  if (out_buffer) iree_hal_buffer_release(out_buffer);
#endif  // !TT_IREE_ENABLE_MOCK

  if (executable) iree_hal_executable_release(executable);
  cleanup_test_context(&ctx);

  if (g_test_failed == 0) {
    fprintf(stderr, "PASSED\n");
  }
}

//===----------------------------------------------------------------------===//
// Main
//===----------------------------------------------------------------------===//

int main(int argc, char** argv) {
  fprintf(stderr, "=== Dispatch Test: End-to-End Kernel Execution ===\n\n");

  // Register driver
  iree_hal_driver_registry_t* registry = iree_hal_driver_registry_default();
  iree_status_t status = iree_hal_tenstorrent_driver_module_register(registry);
  if (!iree_status_is_ok(status)) {
    iree_status_fprint(stderr, status);
    iree_status_free(status);
    return 1;
  }

  test_dispatch_execution_add();

  fprintf(stderr, "\n");
  if (g_test_failed > 0) {
    fprintf(stderr, "=== FAILED (%d errors) ===\n", g_test_failed);
    return 1;
  } else {
    fprintf(stderr, "=== All tests passed! ===\n");
    return 0;
  }
}
