// runtime/src/iree/hal/drivers/tenstorrent/tt_executable.cc

#include "iree/hal/drivers/tenstorrent/tt_executable.h"

#include <vector>
#include <string>

#include "iree/base/api.h"
#include "iree/hal/drivers/tenstorrent/tt_device.h"

#ifndef TT_IREE_ENABLE_MOCK
#include "tt-metalium/host_api.hpp"
#include "tt-metalium/tt_metal.hpp"
#include "tt-metalium/tensor_accessor_args.hpp"
#endif

//===----------------------------------------------------------------------===//
// Executable Structure
//===----------------------------------------------------------------------===//

typedef struct iree_hal_tt_executable_t {
  iree_hal_resource_t resource;
  iree_allocator_t host_allocator;
  iree_hal_device_t* device;

  iree_host_size_t entry_point_count;
  
  // Storage for kernel parameters (Programs are managed here)
  iree_hal_tt_kernel_params_t entry_points[];
} iree_hal_tt_executable_t;

// Forward declare vtable (defined at end of file)
extern const iree_hal_executable_vtable_t iree_hal_tt_executable_vtable;

static iree_hal_tt_executable_t* iree_hal_tt_executable_cast(iree_hal_executable_t* base) {
  IREE_HAL_ASSERT_TYPE(base, &iree_hal_tt_executable_vtable);
  return (iree_hal_tt_executable_t*)base;
}

//===----------------------------------------------------------------------===//
// Helper: Create TT-Metal Program from Source (JIT)
//===----------------------------------------------------------------------===//

#ifndef TT_IREE_ENABLE_MOCK
static iree_status_t iree_hal_tt_create_program_for_entry_point(
    iree_hal_device_t* device,
    iree_hal_tt_kernel_params_t* params) {

  try {
    // 1. Create Program
    params->program = new tt::tt_metal::Program();
    tt::tt_metal::Program& program = *static_cast<tt::tt_metal::Program*>(params->program);

    // 2. Define Core Range (single core (0,0) for PoC)
    tt::tt_metal::CoreCoord core_coord = {0, 0};
    tt::tt_metal::CoreRange core(core_coord);

    // 3. Create Circular Buffers (3 CBs as required by custom_sfpi_add kernels)
    // Using Float16_b to match the reference example
    constexpr uint32_t tiles_per_cb = 2;
    constexpr uint32_t tile_size_bytes = 32 * 32 * 2;  // bfloat16: 2048 bytes

    // CB c_0: input0
    tt::CBIndex src0_cb_index = tt::CBIndex::c_0;
    tt::tt_metal::CreateCircularBuffer(program, core,
        tt::tt_metal::CircularBufferConfig(
            tiles_per_cb * tile_size_bytes,
            {{src0_cb_index, tt::DataFormat::Float16_b}})
            .set_page_size(src0_cb_index, tile_size_bytes));

    // CB c_1: input1
    tt::CBIndex src1_cb_index = tt::CBIndex::c_1;
    tt::tt_metal::CreateCircularBuffer(program, core,
        tt::tt_metal::CircularBufferConfig(
            tiles_per_cb * tile_size_bytes,
            {{src1_cb_index, tt::DataFormat::Float16_b}})
            .set_page_size(src1_cb_index, tile_size_bytes));

    // CB c_16: output
    tt::CBIndex dst_cb_index = tt::CBIndex::c_16;
    tt::tt_metal::CreateCircularBuffer(program, core,
        tt::tt_metal::CircularBufferConfig(
            tiles_per_cb * tile_size_bytes,
            {{dst_cb_index, tt::DataFormat::Float16_b}})
            .set_page_size(dst_cb_index, tile_size_bytes));

    // 4. Create compile-time args for TensorAccessor (DRAM interleaved buffers)
    std::vector<uint32_t> reader_compile_args;
    tt::tt_metal::TensorAccessorArgs::create_dram_interleaved().append_to(reader_compile_args);
    tt::tt_metal::TensorAccessorArgs::create_dram_interleaved().append_to(reader_compile_args);

    std::vector<uint32_t> writer_compile_args;
    tt::tt_metal::TensorAccessorArgs::create_dram_interleaved().append_to(writer_compile_args);

    // 5. Create Kernels with compile-time args
    // Reader (reads tiles from DRAM to circular buffers)
    params->reader_kernel_id = tt::tt_metal::CreateKernel(
        program,
        "tt_metal/programming_examples/custom_sfpi_add/kernels/dataflow/read_tiles.cpp",
        core,
        tt::tt_metal::DataMovementConfig{
            .processor = tt::tt_metal::DataMovementProcessor::RISCV_0,
            .noc = tt::tt_metal::NOC::RISCV_0_default,
            .compile_args = reader_compile_args});

    // Writer (writes tiles from circular buffers to DRAM)
    params->writer_kernel_id = tt::tt_metal::CreateKernel(
        program,
        "tt_metal/programming_examples/custom_sfpi_add/kernels/dataflow/write_tile.cpp",
        core,
        tt::tt_metal::DataMovementConfig{
            .processor = tt::tt_metal::DataMovementProcessor::RISCV_1,
            .noc = tt::tt_metal::NOC::RISCV_1_default,
            .compile_args = writer_compile_args});

    // Compute (performs tile addition using SFPU)
    params->compute_kernel_id = tt::tt_metal::CreateKernel(
        program,
        "tt_metal/programming_examples/custom_sfpi_add/kernels/compute/tiles_add.cpp",
        core,
        tt::tt_metal::ComputeConfig{});

  } catch (const std::exception& e) {
    return iree_make_status(IREE_STATUS_INTERNAL,
                           "Failed to create TT-Metal program: %s", e.what());
  }

  return iree_ok_status();
}
#endif

//===----------------------------------------------------------------------===//
// Executable Creation
//===----------------------------------------------------------------------===//

iree_status_t iree_hal_tt_executable_create(
    iree_hal_device_t* device,
    const iree_hal_executable_params_t* params,
    iree_allocator_t host_allocator,
    iree_hal_executable_t** out_executable) {
  IREE_ASSERT_ARGUMENT(device);
  IREE_ASSERT_ARGUMENT(params);
  IREE_ASSERT_ARGUMENT(out_executable);
  *out_executable = NULL;

  // For PoC: hardcode 1 entry point. In production, parse executable_data FlatBuffer
  // to determine the actual number of entry points.
  iree_host_size_t entry_point_count = 1;
  
  // Calculate size for executable + array of kernel params
  iree_host_size_t total_size = sizeof(iree_hal_tt_executable_t) + 
                               entry_point_count * sizeof(iree_hal_tt_kernel_params_t);

  iree_hal_tt_executable_t* executable = NULL;
  IREE_RETURN_IF_ERROR(iree_allocator_malloc(host_allocator, total_size, (void**)&executable));

  iree_hal_resource_initialize(&iree_hal_tt_executable_vtable, &executable->resource);
  executable->host_allocator = host_allocator;
  executable->device = device;
  executable->entry_point_count = entry_point_count;

  // Initialize entry points
  for (iree_host_size_t i = 0; i < entry_point_count; ++i) {
    iree_hal_tt_kernel_params_t* kernel_params = &executable->entry_points[i];
    
    // In a full implementation, you would:
    // 1. Read FlatBuffer to get kernel source/metadata for entry point [i].
    // 2. Call iree_hal_tt_create_program_for_entry_point(...)

#ifndef TT_IREE_ENABLE_MOCK
    iree_status_t status = iree_hal_tt_create_program_for_entry_point(
        device, kernel_params);
        
    if (!iree_status_is_ok(status)) {
      iree_hal_executable_destroy((iree_hal_executable_t*)executable);
      return status;
    }
#else
    // Mock mode init
    kernel_params->program = NULL;
#endif
  }

  *out_executable = (iree_hal_executable_t*)executable;
  return iree_ok_status();
}

static void iree_hal_tt_executable_destroy(iree_hal_executable_t* base_executable) {
  iree_hal_tt_executable_t* executable = iree_hal_tt_executable_cast(base_executable);
  
#ifndef TT_IREE_ENABLE_MOCK
  for (iree_host_size_t i = 0; i < executable->entry_point_count; ++i) {
    iree_hal_tt_kernel_params_t* params = &executable->entry_points[i];
    if (params->program) {
      delete static_cast<tt::tt_metal::Program*>(params->program);
    }
  }
#endif

  iree_allocator_free(executable->host_allocator, executable);
}

iree_status_t iree_hal_tt_executable_lookup_kernel_params(
    iree_hal_executable_t* base_executable,
    int32_t entry_point,
    const iree_hal_tt_kernel_params_t** out_params) {
  iree_hal_tt_executable_t* executable = iree_hal_tt_executable_cast(base_executable);
  
  if (entry_point < 0 || entry_point >= executable->entry_point_count) {
    return iree_make_status(IREE_STATUS_OUT_OF_RANGE, "invalid entry point ordinal %d", entry_point);
  }

  *out_params = &executable->entry_points[entry_point];
  return iree_ok_status();
}

iree_status_t iree_hal_tt_executable_lookup_kernel_params_mutable(
    iree_hal_executable_t* base_executable,
    int32_t entry_point,
    iree_hal_tt_kernel_params_t** out_params) {
  iree_hal_tt_executable_t* executable = iree_hal_tt_executable_cast(base_executable);

  if (entry_point < 0 || entry_point >= executable->entry_point_count) {
    return iree_make_status(IREE_STATUS_OUT_OF_RANGE, "invalid entry point ordinal %d", entry_point);
  }

  *out_params = &executable->entry_points[entry_point];
  return iree_ok_status();
}

const iree_hal_executable_vtable_t iree_hal_tt_executable_vtable = {
    .destroy = iree_hal_tt_executable_destroy,
};
