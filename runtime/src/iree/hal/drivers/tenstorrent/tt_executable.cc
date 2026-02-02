// runtime/src/iree/hal/drivers/tenstorrent/tt_executable.cc

#include "iree/hal/drivers/tenstorrent/tt_executable.h"

#include <vector>
#include <string>

#include "iree/base/api.h"
#include "iree/hal/drivers/tenstorrent/tt_device.h"

#ifndef TT_IREE_ENABLE_MOCK
#include "tt-metalium/host_api.hpp"
#include "tt-metalium/tt_metal.hpp"
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

static const iree_hal_executable_vtable_t iree_hal_tt_executable_vtable;

static iree_hal_tt_executable_t* iree_hal_tt_executable_cast(iree_hal_executable_t* base) {
  IREE_HAL_ASSERT_TYPE(base, &iree_hal_tt_executable_vtable);
  return (iree_hal_tt_executable_t*)base;
}

//===----------------------------------------------------------------------===//
// Helper: Create TT-Metal Program from Source (JIT)
//===----------------------------------------------------------------------===//

#ifndef TT_IREE_ENABLE_MOCK
static iree_status_t iree_hal_tt_create_program_for_entry_point(
    tt::tt_metal::IDevice* tt_device,
    iree_string_view_t kernel_name,
    // In real implementation, pass source code strings here
    iree_hal_tt_kernel_params_t* params) {
  
  try {
    // 1. Create Program
    // Note: Program is a lightweight container, typically created on heap or stack.
    // Since we need it to persist, we might need a wrapper or manage its lifetime carefully.
    // TT-Metal Program is movable but not copyable.
    params->program = new tt::tt_metal::Program();
    tt::tt_metal::Program& program = *static_cast<tt::tt_metal::Program*>(params->program);

    // 2. Define Core Range (e.g., single core (0,0) for PoC)
    CoreCoord core_coord = {0, 0};
    CoreRange core = {.start = core_coord, .end = core_coord};

    // 3. Create Circular Buffers (L1 Memory)
    // This configuration should come from the compiler (FlatBuffer).
    // Hardcoded example for "Add" op:
    uint32_t cb_index = 0;
    uint32_t num_tiles = 1;
    uint32_t tile_size = 32 * 32 * 2; // bfloat16 assumption
    
    tt::tt_metal::CircularBufferConfig cb_config = tt::tt_metal::CircularBufferConfig(
        num_tiles * tile_size, {{cb_index, tt::tt_metal::DataFormat::Float16_b}})
        .set_page_size(cb_index, tile_size);
    tt::tt_metal::CreateCircularBuffer(program, core, cb_config);

    // 4. Create Kernels
    // These paths/sources should be extracted from the FlatBuffer.
    // For PoC, we assume standard kernels or use placeholders.
    
    // Reader
    params->reader_kernel_id = tt::tt_metal::CreateKernel(
        program,
        "tt_metal/kernels/dataflow/reader_unary.cpp", // Placeholder path
        core,
        tt::tt_metal::DataMovementConfig{
            .processor = tt::tt_metal::DataMovementProcessor::RISCV_1,
            .noc = tt::tt_metal::NOC::RISCV_1_default});

    // Writer
    params->writer_kernel_id = tt::tt_metal::CreateKernel(
        program,
        "tt_metal/kernels/dataflow/writer_unary.cpp", // Placeholder path
        core,
        tt::tt_metal::DataMovementConfig{
            .processor = tt::tt_metal::DataMovementProcessor::RISCV_0,
            .noc = tt::tt_metal::NOC::RISCV_0_default});

    // Compute
    std::vector<uint32_t> compute_args = {};
    params->compute_kernel_id = tt::tt_metal::CreateKernel(
        program,
        "tt_metal/kernels/compute/eltwise_binary.cpp", // Placeholder path
        core,
        tt::tt_metal::ComputeConfig{
            .compile_args = compute_args});

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

  iree_host_size_t entry_point_count = params->pipeline_layout_count;
  
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
    tt::tt_metal::IDevice* tt_device = iree_hal_tt_device_handle(
        (iree_hal_tt_device_t*)device);
        
    iree_string_view_t name = iree_make_cstring_view("entry_point");
    
    iree_status_t status = iree_hal_tt_create_program_for_entry_point(
        tt_device, name, kernel_params);
        
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

static const iree_hal_executable_vtable_t iree_hal_tt_executable_vtable = {
    .destroy = iree_hal_tt_executable_destroy,
};
