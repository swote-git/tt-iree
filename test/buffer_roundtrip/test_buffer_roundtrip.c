// Copyright 2025 The tt-iree Authors
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// Buffer Round-Trip Test
// ======================
// This test validates the core Week 2 functionality:
//   1. Device initialization
//   2. Buffer allocation
//   3. Tile layout conversion (row-major ↔ 32x32 tiles)
//   4. Host ↔ Device data transfer
//
// Success criteria:
//   - Data written to device matches data read back
//   - Tile packing/unpacking is bit-exact

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "iree/base/api.h"
#include "iree/hal/api.h"
#include "iree/hal/drivers/tenstorrent/api.h"
#include "iree/hal/drivers/tenstorrent/registration/driver_module.h"

// Test helper: Create sequential test data
static void fill_sequential(float* data, int size) {
  for (int i = 0; i < size; i++) {
    data[i] = (float)i;
  }
}

// Test helper: Compare float arrays with tolerance
static bool arrays_equal(const float* a, const float* b, int size, float tol) {
  for (int i = 0; i < size; i++) {
    if (fabsf(a[i] - b[i]) > tol) {
      fprintf(stderr, "Mismatch at index %d: %.6f != %.6f\n", i, a[i], b[i]);
      return false;
    }
  }
  return true;
}

//===----------------------------------------------------------------------===//
// Test 1: Tile Layout Conversion
//===----------------------------------------------------------------------===//

static iree_status_t test_tile_conversion(void) {
  printf("Test 1: Tile Layout Conversion\n");
  printf("================================\n");
  
  // Test with 64x64 matrix (4 tiles: 2x2 grid)
  const int rows = 64;
  const int cols = 64;
  const int size = rows * cols;
  
  float* row_major = (float*)malloc(size * sizeof(float));
  float* tiled = (float*)malloc(size * sizeof(float));
  float* unpacked = (float*)malloc(size * sizeof(float));
  
  // Fill with sequential data
  fill_sequential(row_major, size);
  
  // Pack: row-major → tiles
  iree_hal_tt_pack_to_tiles(row_major, tiled, rows, cols);
  
  // Unpack: tiles → row-major
  iree_hal_tt_unpack_from_tiles(tiled, unpacked, rows, cols);
  
  // Verify: unpacked should match original
  bool success = arrays_equal(row_major, unpacked, size, 1e-6f);
  
  if (success) {
    printf("✓ Tile conversion is bit-exact\n");
    
    // Print sample to verify tile ordering
    printf("  Sample values:\n");
    printf("    Original[0] = %.1f, Tiled[0] = %.1f (should match)\n",
           row_major[0], tiled[0]);
    printf("    Original[32] = %.1f, Tiled[32] = %.1f (should match within tile)\n",
           row_major[32], tiled[32]);
    printf("    Original[1024] = %.1f, Tiled[1024] = %.1f (start of 2nd tile)\n",
           row_major[1024], tiled[1024]);
  } else {
    printf("✗ Tile conversion failed\n");
  }
  
  free(row_major);
  free(tiled);
  free(unpacked);
  
  return success ? iree_ok_status() 
                 : iree_make_status(IREE_STATUS_DATA_LOSS, "tile conversion mismatch");
}

//===----------------------------------------------------------------------===//
// Test 2: Buffer Allocation
//===----------------------------------------------------------------------===//

static iree_status_t test_buffer_allocation(iree_hal_device_t* device) {
  printf("\nTest 2: Buffer Allocation\n");
  printf("==========================\n");
  
  iree_hal_allocator_t* allocator = iree_hal_device_allocator(device);
  
  // Allocate 32x32 float buffer (4KB)
  const iree_device_size_t buffer_size = 32 * 32 * sizeof(float);
  
  iree_hal_buffer_params_t params = {
      .type = IREE_HAL_MEMORY_TYPE_DEVICE_LOCAL,
      .usage = IREE_HAL_BUFFER_USAGE_TRANSFER | 
               IREE_HAL_BUFFER_USAGE_DISPATCH_STORAGE,
  };
  
  iree_hal_buffer_t* buffer = NULL;
  iree_status_t status = iree_hal_allocator_allocate_buffer(
      allocator, params, buffer_size, &buffer);
  
  if (iree_status_is_ok(status)) {
    printf("✓ Buffer allocated: %zu bytes\n", (size_t)buffer_size);
    
    // Verify buffer properties
    iree_device_size_t allocated_size = iree_hal_buffer_allocation_size(buffer);
    printf("  Allocated size: %zu bytes\n", (size_t)allocated_size);
    
    if (allocated_size >= buffer_size) {
      printf("✓ Buffer size correct\n");
    } else {
      printf("✗ Buffer size mismatch\n");
      status = iree_make_status(IREE_STATUS_INTERNAL, "size mismatch");
    }
    
    iree_hal_buffer_release(buffer);
  } else {
    printf("✗ Buffer allocation failed: %s\n", 
           iree_status_code_string(iree_status_code(status)));
  }
  
  return status;
}

//===----------------------------------------------------------------------===//
// Test 3: Buffer Round-Trip (THE BIG ONE)
//===----------------------------------------------------------------------===//

static iree_status_t test_buffer_roundtrip(iree_hal_device_t* device) {
  printf("\nTest 3: Buffer Round-Trip\n");
  printf("==========================\n");
  
  iree_hal_allocator_t* allocator = iree_hal_device_allocator(device);
  
  // Test with single 32x32 tile
  const int rows = 32;
  const int cols = 32;
  const int size = rows * cols;
  const iree_device_size_t buffer_size = size * sizeof(float);
  
  // Allocate buffer
  iree_hal_buffer_params_t params = {
      .type = IREE_HAL_MEMORY_TYPE_DEVICE_LOCAL,
      .usage = IREE_HAL_BUFFER_USAGE_TRANSFER | 
               IREE_HAL_BUFFER_USAGE_DISPATCH_STORAGE,
  };
  
  iree_hal_buffer_t* buffer = NULL;
  IREE_RETURN_IF_ERROR(iree_hal_allocator_allocate_buffer(
      allocator, params, buffer_size, &buffer));
  
  printf("✓ Buffer allocated\n");
  
  // Prepare test data
  float* input_data = (float*)malloc(buffer_size);
  float* output_data = (float*)malloc(buffer_size);
  fill_sequential(input_data, size);
  
  printf("✓ Test data prepared (sequential 0..%d)\n", size - 1);
  
  // Step 1: Write to device
  iree_status_t status;
  {
    iree_hal_buffer_mapping_t mapping;
    status = iree_hal_buffer_map_range(
        buffer,
        IREE_HAL_MAPPING_MODE_SCOPED,
        IREE_HAL_MEMORY_ACCESS_WRITE,
        /*offset=*/0,
        buffer_size,
        &mapping);
    
    if (iree_status_is_ok(status)) {
      memcpy(mapping.contents.data, input_data, buffer_size);
      
      status = iree_hal_buffer_unmap_range(buffer, 0, buffer_size, &mapping);
      
      if (iree_status_is_ok(status)) {
        printf("✓ Data written to device (with tile conversion)\n");
      }
    }
  }
  
  if (!iree_status_is_ok(status)) {
    printf("✗ Write failed: %s\n", 
           iree_status_code_string(iree_status_code(status)));
    goto cleanup;
  }
  
  // Step 2: Read from device
  {
    iree_hal_buffer_mapping_t mapping;
    status = iree_hal_buffer_map_range(
        buffer,
        IREE_HAL_MAPPING_MODE_SCOPED,
        IREE_HAL_MEMORY_ACCESS_READ,
        /*offset=*/0,
        buffer_size,
        &mapping);
    
    if (iree_status_is_ok(status)) {
      memcpy(output_data, mapping.contents.data, buffer_size);
      
      status = iree_hal_buffer_unmap_range(buffer, 0, buffer_size, &mapping);
      
      if (iree_status_is_ok(status)) {
        printf("✓ Data read from device (with tile conversion)\n");
      }
    }
  }
  
  if (!iree_status_is_ok(status)) {
    printf("✗ Read failed: %s\n", 
           iree_status_code_string(iree_status_code(status)));
    goto cleanup;
  }
  
  // Step 3: Verify data integrity
  bool data_matches = arrays_equal(input_data, output_data, size, 1e-6f);
  
  if (data_matches) {
    printf("✓ Round-trip successful! Data matches exactly.\n");
    printf("  Sample: input[0]=%.1f, output[0]=%.1f\n", 
           input_data[0], output_data[0]);
    printf("  Sample: input[100]=%.1f, output[100]=%.1f\n",
           input_data[100], output_data[100]);
  } else {
    printf("✗ Round-trip failed: data mismatch\n");
    status = iree_make_status(IREE_STATUS_DATA_LOSS, "data mismatch");
  }

cleanup:
  free(input_data);
  free(output_data);
  iree_hal_buffer_release(buffer);
  
  return status;
}

//===----------------------------------------------------------------------===//
// Main Test Runner
//===----------------------------------------------------------------------===//

int main(int argc, char** argv) {
  printf("==============================================\n");
  printf("tt-iree Week 2: Buffer Round-Trip Test Suite\n");
  printf("==============================================\n\n");
  
  // Initialize IREE
  iree_status_t status = iree_ok_status();
  
  // Test 1: Pure tile conversion (no device needed)
  status = test_tile_conversion();
  if (!iree_status_is_ok(status)) {
    fprintf(stderr, "Tile conversion test failed\n");
    return 1;
  }
  
  // Initialize HAL driver registry
  iree_hal_driver_registry_t* registry = NULL;
  status = iree_hal_driver_registry_allocate(iree_allocator_system(), &registry);
  if (!iree_status_is_ok(status)) {
    fprintf(stderr, "Failed to create driver registry\n");
    return 1;
  }
  
  // Register Tenstorrent driver
  status = iree_hal_tenstorrent_driver_module_register(registry);
  if (!iree_status_is_ok(status)) {
    fprintf(stderr, "Failed to register Tenstorrent driver\n");
    iree_hal_driver_registry_free(registry);
    return 1;
  }
  
  // Create driver
  iree_hal_driver_t* driver = NULL;
  status = iree_hal_driver_registry_try_create(
      registry, IREE_SV("tenstorrent"), iree_allocator_system(), &driver);
  
  if (!iree_status_is_ok(status)) {
    fprintf(stderr, "Failed to create Tenstorrent driver\n");
    iree_hal_driver_registry_free(registry);
    return 1;
  }
  
  printf("\n✓ Tenstorrent driver created\n");
  
  // Create device
  iree_hal_device_t* device = NULL;
  status = iree_hal_driver_create_device_by_id(
      driver, /*device_id=*/0, /*param_count=*/0, /*params=*/NULL,
      iree_allocator_system(), &device);
  
  if (!iree_status_is_ok(status)) {
    fprintf(stderr, "Failed to create device: %s\n",
            iree_status_code_string(iree_status_code(status)));
    iree_hal_driver_release(driver);
    iree_hal_driver_registry_free(registry);
    return 1;
  }
  
  printf("✓ Device created (Device ID: 0)\n");
  
  // Run device-dependent tests
  status = test_buffer_allocation(device);
  if (iree_status_is_ok(status)) {
    status = test_buffer_roundtrip(device);
  }
  
  // Cleanup
  iree_hal_device_release(device);
  iree_hal_driver_release(driver);
  iree_hal_driver_registry_free(registry);
  
  printf("\n==============================================\n");
  if (iree_status_is_ok(status)) {
    printf("✓ ALL TESTS PASSED - Week 2 Complete!\n");
    printf("==============================================\n");
    return 0;
  } else {
    printf("✗ TESTS FAILED\n");
    printf("==============================================\n");
    return 1;
  }
}