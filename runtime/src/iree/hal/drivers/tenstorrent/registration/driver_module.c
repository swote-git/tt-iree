// Copyright 2025 The tt-iree Authors
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/hal/drivers/tenstorrent/registration/driver_module.h"

#include "iree/hal/drivers/tenstorrent/api.h"

//===----------------------------------------------------------------------===//
// Driver factory
//===----------------------------------------------------------------------===//

static iree_status_t iree_hal_tenstorrent_driver_factory_enumerate(
    void* self,
    iree_host_size_t* out_driver_info_count,
    const iree_hal_driver_info_t** out_driver_infos) {
  static const iree_hal_driver_info_t driver_infos[] = {
      {
          .driver_name = IREE_SVL("tenstorrent"),
          .full_name = IREE_SVL("Tenstorrent AI Accelerator (P100A/Wormhole)"),
      },
  };
  *out_driver_info_count = IREE_ARRAYSIZE(driver_infos);
  *out_driver_infos = driver_infos;
  return iree_ok_status();
}

static iree_status_t iree_hal_tenstorrent_driver_factory_try_create(
    void* self,
    iree_string_view_t driver_name,
    iree_allocator_t host_allocator,
    iree_hal_driver_t** out_driver) {
  
  // Only respond to "tenstorrent" driver name
  if (!iree_string_view_equal(driver_name, IREE_SV("tenstorrent"))) {
    return iree_make_status(IREE_STATUS_UNAVAILABLE,
                           "no driver '%.*s'",
                           (int)driver_name.size, driver_name.data);
  }
  
  return iree_hal_tenstorrent_driver_create(
      IREE_SV("tenstorrent"),
      host_allocator,
      out_driver);
}

//===----------------------------------------------------------------------===//
// Public API
//===----------------------------------------------------------------------===//

IREE_API_EXPORT iree_status_t iree_hal_tenstorrent_driver_module_register(
    iree_hal_driver_registry_t* registry) {
  static const iree_hal_driver_factory_t factory = {
      .self = NULL,
      .enumerate = iree_hal_tenstorrent_driver_factory_enumerate,
      .try_create = iree_hal_tenstorrent_driver_factory_try_create,
  };
  return iree_hal_driver_registry_register_factory(registry, &factory);
}
