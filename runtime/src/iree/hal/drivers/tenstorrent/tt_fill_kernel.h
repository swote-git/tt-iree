// Copyright 2026 The tt-iree Authors
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_HAL_DRIVERS_TENSTORRENT_TT_FILL_KERNEL_H_
#define IREE_HAL_DRIVERS_TENSTORRENT_TT_FILL_KERNEL_H_

namespace iree::hal::tenstorrent {

inline constexpr char kDeviceFillSource[] = R"ttfill(
#include <cstdint>
#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/circular_buffer.h"

void kernel_main() {
  constexpr auto accessor = TensorAccessorArgs<0>();
  const auto target = TensorAccessor(accessor, get_arg_val<uint32_t>(0));
  uint32_t offset = get_arg_val<uint32_t>(1);
  uint32_t remaining = get_arg_val<uint32_t>(2);
  const uint32_t pattern = get_arg_val<uint32_t>(3);
  const uint32_t scratch = get_write_ptr(0);
  auto words = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(scratch);
  for (uint32_t i = 0; i < 2048 / 4; ++i) words[i] = pattern;
  while (remaining) {
    const uint32_t in_page = offset % 2048;
    uint32_t count = 2048 - in_page;
    if (count > remaining) count = remaining;
    // Matching source/destination low address bits permits byte-range writes.
    // Only the requested bytes are sent; adjacent arena tensors are untouched.
    noc_async_write(scratch + in_page,
                    target.get_noc_addr(offset / 2048, in_page), count);
    noc_async_write_barrier();
    offset += count;
    remaining -= count;
  }
}
)ttfill";

} // namespace iree::hal::tenstorrent

#endif // IREE_HAL_DRIVERS_TENSTORRENT_TT_FILL_KERNEL_H_
