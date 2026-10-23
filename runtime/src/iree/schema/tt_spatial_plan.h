// Copyright 2026 The tt-iree Authors
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_SCHEMA_TT_SPATIAL_PLAN_H_
#define IREE_SCHEMA_TT_SPATIAL_PLAN_H_

#include "iree/base/api.h"

#ifdef __cplusplus
extern "C" {
#endif

#define TT_IREE_SPATIAL_NO_BINDING UINT32_MAX
#define TT_IREE_SPATIAL_NO_CORE UINT32_MAX
#define TT_IREE_SPATIAL_MAX_CORES 1024u
#define TT_IREE_SPATIAL_MAX_RESOURCES 1024u
#define TT_IREE_SPATIAL_MAX_ROUTES 4096u
#define TT_IREE_SPATIAL_MAX_SHARDS 16384u

typedef enum tt_iree_spatial_memory_space_e {
  TT_IREE_SPATIAL_MEMORY_DRAM = 1,
  TT_IREE_SPATIAL_MEMORY_L1 = 2,
} tt_iree_spatial_memory_space_e;

typedef enum tt_iree_spatial_sharding_e {
  TT_IREE_SPATIAL_SHARDING_NONE = 0,
  TT_IREE_SPATIAL_SHARDING_PARTITIONED = 1,
  TT_IREE_SPATIAL_SHARDING_REPLICATED = 2,
} tt_iree_spatial_sharding_e;

typedef enum tt_iree_spatial_storage_e {
  TT_IREE_SPATIAL_STORAGE_LINEAR = 0,
  TT_IREE_SPATIAL_STORAGE_TILE_32X32_FACE = 1,
} tt_iree_spatial_storage_e;

typedef enum tt_iree_spatial_lifetime_e {
  TT_IREE_SPATIAL_LIFETIME_DISPATCH = 0,
  // A nonzero resource_key identifies an allocation retained by an execution
  // session. Access epochs remain local to this invocation, not graph ordinals.
  TT_IREE_SPATIAL_LIFETIME_SESSION = 1,
} tt_iree_spatial_lifetime_e;

typedef enum tt_iree_spatial_transfer_e {
  TT_IREE_SPATIAL_TRANSFER_READ = 0,
  TT_IREE_SPATIAL_TRANSFER_WRITE = 1,
  TT_IREE_SPATIAL_TRANSFER_MULTICAST = 2,
} tt_iree_spatial_transfer_e;

// Coordinates are logical worker coordinates, never harvested/physical NoC IDs.
typedef struct tt_iree_spatial_core_t {
  uint16_t x;
  uint16_t y;
  // Uses TT_IREE_TTEX_AOT_KERNEL_ROLE_BIT(reader/writer/compute).
  uint32_t kernel_roles;
} tt_iree_spatial_core_t;

typedef struct tt_iree_spatial_shard_t {
  uint32_t core; // Index in cores, or NO_CORE for unsharded DRAM.
  uint64_t logical_byte_offset;
  uint64_t byte_length;
  // Relative to the plan-owned per-core L1 arena, not a physical L1 address.
  uint64_t arena_byte_offset;
} tt_iree_spatial_shard_t;

typedef struct tt_iree_spatial_resource_t {
  // HAL binding view or plan-internal scratch. byte_length includes padding in
  // the declared storage format; tensor indexing remains in TensorDesc.
  uint32_t binding;
  uint64_t binding_byte_offset;
  uint64_t byte_length;
  uint8_t element_type; // Existing TensorElementType enum.
  uint8_t storage;
  uint8_t memory_space;
  uint8_t sharding;
  uint8_t lifetime;
  uint64_t resource_key;
  // Half-open access interval in plan-local ordered epochs. DISPATCH storage
  // may be reused after last_epoch; SESSION storage remains reserved
  // throughout.
  uint32_t first_epoch;
  uint32_t last_epoch;
  uint32_t alignment;
  iree_host_size_t shard_count;
  const tt_iree_spatial_shard_t *shards;
} tt_iree_spatial_resource_t;

typedef struct tt_iree_spatial_route_t {
  uint8_t transfer;
  uint8_t noc; // 0 or 1; runtime resolves endpoints to physical coordinates.
  uint32_t epoch;
  uint32_t source_resource;
  uint32_t source_shard;
  uint64_t source_byte_offset; // Relative to the source shard.
  uint32_t destination_resource;
  iree_host_size_t destination_count;
  const uint32_t *destination_shards;
  uint64_t destination_byte_offset; // Same relative offset at each receiver.
  uint64_t byte_length;
  // Earlier route indices whose destination writes must be visible before
  // this transfer starts. These are logical edges, not hardware semaphore IDs.
  iree_host_size_t dependency_count;
  const uint32_t *dependencies;
} tt_iree_spatial_route_t;

typedef struct tt_iree_spatial_plan_t {
  uint32_t epoch_count;
  // Resource arena only; CBs, code, stacks and semaphore allocations are extra.
  uint64_t l1_bytes_per_core;
  iree_host_size_t core_count;
  const tt_iree_spatial_core_t *cores;
  iree_host_size_t resource_count;
  const tt_iree_spatial_resource_t *resources;
  iree_host_size_t route_count;
  const tt_iree_spatial_route_t *routes;
} tt_iree_spatial_plan_t;

// Validates the static contract, not device capacity or kernel/dataflow safety.
iree_status_t tt_iree_validate_spatial_plan(const tt_iree_spatial_plan_t *plan,
                                            uint16_t binding_count);

// Validates an already FlatBuffer-verified SpatialPlanDef table. The pointer is
// borrowed for this call only. This never creates allocations on a TT device.
iree_status_t tt_iree_validate_spatial_plan_table(const void *table,
                                                  uint16_t binding_count);

#ifdef __cplusplus
}
#endif

#endif // IREE_SCHEMA_TT_SPATIAL_PLAN_H_
