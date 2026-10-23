// Copyright 2026 The tt-iree Authors
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/schema/tt_spatial_plan.h"

#include <algorithm>
#include <unordered_set>
#include <utility>
#include <vector>

#include "iree/schema/tt_executable_builder_util.h"
#include "tt_executable_def_reader.h"

#define ns(x) FLATBUFFERS_WRAP_NAMESPACE(iree_tenstorrent_hal, x)

namespace {
iree_status_t Invalid(const char *message) {
  return iree_make_status(IREE_STATUS_INVALID_ARGUMENT, "TTEX spatial plan: %s",
                          message);
}

bool Fits(uint64_t offset, uint64_t length, uint64_t size) {
  return offset <= size && length <= size - offset;
}

bool Overlaps(uint64_t a, uint64_t a_size, uint64_t b, uint64_t b_size) {
  // Called only after both ranges have been checked against their arena.
  return a < b + b_size && b < a + a_size;
}

struct L1Use {
  uint32_t core;
  uint64_t offset;
  uint64_t length;
  uint32_t begin;
  uint32_t end;
};
} // namespace

iree_status_t tt_iree_validate_spatial_plan(const tt_iree_spatial_plan_t *plan,
                                            uint16_t binding_count) {
  if (!plan || !plan->epoch_count || !plan->core_count || !plan->cores ||
      plan->core_count > TT_IREE_SPATIAL_MAX_CORES || !plan->resource_count ||
      !plan->resources ||
      plan->resource_count > TT_IREE_SPATIAL_MAX_RESOURCES ||
      plan->route_count > TT_IREE_SPATIAL_MAX_ROUTES ||
      (plan->route_count && !plan->routes)) {
    return Invalid("invalid counts or missing arrays");
  }
  constexpr uint32_t kRoles =
      TT_IREE_TTEX_AOT_KERNEL_ROLE_BIT(TT_IREE_TTEX_KERNEL_ROLE_READER) |
      TT_IREE_TTEX_AOT_KERNEL_ROLE_BIT(TT_IREE_TTEX_KERNEL_ROLE_WRITER) |
      TT_IREE_TTEX_AOT_KERNEL_ROLE_BIT(TT_IREE_TTEX_KERNEL_ROLE_COMPUTE);
  std::unordered_set<uint32_t> coordinates;
  for (iree_host_size_t i = 0; i < plan->core_count; ++i) {
    const auto &core = plan->cores[i];
    if (!core.kernel_roles || (core.kernel_roles & ~kRoles) ||
        !coordinates.insert((uint32_t(core.y) << 16) | core.x).second) {
      return Invalid("duplicate core or invalid kernel roles");
    }
  }
  std::unordered_set<uint64_t> resource_keys;
  std::vector<L1Use> l1_uses;
  iree_host_size_t total_shards = 0;
  for (iree_host_size_t i = 0; i < plan->resource_count; ++i) {
    const auto &resource = plan->resources[i];
    const uint32_t element_bytes =
        resource.element_type == TT_IREE_TTEX_TENSOR_ELEMENT_TYPE_BF16  ? 2
        : resource.element_type == TT_IREE_TTEX_TENSOR_ELEMENT_TYPE_F32 ? 4
                                                                        : 0;
    if (!element_bytes || !resource.byte_length ||
        resource.byte_length % element_bytes ||
        !Fits(resource.binding_byte_offset, resource.byte_length, UINT64_MAX) ||
        resource.binding_byte_offset % element_bytes ||
        (resource.binding != TT_IREE_SPATIAL_NO_BINDING &&
         resource.binding >= binding_count) ||
        (resource.binding == TT_IREE_SPATIAL_NO_BINDING &&
         resource.binding_byte_offset) ||
        !resource.alignment ||
        (resource.alignment & (resource.alignment - 1)) ||
        resource.alignment < element_bytes ||
        resource.first_epoch >= resource.last_epoch ||
        resource.last_epoch > plan->epoch_count ||
        resource.storage > TT_IREE_SPATIAL_STORAGE_TILE_32X32_FACE ||
        resource.sharding > TT_IREE_SPATIAL_SHARDING_REPLICATED ||
        resource.lifetime > TT_IREE_SPATIAL_LIFETIME_SESSION ||
        !resource.shard_count || !resource.shards ||
        resource.shard_count > TT_IREE_SPATIAL_MAX_SHARDS - total_shards) {
      return Invalid("invalid resource binding, storage, lifetime or shards");
    }
    total_shards += resource.shard_count;
    if ((resource.lifetime == TT_IREE_SPATIAL_LIFETIME_SESSION &&
         (!resource.resource_key ||
          !resource_keys.insert(resource.resource_key).second)) ||
        (resource.lifetime == TT_IREE_SPATIAL_LIFETIME_DISPATCH &&
         resource.resource_key)) {
      return Invalid("session resources need unique nonzero keys only");
    }
    if ((resource.memory_space != TT_IREE_SPATIAL_MEMORY_DRAM &&
         resource.memory_space != TT_IREE_SPATIAL_MEMORY_L1) ||
        (resource.memory_space == TT_IREE_SPATIAL_MEMORY_DRAM &&
         resource.sharding != TT_IREE_SPATIAL_SHARDING_NONE) ||
        (resource.sharding == TT_IREE_SPATIAL_SHARDING_NONE &&
         resource.shard_count != 1)) {
      return Invalid("invalid memory space or unsharded resource");
    }
    std::vector<std::pair<uint64_t, uint64_t>> ranges;
    std::unordered_set<uint32_t> replica_cores;
    const uint64_t storage_unit =
        resource.storage == TT_IREE_SPATIAL_STORAGE_TILE_32X32_FACE
            ? uint64_t(element_bytes) * 32 * 32
            : element_bytes;
    if (resource.byte_length % storage_unit)
      return Invalid("storage length is not a complete element/tile");
    for (iree_host_size_t j = 0; j < resource.shard_count; ++j) {
      const auto &shard = resource.shards[j];
      if (!shard.byte_length ||
          !Fits(shard.logical_byte_offset, shard.byte_length,
                resource.byte_length) ||
          shard.logical_byte_offset % storage_unit ||
          shard.byte_length % storage_unit) {
        return Invalid(
            "shard range is out of bounds or splits an element/tile");
      }
      if (resource.memory_space == TT_IREE_SPATIAL_MEMORY_DRAM) {
        if (shard.core != TT_IREE_SPATIAL_NO_CORE || shard.arena_byte_offset)
          return Invalid("DRAM endpoint must not encode a worker L1 address");
      } else {
        if (shard.core >= plan->core_count ||
            shard.arena_byte_offset % resource.alignment ||
            !Fits(shard.arena_byte_offset, shard.byte_length,
                  plan->l1_bytes_per_core)) {
          return Invalid(
              "L1 shard exceeds its logical core/arena or alignment");
        }
        l1_uses.push_back({shard.core, shard.arena_byte_offset,
                           shard.byte_length,
                           resource.lifetime == TT_IREE_SPATIAL_LIFETIME_SESSION
                               ? 0u
                               : resource.first_epoch,
                           resource.lifetime == TT_IREE_SPATIAL_LIFETIME_SESSION
                               ? plan->epoch_count
                               : resource.last_epoch});
      }
      if (resource.sharding == TT_IREE_SPATIAL_SHARDING_REPLICATED) {
        if (shard.logical_byte_offset ||
            shard.byte_length != resource.byte_length ||
            !replica_cores.insert(shard.core).second)
          return Invalid(
              "replicas must cover the whole resource on distinct cores");
      } else {
        ranges.emplace_back(shard.logical_byte_offset, shard.byte_length);
      }
    }
    if (resource.sharding != TT_IREE_SPATIAL_SHARDING_REPLICATED) {
      std::sort(ranges.begin(), ranges.end());
      uint64_t next = 0;
      for (const auto &range : ranges) {
        if (range.first != next)
          return Invalid("partition has a hole or overlapping logical shards");
        next += range.second;
      }
      if (next != resource.byte_length)
        return Invalid("shards do not cover the resource");
    }
  }
  // Intervals allow safe intra-dispatch arena reuse, but SESSION storage is
  // pinned even outside its local access interval. Aliases are not supported.
  std::sort(l1_uses.begin(), l1_uses.end(), [](const L1Use &a, const L1Use &b) {
    return std::make_pair(a.core, a.offset) < std::make_pair(b.core, b.offset);
  });
  for (size_t i = 0; i < l1_uses.size(); ++i) {
    const auto &a = l1_uses[i];
    for (size_t j = i + 1; j < l1_uses.size(); ++j) {
      const auto &b = l1_uses[j];
      if (b.core != a.core || b.offset >= a.offset + a.length)
        break;
      if (Overlaps(a.offset, a.length, b.offset, b.length) && a.begin < b.end &&
          b.begin < a.end)
        return Invalid("simultaneously live L1 shards overlap");
    }
  }
  for (iree_host_size_t i = 0; i < plan->route_count; ++i) {
    const auto &route = plan->routes[i];
    if (route.transfer > TT_IREE_SPATIAL_TRANSFER_MULTICAST || route.noc > 1 ||
        route.epoch >= plan->epoch_count || !route.byte_length ||
        route.source_resource >= plan->resource_count ||
        route.destination_resource >= plan->resource_count ||
        !route.destination_count || !route.destination_shards ||
        route.destination_count > plan->core_count ||
        route.dependency_count > i ||
        (route.dependency_count && !route.dependencies)) {
      return Invalid("invalid NoC route fields or references");
    }
    const auto &source = plan->resources[route.source_resource];
    const auto &destination = plan->resources[route.destination_resource];
    if (route.source_shard >= source.shard_count ||
        route.epoch < source.first_epoch || route.epoch >= source.last_epoch ||
        route.epoch < destination.first_epoch ||
        route.epoch >= destination.last_epoch ||
        !Fits(route.source_byte_offset, route.byte_length,
              source.shards[route.source_shard].byte_length)) {
      return Invalid("NoC source/access lifetime is out of bounds");
    }
    if ((route.transfer == TT_IREE_SPATIAL_TRANSFER_READ &&
         destination.memory_space != TT_IREE_SPATIAL_MEMORY_L1) ||
        (route.transfer == TT_IREE_SPATIAL_TRANSFER_WRITE &&
         source.memory_space != TT_IREE_SPATIAL_MEMORY_L1) ||
        (route.transfer == TT_IREE_SPATIAL_TRANSFER_MULTICAST &&
         (source.memory_space != TT_IREE_SPATIAL_MEMORY_L1 ||
          destination.memory_space != TT_IREE_SPATIAL_MEMORY_L1 ||
          route.destination_count < 2)) ||
        (route.transfer != TT_IREE_SPATIAL_TRANSFER_MULTICAST &&
         route.destination_count != 1)) {
      return Invalid("NoC operation does not match its endpoint spaces/counts");
    }
    std::unordered_set<uint32_t> receivers;
    for (iree_host_size_t j = 0; j < route.destination_count; ++j) {
      const uint32_t shard = route.destination_shards[j];
      if (shard >= destination.shard_count ||
          !receivers.insert(destination.shards[shard].core).second ||
          !Fits(route.destination_byte_offset, route.byte_length,
                destination.shards[shard].byte_length)) {
        return Invalid("NoC destination is duplicate or out of bounds");
      }
    }
    std::unordered_set<uint32_t> dependencies;
    for (iree_host_size_t j = 0; j < route.dependency_count; ++j) {
      uint32_t dependency = route.dependencies[j];
      if (dependency >= i || !dependencies.insert(dependency).second ||
          plan->routes[dependency].epoch > route.epoch) {
        return Invalid("NoC dependencies must be unique earlier routes/epochs");
      }
    }
  }
  return iree_ok_status();
}

iree_status_t tt_iree_validate_spatial_plan_table(const void *table,
                                                  uint16_t binding_count) {
  if (!table)
    return Invalid("missing table");
  auto fb = static_cast<ns(SpatialPlanDef_table_t)>(table);
  auto fb_cores = ns(SpatialPlanDef_cores(fb));
  auto fb_resources = ns(SpatialPlanDef_resources(fb));
  auto fb_routes = ns(SpatialPlanDef_routes(fb));
  size_t nc = ns(SpatialCoreDef_vec_len(fb_cores));
  size_t nr = ns(SpatialResourceDef_vec_len(fb_resources));
  size_t nt = ns(SpatialRouteDef_vec_len(fb_routes));
  if (!nc || nc > TT_IREE_SPATIAL_MAX_CORES || !nr ||
      nr > TT_IREE_SPATIAL_MAX_RESOURCES || nt > TT_IREE_SPATIAL_MAX_ROUTES)
    return Invalid("wire plan exceeds descriptor limits");
  std::vector<tt_iree_spatial_core_t> cores(nc);
  std::vector<tt_iree_spatial_resource_t> resources(nr);
  std::vector<std::vector<tt_iree_spatial_shard_t>> shards(nr);
  std::vector<tt_iree_spatial_route_t> routes(nt);
  std::vector<std::vector<uint32_t>> receivers(nt), dependencies(nt);
  for (size_t i = 0; i < nc; ++i) {
    auto c = ns(SpatialCoreDef_vec_at(fb_cores, i));
    if (!c)
      return Invalid("null core table");
    cores[i] = {ns(SpatialCoreDef_x(c)), ns(SpatialCoreDef_y(c)),
                ns(SpatialCoreDef_kernel_roles(c))};
  }
  size_t total_shards = 0;
  for (size_t i = 0; i < nr; ++i) {
    auto r = ns(SpatialResourceDef_vec_at(fb_resources, i));
    if (!r)
      return Invalid("null resource table");
    auto s = ns(SpatialResourceDef_shards(r));
    size_t count = ns(SpatialShardDef_vec_len(s));
    if (!count || count > TT_IREE_SPATIAL_MAX_SHARDS - total_shards)
      return Invalid("wire plan exceeds shard limits");
    total_shards += count;
    shards[i].resize(count);
    for (size_t j = 0; j < count; ++j) {
      auto value = ns(SpatialShardDef_vec_at(s, j));
      if (!value)
        return Invalid("null shard table");
      shards[i][j] = {ns(SpatialShardDef_core(value)),
                      ns(SpatialShardDef_logical_byte_offset(value)),
                      ns(SpatialShardDef_byte_length(value)),
                      ns(SpatialShardDef_arena_byte_offset(value))};
    }
    resources[i] = {ns(SpatialResourceDef_binding(r)),
                    ns(SpatialResourceDef_binding_byte_offset(r)),
                    ns(SpatialResourceDef_byte_length(r)),
                    ns(SpatialResourceDef_element_type(r)),
                    ns(SpatialResourceDef_storage(r)),
                    ns(SpatialResourceDef_memory_space(r)),
                    ns(SpatialResourceDef_sharding(r)),
                    ns(SpatialResourceDef_lifetime(r)),
                    ns(SpatialResourceDef_resource_key(r)),
                    ns(SpatialResourceDef_first_epoch(r)),
                    ns(SpatialResourceDef_last_epoch(r)),
                    ns(SpatialResourceDef_alignment(r)),
                    count,
                    shards[i].data()};
  }
  for (size_t i = 0; i < nt; ++i) {
    auto r = ns(SpatialRouteDef_vec_at(fb_routes, i));
    if (!r)
      return Invalid("null route table");
    auto d = ns(SpatialRouteDef_destination_shards(r));
    auto w = ns(SpatialRouteDef_dependencies(r));
    size_t nd = flatbuffers_uint32_vec_len(d),
           nw = flatbuffers_uint32_vec_len(w);
    if (nd > nc || nw > i)
      return Invalid("wire route exceeds reference limits");
    receivers[i].resize(nd);
    dependencies[i].resize(nw);
    for (size_t j = 0; j < nd; ++j)
      receivers[i][j] = flatbuffers_uint32_vec_at(d, j);
    for (size_t j = 0; j < nw; ++j)
      dependencies[i][j] = flatbuffers_uint32_vec_at(w, j);
    routes[i] = {ns(SpatialRouteDef_transfer(r)),
                 ns(SpatialRouteDef_noc(r)),
                 ns(SpatialRouteDef_epoch(r)),
                 ns(SpatialRouteDef_source_resource(r)),
                 ns(SpatialRouteDef_source_shard(r)),
                 ns(SpatialRouteDef_source_byte_offset(r)),
                 ns(SpatialRouteDef_destination_resource(r)),
                 nd,
                 receivers[i].data(),
                 ns(SpatialRouteDef_destination_byte_offset(r)),
                 ns(SpatialRouteDef_byte_length(r)),
                 nw,
                 dependencies[i].data()};
  }
  tt_iree_spatial_plan_t plan = {ns(SpatialPlanDef_epoch_count(fb)),
                                 ns(SpatialPlanDef_l1_bytes_per_core(fb)),
                                 nc,
                                 cores.data(),
                                 nr,
                                 resources.data(),
                                 nt,
                                 routes.data()};
  return tt_iree_validate_spatial_plan(&plan, binding_count);
}
