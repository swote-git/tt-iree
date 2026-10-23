// Copyright 2026 The tt-iree Authors
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include <array>
#include <functional>

#include "flatcc/flatcc_builder.h"
#include "iree/hal/drivers/tenstorrent/tt_executable.h"
#include "iree/schema/tt_executable_builder_util.h"
#include "iree/testing/gtest.h"
#include "iree/testing/status_matchers.h"
#include "tt_executable_def_builder.h"
#include "tt_executable_def_reader.h"
#include "tt_executable_def_verifier.h"

#define ns(x) FLATBUFFERS_WRAP_NAMESPACE(iree_tenstorrent_hal, x)

namespace {
constexpr uint32_t kRoles = (1u << 1) | (1u << 2) | (1u << 3);

struct PlanFixture {
  std::array<tt_iree_spatial_core_t, 3> cores = {
      {{0, 0, kRoles}, {1, 0, kRoles}, {2, 0, kRoles}}};
  tt_iree_spatial_shard_t dram = {TT_IREE_SPATIAL_NO_CORE, 0, 2048, 0};
  tt_iree_spatial_shard_t sender = {0, 0, 2048, 0};
  std::array<tt_iree_spatial_shard_t, 2> replicas = {
      {{1, 0, 2048, 0}, {2, 0, 2048, 0}}};
  std::array<tt_iree_spatial_shard_t, 2> partitions = {
      {{1, 0, 1024, 2048}, {2, 1024, 1024, 2048}}};
  std::array<tt_iree_spatial_resource_t, 4> resources = {};
  uint32_t single_receiver = 0;
  std::array<uint32_t, 2> multicast_receivers = {{0, 1}};
  uint32_t dependency = 0;
  std::array<tt_iree_spatial_route_t, 2> routes = {};
  tt_iree_spatial_plan_t plan = {};

  PlanFixture() {
    for (auto &r : resources) {
      r.binding = TT_IREE_SPATIAL_NO_BINDING;
      r.byte_length = 2048;
      r.element_type = TT_IREE_TTEX_TENSOR_ELEMENT_TYPE_BF16;
      r.storage = TT_IREE_SPATIAL_STORAGE_TILE_32X32_FACE;
      r.memory_space = TT_IREE_SPATIAL_MEMORY_L1;
      r.last_epoch = 3;
      r.alignment = 32;
      r.shard_count = 1;
    }
    resources[0].binding = 0;
    resources[0].memory_space = TT_IREE_SPATIAL_MEMORY_DRAM;
    resources[0].shards = &dram;
    resources[1].shards = &sender;
    resources[2].binding = 1;
    resources[2].sharding = TT_IREE_SPATIAL_SHARDING_REPLICATED;
    resources[2].first_epoch = 1;
    resources[2].shard_count = replicas.size();
    resources[2].shards = replicas.data();
    resources[3].storage = TT_IREE_SPATIAL_STORAGE_LINEAR;
    resources[3].sharding = TT_IREE_SPATIAL_SHARDING_PARTITIONED;
    resources[3].shard_count = partitions.size();
    resources[3].shards = partitions.data();
    routes[0] = {TT_IREE_SPATIAL_TRANSFER_READ,
                 0,
                 0,
                 0,
                 0,
                 0,
                 1,
                 1,
                 &single_receiver,
                 0,
                 2048,
                 0,
                 nullptr};
    routes[1] = {TT_IREE_SPATIAL_TRANSFER_MULTICAST,
                 1,
                 1,
                 1,
                 0,
                 0,
                 2,
                 multicast_receivers.size(),
                 multicast_receivers.data(),
                 0,
                 2048,
                 1,
                 &dependency};
    plan = {3,
            4096,
            cores.size(),
            cores.data(),
            resources.size(),
            resources.data(),
            routes.size(),
            routes.data()};
  }

  tt_iree_ttex_entry_point_desc_t Entry() const {
    tt_iree_ttex_entry_point_desc_t ep = {};
    ep.name = "spatial_contract";
    ep.binding_count = 3;
    ep.workgroup_size[0] = ep.workgroup_size[1] = ep.workgroup_size[2] = 1;
    ep.spatial_plan = &plan;
    return ep;
  }
};

void ExpectInvalid(iree_status_t status) {
  EXPECT_EQ(iree_status_code(status), IREE_STATUS_INVALID_ARGUMENT);
  iree_status_ignore(status);
}

void ExpectRuntimeStatus(iree_byte_span_t data, iree_status_code_t expected) {
  iree_hal_executable_params_t params;
  iree_hal_executable_params_initialize(&params);
  params.executable_format = iree_make_cstring_view("tenstorrent-ttex-fb");
  params.executable_data =
      iree_make_const_byte_span(data.data, data.data_length);
  iree_hal_executable_t *executable = nullptr;
  // A sentinel proves preflight never touches the device or creates a program.
  uint64_t unused_device = 0;
  auto status = iree_hal_tt_executable_create(
      reinterpret_cast<iree_hal_device_t *>(&unused_device), &params,
      iree_allocator_system(), &executable);
  EXPECT_EQ(iree_status_code(status), expected);
  iree_status_ignore(status);
  EXPECT_EQ(executable, nullptr);
}

// Bypass the checked descriptor builder to exercise the production parser.
iree_byte_span_t RawWire(uint32_t version, bool with_plan,
                         uint32_t shard_core) {
  flatcc_builder_t builder;
  flatcc_builder_init(&builder);
  auto core = ns(SpatialCoreDef_create)(&builder, 0, 0, kRoles);
  auto cores = ns(SpatialCoreDef_vec_create)(&builder, &core, 1);
  auto shard = ns(SpatialShardDef_create)(&builder, shard_core, 0, 2048, 0);
  auto shards = ns(SpatialShardDef_vec_create)(&builder, &shard, 1);
  auto resource = ns(SpatialResourceDef_create)(
      &builder, TT_IREE_SPATIAL_NO_BINDING, 0, 2048,
      TT_IREE_TTEX_TENSOR_ELEMENT_TYPE_BF16, TT_IREE_SPATIAL_STORAGE_LINEAR,
      TT_IREE_SPATIAL_MEMORY_L1, TT_IREE_SPATIAL_SHARDING_NONE,
      TT_IREE_SPATIAL_LIFETIME_DISPATCH, 0, 0, 1, 32, shards);
  auto resources = ns(SpatialResourceDef_vec_create)(&builder, &resource, 1);
  auto routes = ns(SpatialRouteDef_vec_create)(&builder, nullptr, 0);
  auto plan =
      ns(SpatialPlanDef_create)(&builder, 1, 4096, cores, resources, routes);
  EXPECT_NE(plan, 0);
  ns(ExecutableDef_start_as_root)(&builder);
  ns(ExecutableDef_version_add)(&builder, version);
  ns(ExecutableDef_entry_points_start)(&builder);
  ns(ExecutableDef_entry_points_push_start)(&builder);
  ns(EntryPointDef_binding_count_add)(&builder, 3);
  if (with_plan)
    ns(EntryPointDef_spatial_plan_add)(&builder, plan);
  ns(ExecutableDef_entry_points_push_end)(&builder);
  ns(ExecutableDef_entry_points_end)(&builder);
  ns(ExecutableDef_end_as_root)(&builder);
  size_t size = 0;
  void *bytes = flatcc_builder_finalize_aligned_buffer(&builder, &size);
  iree_byte_span_t result = {};
  auto status = iree_allocator_clone(iree_allocator_system(),
                                     iree_make_const_byte_span(bytes, size),
                                     reinterpret_cast<void **>(&result.data));
  EXPECT_TRUE(iree_status_is_ok(status));
  iree_status_ignore(status);
  result.data_length = size;
  flatcc_builder_aligned_free(bytes);
  flatcc_builder_clear(&builder);
  return result;
}

TEST(SpatialPlanTest, RoundTripAllFiveContracts) {
  PlanFixture fixture;
  fixture.resources[2].lifetime = TT_IREE_SPATIAL_LIFETIME_SESSION;
  fixture.resources[2].resource_key = 0x123456789ull;
  auto ep = fixture.Entry();
  iree_byte_span_t data = {};
  IREE_ASSERT_OK(tt_iree_build_ttex_executable_def_with_version(
      iree_allocator_system(), TT_IREE_TTEX_VERSION_19, 1, &ep, &data));
  ASSERT_EQ(ns(ExecutableDef_verify_as_root)(data.data, data.data_length), 0);
  auto root = ns(ExecutableDef_as_root)(data.data);
  EXPECT_EQ(ns(ExecutableDef_version)(root), TT_IREE_TTEX_VERSION_19);
  auto entry =
      ns(EntryPointDef_vec_at)(ns(ExecutableDef_entry_points)(root), 0);
  auto plan = ns(EntryPointDef_spatial_plan)(entry);
  ASSERT_NE(plan, nullptr);
  IREE_ASSERT_OK(tt_iree_validate_spatial_plan_table(plan, ep.binding_count));
  EXPECT_EQ(ns(SpatialPlanDef_epoch_count)(plan), 3u);
  EXPECT_EQ(ns(SpatialPlanDef_l1_bytes_per_core)(plan), 4096u);
  auto cores = ns(SpatialPlanDef_cores)(plan);
  EXPECT_EQ(ns(SpatialCoreDef_x)(ns(SpatialCoreDef_vec_at)(cores, 2)), 2u);
  EXPECT_EQ(
      ns(SpatialCoreDef_kernel_roles)(ns(SpatialCoreDef_vec_at)(cores, 0)),
      kRoles);
  auto resources = ns(SpatialPlanDef_resources)(plan);
  auto replica = ns(SpatialResourceDef_vec_at)(resources, 2);
  EXPECT_EQ(ns(SpatialResourceDef_memory_space)(replica),
            TT_IREE_SPATIAL_MEMORY_L1);
  EXPECT_EQ(ns(SpatialResourceDef_sharding)(replica),
            TT_IREE_SPATIAL_SHARDING_REPLICATED);
  EXPECT_EQ(ns(SpatialResourceDef_storage)(replica),
            TT_IREE_SPATIAL_STORAGE_TILE_32X32_FACE);
  EXPECT_EQ(ns(SpatialResourceDef_lifetime)(replica),
            TT_IREE_SPATIAL_LIFETIME_SESSION);
  EXPECT_EQ(ns(SpatialResourceDef_resource_key)(replica), 0x123456789ull);
  auto partition = ns(SpatialResourceDef_vec_at)(resources, 3);
  auto shard =
      ns(SpatialShardDef_vec_at)(ns(SpatialResourceDef_shards)(partition), 1);
  EXPECT_EQ(ns(SpatialShardDef_logical_byte_offset)(shard), 1024u);
  EXPECT_EQ(ns(SpatialShardDef_arena_byte_offset)(shard), 2048u);
  auto route = ns(SpatialRouteDef_vec_at)(ns(SpatialPlanDef_routes)(plan), 1);
  EXPECT_EQ(ns(SpatialRouteDef_noc)(route), 1u);
  EXPECT_EQ(ns(SpatialRouteDef_transfer)(route),
            TT_IREE_SPATIAL_TRANSFER_MULTICAST);
  EXPECT_EQ(
      flatbuffers_uint32_vec_len(ns(SpatialRouteDef_destination_shards)(route)),
      2u);
  EXPECT_EQ(
      flatbuffers_uint32_vec_at(ns(SpatialRouteDef_dependencies)(route), 0),
      0u);
  ExpectRuntimeStatus(data, IREE_STATUS_UNIMPLEMENTED);
  iree_allocator_free(iree_allocator_system(), data.data);
}

TEST(SpatialPlanTest, RequiresExplicitVersionAndCompletePlans) {
  PlanFixture fixture;
  auto ep = fixture.Entry();
  iree_byte_span_t data = {};
  ExpectInvalid(tt_iree_build_ttex_executable_def(iree_allocator_system(), 1,
                                                  &ep, &data));
  EXPECT_EQ(data.data, nullptr);
  ep.spatial_plan = nullptr;
  ExpectInvalid(tt_iree_build_ttex_executable_def_with_version(
      iree_allocator_system(), TT_IREE_TTEX_VERSION_19, 1, &ep, &data));
  IREE_ASSERT_OK(tt_iree_build_ttex_executable_def(iree_allocator_system(), 1,
                                                   &ep, &data));
  auto root = ns(ExecutableDef_as_root)(data.data);
  EXPECT_EQ(ns(ExecutableDef_version)(root), TT_IREE_TTEX_VERSION_18);
  EXPECT_EQ(ns(EntryPointDef_spatial_plan)(ns(EntryPointDef_vec_at)(
                ns(ExecutableDef_entry_points)(root), 0)),
            nullptr);
  iree_allocator_free(iree_allocator_system(), data.data);
}

TEST(SpatialPlanTest, RejectsInvalidDescriptors) {
  const std::function<void(PlanFixture &)> mutations[] = {
      [](auto &f) { f.plan.epoch_count = 0; },
      [](auto &f) { f.plan.core_count = TT_IREE_SPATIAL_MAX_CORES + 1; },
      [](auto &f) { f.plan.cores = nullptr; },
      [](auto &f) { f.cores[1] = f.cores[0]; },
      [](auto &f) { f.cores[0].kernel_roles = 1; },
      [](auto &f) { f.resources[0].binding = 3; },
      [](auto &f) { f.resources[0].binding_byte_offset = UINT64_MAX - 1; },
      [](auto &f) { f.resources[0].element_type = 255; },
      [](auto &f) { f.resources[0].memory_space = 255; },
      [](auto &f) { f.resources[0].storage = 255; },
      [](auto &f) { f.resources[0].sharding = 255; },
      [](auto &f) { f.resources[0].lifetime = 255; },
      [](auto &f) { f.resources[0].alignment = 3; },
      [](auto &f) { f.resources[0].shards = nullptr; },
      [](auto &f) {
        f.resources[0].shard_count = TT_IREE_SPATIAL_MAX_SHARDS + 1;
      },
      [](auto &f) { f.resources[0].first_epoch = 3; },
      [](auto &f) { f.resources[0].last_epoch = 4; },
      [](auto &f) { f.resources[0].resource_key = 1; },
      [](auto &f) {
        f.resources[0].lifetime = TT_IREE_SPATIAL_LIFETIME_SESSION;
      },
      [](auto &f) { f.dram.core = 0; },
      [](auto &f) { f.sender.core = 3; },
      [](auto &f) {
        f.sender.arena_byte_offset = 32;
        f.plan.l1_bytes_per_core = 2048;
      },
      [](auto &f) { f.sender.arena_byte_offset = 1; },
      [](auto &f) { f.sender.byte_length = 2046; },
      [](auto &f) { f.replicas[1].core = f.replicas[0].core; },
      [](auto &f) { f.partitions[1].logical_byte_offset = 1022; },
      [](auto &f) { f.partitions[1].byte_length = 1022; },
      [](auto &f) { f.partitions[0].arena_byte_offset = 0; },
      [](auto &f) { f.routes[0].transfer = 255; },
      [](auto &f) { f.routes[0].noc = 2; },
      [](auto &f) { f.routes[0].source_resource = 4; },
      [](auto &f) { f.routes[0].source_shard = 1; },
      [](auto &f) { f.routes[0].source_byte_offset = UINT64_MAX; },
      [](auto &f) { f.routes[0].destination_byte_offset = 2; },
      [](auto &f) { f.routes[0].destination_resource = 4; },
      [](auto &f) { f.routes[0].epoch = 3; },
      [](auto &f) { f.routes[1].epoch = 0; },
      [](auto &f) { f.routes[1].byte_length = 2049; },
      [](auto &f) { f.routes[1].destination_count = 1; },
      [](auto &f) { f.multicast_receivers[1] = 0; },
      [](auto &f) { f.multicast_receivers[1] = 2; },
      [](auto &f) { f.dependency = 1; },
      [](auto &f) { f.routes[1].dependencies = nullptr; },
  };
  for (size_t i = 0; i < std::size(mutations); ++i) {
    SCOPED_TRACE(i);
    PlanFixture fixture;
    mutations[i](fixture);
    ExpectInvalid(tt_iree_validate_spatial_plan(&fixture.plan, 3));
    auto ep = fixture.Entry();
    iree_byte_span_t data = {};
    ExpectInvalid(tt_iree_build_ttex_executable_def_with_version(
        iree_allocator_system(), TT_IREE_TTEX_VERSION_19, 1, &ep, &data));
    EXPECT_EQ(data.data, nullptr);
  }
}

TEST(SpatialPlanTest, ArenaReuseRespectsLifetimeAndSessionPinning) {
  PlanFixture fixture;
  fixture.plan.route_count = 0;
  fixture.resources[2].first_epoch = 0;
  fixture.resources[2].last_epoch = 1;
  fixture.resources[3].first_epoch = 1;
  for (auto &shard : fixture.partitions)
    shard.arena_byte_offset = 0;
  IREE_ASSERT_OK(tt_iree_validate_spatial_plan(&fixture.plan, 3));
  fixture.resources[2].lifetime = TT_IREE_SPATIAL_LIFETIME_SESSION;
  fixture.resources[2].resource_key = 7;
  ExpectInvalid(tt_iree_validate_spatial_plan(&fixture.plan, 3));
}

TEST(SpatialPlanTest, SupportsBoundedWriteAfterRead) {
  PlanFixture fixture;
  fixture.routes[1] = {TT_IREE_SPATIAL_TRANSFER_WRITE,
                       1,
                       1,
                       1,
                       0,
                       0,
                       0,
                       1,
                       &fixture.single_receiver,
                       0,
                       2048,
                       1,
                       &fixture.dependency};
  IREE_ASSERT_OK(tt_iree_validate_spatial_plan(&fixture.plan, 3));
}

TEST(SpatialPlanTest, RuntimeRejectsMissingLegacyAndMalformedWirePlans) {
  for (auto item : {std::make_pair(TT_IREE_TTEX_VERSION_18, true),
                    std::make_pair(TT_IREE_TTEX_VERSION_19, false)}) {
    SCOPED_TRACE(item.first);
    auto data = RawWire(item.first, item.second, 0);
    EXPECT_EQ(ns(ExecutableDef_verify_as_root)(data.data, data.data_length), 0);
    auto root = ns(ExecutableDef_as_root)(data.data);
    ASSERT_EQ(ns(ExecutableDef_version)(root), item.first);
    auto entry =
        ns(EntryPointDef_vec_at)(ns(ExecutableDef_entry_points)(root), 0);
    ASSERT_EQ(ns(EntryPointDef_spatial_plan)(entry) != nullptr, item.second);
    ExpectRuntimeStatus(data, IREE_STATUS_INVALID_ARGUMENT);
    iree_allocator_free(iree_allocator_system(), data.data);
  }
  auto data = RawWire(TT_IREE_TTEX_VERSION_19, true, 99);
  ASSERT_EQ(ns(ExecutableDef_verify_as_root)(data.data, data.data_length), 0);
  ExpectRuntimeStatus(data, IREE_STATUS_INVALID_ARGUMENT);
  iree_allocator_free(iree_allocator_system(), data.data);
}
} // namespace
