// Copyright 2025 The tt-iree Authors
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// Unit tests for TTEX FlatBuffer schema: build → verify → parse round-trip.
// These tests run without hardware (mock mode OK) and without the compiler.

#include <cstring>

#include "iree/base/api.h"
#include "iree/schema/tt_executable_builder_util.h"
#include "iree/testing/gtest.h"
#include "iree/testing/status_matchers.h"

// Generated flatcc headers
#include "tt_executable_def_reader.h"
#include "tt_executable_def_verifier.h"

#define ttex_ns(x) FLATBUFFERS_WRAP_NAMESPACE(iree_tenstorrent_hal, x)

namespace iree {
namespace hal {
namespace tenstorrent {
namespace {

TEST(TtexSchemaTest, BuildVerifySingleEntry) {
  iree_allocator_t alloc = iree_allocator_system();

  tt_iree_ttex_entry_point_desc_t ep = {};
  ep.name = "test_add";
  ep.constant_count = 0;
  ep.binding_count = 3;
  ep.flags = 0;
  ep.workgroup_size[0] = 1;
  ep.workgroup_size[1] = 1;
  ep.workgroup_size[2] = 1;
  ep.builtin_program = TT_IREE_TTEX_BUILTIN_PROGRAM_CUSTOM_SFPI_ADD;

  iree_byte_span_t data = {0};
  IREE_ASSERT_OK(tt_iree_build_ttex_executable_def(alloc, 1, &ep, &data));
  ASSERT_NE(data.data, nullptr);
  ASSERT_GT(data.data_length, 0u);

  int ret = ttex_ns(ExecutableDef_verify_as_root(data.data, data.data_length));
  ASSERT_EQ(ret, 0);

  ttex_ns(ExecutableDef_table_t) def = ttex_ns(ExecutableDef_as_root(data.data));
  ASSERT_EQ(ttex_ns(ExecutableDef_version(def)), 0u);

  ttex_ns(EntryPointDef_vec_t) eps = ttex_ns(ExecutableDef_entry_points(def));
  ASSERT_EQ(ttex_ns(EntryPointDef_vec_len(eps)), 1u);

  ttex_ns(EntryPointDef_table_t) ep0 = ttex_ns(EntryPointDef_vec_at(eps, 0));
  ASSERT_STREQ(ttex_ns(EntryPointDef_name(ep0)), "test_add");
  ASSERT_EQ(ttex_ns(EntryPointDef_binding_count(ep0)), 3);
  ASSERT_EQ(ttex_ns(EntryPointDef_constant_count(ep0)), 0);
  ASSERT_EQ(ttex_ns(EntryPointDef_workgroup_size_x(ep0)), 1u);
  ASSERT_EQ(ttex_ns(EntryPointDef_workgroup_size_y(ep0)), 1u);
  ASSERT_EQ(ttex_ns(EntryPointDef_workgroup_size_z(ep0)), 1u);
  ASSERT_EQ(ttex_ns(EntryPointDef_builtin(ep0)), 0u);

  iree_allocator_free(alloc, data.data);
}

TEST(TtexSchemaTest, BuildVerifyMultiEntry) {
  iree_allocator_t alloc = iree_allocator_system();

  tt_iree_ttex_entry_point_desc_t eps[2] = {};
  eps[0].name = "dispatch_0_add";
  eps[0].binding_count = 3;
  eps[0].builtin_program = TT_IREE_TTEX_BUILTIN_PROGRAM_CUSTOM_SFPI_ADD;
  eps[0].workgroup_size[0] = 1;
  eps[0].workgroup_size[1] = 1;
  eps[0].workgroup_size[2] = 1;

  eps[1].name = "dispatch_1_matmul";
  eps[1].binding_count = 3;
  eps[1].builtin_program = TT_IREE_TTEX_BUILTIN_PROGRAM_BF16_MATMUL_TILED;
  eps[1].workgroup_size[0] = 4;
  eps[1].workgroup_size[1] = 2;
  eps[1].workgroup_size[2] = 1;
  eps[1].builtin_m_tiles = 2;
  eps[1].builtin_n_tiles = 3;
  eps[1].builtin_k_tiles = 4;

  iree_byte_span_t data = {0};
  IREE_ASSERT_OK(tt_iree_build_ttex_executable_def(alloc, 2, eps, &data));

  int ret = ttex_ns(ExecutableDef_verify_as_root(data.data, data.data_length));
  ASSERT_EQ(ret, 0);

  ttex_ns(ExecutableDef_table_t) def = ttex_ns(ExecutableDef_as_root(data.data));
  ttex_ns(EntryPointDef_vec_t) parsed_eps = ttex_ns(ExecutableDef_entry_points(def));
  ASSERT_EQ(ttex_ns(EntryPointDef_vec_len(parsed_eps)), 2u);

  ttex_ns(EntryPointDef_table_t) ep1 = ttex_ns(EntryPointDef_vec_at(parsed_eps, 1));
  ASSERT_STREQ(ttex_ns(EntryPointDef_name(ep1)), "dispatch_1_matmul");
  ASSERT_EQ(ttex_ns(EntryPointDef_builtin(ep1)),
            TT_IREE_TTEX_BUILTIN_PROGRAM_BF16_MATMUL_TILED);
  ASSERT_EQ(ttex_ns(EntryPointDef_workgroup_size_x(ep1)), 4u);
  ASSERT_EQ(ttex_ns(EntryPointDef_workgroup_size_y(ep1)), 2u);
  ASSERT_EQ(ttex_ns(EntryPointDef_builtin_m_tiles(ep1)), 2u);
  ASSERT_EQ(ttex_ns(EntryPointDef_builtin_n_tiles(ep1)), 3u);
  ASSERT_EQ(ttex_ns(EntryPointDef_builtin_k_tiles(ep1)), 4u);

  iree_allocator_free(alloc, data.data);
}

TEST(TtexSchemaTest, VerifyRejectsGarbage) {
  uint8_t garbage[] = {0xDE, 0xAD, 0xBE, 0xEF, 0x00, 0x01, 0x02, 0x03};
  int ret = ttex_ns(ExecutableDef_verify_as_root(garbage, sizeof(garbage)));
  ASSERT_NE(ret, 0);
}

TEST(TtexSchemaTest, BuildEmpty) {
  iree_allocator_t alloc = iree_allocator_system();

  iree_byte_span_t data = {0};
  IREE_ASSERT_OK(tt_iree_build_ttex_executable_def(alloc, 0, NULL, &data));
  ASSERT_NE(data.data, nullptr);

  int ret = ttex_ns(ExecutableDef_verify_as_root(data.data, data.data_length));
  ASSERT_EQ(ret, 0);

  ttex_ns(ExecutableDef_table_t) def = ttex_ns(ExecutableDef_as_root(data.data));
  ttex_ns(EntryPointDef_vec_t) parsed_eps = ttex_ns(ExecutableDef_entry_points(def));
  ASSERT_EQ(ttex_ns(EntryPointDef_vec_len(parsed_eps)), 0u);

  iree_allocator_free(alloc, data.data);
}

}  // namespace
}  // namespace tenstorrent
}  // namespace hal
}  // namespace iree
