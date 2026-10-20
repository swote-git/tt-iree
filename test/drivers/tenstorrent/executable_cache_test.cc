// Copyright 2026 The tt-iree Authors
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/hal/drivers/tenstorrent/tt_executable.h"
#include "iree/hal/drivers/tenstorrent/tt_executable_cache.h"
#include "iree/schema/tt_executable_builder_util.h"
#include "utils.h"
#include <filesystem>
#include <fstream>

namespace iree::hal::tenstorrent {
namespace {

using ExecutableCacheTest = TenstorrentTestBase;

TEST_F(ExecutableCacheTest, ReusesIdenticalPreparedExecutable) {
#ifndef TT_IREE_ENABLE_MOCK
  GTEST_SKIP() << "metadata-only cache test runs in mock mode";
#else
  iree_hal_executable_cache_t *cache = nullptr;
  IREE_ASSERT_OK(iree_hal_tt_executable_cache_create(
      device_, iree_make_cstring_view("cache-test"), host_allocator(), &cache));

  tt_iree_ttex_entry_point_desc_t entry = {};
  entry.name = "cached_add";
  entry.binding_count = 3;
  entry.workgroup_size[0] = 1;
  entry.workgroup_size[1] = 1;
  entry.workgroup_size[2] = 1;
  entry.builtin_program = TT_IREE_TTEX_BUILTIN_PROGRAM_CUSTOM_SFPI_ADD;
  entry.dispatch_kind = TT_IREE_TTEX_DISPATCH_KIND_BUILTIN;

  iree_byte_span_t data = {};
  IREE_ASSERT_OK(
      tt_iree_build_ttex_executable_def(host_allocator(), 1, &entry, &data));

  iree_hal_executable_params_t params;
  iree_hal_executable_params_initialize(&params);
  params.executable_format =
      iree_make_cstring_view(IREE_HAL_TT_EXECUTABLE_FORMAT);
  params.executable_data =
      iree_make_const_byte_span(data.data, data.data_length);

  iree_hal_executable_t *first = nullptr;
  iree_hal_executable_t *second = nullptr;
  IREE_ASSERT_OK(
      iree_hal_executable_cache_prepare_executable(cache, &params, &first));
  IREE_ASSERT_OK(
      iree_hal_executable_cache_prepare_executable(cache, &params, &second));
  EXPECT_EQ(first, second);

  // A cache hit must compare content, not just reuse an input pointer.
  data.data[4] ^= 1; // Corrupt the TTEX identifier at the same address.
  iree_hal_executable_t *invalid = nullptr;
  iree_status_t status =
      iree_hal_executable_cache_prepare_executable(cache, &params, &invalid);
  EXPECT_EQ(iree_status_code(status), IREE_STATUS_INVALID_ARGUMENT);
  EXPECT_EQ(invalid, nullptr);
  iree_status_ignore(status);

  iree_hal_executable_release(first);
  iree_hal_executable_release(second);
  iree_hal_executable_cache_release(cache);
  iree_allocator_free(host_allocator(), data.data);
#endif
}

TEST_F(ExecutableCacheTest, CachingModeIsPartOfIdentity) {
#ifndef TT_IREE_ENABLE_MOCK
  GTEST_SKIP() << "metadata-only cache test runs in mock mode";
#else
  iree_hal_executable_cache_t *cache = nullptr;
  IREE_ASSERT_OK(iree_hal_tt_executable_cache_create(
      device_, iree_make_cstring_view("cache-mode-test"), host_allocator(),
      &cache));

  tt_iree_ttex_entry_point_desc_t entry = {};
  entry.name = "cached_add";
  entry.binding_count = 3;
  entry.workgroup_size[0] = 1;
  entry.workgroup_size[1] = 1;
  entry.workgroup_size[2] = 1;
  entry.builtin_program = TT_IREE_TTEX_BUILTIN_PROGRAM_CUSTOM_SFPI_ADD;
  entry.dispatch_kind = TT_IREE_TTEX_DISPATCH_KIND_BUILTIN;
  iree_byte_span_t data = {};
  IREE_ASSERT_OK(
      tt_iree_build_ttex_executable_def(host_allocator(), 1, &entry, &data));

  iree_hal_executable_params_t params;
  iree_hal_executable_params_initialize(&params);
  params.executable_format =
      iree_make_cstring_view(IREE_HAL_TT_EXECUTABLE_FORMAT);
  params.executable_data =
      iree_make_const_byte_span(data.data, data.data_length);
  iree_hal_executable_t *normal = nullptr;
  IREE_ASSERT_OK(
      iree_hal_executable_cache_prepare_executable(cache, &params, &normal));

  params.caching_mode = IREE_HAL_EXECUTABLE_CACHING_MODE_ENABLE_PROFILING;
  iree_hal_executable_t *profiled = nullptr;
  IREE_ASSERT_OK(
      iree_hal_executable_cache_prepare_executable(cache, &params, &profiled));
  EXPECT_NE(normal, profiled);

  iree_hal_executable_release(normal);
  iree_hal_executable_release(profiled);
  iree_hal_executable_cache_release(cache);
  iree_allocator_free(host_allocator(), data.data);
#endif
}

TEST_F(ExecutableCacheTest, SharedArtifactsMaterializeAllPathsAndCleanUp) {
#ifndef TT_IREE_ENABLE_MOCK
  GTEST_SKIP() << "metadata-only artifact test runs in mock mode";
#else
  tt_iree_ttex_entry_point_desc_t entry = {};
  entry.name = "shared_artifacts";
  entry.binding_count = 3;
  entry.builtin_program = TT_IREE_TTEX_BUILTIN_PROGRAM_CUSTOM_SFPI_ADD;
  const uint8_t bytes[] = {0x7f, 'E', 'L', 'F', 0, 1};
  tt_iree_ttex_kernel_source_desc_t kernels[] = {
      {TT_IREE_TTEX_KERNEL_ROLE_READER, "reader"},
      {TT_IREE_TTEX_KERNEL_ROLE_WRITER, "writer"},
      {TT_IREE_TTEX_KERNEL_ROLE_COMPUTE, "compute"}};
  tt_iree_ttex_artifact_file_desc_t artifacts[] = {
      {"kernel/one/brisc.elf", bytes, sizeof(bytes)},
      {"kernel/two/brisc.elf", bytes, sizeof(bytes)}};
  tt_iree_ttex_aot_kernel_bundle_desc_t bundle = {
      TT_IREE_TTEX_TARGET_ARCH_BLACKHOLE, IREE_ARRAYSIZE(kernels), kernels,
      IREE_ARRAYSIZE(artifacts), artifacts};
  entry.aot_kernel_bundle = &bundle;
  iree_byte_span_t data = {};
  IREE_ASSERT_OK(tt_iree_build_ttex_executable_def_with_version(
      host_allocator(), TT_IREE_TTEX_VERSION_3, 1, &entry, &data));
  iree_hal_executable_params_t params;
  iree_hal_executable_params_initialize(&params);
  params.executable_format =
      iree_make_cstring_view(IREE_HAL_TT_EXECUTABLE_FORMAT);
  params.executable_data =
      iree_make_const_byte_span(data.data, data.data_length);
  iree_hal_executable_t *executable = nullptr;
  IREE_ASSERT_OK(iree_hal_tt_executable_create(device_, &params,
                                               host_allocator(), &executable));
  const iree_hal_tt_kernel_params_t *kernel_params = nullptr;
  IREE_ASSERT_OK(iree_hal_tt_executable_lookup_kernel_params(executable, 0,
                                                             &kernel_params));
  const std::filesystem::path root(
      std::string(kernel_params->aot_artifact_root.data,
                  kernel_params->aot_artifact_root.size));
  const auto first = root / artifacts[0].path,
             second = root / artifacts[1].path;
  EXPECT_TRUE(std::filesystem::exists(first));
  EXPECT_TRUE(std::filesystem::exists(second));
  EXPECT_EQ(std::filesystem::file_size(first), sizeof(bytes));
  // Check inode sharing only when the temporary filesystem supports it.
  std::error_code link_error;
  std::filesystem::create_hard_link(first, root / "link-probe", link_error);
  if (!link_error)
    EXPECT_TRUE(std::filesystem::equivalent(first, second));
  std::ifstream stream(second, std::ios::binary);
  const std::string contents((std::istreambuf_iterator<char>(stream)), {});
  EXPECT_EQ(contents,
            std::string(reinterpret_cast<const char *>(bytes), sizeof(bytes)));
  iree_hal_executable_release(executable);
  EXPECT_FALSE(std::filesystem::exists(root));
  iree_allocator_free(host_allocator(), data.data);
#endif
}

} // namespace
} // namespace iree::hal::tenstorrent
