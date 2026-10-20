// Copyright 2026 The tt-iree Authors
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include <algorithm>
#include <chrono>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <string>

#include "TenstorrentAOTCompiler.h"
#include "iree/schema/tt_executable_builder_util.h"
#include "iree/testing/gtest.h"
#include "llrt/tt_elffile.hpp"

namespace {
using namespace mlir::iree_compiler::IREE::HAL;
namespace fs = std::filesystem;

struct TemporaryFiles {
  fs::path root =
      fs::temp_directory_path() /
      ("tt_iree_elf_test_" +
       std::to_string(
           std::chrono::steady_clock::now().time_since_epoch().count()));
  TemporaryFiles() { fs::create_directories(root); }
  ~TemporaryFiles() {
    std::error_code ec;
    fs::remove_all(root, ec);
  }
};

struct KeepDebug {
  std::string old;
  bool had_old = false;
  KeepDebug() {
    if (const char *value = std::getenv("TT_IREE_AOT_KEEP_DEBUG")) {
      had_old = true;
      old = value;
    }
    ::setenv("TT_IREE_AOT_KEEP_DEBUG", "1", 1);
  }
  ~KeepDebug() {
    if (had_old)
      ::setenv("TT_IREE_AOT_KEEP_DEBUG", old.c_str(), 1);
    else
      ::unsetenv("TT_IREE_AOT_KEEP_DEBUG");
  }
};

TEST(AotArtifactTest, StrippingDebugPreservesMetalLoadedAndRelocatedImages) {
  KeepDebug keep_debug;
  TenstorrentAOTKernelBundle bundle;
  std::string error;
  ASSERT_TRUE(
      compilePaddedCopyAOT(TenstorrentAOTTargetArch::Blackhole, bundle, error))
      << error;
  ASSERT_FALSE(bundle.artifacts.empty());
  TemporaryFiles files;
  size_t original_bytes = 0, compact_bytes = 0;
  for (size_t i = 0; i < bundle.artifacts.size(); ++i) {
    SCOPED_TRACE(bundle.artifacts[i].path);
    const auto &artifact = bundle.artifacts[i];
    const std::string original =
        (files.root / (std::to_string(i) + ".elf")).string();
    const std::string compact =
        (files.root / (std::to_string(i) + ".compact.elf")).string();
    {
      std::ofstream output(original, std::ios::binary);
      output.write(reinterpret_cast<const char *>(artifact.data.data()),
                   artifact.data.size());
      ASSERT_TRUE(output.good());
    }
    ASSERT_TRUE(stripTenstorrentAOTArtifactDebugInfo(original, compact, error))
        << error;
    original_bytes += fs::file_size(original);
    compact_bytes += fs::file_size(compact);
    ASSERT_EQ(fs::file_size(original), artifact.data.size());
    for (bool relocate : {false, true}) {
      ll_api::ElfFile before, after;
      before.ReadImage(original);
      after.ReadImage(compact);
      if (relocate) {
        before.MakeExecuteInPlace();
        after.MakeExecuteInPlace();
      }
      const auto &a = before.GetSegments(), &b = after.GetSegments();
      ASSERT_EQ(a.size(), b.size());
      for (size_t j = 0; j < a.size(); ++j) {
        EXPECT_EQ(a[j].address, b[j].address);
        EXPECT_EQ(a[j].lma, b[j].lma);
        EXPECT_EQ(a[j].membytes, b[j].membytes);
        EXPECT_EQ(a[j].relocs, b[j].relocs);
        EXPECT_TRUE(std::equal(a[j].contents.begin(), a[j].contents.end(),
                               b[j].contents.begin(), b[j].contents.end()));
      }
    }
  }
  EXPECT_LT(compact_bytes, original_bytes);
}

TEST(AotArtifactTest, RejectsInPlaceStripAndMissingInput) {
  TemporaryFiles files;
  std::string error;
  const auto path = (files.root / "missing.elf").string();
  EXPECT_FALSE(stripTenstorrentAOTArtifactDebugInfo(path, path, error));
  EXPECT_NE(error.find("distinct"), std::string::npos);
  EXPECT_FALSE(
      stripTenstorrentAOTArtifactDebugInfo(path, path + ".out", error));
  EXPECT_FALSE(error.empty());
}
} // namespace
