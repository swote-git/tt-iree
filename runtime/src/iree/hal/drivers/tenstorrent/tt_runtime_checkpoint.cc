// Copyright 2026 The tt-iree Authors
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/hal/drivers/tenstorrent/tt_runtime_checkpoint.h"

#include <cctype>
#include <cinttypes>
#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <limits>
#include <sstream>
#include <system_error>
#include <utility>

namespace iree::hal::tenstorrent {
namespace {

std::string JsonEscape(const std::string &value) {
  std::string escaped;
  escaped.reserve(value.size());
  for (char c : value) {
    switch (c) {
    case '\\':
      escaped += "\\\\";
      break;
    case '"':
      escaped += "\\\"";
      break;
    case '\n':
      escaped += "\\n";
      break;
    case '\r':
      escaped += "\\r";
      break;
    case '\t':
      escaped += "\\t";
      break;
    default:
      escaped.push_back(c);
      break;
    }
  }
  return escaped;
}

std::string SanitizeFileComponent(const std::string &value) {
  std::string sanitized;
  sanitized.reserve(value.size());
  for (unsigned char c : value) {
    sanitized.push_back(std::isalnum(c) || c == '-' || c == '_' ? c : '_');
  }
  if (sanitized.empty())
    return "dispatch";
  constexpr size_t kMaxComponentLength = 96;
  if (sanitized.size() > kMaxComponentLength)
    sanitized.resize(kMaxComponentLength);
  return sanitized;
}

iree_status_t CheckOutputStream(const std::ofstream &stream,
                                const std::filesystem::path &path) {
  if (stream.good())
    return iree_ok_status();
  return iree_make_status(IREE_STATUS_DATA_LOSS,
                          "failed to write checkpoint output '%s'",
                          path.c_str());
}

iree_status_t ValidateRecord(const RuntimeCheckpointRecord &record) {
  if (record.shape.empty()) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "checkpoint shape must not be empty");
  }
  uint64_t element_count = 1;
  for (uint32_t dimension : record.shape) {
    if (dimension == 0 ||
        element_count > std::numeric_limits<uint64_t>::max() / dimension) {
      return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                              "checkpoint shape has an invalid dimension");
    }
    element_count *= dimension;
  }
  if (element_count != record.values.size()) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "checkpoint shape requires %" PRIu64
                            " values but received %zu",
                            element_count, record.values.size());
  }
  return iree_ok_status();
}

} // namespace

iree_status_t RuntimeCheckpointWriter::InitializeFromEnvironment() {
  const char *output_directory = std::getenv("TT_IREE_CHECKPOINT_OUTPUT");
  if (!output_directory || output_directory[0] == '\0')
    return iree_ok_status();
  return Initialize(output_directory);
}

iree_status_t
RuntimeCheckpointWriter::Initialize(std::string output_directory) {
  std::lock_guard<std::mutex> lock(mutex_);
  if (output_directory.empty()) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "checkpoint output directory is empty");
  }

  std::error_code error;
  std::filesystem::create_directories(output_directory, error);
  if (error) {
    return iree_make_status(IREE_STATUS_PERMISSION_DENIED,
                            "failed to create checkpoint directory '%s': %s",
                            output_directory.c_str(), error.message().c_str());
  }

  output_directory_ = std::move(output_directory);
  next_dispatch_index_ = 1;
  records_.clear();
  enabled_ = true;
  IREE_RETURN_IF_ERROR(WriteManifestLocked());
  std::fprintf(stderr, "tt-iree: checkpoint output: %s\n",
               output_directory_.c_str());
  return iree_ok_status();
}

iree_status_t RuntimeCheckpointWriter::Record(RuntimeCheckpointRecord record) {
  if (!enabled_)
    return iree_ok_status();
  IREE_RETURN_IF_ERROR(ValidateRecord(record));

  std::lock_guard<std::mutex> lock(mutex_);
  record.dispatch_index = next_dispatch_index_;
  char index[32] = {};
  std::snprintf(index, sizeof(index), "%04" PRIu64, record.dispatch_index);
  record.file = "dispatch_" + std::string(index) + "_" +
                SanitizeFileComponent(record.op_code) + ".npy";
  IREE_RETURN_IF_ERROR(WriteNpyLocked(record));
  std::vector<float>().swap(record.values);
  records_.push_back(std::move(record));
  ++next_dispatch_index_;
  return WriteManifestLocked();
}

iree_status_t
RuntimeCheckpointWriter::WriteNpyLocked(const RuntimeCheckpointRecord &record) {
  std::ostringstream shape;
  shape << '(';
  for (size_t i = 0; i < record.shape.size(); ++i) {
    if (i)
      shape << ", ";
    shape << record.shape[i];
  }
  if (record.shape.size() == 1)
    shape << ',';
  shape << ')';

  std::string header =
      "{'descr': '<f4', 'fortran_order': False, 'shape': " + shape.str() +
      ", }";
  constexpr size_t kPreambleSize = 10;
  size_t padding = (64 - ((kPreambleSize + header.size() + 1) % 64)) % 64;
  header.append(padding, ' ');
  header.push_back('\n');
  if (header.size() > std::numeric_limits<uint16_t>::max()) {
    return iree_make_status(IREE_STATUS_RESOURCE_EXHAUSTED,
                            "checkpoint NPY header is too large");
  }

  const std::filesystem::path path =
      std::filesystem::path(output_directory_) / record.file;
  std::ofstream output(path,
                       std::ios::out | std::ios::binary | std::ios::trunc);
  if (!output.is_open()) {
    return iree_make_status(IREE_STATUS_PERMISSION_DENIED,
                            "failed to open checkpoint output '%s'",
                            path.c_str());
  }
  constexpr char kMagic[] = "\x93NUMPY";
  output.write(kMagic, sizeof(kMagic) - 1);
  const char version[2] = {1, 0};
  output.write(version, sizeof(version));
  const uint16_t header_size = static_cast<uint16_t>(header.size());
  const char header_size_le[2] = {
      static_cast<char>(header_size & 0xFF),
      static_cast<char>((header_size >> 8) & 0xFF),
  };
  output.write(header_size_le, sizeof(header_size_le));
  output.write(header.data(), static_cast<std::streamsize>(header.size()));
  output.write(
      reinterpret_cast<const char *>(record.values.data()),
      static_cast<std::streamsize>(record.values.size() * sizeof(float)));
  return CheckOutputStream(output, path);
}

iree_status_t RuntimeCheckpointWriter::WriteManifestLocked() {
  const std::filesystem::path directory(output_directory_);
  const std::filesystem::path temporary_path = directory / "manifest.json.tmp";
  const std::filesystem::path final_path = directory / "manifest.json";
  std::ofstream output(temporary_path, std::ios::out | std::ios::trunc);
  if (!output.is_open()) {
    return iree_make_status(IREE_STATUS_PERMISSION_DENIED,
                            "failed to open checkpoint manifest '%s'",
                            temporary_path.c_str());
  }

  output << "{\n"
         << "  \"schema_version\": 1,\n"
         << "  \"dtype\": \"float32\",\n"
         << "  \"checkpoint_count\": " << records_.size() << ",\n"
         << "  \"records\": [\n";
  for (size_t i = 0; i < records_.size(); ++i) {
    const RuntimeCheckpointRecord &record = records_[i];
    output << "    {\"dispatch_index\": " << record.dispatch_index
           << ", \"op_code\": \"" << JsonEscape(record.op_code)
           << "\", \"builtin_program\": " << record.builtin_program
           << ", \"dispatch_kind\": " << record.dispatch_kind
           << ", \"output_binding\": " << record.output_binding
           << ", \"shape\": [";
    for (size_t d = 0; d < record.shape.size(); ++d) {
      if (d)
        output << ", ";
      output << record.shape[d];
    }
    output << "], \"file\": \"" << JsonEscape(record.file) << "\"}"
           << (i + 1 == records_.size() ? "\n" : ",\n");
  }
  output << "  ]\n}\n";
  IREE_RETURN_IF_ERROR(CheckOutputStream(output, temporary_path));
  output.close();

  std::error_code error;
  std::filesystem::rename(temporary_path, final_path, error);
  if (error) {
    return iree_make_status(IREE_STATUS_DATA_LOSS,
                            "failed to publish checkpoint manifest '%s': %s",
                            final_path.c_str(), error.message().c_str());
  }
  return iree_ok_status();
}

} // namespace iree::hal::tenstorrent
