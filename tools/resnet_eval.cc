// Copyright 2026 The tt-iree Authors
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// Static B1 BF16 dataset runners share session and buffer lifetime handling.
#include <chrono>
#include <cstdio>
#include <filesystem>
#include <fstream>
#include <set>
#include <string>
#include <utility>
#include <vector>

#include "iree/hal/buffer_transfer.h"
#include "iree/runtime/api.h"
#ifdef TT_IREE_EVAL_PROFILER
#include "iree/hal/drivers/tenstorrent/tt_device.h"
#include "iree/hal/drivers/tenstorrent/tt_runtime_profiler.h"
#endif

namespace {
#ifdef TT_IREE_EVAL_MNIST
constexpr iree_hal_dim_t kInputShape[] = {1, 1, 28, 28};
constexpr iree_hal_dim_t kOutputClasses = 10;
#else
constexpr iree_hal_dim_t kInputShape[] = {1, 3, 224, 224};
constexpr iree_hal_dim_t kOutputClasses = 1000;
#endif
constexpr size_t kInputBytes = kInputShape[0] * kInputShape[1] *
                               kInputShape[2] * kInputShape[3] * 2;
constexpr size_t kOutputBytes = kOutputClasses * 2;
using Clock = std::chrono::steady_clock;
double Seconds(Clock::time_point start) {
  return std::chrono::duration<double>(Clock::now() - start).count();
}

iree_status_t Evaluate(iree_runtime_session_t *session,
                       const std::string &input, const std::string &output,
                       size_t index, std::ostream &timings) {
#ifdef TT_IREE_EVAL_PROFILER
  auto *profiler =
      iree_hal_tt_device_runtime_profiler(iree_runtime_session_device(session));
  if (profiler)
    profiler->SetInvocationId(
        index + 1, iree::hal::tenstorrent::RuntimeInvocationPhase::Input);
  struct InvocationScope {
    iree::hal::tenstorrent::RuntimeProfiler *profiler;
    ~InvocationScope() {
      if (profiler)
        profiler->SetInvocationId(0);
    }
  } invocation_scope{profiler};
#endif
  auto start = Clock::now();
  std::vector<uint8_t> input_data(kInputBytes);
  std::ifstream input_stream(input, std::ios::binary);
  if (!input_stream.read(reinterpret_cast<char *>(input_data.data()),
                         input_data.size()) ||
      input_stream.peek() != std::ifstream::traits_type::eof()) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "expected exactly %zu BF16 input bytes: %s",
                            kInputBytes, input.c_str());
  }
  iree_runtime_call_t call;
  IREE_RETURN_IF_ERROR(iree_runtime_call_initialize_by_name(
      session, iree_make_cstring_view("module.main"), &call));
  auto *device = iree_runtime_session_device(session);
  iree_hal_buffer_params_t params = {};
  params.type = IREE_HAL_MEMORY_TYPE_DEVICE_LOCAL;
  params.access = IREE_HAL_MEMORY_ACCESS_ALL;
  params.usage = IREE_HAL_BUFFER_USAGE_DEFAULT;
  iree_hal_buffer_view_t *arg = nullptr;
  iree_status_t status = iree_hal_buffer_view_allocate_buffer_copy(
      device, iree_runtime_session_device_allocator(session), 4, kInputShape,
      IREE_HAL_ELEMENT_TYPE_BFLOAT_16, IREE_HAL_ENCODING_TYPE_DENSE_ROW_MAJOR,
      params, iree_make_const_byte_span(input_data.data(), input_data.size()),
      &arg);
  if (iree_status_is_ok(status)) {
    status = iree_runtime_call_inputs_push_back_buffer_view(&call, arg);
  }
  iree_hal_buffer_view_release(arg);
  double input_seconds = Seconds(start);
  auto invoke_start = Clock::now();
#ifdef TT_IREE_EVAL_PROFILER
  if (profiler)
    profiler->SetInvocationId(index + 1);
#endif
  if (iree_status_is_ok(status)) {
    status = iree_runtime_call_invoke(&call, 0);
  }
  double invoke_seconds = Seconds(invoke_start);
#ifdef TT_IREE_EVAL_PROFILER
  if (profiler)
    profiler->SetInvocationId(
        index + 1, iree::hal::tenstorrent::RuntimeInvocationPhase::Output);
#endif
  auto output_start = Clock::now();
  iree_hal_buffer_view_t *result = nullptr;
  if (iree_status_is_ok(status)) {
    status = iree_runtime_call_outputs_pop_front_buffer_view(&call, &result);
  }
  if (iree_status_is_ok(status) &&
      (iree_vm_list_size(iree_runtime_call_outputs(&call)) != 0 ||
       iree_hal_buffer_view_shape_rank(result) != 2 ||
       iree_hal_buffer_view_shape_dim(result, 0) != 1 ||
       iree_hal_buffer_view_shape_dim(result, 1) != kOutputClasses ||
       iree_hal_buffer_view_element_type(result) !=
           IREE_HAL_ELEMENT_TYPE_BFLOAT_16 ||
       iree_hal_buffer_view_encoding_type(result) !=
           IREE_HAL_ENCODING_TYPE_DENSE_ROW_MAJOR ||
       iree_hal_buffer_view_byte_length(result) != kOutputBytes)) {
    status = iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                              "expected one dense 1x%dxbf16 output",
                              static_cast<int>(kOutputClasses));
  }
  std::vector<uint8_t> output_data(kOutputBytes);
  if (iree_status_is_ok(status)) {
    status = iree_hal_device_transfer_d2h(
        device, iree_hal_buffer_view_buffer(result), 0, output_data.data(),
        output_data.size(), IREE_HAL_TRANSFER_BUFFER_FLAG_DEFAULT,
        iree_infinite_timeout());
  }
  if (iree_status_is_ok(status)) {
    std::ofstream output_stream(output, std::ios::binary);
    output_stream.write(reinterpret_cast<const char *>(output_data.data()),
                        output_data.size());
    output_stream.close();
    if (!output_stream) {
      status = iree_make_status(IREE_STATUS_DATA_LOSS,
                                "failed to write output: %s", output.c_str());
    }
  }
  double output_seconds = Seconds(output_start);
  iree_hal_buffer_view_release(result);
  iree_runtime_call_deinitialize(&call);
#ifdef TT_IREE_EVAL_PROFILER
  if (profiler)
    status = iree_status_join(status, profiler->Flush());
#endif
  if (iree_status_is_ok(status)) {
    timings << index << ',' << input_seconds << ',' << invoke_seconds << ','
            << output_seconds << ',' << Seconds(start) << '\n';
    timings.flush();
    if (!timings)
      return iree_make_status(IREE_STATUS_DATA_LOSS, "timing write failed");
    std::printf("sample %zu completed (invoke %.6f s)\n", index,
                invoke_seconds);
    std::fflush(stdout);
  }
  return status;
}
} // namespace

int main(int argc, char **argv) {
  if (argc < 4 || argc > 5) {
    std::fprintf(
        stderr,
        "usage: %s module.vmfb inputs.tsv timings.csv [device=tenstorrent]\n"
        "TSV rows: absolute_input_path<TAB>absolute_output_path\n",
        argv[0]);
    return 1;
  }
  std::vector<std::pair<std::string, std::string>> samples;
  std::ifstream manifest(argv[2]);
  std::set<std::string> output_paths;
  std::set<std::string> input_paths;
  std::string line;
  while (std::getline(manifest, line)) {
    size_t tab = line.find('\t');
    if (tab == std::string::npos ||
        line.find('\t', tab + 1) != std::string::npos) {
      std::fprintf(stderr, "invalid TSV row\n");
      return 1;
    }
    std::filesystem::path input(line.substr(0, tab)),
        output(line.substr(tab + 1));
    std::error_code error;
    if (!input.is_absolute() || !output.is_absolute() ||
        std::filesystem::file_size(input, error) != kInputBytes || error ||
        std::filesystem::exists(output) ||
        std::filesystem::is_symlink(output) ||
        !std::filesystem::is_directory(output.parent_path()) ||
        !output_paths.insert(std::filesystem::weakly_canonical(output).string())
             .second) {
      std::fprintf(stderr, "invalid input or existing/duplicate output: %s\n",
                   line.c_str());
      return 1;
    }
    samples.emplace_back(input.string(), output.string());
    input_paths.insert(std::filesystem::canonical(input).string());
  }
  const auto timing_path =
      std::filesystem::weakly_canonical(std::filesystem::absolute(argv[3]))
          .string();
  for (const auto &output : output_paths) {
    if (input_paths.count(output)) {
      std::fprintf(stderr, "output would overwrite an input\n");
      return 1;
    }
  }
  if (manifest.bad() || samples.empty() || std::filesystem::exists(argv[3]) ||
      std::filesystem::is_symlink(argv[3]) || output_paths.count(timing_path) ||
      input_paths.count(timing_path) ||
      timing_path == std::filesystem::canonical(argv[1]).string() ||
      timing_path == std::filesystem::canonical(argv[2]).string()) {
    std::fprintf(stderr,
                 "empty/unreadable manifest or existing/aliased timings\n");
    return 1;
  }
  std::ofstream timings(argv[3]);
  if (!timings)
    return 1;
  timings.precision(12);
  timings << "sample,input_read_upload_seconds,invoke_seconds,output_download_"
             "write_seconds,total_seconds\n";
  auto setup_start = Clock::now();
  iree_runtime_instance_options_t options;
  iree_runtime_instance_options_initialize(&options);
  iree_runtime_instance_options_use_all_available_drivers(&options);
  iree_runtime_instance_t *instance = nullptr;
  iree_status_t status = iree_runtime_instance_create(
      &options, iree_allocator_system(), &instance);
  iree_hal_device_t *device = nullptr;
  if (iree_status_is_ok(status)) {
    status = iree_runtime_instance_try_create_default_device(
        instance, iree_make_cstring_view(argc == 5 ? argv[4] : "tenstorrent"),
        &device);
  }
  iree_runtime_session_options_t session_options;
  iree_runtime_session_options_initialize(&session_options);
  iree_runtime_session_t *session = nullptr;
  if (iree_status_is_ok(status)) {
    status = iree_runtime_session_create_with_device(
        instance, &session_options, device, iree_allocator_system(), &session);
  }
  iree_hal_device_release(device);
  if (iree_status_is_ok(status)) {
    status =
        iree_runtime_session_append_bytecode_module_from_file(session, argv[1]);
  }
  std::printf("session setup %.6f s; %zu distinct calls\n",
              Seconds(setup_start), samples.size());
  for (size_t i = 0; i < samples.size() && iree_status_is_ok(status); ++i) {
    status = Evaluate(session, samples[i].first, samples[i].second, i, timings);
  }
  iree_runtime_session_release(session);
  iree_runtime_instance_release(instance);
  int exit_code = static_cast<int>(iree_status_code(status));
  if (!iree_status_is_ok(status)) {
    iree_status_fprint(stderr, status);
    iree_status_ignore(status);
  }
  return exit_code;
}
