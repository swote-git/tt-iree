// Copyright 2026 The tt-iree Authors
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// Included by tt_device.cc after its binding/AOT helpers. Hardware only.
static iree_status_t iree_hal_tt_execute_native_descriptor(
    iree_hal_tt_device_t *device, const iree_hal_tt_kernel_params_t *params,
    const iree_hal_tt_dispatch_command_t &cmd,
    iree_hal_buffer_binding_table_t bindings,
    iree::hal::tenstorrent::RuntimeProfileStages *stages) {
  tt_iree_ttex_entry_point_desc_t ep = {};
  ep.dispatch_kind = params->dispatch_kind;
  ep.matmul_like = params->matmul_like;
  ep.conv2d = params->conv2d;
  ep.linear = params->linear;
  ep.max_pool = params->max_pool;
  ep.global_sum = params->global_sum;
  ep.scalar_div = params->scalar_div;
  ep.bf16_native = params->bf16_native;
  const bool bf16_native = ep.dispatch_kind == TT_IREE_TTEX_DISPATCH_KIND_BF16_NATIVE;
  const unsigned tensor_count = bf16_native ? TT_IREE_BF16_NATIVE_TENSOR_COUNT : TT_IREE_NATIVE_TENSOR_COUNT;
  const unsigned partition_arg = tensor_count * 2;
  const tt_iree_ttex_tensor_desc_t *t[TT_IREE_NATIVE_MAX_TENSOR_COUNT] = {};
  tt_iree_ttex_native_tensors(&ep, t);
  if (!t[2])
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "missing native output");
  std::vector<uint32_t> args(partition_arg);
  for (unsigned i = 0; i < tensor_count; ++i) {
    iree_hal_buffer_ref_t ref = {};
    IREE_RETURN_IF_ERROR(iree_hal_tt_resolve_tensor_buffer_ref(
        *(t[i] ? t[i] : t[2]), cmd, bindings, &ref));
    IREE_RETURN_IF_ERROR(iree_hal_tt_get_interleaved_buffer_runtime_args(
        ref, &args[i * 2], &args[i * 2 + 1]));
  }
  auto *idevice = iree_hal_tt_device_idevice(device);
  if (!idevice)
    return iree_make_status(IREE_STATUS_UNAVAILABLE,
                            "TT-Metal device not initialized");
  auto expected_arch =
      params->aot_target_arch == TT_IREE_TTEX_TARGET_ARCH_BLACKHOLE
          ? tt::ARCH::BLACKHOLE
          : tt::ARCH::WORMHOLE_B0;
  if (idevice->arch() != expected_arch)
    return iree_make_status(IREE_STATUS_FAILED_PRECONDITION,
                            "native AOT architecture mismatch");
  const bool matrix = tt_iree_ttex_is_native_matrix_kind(
      bf16_native ? ep.bf16_native.operation_kind : ep.dispatch_kind);
  const bool uses_tensix =
      matrix && (bf16_native
          ? params->builtin_program == TT_IREE_TTEX_BUILTIN_PROGRAM_BF16_NATIVE_TENSIX
          : params->ttex_version != TT_IREE_TTEX_VERSION_15);
  uint32_t m = 0, n = 0, k = 0, elements = 0;
  if (bf16_native && matrix) {
    m = ep.bf16_native.m;
    n = ep.bf16_native.n;
    k = ep.bf16_native.k;
  } else if (ep.dispatch_kind == TT_IREE_TTEX_DISPATCH_KIND_MATMUL_LIKE) {
    m = ep.matmul_like.m;
    n = ep.matmul_like.n;
    k = ep.matmul_like.k;
  } else if (ep.dispatch_kind == TT_IREE_TTEX_DISPATCH_KIND_CONVOLUTION) {
    m = ep.conv2d.output.shape[1] * ep.conv2d.output.shape[2];
    n = ep.conv2d.output.shape[0];
    k = ep.conv2d.input.shape[0] * ep.conv2d.kernel_h * ep.conv2d.kernel_w;
  } else if (ep.dispatch_kind == TT_IREE_TTEX_DISPATCH_KIND_LINEAR) {
    m = ep.linear.m;
    n = ep.linear.n;
    k = ep.linear.k;
  } else {
    uint64_t count = 1;
    for (unsigned i = 0; i < t[2]->rank; ++i)
      count *= t[2]->shape[i];
    if (count > UINT32_MAX)
      return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                              "native element count overflow");
    elements = static_cast<uint32_t>(count);
  }
  uint32_t cols = cmd.config.workgroup_count[0],
           rows = cmd.config.workgroup_count[1];
  auto grid = idevice->compute_with_storage_grid_size();
  uint32_t mt = (m + 31) / 32, nt = (n + 31) / 32;
  if (!cols || !rows || cols > grid.x || rows > grid.y ||
      cmd.config.workgroup_count[2] != 1 ||
      (matrix && (nt % cols || mt % rows)))
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "native grid cannot partition dispatch");
  try {
    uint64_t start =
        stages ? iree::hal::tenstorrent::RuntimeProfiler::NowNs() : 0;
    auto program = tt::tt_metal::CreateProgram();
    tt::tt_metal::CoreRange range({0, 0}, {cols - 1, rows - 1});
    tt::tt_metal::CoreRangeSet cores(std::set<tt::tt_metal::CoreRange>{range});
    auto cb = [&](uint8_t index, tt::DataFormat format, uint32_t page,
                  uint32_t pages) {
      tt::tt_metal::CreateCircularBuffer(
          program, cores,
          tt::tt_metal::CircularBufferConfig(page * pages, {{index, format}})
              .set_page_size(index, page));
    };
    cb(2, tt::DataFormat::UInt32, 128, 1);
    if (matrix && !uses_tensix) {
      for (uint8_t i : {0, 1, 16})
        cb(i, tt::DataFormat::UInt32, 4096, 1);
    } else if (uses_tensix && bf16_native) {
      cb(0, tt::DataFormat::Float16_b, 2048, 2);
      cb(1, tt::DataFormat::Float16_b, 2048, 2);
      cb(16, tt::DataFormat::Float32, 4096, 2);
      for (uint8_t i : {8, 9, 17})
        cb(i, tt::DataFormat::UInt32, 4096, 1);
      cb(3, tt::DataFormat::UInt32, 128, 1);
    } else if (uses_tensix && params->ttex_version >= TT_IREE_TTEX_VERSION_16) {
      for (uint8_t i : {0, 1, 4, 5, 6, 7, 10, 11})
        cb(i, tt::DataFormat::Int8, 1024, 2);
      for (uint8_t i : {8, 9, 17, 18})
        cb(i, tt::DataFormat::UInt32, 4096, 1);
      cb(12, tt::DataFormat::UInt32, 256, 2);
      cb(3, tt::DataFormat::UInt32, 128, 1);
      cb(16, tt::DataFormat::Int32, 4096, 2);
    } else if (uses_tensix) {
      cb(0, tt::DataFormat::Float16_b, 2048, 2);
      cb(1, tt::DataFormat::Float16_b, 2048, 2);
      if (params->ttex_version >= TT_IREE_TTEX_VERSION_14)
        for (uint8_t i = 4; i <= 7; ++i)
          cb(i, tt::DataFormat::Float16_b, 2048, 2);
      cb(3, tt::DataFormat::UInt32, 128, 1);
      cb(16, tt::DataFormat::Float32, 4096, 2);
    }
    std::vector<uint32_t> compile_args;
    for (unsigned i = 0; i < tensor_count; ++i) {
      tt::tt_metal::TensorAccessorArgs::create_dram_interleaved().append_to(
          compile_args);
      compile_args.back() = 2048;
    }
    tt::tt_metal::DataMovementConfig reader_config{
        .processor = uses_tensix ? tt::tt_metal::DataMovementProcessor::RISCV_1
                                 : tt::tt_metal::DataMovementProcessor::RISCV_0,
        .noc = uses_tensix ? tt::tt_metal::NOC::RISCV_1_default
                           : tt::tt_metal::NOC::RISCV_0_default,
        .compile_args = compile_args};
    tt::tt_metal::DataMovementConfig writer_config{
        .processor = tt::tt_metal::DataMovementProcessor::RISCV_0,
        .noc = tt::tt_metal::NOC::RISCV_0_default,
        .compile_args = compile_args};
    auto bundle =
        static_cast<tt_device_ttex_ns(TTMetalKernelBundleDef_table_t)>(
            params->aot_kernel_bundle);
    if (iree_string_view_is_empty(params->aot_artifact_root))
      throw std::runtime_error("native AOT artifact cache missing");
    std::string root(params->aot_artifact_root.data,
                     params->aot_artifact_root.size);
    auto reader = iree_hal_tt_create_kernel_from_embedded_precompiled(
        program,
        iree_hal_tt_find_aot_kernel_source(bundle,
                                           TT_IREE_TTEX_KERNEL_ROLE_READER),
        cores, reader_config, root);
    tt::tt_metal::KernelHandle writer = 0, compute = 0;
    if (uses_tensix) {
      writer = iree_hal_tt_create_kernel_from_embedded_precompiled(
          program,
          iree_hal_tt_find_aot_kernel_source(bundle,
                                             TT_IREE_TTEX_KERNEL_ROLE_WRITER),
          cores, writer_config, root);
      compute = iree_hal_tt_create_kernel_from_embedded_precompiled(
          program,
          iree_hal_tt_find_aot_kernel_source(bundle,
                                             TT_IREE_TTEX_KERNEL_ROLE_COMPUTE),
          cores, iree_hal_tt_matmul_compute_config(), root);
    }
    for (uint32_t r = 0; r < rows; ++r)
      for (uint32_t c = 0; c < cols; ++c) {
        tt::tt_metal::CoreCoord core{c, r};
        auto core_args = args;
        if (matrix) {
          core_args.insert(core_args.end(), {r * (mt / rows), c * (nt / cols),
                                             mt / rows, nt / cols});
          if (uses_tensix) {
            tt::tt_metal::SetRuntimeArgs(program, writer, core, core_args);
            tt::tt_metal::SetRuntimeArgs(
                program, compute, core,
                {(mt / rows) * (nt / cols), (k + 31) / 32});
          }
        } else {
          uint32_t ordinal = r * cols + c, total = rows * cols;
          uint32_t begin = static_cast<uint64_t>(elements) * ordinal / total;
          uint32_t end =
              static_cast<uint64_t>(elements) * (ordinal + 1) / total;
          core_args.insert(core_args.end(), {begin, end - begin, ordinal});
        }
        tt::tt_metal::SetRuntimeArgs(program, reader, core, core_args);
      }
    tt::tt_metal::distributed::MeshWorkload workload;
    workload.add_program(tt::tt_metal::distributed::MeshCoordinateRange(
                             device->mesh_device->shape()),
                         std::move(program));
    if (stages)
      stages->program_setup_ns +=
          iree::hal::tenstorrent::RuntimeProfiler::NowNs() - start;
    start = stages ? iree::hal::tenstorrent::RuntimeProfiler::NowNs() : 0;
    tt::tt_metal::distributed::EnqueueMeshWorkload(*device->mesh_cq, workload,
                                                   true);
    if (stages)
      stages->enqueue_wait_ns +=
          iree::hal::tenstorrent::RuntimeProfiler::NowNs() - start;
    return iree_ok_status();
  } catch (const std::exception &e) {
    return iree_make_status(IREE_STATUS_INTERNAL,
                            "native descriptor dispatch failed: %s", e.what());
  }
}
