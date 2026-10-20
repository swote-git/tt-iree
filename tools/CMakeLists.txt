# Copyright 2025 The tt-iree Authors
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

if(TT_IREE_BUILD_RUNTIME)
  foreach(_TT_EVAL_TARGET IN ITEMS tt-iree-resnet-eval tt-iree-mnist-eval)
    add_executable(${_TT_EVAL_TARGET} resnet_eval.cc)
    target_link_libraries(${_TT_EVAL_TARGET} PRIVATE iree::runtime::unified)
    target_compile_features(${_TT_EVAL_TARGET} PRIVATE cxx_std_17)
  endforeach()
  target_compile_definitions(tt-iree-mnist-eval PRIVATE TT_IREE_EVAL_MNIST=1)
endif()
