#!/usr/bin/env bash
# Copyright 2026 The tt-iree Authors
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

set -euo pipefail

usage() {
  cat <<'EOF'
Usage: scripts/analyze_resnet50_ir.sh <global-opt.mlir> [output-dir]

Resumes an IREE global-optimization dump with the Tenstorrent target, records
the dispatch/executable inventory, and probes the current full-compile gate.

Environment:
  IREE_COMPILE                 Path to iree-compile.
  TT_IREE_REQUIRE_FULL_COMPILE
                               Require full compilation and exact coverage.
  TT_IREE_EXPECT_EXPORTS       Expected export count (default: 37).
EOF
}

if [[ $# -lt 1 || $# -gt 2 ]]; then
  usage >&2
  exit 2
fi

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
IR_PATH="$(realpath "$1")"
OUTPUT_DIR="${2:-/tmp/tt-iree-resnet50-analysis}"
IREE_COMPILE="${IREE_COMPILE:-${ROOT_DIR}/build-top/third_party/iree/tools/iree-compile}"

if [[ ! -f "${IR_PATH}" ]]; then
  echo "error: input IR does not exist: ${IR_PATH}" >&2
  exit 2
fi
if [[ ! -x "${IREE_COMPILE}" ]]; then
  echo "error: iree-compile is not executable: ${IREE_COMPILE}" >&2
  exit 2
fi

mkdir -p "${OUTPUT_DIR}"
DISPATCH_IR="${OUTPUT_DIR}/dispatch_creation.mlir"
EXECUTABLE_IR="${OUTPUT_DIR}/executable_sources.mlir"
VMFB="${OUTPUT_DIR}/resnet50.vmfb"
COMPILE_LOG="${OUTPUT_DIR}/full_compile.log"
SUMMARY="${OUTPUT_DIR}/summary.txt"

compile_to_phase() {
  local phase="$1"
  local output="$2"
  "${IREE_COMPILE}" "${IR_PATH}" \
    --compile-from=global-optimization \
    --compile-to="${phase}" \
    --iree-hal-target-device=tenstorrent \
    --emit-mlir-bytecode=false \
    -o "${output}"
}

echo "[1/3] Materializing dispatch creation IR"
compile_to_phase dispatch-creation "${DISPATCH_IR}"

echo "[2/3] Materializing executable source IR"
compile_to_phase executable-sources "${EXECUTABLE_IR}"

echo "[3/3] Probing full Tenstorrent compilation"
if "${IREE_COMPILE}" "${IR_PATH}" \
    --compile-from=global-optimization \
    --iree-hal-target-device=tenstorrent \
    -o "${VMFB}" >"${COMPILE_LOG}" 2>&1; then
  FULL_COMPILE_STATUS="PASS"
else
  FULL_COMPILE_STATUS="FAIL"
  rm -f "${VMFB}"
fi

count_matches() {
  local pattern="$1"
  local file="$2"
  local count
  count="$(rg -c "${pattern}" "${file}" 2>/dev/null || true)"
  echo "${count:-0}"
}

CATEGORY_COUNTS="$({
  rg 'hal\.executable\.export public @' "${EXECUTABLE_IR}" || true
} | perl -ne '
  if (/dispatch_\d+_([^ ]+) ordinal/) {
    $name = $1;
    if ($name =~ /^matmul_like_/) { $category = "matmul_like"; }
    elsif ($name =~ /^matmul_/) { $category = "matmul"; }
    elsif ($name =~ /^conv_/) { $category = "conv"; }
    elsif ($name =~ /^reduction_/) { $category = "reduction"; }
    elsif ($name =~ /^elementwise_/) { $category = "elementwise"; }
    elsif ($name eq "slow_memcpy") { $category = "slow_memcpy"; }
    else { $category = "other"; }
    print "$category\n";
  }
' | sort | uniq -c | sort -nr)"

FIRST_ERROR="$(rg -m1 'error: ' "${COMPILE_LOG}" 2>/dev/null || true)"
UNSUPPORTED_EXPORTS="$(perl -0777 -ne '
  while (/error: unsupported Tenstorrent dispatch;.*?sym_name = "([^"]+)"/sg) {
    print "$1\n";
  }
' "${COMPILE_LOG}" | sort -u)"
UNSUPPORTED_EXPORT_COUNT="$(printf '%s\n' "${UNSUPPORTED_EXPORTS}" |
  awk 'NF { count++ } END { print count + 0 }')"
HAL_EXECUTABLE_COUNT="$(count_matches 'hal\.executable\.export public @' "${EXECUTABLE_IR}")"
SERIALIZER_ACCEPTED_COUNT=$((HAL_EXECUTABLE_COUNT - UNSUPPORTED_EXPORT_COUNT))

{
  echo "source=${IR_PATH}"
  echo "source_bytes=$(wc -c < "${IR_PATH}")"
  echo "input=$(rg -m1 'hal\.tensor\.import' "${IR_PATH}" | sed 's/^[[:space:]]*//' || true)"
  echo "output=$(rg -m1 'hal\.tensor\.export' "${IR_PATH}" | sed 's/^[[:space:]]*//' || true)"
  echo "linalg_generic=$(count_matches 'linalg\.generic' "${IR_PATH}")"
  echo "linalg_matmul=$(count_matches 'linalg\.matmul' "${IR_PATH}")"
  echo "flow_dispatches=$(count_matches 'flow\.dispatch\.workgroups' "${DISPATCH_IR}")"
  echo "hal_executables=${HAL_EXECUTABLE_COUNT}"
  echo "categories:"
  echo "${CATEGORY_COUNTS}"
  echo "full_compile=${FULL_COMPILE_STATUS}"
  echo "serializer_accepted_exports=${SERIALIZER_ACCEPTED_COUNT}"
  echo "serializer_unsupported_exports=${UNSUPPORTED_EXPORT_COUNT}"
  echo "first_error=${FIRST_ERROR}"
} | tee "${SUMMARY}"

if [[ -n "${UNSUPPORTED_EXPORTS}" ]]; then
  printf '%s\n' "${UNSUPPORTED_EXPORTS}" >"${OUTPUT_DIR}/unsupported_exports.txt"
else
  rm -f "${OUTPUT_DIR}/unsupported_exports.txt"
fi

echo "analysis_dir=${OUTPUT_DIR}"

if [[ "${TT_IREE_REQUIRE_FULL_COMPILE:-0}" == "1" ]]; then
  EXPECTED_EXPORTS="${TT_IREE_EXPECT_EXPORTS:-37}"
  if [[ "${FULL_COMPILE_STATUS}" != "PASS" ||
        "${HAL_EXECUTABLE_COUNT}" != "${EXPECTED_EXPORTS}" ||
        "${SERIALIZER_ACCEPTED_COUNT}" != "${EXPECTED_EXPORTS}" ||
        "${UNSUPPORTED_EXPORT_COUNT}" != "0" ]]; then
    echo "error: expected ${EXPECTED_EXPORTS}/${EXPECTED_EXPORTS} serializer coverage" >&2
    exit 1
  fi
fi
