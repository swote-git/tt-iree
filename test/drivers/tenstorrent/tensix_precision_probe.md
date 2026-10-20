# Tensix Precision Probe

This manual diagnostic uses project-owned kernels and the TT-Metal public host
API. It does not invoke IREE, TTEX, tensor splitting, fused epilogues, or host
output correction. JIT is used only by this diagnostic, not by production
executable loading. Run one hardware process at a time.

```sh
cmake --build build-top --target tensix_precision_probe
export TT_METAL_RUNTIME_ROOT="$PWD/third_party/tt-metal"
probe="$PWD/build-top/test/drivers/tenstorrent/tensix_precision_probe"

# BF16 storage -> HiFi4 matrix math -> FP32 destination/output.
"$probe" --format bf16 --case two-term --k 2 --output-dir /tmp/bf16-two-term
# Actual F32 CB storage, not merely an F32 external IREE tensor.
"$probe" --format f32 --case two-term --k 2 --output-dir /tmp/f32-two-term
# Sign-magnitude INT8 -> INT32 control; this is not a BF16 integer-valued input.
"$probe" --format int8 --case two-term --k 2 --require-exact \
  --output-dir /tmp/int8-two-term

# Lossless DRAM/CB transfer, with no compute kernel.
"$probe" --format f32 --op bypass --case wave --require-exact \
  --output-dir /tmp/f32-transfer
# Default F32 unpack through Src versus full F32 direct-to-destination unpack.
"$probe" --format f32 --op copy --case wave --output-dir /tmp/f32-src-copy
"$probe" --format f32 --op copy --case wave --f32-unpack-to-dst \
  --require-exact --output-dir /tmp/f32-dst-copy
# Known non-TF32-representable destination bits -> Float32 pack -> DRAM.
"$probe" --format f32 --op fill --fill-bits 3f812345 --require-exact \
  --output-dir /tmp/f32-pack
```

Each output directory must be new. `--help` lists all controls. All matrix
operations use a physical 32x32 tile on logical core `(0,0)`. `--k` zeros unused
reduction positions but does not request an unsupported physical 16x16 tile.
`--repeats` feeds identical tile pairs through the same program and destination
lifecycle, recording bitwise stability. This is not an independent-board test.

The `two-term` case repeats `[-127,-90,0,...]` in every LHS row and
`[-9,94,0,...]` in every RHS column. Every mathematical output is exactly `-7317`.
The `product` case has a scaled identity LHS, so each output has one nonzero
product. `snapshot` extracts the first K32/output tile from the earlier
integer-valued BF16 tail diagnostic. `integer` uses modular signed integer
inputs; `wave` uses nonuniform trigonometric inputs. INT8 `wave` rounds/clamps
these inputs and is mostly zero, so it is not useful evidence of floating-point
accuracy. Use the integer controls for INT8 comparisons.

For `--rhs-component high5`, stored RHS values are masked with `0xfff80000`
(hidden bit plus four fraction bits). `low5` uses the residual after this
masking. Each run gets its own reference for the actual stored component.
These are isolation experiments, not a proposed general F32 codegen algorithm.

## Artifacts and Exit Codes

- `summary.json`: requested fidelity, formats, error counts and repeated-tile
  stability. Counts include all repeats, not just one 1024-element tile.
- `lhs.f32`, `rhs.f32`, `expected.f32`: one row-major 32x32 tile in native-endian
  float32. The reference is double-precision matrix multiplication of the
  **stored quantized inputs**, then conversion to float32.
- `actual.f32`: row-major `[repeats,32,32]` output, decoded to float32.
- `*-packed.bin`: native-endian words in 16x16 face order. BF16 inputs use RNE;
  integer inputs/outputs use Tensix sign-magnitude representation.
- `reader.cpp`, `writer.cpp`, `compute.cpp`: exact kernel strings. Host compile
  arguments and define choices are in the probe source; unused compute strings
  are still saved for bypass experiments.

Without `--require-exact`, exit 0 means measurements were recorded, **not that
the result passed a correctness test**. With this flag, numerical or repeat
differences return 2. API/build/IO/option errors return 1. Numerical failures are
not added to CI as unconditional passing tests. Standard tolerance counts use
`abs(actual-expected) > 1e-5 + 1e-5 * abs(expected)` without widening the gate.

HiFi4 and FP32 destination/output do not imply IEEE FP32 matrix inputs or
intermediate arithmetic. F32 Src unpack may narrow to TF32; direct-to-Dst copy
tests an alternate unpack path and is not a fix for matrix multiplication.

## Reduction Isolation

`--case custom --lhs-file LHS --rhs-file RHS` reads exactly one native-endian,
row-major 32x32 raw F32 tile per operand. Inputs must be finite. They are then
quantized to the selected storage format just like built-in cases. Saved
`lhs.f32`/`rhs.f32` describe the actual stored operands, not a host correction.

`--k-slice-width 1|2|4|8|16|32` splits the logical reduction into multiple
zero-padded input tile pairs. One destination acquisition is kept across all
`matmul_tiles` calls, so accumulation remains in the device's FP32 destination.
No CPU summation, Kahan writer, or intermediate output download is added. Width
32 is the original behavior; width 1 performs 32 separate matmuls. Input packed
files have shape `[repeats,32/width,32,32]` in face order, while output shape is
unchanged. This expensive diagnostic is not a production optimization.

`--dst-check-bits HEX` reads destination bits with SFPU **before commit/pack**
and replaces them with 1.0 for an exact match and 0.0 otherwise. This mode
outputs flags, not matmul values. `math-expected.f32` retains the mathematical
reference; `expected.f32` contains the reference comparison flags. The flag
mode is forbidden for INT8 and bypass operations. Fill/match and fill/nonmatch
controls validate the comparator independently of matmul.

```sh
# Expected -7317 bits versus observed -7316 bits in destination.
"$probe" --format bf16 --case two-term --dst-check-bits c5e4a800 \
  --output-dir /tmp/dst-match-math
"$probe" --format bf16 --case two-term --dst-check-bits c5e4a000 \
  --output-dir /tmp/dst-match-observed
"$probe" --format bf16 --case two-term --k-slice-width 1 --require-exact \
  --output-dir /tmp/independent-products-same-dst

# Use a NumPy-enabled Python environment. All hardware runs are sequential.
/home/swote/.tenstorrent-venv/bin/python \
  test/drivers/tenstorrent/tensix_reduction_diagnostic.py \
  --probe "$probe" --output-dir /tmp/reduction-bf16
/home/swote/.tenstorrent-venv/bin/python \
  test/drivers/tenstorrent/analyze_tensix_reduction.py \
  --input-dir /tmp/reduction-bf16 --output /tmp/reduction-model.json
```

The suite records position-pair grids, operand orientation, signed sums, power
of two scaling, inactive/neighbor outliers, mantissa grids, K slicing, and
independently seeded integer/BF16 cases. `--select NAME` can be repeated. It
does not count known discrepancies as correctness passes. The analyzer tests
an **empirical**, 8-term shared-exponent alignment hypothesis against device
output, including nearest-even/truncation/16-term alternatives. It is not an
RTL implementation, a public hardware precision contract, or a replacement
golden. Residual differences are retained and no predicted values are fed back
into runtime execution.

## Bit Preservation and Prefilled Destination

`--dst-prefill-bits HEX` initializes the acquired FP32 destination with an exact
finite value using SFPU before any matrix instruction. It is only supported
for BF16/F32 matmul. All slices accumulate onto that value. The mathematical
reference includes the initial value before its final Float32 RNE conversion.
Use zero-product controls to check preservation before interpreting tie tests.
With `--dst-check-bits`, the comparison still happens before commit/pack.

```sh
# Prefill 2**24, then add the single product 1143.
"$probe" --format bf16 --case two-term --k 1 --dst-prefill-bits 4b800000 \
  --output-dir /tmp/prefilled-dst

# First freeze the original model, not the post-hoc rounding/order hypotheses.
cp test/drivers/tenstorrent/analyze_tensix_reduction.py /tmp/frozen-tensix-model.py
/home/swote/.tenstorrent-venv/bin/python \
  test/drivers/tenstorrent/tensix_precision_followup.py \
  --probe "$probe" --model /tmp/frozen-tensix-model.py \
  --prior-report /home/swote/dev/report/tt-iree-matrix-reduction-20261007 \
  --output-dir /tmp/tensix-bit-rounding
/home/swote/.tenstorrent-venv/bin/python \
  test/drivers/tenstorrent/analyze_tensix_precision_followup.py \
  --model /tmp/frozen-tensix-model.py \
  --prior-report /home/swote/dev/report/tt-iree-matrix-reduction-20261007 \
  --experiments /tmp/tensix-bit-rounding --output /tmp/tensix-posthoc.json
/home/swote/.tenstorrent-venv/bin/python \
  test/drivers/tenstorrent/tensix_precision_followup_test.py -v
```

The follow-up runner checks a fixed model SHA256. It saves all input hashes
and baseline/RNE predictions before starting 203 sequential hardware conditions.
Identity-matrix products isolate LHS/RHS bit survival, while prefilled
destinations distinguish positive/negative even/odd halfway cases. Prefix-K
and significand sweeps retain the old inputs; new seeded distributions are
compared without refitting. The original 20 nonmatching tiles are listed in CSV.

The post-hoc analyzer separately tests nearest ties toward positive infinity
and alternative K8-partial merge orders. It verifies that its original
away/fused variant is bitwise identical to the frozen predictor. Improved
post-hoc agreement is not a preregistered success or an RTL specification.
Relative-L2 is normalized to device output; near-zero cancellation outputs
need absolute errors, not an output-zero ULP count. Prefill runs are not valid
inputs to the older zero-initial-Dst reduction analyzer.

## Simulator Comparison

`tensix_ttsim_compare.py` replays archived measurements with an unchanged probe.
The launcher selects TT-Metal's simulator backend; no simulator detection or
alternate arithmetic is added to the host probe or device kernels. Keep the
simulator `.so` outside `third_party`, with the Blackhole SoC descriptor copied
next to it as `soc_descriptor.yaml`. TT-Metal v0.75.0 pins ttsim v1.9.6.

```sh
reference=/path/to/archived/measurement
archived_probe=/path/to/the/probe-used-for-that-measurement
simulator=/path/to/libttsim_bh.so
/home/swote/.tenstorrent-venv/bin/python \
  test/drivers/tenstorrent/tensix_ttsim_compare.py \
  --probe "$archived_probe" --reference-probe "$archived_probe" \
  --metal-root "$PWD/third_party/tt-metal" --simulator "$simulator" \
  --measurement "$reference" --output-dir /tmp/tensix-sim-comparison
/home/swote/.tenstorrent-venv/bin/python \
  test/drivers/tenstorrent/tensix_ttsim_compare_test.py -v
```

`--measurement` can be repeated. The launcher validates the supplied host
binary hashes, storage bytes, packed inputs, kernel strings, compile settings,
output shapes, finite values, and repeat stability. It compares **all repeated
outputs bitwise to the archived device output**, not merely to CPU tolerance.
The caller must supply the executable actually used for that archive; matching
two supplied file hashes alone does not establish the archive's provenance.

Default dispatch is slow. An opt-in `--dispatch fast` is available, but ttsim
v1.11.1 timed out on the four-repeat two-term probe under fast dispatch in this
investigation; this is an execution failure, not a floating-point discrepancy.
Each condition has a timeout and a persistent result record. Exit 0 requires
every requested replay to match. Known disagreement with the mathematical
reference is retained separately and is not counted as FP32 correctness.

On 2026-10-07, ttsim v1.9.6 and v1.11.1 each matched all 203 follow-up conditions
and 20 refreshed silicon conditions. The latter refresh was required because
the older archive and its later executable snapshot had different kernel
strings. The refreshed outputs also matched the original 20 outputs bitwise.
For the refreshed suite, debug/comment-stripped TRISC unpack/math/pack ELF image
sets were identical across silicon and both simulators. Reader/writer images
were not identical, so this is not a claim that every device binary matched.

Unmodified arithmetic excerpts from
[ttsim v1.11.1](https://github.com/tenstorrent/ttsim/blob/v1.11.1/src/tensix.cpp)
also matched all 438 archived BF16/F32 matmul conditions in a separate schedule
adapter. The code explicitly implements K8 product alignment, sign-dependent
partial-sum alignment, and destination renormalization. In particular, the
prefill tests' observed ties toward positive infinity are **not a global Dst
rounding rule**: partial-sum alignment uses ties toward positive infinity,
while prior-Dst alignment and final right-shift renormalization use ties away
from zero. This is source-backed diagnosis of the tested path, not an IEEE FP32
guarantee, a replacement golden, or proof across all silicon revisions.
