# ResNet50 Bring-up Target

This document defines the first full-model acceptance target for tt-iree. The
source artifact is an IREE IR dump taken after device-agnostic global
optimization. It must remain a valid compiler input through:

```text
global-optimization IR
  -> dispatch creation
  -> Tenstorrent executable selection and TTEX serialization
  -> VMFB
  -> Tenstorrent HAL execution
  -> 1x1000 logits
```

The model artifact is external and is not copied into this repository. Analyze
it with:

```bash
scripts/analyze_resnet50_ir.sh /path/to/resnet50_global_opt.mlir
```

Set `TT_IREE_REQUIRE_FULL_COMPILE=1` when using the script as an eventual
compile acceptance gate.

## Measured Baseline

The baseline artifact at `/home/swote/dev/test/resnet50_global_opt.mlir` was
measured on 2026-08-20 with IREE v3.11.0:

| Property | Value |
| --- | ---: |
| Input | `tensor<1x3x224x224xf32>` |
| Output | `tensor<1x1000xf32>` |
| Embedded constants | approximately 102 MB |
| `linalg.generic` operations | 334 |
| Named `linalg.matmul` operations | 1 |
| Flow dispatch calls | 58 |
| Unique HAL executables after fusion | 37 |

The 37 executable categories are:

| Category | Count | Meaning |
| --- | ---: | --- |
| `matmul_like` | 22 | 1x1 convolution/projection, often with fused epilogues |
| `conv` | 11 | 7x7 or 3x3 spatial convolution, often with fused epilogues |
| `slow_memcpy` | 1 | Input padding interior copy |
| `reduction` | 1 | Global average pooling reduction |
| `elementwise` | 1 | Post-reduction scaling |
| `matmul` | 1 | Final `1x2048 * 1000x2048` classifier plus bias |

A 2026-09-16 full-compile audit accepts all 37 unique exports. The external
global-optimization IR produces 58 flow dispatch calls and 37 HAL
executables, with `serializer_accepted_exports=37`,
`serializer_unsupported_exports=0`, and `full_compile=PASS`. This establishes
100% compiler classification and TTEX serialization coverage for the measured
artifact. It does not yet establish full-model execution correctness: the
generated VMFB is approximately 1.5 GB because AOT artifacts are duplicated
per executable, constant-file import is incomplete, and a complete hardware
run has not been validated.

The compiler previously mistook the reduction's single `arith.addf` for the
BF16 elementwise add builtin. The add matcher now requires equal static BF16
tensors, identity indexing, and a pure two-input add body, so unsupported
reduction semantics fail compilation instead of producing an invalid program.
`scripts/analyze_resnet50_ir.sh` records accepted and unsupported export counts
and writes rejected names to `unsupported_exports.txt` when any remain.

## Single-Block E2E Fixture

`test/e2e/resnet50_bottleneck.mlir` isolates one stage-2 projection bottleneck
from the measured model. It keeps the model's F32 ABI, channel dimensions, and
operation topology while reducing the spatial extent from 56x56 to 8x8:

```text
input 64x8x8
  -> 1x1 64->64 -> batch norm -> ReLU
  -> 3x3 64->64 -> batch norm -> ReLU
  -> 1x1 64->256 -> batch norm
  +  projection shortcut 1x1 64->256 -> batch norm
  -> ReLU -> output 256x8x8
```

Run the acceptance test with:

```bash
ctest --test-dir build-top --output-on-failure \
  -R '^tt_e2e_resnet50_bottleneck_pipeline$'
```

The test currently provides complementary gates:

- It compiles through Tenstorrent dispatch creation and executable-source
  generation, then requires exactly four exports: three `matmul_like` and one
  `conv`, including the fused batch-normalization and ReLU epilogues.
- It serializes all four exports, including the flattened residual projection,
  into one complete Tenstorrent VMFB.
- It compiles the same fixture to LLVM CPU and executes it with deterministic
  inputs, requiring every output element to equal `0.96875`.
- `test/e2e/resnet50_projection.mlir` separately requires the first projection,
  scalar affine, ReLU, and padded output store to execute through a complete
  Tenstorrent VMFB. The mock result requires a zero border and `0.6875` in the
  logical 64x8x8 region.
- `test/e2e/resnet50_conv2d.mlir` requires the static CHW x OIHW 3x3
  convolution, scalar affine, and ReLU to serialize and execute through a
  complete Tenstorrent VMFB.
- `test/e2e/resnet50_stem_conv.mlir` reproduces the shape-exact 7x7 stride-2
  stem convolution and its 64-channel batch-normalization affine/ReLU. It
  compiles to TTEX v5 and executes in the mock HAL.
- `test/e2e/resnet50_max_pool.mlir` reproduces the shape-exact
  `64x114x114 -> 64x56x56` 3x3 stride-2 max-pool immediately after the stem.
  It compiles to TTEX v6 and executes in the mock HAL.
- `test/e2e/resnet50_per_channel_projection.mlir` reproduces the shape-exact
  `64x56x56` first-stage projection, per-channel affine/ReLU, and padded
  `64x58x58` output. It compiles to TTEX v7, requires a `2x7x1` workgroup
  grid, and executes in the mock HAL.
- `test/e2e/resnet50_input_pad.mlir` reproduces the full model's first
  `3x224x224xf32 -> 3x230x230xf32` zero-padding dispatch and requires it to
  serialize to a complete Tenstorrent VMFB.
- The full bottleneck mock boundary executes all four dispatches, including the
  residual projection epilogue, and checks the expected `0.96875` result.

The first projection is now a compiler-to-runtime E2E test on both the mock
driver and a physical Blackhole device. The complete bottleneck is not yet a
Tenstorrent device E2E test. TTEX v2 describes the F32 1x1 contractions
and static F32 3x3 convolution, including CHW, flattened channel-major, OI, and
OIHW tensor metadata. Tensor descriptors preserve static or runtime-derived
HAL subspan byte offsets and logical strides. The final projection records its
residual tensor and both affine branches before residual add and ReLU. The
runtime validates and preserves all descriptors. Projections gather F32 CHW
or channel-major storage, pad non-tile-aligned dimensions, stage contraction
through the multicore BF16 tiled matmul path on hardware, apply affine,
residual, and ReLU operations on the host, crop, and scatter back to F32
storage. Convolutions reuse the generated matmul AOT bundle via host-assisted
im2col, then apply scalar or runtime per-channel affine/ReLU and scatter to
padded CHW storage.

TTEX v3 adds an offline AOT kernel bundle to add, BF16 matmul, and matmul-like
entry points. During `iree-compile`, the Tenstorrent backend emits reader,
writer, and compute source from compiler-owned templates, invokes tt-metal's
offline compiler in mock device mode, and embeds the generated ELF artifacts
in TTEX. Each bundle names one target architecture. `p100a`/`blackhole` select
Blackhole, while
`wormhole`/`n150`/`n300` select Wormhole B0. The offline compiler keeps only
the products for that architecture. At runtime, the driver rejects an AOT
bundle that does not match the opened device, materializes its artifacts into
a temporary directory, and creates the kernel objects with
`FallbackPolicy::Error`. The embedded source is used only as tt-metal's
portable kernel identity and hash input; the runtime is not allowed to JIT
compile it.

TTEX v4 adds the narrow `PADDED_COPY` dispatch kind with source and destination
tensor descriptors. The compiler accepts only a direct static rank-3 F32 CHW
load/store slice, and the runtime preserves the destination border while
copying the logical source region. The first ResNet50 export uses source shape
`3x224x224`, destination storage shape `3x230x230`, and destination offset
`[0,3,3]`. M1 executes this operation through HAL buffer mapping on the host;
device-side data-movement codegen is deferred until correctness is established.

TTEX v5 adds optional per-output-channel convolution scale and bias vectors.
The compiler recognizes only exact static 3x3 stride-1 and 7x7 stride-2 CHW x
OIHW contractions and collapses the model's batch-normalization mean, common
epsilon multiplier, scale, and bias into channel-wise affine parameters. The
runtime retains v1-v4 compatibility and rejects vector epilogues on older
versions.

TTEX v6 adds the `MAX_POOL` dispatch kind and static input/output descriptors.
The compiler requires the exact ResNet50 CHW 3x3 stride-2, dilation-1, valid
pooling indexing maps, an unused 3x3 window operand, an `arith.maximumf`
reduction, and a negative-infinity initializer. The runtime repeats the shape
and attribute validation and executes a correctness-first host path while
preserving descriptor offsets and output storage borders. Device-side pooling
codegen remains open.

TTEX v7 adds per-output-channel scale and bias vectors to `MatmulLikeDef`.
The compiler collapses constant batch-normalization mean, common epsilon
multiplier, channel scale, and channel bias into one scale/bias pair per output
channel. The runtime validates exactly `N` finite values for each vector and
applies them during F32 output scatter. Older TTEX versions remain accepted;
vector matmul epilogues are rejected before v7. Static matmul-like grids are
factored to divisors no larger than 8 per axis, so the shape-exact `2x98` tile
space runs as a portable `2x7` core grid. Per-channel residual branches are not
yet part of this v7 contract.

TTEX v8 adds logical tensor strides, dispatch-constant-backed byte offsets,
and runtime-bound affine vectors. These fields cover executable reuse where
the stream passes activation and parameter offsets as push constants. It also
adds typed `GLOBAL_SUM`, `SCALAR_DIV`, and `LINEAR` descriptors for the final
three ResNet50 exports. Rank-0 MLIR scalar tensors are represented as
one-element row-major descriptors at the ABI boundary. The global reduction
and scalar division use correctness-first host execution; the classifier pads
`M/N/K`, reuses the generated tiled matmul AOT bundle, crops to `1x1000`, and
adds the F32 bias.

## Important IR Properties

- The model is static-shape F32. Current compute builtins are tile-aligned BF16.
- Convolution and most pointwise operations are canonical `linalg.generic`
  operations. Matching only named Linalg ops is insufficient.
- Dispatch creation fuses batch normalization, residual add, and ReLU with
  contractions. A contract must describe the complete dispatch, not just the
  first contraction found in a function.
- Several executables are reused with different constant offsets. Binding
  ranges and push constants must remain authoritative runtime arguments.
- The stream program uses fills and an embedded parameter file. HAL
  `fill_buffer` and `update_buffer` are implemented through the common
  synchronous transfer path; constant-file loading remains incomplete.
- Spatial extents include 3136, 784, 196, 49, and classifier width 1000. A
  correctness path must support padding instead of requiring every dimension
  to be a multiple of 32.

## Implementation Phases

### R0: Reproducible Acceptance Gate

Objective: keep the external model reproducible without committing its 200 MB
text dump.

- Run the analysis script in local and hardware CI jobs that have the model.
- Preserve `--compile-from=global-optimization` as the supported ingestion
  boundary.
- Track dispatch and executable counts so changes in upstream IREE fusion are
  visible.

Exit: dispatch creation and executable-source generation pass with the expected
58 calls and 37 unique executables.

### R1: HAL Data Movement Substrate

Objective: make the non-compute stream commands executable.

- Implement command-buffer and queue fill/update operations.
- Implement constant file import or an equivalent embedded-constant upload
  path.
- Add a strided pad/copy primitive for the first `slow_memcpy` export.
- Test large subranges and indirect bindings, not only whole buffers.

Exit: model constants and transient fills can be prepared on the device, and
the first padding dispatch executes correctly.

Status as of 2026-09-16: command-buffer and queue fill/update are implemented,
including direct/indirect subranges and immediate update-data snapshots. The
same transfer commands execute in hardware and mock builds. The first strided
padding dispatch now compiles and has descriptor/runtime unit coverage;
physical execution remains to be rechecked when a device is available.
Constant-file import remains before the R1 exit.

### R2: TTEX Dispatch Contract v2

Objective: describe the 37 model dispatches without serializing arbitrary MLIR
or rewriting the backend.

- Keep TTEX v1 builtins compatible with existing add/matmul tests.
- Add a dispatch kind and typed tensor descriptors: shape, element type,
  layout, binding role, and byte range.
- Add contraction/convolution attributes and a small ordered epilogue enum for
  bias, affine scale/shift, residual add, and ReLU.
- Add padding metadata for non-tile-aligned dimensions.
- Version and verify every new field before executable creation.

Exit: all 37 exports serialize and parse, even before every kind can execute.

Status as of 2026-09-16: TTEX v2 adds `MATMUL_LIKE` and `CONVOLUTION` dispatch
kinds, static logical/storage tensor shapes, binding ordinals, element offsets,
and static byte subspans. CHW, flattened channel-major, OI, and OIHW layouts
cover the standalone bottleneck. Contraction descriptors support direct,
scalar affine/ReLU, per-channel affine/ReLU, and scalar
affine/residual/ReLU forms. The runtime accepts all versions from v1 through
v8, validates tensor ranges and residual non-overlap, and executes padded
projections with direct or residual epilogues plus descriptor-driven
convolutions. TTEX v4 adds the `PADDED_COPY` descriptor,
TTEX v5 adds per-channel convolution affine vectors, TTEX v6 adds the narrow
stem max-pool descriptor, and TTEX v7 adds per-channel matmul-like affine
vectors. TTEX v8 completes the measured model contract with runtime offsets,
strides, runtime affine descriptors, and the three typed tail operations. The
R2 serialization exit is complete for the measured artifact.

### R3: Compiler Dispatch Classification

Objective: classify the existing fused executable-source IR deterministically.

- Recognize contraction and convolution semantics from indexing maps and
  iterator types, not export names.
- Recognize only the required pointwise epilogue expression trees.
- Reject ambiguous `linalg.generic` forms with one concise diagnostic.
- Emit a coverage test requiring 37 of 37 exports to receive a descriptor.

Exit: the full model compiles to a VMFB and runtime executable creation reports
which dispatch kinds are not implemented, if any.

Status as of 2026-09-16: exact indexing-map matching covers rank-3 projection,
flattened rank-2 projection, static 3x3 stride-1 and 7x7 stride-2 convolution,
and the bottleneck's residual affine-add-ReLU expression tree. The complete
four-dispatch bottleneck compiles to a Tenstorrent VMFB and loads in the mock
runtime. A direct static rank-3 F32 load/store matcher also classifies the input
padded copy. It also recognizes the exact 3x3 stride-2 stem max-pool reduction.
It also recognizes direct and residual per-channel affine/ReLU on rank-3 and
flattened matmul-like projections, 3x3 stride-2 convolution, runtime-derived
subspan offsets, the exact global reduction and scalar division, and the final
OI-weight classifier with bias. The external full-model audit classifies and
serializes 37 of 37 exports; the R3 coverage exit is complete.

### R4: Matmul-like Correctness Path

Objective: reuse the existing tiled multicore matmul work for 22 projection
executables and the final classifier.

- Support F32 model ABI through an explicit BF16 staging policy for M1.
- Pad M/N/K to tile boundaries and crop outputs.
- Support transposed weight indexing used by the model.
- Apply the required fused epilogues before writing F32-compatible results.
- Keep host staging acceptable for this correctness milestone.

Exit: every `matmul_like` dispatch and the classifier passes isolated reference
tests for the exact ResNet50 shapes.

Status as of 2026-09-16: the shape-reduced 64x64 first projection executes with
a 2x2 workgroup grid, explicit F32-to-BF16 staging on hardware, descriptor-based
CHW/OI gather and padded CHW scatter, and host affine/ReLU. The mock VMFB E2E
checks exact `0.6875` output and zero padding. The same TTEX v3 VMFB now passes
on a physical p100a Blackhole: the 64x10x10 F32 result has a zero border and
`0.6875` throughout the logical 64x8x8 region. Residual epilogues and
non-tile-aligned `M/N/K` padding are implemented in the correctness path.

The shape-exact `64x56x56` projection also compiles and executes in the mock
HAL with its per-channel affine/ReLU and `64x58x58` padded destination. Its 98
spatial tile rows are divided over a `2x7x1` workgroup grid, keeping the static
dispatch within the portable 8x8 core limit. The vector epilogue has descriptor
round-trip and channel-sensitive execution tests. Physical-device validation
of this TTEX v7 path remains open.

The first offline-codegen slice is also complete for this projection. A
hardware-enabled compiler build produces a TTEX v3 bundle containing three
compiler-generated kernel sources and target-specific tt-metal ELF products.
Measured device-less builds on 2026-08-24 produced:

| Target option | TTEX architecture | ELF markers | VMFB size |
| --- | --- | ---: | ---: |
| `p100a` | Blackhole | 180 | 42,526,129 bytes |
| `wormhole` | Wormhole B0 | 140 | 31,987,949 bytes |

The earlier architecture-agnostic bundle contained 320 ELF markers and was
74,501,505 bytes. `tt_e2e_compile_resnet50_projection` verifies that the VMFB
contains AOT ELFs and generated compute source but no path to tt-metal's test
kernels. `tt_e2e_check_target_arch_options` verifies CLI propagation, target
attributes, and rejection of unknown device IDs.

This remains a correctness bridge rather than the final codegen architecture:

- AOT generation covers elementwise BF16 add, fixed 32x32 BF16 matmul, static
  and dynamic tiled BF16 matmul, F32 matmul-like/convolution exports, and the
  final F32 linear classifier.
- One shape-independent matmul ELF set is reused across fixed, multicore, and
  dynamic shapes. Matrix dimensions and per-core tile counts are runtime args.
- Bundles are architecture-specific but still include every offline compile
  configuration produced for that architecture; narrowing by exact product or
  core topology remains future size work.
- Each AOT entry point materializes its artifacts once during executable
  creation. Dispatches reuse that immutable directory, and executable
  destruction removes it. Cross-executable artifact deduplication is not yet
  implemented.
- Compiler-produced TTEX v3 add and matmul entries use embedded AOT artifacts.
  Runtime source-path kernel creation and the legacy `TT-METAL` executable path
  have been removed. TTEX entries without AOT remain usable for descriptor and
  command-recording validation, but execution rejects them.
- Physical validation used KMD 2.10.0 and firmware bundle 19.4.1 on a p100a
  Blackhole exposing an 11x10 worker grid and 3984 MB DRAM. The runtime loads
  the architecture-specific TTEX v3 artifacts with source fallback disabled.
  Add, fixed 32x32 matmul, multicore 64x64 matmul, and dynamic 64x64 matmul
  produced the expected constant outputs 3, 64, 128, and 128 respectively.
- `tt_e2e_resnet50_projection_runtime` compiles for `p100a`, launches the
  physical Tenstorrent device, and checks the padded projection result. It is
  serialized and labeled `hardware;aot` in non-mock TTNN builds.

### R5: Convolution Correctness Path

Objective: execute the 11 spatial convolution descriptors.

- Bring up 1x1 projections through the matmul path first.
- Add 3x3 stride-1, 3x3 stride-2, and initial 7x7 stride-2 variants.
- Use static shape descriptors and fixed NCHW layout for M1.
- Add per-layer tests using weights and activations extracted from the model.

Exit: all stem and residual-block convolution dispatches match a CPU reference
within the selected BF16 tolerance.

Status as of 2026-09-16: all 11 measured convolution exports serialize with
compiler-generated matmul AOT bundles, including 7x7 stride-2 and 3x3
stride-1/stride-2 forms. Mock execution covers im2col, scalar and runtime
per-channel affine/ReLU, padding, cropping, and strided output scatter.
Physical-device validation of every exact model shape remains open.

### R6: Reduction and Full Graph

Objective: complete the tail and execute the public `main` entry point.

- Move max-pool from host execution to generated device code.
- Implement global-average-pool and post-reduction elementwise work.
- Validate fence, constant lifetime, and transient buffer reuse behavior.
- Compare final logits and top-1 result against a CPU reference.

Exit: `iree-run-module --device=tenstorrent --function=main` completes with
matching logits for one known input.

Status as of 2026-09-16: TTEX v8 classifies and executes the typed global-sum,
scalar-division, and biased classifier operations through correctness-first
host-assisted paths, and the full model compiles to VMFB. Full public `main`
execution remains open pending constant-file import, AOT bundle deduplication,
and end-to-end hardware validation against reference logits.

## M1 Hardcoding Boundary

The following may remain fixed for the first full-model smoke path:

- batch size 1 and all current static shapes;
- NCHW F32 public ABI with internal BF16 compute staging;
- Blackhole unit mesh and one device;
- ResNet50-required convolution strides, kernels, and epilogues only;
- host-assisted layout conversion, padding, and cropping;
- correctness tolerance instead of performance targets.

The following must be generalized now:

- executable dispatch kind instead of one add/matmul program ID;
- per-export binding and constant counts;
- tensor shapes, element types, layouts, and binding byte ranges;
- workgroup/core-grid metadata;
- safe handling of reused executables with different runtime offsets;
- TTEX version validation and unsupported-descriptor diagnostics.

Performance work such as device-resident layout conversion, weight prepacking,
program caching, and cross-dispatch fusion comes after the first full graph is
correct.
