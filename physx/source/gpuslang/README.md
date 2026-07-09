# PhysX GPU — Slang Migration (gpuslang)

This directory contains the CUDA→Slang migration of the PhysX GPU pipeline:
compute kernels rewritten in [Slang](https://shader-slang.org/), executed
through explicit **Vulkan** and **Direct3D 12** compute backends instead of the
CUDA driver API. The goal is byte-identical data structures and semantics with
the original CUDA kernels, verified by a self-contained test suite that runs
the Slang kernels on a real (or software) Vulkan device against CPU reference
implementations transcribed from the CUDA sources.

```
gpuslang/
├── include/
│   └── PxgSlangComputeDevice.h      device abstraction (buffers, pipelines, dispatch)
├── src/
│   ├── vulkan/PxgVulkanComputeDevice.cpp   Vulkan backend (SPIR-V, any 1.2+ device)
│   └── dx12/PxgDx12ComputeDevice.cpp       D3D12 backend (DXIL, Windows)
├── shaders/
│   ├── pxgSlangCommon.slang         math + GPU struct layouts (mirrors CUDA headers)
│   ├── pxgSlangScan.slang           portable workgroup scan primitives
│   ├── radixSort.slang              gpucommon radix sort (histogram + stable reorder)
│   ├── radixSort64.slang            64-bit key support (copyHigh32Bits chaining)
│   ├── radixSortGather.slang        rank-indexed 64-bit gather/copy family
│   ├── scan.slang                   device-wide prefix sums (algorithms.cu)
│   ├── scan4x4.slang                int4x4 16-counter scans (PxGpuRadixSort)
│   ├── radixFourBit.slang           PxGpuRadixSort count/reorder passes
│   ├── reorderData.slang            float4 payload gather (reorderKernel)
│   ├── preIntegration.slang         solver pre-integration (PGS + TGS)
│   ├── integration.slang            solver integration, PGS + TGS + sleep state machine
│   ├── updateBounds.slang           transform cache + world AABB update
│   ├── solverPGS.slang              PGS solver: iteration, conclude, writeback, averaging
│   ├── solverTGS.slang              TGS solver: iteration, conclude, writeback, averaging
│   ├── solverZeroBodies.slang       pre-solve body-velocity reset (PGS + TGS)
│   ├── solverInitVelocitiesTGS.slang TGS pre-solve velocity seeding (slab scatter)
│   ├── solverApplyGravityTGS.slang   TGS substep gravity + external accel (lock flags, inertia)
│   ├── solverDmaBack.slang           force-change elem DMA-back + constraint residual gather
│   ├── npPairManagement.slang        narrow-phase sparse-remove keep/drop core + manifold init
│   ├── npPairRemoveSwap.slang        narrow-phase sparse-remove Stage5 data swap (+ CvxTri)
│   ├── npSphereCollision.slang       sphere primitive contacts (sphere/plane/capsule/box)
│   ├── npCapsuleCollision.slang      capsule two-contact routines (plane/capsule, seg-seg)
│   ├── npMaterialCombine.slang       contact-patch material combine (friction/restitution/damping)
│   ├── npContactPatch.slang          contact-stream alloc + status-flag packing (atomic)
│   ├── npInsertPatch.slang           PxContactPatch header write (material combine + pack)
│   ├── npCompressContacts.slang      contact-count histogram + friction-patch offset rescale
│   ├── npCompressContacts2.slang     cross-block PxGpuContactPair compaction (Stage2)
│   ├── npTriangle.slang              triangle closest-point (Voronoi) + line-triangle test
│   ├── npSegTriangle.slang           segment-vs-triangle squared distance (capsule/mesh)
│   ├── npCapTriangle.slang           capsule-triangle contacts (endpoint + edge-edge)
│   ├── npTriTri.slang                triangle-triangle SAT separation test
│   ├── npBoxClip.slang               box-box clipping primitives (point-in-poly, segment-AABB)
│   ├── npBoxIncident.slang           box-box width-4 incident-face select + bounds4 reduction
│   ├── npBoxContacts.slang           box-box clipping core (calculateContacts: area/corner/edge)
│   ├── npBoxSat.slang                box-box 15-axis SAT + feature/sign selection
│   ├── npBoxFeature.slang            box-box feature->newTransform reference-frame setup
│   ├── npBoxReduce.slang             box-box Stage-2 contact compaction + 4-point reduction
│   ├── pxgSlangPrepCommon.slang     constraint-prep inputs + Dy 1D solver-constant helpers
│   ├── constraintPrepPGS.slang      PGS contact + 1D joint constraint prep
│   ├── constraintPrepTGS.slang      TGS contact + 1D joint constraint prep
│   ├── thresholdStream.slang        force-threshold reporting accumulation
│   ├── thresholdForceChange.slang   cross-frame found/lost/persistent force-change diff
│   ├── broadphaseSetup.slang        broad-phase integer-AABB + SAP-box setup
│   ├── broadphaseReports.slang      broad-phase found/lost pair report accumulation
│   ├── broadphaseIncrementalHistogram.slang  incremental-SAP comparison-count histograms
│   ├── broadphaseEndPtHistogram.slang        start/end-point projection histograms + ordered handles
│   ├── broadphaseIncrementalSap.slang        incremental sweep-and-prune found/lost pair generation
│   ├── broadphaseRegionHistogram.slang       region accumulator prefix-sum scan pair
│   ├── broadphaseCreateRegions.slang         region-span + overlap-check-count producer
│   ├── broadphaseActiveRegionHistogram.slang per-region handle compaction (flags/scan/scatter)
│   ├── broadphaseOverlapCheckHistogram.slang overlap-check counts + 64-bit region scan
│   ├── broadphaseNewBoundsPairs.slang        new-bounds region found-pair generation
│   ├── broadphaseMarkPairs.slang             incremental delete/update projection marking
│   ├── broadphaseClearNewFlag.slang          clear new-box flag + completion signal
│   ├── broadphaseAggregateBounds.slang       aggregate bounds union + changed-bitmap refresh
│   ├── broadphaseAggregateDirty.slang        dirty-aggregate double-buffer merge + bound scatter
│   ├── broadphaseAggregatePairsAdd.slang     agg-pair free-list slot alloc + overflow finalize
│   ├── broadphaseAggregateFinalize.slang     agg report copy + free-count + added/deleted setEmpty
│   ├── broadphaseAggregatePairsRemove.slang  agg-pair remove compaction (2-pass, into free list)
│   ├── broadphaseAggregateBoxPruning.slang   bipartite SAP box-pruning sweep (agg collision core)
│   ├── broadphaseAggPairCollision.slang      aggregate-pair collision dispatch (found/lost)
│   ├── broadphaseAggSelfCollision.slang      aggregate self-collision (found/lost within one agg)
│   ├── broadphaseAggSortProjections.slang    build sorted aggregate projection SAP structure
│   ├── bodyUpdate.slang             CPU->GPU body/shape DMA mirrors
│   ├── directApi.slang              Direct GPU API rigid-body get/set kernels
│   ├── memCopy.slang                balanced copy dispatcher (PxgCopyManager)
│   ├── clampValues.slang            counter clamp utilities
│   └── skinning.slang               deformable skinning (cloth Phong / softbody barycentric)
└── tests/                           verification suite (CMake + ctest)
```

See [MIGRATION_STATUS.md](MIGRATION_STATUS.md) for the per-kernel migration
coverage — this directory is the migration *architecture* plus a verified
vertical slice through it, not yet a complete port of all 61 `.cu` files.

## Design

### Device abstraction

`PxgSlangComputeDevice` replaces the CUDA driver layer
(`PxCudaContextManager` / `PxgKernelWrangler`):

| CUDA concept | gpuslang equivalent |
|---|---|
| `CUdeviceptr` + kernel pointer args | storage buffers bound by declaration order |
| kernel scalar arguments | push constants (Vulkan) / root constants (D3D12) |
| `cuLaunchKernel(grid, block)` | `dispatch(pipeline, gridX, gridY, gridZ, …)` |
| single CUDA stream ordering | `barrier()` between recorded dispatches |
| `cuStreamSynchronize` | `flush()` |
| CUBIN / fatbin modules | SPIR-V (Vulkan) / DXIL (D3D12), compiled offline by `slangc` |

Descriptor-based APIs cannot chase raw device pointers, which the CUDA code
embeds in descriptor structs (`PxgRadixSortDesc`, `PxgSolverCoreDesc`,
`PxgShape::hullOrMeshPtr`). The port therefore:

- binds each buffer member of a descriptor struct explicitly, and passes the
  scalar members as push constants;
- turns embedded data pointers (e.g. the cooked convex hull pointer) into
  **byte offsets into a bound geometry blob buffer** — the blob layout itself
  (convexFormat.h) is unchanged.

### Portability of warp-level code

The CUDA kernels lean heavily on `__shfl_sync` with a hard-wired 32-thread
warp. Vulkan/D3D12 subgroups vary in width (8–64) and the software devices
used for CI (llvmpipe, WARP) differ again, so the port replaces warp shuffles
with **groupshared-memory scans over 32-lane "virtual warps"**
(`pxgSlangScan.slang`), preserving the packed 8-bit/16-bit counter ranking
schemes of the originals. This is correctness-first; a wave-intrinsic
specialization (via `WaveActiveSum`/`WavePrefixSum` guarded by subgroup-size
queries) can be layered on later without touching call sites.

### Struct layout fidelity

`pxgSlangCommon.slang` re-declares the GPU data structures (`PxgBodySim`,
`PxgSolverBodyData`, `PxgShapeSim`, …) with the exact byte layout of the C++
headers, using only scalar/`float4`/`uint2` members so std430 rules cannot
introduce padding (a bare `float3` would be 16-byte aligned). The test suite
pins every layout with `static_assert`s against hand-mirrored C++ structs.

## Building and running the verification suite

Requirements: C++17 compiler, CMake ≥ 3.16, Vulkan SDK (headers + loader),
any conformant Vulkan ≥ 1.2 ICD (GPU driver **or** CPU implementations such as
Mesa llvmpipe — the suite runs entirely on llvmpipe in GPU-less containers),
and `slangc` (github.com/shader-slang/slang releases, or the copy inside the
`slangtorch` Python wheel).

```bash
cmake -S physx/source/gpuslang/tests -B build-gpuslang -DCMAKE_BUILD_TYPE=Release
cmake --build build-gpuslang -j
ctest --test-dir build-gpuslang --output-on-failure
```

On a machine with a discrete GPU, set `PXG_PREFER_SOFTWARE_DEVICE=0` to run
the suite on the real device. On Windows, `PXG_TEST_BACKEND=dx12` selects the
D3D12 backend (kernels compiled with `slangc -target dxil`).

The suite (119 GPU test cases + 148 DX12 cross-compile checks) verifies:

- **scan** — device-wide inclusive/exclusive prefix sums vs serial reference,
  including block-boundary sizes and multi-hundred-thousand element inputs;
- **radixSort** — full 8-pass key+rank sorts vs `std::stable_sort`, including
  stability under heavy duplicates, non-multiple-of-4 sizes and 1M elements;
- **preIntegration** — unconstrained velocity update, damping, velocity
  clamping, lock flags, gyroscopic forces and world-space √(I⁻¹) tensors vs a
  C++ transcription of the CUDA kernel, over randomized rigid-body populations;
- **integration** — post-solver velocity finalization, the complete
  sleep/freeze/wake-counter state machine (flag outputs compared exactly) and
  closed-form quaternion pose integration vs a C++ transcription;
- **updateBounds** — world pose composition and AABBs for sphere / capsule /
  box / convex-hull / fallback geometry, plus frozen/unfrozen/updated/
  activate/deactivate bookkeeping, compared field-by-field;
- **integration (TGS)** — the TGS velocity-pool layout and delta-transform
  pose accumulation, sharing the verified sleep state machine;
- **radix sort (64-bit)** — two chained 8-pass sorts through the
  copyHigh32Bits gather, checked against `std::stable_sort` of PxU64;
- **MemCopyBalanced / clamp utilities** — the PxgCopyManager balanced copy
  dispatcher over random descriptor batches;
- **skinning** — area-weighted normal accumulation (portable CAS float
  atomics), Phong cloth interpolation and barycentric softbody interpolation
  over strided vertex views;
- **PxGpuRadixSort** — the second, general-purpose radix sort
  (algorithms.cu): int4x4 histogram scans, the count → computeBlockSum →
  reorder pass composition with reorder-tracking, partial bit widths and the
  float4 payload gather kernel;
- **PGS solver iteration** — solveBlockPartition over mixed contact/joint
  batches: normal impulses with rigid and compliant (acceleration-spring)
  restitution, static/dynamic friction clamping with broken flags, 1D joint
  rows with spring modes and residual writeback, and the mass-splitting
  slab reference counts;
- **TGS solver iteration** — the sub-timestep solver: accumulated-delta
  contact separations, paired anisotropic friction with torsional anchors,
  and 1D joints re-anchored through per-body delta transforms with
  ortho-axis projection;
- **TGS conclude / writeback / velocity averaging** — joint bias removal,
  contact force and joint impulse/breakage writeback with friction patch
  export and the threshold stream, the 6-component mass-splitting velocity
  averaging with slab-count reset, the per-substep delta-pose quaternion
  integration (including body lock flags and kinematics) and the
  accumulated-delta motion-velocity writeback;
- **constraint prep (PGS + TGS)** — the contact and 1D-joint block
  preparation that turns cooked contact/joint data into the solver's block
  SoA constraints: friction-patch correlation (including previous-frame
  anchor carry-over), contact/friction Jacobians and response terms,
  compliant and TGS coefficient derivations with torsional friction, and the
  1D-joint row orthogonalization plus PGS/TGS solver-constant computation;
- **threshold stream** — the force-threshold reporting pipeline: two-pass
  (nodeIndexA, nodeIndexB) sort-key extraction and stream reorganize, the
  device-wide segmented scan that accumulates each body pair's normal force,
  and the exceeded-force detection and compaction, verified across the
  32-block boundary;
- **threshold force-change diff** — the cross-frame event pipeline: binary
  searching the sorted previous-exceeded set (with the ThresholdStreamElement /
  PxNodeIndex 64-bit comparators and the backward run-walk that matches
  shapeInteraction), the persistent-mask complement, the two-pass device-wide
  exclusive prefix over the previous|current|persistent mask, and the
  compaction of the found / lost / persistent force-change events;
- **broad-phase setup** — the incremental-SAP foundation: quantizing float
  AABBs into the sortable integer projection layout (contact-distance
  inflation, monotonic float encoding, axis bit-shifts and environment-ID
  packing), appending newly created bounds' projections and sort handles, and
  building the 1D SAP boxes from the sorted handles;
- **broad-phase reports** — the found/lost pair report accumulation: classifying
  each pair as aggregate vs actor (VolumeData::isSingleActor), the two-stage
  between-block prefix that stably compacts the four found/lost × agg/actor
  categories, and the actor-report copy-out with overflow flagging;
- **broad-phase incremental histograms** — the incremental-SAP comparison-count
  stage: per-projection overlap-comparison counts derived as differences of two
  prefix-sum histogram entries (selected from the 2×2×3 start/end ×
  this/last-frame × axis table by each projection's start/end kind and down/up
  pass, with the down-pass rank clamp and deleted-projection exclusion), and the
  device-wide two-pass exclusive prefix (per-projection running total, per-block
  totals and per-axis grand total) verified across the 32-block boundary;
- **broad-phase endpoint histograms** — the upstream producer of those
  histograms: per-projection start / end / all-end classification with a
  packed three-counter block scan (10-bit fields), promoted to the device-wide
  start/end/total-end point histograms and their per-block totals, and the
  compacting scatter of non-deleted handles into the ordered start/end point
  handle arrays, over incremental (with and without the removed-handle rescan)
  and full-rebuild modes;
- **broad-phase incremental SAP** — the sweep-and-prune core
  (performIncrementalSAP) that consumes the two histogram stages: binary-search
  work distribution over the per-projection comparison prefix (grid-stride loop),
  up/down-sweep swap-partner resolution through the ordered start/end handles,
  integer-AABB overlap-change detection (new/old bounds inverted by sweep
  direction, with first-axis-of-separation dedup and group/environment-ID
  filtering) and the atomic-compacted found/lost pair emission, verified by
  comparing the produced pair sets and counts against a transcription of the
  kernel logic;
- **broad-phase region histogram** — the region-accumulator prefix-sum scan pair
  (the shared computeHistogram/outputHistogram helpers behind the four
  start-region/region wrappers): a plain device-wide two-pass exclusive prefix
  sum plus grand total over a flat integer array, verified against
  `std::exclusive_scan` across warp/block/grid-stride boundary sizes and a
  wide-value case;
- **broad-phase create regions** — the region producer (createRegionsKernel)
  that maps each box's per-axis endpoint positions through the total-end-point
  histogram into a coarse min/max region-index span, writing regionRange, the
  collision-free start-region scatter and the per-projection region
  overlap-check counts that the region prefix-sum scan then accumulates;
- **broad-phase active-region histogram** — the per-region handle-compaction
  pipeline: stamping start/new flags into each region slice a box touches
  (binary-search work distribution + region-cell decode), the in-place
  per-region exclusive prefix of the packed 16/16 start+active flags (a
  512-thread block scan with block-accumulator carry), and the scatter of each
  box's handle into the compacted ordered start/active handle arrays, verified
  end-to-end against a transcription over a synthetic region layout;
- **broad-phase overlap-check histogram** — the overlap-check counting +
  scan trio: the per-comparison overlap-candidate count from the ordered-handle
  offset tables (with the handle/region record), and the device-wide **64-bit**
  (`PxU64` → `uint64_t`) two-pass exclusive prefix sum + grand total, verified
  against a `uint64` exclusive scan whose total exceeds 2³² to exercise the
  64-bit path (the Vulkan device enables `shaderInt64` when available);
- **broad-phase new-bounds pairs** — the final region kernel
  (generateFoundPairsForNewBoundsRegion) that consumes the whole region
  pipeline: a 64-bit grid-stride binary-search over the overlap-check total,
  region-slice candidate resolution from the compacted ordered handle lists, the
  max-of-min-corners dedup, group/environment filtering, integer-AABB
  intersection and atomic-compacted found-pair output — verified end-to-end by
  rebuilding the entire region pipeline on the CPU to feed consistent inputs and
  comparing the produced pair set against a transcription.

## Known limitations

- Kernel coverage is the vertical slice listed in MIGRATION_STATUS.md; the
  remaining CUDA modules (narrowphase, broadphase, articulations, softbody,
  particles, …) still require porting on this foundation.
- The D3D12 backend compiles only on Windows and has not been executed in this
  (Linux) development environment; the per-kernel HLSL cross-compilation is
  enforced by ctest on all platforms.
- The articulation branch of `updateTransformCacheAndBoundArrayLaunch` is not
  ported yet (it walks per-articulation link arrays through embedded device
  pointers, which needs the articulation data redesign first).
- Buffers use host-visible memory for simplicity (ideal for the verification
  suite and software devices); a production integration wants device-local
  heaps with staging rings.
- Shaders are built with `-O0` because the trimmed `slangc` used in CI lacks
  the optional `slang-glslang` library that backs slangc's spirv-opt pass;
  drivers still run their own optimizers. Use a full slang release for
  optimized offline builds.
