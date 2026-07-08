# CUDA → Slang Migration Status

Honest per-file status of the migration of the PhysX 5.6.1 GPU pipeline
(61 `.cu` files, 62 `.cuh` headers, ~95k lines of device code) to Slang with
Vulkan/D3D12 backends.

**Legend**
- ✅ **ported + verified** — Slang kernel exists, runs on Vulkan, passes the
  verification suite against a CPU transcription of the CUDA original
- 🟡 **partially ported** — core paths ported + verified, listed gaps remain
- ⬜ **not started** — still CUDA-only

## gpucommon

| CUDA source | Kernels | Status | Slang |
|---|---|---|---|
| `radixSortImpl.cu` + `RadixSort.cuh` | radixSortMultiBlockLaunch\*, radixSortMultiCalculateRanksLaunch\*, radixSortCopyHigh32Bits | ✅ | `radixSort.slang` + `radixSort64.slang` (histogram + stable reorder + 64-bit key chaining; the `WithCount`/`WithoutCount`/`BlockDesc` entry variants collapse into push-constant differences; `radixSortDoubleCopyHigh32Bits` = two dispatches; single-warp in-block variants ⬜) |
| `reduction.cuh` (header) | warpScan/blockScan/warpReduction | ✅ | `pxgSlangScan.slang` (portable groupshared equivalents) |
| `MemCopyBalanced.cu` | MemCopyBalanced, clampMaxValue, clampMaxValues | ✅ | `memCopy.slang` + `clampValues.slang` (CopyDesc pointers → buffer byte offsets) |
| `utility.cu` | interleaveBuffers + deformable-skinning kernels (zero/areaWeighted/normalize normals, cloth Phong + softbody barycentric interpolation) | ✅ | `skinning.slang` (PxTypedBoundedData pointers → strided buffer views; float atomicAdd → portable CAS loop) |
| `updateCacheAndBound.cuh` (header) | updateBounds/getAbsPose/setTransformCache | ✅ | `updateBounds.slang` |
| other `.cuh` (atomic, copy, sparseRemove, contactReduction, gridCal, MemoryAllocator) | device helpers | ⬜ | ported on demand with their consumers (atomic.cuh AtomicAdd3 ✅ in `skinning.slang`) |

## gpusolver (rigid-body PGS/TGS solver)

| CUDA source | Kernels | Status | Slang |
|---|---|---|---|
| `preIntegration.cu/.cuh` | preIntegrationLaunch, initStaticKinematics | ✅ | `preIntegration.slang` |
| `integration.cu` + `integration.cuh` | integrateCoreParallelLaunch | ✅ | `integration.slang` (PGS variant incl. full sleep/freeze/wake state machine) |
| `integrationTGS.cu` | integrateCoreParallelLaunchTGS | ✅ | `integration.slang` (integrateCoreTGS: absolute velocities, delta-transform pose accumulation) |
| `preIntegrationTGS.cu` | preIntegrationLaunchTGS | ✅ | covered by `preIntegration.slang` — the TGS entry is the identical kernel with the runtime skipGravityApplication flag, which the port already exposes as a push constant |
| `solverMultiBlock.cu` + `solverBlock.cuh` | rigid PGS pipeline | 🟡 | `solverPGS.slang` — solveBlockPartition (solveContactBlock + solve1DBlock incl. compliant/accel-spring paths, mass-splitting reference counts) ✅, concludeBlocks ✅, writebackBlocks ✅ (force buffer, joint impulse/break writeback, friction patch export, threshold stream via atomics), writeBackBodies ✅, computeAverageSolverBodyVelocity + propagateSolverBodyVelocity ✅ (multi-slab velocity averaging incl. slab-count reset); remaining: ZeroBodies/markActiveSlab/initialization glue, articulation paths (`solver.cu` artiSolveBlockPartition, solveExt*), contact-error accumulator stats ⬜ |
| `solverMultiBlockTGS.cu` + `solverBlockTGS.cuh` | rigid TGS pipeline | 🟡 | `solverTGS.slang` — solveBlockPartitionTGS (rigid path of solveBlockUnified): solveContactBlockTGS (accumulated-delta separation, compliant/accel-spring restitutionXdt path, paired anisotropic friction + torsional anchors) and solve1DBlockTGS (delta-transform Jacobian re-anchoring, ortho-axis projection, spring modes, residuals) ✅; concludeBlocksTGS ✅ (conclude1DBlockTGS bias removal; the CUDA contact conclude body is compiled out and stays a no-op), writebackBlocksTGS ✅ (force buffer, joint impulse/break writeback, friction patch export, threshold stream), writeBackBodiesTGS ✅ (accumulated deltas → motion velocities), computeAverageSolverBodyVelocityTGS + propagateAverageSolverBodyVelocityTGS ✅ (6-component mass-splitting velocity averaging incl. slab-count reset, per-substep delta-pose quaternion integration with lock flags); remaining: ZeroBodiesTGS/initializeSolverVelocitiesTGS/applyTGSSubstepGravity glue, solveStaticBlockTGS, articulation paths ⬜ |
| `constraintBlockPrep.cu` + `contactConstraintBlockPrep.cuh` + `jointConstraintBlockPrep.cuh` | rigid PGS constraint prep | 🟡 | `constraintPrepPGS.slang` — contactConstraintBlockPrepareParallelLaunch (friction correlation: correlatePatches/getFrictionPatches/growPatches, and setupFinalizeSolverConstraintsBlock: contact Jacobians, resp/coeff terms, compliant contacts, tangent friction rows) ✅, jointConstraintBlockPrepareParallelLaunch (preprocessRows + orthogonalize, setUp1DConstraintBlock, intializeBlock1D: PGS solver constants, spring/bounce/drive-limit handling) ✅; articulation (`extContactConstraintPrepareLaunch`, ext 1D) paths ⬜ |
| `constraintBlockPrepTGS.cu` + `jointConstraintBlockPrepTGS.cuh` | rigid TGS constraint prep | 🟡 | `constraintPrepTGS.slang` — contactConstraintBlockPrepareParallelLaunchTGS (restitutionXdt/scaledBias TGS coefficients, torsional friction anchor, kinematic target-velocity bias) ✅, jointConstraintBlockPrepareParallelLaunchTGS (setUp1DConstraintBlockTGS + intializeBlock1DTGS: biasScale/velMultiplier/targetVel constants, maxBiasVelocity, ortho-axis header setup, solve-hint promotion) ✅; articulation paths ⬜ |
| `constraintBlockPrePrep.cu` | constraint pre-prep (partition remap, friction-patch counts) | ⬜ | produces the batch/partition layout the block prep consumes |
| `artiConstraintPrep2.cu` | articulation constraint prep | ⬜ | |
| `accumulateThresholdStream.cu` | force-threshold reporting | 🟡 | `thresholdStream.slang` — sort-key extraction for the two-pass (nodeIndexA, nodeIndexB) sort (initialRanksAndBodyIndexB/A) + stream reorganize ✅, device-wide segmented force accumulation (computeAccumulateThresholdStream + outputAccumulateThresholdStream + writeoutAccumulatedForcePerObject) ✅, exceeded-force detection + compaction (computeExceededForceThresholdElementIndice + outputExceededForceThresholdElementIndice) ✅; the cross-frame found/lost/persistent event diff (setThresholdElementsMask → createForceChangeThresholdElements) ⬜ |

## gpusimulationcontroller

| CUDA source | Kernels | Status | Slang |
|---|---|---|---|
| `algorithms.cu` | scanPerBlockKernel, addBlockSumsKernel, scanPerBlockKernel4x4, addBlockSumsKernel4x4, radixFourBitCountPerBlockKernel, radixFourBitReorderKernel, reorderKernel | ✅ | `scan.slang`, `scan4x4.slang`, `radixFourBit.slang`, `reorderData.slang` — the complete PxGpuRadixSort pipeline (PxgAlgorithms.cpp); PxU64-packed rank scans become uint4-lane scans, the PxU16 offsets buffer is widened to 32-bit |
| `updateTransformAndBoundArray.cu` | updateTransformCacheAndBoundArrayLaunch | 🟡 | `updateBounds.slang` — rigid-body path ✅ incl. all geometry types; articulation branch and `updateChangedAABBMgrHandlesLaunch`/`mergeTransformCacheAndBoundArrayChanges` ⬜ |
| `updateBodiesAndShapes.cu` | body/shape DMA + direct GPU API | 🟡 | `bodyUpdate.slang` (updateBodiesLaunch ✅, updateBodiesLaunchDirectAPI ✅ incl. CMassLocalPose body2World re-derivation, updateShapesLaunch ✅, updateBodyExternalVelocitiesLaunch ✅), `directApi.slang` (get/set pose, velocities, accelerations, force, torque ✅), setRigidDynamicGlobalPose ✅ in `updateBounds.slang` (incl. node-index binary search + per-shape bounds refresh); copyUserData = MemCopyBalanced ✅; articulation kernels (newArticulationsLaunch, updateArticulationsLaunch, updateJointsLaunch, D6 joint getters) ⬜ |
| `rigidDeltaAccum.cu`, `SDFConstruction.cu`, `softBody*.cu`, `FEMCloth*.cu`, `particlesystem.cu`, `diffuseParticles.cu`, `anisotropy.cu`, `isosurfaceExtraction.cu`, `sparseGridStandalone.cu` | deformables / particles | ⬜ | |

## gpubroadphase / gpunarrowphase / gpuarticulation

| Module | Files | Status | Notes |
|---|---|---|---|
| gpubroadphase | `broadphase.cu`, `aggregate.cu` | 🟡 | incremental SAP; heavy consumer of the ported radix sort + scans. `broadphaseSetup.slang` ports the foundation kernels: translateAABBsLaunch (float `PxBounds3` → sortable `PxgIntegerAABB` with contact-distance inflation, monotonic float encoding, axis bit-shift and env-ID packing) ✅, markCreatedPairsLaunch (append new bounds' projections + start/end sort handles, radix padding) ✅, initializeSapBox1DLaunch (SAP-box endpoint indices from the sorted handles) ✅, initializeRadixRanks (identity ranks for the endpoint radix sort) ✅, updateHandles (reorder the previous frame's handles through the sort permutation) ✅. `broadphaseReports.slang` ports the found/lost pair report accumulation: accumulateReportsStage_1 + accumulateReportsStage_2 (classify each pair as aggregate vs actor via `VolumeData::isSingleActor` and stably compact the four found/lost × agg/actor categories through a 2-stage between-block prefix) ✅, copyReports (actor found/lost reports → mapped output, overflow flag) ✅. `broadphaseIncrementalHistogram.slang` ports the incremental comparison-count stage: computeIncrementalComparisonHistograms_Stage1 + computeIncrementalComparisonHistograms_Stage2 (per-projection overlap-comparison counts as a difference of two prefix-sum histogram entries selected from the 2×2×3 [isStartHandle][isDownPass][axis] table, deleted projections excluded, followed by a device-wide two-pass exclusive prefix producing the per-projection running total, per-block totals and grand total per axis) ✅. `broadphaseEndPtHistogram.slang` ports the upstream producer of those histograms: computeEndPtsHistogram + outputEndPtsHistogram (firstPassEndPtsHistogram/secondPassEndPtsHistogram) — per-projection start/end/all-end (end2) classification with a packed three-counter (10-bit fields) block scan, promoted device-wide to the start/end/total-end point histograms and their per-block totals, plus the compacting scatter of non-deleted handles into the ordered start/end point handle arrays; incremental (with/without removals) and full-rebuild frame handling ✅; the incremental SAP sweep proper, the region/active/overlap-check histogram builders and pair generation ⬜ |
| gpunarrowphase | 25 files (GJK/EPA, convex/mesh/HF, midphase, contact management) | ⬜ | largest module; `updateBounds.slang` established the shape/geometry data-structure mapping (hull blob offsets) it needs |
| gpuarticulation | 4 files (Featherstone forward/inverse dynamics) | ⬜ | |

## Host-side runtime

| Component | Status | Notes |
|---|---|---|
| Compute device abstraction (buffers, pipelines, dispatch, sync) | ✅ | `PxgSlangComputeDevice` |
| Vulkan backend | ✅ | verified on llvmpipe (CI) — any Vulkan ≥ 1.2 device |
| D3D12 backend | 🟡 | complete implementation (WARP + hardware adapters, root constants, structured UAVs); compiles/executes only on Windows — per-kernel HLSL cross-compilation is ctest-enforced everywhere |
| Offline shader build (slangc → SPIR-V/DXIL) | ✅ | CMake rules in `tests/CMakeLists.txt` |
| Replacement of `PxgCudaMemoryAllocator`/`PxgHeapMemAllocator` pools | ⬜ | tests use direct buffer allocation |
| Integration into `PxgPhysXGpu` module loading (`PhysXGpu` DLL) | ⬜ | requires replacing the CUDA-context plumbing in `physxgpu/` |

## Verification

All ported kernels are covered by `tests/` (59 GPU test cases, run per commit
via ctest on a Vulkan device — llvmpipe in CI):

| Kernel | Reference | Checks |
|---|---|---|
| radix sort (32-bit) | `std::stable_sort` | keys + rank payload (exact), stability under duplicates, sizes 1 … 2²⁰ incl. non-multiples of 4 and chunking boundaries |
| radix sort (64-bit) | `std::stable_sort` of PxU64 | two chained 8-pass sorts + copyHigh32Bits gather, duplicate-heavy low/high words |
| scan | serial prefix sum | inclusive + exclusive, totals, sizes 1 … 2¹⁸+39 |
| scan 4x4 / PxGpuRadixSort | component-wise scan; `std::stable_sort` | int4x4 per-block scans, the full count → computeBlockSum → reorder pipeline incl. reorder-tracking map, partial bit widths (odd nibble rounding), duplicate-heavy keys, float4 payload gather |
| preIntegration | C++ transcription of `preIntegration.cuh` | velocities (momocity format), √(I⁻¹) tensor, solver body data, lock/gyro/damping/clamp branches, node-index bookkeeping |
| integration (PGS) | C++ transcription of `integration.cuh` | velocities, quaternion pose step, wake counter (tolerance), sleep/freeze flags (exact), bodySim write-back, non-zero deltaV offset |
| integration (TGS) | C++ transcription of `integrateCoreTGS` | TGS velocity-pool layout, delta-transform pose accumulation, sleep state machine (flags exact) |
| updateBounds | C++ transcription of `updateCacheAndBound.cuh` | AABBs for 5 geometry classes, transform cache, frozen/unfrozen/updated/activate/deactivate (exact) |
| MemCopyBalanced | `memcpy` | 40 random descriptors incl. empty payloads, byte-offset addressing |
| body/shape DMA | C++ transcription of `updateBodiesAndShapes.cu` | scatter by embedded index (bit-exact), direct-API partial copies (velocity .w-merge, first-transfer, body2Actor-change pose re-derivation), external velocity updates |
| direct GPU API | C++ transcription | pose composition (body2World·body2Actor⁻¹ and inverse), velocity/acceleration getters, velocity/force/torque setters incl. world-space inertia transform, setGlobalPose with sorted node-index binary search + bounds/cache refresh |
| PGS solver iteration | C++ transcription of `solverBlock.cuh` | mixed contact/joint batches with 1–32 lanes, rigid + compliant (accel-spring) restitution, static/dynamic friction clamping incl. broken flags, joint spring modes, impulse clamping, per-row residuals, >32-slab mass-splitting reference counts, velocity write-back |
| PGS conclude/writeback | C++ transcription | bias removal (coeff swap / friction bias zeroing), force-buffer writeback, joint impulse + breakage writeback, friction patch export, threshold stream (order-independent compare), report-threshold gating |
| velocity averaging | C++ transcription | per-slab delta averaging with active-slab counts (4/12/40 slabs incl. multi-word), slab-count reset, propagation into remapped slab slots |
| TGS solver iteration | C++ transcription of `solverBlockTGS.cuh` | 6-float4 velocity+delta lane layout, delta-derived separations, compliant (accel-spring) contacts, paired friction with sqrt-clamping + torsional anchors, joint re-anchoring through delta transforms, ortho-target projection, inequality/spring bias modes |
| TGS conclude/writeback | C++ transcription | joint bias-scale removal (KEEP_BIAS gating, spring coefficient zeroing; contact conclude verified as the CUDA no-op), force-buffer writeback, joint impulse + breakage writeback (uchar4-packed row counts), friction patch export incl. torsional-anchor row counts, threshold stream (order-independent compare) |
| TGS velocity averaging | C++ transcription | 6-component per-slab delta averaging with active-slab counts (4/12/40 slabs), slab-count reset, propagation into 3-float4 slab slots, per-substep delta-pose quaternion integration incl. per-body lock flags and kinematic bodies (position + velocity iterations), accumulated-delta motion-velocity writeback |
| constraint prep — contact (PGS + TGS) | C++ transcription of `contactConstraintBlockPrep.cuh` + `constraintPrepShared.cuh` | friction-patch correlation (correlate/getFrictionPatches previous-frame carry-over/growPatches anchor building), contact Jacobians with offset-slop zeroing, unit responses, rigid + compliant (accel-spring) coefficients, tangent friction rows, TGS restitutionXdt/scaledBias and torsional friction anchors, kinematic target-velocity bias; verified field-by-field incl. friction-anchor world transforms |
| constraint prep — 1D joint (PGS + TGS) | C++ transcription of `jointConstraintBlockPrep(TGS).cuh` + `DyCpuGpu1dConstraint.h` | row sort + orthogonalization (in-place Jacobian modification checked), √(I⁻¹)-scaled angular Jacobians, PGS/TGS solver constants (spring/accel-spring/bounce/hard), min/max impulse with drive-limits-as-forces, break impulses, PxU16/uchar4-packed row counts, TGS ortho-axis header setup + angular solve-hint promotion |
| threshold stream | C++ transcription of `accumulateThresholdStream.cu` | two-pass sort-key extraction (nodeIndexB then nodeIndexA) + word-exact stream reorganize (accumulatedForce preserved), device-wide segmented normal-force accumulation with per-pair write indices (37 … 2001 elements spanning the 32-block boundary), per-pair segment force via prefix-difference, exceeded-force flags/count and compacted exceeded-element array; parallel float-summation tolerances on prefix sums, exact on indices/flags/counts |
| broadphase setup | C++ transcription of `broadphase.cu` | float→integer AABB quantization (contact-distance inflation, monotonic `encodeFloat`, axis bit-shifts, environment-ID packing incl. invalid-ID ground-plane extents and min-even/max-odd parity), created-bounds projection/handle append with radix padding, SAP-box endpoint assignment from sorted handles, identity radix-rank init and handle reorder through a random sort permutation; found/lost pair report classification (aggregate vs actor) with stable between-block compaction and actor-report copy-out; all outputs compared bit-exactly |
| broadphase incremental histogram | C++ transcription of `broadphase.cu` | per-projection overlap-comparison counts as histogram-entry differences over all three axes (2×2×3 start/end × this/last-frame histogram selection by isStartHandle/isDownPass, deleted-projection exclusion, down-pass rank clamp exercised by out-of-range ranks), and the device-wide two-pass exclusive prefix (within-block running total, per-block totals and grand total) verified bit-exactly across the 32-block boundary with 2 iterations per block |
| broadphase endpoint histogram | C++ transcription of `broadphase.cu` | per-projection start / end / all-end (end2 incl. deleted) classification, the packed three-counter block scan promoted device-wide to the start/end/total-end point histograms and their per-block totals, and the compacting scatter of non-deleted handles into the ordered start/end point handle arrays — all three axes, over incremental (no-removal and with-removal frame-1 rescan) and full-rebuild (non-incremental) modes crossing the 32-block boundary; the deleted-only scatter guard (not the "new" bit) exercised by handles with the new flag set, verified bit-exactly |
| clampMaxValue(s) | scalar clamp | in-place counter clamping |
| skinning | C++ transcription of `utility.cu` | area-weighted vertex normals (atomic accumulation), Phong cloth interpolation incl. out-of-triangle uv projection and thickness clamping, barycentric softbody interpolation, buffer interleaving, strided-view addressing |

Floating-point comparisons use relative tolerance 1e-4…1e-5 (GPU `sin/cos/
sqrt` and FMA contraction differ legitimately from the CPU); all integer,
flag and index outputs are compared exactly.
