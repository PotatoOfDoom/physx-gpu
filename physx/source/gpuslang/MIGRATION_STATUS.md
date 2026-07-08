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
| `radixSortImpl.cu` + `RadixSort.cuh` | radixSortMultiBlockLaunch\*, radixSortMultiCalculateRanksLaunch\* | ✅ | `radixSort.slang` (histogram + stable reorder; the `WithCount`/`WithoutCount`/`BlockDesc` entry variants collapse into push-constant differences; single-warp variants + `radixSortCopyHigh32Bits` for 64-bit keys ⬜) |
| `reduction.cuh` (header) | warpScan/blockScan/warpReduction | ✅ | `pxgSlangScan.slang` (portable groupshared equivalents) |
| `MemCopyBalanced.cu` | balanced copy dispatcher | ⬜ | |
| `utility.cu` | fill/clear/misc utilities | ⬜ | |
| `updateCacheAndBound.cuh` (header) | updateBounds/getAbsPose/setTransformCache | ✅ | `updateBounds.slang` |
| other `.cuh` (atomic, copy, sparseRemove, contactReduction, gridCal, MemoryAllocator) | device helpers | ⬜ | ported on demand with their consumers |

## gpusolver (rigid-body PGS/TGS solver)

| CUDA source | Kernels | Status | Slang |
|---|---|---|---|
| `preIntegration.cu/.cuh` | preIntegrationLaunch, initStaticKinematics | ✅ | `preIntegration.slang` |
| `integration.cu` + `integration.cuh` | integrateCoreParallelLaunch | ✅ | `integration.slang` (PGS variant incl. full sleep/freeze/wake state machine) |
| `integrationTGS.cu` | integrateCoreParallelLaunchTGS | ⬜ | shares ~90% of `integration.slang`; needs the TGS deltaBody2World path |
| `preIntegrationTGS.cu` | TGS pre-integration | ⬜ | analogous to `preIntegration.slang` |
| `solver.cu`, `solverMultiBlock.cu`, `solverMultiBlockTGS.cu` | constraint solver iterations | ⬜ | largest remaining solver work item |
| `constraintBlockPrePrep.cu`, `constraintBlockPrep.cu`, `constraintBlockPrepTGS.cu` | constraint prep | ⬜ | |
| `artiConstraintPrep2.cu` | articulation constraint prep | ⬜ | |
| `accumulateThresholdStream.cu` | threshold streams | ⬜ | needs radix sort + scan (both ported) |

## gpusimulationcontroller

| CUDA source | Kernels | Status | Slang |
|---|---|---|---|
| `algorithms.cu` | scanPerBlockKernel, addBlockSumsKernel | ✅ | `scan.slang` (`4x4`/radix-reorder variants ⬜) |
| `updateTransformAndBoundArray.cu` | updateTransformCacheAndBoundArrayLaunch | 🟡 | `updateBounds.slang` — rigid-body path ✅ incl. all geometry types; articulation branch and `updateChangedAABBMgrHandlesLaunch`/`mergeTransformCacheAndBoundArrayChanges` ⬜ |
| `updateBodiesAndShapes.cu` | body/shape DMA updates | ⬜ | |
| `rigidDeltaAccum.cu`, `SDFConstruction.cu`, `softBody*.cu`, `FEMCloth*.cu`, `particlesystem.cu`, `diffuseParticles.cu`, `anisotropy.cu`, `isosurfaceExtraction.cu`, `sparseGridStandalone.cu` | deformables / particles | ⬜ | |

## gpubroadphase / gpunarrowphase / gpuarticulation

| Module | Files | Status | Notes |
|---|---|---|---|
| gpubroadphase | `broadphase.cu`, `aggregate.cu` | ⬜ | incremental SAP; heavy consumer of the ported radix sort + scans |
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

All ported kernels are covered by `tests/` (17 GPU test cases, run per commit
via ctest on a Vulkan device — llvmpipe in CI):

| Kernel | Reference | Checks |
|---|---|---|
| radix sort | `std::stable_sort` | keys + rank payload (exact), stability under duplicates, sizes 1 … 2²⁰ incl. non-multiples of 4 and chunking boundaries |
| scan | serial prefix sum | inclusive + exclusive, totals, sizes 1 … 2¹⁸+39 |
| preIntegration | C++ transcription of `preIntegration.cuh` | velocities (momocity format), √(I⁻¹) tensor, solver body data, lock/gyro/damping/clamp branches, node-index bookkeeping |
| integration | C++ transcription of `integration.cuh` | velocities, quaternion pose step, wake counter (tolerance), sleep/freeze flags (exact), bodySim write-back, non-zero deltaV offset |
| updateBounds | C++ transcription of `updateCacheAndBound.cuh` | AABBs for 5 geometry classes, transform cache, frozen/unfrozen/updated/activate/deactivate (exact) |

Floating-point comparisons use relative tolerance 1e-4…1e-5 (GPU `sin/cos/
sqrt` and FMA contraction differ legitimately from the CPU); all integer,
flag and index outputs are compared exactly.
