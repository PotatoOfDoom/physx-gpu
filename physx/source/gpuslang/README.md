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
│   ├── scan.slang                   device-wide prefix sums (algorithms.cu)
│   ├── preIntegration.slang         PGS solver pre-integration
│   ├── integration.slang            PGS solver integration + sleep state machine
│   └── updateBounds.slang           transform cache + world AABB update
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

The suite (17 GPU test cases + 8 DX12 cross-compile checks) verifies:

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
  activate/deactivate bookkeeping, compared field-by-field.

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
