# PhysX GPU — Slang/Vulkan port (gpuslang)

This directory contains the port of the PhysX 5.6.1 GPU pipeline from CUDA to
[Slang](https://shader-slang.org/). The kernels run on an explicit **Vulkan**
compute backend, with a **Direct3D 12** backend alongside it, instead of the
CUDA driver API. The unmodified engine host code (the `gpu*` modules:
broadphase, narrowphase, solver, articulations, simulation controller,
softbody, cloth, particles) drives the Slang kernels through a CUDA-driver
shim. The goal is bit-identical results with the CUDA kernels under an IEEE
reading of the CUDA source. It is checked bitwise against host oracles
transcribed from the CUDA, and end to end by stepping real `PxScene`s.

Status in one paragraph: **every kernel this build can launch is mapped and
wired** (P9.95 complete). The full GPU pipeline steps real scenes over Vulkan,
and the suite and scene harness are green. Development and verification have
happened **only on Linux with Mesa llvmpipe** (software Vulkan). The port has
never run on real GPU hardware. The D3D12 backend has never been built or
executed. The SDK's own build scripts produce the port as a standalone
`PhysXGpu` module with no CUDA toolkit (`PX_GPU_BACKEND=VULKAN` presets, see
[The SDK build](#the-sdk-build-px_gpu_backendvulkan)); its Windows build has
never been compiled with MSVC.

Status documents:

- [RUNTIME_STATUS.md](RUNTIME_STATUS.md) is the authoritative tranche log
  (Phase 2, P0–P9.x). Its header block summarises every tranche, newest first,
  with the current test and scene counts. The detailed entries since about P9.4x
  are in German.
- [MIGRATION_STATUS.md](MIGRATION_STATUS.md) is the Phase 1 per-file record of the
  CUDA math-core port. Phase 1 is closed. Its Status column now reflects the
  current per-file state, which matches the manifest and the checkers below.
  Unreachable kernels are marked ⛔, with the reason. The long notes in its rows
  are the Phase 1 log.
- [RUNTIME_PLAN.md](RUNTIME_PLAN.md) is the original Phase 2 plan.

```
gpuslang/
├── include/
│   ├── PxgSlangComputeDevice.h     device abstraction (buffers, textures, pipelines, dispatch, sync)
│   ├── PxgCudaShim.h               CUDA driver-API surface over the device (flat address map, launches)
│   ├── PxgSlangHeapAllocator.h     sub-allocator
│   ├── PxgSlangEnv.h               environment switches shared by bridge and tests (PXG_DEVICE_V2)
│   └── PxgSlangKernelManifest.h    GENERATED: CUDA kernel id -> SPIR-V module + entry (tools/gen_kernel_manifest.py)
├── src/
│   ├── vulkan/PxgVulkanComputeDevice.cpp   Vulkan backend (SPIR-V, Vulkan >= 1.2)
│   ├── dx12/PxgDx12ComputeDevice.cpp       D3D12 backend (DXIL, Windows only; never built, see below)
│   └── PxgCudaShim.cpp
├── bridge/                         the engine hook: PxPhysXGpu C-ABI factory (PhysXGpuVulkan.cpp),
│                                   PxCudaContext/Manager over the shim, kernel table + launch ABI rows
│                                   (PxgVulkanKernelAbi.cpp), driver/kernel-init shims, memory manager
├── cmake/                          shared by the SDK build (source/compiler/cmakegpuvulkan), tests and sdktest:
│                                   PxgShaders (slangc + post-processing), PxgShaderList (every shader),
│                                   PxgEngineShaders (the manifest's subset), PxgVulkanGpuSources (module sources)
├── nocuda/                         CUDA header stand-ins so the gpu* host code compiles without a toolkit
├── shaders/                        555 .slang files: pxgSlang* shared modules (79), the *Bda.slang engine
│                                   kernels (buffer-device-address ABI), Phase-1 cores and fixture shaders
├── tests/                          standalone verification suite (CMake + ctest) and the host oracles (*Ref.h)
├── sdktest/                        full-SDK builds: sdk-step (CPU) and gpu-sdk-step (GPU scene harness)
└── tools/                          scene/suite/validation runners, mutation sweeps, static checkers, foldscan
```

## Coverage

The kernel registry (`PxgKernelNames.h`) has **501** kernels.
`include/PxgSlangKernelManifest.h` maps **428** of them to an entry point in 178
SPIR-V modules. Of those, **415** have a launch-ABI row in
`bridge/PxgVulkanKernelAbi.cpp`, which makes them launchable by the engine
(`tools/check_launch_abi.py`). Everything that is not both mapped and wired is
unreachable in this build. `tools/check_unmapped_reachability.py` and the
ratchet files classify these kernels:

| Kernels | State | Why it is unreachable |
|---|---|---|
| 13 | mapped, no ABI row (`KNOWN_UNWIRED`) | Owned only by objects the PhysicsGpu factory (`physxgpu/src/PxgPhysXGpu.cpp`) constructs. That factory is not compiled here, and `PxGpuCreatePhysicsGpu()` returns NULL. These are the generic scan/radix utilities (incl. `reorderKernel`, which has no caller even upstream), `interleaveBuffers`, and the five skinning kernels. |
| 58 | unmapped, STANDALONE | App-facing features behind the same factory: isosurface extraction, SDF builder, sparse-grid provider, anisotropy, smoothed positions. |
| 11 | unmapped, ORPHAN | Nothing launches them, even upstream. |
| 4 | unmapped, ENGINE-named but dead (`tools/engine_reachable_unmapped.txt`) | `trimeshHeightfieldNarrowphase` sits in a dispatch bucket that no classifier produces. `sb_plasticDeformLaunch`/`sb_plasticDeformLaunch2` have a single caller, and it is commented out upstream. `copyUserData` belongs to an owner class that is never constructed. |

Both lists are ratchets: a new unwired or reachable-unmapped kernel fails the
build, and so does a stale entry. The articulation branch of
`updateTransformCacheAndBoundArrayLaunch` is ported (it reads the link poses out
of the articulation pool).

## Design

### Device abstraction and the CUDA shim

`PxgSlangComputeDevice` replaces the CUDA driver layer. `PxgCudaShim` and the
bridge implement the `PxCudaContext` / `PxCudaContextManager` surface the engine
calls on top of it:

| CUDA concept | gpuslang equivalent |
|---|---|
| `CUdeviceptr` (flat address) | Vulkan buffer device address; the shim keeps an interval map address → (buffer, offset) |
| `cuLaunchKernel` parameter pack | passed through **unchanged** as the push-constant block. Engine kernels (`*Bda.slang`) dereference the embedded addresses as BDA pointers, so the CUDA pointer-passing model is kept 1:1 |
| `cuLaunchKernel(grid, block)` | `dispatch` / `dispatchIndirect` with the CUDA block shape as `numthreads` |
| stream ordering / `cuStreamSynchronize` | recorded dispatches with barriers; `flush()`; timeline semaphores in device-local mode |
| `cuTexObject` / `tex3D<float>` | bindless sampled images (set 1, binding 0), with filtering emulated in the shader (below) |
| CUBIN / fatbin | SPIR-V (Vulkan) / DXIL (D3D12), compiled offline by `slangc` |

The launch-ABI row for each kernel fixes the size of its argument pack. The
shim's size check is exact. A mismatched pack is refused and never runs
silently.

The early Phase-1 fixture kernels (radix sort, scans, the first solver and
broadphase ports) still use the older model: each buffer member of a
descriptor struct is bound explicitly, and embedded pointers become byte offsets
into a bound blob.

### Memory modes

The device has two memory modes, and the suite runs in both:

- **Default:** host-visible buffers.
- **`PXG_DEVICE_V2=1`:** device-local buffers with staged transfers and
  timeline-semaphore synchronisation.

The engine bridge and `gpuslang-tests` parse `PXG_DEVICE_V2` with the same
helper (`include/PxgSlangEnv.h`):

- unset, empty or `0`: default mode;
- `1`: device-local mode;
- any other value: default mode, with a one-time warning on stderr that names
  the value.

The bridge prints the chosen mode once at startup, for example
`[PhysXGpuVulkan] device: llvmpipe (...), memory mode: host-visible (default)`.
`gpuslang-tests` appends `[v2: device-local + staged]` to its backend line.
Earlier versions of the bridge treated any set value, `0` included, as on.

### Warp-level code

The CUDA kernels assume a 32-lane warp (`__shfl_sync`, ballots, warp scans).
The port emulates these with **groupshared memory over 32-lane virtual warps**
(`pxgSlangScan` and the warp/atomic libraries), so no engine kernel depends on
the subgroup width. `tools/check_subgroup_ops.py` pins this; only three probes
use subgroup operations. Workgroup barriers need block-uniform control flow,
whereas CUDA only needs warp-uniform flow. `check_barrier_uniformity.py` and
`check_helper_uniformity.py` check this statically.

### Struct layout fidelity

The GPU structs are re-declared with the exact byte layout of the C++ headers,
using scalar, `float4` and `uint2` members so that std430 cannot insert padding.
Layouts are pinned by `static_assert`s and the `*DescLayout` / `npAbiLayout`
suites. `check_push_block.py` overlays every shader push block on its ABI row,
placing the arguments the way the bridge does, and compares member names
against the CUDA parameter names.

### `tex3D` (SDF) emulation

Since P9.96, CUDA's `tex3D<float>` (unnormalised coordinates, clamp-to-edge,
trilinear) is emulated in `pxgSlangTex3d` from eight texel loads, with **no
sampler**. The emulation follows the CUDA Programming Guide's linear-filtering
formula: clamp the coordinate to [0.5, N−0.5], compute `x−0.5` and `floor`,
round α to the nearest 1/256 with ties going up, and form the eight-term sum.
On llvmpipe, UNORM decoding was measured exhaustively and matches the engine's
own decoder. The emulation costs speed: the SDF fixture suite is about 2.5×
slower, and the `sdfmesh` scene about 11% slower.

## Verification

The standard is **bitwise** comparison, for floats as well as integers. Since
P9.86, the float-equality helper `eqFZ` is bitwise, including the sign of zero.
Some older Phase-1 fixtures still carry relative tolerances. Those cover
order-nondeterministic float sums such as parallel scans and atomic
accumulation.

- **Literal host oracles.** The reference for each kernel is a C++ transcription
  of the CUDA, often written over the engine's own `PxVec3`/`PxQuat`/
  `PxTransform`. The test binary is compiled with `-ffp-contract=off`.
- **Bitwise IEEE reading of the CUDA.** slangc emits no `NoContraction`, and
  Vulkan drivers may contract. Every float-computing engine shader therefore has
  its SPIR-V stamped with `NoContraction` (`tools/spv_nocontract.py`) and
  `SignedZeroInfNanPreserve`. Spelling is kept literal: `PxMin`/`PxMax` are
  selects, `transformInv` is a conjugate rotate, and the association order of
  sums matches the CUDA. The sign of zero is treated as observable. This rests on
  measured llvmpipe facts recorded in RUNTIME_STATUS:
  - llvmpipe folds `0 - x` and `x + 0` even under `SignedZeroInfNanPreserve`;
  - an undecorated `OpDot` sums `x+(y+z)`, while a stamped one sums `(x+y)+z`
    like the CUDA.

  P9.86–P9.94 audited every engine-reached file for these hazards. Directed
  fixtures, each with a mirror twin, kill the reverts.
- **Mutation sweeps.** No change ships without a sweep showing that the fixtures
  discriminate. Each runner restores the source and the SPIR-V after every
  mutant. Per-run timeouts are derived from the measured baseline
  (max(floor, 3× baseline)), and exit code 137 at the bound counts as a hang.
  - `tools/run_mutation_sweep.sh <shader-stem> <suite> <mutants-file>` sweeps
    shaders (`MUTATION_SWEEP_SPV_ONLY=1` for one `.spv` per mutant).
  - `tools/run_scene_mutation_sweep.sh <shader-stem> <scene> <mutants-file>`
    sweeps engine wiring whose only observer is a scene.
  - `tools/run_host_mutation_sweep.py <source> <ctest-regex> <mutants-file>`
    sweeps host C++.
- **Static checkers.** There are 20 `tools/check_*.py` checkers, all run as
  ctests. Most of them run a self-test or negative control first, so a green
  result means the checker is working, not just quiet. They cover:
  - launch-ABI rows and host packs vs. rows vs. CUDA signatures (`check_launch_abi*.py`);
  - push-block overlay and names (`check_push_block.py`);
  - unmapped-kernel reachability and disabled/twin kernels;
  - barrier and helper uniformity, block/grid-stride coverage and fixture grids;
  - groupshared limits, subgroup ops, speculated loads and sub-word stores;
  - vector/component divide and `length` spellings, and geometry-type constants;
  - MSVC portability of the Vulkan PhysXGpu's sources (`check_msvc_portability.py`).
- **Foldscan.** `tools/foldscan_all.py <build-gpuslang> <out> [-j N]` scans the
  compiled `-O0 -g2` SPIR-V of every engine-reached shader for literal-zero fold
  hazards: `FSub(0,x)`, `FAdd(x,0)`, and float `FMin`/`FMax`/`FClamp`. Results
  are compared against the reviewed baseline `foldscan_reviewed.json`. The
  expected state is `UNREVIEWED 0`.
- **Khronos validation (P9.98).** `tools/run_validation.sh` runs the suite (both
  memory modes) and scene rows under `VK_LAYER_KHRONOS_validation` (baseline
  VVL 1.3.275), with `tools/validation/vk_layer_settings.txt` (core +
  synchronisation + thread safety). It fails on any message not covered by
  `tools/validation_waivers.txt`. It also fails on the port's own signals:
  - a refusal outside the fixtures that provoke it;
  - a push range widened for an entry the waivers do not list;
  - an allocation peak above the device limit.

  The synchronisation checker cannot follow buffer-device-address accesses, so
  ordering between engine kernels is not validated by the layer.
- **Runtime limit refusals (P9.98/P9.99).** Before building a pipeline, the
  Vulkan device reads each entry point's SPIR-V and checks it against the device
  limits: push block, groupshared size, workgroup size and invocations, and a
  required subgroup size against both the device and the workgroup shape.
  Dispatch grids are also checked against `maxComputeWorkGroupCount`. A
  violation is refused with a reason on stderr (handle 0; the shim reports a
  refused launch). It does not crash. Failed allocations and out-of-range
  transfers are refused in the same way. Smaller devices can be simulated with
  `PXG_VK_LIMIT_PUSH_CONSTANTS`, `PXG_VK_LIMIT_SHARED_MEMORY`,
  `PXG_VK_LIMIT_WORKGROUP_INVOCATIONS`, `PXG_VK_LIMIT_WORKGROUP_SIZE` and
  `PXG_VK_LIMIT_WORKGROUP_COUNT` (these can only lower a limit). The port's
  largest push block is 192 B (some drivers allow only 128). Its largest
  groupshared footprint is 31744 B (the Vulkan minimum is 16384).

### The test suite (`ctest`): 892 tests

| Group | Count | What it is |
|---|---|---|
| `gpuslang.<suite>` | 512 | GPU fixture suites in `gpuslang-tests`, each with many cases: Phase-1 cores; engine-ABI kernel fixtures for broadphase, narrowphase (convex/mesh/heightfield/SDF/tri-tri), solver PGS/TGS, articulations, softbody, FEM cloth and particles (PBD, diffuse); the runtime and shim (`p0`–`p6`, `p3CudaContext`, `p4Loader`, `pipelineRefusal`, …) |
| `gpuslang.hlsl.<entry>` | 356 | D3D12 path: `slangc -target hlsl -profile cs_6_0` cross-compile of each listed entry (HLSL only, no DXIL, no execution) |
| tool tests | 24 | the 20 checkers, `sweepTimeoutPolicy`, `sweepRunners`, `vvlTriage`, and `validationSubset` (skipped, exit 77, when no validation layer is found) |

### The scene harness (`gpu-sdk-step`, `tools/run_scenes.sh`): 85 rows

`gpu-sdk-step` builds a real `PxScene` from the CPU SDK with
`eENABLE_GPU_DYNAMICS` and the GPU broadphase. The engine's GPU module pointer
is aimed at the bridge's statically linked `PxPhysXGpu` factory, in place of
the CUDA `libPhysXGpu_64.so`. Every `Pxg*` subsystem is then constructed over
Vulkan and stepped. Each scene checks physical outcomes: rest heights, contact
and event sequences, and bit-identity invariants. The reference is either
analytic or the CPU solver.

- Scenes are selected with `PXG_SCENE=<name>` and stepped with `PXG_GPU_STEP=1`.
  The solver is TGS by default and PGS with `PXG_SOLVER=pgs`.
- `tools/run_scenes.sh` scores **85 rows**: 64 named scenes plus 21 explicit
  variants. The variants are:
  - the settling bag;
  - softbody on mesh / convex-core / heightfield / SDF-mesh floors;
  - `softbodyhex` box;
  - 9 PGS rows;
  - 3 `slabgrid` modes;
  - 3 `particle2` arms.
- Every run ends with a scored refusal control: `box` under
  `PXG_VK_LIMIT_WORKGROUP_INVOCATIONS=128` must fail cleanly with rc 2 and must
  not crash.
- A missing verdict line is scored as its own outcome and never counted as a
  pass.
- Runner options:
  - `--strict` sets `PXG_REQUIRE_MAPPED=1`, so any launch of an unmapped kernel
    fails the scene;
  - `--resume <log>` re-scores the rows that log already lists as ok;
  - `--diagnostics` adds the unscored control modes.

A full run takes over an hour on llvmpipe.

## Building and running

Requirements:

- C++17 compiler and CMake ≥ 3.16;
- Vulkan SDK headers and loader;
- a Vulkan ≥ 1.2 ICD (Mesa llvmpipe is what everything has been run on). The
  engine path needs `bufferDeviceAddress` and `shaderInt64`, and SDF textures
  need descriptor indexing;
- `slangc` (github.com/shader-slang/slang releases; developed with 2026.14.1);
- Python 3 for the tools.

```bash
# 1. Kernels + verification suite (also builds the shaders into build-gpuslang/shaders)
cmake -S physx/source/gpuslang/tests -B build-gpuslang -DCMAKE_BUILD_TYPE=Release
cmake --build build-gpuslang -j
ctest --test-dir build-gpuslang -j4 --output-on-failure      # sets PXG_SHADER_DIR itself
sh physx/source/gpuslang/tools/run_suite.sh build-gpuslang   # both memory modes, archives failures

# 2. Full SDK + GPU scene harness (CPU SDK and the unmodified gpu* host modules, no CUDA toolkit)
cmake -S physx/source/gpuslang/sdktest -B build-sdk -DCMAKE_BUILD_TYPE=Release
cmake --build build-sdk -j --target gpu-sdk-step
cd build-sdk
PXG_SHADER_DIR=../build-gpuslang/shaders PXG_SCENE=box PXG_GPU_STEP=1 ./gpu-sdk-step
sh ../physx/source/gpuslang/tools/run_scenes.sh              # all 85 rows + refusal control
cd ..

# 3. Validation-layer run (needs VK_LAYER_KHRONOS_validation; PXG_VVL_DIR = extracted
#    vulkan-validationlayers package if none is installed)
sh physx/source/gpuslang/tools/run_validation.sh subset            # fast, both modes
sh physx/source/gpuslang/tools/run_validation.sh suite             # whole suite, both modes
sh physx/source/gpuslang/tools/run_validation.sh 'scene:box|PXG_SCENE=box'
```

Notes:

- If `slangc` is not on `PATH`, pass `-DPXG_SLANGC=/path/to/slangc`.
- Running `build-gpuslang/gpuslang-tests <suite>` directly needs
  `PXG_SHADER_DIR=<build-gpuslang>/shaders`. `run_scenes.sh` and
  `run_validation.sh` find `../build-gpuslang/shaders` themselves.
- `gpuslang-tests` prefers a software device unless
  `PXG_PREFER_SOFTWARE_DEVICE=0`. The engine bridge prefers hardware unless
  `PXG_PREFER_SOFTWARE_DEVICE=1`.
- On Windows, `PXG_TEST_BACKEND=dx12` would select the D3D12 backend. This has
  never been tried.
- Shaders are built with `-O0 -fvk-use-entrypoint-name` (`PXG_SLANG_FLAGS`).
  `-O0` keeps slangc from invoking spirv-opt, which needs the optional
  `slang-glslang` library. Drivers still optimise.
- `build-gpuslang` also produces:
  - `pxg-caps`, a device-profile probe;
  - `libPhysXGpuVulkan_64.so`, a C-ABI export library used by the loader test.
    It contains only the broadphase/AABB subsystem seams, not the full pipeline.
- `tools/lvprepro/` is a standalone Vulkan harness for deciding whether a fault
  is in a shader or in llvmpipe.

### The SDK build (`PX_GPU_BACKEND=VULKAN`)

The official scripts build the port as the SDK's GPU module, with no CUDA
toolkit, through four presets: `linux-gcc-vulkan`, `linux-clang-vulkan`,
`vc17win64-vulkan` and `vc18win64-vulkan`.

```bash
cd physx
./generate_projects.sh linux-gcc-vulkan          # Windows: generate_projects.bat vc18win64-vulkan
cmake --build compiler/linux-gcc-vulkan-release -j
# -> bin/linux.x86_64/release/libPhysXGpu_64.so + bin/linux.x86_64/release/PhysXGpuShaders/*.spv
```

- `compiler/public/CMakeLists.txt` takes the switch `PX_GPU_BACKEND`
  (`CUDA` by default, or `VULKAN`). With `VULKAN` and the GPU projects on, it
  neither enables the CUDA language nor looks for the toolkit. It adds
  `source/compiler/cmakegpuvulkan` instead of `source/compiler/cmakegpu`, which
  stays untouched. The snippet render library then skips its CUDA–GL interop.
- `cmakegpuvulkan` builds **one** shared `PhysXGpu` with the name, output
  directory, flags and CRT of the CUDA module. It compiles:
  - the gpu* host modules;
  - the bridge and the Vulkan device (`cmake/PxgVulkanGpuSources.cmake`, which
    `sdktest` uses as well);
  - the PhysX dependency sources the CUDA module also compiles into itself (read
    from `cmakegpu/PhysXGpuDependencies.cmake`).
- On Linux the module links with `-Wl,--no-undefined`, so a missing symbol fails
  at link time as it would under MSVC.
- It compiles the 178 shader files the kernel manifest maps
  (`cmake/PxgEngineShaders.cmake`) into `PhysXGpuShaders/` next to the library.
  It uses the same `pxg_compile_shader` calls as the suite: `cmake/PxgShaderList.cmake`
  is the one list, and the configure step fails if the manifest names a file or
  entry the list does not export.
- The bridge loads each `.spv` from `$PXG_SHADER_DIR` only, when that is set and
  not empty (no fallback). Otherwise it tries `PhysXGpuShaders/` next to the
  module, then the working directory.
- `PxCreateCudaContextManager` and `PxSetPhysXGpuFoundationInstance` hand the
  application's foundation to the module, as the CUDA module does. Without that, a
  separately loaded module dereferences a null foundation: on Windows always, and
  on Linux with the static SDK libraries.
- The Linux loader (`PxPhysXGpuModuleLoader.cpp`) still loads `libcuda.so.1` first
  when it exists, but no longer refuses to load the GPU module without it.
- `PxGetPhysicsGpu()` is NULL (see below), so the skinning and isosurface snippets
  are not generated with this backend.
- `PX_GENERATE_GPU_STATIC_LIBRARIES` and `PX_USE_NVTX` stop the configure step
  with a message.
- slangc comes from `-DPXG_SLANGC`, else `$SLANG_DIR/bin` and `$SLANG_DIR`, then
  `PATH`, then `$VULKAN_SDK/Bin|bin`. A version other than
  2026.14.1 configures with a warning.
- Windows: MSVC has never compiled this. `tools/check_msvc_portability.py` (ctest
  `gpuslang.msvcPortability`) scans the module's sources for the failure classes
  gcc and clang let through:
  - RTTI under `/GR-`;
  - GNU attributes, builtins and keywords;
  - unguarded POSIX code;
  - a raw `<windows.h>`;
  - a non-literal `__declspec(align())`;
  - Win32 APIs hidden by `PxWindowsInclude.h`'s `NO*` macros (`NONLS`,
    `NOUSER`/`NOMB`, `NOGDI`).

  The UTF-8/UTF-16 path conversion the Windows module uses is platform-neutral
  and tested on Linux (ctest `gpuslang.p4ModulePath`).

  It is a guard, not a compiler; a real Visual Studio build is still owed.

## Known limitations

- **Linux + llvmpipe only.** Everything above was developed and measured on Mesa
  llvmpipe (Mesa 25.2.8 / LLVM 20.1.2 at the last measurement). No run on real
  GPU hardware or on any other Vulkan driver has happened. The device-limit and
  subgroup checks exist so that a smaller device refuses instead of crashing,
  but no real device has exercised them.
- **D3D12 has never been built or executed.** The backend compiles only on
  Windows. Its edits since P9.98 (the limit and range checks) are
  syntax-checked only. The HLSL ctests cover the cross-compile of a subset of
  entries, not DXIL or execution. Several kernels have no `cs_6_0` lowering
  (`WorkgroupCount()`).
- **Bit equality with NVIDIA hardware is not provable for every path.** The
  claim is equality with an IEEE reading of the CUDA source, measured on
  llvmpipe. Known unknowns:
  - `tex3D` filtering: the α rounding and the blend arithmetic are undocumented
    by NVIDIA (P9.96 K1/K2). The [0.5, N−0.5] clamp follows a hardware
    measurement, not the guide text (K4).
  - `OpDot` and other driver-lowered operations: their association order is
    driver-defined. llvmpipe matches the CUDA when stamped; other drivers are
    unmeasured.
  - Atomic accumulation order: a few cloth/softbody scenes vary in the 5th
    decimal from run to run on llvmpipe, within scene tolerances.
- **The Windows module has never been built.** The `vc17win64-vulkan` /
  `vc18win64-vulkan` presets exist and the MSVC-specific failure classes are
  checked from Linux, but no Visual Studio build has run. The scene harness and
  the suite exercise the bridge linked statically (`sdktest`); the shared module
  the SDK build produces is what an unmodified application loads. The PhysicsGpu factory
  (SDF builder, isosurface, sparse grid, array converter, skinning;
  `PxGetPhysicsGpu()`) is not compiled, so those app-facing utilities are
  unavailable (see Coverage).
- **Performance has not been a goal.** The design is correctness-first:
  - groupshared warp emulation instead of subgroup intrinsics;
  - `-O0` SPIR-V;
  - `tex3D` emulation that costs 9 descriptor accesses per sample.
- **Upstream bug candidates.** Porting turned up a number of suspected defects in
  the CUDA/host code. Examples: a permuted launch pack for
  `trimeshHeightfieldNarrowphase`, `ps_stepParticlesLaunch` stepping a removed
  particle system, and a race in `computeThresholdElementMaskIndices`. They are
  documented in RUNTIME_STATUS.md (numbered up to 60 by P9.83b, with more noted
  per tranche since). The port reproduces the upstream behaviour. **None has
  been reported upstream.**
