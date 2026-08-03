# Phase 2 runtime status

Progress tracker for the Vulkan production runtime (RUNTIME_PLAN.md P0–P7).
Phase 1 (bit-exact CUDA→Slang math port, 673 kernels) is closed; see
MIGRATION_STATUS.md.  Every item below is committed on
`claude/physx-cuda-slang-migration-mqbz7u` and verified against the standalone
gpuslang harness on llvmpipe (Vulkan 1.4) in BOTH memory modes — default
(host-visible) and `PXG_DEVICE_V2=1` (device-local + staged + timeline).

Test count: **676/676** green in both modes (`ctest -N` in `build-gpuslang`).
*(This line read 696/696 before P8.20 and had drifted: `run_suite.sh` has been
reporting 673/674/675 for several tranches while the header kept an older
figure.  It is now the number `ctest` prints.)*

## Done

| Step | Commit | What shipped |
|---|---|---|
| P0 profile | 3dfd7e4 | `pxg-caps` probe + subgroup/BDA spikes. **Risk R1 confirmed**: llvmpipe subgroup window fixed at [8,8] → warp-32 needs native (hardware) + emulation (CI) paths. BDA launch model proven. |
| P1.1+P1.2 | d292c36 | Timeline submission (4 frames in flight, one timeline semaphore, submit/wait/poll, flush=wait(submit)); device-local buffers + persistently mapped 8 MiB staging ring with per-buffer barriers; `PXG_DEVICE_V2` mode. |
| P1.2 heap | 39ba000 | `PxgSlangHeapAllocator`: buddy sub-allocator with `PxgHeapMemoryAllocator` semantics (128 B align, deferred frees, grow-and-copy resize). |
| P1.1 fix | 8c7d726 | Adversarial-review follow-up: consecutive submits wait on the previous ticket → single-CUDA-stream ordering (Vulkan does not order same-queue submissions itself); destroy flushes open recordings. |
| P2.1 warp lib | eb2bfe5 | `pxgSlangWarpInc.slang`: CUDA-exact shuffles/ballots/butterfly reductions/scans (width-parameterized, partial-mask aware), emulation (any subgroup size) + native@32 paths. Bit-level battery, 4 mutants. |
| P2.2 atomics | b2aab7f | CAS-loop float min/max/add (atomic.cuh parity), split 64-bit OR, `__popcll`/`__ffsll`/`__clzll`. Order-invariant contention battery, 7 mutants. |
| tex3D | 28e2805 | 3D sampled images (R32F, trilinear clamp-to-edge), `createPipelineTex`/`dispatchTex`. Voxel centers bitwise + filtered points vs double trilinear reference (tolerance path, documented replacement). |
| P3 shim core | bbc00d3 | `PxgCudaShim`: flat-address allocs (interior-pointer resolve), HtoD/DtoH/DtoD/memset, kernel launch via manifest + BDA nested-pointer marshalling. |
| P3 stream/event | b39004d | streamCreate/Destroy/Synchronize, eventCreate/Destroy/Record/Synchronize/Query, streamWaitEvent — streams as ordering handles, events as timeline tickets. |
| P1.6 copy/fill | 7b75b66 | Device-side `vkCmdCopyBuffer` (DtoD) + `vkCmdFillBuffer` (D32); fixed a latent host-visible stream-ordering gap (upload/download now drain prior GPU work, per cuMemcpy*'s stream ordering). |
| P2.1 native val. | 1a2bfe2 | `npWarpNative.slang` (`PXG_WARP_NATIVE=1`) compile/lowering/driver validation of the native WaveShuffle/WaveActiveBallot path; hardware exec of 32-lane semantics stays P5. |
| P1.5 pipeline cache | 7b84aaf | Per-device `VkPipelineCache`, seeded from / written back to an on-disk blob; corrupt blob tolerated. |
| P6 indirect | f394c31 | `dispatchIndirect` (`vkCmdDispatchIndirect`) with a device-computed grid + compute→indirect barrier; all buffers gain INDIRECT usage. |
| review follow-up | 9ca8957 | Device→host domain barrier on host-visible `download`; cache short-write cleanup; fillBuffer/dispatchIndirect alignment asserts (adversarial review, no BLOCKER/MAJOR). |
| P1.6 timestamps | 1b68913 | Per-device timestamp `VkQueryPool`; `writeTimestamp`/`getTimestampNanos` (wrap-safe, order-independent) for per-stage GPU timing. |
| P3.1 PxCudaContext | dad70a6 | `bridge/PxgVulkanCudaContext` implements the REAL `physx::PxCudaContext` against actual PhysX headers (no CUDA toolkit); a Slang kernel runs end-to-end through the abstract interface. |
| P3/P4 manifest | 8b55cbc | `tools/gen_kernel_manifest.py` -> generated `PxgSlangKernelManifest.h`; 501 registry kernels, 123 (24.6%) mapped to a ported Slang entry by exact name. |
| P3.4 PxCudaContextManager | 1e13214 | `PxgVulkanCudaContextManager` implements the REAL `physx::PxCudaContextManager` (the object `PxSceneDesc::cudaContextManager` receives); hands out the context, reports device props, drives the deprecated `*Internal` buffer helpers. |
| P3.5 C-ABI export | 1cbebd9 | `bridge/PhysXGpuVulkan.cpp`: the `libPhysXGpuVulkan_64.so` drop-in — exports the loader's C symbols (`PxCreateCudaContextManager` wired to the bridge, the CUDA function/module table getters, `PxGpuCreatePhysicsGpu`); `p4Loader` dlopen test. `CudaKernelWrangler.h` de-coupled from `<cuda.h>`. |
| P4.1 foundation | 8f94208 | `physx-foundation` static lib (15 Fd*/unix files, PIC) built in the harness — the engine-build keystone: real PhysX engine `.cpp` files now compile+link against the bridge with no CUDA toolkit. `p4Foundation` creates a `PxFoundation` + `PX_NEW`/`PxArray`. |
| P4.2 memory seam | c6a65eb | The REAL gpucommon `PxgHeapMemoryAllocatorManager` buddy allocator (unmodified engine code) runs over `PxgVulkanMemoryManager : PxsMemoryManager` on the Vulkan context; flat-address device allocs resolved at interior pointers. `p4Heap` roundtrips 6 device buffers. |
| P4.3 wrangler seam | a34e9d7 | The REAL cudamanager `KernelWrangler` + gpucommon `PxgCudaKernelWranglerManager` (the exact concrete types the gpu-subsystem ctors demand) run over the Vulkan context. `PxgVulkanKernelTable.cpp` builds the live CUDA function table from the 123 mapped manifest rows; the wrangler ctor name-resolves every kernel through our `moduleGetFunction` -> `toFunc(id)=id+1`. `p4Wrangler` proves the engine wrangler resolves all 123 mapped registry kernels to our launch encoding — which also pins manifest↔gpucommon enum alignment and `entry==cudaName`. `PXG_NO_CUDA_TOOLKIT` guards the toolkit-only includes + CUDART shims in `CudaKernelWrangler.cpp`. |
| P4.4 broadphase seam | 7096bba | The FIRST concrete gpu subsystem over Vulkan: the REAL `PxgCudaBroadPhaseSap` (the type `createGpuBroadPhase` static_casts to) constructs on the Vulkan context via both prior seams, taking the context lock, allocating its ~40 device buffers + host-mapped descriptors through the buddy allocator, and creating its stream+event through the bridge — including the first direct-driver seam, `cuCtxGetStreamPriorityRange`, shimmed by `PxgVulkanDriverShim.cpp`. Needed the no-CUDA-toolkit header stand-ins (`nocuda/vector_types.h` etc., ABI-exact vs cuda.h) + a small `PxgCudaToolkitShim.h`, and `PXG_NO_CUDA_TOOLKIT` guards on 6 gpu headers. `p4Broadphase` constructs it, proves device bytes are charged to the eBROADPHASE heap group, and round-trips real data through the shared pool. |
| P4.5 AABB manager seam | (this commit) | The REAL `PxgAABBManager` (the type `createGpuAABBManager` static_casts to) constructs over the Vulkan context, wrapping the real `PxgCudaBroadPhaseSap` and driven by both prior seams. Reuses the P4.4 no-toolkit mechanism verbatim — no new shim, no new driver seam. Adds the CPU-side `Bp::AABBManagerBase`/`BpFiltering` base + the geometry bounds path (`Gu::computeBounds` and its convex-shape helpers) the `PxgBoundsArray` vtable pulls. `p4Aabb` constructs the manager and proves the bp↔aabb coupling: the broadphase's `getAABBManager()` now points back at it (mutation on the engine's `setGPUAABBManager` call is caught). |
| P5 SDK milestone A | 30e9508 | **The full unmodified PhysX CPU SDK builds from source and a real `PxScene` steps.** ~340 CPU `.cpp` (foundation, common, geomutils, lowlevel*, lowleveldynamics, scenequery, simulationcontroller, Np core) build as one static lib in the harness — no packman, no CUDA toolkit — needing only `PX_PHYSX_STATIC_LIB` + `PX_PUBLIC_RELEASE=1`, **zero engine edits**. Test `sdk-step` drops a dynamic box onto a static ground and steps 4 s @ 60 Hz: it falls (y 5.000→4.997 on step 1), reaches the ground, settles at centre y=0.500, and comes to rest (\|v\|=0.000). |
| P5 driver-seam closure | a71b28b | The remaining direct-CUDA-driver seams shimmed (CUDA texture/array API for SDF/heightfield — inert, off the rigid path; raw `cuMemcpyDtoH/HtoD/DtoD` routed through the live bridge context; `cuStreamQuery`→ready; the `CUDA_ERROR_*` codes) so **all ~50 GPU host `.cpp` also compile no-toolkit** — the whole engine (CPU + GPU) now assembles against the Vulkan bridge. gpuslang suite still 696/696. |
| P5 SDK milestone B | _this_ | **A real GPU `PxScene` is constructed and stepped over Vulkan.** The 6 bridge factories in `PhysXGpuVulkan.cpp` now placement-new the real concrete subsystems (`PxgCudaBroadPhaseSap` / `PxgAABBManager` / `PxgBoundsArray` / `PxgNphaseImplementationContext` / `PxgSimulationController` / `Pxg{,TGS}DynamicsContext`), verbatim as `physxgpu`. All 48 gpu\* host `.cpp` (broadphase/common/narrowphase/solver/simulationcontroller/articulation) + the bridge + the gpuslang backend link statically against the CPU SDK (`physx-gpuhost` lib), resolving every concrete-ctor symbol. Test `gpu-sdk-step` builds a `PxScene` with the Vulkan `cudaContextManager` + `eENABLE_GPU_DYNAMICS` + `eGPU` broadphase + TGS: **the whole `Pxg*` pipeline constructs live over Vulkan** (llvmpipe), two actors add, and `simulate()`/`fetchResults()` drive the real GPU broadphase to the kernel-launch boundary. Remaining frontier: per-kernel launch marshalling (the CUDA param-pack → Slang binding + manifest `pushConstantSize`) — the first launch (`MemCopyBalanced`) is where stepping currently stops. |

## Empirical device profile (CI = llvmpipe, LLVM 20.1.2)

subgroup 8/8/8 fixed (required-32 impossible → emulation on CI); BDA yes;
timelineSemaphore yes; VK_EXT_shader_atomic_float (buffer+shared) yes;
shaderFloat64/Int64 yes; vulkanMemoryModel yes; 32 KiB shared.

## Open

- **P2 native path** — hardware execution at subgroup 32 (P5; CI's llvmpipe
  is fixed at 8).  Compile/lowering/driver validation is done (1a2bfe2).
- **P1 remainder** — specialization constants (per-module block-size
  variants), multi-queue SubmissionContexts (P1 serializes on one queue in
  submission order).  Pipeline cache, copy/fill utilities, indirect dispatch,
  and timestamp queries are done.
- **P3 DONE** — the real `PxCudaContext`/`PxCudaContextManager` virtual wrappers
  + kernel-wrangler forwarding onto the shim (P3.1/2/4, hardened per adversarial
  review), the generated 501-entry kernel manifest (P4 manifest), and the
  `PhysXGpuVulkan` C-ABI export library (P3.5: exports the loader's C symbols,
  `PxCreateCudaContextManager` wired to the bridge, `p4Loader` dlopen test).
  `CudaKernelWrangler.h` is de-coupled from `<cuda.h>`.  All against the real
  PhysX headers with no CUDA toolkit.
- **P4** — full-engine integration.  The two load-bearing coupling seams the
  `PxgPhysXGpu.cpp` `static_cast` sites demand are now CLOSED against the real,
  unmodified engine code over the Vulkan context: the **memory manager**
  (P4.2 — real `PxgHeapMemoryAllocatorManager` buddy allocator) and the
  **kernel wrangler** (P4.3 — real `KernelWrangler` + `PxgCudaKernelWranglerManager`,
  resolving all 123 mapped kernels to our launch encoding).  Remaining: the
  concrete `Pxg*` subsystems (broadphase / AABB / narrowphase / simulation
  controller / dynamics) rebuilt on the Vulkan context and driven through these
  two seams.  The **first concrete subsystem is now up (P4.4)**: the real
  `PxgCudaBroadPhaseSap` constructs over Vulkan through both seams, which also
  built the reusable no-CUDA-toolkit header layer (`nocuda/` ABI-exact vector-type
  stand-ins + `PxgCudaToolkitShim.h`) and closed the first direct-driver seam
  (`cuCtxGetStreamPriorityRange`).  The **AABB manager is up too (P4.5)** — the
  real `PxgAABBManager` constructs over Vulkan wrapping the broadphase, reusing the
  P4.4 mechanism with no new shim.
  **Coupling boundary found (P4.6):** broadphase + AABB construct in isolation
  because their ctors take only the two GPU seams; the remaining trio (narrowphase,
  dynamics, sim controller) is mutually recursive and rooted in the CPU sim core
  (`PxsContext`/`SimpleIslandManager`/`IslandSim`), so they cannot be meaningfully
  constructed alone — the narrowphase ctor even reads the dynamics context's solver
  stream.  The honest path to constructing and STEPPING them is a full-PhysX-SDK
  link (CPU core + GPU modules + a `PxScene` on the Vulkan bridge), during which the
  last direct-driver seam (the CUDA texture-object/array API for SDF/heightfield,
  to be backed by the existing gpuslang tex3D Vulkan images — not stubbed) also
  closes.  See RUNTIME_PLAN.md P4.6 for the full scoping.  Also remaining: the
  remaining ~direct CUDA-driver call-sites (the `cuTexObject` trio, `cuStreamQuery`,
  one raw `cuLaunchKernel`, scattered `cuStreamSynchronize`/`cuMemcpyDtoH`); each
  subsystem's `update()`/kernel-launch path exercised (construction is proven,
  stepping is not); and the manifest's `pushConstantSize` + the 378 unmapped kernel
  rows filled during per-module bring-up.  This moves verification from harness
  batteries to real ScScene scenes — the qualitative shift from isolated tranches
  to full-engine integration.
- **P5** — cross-vendor validation (NVIDIA/AMD RDNA+/Intel Xe+), per-device
  tex3D filter tolerance calibration.
- **P6/P7** — async compute, indirect dispatch, DX12 backend, interop.

## Verification doctrine (unchanged from Phase 1)

Every device-layer or library change: focused test in both memory modes +
mutation battery proving the test has teeth + full-suite regression in both
modes before commit. Adversarial review for non-trivial synchronization code
(the P1.1 ordering gap and the P1.6 host-visible ordering gap were both found
this way / by the wiring, not by the happy path).

## P7.0 — MILESTONE: the FIRST live FEMCloth frame — a triangle-mesh cloth falls under gravity on Vulkan

The deformable-surface (cloth) subsystem opens exactly as the particle arc did with
P6.0: a triangle-mesh cloth FALLING under gravity through the full `Pxg*` engine on
llvmpipe. Three FEMCloth orchestration kernels; `PXG_SLANG_KERNEL_MAPPED` 180 → **183**,
all in new file `femClothSimBda.slang`, ported line-for-line from
`gpusimulationcontroller/src/CUDA/FEMCloth.cu`:
- `cloth_preIntegrateLaunch` (`FEMCloth.cu:48`, ABI `{8,8,12,4,1,8,1,1,8,8,8}`=80B) — the
  free-fall driver: gravity predict AND the PGS position integration in one shot
  (`positions[i] = oldPos + vel*dt`, the observable fall). The cloth vertex arrays ALIAS
  the user `PxDeformableSurface` buffers directly (`PxgSimulationCore.cpp:1850`), so this
  write is what a host DtoH read of `getPositionInvMassBufferD()` sees — NO ingest/egress
  kernel. Launch `PxgFEMClothCore.cpp:177`.
- `cloth_finalizeVelocitiesLaunch` (`:1227`, ABI `{8,8,4,4,1}`=32B) — velocity settling +
  the block-wide wake flag (`mIsActive` device `atomicOr`). Its two-level warp `__any_sync`
  lowers to a groupshared `InterlockedOr` (block-wide OR — semantically identical).
- `cloth_sleeping` (`:1313`, ABI `{8,4,8,4,4,8,8}`=48B) — wake-counter bookkeeping. REQUIRED
  or the cloth deactivates (wakeCounters→0 in ~24 frames) and the fall freezes. Its
  `__ballot_sync` lowers to a groupshared `InterlockedOr` ballot, and the CUDA
  early-return+`__syncthreads` is restructured to an `if (gid < n)` guard so ALL threads
  reach both barriers (llvmpipe requires uniform barrier control flow).

New descriptor mirror `pxgSlangFEMClothDesc.slang`: byte-exact `PxgFEMClothGpu` of the engine
`class PxgFEMCloth` (`PxgFEMCloth.h:69`), flat pointer+scalar block, sizeof **400**; the packed
`PxU8 mActorFlags|PxU8 mBodyFlags|PxU16 mSurfaceFlags` @376 folds to one `mFlagsWord` (low byte =
mActorFlags; gravity-disable test). New scene `PXG_SCENE=cloth` in `testGpuSdkStep.cpp`: a
vert-mapped triangle grid cooked with `buildGPUData`, a `PxDeformableSurface` spawned high with
one live rigid body below (the zero-dynamic-rigid null-deref trap), DtoH mean-vertex-y readback.

**KEY EVIDENCE — the FEMCloth free-fall (`PXG_SCENE=cloth PXG_SOLVER=pgs`):**
```
step  0: mean y = 4.99727 (dy = -0.00273)    step 30: mean y = 3.64840 (dy = -0.08448)
step 10: mean y = 4.82015 (dy = -0.02997)    step 50: mean y = 1.38665 (dy = -0.13897)
step 20: mean y = 4.37052 (dy = -0.05723)    step 59: mean y = 0.01325 (dy = -0.16350)
cloth free fall: mean y 5.0000 -> 0.0133 (dropped 4.9867 over 1.00s)
[  OK  ] gpu-sdk-step: FEMCloth fell monotonically under gravity on Vulkan (the first live FEMCloth frame)
```
Monotonic, with dy accelerating each step (−0.0027 → −0.1635) — exactly semi-implicit
free-fall. All 3 cloth kernels DISPATCH + launch OK in the trace. Deviations (flagged, all
free-fall-preserving): `preIntegrate`'s `adaptiveCollisionPairUpdate` warpReduction (its only
subgroup op) is OMITTED — the scene sets `nbCollisionPairUpdatesPerTimestep != 0` so that
branch is dead; the `*updateContactPairs = !isTGS` PxU8 store is elided (buffer is exactly
`sizeof(PxU8)`, no mapped kernel consumes it on the free-fall path). New unit test
`gpuslang.femClothSim` (5 cases: descLayout, PreIntegratePGS, PreIntegrateTGS, Finalize,
Sleeping) bit-exact vs a CPU transcription of the .cu, both memory modes; the P6.1 particle
BDA re-points verified intact (manifest hand-edited, NOT regenerated). Full suite **673/673**
both memory modes; rigid regression 8×2 = **16/16** — all as re-verified end-to-end by the
build agent BEFORE the environment note below.

**P7.0 RE-VERIFIED on a fresh container — the `libvulkan_lvp.so` SIGSEGV was NOT an
environmental degradation; it was a MISSING `PXG_SHADER_DIR` (root-caused + fixed).** The
earlier "environment degraded mid-verification" note (the SIGSEGV inside `libvulkan_lvp.so`,
null deref `at 50`, on the first core kernel `updateShapesLaunch` of every `gpu-sdk-step`
scene, both memory modes) was a MISDIAGNOSIS. The real, fully-reproduced mechanism:
1. `gpu-sdk-step` loads each kernel's SPIR-V blob at first launch via `envSpirvLoader`
   (`bridge/PhysXGpuVulkan.cpp`), resolving the manifest `spirvFile` against `$PXG_SHADER_DIR`.
2. With `PXG_SHADER_DIR` UNSET, `bodyUpdate.spv` (the first engine kernel, `updateShapesLaunch`)
   is not found → **empty blob** → empty `VkShaderModule`.
3. `vkCreateComputePipelines` returns **VK_ERROR_UNKNOWN (-13)**; `PXG_VK_CHECK` is `fprintf` +
   `assert(false)`, and `assert` is a no-op under `NDEBUG` (Release), so the failure is
   **non-fatal** → the shim binds the **NULL `VkPipeline`**.
4. lavapipe null-derefs that pipeline in `handle_pipeline` (`lvp_execute.c:1034`) at command-
   buffer replay on the `vk_queue` submit thread → the `at 50` SIGSEGV.
Ruled out as causes (each tested): Mesa point-release (`25.2.8-0ubuntu0.24.04.1` and `.2` crash
identically), the `MESA_VK_ENABLE_SUBMIT_THREAD` submit thread, `LP_NUM_THREADS` raster
threads, the Mesa shader disk cache, and the app pipeline cache. The unit test
`gpuslang.bodyUpdate` builds the SAME `updateShapesLaunch` pipeline and PASSES (it resolves
its own shader dir) — proving the SPIR-V is fine and the fault is purely the missing runtime
shader path. **Fix: run `gpu-sdk-step` with `PXG_SHADER_DIR=<build-gpuslang>/shaders`** (the
harness shader-output dir); `envSpirvLoader` now also logs a loud one-time warning naming the
missing file + the `PXG_SHADER_DIR` hint, so the silent-empty-blob → NULL-pipeline trap can
never masquerade as a driver crash again.

**Re-verification (fresh container, Mesa `25.2.8` / llvmpipe LLVM `20.1.2` — the exact CI
profile — + slangc `2026.12.2`, both provisioned from scratch):** with `PXG_SHADER_DIR` set,
`PXG_SCENE=cloth` reproduces the money shot **BIT-EXACT** to the trajectory above (mean y
`5.0000 → 4.99727 → 4.82015 → 4.37052 → 3.64840 → 1.38665 → 0.01325`, dy monotone
`−0.00273 → −0.16350`, prints the `[  OK  ]` line); `box` falls `5.000 → 0.500` and rests;
`particle` free-falls `100.75 → 80.9665`. Full gpuslang suite **673/673 in BOTH memory modes**;
rigid regression 8×2 = **16/16**. P7.0's code was always good; only the invocation was missing
the shader path.

## P7.1 tranche 1 — the cloth IN-PLANE MEMBRANE energy solve on the BDA engine ABI

The first step of the P7.0→draping arc (the analog of the particle P6.2/P6.3 tranches): the
per-triangle in-plane (membrane / ARAP + area) energy solve — the internal force that makes the
cloth resist stretching — is brought onto the BDA engine ABI. Two kernels;
`PXG_SLANG_KERNEL_MAPPED` 183 → **185**, ported line-for-line from `FEMCloth.cu`:
- `cloth_solveNonSharedTriangleEnergyLaunch` (`FEMCloth.cu:669`, ABI `{8,8,8,4,4,1}`=40B, launch
  `PxgFEMClothCore.cpp:1556`) — one large partition per launch: over the graph-colored non-shared
  triangle partition `[accumCP[p-1], accumCP[p])`, solve each triangle's in-plane energy and fold
  the position change back into velocity + accumulated delta (`updateNonSharedTriangle`, `:610`).
- `cloth_solveNonSharedTriangleEnergyClusterLaunch` (`:719`) — the small-partition twin: loops the
  remaining partitions in one launch with a groupshared barrier between them (barrier in UNIFORM
  control flow — `nbPartitions` is uniform — so llvmpipe-safe).

New shader `femClothSolveBda.slang`; the in-plane math lives in new module
`pxgSlangClothMembrane.slang` — a VERBATIM lift (code byte-for-byte identical, verified by diff) of
the Phase-1 `npClothSharedEnergy` membrane leaf (`membraneSolve` = `membraneEnergySolvePerTriangle`
FEMClothUtil.cuh:436-527, composing extractRotation2D / arapConstraint2D / areaConstraint2D + inline
Lamé params), so the engine BDA solve and the Phase-1 verification shaders share ONE bit-exact copy;
NoContraction-stamped (`PXG_NOCONTRACT_SHADERS`) like its Phase-1 twin. New descriptor: byte-exact
`PxsDeformableSurfaceMaterialGpu` mirror of `PxsDeformableSurfaceMaterialData` (sizeof 32); the
PxgFEMCloth fields the solve reads (`mMaterialIndices`@56,
`mOrderedNonSharedTriangleVertexIndices_triIndex`@80, `mOrderedNonSharedTriangleLambdas`@96,
`...RestPoseInv`@104 = QInv, `mNonSharedTriAccumulatedPartitionsCP`@192,
`mNbNonSharedTriPartitions`@344) pinned with offsetof static_asserts in the test TU.

**Correctness is proven by the unit test, not the scene** (the P6.2/6.3 precedent): the P7.0
5×5-grid cloth has **0 non-shared-triangle partitions** — its triangles route through the
*shared/pair* energy path (`cloth_solveTrianglePairEnergyLaunch`, still unmapped — the next
tranche), so `solveNonSharedTriangles` launches nothing and the free-fall is **byte-unchanged**
(mean y 5.0000 → 0.01325, monotone, exit 0). New unit test `gpuslang.femClothSim` +2 cases
(`SolveNonSharedPGS`/`TGS`): one triangle STRETCHED to 1.5× rest edge in a single partition,
dispatched, then `positions`/`velocities`/`accumulatedDelta`/`lambdas` compared **bit-exact** vs a
CPU transcription of `updateNonSharedTriangle` driving the Phase-1-verified `refMembrane` —
mutation-checked (a 1-line invDt perturbation in the shader makes it fail on the velocity
writeback, ref 4.07 vs got 6.05). Full suite **673/673 both memory modes** (the femClothSim entry
now runs 7 cases); rigid regression 8×2 = **16/16** (the manifest gains only two previously-`nullptr`
cloth rows — hand-edited, NOT regenerated — so every rigid/particle dispatch is byte-unaffected).

Deviations (flagged, all behavior-preserving): the CUDA `USE_BLOCK_COPY` groupshared descriptor copy
→ direct `Ptr<>` reads (P7.0 precedent); `mMaterialIndices[triIdx]` PxU16 load → 32-bit word read +
half extraction (llvmpipe has no 16-bit storage); the per-triangle lambda array I/O is hoisted OUT of
`membraneSolve` into `updateNonSharedTriangle` (the verified npClothSharedEnergy float2-slot pattern —
same read-at-start / write-at-end, PGS only). **Recovery note:** this tranche's shader + scaffolding
were drafted by a background worktree agent that a container suspend interrupted before it built or
tested; the code survived on disk and was salvaged — the math verified byte-identical to the Phase-1
module by diff, the wiring audited against the `.cu`, and the functional bit-exact unit test authored
to close it.

## P7.1 tranche 2 — the cloth triangle-PAIR (shared-energy + bending) solve on the BDA engine ABI

The energy solve the P7.0 5×5-grid cloth ACTUALLY routes through (its triangles have 0 non-shared
partitions — all edges are shared triangle *pairs*). Three kernels; `PXG_SLANG_KERNEL_MAPPED`
185 → **188**, ported line-for-line from `FEMCloth.cu`:
- `cloth_solveTrianglePairEnergyLaunch` (`:861`, ABI `{8,8,8,4,4,1,1}` — adds an `isSharedTrianglePair`
  bool vs the non-shared solve) + its cluster twin `...ClusterLaunch` — the 4-vertex (x0,x1,x2,x3)
  pair solve (`updateTrianglePairs`, `:768`): for a SHARED pair, `clothSharedEnergySolvePerTrianglePair`
  (membrane on both triangles + bending); for a NON-shared pair, `bendingEnergySolvePerTrianglePair`
  (bending only). Uses the COPIES/REMAP scheme — each vertex belongs to several pairs, so each pair
  writes its correction to its own copy slot (`mPosition_InvMassCP` via the remap-output CP).
- `cloth_averageTrianglePairVertsLaunch` (`:522`, ABI `{8,8,4,1}`) — the Jacobi pass that averages
  each vertex's per-pair copies back by its reference count.

New shader `femClothSolvePairBda.slang`; the bending math is new module `pxgSlangClothBending.slang`
(`bendingSolve` = `bendingEnergySolvePerTrianglePair`, FEMClothUtil.cuh:543-655 — the Discrete-Shells
XPBD dihedral constraint, structurally identical to the Phase-1 `npClothBending`). The shared-energy
driver reuses `pxgSlangClothMembrane` (tranche 1) for the two membrane calls.

**THE atan2 SEAM (documented tolerance, the tex3D precedent).** The bending dihedral uses
`atan2(sinAngle, cosAngle)`; `atan2f` is a libm transcendental with NO cross-backend bit-exact
guarantee (same class as sinf/cosf/tex3D filtering). The Phase-1 `npClothBending` harness dodges it by
taking the angle as a host-precomputed INPUT; the ENGINE kernel has no host, so it evaluates `atan2`
**on-device**. Therefore: the membrane half, the copies/remap/average machinery, and everything up to
and INCLUDING `sinAngle`/`cosAngle` are **bit-exact**; the on-device `atan2` and everything downstream
(C, the 2π wrap, clamp, deltaLambda, position update) is a **tight-tolerance** path — `relTol 1e-5` +
abs floor `1e-6` (velocity `6e-5 = 1e-6·invDt`), justified by the *measured* ~1e-7-relative divergence
between llvmpipe's `GLSL.std.450 Atan2` and glibc, which propagates to ~5e-7 in the displacement. A
symmetric-coplanar (FLAT) pair hits `atan2(0,+1) == +0` (an IEEE-exact special value), so its bending
is **bit-exact** too.

**Scene — the 5×5 mesh now runs its energy solve LIVE:** `PXG_SCENE=cloth` dispatches
`cloth_solveTrianglePairEnergyClusterLaunch` ×960 + `cloth_averageTrianglePairVertsLaunch` ×960 over
the run, **zero refused launches** (was `Could not find cloth_solveTrianglePairEnergyLaunch`), exit 0;
the free-fall is **byte-unchanged** (mean y 5.0000 → 0.01325 — a true no-op on the coherent flat cloth,
no spurious deformation, no crash). New unit tests `gpuslang.femClothSim` +9 cases (SharedMembrane /
SharedBend / NonSharedFlat / NonSharedBent / Cluster × PGS/TGS): a triangle pair (bent out of plane
for the bend cases, coplanar for the flat) with the copies/remap arrays set for a single pair,
dispatched through solve + average, compared vs a CPU transcription — **bit-exact** for the membrane +
flat-bending + machinery, **within-tolerance** for the on-device-atan2 bent bending — mutation-checked
(a +0.01 rad angle perturbation fails 7 cases; the tight tolerance has teeth). Full suite **673/673
both memory modes**; rigid regression 8×2 = **16/16** (three previously-`nullptr` cloth rows hand-edited,
NOT regenerated — rigid/particle dispatch byte-unaffected).

Also fixed a latent build-dep gap surfaced by the mutation check: the cloth Slang modules
(`pxgSlangFEMClothDesc` / `...ClothMembrane` / `...ClothBending`) were absent from
`SHADER_COMMON_DEPS`, so editing an imported module did not re-trigger dependent-shader recompiles
(stale `.spv` on incremental builds — clean builds were always correct). All three are now tracked.

Deviations (flagged, behavior-preserving): descriptor block-copy → direct `Ptr<>` reads; PxU16 loads →
32-bit word + half extraction; the copies-`w`-flag reset and the remap writeback transcribed verbatim;
material lookup by `u32(restData.y/z)`; the atan2 tolerance seam above (the ONLY non-bitwise part).
**Recovery/verification note:** the tranche was drafted by a background worktree agent (this one ran to
completion), then salvaged and INDEPENDENTLY re-verified in the main tree — math diffed against the
Phase-1 modules, the atan2 tolerance re-derived and mutation-checked for teeth, and the full
suite/regression/scene re-run before commit.

## P7.1 tranche 3a — the cloth RIGID-ATTACHMENT constraint kernels on the BDA engine ABI

The machinery to PIN a cloth vertex to a rigid/world anchor (the first half of a hanging/draping
cloth; the scene that uses it is tranche 3b). Three kernels; `PXG_SLANG_KERNEL_MAPPED` 188 → **191**,
ported line-for-line from `FEMClothConstraintPrep.cu` + `FEMClothExternalSolve.cu`:
- `cloth_rigidAttachmentPrepareLaunch` (launch `PxgFEMClothCore.cpp:226`, ABI
  `{8,8,8,8,4,8,8,8,8,8}`=80B) — builds a `PxgFEMRigidAttachmentConstraint` from each
  `PxgFEMRigidAttachment` (the anchor world pose, the cloth element + barycentric, the effective
  mass / raXn rows), zeroing the rigid-side `rigidDeltaVel` slots.
- `cloth_solveRigidClothAttachmentLaunch` (PGS) + `...TGSLaunch` (launch `:1858`, ABI
  `{8,8,4,8,8,8,8,4,8}`=72B) — applies the constraint: pull the cloth vertex toward the anchor
  (scatter the position correction into `mDeltaPos`), accumulate the rigid-side reaction.

New shader `femClothRigidAttachBda.slang`. New byte-exact mirrors (offset-probed vs the real headers,
static_assert-pinned in the test TU): `PxNodeIndex` (8B, static/articulation bit-ops on the u64),
`PxgFEMRigidAttachment` (96B, pinned vs `PxgSimulationCoreDesc.h`), `PxgFEMRigidAttachmentConstraint`
(the RAC_* 32-slab SoA block, replicated host-side because `PxgFEMCore.h` pulls `<cuda.h>` and can't
be included in the toolkit-free test TU — cited to `PxgFEMCore.h:191`). Reuses the P5 prep/solver-desc
mirrors (`PxgPrePrepDescGpu` / `PxgConstraintPrepareDescGpu` / `PxgSolverCoreDescGpu` /
`PxgSolverSharedDesc{PGS,TGS}Gpu` / `PxgSolverBodyData` / `PxgSolverTxIData` / `PxgAlignedTransform`).

**Fully BIT-EXACT — no transcendental/tolerance seam** (the attachment solve is plain linear algebra;
`getNormalized`'s `PxRecipSqrt` is `1/sqrtf`, IEEE-correctly-rounded). New unit tests
`gpuslang.femClothSim` +3 cases (`RigidAttachDescLayout`/`PGS`/`TGS`): one cloth vertex + one static
world anchor, `prepare → solve`, comparing the prepared constraint block AND the solved `mDeltaPos`
bit-exact vs a CPU transcription (private `p73ref` namespace, verbatim `PxQuat::rotate`/`getNormalized`/
`PxMat33` leaves). The pull is toward the anchor (error = anchor − clothPos); independently
mutation-checked in the main tree (a `dp.x += 0.0002` scatter perturbation + forced-fresh `.spv`
recompile fails both attach cases). Full suite **673/673 both memory modes**; rigid regression
8×2 = **16/16** (three previously-`nullptr` cloth rows hand-edited, NOT regenerated).

Deviations (flagged, behavior-preserving, all documented in the shader header): the **articulation
anchor branch** is deferred (a rigid/world anchor is never an articulation — the `psContactSolveTgsBda`
precedent of sinking `outArtiVelocity` behind the `isArticulation()` guard); the **dynamic-rigid
reaction** writeback is deferred (CUDA guards it behind `!isStaticBody()`, never taken for a static
anchor — the prepare still zeroes `rigidDeltaVel` as the CUDA does, verified in the test); the
**cone-limit** path is deferred (a `PxSinCos` seam; `isConeLimitedEnabled()` is false for a plain pin,
so the plain velError/tgsError IS the exact CUDA false-path); `atomicAdd(mDeltaPos)` → plain
read-modify-write (bit-identical for the single-attachment test; multi-attachment averaging is the
scene tranche's concern). This tranche was drafted by a background worktree agent, then salvaged and
INDEPENDENTLY re-verified in the main tree (build, both-mode suite, my own mutation check with a
forced-fresh recompile, regression) before commit.

## P7.1 tranche 3bc — MILESTONE: a FEMCloth HANGS from a pinned edge and drapes under gravity (the cloth money shot)

The culmination of the P7.1 cloth arc: a triangle-mesh `PxDeformableSurface` pinned along its top
edge now **HANGS and drapes to rest** on the Slang/Vulkan (llvmpipe) engine path — the cloth analog
of the P6.5 settling-bag particle money shot. Two parts:

**3c — the missing link `cloth_applyExternalDeltasLaunch`** (`FEMClothExternalSolve.cu:1053`, ABI
`{8,8,4}`, launch `PxgFEMClothCore.cpp:2113`, block dim SB_SOLVETETRA=64). `PXG_SLANG_KERNEL_MAPPED`
191 → **192** (manifest hand-edited). This per-vertex kernel folds the accumulated external
correction — `mDeltaPos` (`.w` = the running per-vertex count) written by the attachment/contact
solves — into `mPosition_InvMass` / `mVelocity_InvMass` / `mAccumulatedDeltaPos` (`scale = 1/.w`,
`vel += delta/scale·invDt`, `pos += delta`), then clears it. **It is the ONLY consumer of
`mDeltaPos`; without it every cloth attachment (and cloth-cloth / contact) correction was
dead-ended — surfaced by the tranche-3b scene agent's honest report** (the pin dispatched but moved
nothing). Added to `femClothRigidAttachBda.slang` (NoContraction-stamped). New unit test
`gpuslang.femClothSim` `ApplyExternalDeltas`: three vertices (counts 2 / 0 / 3), bit-exact vs the
`.cu` (the count-0 vertex must stay untouched, the others fold + clear) — mutation-checked
(`pos += delta·1.0001` fails the two applied vertices under a forced-fresh recompile).

**3b — the `PXG_SCENE=clothhang` scene.** A vertical triangle-mesh curtain (top row at y=5, hanging
to y=3) is pinned along its **top row (5 distinct vertices → one rigid attachment each**, so the
exact single-attachment 3a path, no averaging kernel) to a static world anchor via the REAL
deformable-surface attachment API (`PxPhysics::createDeformableAttachment`, type eVERTEX↔eWORLD →
`Sc::Scene::addRigidAttachment(body=NULL)` → `PxgSimulationController::addRigidAttachment`). It steps
under gravity with the live P7.1.1/.2 energy solve + the P7.1.3a attachment prepare/solve + this 3c
apply.

**KEY EVIDENCE (`PXG_SCENE=clothhang PXG_SOLVER=pgs`, 120 steps):**
```
top row    y 5.0000 -> 5.0000  (max |dev| = 0.00000)   -- pinned edge holds exactly
bottom row y 3.0000 -> 2.9869  (dropped 0.0131)        -- drapes + stabilizes
bottom rest |dy| over the last 30 steps = 0.000000     -- converged to rest
cloth_applyExternalDeltasLaunch OK x1120 (+ attachment prepare/solve OK every iteration)
[  OK  ] gpu-sdk-step: FEMCloth HANGS from its pinned edge and drapes under gravity
```
The pinned top row stays at y=5 to the ULP; the free cloth hangs as a coherent draped sheet (the
energy solve holds its shape against gravity) and comes to REST — NOT free-fall (the P7.0 free-fall
dropped 4.99 m; this drops 0.013 m of stretch and stops). Full suite **673/673 both memory modes**
(femClothSim now 20 cases); rigid regression 8×2 = **16/16**; the `cloth` free-fall + `box` +
`particle` scenes are byte-unchanged.

Deferred (the next frontier, all currently refuse cleanly + are no-ops for a non-self-intersecting
hanging curtain): cloth **self-collision** (`cloth_selfCollisionMidphaseLaunch` + its sort/reorder
chain) and cloth-vs-primitive **contact** — needed for a cloth that drapes OVER an object or folds
onto itself. The TGS attachment path also freezes the curtain (TGS cloth position-integration not
fully mapped), so the hang is a **PGS** result, exactly as the particle settling bag was first a PGS
result. This closes the cloth attachment loop end-to-end; 3c was ported + verified in the main tree,
3b salvaged from the (honest-reporting) scene agent, and the money shot re-run green here.

## P7.1 tranche 4a — cloth-vertex-vs-PRIMITIVE contact GENERATION on the BDA engine ABI

The first tranche of the cloth-drape-on-a-floor arc (the analog of the particle P6.4a contact
generator). Three kernels; `PXG_SLANG_KERNEL_MAPPED` 192 → **195**, ported line-for-line from
`gpunarrowphase/src/CUDA/femClothMidPhase.cu` + `femClothPrimitives.cu`:
- `cloth_midphaseGenerateVertexPairsLaunch` (`CLOTH_MIDPHASE_VERTEX_PRIMS`, launch
  `PxgNarrowphaseCore.cpp:6317`, ABI `{4,4,8×8}`=72B, grid (32, numTests, 1)) — broadphase→midphase
  cloth-vertex ↔ rigid-AABB candidate-pair emit.
- `cloth_planeVertContactGenLaunch` (`CLOTH_PLANE_VERTEX_CG`, `:5853`) — vertex-vs-PLANE contact
  generation (the floor drape).
- `cloth_SphereVertexContactGenLaunch` (`CLOTH_VERT_SPHERE_CG`, `:5776`) — vertex-vs-SPHERE (the
  drape-over-a-sphere). Both share ABI `{4,8×8,4,80}`=160B (a `PxgFEMContactWriter` marshalled by
  value @80, the psContactGen precedent).

New shader `femClothContactGenBda.slang`. Reuses the REAL narrowphase mirrors (`PxgContactManagerInput`
/ `PxgShape` / `PxsCachedTransform` / `PxgRigidFilterPair` / `PxBounds3`, offset-pinned); the
`PxgFEMContactWriter` info tail (`PxgFemOtherContactInfo`, sizeof 16) mirrored locally + self-asserted
(`PxgFEMCore.h` pulls the CUDA buffer stack, not test-includable). **Fully BIT-EXACT — no
transcendental/tolerance seam** (plane/sphere/midphase use only mul/add/sqrt/1÷x, IEEE-correctly-
rounded). New unit tests `gpuslang.femClothSim` +5 cases (ContactGenDescLayout / MidphaseVertexPairs
incl. a no-penetration→0-pairs case / PlaneVertContactGen{,NoContact} / SphereVertContactGen): one
cloth vertex penetrating a plane/sphere → midphase + contact-gen → the contact (normal, penetration,
indices) compared bit-exact vs a CPU transcription of the `.cu`; independently mutation-checked in the
main tree (a +0.01 perturbation of the stored plane normal + forced-fresh `.spv` recompile fails the
plane case: ref n.x −5.96e-08 vs got 0.0099). Full suite **673/673 both memory modes**; rigid
regression 8×2 = **16/16** (three previously-`nullptr` cloth rows hand-edited, NOT regenerated — no
scene uses these yet, so every existing dispatch is byte-unaffected).

Deviations (flagged, behavior-preserving, documented in the shader header): the CUDA
`__ballot_sync`/`__shfl_sync`/lane-0-atomicAdd contact compaction → per-32-lane-tile groupshared
`InterlockedOr`-ballot + `countbits` prefix + one `InterlockedAdd` (identical contact SET + totals;
the atomicAdd-order buffer index is meaningless — radix-sorted downstream); warp-cooperative
`LoadShapePair` → per-thread `Ptr<>` loads; the midphase `gridDim.x` hardcoded to `MIDPHASE_GRID_X=32`
(compute has no gridDim built-in; matches `numBlocks=32` — psContactGen `PS_GRID` precedent, dispatch
grid must be (32,·,·)); the sphere kernel's eCAPSULE branch deferred (a capsule is never launched into
the sphere path); `setRigidMaterialIndex(mat)+markInCollision(false)` folded to one store of
`mat & 0x7fffffff` (the two RMWs provably net to that word).

**Scouting map for the rest of the cloth-drape arc** (recommended next tranches, mirroring particle
P6.4b): **4b-1 SORT** — `fem_reorderRigidContactsLaunch` (`FEM_REORDER_RS_CONTACTS`, launch
`PxgFEMCore.cpp:231`) + wire the already-mapped radix path (`RS_MULTIBLOCK`/`CALCULATERANKS`/
`COPY_HIGH_32BITS`/`COPY_VALUE`). **4b-2 PREPARE** — `cloth_rigidContactPrepareLaunch`
(`CLOTH_RIGID_CONTACTPREPARE`, `PxgFEMClothCore.cpp:326`, builds `PxgFemRigidConstraintBlock`).
**4b-3 SOLVE** — `cloth_queryRigidClothContactReferenceCount{,TGS}Launch` +
`cloth_solveRigidClothCollision{,TGS}Launch` (`:1733`/`:1778`; scatters into `mDeltaPos`, folded by
the already-mapped `cloth_applyExternalDeltasLaunch`; the rigid-side reaction `accumulateRigidDeltas`
is deferrable for a static floor, the P7.1.3a static-anchor precedent). **4c** — the
`PXG_SCENE=clothdrape` floor scene. This tranche was drafted by a background worktree agent then
salvaged + INDEPENDENTLY re-verified in the main tree (build, both-mode suite, my own forced-fresh
mutation check, regression) before commit.

## P7.1 tranche 4b-1 — cloth rigid-contact SORT / reorder on the BDA engine ABI

The second tranche of the cloth-drape arc (the analog of particle P6.4b-1): reorder the P7.1.4a-generated
cloth-vs-rigid contacts into the by-rigid `…Sorted…` buffers the prepare/solve read. One kernel;
`PXG_SLANG_KERNEL_MAPPED` 195 → **196**:
- `fem_reorderRigidContactsLaunch` (`FEM_REORDER_RS_CONTACTS`, launch `PxgFEMCore.cpp:231`, ABI
  `{8}×11`=88B, block/grid SB_REORDERCONTACTS 256/1024) — a pure gather: `remapIndex =
  remapByRigid[index]`, then copy `contacts / normalpens / barycentrics / contactInfos[remapIndex]`
  into the four sorted arrays and split the sorted info's `pairInd0` (PxU64 rigid id) into
  `sortedRigidIds[index]`. New shader `femClothContactSortBda.slang` (mirrors the particle
  `psContactSortBda` precedent — a distinct pipeline stage, no imports).

The radix sort that produces `remapByRigid` is **already mapped** — `PxgFEMClothCore::sortContacts()`
drives the plain `RS_MULTIBLOCK`/`RS_CALCULATERANKS_MULTIBLOCK` (P6.4b-1) + `RS_COPY_HIGH_32BITS`/
`RS_COPY_VALUE`, and `updateGPURadixSortBlockDesc` is a host function (no kernel). No sort-setup kernel
is missing. **Fully BIT-EXACT** (a pure index-remapped load/store — zero float arithmetic, so NOT in
`PXG_NOCONTRACT_SHADERS`; the float4 lanes move as opaque bit patterns). New unit tests
`gpuslang.femClothSim` +2 cases (`ReorderRigidContacts` NC=5000; `…MultiIter` NC=262921 > grid·block,
forcing the second grid-stride iteration): a shuffled `[0,NC)` rank permutation → reorder → every
sorted point/normalPen/barycentric/info is the exact permutation + `sortedRigidIds[i]==info.pairInd0`,
bit-exact; independently mutation-checked in the main tree (`barycentrics[remapIndex]→[index]` +
forced-fresh `.spv` recompile fails both cases). Full suite **673/673 both memory modes**; rigid
regression 8×2 = **16/16** (one previously-`nullptr` row hand-edited; no scene uses it yet).

Deviations (behavior-preserving): NO NoContraction (pure gather); the CUDA grid-stride early-`return`
kept verbatim (no groupshared/barrier); operator precedence `(n+B·G-1)/B*G` preserved verbatim
(exercised by the multi-iter case); PxU64 rigid id carried as `uint2` halves. Drafted by a background
worktree agent, salvaged + INDEPENDENTLY re-verified in the main tree (build, both-mode suite, my own
forced-fresh mutation check, regression) before commit. Remaining: 4b-2 PREPARE
(`cloth_rigidContactPrepareLaunch`) → 4b-3 SOLVE (`cloth_queryRigidClothContactReferenceCount{,TGS}` +
`cloth_solveRigidClothCollision{,TGS}`) → 4c the `clothdrape` floor scene.

## P7.1 tranche 4b-2/3 — cloth rigid-contact PREPARE + SOLVE on the BDA engine ABI

The third tranche of the cloth-drape arc (the analog of particle P6.4b-2): turn the generated + sorted
cloth-vs-rigid contacts into a corrective cloth-vertex push. Five kernels; `PXG_SLANG_KERNEL_MAPPED`
196 → **201**, ported line-for-line from `FEMClothConstraintPrep.cu` + `FEMClothExternalSolve.cu`:
- `cloth_rigidContactPrepareLaunch` (`CLOTH_RIGID_CONTACTPREPARE`, launch `PxgFEMClothCore.cpp:328`,
  ABI `{8×10,4,8,1}`=104B) — builds a `PxgFemRigidConstraintBlock` (effective mass / normal / raXn
  response rows / bias) from each sorted contact.
- `cloth_queryRigidClothContactReferenceCount{,TGS}Launch` (`CLOTH_QUERY_RIGID_CLOTH_REFERENCE_COUNT[_TGS]`,
  `:1741`, ABI `{8×8,4,8,8}`=88B) — accumulates each cloth vertex's contact reference count into
  `mDeltaPos.w` (so the already-mapped `cloth_applyExternalDeltasLaunch` averages correctly).
- `cloth_solveRigidClothCollision{,TGS}Launch` (`CLOTH_SOLVE_RIGID_CLOTH_COLLISION[_TGS]`, `:1797`,
  ABI `{8×11,4,8,8}`=112B) — solve the constraint, scatter the cloth-side correction
  `deltaPos = −(δλN·normal + δλT·tangent)·dt` into `mDeltaPos`.

New shader `femClothRigidContactBda.slang` (NoContraction-stamped — the dot/normalize bias math is
bitwise vs the `-ffp-contract=off` CPU ref, the psContactSolveTgsBda precedent). New mirror
`PxgFemRigidConstraintBlock` (PxgFEMCore.h:167, the 32-slab SoA; host-replicated + self-asserted since
the header pulls the CUDA stack, the P7.1.3a precedent). **Fully BIT-EXACT** (linear algebra + sqrt, no
transcendental seam) in PGS AND TGS, BOTH memory modes. New unit tests `gpuslang.femClothSim` +2 cases
(`RigidContactPGS`/`TGS`): one cloth vertex penetrating a static floor → a sorted contact → prepare →
queryRefCount → solve → the prepared constraint block (7 fields incl. non-zero response rows) AND the
solved `mDeltaPos` push the vertex OUT, bit-exact vs a CPU transcription; independently mutation-checked
in the main tree (dropping the `deltaPos.y` negation + forced-fresh `.spv` recompile fails both cases:
ref y=+0.0038 vs got −0.0038). Full suite **673/673 both memory modes**; rigid regression 8×2 = **16/16**.

**This closes the cloth-vs-rigid contact SOLVE chain end-to-end** (generate → sort → prepare → solve →
`applyExternalDeltas`); only the scene (4c) remains to make a cloth actually settle on a floor.
Deferrals (each behavior-preserving for a static floor, documented in-shader): the articulation branch
(a static floor is never an articulation); the dynamic-rigid reaction (`computeRigidChange` behind
`!isStaticBody()`, never taken — the prepare still writes the count-0 the CUDA writes); friction
(`getCombinedFriction()==0` for the NULL-material query path → `δλT ≡ −0`, cloth push bit-exact with a
frictionless material; material-strided friction is a follow-on); `atomicAdd(mDeltaPos)` → plain
read-modify-write (single-contact test). Drafted by a background worktree agent, salvaged +
INDEPENDENTLY re-verified in the main tree (build, both-mode suite, my own forced-fresh mutation check,
regression) before commit. Remaining: **4c** — the `PXG_SCENE=clothdrape` floor scene (cloth falls,
the contact pipeline catches it, it settles).

## P7.1 tranche 4c — MILESTONE: a FEMCloth falls onto a floor and SETTLES under contact (the cloth-contact money shot)

The payoff of the cloth-vs-rigid contact arc: a triangle-mesh `PxDeformableSurface` dropped above a
STATIC floor now **falls, is caught by the contact pipeline, and settles resting ON the floor** — the
second cloth money shot (after the P7.1.3 hanging cloth), the direct analog of the P6.5 particle
settling bag. A PURE SCENE tranche (no new kernels — the whole gen→sort→prepare→solve→apply chain is
mapped by P7.1.4a/4b). New `PXG_SCENE=clothdrape` in `testGpuSdkStep.cpp`: a horizontal 5×5 tri-mesh
cloth spawned flat at y=1.5 with **invMass=1 everywhere, NO pins/attachments, no zero-mass bypass** —
only the floor CONTACT can catch it. The floor is a static infinite `PxPlane` at y=0 (the direct
cloth-vs-plane path `cloth_planeVertContactGenLaunch`, no midphase); one live dynamic box far aside
keeps the rigid PGS island non-empty.

**KEY EVIDENCE (`PXG_SCENE=clothdrape PXG_SOLVER=pgs`, 180 steps):**
```
mean y 1.5000 -> 1.320 (s10) -> 0.148 (s30, free-fall at g) -> 0.00000 (s40, caught) ->
        0.00000 ... 0.00000 (rest); final min y = -0.0000 (no vertex tunnels below 0);
        rest |dy| over the last 45 steps = 0.000000
[  OK  ] gpu-sdk-step: FEMCloth fell onto the floor and SETTLED under contact
```
It free-falls at the gravity rate until ~y=0.15, then the plane contact decelerates it to rest exactly
at the surface (min y never < 0 → no tunnelling). The contact kernels DISPATCH OK in the trace:
`cloth_planeVertContactGenLaunch` / `fem_reorderRigidContactsLaunch` / `cloth_rigidContactPrepareLaunch`
×59, `cloth_queryRigidClothContactReferenceCountLaunch` / `cloth_solveRigidClothCollisionLaunch` ×236,
`cloth_applyExternalDeltasLaunch` ×1180, **zero refusals among them** (contact-gen stops at step 59
because the settled cloth then sleeps — physically correct). Full unit suite **673/673 both memory
modes** (unchanged — SDK-scene-only edit); rigid regression 8×2 = **16/16**; `cloth`/`clothhang`/
`particle` scenes unchanged.

**Solver scope: this is a PGS result** (as the P7.0/3b/4b cloth milestones were). Under TGS all three
cloth scenes (`cloth`, `clothhang`, `clothdrape`) leave the sheet frozen — cloth gravity integration on
this Vulkan/llvmpipe path is PGS-only (a TGS cloth-position-integration gap, pre-existing, NOT caused by
contacts) — so `clothdrape` honestly prints `[FAIL]` under TGS exactly as `cloth`/`clothhang` do; it was
NOT special-cased to fake a TGS pass.

**Honest gap flagged (a real follow-on):** three single-copy radix-sort helpers used by
`PxgFEMClothCore::sortContacts` — `clampMaxValues`, `radixSortCopy`, `radixSortCopyHigh32Bits` — REFUSE
with a push-constant-size ABI mismatch (`paramBlobSize != manifest.pushConstantSize`). Harmless for
THIS scene (every cloth-vs-plane contact references the SAME single static rigid, so the sort-by-rigid
key is trivially uniform and the reorder/prepare/solve see the correct order regardless), and the cloth
settles exactly — but it is a genuine gap the cloth contact sort would hit with MULTIPLE distinct rigid
contact targets. It is outside this tranche's scope; the fix (align those single-copy radix kernels'
push-constant ABI, the double-copy variants used by the particle path are already fine) is a scoped
follow-on. This tranche was drafted by a background worktree agent (honest-reporting) then salvaged +
INDEPENDENTLY re-verified in the main tree (rebuild, the clothdrape money shot, rigid regression 16/16,
all cloth/box/particle scenes) before commit.

## P7.2 tranche 1 — cloth SELF-collision midphase + contact GENERATION (VT + EE) on the BDA engine ABI

Opens the cloth fold-on-itself arc: detect the cloth's own vertex-vs-triangle (VT) and edge-vs-edge
(EE) self-collisions. Two kernels; `PXG_SLANG_KERNEL_MAPPED` 201 → **203**, ported line-for-line from
`gpunarrowphase/src/CUDA/femClothClothMidPhase.cu`:
- `cloth_selfCollisionMidphaseVTLaunch` (`CLOTH_SELFCOLLISION_MIDPHASE_VT`) +
  `cloth_selfCollisionMidphaseEELaunch` (`CLOTH_SELFCOLLISION_MIDPHASE_EE`) — launch
  `PxgFEMClothCore.cpp:698`, shared ABI `{4,8,8,8,8,4,8,8,8}`=72B, grid `(numQueries, nbActiveCloths, 1)`
  block `(32,1,1)`. Both walk the cloth's own BV32 triangle BVH and append a `PxgFemFemContactInfo` per
  penetrating VT / crossing EE pair DIRECTLY (no separate self-contact-gen kernel).

New shader `femClothSelfCollisionBda.slang` — the BV32 warp-cooperative tree walk
(`bv32TreeTraversal` + `BV32DataPacked` decode) + the closest-point tests (`closestPtPointTriangle`,
`closestPtLineLine`) rebuilt as the BDA sibling of the proven Phase-1 `npBv32`/`npClothCloth`/`npClosest`/
`npEdgeEdge` transcriptions (`Ptr<>` reads + push constants instead of descriptor-set buffers).
NoContraction-stamped. No new device mirror — the existing `PxgFEMClothGpu` carries every field
(`mTriMeshData`/`mTrianglesWithActiveEdges`/`mTriangleVertexIndices`/`mRestPosition`/
`mSelfCollisionFilterDistance`…); the `PxgFemFemContactInfo` (16B) + BV32-node byte offsets (min@0/max@512/
data@1024, stride 1168) are mirrored + self-asserted in the test TU. **Fully BIT-EXACT — no transcendental
seam** (VT/EE closest-point is pure mul/add/sqrt/1÷x/dot/cross; the midphase stores only the integer
contact record, and the float math merely gates contact presence, with the test geometry far from every
threshold). New unit tests `gpuslang.femClothSim` +2 cases (`SelfCollisionVT`: a vertex 0.05 above a
triangle → 1 VT contact, a lifted vertex → 0; `SelfCollisionEE`: two crossing edge pairs → 2 EE contacts
with the exact `auxInd = 0x40000000|e0|(e1<<14)` encoding, lifted apart → 0), bit-exact vs a CPU
transcription; independently mutation-checked in the main tree (`writeContactInfo` pairInd0+1 +
forced-fresh `.spv` recompile fails the VT case: got 4 ref 3). Full suite **673/673 both memory modes**;
rigid regression 8×2 = **16/16** (two previously-`nullptr` rows hand-edited; no scene uses them yet).

Deviations (behavior-preserving, documented in-shader): one 32-thread group == one warp == one query
(the CUDA 2-warps/block would put the BV32 `GroupMemoryBarrierWithGroupSync` in non-uniform cross-warp
control flow — the proven npBv32 lowering; dispatch grid `(numQueries, nbActiveCloths, 1)`);
`__ballot_sync`/`__shfl_sync`/`warpScanExclusive` → groupshared `InterlockedOr`/broadcast/`countbits`
prefix; per-warp scratch pointer cache → direct `Ptr<>` reads; `auxInd` zero-init (pins the dead
InCollision bit + VT's unused aux payload the CUDA leaves indeterminate — every bit the solve reads is
still written by the same mark*/setAux* sequence); the different-cloth `distSqInRestPose` NULL guard is
dead on the self path (rest buffer always non-null) and omitted; `clothClothPairs` carried for ABI parity
but unread (the self path applies no filter-pair lookup, only the rest-distance test); self-only branch
folding (`clothId0==clothId1` makes several `!=`-guards compile-time constant); EE over-write-by-≤8
bug-compat preserved (`index<maxContacts` gate but `[index+li]` write, matching the CUDA).

**Scouting for the rest of the self-collision arc** (there is NO sort step — self-contacts are consumed
in place, unlike the rigid 4b-1 radix reorder): P7.2 tranche 2 = PREPARE `cloth_clothContactPrepareLaunch`
(`CLOTH_CLOTH_CONTACTPREPARE`, `PxgFEMClothCore.cpp:499`) → per-substep QUERY-COUNT
`cloth_queryClothClothContact{VT,EE}CountLaunch` (`:1981/:1982`) + SOLVE `cloth_solveClothClothDelta{VT,EE}Launch`
(`:2003/:2004`, scatter into `mDeltaPos`, folded by the already-live `cloth_applyExternalDeltasLaunch`) +
`cloth_updateClothContactValidityLaunch` (`:647`). P7.2 tranche 3 = a `clothselfcollide` fold-on-itself
scene. Drafted by a background worktree agent, salvaged + INDEPENDENTLY re-verified in the main tree
(build, both-mode suite, my own forced-fresh mutation check, regression) before commit.

## P7.2 tranche 2 — cloth SELF-collision SOLVE (prepare + count + solve VT/EE + validity) on the BDA engine ABI

Closes the cloth-fold-on-itself pipeline the tranche-1 midphase opened: turn the detected VT/EE
self-contacts into position corrections. Six kernels; `PXG_SLANG_KERNEL_MAPPED` 203 → **209**, ported
line-for-line from `gpusimulationcontroller/src/CUDA/FEMClothConstraintPrep.cu` + `FEMClothExternalSolve.cu`:
- `cloth_clothContactPrepareLaunch` (`CLOTH_CLOTH_CONTACTPREPARE`, `PxgFEMClothCore.cpp:505`) — recomputes
  the exact rest-pose distance filter and stamps each surviving `PxgFemFemContactInfo`; ABI `{8,8,8,4,8,8}`=48B
  (the `maxContacts` PxU32 @24 tail-pads to realign the following pointer).
- `cloth_queryClothClothContact{VT,EE}CountLaunch` (`CLOTH_QUERY_CLOTH_CONTACT_{VT,EE}_COUNT`,
  `:1983`) — per-substep reference-count of each vertex's self-contacts into `mDeltaPos.w`; shared ABI
  `{8,8,8}`=24B.
- `cloth_solveClothClothDelta{VT,EE}Launch` (`CLOTH_SOLVE_CLOTH_{VT,EE}_COLLISION`, `:2005`) — scatters
  BOTH endpoints' corrections into `mDeltaPos` (folded into position by the already-live
  `cloth_applyExternalDeltasLaunch`); shared ABI `{8,8,8,4}`=32B (`dt` PxReal @24).
- `cloth_updateClothContactValidityLaunch` (`CLOTH_UPDATE_CLOTH_CONTACT_VALIDITY`, `:652`) — per-vertex
  contact-pair validity refresh + `prevPositionInContactOffset` snapshot; ABI `{8,8,1,1,8,8,8,4}`=56B (two
  `bool` flags @16/@17).

New shader `femClothClothContactBda.slang` — the closest-point constraint math (`closestPtPointTriangle`,
`closestPtLineLine`, barycentric weight scatter) + the `mDeltaPos` RMW accumulation rebuilt as the BDA
sibling of the proven Phase-1 cloth-cloth solve, reading every descriptor / `PxgFemFemContactInfo` record
through `Ptr<>` + push constants. NoContraction-stamped; deliberately absent from `PXG_HLSL_ENTRIES` (pure
BDA, no `cs_6_0` lowering — same precedent as `femClothRigidContactBda` / `femClothSelfCollisionBda`); its
five imported modules already ride in `SHADER_COMMON_DEPS`. No new device mirror — the existing
`PxgFEMCloth`/`PxgFemFemContactInfo` carry every field. **Fully BIT-EXACT — both memory modes**: the
friction branch's `PxRecipSqrt` transcendental seam is DEAD behind `frictionCoefficient != 0`, and the
solve runs the deferred null-material (frictionless) target the launch site uses, so every executed op is
mul/add/sqrt/1÷x/dot/cross. New unit tests `gpuslang.femClothSim` +2 cases (`runClothSelfContactVT`:
prepare → query → VT-solve on a penetrating vertex-triangle self-pair → both endpoints pushed apart, the
query count landing in `mDeltaPos.w`, validity snapshot; `runClothSelfContactEE`: the same chain on a
crossing edge-edge self-pair), each bit-exact vs a `p72ref` CPU transcription of the closest-point +
constraint math. Independently mutation-checked in the main tree (`deltaLambdaN * normal` → `* 1.01f` +
forced-fresh `.spv` recompile fails every `mDeltaPos[0..3]` VT + EE `MM` compare). Full suite **673/673
both memory modes**; rigid regression 8×2 = **16/16**; the `cloth` + `clothdrape` scenes still settle.

Deviations (behavior-preserving, documented in-shader): the CUDA `atomicAdd` into `mDeltaPos` → a
groupshared/`Ptr<>` read-modify-write (single-thread-per-contact scatter, no cross-lane contention on the
test geometry — the proven external-solve lowering); the friction-material fetch + `PxRecipSqrt` tangent
solve carried for ABI parity but frictionless-dead as above; the `bool` flags marshalled as the CUDA's
1-byte `PxU8` payload; ABI-parity args (`updateContactPairs`, the adaptive/force flags) present in the
push blob but unread on the frictionless self path. Drafted by a background worktree agent, salvaged +
INDEPENDENTLY re-verified in the main tree (build, both-mode 673/673 suite, my own forced-fresh mutation
check with teeth, regression, `cloth`/`clothdrape` scenes) before commit.

**Scouting for the rest of the self-collision arc**: P7.2 tranche 3 = a `clothselfcollide` fold-on-itself
scene (drop a cloth so it lands on itself and the self-contacts stop it from interpenetrating — the
self-collision money shot), wiring the tranche-1 midphase + this solve into the per-substep loop. Then the
flagged multi-rigid sort-helper ABI (single-copy radix BDA rewrite) + cloth TGS integration close out P7,
after which **SOFTBODY** (0/56 kernels — the next big subsystem) opens.

## P8.0 tranche 1 — OPENS THE SOFTBODY (FEM-volume) SUBSYSTEM: the first live softbody frame on the BDA engine ABI

The softbody analog of the P7.0 FEMCloth free-fall: the first of the ~56 softbody simulation
kernels, ported line-for-line from `gpusimulationcontroller/src/CUDA/softBodyGM.cu`.
`PXG_SLANG_KERNEL_MAPPED` 209 → **210**:
- `sb_gm_preIntegrateLaunch` (`SB_SIM_PREINTEGRATION`, `softBodyGM.cu:72`, launch
  `PxgSoftBodyCore.cpp:165`) — the softbody free-fall predict/integrate driver: gravity predict
  AND (on the PGS path) the position integration `mSimPosition_InvMass[i] = oldPos + vel*dt`
  (the observable fall), plus the full-kinematic (`mBodyFlags & eKINEMATIC`) and
  partially-kinematic (`mVolumeFlags & ePARTIALLY_KINEMATIC`) branches, the damping predict,
  the max-velocity clamp and both the TGS and PGS position paths — every null/flag branch and
  the thread mapping preserved. Grid `(numBlocks, nbActiveSoftBodies, 1)`, block
  `(32, SB_PREINTEGRATION/32, 1)`, `blockIdx.y` selects the softbody via `activeId[blockIdx.y]`.
  ABI `{8,8,12,4,1,8,1}` = **56 bytes** (two device ptrs, `PxVec3 gravity`(12)@16, `PxReal dt`@28,
  `bool isTGS`@32, `PxReal* speculativeCCDContactOffset`@40 after the 4-byte tail-pad realigns,
  `bool externalForcesEveryTgsIterationEnabled`@48).

New shaders `softBodySimBda.slang` + `pxgSlangSoftBodyDesc.slang` — the first softbody device
mirror: `PxgSoftBodyGpu`, a BYTE-EXACT transcription of the engine `PxgSoftBody` struct (every
field offset AND the array stride, self-asserted with `offsetof` against the real engine header
in the test TU — the byte-packed `mActorFlags`@392 / `mBodyFlags`@393 / `mVolumeFlags`@394 flag
triple + explicit `_pad396` so the following pointer lands at @400). The kernel reads the
descriptor array through `Ptr<>` + push constants (pure BDA, excluded from `PXG_HLSL_ENTRIES`),
NoContraction-stamped. **Fully BIT-EXACT — both memory modes, NO transcendental seam**: the
predict/integrate path is pure mul/add; the max-velocity clamp's `1/PxSqrt(velMagSq)` is DEAD on
the default scene (`mMaxLinearVelocity == PX_MAX_REAL` short-circuits the clamp false). New unit
test `gpuslang.softBodySim` (`runPreIntegrate` for both the PGS and TGS paths): a small softbody
with free + fixed verts, gravity `(0,-9.81,0)`, dispatched and compared f2u-bit-exact to a CPU
transcription of the same gravity/damping predict + PGS position integrate (and the TGS path where
`delta` stays 0 but velocity still receives gravity). Independently mutation-checked in the main
tree (`vel += gravity*dt` → `*1.01f` + forced-fresh `.spv` recompile fails 21 f2u compares:
`vel MM 0 ref(0,-0.163228,0) got(0,-0.16486,0)`). `softBodySim` passes bit-exact in BOTH memory
modes; the rest of the suite passes too (673 prior + this new case), with the lone exception of
`gpuslang.npBv32`, which fails `vkCreateComputePipelines … VK_ERROR_UNKNOWN (-13)` even in isolation
— an ENVIRONMENTAL llvmpipe pipeline-creation degradation (the container's long suspend/resume, the
same class as the documented `gpuslang.scan` flake), NOT a logic regression from this change (every
BV32-using test that DOES build its pipeline, e.g. `femClothSim`, still passes bit-exact).

Deviations (behavior-preserving, documented in-shader): the kernel's ONLY warp op — the
speculative-CCD reduction (`warpReduction<MaxOpFloat>` + `__syncthreads` + `AtomicMax` on
`speculativeCCDContactOffset[id]`) — is OMITTED because it is gated by the OPT-IN
`eENABLE_SPECULATIVE_CCD` body flag (unset on the default free-fall scene), so the whole block +
its `__shared__ maxMagnitudeVelSq[32]` are dead; keeping it out means the shader has NO groupshared
and NO barriers, so there is no 8-vs-32-lane subgroup hazard on llvmpipe (the exact position the
cloth precedent took for its dead `adaptiveCollisionPairUpdate` reduction; `speculativeCCDContactOffset`
stays in the push blob for ABI parity but is unread). The two per-vertex `bool*` byte stores
(`mVertsAreDeformed`/`mVertsCanNotDeform` = false) are elided (8-bit storage on llvmpipe; the cloth
`updateContactPairs` precedent) — no MAPPED kernel consumes them on the free-fall path; the
`mAwake = false` reset IS kept (PxU32, 4-byte, faithful).

**Scouting for the softbody arc** (the minimal "a softbody falls and settles" pipeline, analogous to
how cloth needed only 3 kernels for its first frame): next tranches = `sb_refitBoundLaunch` (BV32
refit) → the FEM-volume solve chain (`sb_gm_cp_solveTetrahedronsPartitionLaunch` +
`sb_updateTetrahedraRotationsLaunch` + `sb_gm_applyExternalDeltasLaunch`) → `sb_gm_stepSoftbodyLaunch`
+ `sb_gm_finalizeVelocitiesLaunch` + `sb_sleeping`, then softbody-vs-rigid contact
(`sb_rigidContactPrepareLaunch` + `sb_solveRigidSoftCollisionLaunch`) for a softbody-on-floor scene.
56 softbody kernels total; this is 1.

## P7.2 tranche 3a — the cloth BVH BOUNDS REFIT: the self-collision midphase now queries a LIVE tree

Tranches 1 + 2 gave the cloth a self-collision midphase and solve, but the midphase walked the
cloth's own BV32 triangle BVH **as cooked** — in rest pose.  `cloth_refitBoundLaunch`
(`CLOTH_REFIT_BOUND`, FEMCloth.cu:338, launch PxgFEMClothCore.cpp:563) is the per-step rebuild that
was missing, and without it a folded cloth generates **zero** self-contacts no matter how deeply it
penetrates itself: every midphase box test is against last-cook geometry.

`shaders/femClothRefitBda.slang` ports it line-for-line as the BDA sibling of
`femClothSelfCollisionBda.slang` (same `Gu::BV32DataPacked` 1168-byte decode, same
`pxgSlangFEMClothDesc` mirror).  It walks the `BV32DataDepthInfo` levels bottom-up, recomputes every
node's 32 per-child AABBs from the CURRENT `mPosition_InvMass`, aggregates each node's own bound into
`mPackedNodeBounds`, and expands the root into the broad-phase `boundArray` slot.  ABI row
`{8,8,4,8,8,8,8,4,8}` (72-byte blob); manifest hand-edited, **MAPPED 210 → 211**.

**Warp lowering.** One 32-thread group per cloth replaces the CUDA (32, `SB_REFIT_WAPRS_PER_BLOCK`=32)
block, so the per-level node loop runs serially in the single warp.  This is forced, not cosmetic:
CUDA `__syncthreads()`-couples 32 warps that each run a data-dependent inner loop, and the groupshared
lowering of `__ballot_sync` / `warpShuffleMin`/`Max` needs a barrier every lane reaches uniformly —
impossible with 32 divergent warps in one group.  Output-identical: same-level nodes have no data
dependency on each other (each reads only children written in a PREVIOUS level, and writes only its
own slots), and min/max is exact + associative + commutative, so both the serialization and the
reduction-order change are bit-preserving, not merely equivalent.

**Two llvmpipe landmines, both bisected rather than guessed:**

1. **`k &= k-1` miscompiles.** CUDA's leaf walk is
   `for (k = resultWarp; k; k = clearLowestSetBit(k))` — the canonical popcount loop idiom.  Under
   llvmpipe's LLVM 20.1 backend this **SIGSEGVs in the driver worker thread** as soon as the loop body
   is non-trivial; bisection pinned it to that loop, with the primitive gather and the groupshared
   reduction EACH reproducing it alone, while an empty-body bit walk runs clean (so: codegen bug, not
   a Slang/SPIR-V semantics issue).  Replaced with a fixed 32-iteration scan over the same ballot
   mask — identical set bits, identical ascending order, identical bits out.
2. **A buffer ending exactly on its last element faults.** llvmpipe widens the final
   `remapPackedNodeIndex` load past the last word, so a test blob that ends on it faults when it lands
   at the end of a mapped region — which is why this only reproduced when the case ran AFTER the rest
   of the suite, never in isolation.  A 16-byte trailer on the test blob fixes it; the refit stays
   bit-exact with the trailer, which it could not be if the kernel were really consuming a 4th remap
   entry.  The engine blob is a cooked mesh-data block with its own trailing sections, so it never
   ends on that boundary.

**Verification.** `femClothSimRefitBound` builds a deliberately TWO-level tree (root with 2 internal
children → two leaf nodes over 6 triangles) so both aggregation paths run — the leaf triangle-vertex
ballot walk AND the `!isLeaf → read mPackedNodeBounds[childOffset]` path, which is only correct if the
deeper level really ran first.  Every node AABB slot, every `mPackedNodeBounds` entry and the
broad-phase slot are compared **bit-exact** (raw float bits) against a CPU reference that transcribes
the 32-lane semantics exactly; neighbouring broad-phase slots are checked untouched (elemIndex
indirection proof) and `updateContactPairs == 0` must skip the refit entirely.  Pure min/max/mul/add,
no transcendental — **fully bit-exact, no tolerance seam**.  Forced-fresh mutation checks (3, each
`rm`-ing the .spv first): perturbing the leaf min, flipping the walk top-down (the reference then
still holds the poison sentinel — proof the bottom-up order is load-bearing), and flipping the
broad-phase expansion sign are all caught with concrete f2u mismatches.  Full suite **674/674 in both
memory modes** (default + `PXG_DEVICE_V2=1`), rigid regression 16/16.

## P7.2 tranche 3b — MILESTONE: a FEMCloth CATCHES ITSELF (`PXG_SCENE=clothselfcollide` goes green)

The self-collision arc closes: a hairpin-folded cloth's free upper layer now **rests on the shelf
made of its own fixed lower layer** instead of drooping through it.  A/B over repeated runs
(`PXG_SELFCOLL` is the only difference): **ON mean y 5.05-5.42 / min 4.18-4.94** vs **OFF mean 4.243 /
min 3.634**, shelf at y=5.00.  The ON path is genuinely non-deterministic (contact-slot atomics +
order-dependent `mDeltaPos` accumulation); the OFF control has no self-contacts and repeats bit-for-bit.

Getting here needed three fixes, each found by instrumenting the live pipeline rather than guessing —
a temporary bridge hook that read the gate, the contact counters, the per-contact validity bits and
`mDeltaPos` straight out of the push blob after each launch:

**1. The gate was never opened (a REGRESSION the P7.0 port planted).**  Every self-collision kernel —
midphase VT/EE, refit, contact prepare — early-outs on `*updateContactPairs == 0`.  In PGS that flag is
set by `cloth_preIntegrateLaunch` (`*updateContactPairs = !isTGS`, FEMCloth.cu:93), and the P7.0 port
had deliberately ELIDED that store: at the time nothing mapped consumed it, and it is a `PxU8` write
(no 8-bit storage on llvmpipe).  Correct then, silently fatal from tranche 1 onward — the whole arc
ran every frame and did nothing.  Restored as a 4-byte store: the engine itself clears that buffer with
`memsetD32Async` (PxgFEMClothCore.cpp:1397) and every consumer reads `word & 0xFF`, so the full word is
ours and the observable byte is identical.  Verified by removal: without the store the scene falls to
exactly the OFF numbers (min 3.634).

**2. `mRestDistance` was 0, so the solve threw away every contact it was handed.**  With the gate open
the midphase generated 76 VT + up to 29 EE contacts per step, all flagged valid — and `maxDelta` stayed
0.  `countContact` (FEMClothExternalSolve.cu:870) keeps a pair only while its separation is inside
`cloth0->mRestDistance + cloth1->mRestDistance`, and `mRestDistance` IS the shape's **rest offset**
(PxgSimulationController.cpp:1770).  The scene set only `contactOffset`, leaving rest offset at its 0
default, so every contact was immediately marked not-in-collision.  `setRestOffset(0.05f)` gives a 0.1
separation band — it exceeds the `dropH=0.05` the layers spawn at, and stays within the contact offset.

**3. Border vertices sat exactly on the shelf's edge.**  Vertex-triangle self-collision can only hold a
vertex that lies over a shelf TRIANGLE.  With both layers on the same footprint the upper layer's border
vertices sat on the shelf's outer edge with nothing beneath them and drooped (measured: interior
5.0-5.5, border 4.66-4.91) — correct cloth behaviour, not a solver gap.  The scene now insets the upper
layer to 0.9, giving the shelf an overhang the way a real sheet on a table is smaller than the table.
(A 0.7 inset is WORSE — it halves the contact count and turns the crease into a ramp the layer slides
down.  Tuning `dropH` up to the rest band, raising the rest offset to 0.07/0.09, damping 0.5, 16
substeps and a 12x7 mesh were all tried and all measured worse; the recorded settings are the best of
that sweep.)

**Scene thresholds are set from a measured 8-run spread, not a guess**: mean > baseY-0.4 and
min > baseY-1.1, each ~0.3 clear of BOTH the ON and OFF ranges, so either one separates the two on its
own; the control asserts the mirror image.  The min threshold is deliberately the looser of the two —
this contact model tests ABSOLUTE separation with no penetration recovery, so a vertex that loses its
contact band in one substep is never pulled back, and the worst-vertex statistic scatters ~0.75 wide.
The "did the sim actually run" guard now measures the largest mean excursion over the WHOLE run rather
than start-vs-end: a well-held layer settles back within ~0.003 of its spawn height, which an endpoint
comparison read as a frozen sim.

Suite 674/674 in both memory modes, rigid regression 16/16, cloth/clothhang/clothdrape/particle green.

## P8.0 tranche 2 — the SOFTBODY BVH bounds refit (`sb_refitBoundLaunch`)

Second kernel of the softbody free-fall chain.  `sb_gm_preIntegrateLaunch` (tranche 1) moves the
vertices; from that moment the softbody's cooked BV32 TETRAHEDRON tree and its broad-phase bound are
stale, so the broad phase, the midphase and contact generation all still see the body where it was
cooked.  `shaders/softBodyRefitBda.slang` ports softBody.cu:99 (launch PxgSoftBodyCore.cpp:284) as the
direct sibling of the cloth refit — same algorithm over tetrahedra instead of triangles — reusing that
port's BV32 decode and warp lowering verbatim.  ABI row `{8,8,4,8,8,8,8,8}` (64-byte blob); manifest
hand-edited, **MAPPED 211 → 212**.

Differences from the cloth sibling, all straight from the .cu: the primitive is a TETRAHEDRON (4
vertices, min/max folded pairwise v0v1 / v2v3 / the two); the tree hangs off `mTetMeshData` with
`mTetIndices` and the COLLISION-mesh `mPosition_InvMass` (not the `mSim*` fields); there is no
`updateContactPairs` gate (a cloth-only mechanism); and the broad-phase tail copies the root bound
VERBATIM with no expansion ("the broad phase will add contact dist automatically"), additionally
republishing the bp/np contact distance as `speculativeCCDContactOffset + mOriginalContactOffset` when
`PxDeformableBodyFlag::eENABLE_SPECULATIVE_CCD` is set.  The llvmpipe `k &= k-1` miscompile and the
buffer-trailer load-width artifact — both bisected on the cloth sibling — are pre-empted here.

**The test geometry is built for mutation teeth, and getting there took three rebuilds.**  The
first attempt (a plain shared-vertex mesh) let a mutant that folds `worldV3` out of the min/max PASS:
every dropped vertex also appeared in another slot of another tet in the same leaf, which re-supplied
the extreme.  Adding one dedicated extremal vertex per slot fixed two of four slots.  The real cause is
structural: the kernel's leaf walk aggregates across ALL of a node's leaf children and then writes that
one aggregate into EVERY child slot, so several extremal vertices under one node mask each other — one
tet per leaf child is not enough, it has to be one tet per NODE.  The final tree is a root with 8
internal children, each a node holding a single tet, and each tet carries in one chosen slot a vertex
uniquely extremal in all three axes (v8/v9 for x, v10/v11 for y, v12/v13 for z, v14/v15 for w, min and
max).  All **8 slot mutants** (4 slots x min/max) are now caught, plus bottom-up→top-down, a
broad-phase bound expansion, a childOffset shift, and ignoring the speculative-CCD gate.

Everything is compared **bit-exact** (raw float bits) against a CPU reference transcribing the 32-lane
semantics: every node AABB slot, every `mPackedNodeBounds` entry, the broad-phase slot (with the
neighbours checked untouched — elemIndex indirection proof) and both contact-distance arrays, with the
speculative-CCD branch covered BOTH ways.  Pure min/max plus one addition, no transcendental —
**fully bit-exact, no tolerance seam**.  Suite 674/674 in both memory modes (one `gpuslang.scan`
SegFault on the first V2 run, the known llvmpipe flake — clean on re-run), rigid regression 16/16,
cloth / clothdrape / clothselfcollide / particle green.

**Remaining softbody free-fall chain**: `sb_gm_stepSoftbodyLaunch` → `sb_gm_finalizeVelocitiesLaunch`
→ `sb_sleeping` completes the minimal "a softbody falls and settles" path (2 of 56 softbody kernels
mapped so far), then the FEM-volume solve chain, then softbody-vs-rigid contact for a scene.

## P8.0 tranche 3 — the softbody free-fall chain CLOSES (step + finalize + sleeping)

Three kernels, all in `shaders/softBodySimBda.slang` alongside the tranche-1 preIntegrate, complete
the minimal "a softbody falls, settles and goes to sleep" path:

| kernel | source | what it does |
|---|---|---|
| `sb_gm_stepSoftbodyLaunch` | softBodyGM.cu:232 | the substep integrator — advances position by `vel*dt` and accumulates the delta into `mSimDeltaPos`, optionally re-applying gravity/damping/max-velocity-clamp first |
| `sb_gm_finalizeVelocitiesLaunch` | softBodyGM.cu:3136 | turns the accumulated delta back into a velocity, applies settling damping / velocity averaging, ORs the awake flag |
| `sb_sleeping` | softBodyGM.cu:3219 | per-body wake-counter bookkeeping + the state-changed bitmap |

ABI rows `{8,8,4,12,1}` (40 B), `{8,8,4,4,4,1}` (32 B) and `{8,4,8,4,4,8,8}` (48 B); manifest
hand-edited, **MAPPED 212 → 215**.  These are the softbody twins of the cloth trio in
`femClothSimBda.slang` and reuse its warp lowerings; the softbody-specific differences are that every
buffer is the SIMULATION mesh (`mSimPosition_InvMass` @96 / `mSimVelocity_InvMass` @88 / `mSimDeltaPos`
@128 / `mNumVertsGM` @372), the awake flag is `mAwake` @408, and finalize takes an explicit
`velocityScale` and blends `tVel*velocityScale + deltaVel*(1-velocityScale)` where the cloth kernel
rescales by a sqrt ratio — **different formulas, the cloth line must not be copied**.

One trap worth naming: `sb_gm_stepSoftbodyLaunch` launches **(1024,1,1)** with a plain 1-D index,
while its own file-mate `sb_gm_preIntegrateLaunch` launches (32,32,1) and builds the index from
`threadIdx.x + WARP_SIZE*threadIdx.y + ...`.  Copying the neighbouring kernel's thread mapping would
be silently wrong.  Its arg pack also orders `dt` BEFORE `gravity`, the opposite of preIntegrate's.

**A REGRESSION FOUND AND FIXED IN THE CLOTH PORT.**  While transcribing `sb_sleeping`'s
`(startIdx + threadIdx.x) < ((numActive + 31) / 32)` store guard it became clear the P7.0
`cloth_sleeping` port had dropped it: `stateChangedMask` is a `PxgBitMap` sized to the body count, so
with one live cloth exactly one word is live while all 32 lanes were storing — zeros written past the
end of the map.  The guard is restored in `femClothSimBda.slang`, and `femClothSimSleeping`'s
assertion — which had encoded the buggy behaviour by expecting zeros in the out-of-range words — now
requires the sentinel to survive instead.  The corrected test fails against the unguarded shader,
which is how the fix was confirmed.

**Verification.**  9 new bit-exact cases (raw float bits vs an `-ffp-contract=off` CPU transcription):
step x4 (plain / external-forces / velocity-clamp / gravity-disabled), finalize x4 (awake / asleep /
averaging / exactly-on-the-sleep-threshold), sleeping x1.  **19 mutants, all caught** — and two of
them only after the cases were sharpened:
  - the `eDISABLE_GRAVITY` gate had NO teeth until a case actually set the flag (every other case left
    it clear, so a mutant that always applies gravity passed);
  - the `awake = magSq >= tolerance*tolerance` comparison had no teeth until a case put every vertex
    EXACTLY on the threshold (`delta = (tolerance,0,0)` makes `magSq == tolerance*tolerance` to the
    bit).  The mixed case always contained a vertex above the settle tolerance, which set awake through
    the else-branch regardless — so both a `>=`→`>` flip and a 4x threshold loosening passed.

`PxSqrt` in the max-velocity clamp is IEEE-754 correctly rounded (unlike rsqrt / the trig family), so
it stays bit-exact — **no tolerance seam anywhere in this tranche**.  Suite 674/674 in both memory
modes (one `gpuslang.npRigidSumBatches` SegFault on one run — the known llvmpipe flake; the test
passes 3/3 in isolation and the suite re-run is clean), rigid regression 16/16,
cloth / clothhang / clothdrape / clothselfcollide / particle green.

**Softbody status: 5 of 56 kernels mapped.**  Next is the FEM-volume solve chain
(`sb_gm_cp_solveTetrahedronsPartitionLaunch` + `sb_updateTetrahedraRotationsLaunch` +
`sb_gm_applyExternalDeltasLaunch`), then softbody-vs-rigid contact, then a softbody scene.

## P8.0 tranche 4 — the softbody FEM-volume solve BRACKET (zeroTetMultipliers + applyExternalDeltas)

The two kernels that open and close each softbody solve iteration, in the new
`shaders/softBodySolveBda.slang`:

| kernel | source | what it does |
|---|---|---|
| `sb_gm_zeroTetMultipliers` | softBodyGM.cu:1801 | clears every tetrahedron's Lagrange-multiplier spatial vector at the top of the solve — without it the solver accumulates into last frame's multipliers and the material response drifts frame over frame |
| `sb_gm_applyExternalDeltasLaunch` | softBodyGM.cu:2609 | folds the accumulated per-vertex correction (`mSimDelta`, written by the contact/attachment solves) back into position, velocity and `mSimDeltaPos`, then clears the delta buffer — the softbody twin of the `cloth_applyExternalDeltasLaunch` the cloth arc has used since P7.1 |

ABI rows `{8,8}` (16 B) and `{8,8,4}` (24 B); manifest hand-edited, **MAPPED 215 → 217**.  Both are
pure load/scale/store: no transcendental, no warp op, no cross-lane communication — **fully bit-exact,
no tolerance seam**.

The multipliers live in a `PxgSpatialVectorBlock` (PxgArticulation.h:77), the 32-lane SoA block
`float4 mTopxyz_bx[32]` @0 then `float2 mbyz[32]` @512 → 768-byte stride, so zeroing tet `t` means two
stores at block `t/32`, lane `t&31`.  `sb_gm_applyExternalDeltasLaunch`'s opening `blockCopy` of the
whole 432-byte descriptor into `__shared__` + `__syncthreads` is a read-caching optimisation only and
is dropped (every lane reads through `Ptr<>` instead), which also removes the kernel's only barrier.

**Verification.** 2 new bit-exact cases, **10 mutants all caught**.  The zeroTetMultipliers case uses
`numTets = 37` — deliberately not a multiple of 32 — and over-dispatches by a block, so one case pins
the block/lane split, the `groupThreadIdx < numTets` guard, and the fact that ONLY the two words
`storeSpatialVector` writes are touched (every other word of every block keeps its poison).  The
applyExternalDeltas case spans the `delta.w` reference count — 0, 1, 3, 7, 2.5, 0.5, 4, 12 — with
explicit assertions that vertex 3 (`w == 7`) is averaged by exactly 1/7 and vertex 5 (`w == 0.5`) is
NOT scaled up, which is what gives the `1.0f / max(delta.w, 1.0f)` line teeth: a port that dropped
either the divide or the clamp matches the `w == 0` / `w == 1` vertices and only differs there.

Suite 674/674 in both memory modes, rigid regression 16/16, cloth / clothdrape / clothselfcollide /
particle green.  **Softbody now 7 of 56 kernels mapped.**  Remaining in the solve chain:
`sb_gm_cp_solveTetrahedronsPartitionLaunch` (the constraint solve itself) and
`sb_updateTetrahedraRotationsLaunch` (the APD / `extractRotation` polar decomposition — the first
softbody kernel with a genuine transcendental seam), then softbody-vs-rigid contact, then a scene.

## P8.0 tranche 5 — the tetrahedron ROTATION extractors, and the arc's FIRST TOLERANCE SEAM

`sb_gm_updateTetrahedraRotationsLaunch` (softBodyGM.cu:419, sim mesh) and
`sb_updateTetrahedraRotationsLaunch` (:461, collision mesh), both in
`shaders/softBodySolveBda.slang`.  Each computes the deformation gradient F = P·Qinv from its
tetrahedron's displacement field, seeds the quaternion from the stored rotation (or
`mInitialRotation` on the first frame, detected via the zero-initialised memory), and runs Mueller's
iterative polar decomposition `extractRotation(F, q, 100)`.  ABI row `{8,8}` (16 B, shared with
`sb_gm_zeroTetMultipliers`); manifest hand-edited, **MAPPED 217 → 219**.

They differ in exactly three places: the buffer set (sim `mSimTetIndices` @80 /
`mSimPosition_InvMass` @96 / `mSimTetraRestPoses` @136 / `mSimTetraRotations` @168 / `mNumTetsGM`
@376, versus collision `mTetIndices` @16 / `mPosition_InvMass` @64 / `mTetraRestPoses` @40 /
`mTetraRotations` @56 / `mNumTets` @368); the rest-pose LAYOUT (sim reads through `loadPxMat33` from a
`PxgMat33Block` — 1152-byte stride, the 9 elements split 4/4/1 across `mCol0`/`mCol1`/`mCol2` — while
collision reads a flat 36-byte `PxMat33`); and the enabling test (collision runs only under
`eCOMPUTE_STRESS_TENSOR` or when self-collision is ENABLED; sim runs unconditionally).  Both use
`maxIter = 100`; the 100-vs-4 split lives only in `computeGMTetrahedronRotations`, which neither
kernel reaches.

### The seam

`extractRotation` builds each update as `PxQuat(w, omega)`, and that constructor (PxQuat.h:86)
evaluates `PxSinCos` → sinf/cosf, which have no cross-backend rounding guarantee.  The `.cu` picks
this variant on the default path (the `sb_extractRotationAPD` call sits commented out directly above),
so **the seam cannot be avoided by configuration** — which is exactly why `npSbExtractRotAPD.slang`
documents the Mueller variant as non-portable and ports the APD one instead.  Everything else in the
chain stays bit-exact: `PxVec3::normalize` and `PxQuat::getNormalized` both go through `PxSqrt`
(IEEE-754 correctly rounded), and the quaternion product and rotation-matrix build are pure arithmetic.

**Two measurement mistakes had to be corrected before the seam could be budgeted honestly**, and both
were caught by the numbers rather than assumed away:
  - a raw quaternion compare read **2.1e9 ULP** on the seeded case — because q and −q are the SAME
    rotation and the iteration is free to land on either representative.  The quaternion is now
    sign-normalised before any ULP is reported;
  - the natural-looking `acos(|dot|)` angle metric is badly conditioned near d == 1: a 1-ULP
    difference reads as **3e-4 rad**, which says nothing about the port.  The primary metric is now
    the Frobenius distance between the two rotation MATRICES — sign-agnostic by construction, no
    amplification, and it is what the solver actually consumes.

Measured, and the two regimes genuinely differ: seeded from the STORED rotation (near the fixed
point) gives **1 ULP / 6e-8**; seeded from `mInitialRotation` (~60° away) gives **860 ULP / 2e-5**.
The iteration AMPLIFIES the seam rather than washing it out, because the `w < eps` early-exit is
itself seam-dependent — a last-bit sin/cos difference moves w across the threshold and the two sides
stop one iteration apart.  2e-5 on the rotation matrix is ~0.001°, irrelevant to the solver, but it is
a real divergence and is budgeted as such.  Guarantees: **≤ 4096 ULP and < 1e-4 matrix distance**,
~5x above the measurement.

**The seam does not cost the test its teeth**, which is the point that mattered: 5 mutants (iteration
count 100→3, a wrong `PxgMat33Block` decode, F = Qinv·P instead of P·Qinv, eps 1e-6→1e-3, and a
swapped quaternion product) all land at **1e-2 to 1.9** matrix distance — 500x to 100000x above the
budget.  The tolerance sits in a wide, empty gap between the noise floor and the weakest real error.

**Two findings from the adversarial analysis, both verified by hand:**
  - `PxgSoftBodyCore::updateSimTetraRotations()` — the launcher for the SIM-mesh kernel — has **zero
    call sites** in the tree.  `SB_GM_UPDATETETRAROTATIONS` is never dispatched by the shipping
    pipeline; the sim-mesh rotations are maintained inside the solve kernels via
    `computeGMTetrahedronRotations` instead.  Only the collision variant actually runs, once per
    frame, from PxgContext.cpp:2024.  The sim kernel is ported and correct, but it is currently dead
    weight — worth knowing before anyone spends time on a scene that expects it to fire;
  - the upcoming solve kernel has a **degenerate-rest-pose guard** that must not be dropped:
    softBodyGM.cu:1123 `if(!(Qinv == PxMat33(PxZero))) { tetrahedronsSolveGM(...); }` (the scatter
    still runs, with a zero delta), and :1178 `if(Qinv == PxMat33(PxZero)) return;` — placed AFTER the
    rotation write, so the quaternion is persisted before the early return.

Suite 674/674 in both memory modes, rigid regression 16/16, cloth / clothselfcollide / particle green.
**Softbody now 9 of 56 kernels mapped.**

## P8.0 tranche 6 — the FEM-volume constraint SOLVE lands (and the "driver bug" was mine)

`sb_gm_cp_solveTetrahedronsPartitionLaunch` (softBodyGM.cu:1408) — the core of the softbody solver —
is ported, green and mapped, together with its whole device tree: `tetrahedronsSolveGM` (the material
dispatcher), both constitutive branches (`tetrahedronsSolveInnerNeoHookean` + `volumeSolve` for
eNEO_HOOKEAN, `ARAP_constraint` + `volume_constraint` for eCO_ROTATIONAL) and the shared compliance
primitives (`compute_dCdx`, `compute_deltaLambdaXgradC`).  ABI row `{8,8,4,4,4,1,1,8}` (40-byte blob);
manifest hand-edited, **MAPPED 219 → 220**.

### The bug, and why in-harness bisection could not find it

An earlier commit recorded this kernel as blocked on an llvmpipe fault, with a long list of ruled-out
causes.  **That conclusion was wrong.**  The defect was in the port:

```slang
Ptr<uint>(matIdxAddr & ~3UL)[0]     // the mOrderedMaterialIndices PxU16 read
```

**Slang evaluates `~3UL` in 32 bits.**  The resulting `0xFFFFFFFC` is zero-extended for the 64-bit
AND, silently clearing the TOP HALF of the buffer-device address.  The read lands at a garbage
address and the driver faults — correctly.  `~uint64_t(3)`, `0xFFFFFFFFFFFFFFFCUL` and
`- (addr % 4UL)` all run clean on the same shader and data.  The `UL` suffix does not make the
*complement* 64-bit; this is a trap for any BDA port and is now called out at the site.

The in-harness bisection had produced a **misleading correlation** — "it faults whenever the computed
delta is non-zero" — that survived every experiment because it was real but not causal: `material` is
read only on the path the delta enables, so with the solve stubbed out the bad read is dead code and
gets optimised away.  Chasing that correlation ruled out the arithmetic, the buffer sizes, the
signature width, the indexed local arrays and the shader size — all correctly, and all irrelevant.

### What actually settled it

`tools/lvprepro/` — a standalone Vulkan harness (no PhysX, no descriptor sets, no test framework)
that dispatches a real compiled `.spv` against synthetic data, building the `PxgSoftBody` descriptor
by raw byte offset.  The same shader faulted the same way there, which removed PhysX from the
picture; two substitutions on the masking expression then isolated the cause exactly.  The harness
and its README are kept, because "shader bug or driver bug?" recurs and the suite cannot answer it.

**Lesson recorded for the next tranche:** a correlation that survives many experiments is not a
cause.  Once the in-harness search space was exhausted, the right move was to leave the harness — not
to keep subdividing inside it.

### Verification

5 cases (co-rotational PGS / damped / TGS, neo-Hookean PGS / TGS) against a full CPU transcription of
the device tree.  **A tolerance seam is inherited** — not from the constitutive maths, which is
mul/add/sub/div/cross/dot plus one `PxSqrt` and would be bit-exact alone, but from the rotation
refresh (`computeGMTetrahedronRotations` → `extractRotation` → `PxQuat(w, omega)` → sin/cos, the
tranche-5 seam).  Every solve term is a function of that quaternion.  Measured across the five cases:
**10-248 ULP, up to 2.24e-5 relative**, the damped co-rotational case widest (its denominator carries
`1 + damping*dtInv` and a `scaledDamping*velocity` term, both of which amplify a last-bit rotation
difference).  Budget: 4096 ULP AND 1e-3 relative, ~15x above the measurement.

**The seam does not cost the test its teeth.**  5 mutants — ARAP sign flip (F-R → R-F), neo-Hookean
C offset 3 → 2, `alphaTilde0` missing its `2*mu`, the `dCdx` transpose dropped, and lambda assigned
instead of accumulated — land at **relative 0.26 to 217**, i.e. 12000x to 10000000x above the budget.
The tolerance sits in a vast empty gap.

Suite 674/674 in both memory modes, rigid regression 16/16, cloth / clothselfcollide / particle green.
**Softbody now 10 of 56 kernels mapped**, and the FEM-volume solve — the piece everything else in the
softbody arc was waiting on — is live.

## P8.0 tranche 7 — the softbody step CLOSES: copy-merge, Jacobi apply, and the skin

Three kernels, ported into `softBodySolveBda.slang`, that together take the solve loop from
"computes a correct delta" to "moves a mesh anyone can look at":

| kernel | .cu | launch | ABI row | blob |
|---|---|---|---|---|
| `sb_gm_cp_averageVertsLaunch` | softBodyGM.cu:1509 | PxgSoftBodyCore.cpp:1298 | `{8,8,4}` | 24 B |
| `sb_gm_applyDeformationDeltasLaunch` | softBodyGM.cu:2665 | PxgSoftBodyCore.cpp:1269 | `{8,8,4}` | 24 B |
| `sb_gm_updateTetModelVertsLaunch` | softBodyGM.cu:1589 | PxgSoftBodyCore.cpp:828 | `{8,8}` | 16 B |

Manifest hand-edited, **MAPPED 220 → 223**; softbody **10 → 13 of 56**.

**Why these three and not any other three.** The tranche-6 solve writes each vertex once *per
partition copy* into `mSimPosition_InvMassCP`, past a leading block of `numElements *
numVertsPerElement` element-corner entries. Nothing downstream reads that buffer.
`averageVerts` is what folds the copies back onto the canonical vertex — without it the solve
runs, produces correct numbers, and the mesh does not move at all. `applyDeformationDeltas` is
the same closing move for the Jacobi vertex subset. And `updateTetModelVerts` is the last
kernel of the step: the solver moves the SIMULATION mesh, but narrowphase, the render readback
and the user's `PxDeformableVolume` vertex buffer all read the COLLISION mesh, which only
exists as a barycentric skin over the sim tetrahedra. It is the kernel that makes softbody
motion observable outside the solver.

**The engine's own block-dim mismatch, mirrored rather than fixed.** `averageVerts` and
`updateTetModelVerts` compute `numBlocks` from `SB_SOLVETETRA` (64) and then launch with
`SB_PREINTEGRATION` (1024) threads per block — a 16x over-dispatch that the
`groupThreadIdx < numVerts` guard makes harmless. The port keeps `[numthreads(1024,1,1)]` so
that `threadIdx.x + blockIdx.x * blockDim.x` reproduces the same index for the same
(block, thread) pair. `applyDeformationDeltas` is consistent at 64.

**No tolerance seam.** Not one transcendental between the three; everything is compared through
raw float bits (`f2u`). The shader is in `PXG_NOCONTRACT_SHADERS`, which matters most for
`updateTetModelVerts`' four-term barycentric blend `p0*b.x + p1*b.y + p2*b.z + p3*b.w` — a
textbook FMA-fusion candidate.

### Verification

Four cases (`softBodySimAverageVerts`, `…Kinematic`, `softBodySimApplyDeformationDeltas`,
`softBodySimUpdateTetModelVerts`), all bit-exact against a CPU transcription. The fixtures are
built so the plausible wrong port is *distinguishable*: the copy buffer's leading element-corner
block is poisoned far from the real copies; per-vertex copy counts vary 1/2/3/1/2/1 so both
vertex 0's implicit start-at-0 and the general `accum[i-1]` case are live; one vertex is pinned
(`invMass == 0`) but owns two copies; the velocity's `w` channel is *never* equal to `pos.w`;
the Jacobi index list is a permuted, non-contiguous subset; the sim tets' corner indices are
permuted so a transposed blend shows immediately; and `mNumVerts` is deliberately different
from `mNumVertsGM`.

**10 mutants, 10 caught** (forced-fresh: `.spv` removed before every rebuild):
the copy-buffer `offset` dropped; the `1/numCopies` divide dropped; velocity `w` taken from
the averaged value instead of `pos_invMass`; the `eKINEMATIC` early-out dropped; the
`pos_invMass.w > 0` gate dropped; the Jacobi indirection replaced by the lane index;
`mJacobiScale` replaced by the external-delta rule `1/max(delta.w,1)`; the delta clear dropped;
the skin guard switched to `mNumVertsGM`; the barycentric `z`/`w` swapped; the remap
indirection dropped; and a write to the sim mesh added.

Suite **674/674 in both memory modes**, rigid regression **16/16**, cloth / clothdrape /
clothselfcollide / particle green.

## P8.0 — MILESTONE: a FEM-volume softbody runs live on Vulkan (`PXG_SCENE=softbody`)

`testGpuSdkStep.cpp` gains the softbody money shot: a hand-authored tetrahedral cube (2x2x2 cells,
27 vertices, 40 tets) built through `immediateCooking::createDeformableVolumeMesh`, attached as a
real `PxDeformableVolume`, and stepped by the engine over the Vulkan bridge. Three modes, all green:

| invocation | what it proves | measured |
|---|---|---|
| `PXG_SCENE=softbody` | the integrate chain | drop 4.98675 vs 4.98675 expected, **0.0000% off** |
| `+ PXG_SB_STRETCH=1.6` | the constitutive solve under gravity | y-extent 1.600 → 1.076, **87.4% recovered** |
| `+ PXG_SB_NOGRAVITY=1` | the solve converges and HOLDS | 1.600 → 1.035 (**94.2%**), then constant to **0.0e+00** over 20 steps |

Run any of them with `PXG_SOLVER=pgs PXG_GPU_STEP=1`.

### The observable is the COLLISION mesh, on purpose

The two meshes are produced by different kernels. The SIM mesh (`getSimPositionInvMassBufferD()`)
is what `sb_gm_preIntegrateLaunch` predicts, `sb_gm_cp_solveTetrahedronsPartitionLaunch` corrects
and `sb_gm_cp_averageVertsLaunch` folds back. The COLLISION mesh (`getPositionInvMassBufferD()`)
only ever moves because `sb_gm_updateTetModelVertsLaunch` skins it onto the sim tetrahedra. They
are separate device allocations (the scene asserts the two pointers differ, so the skin check
cannot be satisfied by aliasing), and both are read back every step.

**Two A/B runs give the scene its teeth**, each done by hand-editing one manifest row to `nullptr`:

- `sb_gm_updateTetModelVertsLaunch` unmapped → the collision mesh freezes at exactly `4.00000`
  while the sim mesh falls to `3.42775`. Skin gap 0.57225, two checks fire.
- `sb_gm_cp_averageVertsLaunch` unmapped → the stretched cube's y-extent stays at exactly
  `1.60000` forever. The solve still runs; its output simply never reaches the mesh. **That merge
  is the sole conduit from the partitioned solve to the simulation mesh.**

So the scene is not merely "a softbody moved" — it fails loudly if either tranche-7 kernel is
removed.

### Two things this scene does NOT do, stated plainly

**It does not settle.** The cube passes straight through the ground box and ends at mean
y = −0.987. The whole softbody-vs-rigid contact arc (`SB_MIDPHASE_PRIMITIVES`, `SB_PRIMITIVES_CG`,
`SB_OTHER_CONTACT_REMAP_TO_SIM`, `SB_RS_CONTACTPREPARE`, `SB_SOLVE_RIGID_SOFT_COLLISION`) is still
unmapped, so there is nothing to catch it. This is a contact-free free-fall scene, exactly as
`PXG_SCENE=cloth` was at P7.0.

**It does not drive `sb_gm_applyDeformationDeltasLaunch`.** The cooker pins
`mGridModelNbPartitions` to 8 for a one-tet-per-element mesh, so the Jacobi remainder is empty and
the host never takes that branch. The kernel stays covered by its unit test only. (The other two
tranche-7 kernels do run — `averageVerts` because `mUsePartitionAveraging` is set for any mesh with
one tet per element, `updateTetModelVerts` twice per frame.)

Eleven kernels are refused per step in this scene; all of them contribute only contact, attachment
or stress corrections that are identically zero without contacts. The refusals are non-fatal.

### The outward creep, and why it is not a port defect

With gravity ON and a stretched spawn, the cube relaxes to y-extent ~1.034 by step 20 and then
creeps back OUT to ~1.13 by step 120, decelerating. Two experiments placed it:

- **Iteration count is not the cause.** 4x the XPBD position iterations moves the step-119 extent
  only 1.128 → 1.109, so it is not a convergence residual.
- **Gravity is.** With `PXG_SB_NOGRAVITY=1` the identical solve converges to 1.03481 and is then
  constant to five decimal places from step 60 through 120. The constitutive damping term couples
  to the vertices' ABSOLUTE velocity, and in free fall that grows without bound.

So the falling case only asserts that a substantial contraction happened; the tight
"relax and hold" assertion lives in the gravity-free mode where the solve is isolated. The
remaining 3.5% is XPBD's finite compliance, not drift.

`PXG_SB_NOGRAVITY` also earns its keep as coverage: it puts the tranche-1 pre-integrate gravity
gate (`PxActorFlag::eDISABLE_GRAVITY`) on its skip path — the one branch its unit test could not
give teeth to, because no case set the flag. The scene budgets the check against what gravity WOULD
have done (2% of 19.79 at 120 steps) rather than against zero, because a stretched cube's
relaxation shifts its centroid ~0.018.

### Notes from building it

- `physxextensions` is not linked into this harness (only `ExtJoint.cpp` / `ExtD6Joint.cpp`), so
  `PxTetMaker` and `PxDeformableVolumeExt` are unavailable. The tet mesh is hand-authored and the
  host-mirror fill open-codes what `allocateAndInitializeHostMirror` + `transform` + `copyToDevice`
  do.
- The alternating 5-tet cube split does **not** produce consistent handedness. Each tet's signed
  volume is checked and two corners swapped when it comes out negative.
- `buildGPUData = true` is mandatory and fails **silently**: without it the tet cooker leaves
  `mGRB_primIndices` NULL and then dereferences it.
- `setSolverIterationCounts` is called through `PxDeformableBody`; the same-named
  `PxDeformableVolume` override is `PX_DEPRECATED` and hides the live one.

Suite **674/674 in both memory modes**, rigid regression **16/16**, cloth / clothhang / clothdrape /
clothselfcollide / particle green.

## P8.1 — the cloth delta accumulators get REAL atomics (and a confirmed, unfixed grid defect)

Three cloth shaders accumulated `mDeltaPos` with a plain load-add-store where CUDA uses
`atomicAdd`. Each carried the same written justification:

> *"a contact set with no two contacts sharing a cloth vertex within one pass has no
> cross-thread contention — the read-modify-write is then bit-identical to the atomic"*

**That condition never holds.** A cloth TRIANGLE contact bumps all three of its vertices and
adjacent triangles share vertices by construction; a folded cloth's self-collision has many
contacts per vertex. Under contention the RMW drops updates — and because `mDeltaPos.w` is the
DIVISOR `applyExternalDeltas` mass-averages with (`scale = 1/max(w,1)`), a lost count update
desynchronises the accumulated `.xyz` from its own divisor. The correction is then applied with
the wrong weight, which presents as an unstable or under-responsive solve rather than as a lost
write.

| file | site |
|---|---|
| `femClothClothContactBda.slang` | `deltaBumpW` / `deltaAddXYZ` (self-contact) |
| `femClothRigidContactBda.slang` | the query `.w` bumps (4) + the solve `.xyz` scatters (2) |
| `femClothRigidAttachBda.slang` | `applyClothDelta` |

`solverTGS.slang:2443` was checked and is NOT an instance — it already runs a proper CAS loop,
and its non-atomic pre-test mirrors the `.cu` exactly.

New module `pxgSlangAtomicFloat.slang`. Slang's `InterlockedAdd` on a `Ptr<float>` emits
`OpAtomicFAddEXT` (capability `AtomicFloat32AddEXT`, extension
`SPV_EXT_shader_atomic_float_add`); the bridge now enables `VK_EXT_shader_atomic_float` +
`shaderBufferFloat32AtomicAdd` where the device advertises them — llvmpipe does. Note that
`VkDeviceCreateInfo` previously enabled **no device extensions at all**, so this adds the
`ppEnabledExtensionNames` path as well, gated on `vkEnumerateDeviceExtensionProperties`.

### What it bought

The determinism the P7.2 tranche documented as inherent — *"the ON path is genuinely
NON-deterministic (the contact-slot atomics and the mDeltaPos accumulation are
order-dependent)"* — was mostly this bug. `clothselfcollide`, self-collision ON:

| | mean spread | min spread |
|---|---|---|
| before (non-atomic) | 4.971 – 5.199 (**0.23**) | 4.199 – 4.943 (**0.74**) |
| after (atomic) | 5.045 – 5.050 (**0.006**) | 4.597 – 4.622 (**0.025**) |

Float addition is still not associative, so an atomic accumulation remains order-dependent and
not bit-reproducible — but that is parity with CUDA's `atomicAdd`, not a regression. What
disappeared is the 30x wider spread that came from dropping writes outright.

### The grid defect: CONFIRMED, and deliberately NOT fixed yet

While reading the launch site for the softbody midphase, the same launch shape turned up a
separate, real defect in `femClothSelfCollisionBda`:

- the engine packs `MIDPHASE_WARPS_PER_BLOCK = 2` warps per block and launches
  `ceil(N / 2)` blocks (`PxgFEMClothCore.cpp:714`);
- the Vulkan bridge **drops `blockDim`** and passes only `gridDim` through
  (`PxgVulkanCudaContext.cpp:293`);
- the shader is `[numthreads(32,1,1)]` and indexes `globalWarpIdx = gid.x`.

So it covers only the first half of the cloth vertices (VT) and of the
triangles-with-active-edges (EE). That is arithmetic, not conjecture: `mMaxNbClothVerts` is the
plain vertex count (`PxgSimulationCore.cpp:1959`).

A tree-wide sweep for this class found **no other instance**: `sb_refitBoundLaunch` /
`cloth_refitBoundLaunch` are correct (`numBlocks == nbActiveBodies`, so `gid.x` IS the body
index), `convexConvexNphase_stage2Kernel` is correct (`GJK_EPA_WARPS_PER_BLOCK == 1`), and the
PGS/TGS solver kernels index flat off their descriptors and are covered by the 16/16 rigid
regression.

The fix (a group-uniform serial loop over the block's warps, keeping one group per warp so the
traversal barriers stay uniform) was written and **verified correct**: contacts go VT 76 → 152
and EE 34 → 70, exactly the missing half, and the VT unit test re-dispatched at the engine's own
`ceil(N/2)` grid still asserts `gCount == 1`, which rules out duplication.

**But with the correct contact set the scene gets markedly worse** — the upper layer falls to
mean 4.066 / min 3.226, against 5.047 with half coverage and 3.985 with self-collision off. With
the atomics in place this is now reproducible to four decimals, so it is a real effect and not
race noise. The prepare and solve kernels were checked and use a fixed grid-stride grid, so
there is no second mismatch feeding a divisor/numerator desync.

**The cause is not yet understood, so the fix is withheld rather than landed.** Turning a green
money shot red on a theory is the exact mistake the tranche-6 write-up records; the next step is
to instrument per-vertex `mDeltaPos.w` and `.xyz` across the two coverage regimes rather than to
keep reasoning about it.

### Second investigation round: what the grid defect actually exposes

The fix was re-applied and the scene instrumented per step. The mechanism is now visible, and
it is NOT lost contacts:

| step | half coverage | full coverage |
|---|---|---|
| 0 | mean 5.083 | 5.098 |
| 5 | 5.195 | 5.264 |
| 20 | 5.255 (peak) | **5.415** (peak) |
| 60 | 5.195 | 4.827 |
| 100 | 5.057 | 4.124 |

The upper layer is pushed **above its own spawn height (5.05)** in BOTH regimes — self-collision
is meant to arrest the fall, not lift the cloth. Full coverage roughly doubles that push, the two
layers separate, and the upper layer then passes **through** the shelf (min y 4.31 at step 60,
shelf at 5.00) instead of being re-caught. It tunnels; it does not settle low.

Ruled out this round, each by measurement rather than by reading:

- **contact-buffer overflow** — capacity 1 048 576, counters in the tens; `checkBufferOverflows`
  never fires;
- **duplication by the sub-warp loop** — the VT unit test re-dispatched at the engine's own
  `ceil(N/2)` grid still asserts `gCount == 1`;
- **the count pass** — the Slang `countContact` matches `FEMClothExternalSolve.cu:861` exactly:
  `.w` bumped by `|coeff_i|` (not by 1), the same `> 1e-3` gate, the same
  `distSq > restDist^2 -> markInCollision(false)` early-out;
- **the divisor and the clear** — `scale = 1/delta.w` gated on `w != 0`, whole-`float4` clear,
  identical to `cloth_applyExternalDeltasLaunch` (`FEMClothExternalSolve.cu:1053`);
- **a second grid mismatch downstream** — the contact prepare and both solve kernels launch on a
  fixed `SB_UPDATEROTATION` grid with a grid-stride loop, so they see every contact regardless;
- **solver iteration budget** — 8 / 16 / 32 collision substeps give 4.0658 / 4.0657 / 4.0656.
  Identical. More iterations cannot help once the layers have separated.

So every link of the chain that can be compared against the `.cu` by reading has been compared,
and matches. What remains is the per-contact correction magnitude, or the contact-refresh cadence
under larger displacements — and neither can be settled in a 40-vertex draping scene with no CUDA
ground truth to compare against. The next step is a unit-level fixture that drives
count -> solve -> apply end to end on a geometry whose correct outcome is known analytically, run
at both coverage levels. `PXG_CLOTH_SUBSTEPS` was added to the scene so this margin stays
measurable rather than assumed.

### Third round: two analytic fixtures, and the solve chain is EXONERATED

The existing self-contact tests compare the port against a CPU transcription of the same
logic, so they agree with each other whether or not that logic is right. Two new fixtures
in `testFemClothSim.cpp` assert an **analytically known** outcome instead, driving the real
chain end to end (`prepare -> queryCount -> solve -> applyExternalDeltas`):

**`femClothSimSelfContactRedundancy{1,2,4}`** — a free vertex above a PINNED triangle,
`gap = 0.04` against a total rest distance of `0.10`. The pinned triangle drops out of the
constraint (invMass 0), so one pass must place the vertex at exactly the rest distance. With
N identical contacts describing the one overlap:

| N | `mDeltaPos.w` | `mDeltaPos.z` | displacement |
|---|---|---|---|
| 1 | 1.0000 | 0.060000 | **0.060000** |
| 2 | 2.0000 | 0.120000 | **0.060000** |
| 4 | 4.0000 | 0.240000 | **0.060000** |

Count and accumulator both scale with N and the divide cancels them exactly. **Redundant
contacts do not over-respond.**

**`femClothSimSelfContactTwoWay`** — this is what the grid fix actually adds: not copies, but
the *opposite direction*. Two parallel FREE triangles, `gap = 0.04`; direction 1 is
`(a0, TB)`, direction 2 is `(b0, TA)`. Both are genuine, distinct constraints on distinct
endpoint sets, so the mass-averaging above does not relate them.

```
ONE dir:    separation 0.040000 -> 0.100000   (rest distance 0.100000)
BOTH dirs:  separation 0.040000 -> 0.100000
over-separation factor: 1.0000
```

**The two directions agree rather than add.** Resolving the same overlap from both sides
lands on the rest distance, not twice past it.

So the `count -> solve -> apply` arithmetic is correct under both regimes that could plausibly
have produced the over-response, now pinned against closed-form answers rather than against a
transcription. Combined with the six causes excluded in the previous round, **the solve chain
is exonerated**: whatever makes the corrected contact set destabilise `clothselfcollide` is
not in the contact arithmetic.

What that leaves is the contact SET and its cadence — the midphase finding contacts it should
not, or the pair-refresh rate (`setNbCollisionPairUpdatesPerTimestep`) being too coarse for the
larger displacements the corrected set produces — or genuine physics the scene was never tuned
for, since its thresholds were calibrated against half the contacts. The fixtures are the
permanent artefact: any future change to this chain now has an analytic reference, not a
mirror.

### Fourth round: the overshoot is FOUND, measured, and inherited from upstream

The two fixtures above both put every participant in exactly **one** contact per role — which
turns out to be the one configuration where the arithmetic is neutral. The realistic case is
one vertex penetrating **K distinct** opposing triangles at once, which is what a dense fold
produces. `femClothSimSelfContactManyTriangles` sweeps it:

| K | count[vertex] | count[corner] | vertex Δ | corner Δ | overshoot |
|---|---|---|---|---|---|
| 1 | 1.0 | 0.3333 | −0.054000 | +0.018000 | **1.2000×** |
| 2 | 2.0 | 0.3333 | −0.056842 | +0.009474 | 1.1053× |
| 4 | 4.0 | 0.3333 | −0.058378 | +0.004865 | 1.0541× |
| 8 | 8.0 | 0.3333 | −0.059178 | +0.002466 | 1.0274× |
| 16 | 16.0 | 0.3333 | −0.059586 | +0.001241 | 1.0138× |

**The pass overshoots.** A lone all-free vertex-triangle contact at the centroid closes 120% of
the penetration deficit instead of 100%.

**The cause.** `mDeltaPos.w` accumulates `PxAbs(coeffs[j])` — a *weight* — not `1`, despite the
`.cu`'s own comment at that site calling it "how many times each vertex is referenced"
(`FEMClothExternalSolve.cu:877-896`). Since the solve multiplies the inverse masses by those
counts and the apply divides by them, the effective denominator becomes `Σ m_j |w_j|³` where
XPBD wants `Σ m_j w_j²`. The ratio between the two IS the overshoot factor. It is bounded by
the vertex side's weight of 1 (hence ≤ 1.2 for a centroid hit) and decays as the vertex-side
count comes to dominate the denominator — which is exactly the measured shape.

**This is upstream PhysX 5.6.1 arithmetic, not a migration regression.** The `.cu` bumps by
`PxAbs(coeffs[j])`; `femClothClothContactBda.slang:505-513` does the same. The port is
faithful. The fixture is therefore written as a **regression fence** — it pins the inherited
behaviour and its monotone decay, so a future port change that alters either fails loudly —
not as a correctness claim.

**How this closes the loop on the grid defect.** The overshoot is per *pass*, and
`applyExternalDeltas` runs many times per frame (VT and EE per collision substep, times the
substep count, plus the solver iterations). A few percent of over-separation compounded that
often is exactly the "layer pushed above its own spawn height" seen in the scene. At half
coverage only half the vertices ever act as the *vertex side* of a contact — the side that
carries weight 1 and absorbs the overshoot — so the effect is roughly halved. Fixing the grid
doubles the number of overshooting pushes, the fold separates, and the layer then falls
through. **The grid fix is correct; what it amplifies is inherited upstream behaviour.**

That reframes the remaining work: this is no longer a hunt for a port defect, it is a decision
about whether to deviate from upstream (bump `.w` by 1 instead of `|coeff|`, which would make
the factor identically 1) or to tune the scene around the inherited behaviour. That is a
judgement call about fidelity-versus-stability and belongs with the project owner, not in a
silent code change.

### Resolution: the grid fix LANDS, and the scene was the thing that was wrong

Staying faithful to upstream. The grid fix is in
(`femClothSelfCollisionBda.slang`, a group-uniform serial loop over the block's warps, with
the unit tests re-dispatched at the engine's own `ceil(N/2)` grid), and `clothselfcollide`
was rebuilt around the corrected contact set.

**Density was not the lever.** Sweeping the mesh from 40 to 144 vertices changed nothing —
every configuration landed at mean ≈ 4.03–4.07. **The initial condition was.** The scene
spawned the upper layer 0.05 above the shelf while the contact rest distance is 0.10, i.e.
*inside* it — handing the solve a 0.05 penetration deficit on frame one, to be overshot
every pass forever. With half the contacts that was survivable; with all of them it ejected
the fold.

The layer now starts at 0.30 above the shelf and **falls**:

| | min y | mean y |
|---|---|---|
| self-collision ON | **5.0999** | 5.1105 |
| expected rest distance | 5.1000 | |
| control (OFF) | 3.1180 | 3.8751 |

It settles **on** the contact rest distance, 0.0001 off, and the run is **bit-identical over
five runs** — the scene that P7.2 documented as *"genuinely NON-deterministic (the contact-slot
atomics and the mDeltaPos accumulation are order-dependent)"*. The atomics removed the
dropped writes; removing the artificial deficit removed the rest.

The pass criterion is correspondingly sharper. It no longer asks whether the layer stayed
"roughly up there" against a threshold read off an 8-run spread; it asserts that the layer
settles **on** `baseY + 2*restOffset` to within 0.03, which is a physical prediction of the
contact model rather than a fitted number.

This closes the arc. The port is faithful to upstream at every step that was questioned; the
one real defect (the non-atomic accumulators) is fixed; the one real coverage bug is fixed;
and the inherited per-pass overshoot is measured, fenced by
`femClothSimSelfContactManyTriangles`, and no longer papered over by a scene that was
accidentally tuned to hide it.

### A structural gap in the verification protocol

The unit tests **choose their own dispatch grid**, so they cannot catch an engine-grid mismatch
by construction — the self-collision test dispatches `gridX = numVerts`, a grid the engine never
issues, which is precisely what made the buggy indexing look correct. Any test for a kernel whose
engine launch packs warps per block has to mirror the engine's grid formula, not a convenient one.

Suite **674/674 in both memory modes**, rigid regression **16/16**, cloth / clothhang /
clothdrape / clothselfcollide / particle / softbody green.

## P8.2 tranche a — the softbody-vs-rigid MIDPHASE opens the contact chain

`sb_midphaseGeneratePairsLaunch` (softbodyMidPhase.cu:199, launch PxgNarrowphaseCore.cpp:4221) is
ported into the new `softBodyContactGenBda.slang`. One warp per contact-manager pair walks the
softbody's own BV32 tetrahedron BVH against the rigid's fattened world AABB and appends
`uint4(cmIdx, tetIdx, 0, 0)` to the shared collision stack. ABI row `{4,4,8,8,8,8,8,8,4,8,8}`
(80-byte blob); manifest hand-edited, **MAPPED 223 → 224**; softbody **13 → 14 of 56**.

This is the first of the seven kernels standing between the `PXG_SCENE=softbody` cube and actually
settling on the ground instead of passing through it.

**Almost nothing here was written from scratch.** The BV32 walk, the leaf/child decode and the
groupshared ballot/scan lowering come verbatim from `femClothSelfCollisionBda`; the traverser body
(surface-hint gate, `isValidTet`, `tetBoundingBox`) is the already-unit-tested transcription in
`npMidphaseDeform.slang` lifted onto `Ptr<>` reads; the `PxgContactManagerInput` / `PxgShape`
mirrors and the `bounds[rigidCacheRef]` read are `femClothContactGenBda`'s.

**The grid was right from the first line.** The engine packs `MIDPHASE_WARPS_PER_BLOCK = 2` warps
per block and the bridge forwards only `gridDim`, so the block's warps are walked serially in a
group-uniform loop. That is the same mismatch that shipped undetected in `femClothSelfCollisionBda`
and cost the whole P8.1 investigation; here the *test* dispatches the engine's own grid
(`ceil(numTests/2) = 1` group for two pairs) so a regression fails loudly.

### Verification

`softBodySimMidphaseGeneratePairs` drives two contact-manager pairs — pair 0 with the tetrahedron
mesh as shape0, pair 1 with it as shape1, so both branches of the shape swap run — over five
tetrahedra that pin each gate independently:

| tet | property | expected |
|---|---|---|
| 0 | surface, valid, inside the box | **emitted** |
| 1 | interior (hint 0), inside | dropped by the surface-hint gate |
| 2 | surface, valid, far outside | dropped by the tet-bounds test |
| 3 | surface, **degenerate**, inside | dropped by `isValidTet` |
| 4 | surface, valid, at x = 2.09 | **emitted** — only the fully fattened box reaches it |

Every child bound in the tree is deliberately generous, so the node-level cull can never stand in
for a primitive-level gate: each tet must be rejected by its *own* test.

**11 mutants, 11 caught** (forced-fresh `.spv` removal before every rebuild): sub-warp loop dropped,
surface-hint gate dropped, the `& ~uint64_t(3)` byte-extract alignment dropped, `isValidTet` dropped,
tet-bounds test dropped, leaf primitive index off by one, shape swap dropped (SIGSEGV — the rigid's
`particleOrSoftbodyId` is `0xFFFFFFFF`), fattening dropped, fattening from only the rigid's contact
distance, fattening from only the softbody's, the `stackIndex < stackSize` drop removed, and
`stackSizeBytes` used without dividing by `sizeof(uint4)`.

Three of those exposed real gaps in the fixture before it had teeth on them, and each was closed:
the contact distances are now **asymmetric** (0.08 + 0.02) with tet 4 at 2.09, so neither half alone
reaches it; and the overflow run appends **poison padding** past the undersized stack, because
counting filled slots cannot see a write that lands beyond the end.

The overflow case is its own assertion: `pushOntoStackFullWarp` reserves with the atomic *before*
testing capacity, so the pair count must report the true total (4) while at most `stackSize` entries
are written. The engine depends on exactly that to size the stack for the next frame.

Suite **674/674 in both memory modes**, rigid regression **16/16**, all six deformable scenes green.

## P8.2 tranche b — softbody-vs-rigid contact GENERATION, plane path

`sb_primitiveContactGenLaunch` (softbodyPrimitives.cu:573, launch PxgNarrowphaseCore.cpp:4287)
turns the midphase's `(cmIdx, tetIdx)` pairs into contact points. ABI row
`{4,8,8,8,8,8,8,8,8,8,4,8,4,8,80}`; manifest **MAPPED 224 → 225**; softbody **14 → 15 of 56**.

**This port implements the PLANE path only, and that is a deliberate, documented decision.**
The CUDA kernel branches on the rigid's geometry type: `ePLANE` goes through `tetPlaneCollide`
(four lanes testing the tet's four corners, no GJK), while `eSPHERE` / `eCAPSULE` / `eBOX` /
`eCONVEXMESH` go through GJK + EPA over `squawk::EpaScratch` — a much larger body of machinery
and its own tranche. The non-plane branch here emits nothing and returns.

That is **not a regression** — until now the whole kernel was unmapped, so every rigid type
produced zero softbody contacts — but it **is** silent from the engine's point of view: a box
ground still generates no contacts. The test pins that boundary explicitly (see below) so the
GJK tranche has to change it consciously.

**Correction to an earlier claim.** This tranche does *not* make the softbody cube settle. I
said it would; that was wrong. Contact generation is one of seven kernels — the contacts still
need `sb_other_contact_remap_to_sim`, the reorder, `sb_rigidContactPrepare`, the reference-count
query and the collision solve before anything moves. The plane path shortens the remaining road;
it does not end it.

### Two things worth recording

**The push blob is 192 bytes.** `PxgFEMContactWriter` is passed BY VALUE as the last CUDA
argument, so it is inlined into the push constant at @112 (nine pointers + a `PxU32`).
`paramAlign(80)` returns 16, which happens to land it at @112 either way. llvmpipe reports
`maxPushConstantsSize = 256` so it fits — but **Vulkan only guarantees 128**, making this the
first kernel in the port that would fail to load on a device at the floor. Moving the writer to
a descriptor set is the fix if that ever matters.

**A guarded load in a zero-trip loop still faults.** `findRigidPair`'s binary search cannot
execute its body when `nbPairs == 0` (`l == r == 0`), so by the language's semantics the pointer
is never dereferenced. The backend **speculates** the load anyway, and with a null filter-pair
address that SIGSEGVs. Found by bisection, not by guessing: a probe placed just before the call
survives, one placed just after it faults. The fix is an explicit `numSoftBodyRigidPairs > 0`
guard that the source language says is redundant. Same family as the `~3UL` trap in
`softBodySolveBda` — and the mutation sweep confirms it, since removing the guard reproduces the
SIGSEGV on demand.

### Verification

`softBodySimPrimitiveContactGenPlane` drives two contact-manager pairs over one tet straddling
the plane. The corners split 3/1 — three within the 0.10 contact distance, one 5.0 above:

```
contact 0: point(0.0000, -0.3000, 0.0000) rest=0.0200 normal(0, 1, 0) pen=-0.3000
contact 1: point(1.0000, -0.1000, 0.0000) rest=0.0200 normal(0, 1, 0) pen=-0.1000
contact 2: point(0.0000,  0.0500, 1.0000) rest=0.0200 normal(0, 1, 0) pen= 0.0500
```

Each contact's point is the corner itself, the penetration is its signed plane distance, and the
normal is the plane's world `+y`. The `static_assert`s on the 192-byte blob are the guard on the
inlined writer — one wrong field offset and every contact lands in the wrong buffer.

**11 mutants, 11 caught**: wrong plane axis (`.y` instead of `.x`), contact distance dropped,
`transformInv` reduced to a translation, penetration sign flipped, the material index written
without clearing bit 31, the sort identity seed replaced by 0, `stackSizeNeeded` in pairs instead
of bytes, the plane gate removed, the filter-pair guard removed (SIGSEGV), `restDistance[0]`
instead of `restDistance[cmIdx]`, and the shape swap dropped.

Three of those needed the fixture sharpened first: a **BOX pair** was added so removing the plane
gate has an observable effect; the plane pair was moved to **cmIdx 1** so `restDistance[cmIdx]`
can be told from `restDistance[0]`; and both are now asserted directly.

Suite **674/674 in both memory modes**, rigid regression **16/16**, all six deformable scenes
green.

## P8.24 — `pxgSlangTriBox`: the SAT extracted, with the extraction PROVEN inert

The second half of `midphaseGeneratePairs` is a BV32 traversal whose per-triangle
test is `Gu::intersectTriangleBox_RefImpl<true>` — the Akenine-Moller
separating-axis test, thirteen axes.  Before writing the BDA midphase this
tranche checks what already exists, and the answer is: all of it.  Phase 1
transcribed the whole SAT into the harness shader `npMidphaseCvx.slang`, where it
has a passing fixture.

So this is an extraction, not a port: `pxgSlangTriBox.slang` now owns
`tbPlaneBoxOverlap` and `tbIntersectTriangleBox`, and `npMidphaseCvx` imports
them.  The same move P8.15 made for the cloth midphase and P8.6 for the GJK/EPA
drivers, taken for the reason this port has already paid for once — a mutant that
deleted an OUTER gate survived because a DUPLICATED inner copy still culled.  One
copy, one place to mutate.

### The extraction is proven inert, not assumed inert

A refactor that "still passes the tests" is a weaker claim than this project
should accept for geometry, so the `.spv` was compared directly across the move:

```
before extraction: 46122dc9b0ed26d9e31d64f9a70df7df1113646a09a22d7f3642bb2a36670186
after  extraction: 6048f41024059957492d0aa8b4cdb022fd5042e943c07d325b4f6b43b62131b2
```

Different — which proves nothing on its own, since moving code into a module
renumbers every SPIR-V id.  Disassembling both and comparing the OPCODE STREAM
does settle it: the histogram is identical, and so is the full **1562-instruction
opcode SEQUENCE**, position for position.  The two binaries differ only in id
numbering.  Together with the `gpuslang.npMidphaseCvx` fixture staying green,
that is a behaviour-preservation argument rather than an absence of evidence.

### Two details recorded in the module, because only literal transcription keeps them

- **`planeBoxOverlap` is asymmetric.**  The near-corner test rejects on `> 0.0f`
  and the far-corner test accepts on `>= 0.0f`
  (`GuIntersectionTriangleBoxRef.h:99-104`).  A triangle plane exactly touching
  the far corner counts as an overlap; one exactly touching the near corner does
  not.  Making the two agree in either direction changes the emitted triangle set
  on any exactly-tangent plane.
- **The three vertex-in-box checks are an early-out for the whole conjunction**,
  not a fourteenth axis.  They can only fire where the conjunction would also
  succeed, so they change no result — but they change which float expressions are
  evaluated, and under NaN input they are the only path that returns true without
  touching an axis test.

### A contraction asymmetry that is deliberate, and worth knowing before it surprises someone

Every AXISTEST is literally `p = a*v.y - b*v.z` compared against
`rad = fa*extents.y + fb*extents.z` — mul feeding add, exactly the shape
contraction moves.  `npMidphaseCvx`, the Phase-1 harness consumer, is **not** in
`PXG_NOCONTRACT_SHADERS`.  The BDA consumer that follows **will** be, as every
other mesh midphase in this port is.  So the two will compile this one source to
different arithmetic on near-tangent triangles.  That is correct rather than an
oversight — the stamp is a property of the `.spv`, not of the module, and it is
the engine-path shader whose float behaviour has to match nvcc's contracted
reference — but it does mean the harness fixture's expectations do not transfer
unchanged to the BDA shader, and the BDA shader will need its own.

### Verification

Forced-fresh `.spv`, suite through `tools/run_suite.sh` — **676/676 in both
memory modes** (unchanged: an extraction adds no fixtures).  Scenes spot-checked
green (`box`, `convex`, `plane`, `clothdrape`, `softbodystack`).  A pre-existing
slangc warning at `npMidphaseCvx.slang:141` (`if (currentNodeIndex & 1)`, implicit
uint-to-bool) sits in untouched traversal code and is not from this change.

## P8.23 — the PCM manifold GATE, extracted and certified before the traversal

`midphaseGeneratePairs` splits in two at `fullContactGen = invalidate ||
lostContacts` (`convexMeshMidphase.cu:263`).  P8.22 established that the SMALLER
half — the gate — is the risky one, because it is the only stateful part of the
kernel.  This tranche ports that half on its own, the same move P8.6 made for the
GJK and EPA drivers and P8.15 for the cloth midphase.

New module `pxgSlangPcmManifold.slang` exports `mmInvalidateManifold` and
`mmRefreshSlot`; `pcmManifoldBda.slang` drives them as a standalone dispatch.
**Neither is in the manifest** — this is not an engine kernel yet, and
`convexTrimeshMidphaseBda` will import the module and inline the same sequence.
Every scene number is therefore unchanged, which is the expected result and was
confirmed rather than assumed (`softbody` box `|dy|` 0.000929, identical to P8.20).

**What stays in the caller, and why.**  `refreshManifolds` ends with a warp-wide
`__any_sync` and a lane-0 `mNbManifolds = 0`.  The module returns the PER-LANE
`lostContact` and leaves both to the consumer: the group-wide OR is the caller's
groupshared emulation of a ballot (this port has no subgroup-native path), and
keeping the barrier in the caller keeps it in provably UNIFORM control flow.  The
CUDA has the same property for the same reason — its `__any_sync` sits OUTSIDE
`if (curManifoldIdx < numManifolds)` so every lane reaches it.  A
`GroupMemoryBarrierWithGroupSync` in divergent flow does not error on llvmpipe,
it hangs.

### The fixtures, and the four times they were the thing that was wrong

**21 mutants, 21 killed.**  Getting there took four rounds, and every survivor
turned out to be a hole in the TEST rather than an equivalent mutant.  Recording
them because each is a trap the next fixture in this project can fall into:

- *no boundary case.*  `deltaQ <= thresholdQ` survived because no record had a
  quaternion dot of exactly 0.9998.  Since `aToB.q` is identity in that fixture,
  `dot(aToB.q, prevQ)` is exactly `prevQ.w`, so the boundary is constructible
  by writing 0.9998 straight into the stored quaternion.
- *an accidentally-signed input.*  Dropping `abs()` from the position delta
  survived because every stored transform happened to sit BEHIND `aToB`, making
  each delta positive.  Fixed by a record whose stored position is AHEAD, so the
  delta is −0.06 and only its magnitude crosses.
- *a degenerate rotation.*  Dropping the `q.w * t` term from `quatRotate`
  survived because every dispatch used an identity quaternion, whose vector part
  is zero — so `t` is zero and the term vanishes.  The multi-frame fixture now
  rotates by a tenth of a degree: enough to make the term load-bearing, small
  enough that the records still must survive all six frames.
- *a builder that measured the wrong thing.*  Reducing the group-wide OR to lane
  0 survived because the fixture's `pointB` was placed at `pointA.y +
  penetration`, which leaves a residual of `2*penetration` in the projected
  difference — large enough that EVERY slot was already "lost", lane 0 included.
  The correct placement is `pointA - normal*pen + (drift,0,0)`, which makes the
  measured difference exactly the drift and nothing else.  With that fixed, a
  record that drifts on slot 7 ALONE distinguishes a real reduction from lane 0.

Fixing the last one turned `projectedPoint` using `penNew` instead of `penOld`
into a genuinely EQUIVALENT mutant: under an identity transform `penNew ==
penOld` exactly, so the two lines are the same expression.  They are only
distinguishable when the transform moves the point along its own normal, so the
frame fixture now runs a SECOND dispatch translating 0.03 along the contact
normal — with `penOld` the projected difference carries that 0.03 and breaks the
0.02 threshold, with `penNew` it cancels and does not.  It is a second dispatch
rather than a change to the first because the first has a job the new transform
would destroy: a normal-direction move loses every slot, which would let the
lane-0 mutant pass again.

### The multi-frame fixture

`pcmManifoldRefreshMulti` dispatches SIX times over an evolving transform with a
CPU model carried forward in lockstep, so each frame's input is the previous
frame's output.  It exists because of a property stated in the module header
before any code was written: **a single-dispatch golden-vector fixture cannot
detect a bug in this gate.**  A port that always reported "regenerate
everything" yields correct contacts on frame one and in every isolated fixture,
while having silently destroyed PCM caching.  Mutant M20 is exactly that port —
`mNbManifolds` zeroed unconditionally — and it is killed only by the assertion
that the surviving records still hold their manifold count after six frames.

### Verification

Forced-fresh `.spv`, suite through `tools/run_suite.sh` — **676/676 in both
memory modes** (675 → 676: the new `gpuslang.pcmManifold` suite).  Scenes spot-
checked unchanged (`box`, `plane`, `softbodystack`, `clothselfcollide` green;
`softbody` box `|dy|` 0.000929, bit-identical to P8.20), which is the expected
outcome for a shader no manifest entry points at.

## P8.22 — `PXG_SCENE=meshfloor`: the assertion that was missing, and the convex-trimesh recon

P8.21 concluded that a dynamic rigid cannot collide with a cooked triangle mesh
and that no scene said so.  This tranche makes the scene say so, and does the
reading the port needs.  **No shader, manifest or ABI changed here.**

### The red control

`PXG_SCENE=meshfloor` is `PXG_SCENE=plane` with the floor swapped for the SAME
20 x 20 two-triangle cooked quad that `PXG_SB_FLOOR=mesh` gives the softbody, so
the two settle numbers are directly comparable.  A half-0.5 box dropped from
y = 5 must rest at centre y = 0.5.  Measured today:

```
  [boxOnMesh] start y=5.000, y after 1 step=4.9973, final y=-73.8070 (rest 0.500), |v|=39.2401
FAIL: [boxOnMesh] not at rest (|v|=39.240)
FAIL: [boxOnMesh] below rest band / sank (finalY=-73.807)
```

with all six `convexTrimesh` kernels failing to launch.  The box does not
"settle badly" — it passes through the floor and is still accelerating.  **This
scene is EXPECTED RED until the chain lands**, and it is deliberately not in the
rigid regression list yet.  It costs nothing to keep red: `gpu-sdk-step` scenes
are driven by hand, not by `ctest`, so the 675-test suite is unaffected.

### Recon for the port, and one claim that needed checking

A read-only survey of `midphaseGeneratePairs` produced four things worth
committing before any code is written.

**1. The traversal stack depth is NOT one number, and the deformable ports are
correct.**  A review flagged that the shipped BDA midphases use `STACK_SIZE =
192` while `convexNpCommon.h:171` declares `int sBv32Nodes[320]; //10 depth`,
i.e. that P8.10 and P8.17 might be under PhysX's own bound.  Checked: they are
not.  There are two scratch structs.  The DEFORMABLE path's
`femMidphaseScratch` (`gpucommon/src/CUDA/femMidphaseScratch.cuh:57`) declares
`int sBv32Nodes[192]; // 6 depth of the bv32 tree` — which is exactly what
`pxgSlangClothMidphase`, `softBodyMeshMidphaseBda` and `femClothMeshMidphaseBda`
mirror, comment and all.  The 320 belongs to the CONVEX path's `MidphaseScratch`
alone.  **The convex port must use 320; copying 192 from the deformable template
would silently corrupt traversal on a deep mesh.**

**2. Half of `midphaseCore` is not the traversal, and that half is the risk.**
The kernel splits at `fullContactGen = invalidate || lostContacts`
(`convexMeshMidphase.cu:263`).  Only if that is true does the BV32 traversal run.
The gate itself is `invalidateManifold` + `refreshManifolds` (`manifold.cuh`),
and it is the ONLY stateful part of the kernel: `manifold.cuh:84` writes
penetration back into a persistent 1216-byte record and `:97` destructively
zeroes `mNbManifolds`.  Everything else is a pure function of this frame.

The consequence for testing is sharp and worth stating before it bites: **a
single-dispatch golden-vector fixture cannot detect a bug in the gate.**  A port
that is conservative (always `fullContactGen = true`) yields CORRECT contacts on
frame 1 and in every isolated fixture, while having destroyed PCM caching — the
failure appears as solver warm-start drift many frames later.  The gate needs a
multi-frame test, not a one-shot one.

**3. The gate also carries the only lane-count hazard with a state consequence.**
`manifold.cuh:52` is `PxU32 i = threadIdxInWarp;` with NO loop, covering 24 slots
(`MULTIMANIFOLD_MAX_MANIFOLDS 4 x SUBMANIFOLD_MAX_CONTACTS 6`).  It assumes
`WARP_SIZE >= 24`.  Under this port's sanctioned mapping — 32-lane virtual warps
over groupshared scans, `[numthreads(32,1,1)]`, four subgroups of 8 per group —
that holds.  It only breaks if someone "optimises" the group to 8 wide, and then
it does not merely drop work: 16 of 24 cached contacts keep last frame's
penetration while 8 are current.

**4. Two BDA alignment traps, both from odd-sized structs.**  `PxBounds3` is 24
bytes, so `bounds[i]` sits at byte `24*i` — 8-mod-16 for odd `i`, which makes a
`Ptr<float4>` read MISALIGNED.  `ConvexMeshPair` is 56 bytes with the same
problem at `cvxTrimeshPair[cmIdx]`, which is precisely why upstream writes it as
a 14 x 4-byte scatter (`dataReadWriteHelper.cuh:384`).  Both must be read and
written as scalars.

Also recorded for the port: `cmOutputs` (argument 10 of 15) is **never touched**
by `midphaseCore` — it is passed and ignored; the two launch sites differ only in
which bucket they serve (sphere/capsule-vs-mesh at `:1950`, box/convex-vs-mesh at
`:2554`) and in one memset; and a BV32 leaf's `nbPrimitives` field is 6 bits (up
to 63) while `bv32Traversal.cuh` maps one lane per primitive and silently drops
past 32 — so the port must clamp to 32 bug-for-bug rather than write the obvious
loop.

### Verification

`meshfloor` fails as designed and is documented as such.  Every other scene
unchanged; the suite is untouched (675/675) because no shader or manifest moved.

## P8.21 scouting — the failed-launch inventory: two failure modes, and what is actually missing

P8.20 found three kernels that had been refused at every launch for the whole
softbody arc without anyone noticing.  That is a failure CLASS, not an incident,
so before picking another kernel by intuition this tranche measures it across
every scene.

### There are TWO failure modes, and only one of them is traced

- **REFUSED** — the kernel IS in the manifest with valid SPIR-V, but has no ABI
  row, so `pushConstantSizeFor()` returns 0 and the shim rejects the blob.
  `PXG_LAUNCH_TRACE=1` prints `[pxg-launch] ... REFUSED` for these.
- **No `CUfunction` at all** — the manifest entry is `nullptr` (unmapped).
  `fromFunc()` fails and `PxgVulkanCudaContext::launchKernel` returns before
  `traceDispatch()` ever runs, so these produce **NO trace line**.  The only
  evidence is the engine's own `PxGetFoundation().error(... "fail to launch
  kernel!!")` on stderr.

A sweep that greps only the trace therefore misses every unmapped kernel, which
is exactly how the `convexTrimesh` chain below stayed invisible.  Both sweeps:

```
PXG_LAUNCH_TRACE=1 ... | grep REFUSED                      # mapped, no ABI row
                    ... | grep -oE "GPU [A-Za-z0-9_]+ fail to launch kernel"   # unmapped
```

### Mode 1 (REFUSED), all scenes: clean

Across box / stack / plane / sphere / capsule / capsulebox / convex / joint /
particle / cloth / clothhang / clothdrape / clothselfcollide / softbody (all
floors) / softbodyselfcollide, the only refusals left after P8.20 are
`removeContactManagers_Stage5` (1–3 per run) and `initializeManifolds` (1 per
run).  Both fire in setup/teardown, not per step.  Nothing per-step is refused
anywhere.

### Mode 2 (unmapped), and the two things it found

| scene | kernel | per run |
|---|---|---|
| `clothdrape` + `PXG_CLOTH_FLOOR=mesh` | `cloth_midphaseVertexMeshLaunch` | 174 |
| `clothdrape` + `PXG_CLOTH_FLOOR=mesh` | `cloth_meshVertexContactGenLaunch` | 174 |
| `softbody` + `PXG_SB_FLOOR=mesh`, `clothdrape` + mesh | `midphaseGeneratePairs` (`convexTrimeshMidphase`) | 13 / 10 |
| " | `convexTrimeshNarrowphase`, `sortTriangleIndices`, `convexTrimeshPostProcess`, `convexTrimeshCorrelate` | 13 / 10 |

**Finding A — the cloth VERTEX-vs-mesh pair is launched every step, and its
contact writer is DISABLED UPSTREAM.**  P8.14's survey called the four cloth
vertex-vs-mesh/HF kernels "dead".  That verdict is right in its effect and wrong
in its reasoning: they are launched 174 times per run, once per step, not never.
But `cloth_meshVertexContactGenLaunch` (`femClothPrimitives.cu:2910`) declares
`bool hasContacts = false;` and its ONLY assignment is commented out in
upstream's own source:

```cpp
    if ((pen <= cDistance) && normal.dot(n) > 0.f)
    {
        pairInd0 = rigidId.getInd();
        pairInd1 = PxEncodeClothIndex(clothId, clothVertIndex);
        contact = make_float4(closestPt.x, closestPt.y, closestPt.z, restDistance[cmIdx]);
        //hasContacts = true;                      // <-- upstream, femClothPrimitives.cu:3056
    }
```

so the `__ballot_sync` is always empty and `writeRigidVsDeformableContact32` is
never reached.  The sibling at `:2737` is commented out identically.  The kernel
computes a full closest-point-on-triangle query and throws the answer away; its
only surviving effect is the `atomicMax` on `stackSizeNeededOnDevice`.

Porting it line-for-line — which is this project's rule — would faithfully
reproduce a no-op.  **These two kernels are removed from the queue with evidence
rather than by verdict**, and that is also why `PXG_CLOTH_FLOOR=mesh` settles
correctly (P8.18, min y 0.0496) while both halves of the pair fail to launch:
they would have contributed nothing either way.

**Finding B — the real gap is the rigid convex-vs-trimesh narrowphase**, six
unmapped kernels that fire in BOTH mesh-floor scenes:

| manifest name | id | CUDA |
|---|---|---|
| `midphaseGeneratePairs` | `CONVEX_TRIMESH_MIDPHASE` | `convexMeshMidphase.cu:378` |
| `convexTrimeshNarrowphase` | `CONVEX_TRIMESH_CORE` | |
| `sortTriangleIndices` | `CONVEX_TRIMESH_SORT_TRIANGLES` | |
| `convexTrimeshPostProcess` | `CONVEX_TRIMESH_POST_PROCESS` | |
| `convexTrimeshCorrelate` | `CONVEX_TRIMESH_CORRELATE` | |
| `convexTrimeshFinishContacts` | `CONVEX_TRIMESH_FINISHCONTACTS` | |

This is not a deformable gap — it is the pair that lets ANY dynamic rigid body
collide with a cooked triangle mesh, and it has been missing the whole time.  The
mesh-floor scenes carry a dynamic rigid box (the one the PGS island needs) which
currently falls straight through the mesh floor; no scene asserts on it, which is
why it never surfaced.  Phase 1 already ported the underlying math into harness
shaders (`npConvexTri`, `npCvxTriEmit`, `npCvxTriFind`, `npCvxTriPurge`,
`npCvxTriWorldNormal`, `npCvxTriNpSetup`, `npMidphaseCvx`), so the work is the
usual lift onto the BDA engine ABI rather than a fresh port of the geometry.

Step 1 is `midphaseGeneratePairs`: pack
`{4,4,8,8,8,8,8,8,8,8,8,8,8,8,4}` = 15 args, 112-byte blob (well inside the
256-byte push-constant limit), launched at `PxgNarrowphaseCore.cpp:1950` and
`:2554` with grid `(numBlocks,1,1)` block `(32, MIDPHASE_WARPS_PER_BLOCK, 1)`.
`midphaseCore` (`convexMeshMidphase.cu:167`) is ~200 lines of warp code over a
BV32 traversal, comparable in size to the softbody mesh midphase of P8.10.

### Verification

Measurement only — no shader, manifest, ABI or scene file changed in this commit.

## P8.20 — the softbody settle failure: three REFUSED launches, and an undamped material

This tranche started as a bisect and ended as two unrelated findings, neither of
which was the one P8.19 predicted.  Both are recorded here with the measurements
that decided them, including the places where my own earlier notes were wrong.

### What P8.19 handed over, and why the bisect found nothing

P8.19 recorded `PXG_SCENE=softbody` failing its settle check and named a bisect as
the next step, on the strength of two claims.  **Both were wrong**, and correcting
them is the first result of this tranche:

- *"the number is identical across box, plane, sphere and mesh"* — it is not.
  Re-measured: box **0.015430**, plane **0.012998**, sphere **0.014072**, mesh
  **0.011728**.  The P8.19 note generalised from a single box run.
- *"the last recorded green run predates the cloth arc, so the window is wide"* —
  there is no window.  The failure reproduces at **P8.11 (`81e1a5b`)**, at
  **P8.7 (`4bf5b57`)** — the commit whose message is "the softbody settles on the
  ground box" — and at **P8.2 tranche e (`24a4b94`)**, the commit that recorded
  `rest |dy| = 0.000002`, all with the SAME numbers as HEAD, byte for byte.

Each of those probes wiped every `.spv` first: checking out an older commit does
NOT retrigger an importer's SPIR-V when only a module changed, and the first pass
of this bisect was invalid for exactly that reason (three commits apart, three
identical numbers, because all three ran HEAD's shaders).  The forced-fresh rerun
gave the same answer, so the conclusion stands on its own.

Ruled out along the way, each by measurement rather than argument: run-to-run
nondeterminism (three runs, bit-identical), llvmpipe thread count
(`LP_NUM_THREADS` 1/2/4/8, bit-identical — and the scene drives PhysX through a
single-threaded `SyncDispatcher` anyway), and toolchain drift (slangc `2026.12.2`,
Mesa `25.2.8-0ubuntu0.24.04.2`, LLVM `20.1.2` — the exact recorded profile).

**So the numbers in this document for `PXG_SCENE=softbody` — plane `0.000002`, box
`0.000899`, sphere `0.000575` — are not reproducible at their own commits.**  I
cannot explain how they were produced and am not going to invent a mechanism; what
follows is measured on the tree as it stands.

### Finding 1 — three kernels were REFUSED at every launch, silently

`PXG_LAUNCH_TRACE=1` on the softbody scene showed three kernels refused on every
step, with `PxgSoftBodyCore` logging `fail to launch kernel!!` and stepping on:

| kernel | launched from | CUDA |
|---|---|---|
| `clampMaxValues` | `PxgSoftBodyCore.cpp:474`, `PxgFEMClothCore.cpp:826` | `MemCopyBalanced.cu:95` |
| `radixSortCopyHigh32Bits` | `PxgSoftBodyCore.cpp:559`/`:645`, `PxgFEMClothCore.cpp:935`/`:1019` | `radixSortImpl.cu:119` |
| `radixSortCopy` | `PxgSoftBodyCore.cpp:614`, `PxgFEMClothCore.cpp:991`/`:1078` | `radixSortImpl.cu:163` |

All three were in the manifest with valid SPIR-V — pointing at the **Phase-1
resource-binding harness** (`clampValues.slang`, `radixSort64.slang`,
`radixSortGather.slang`), which takes bound `RWStructuredBuffer`s plus indices in
the push constant.  The engine hands them the CUDA pack of bare device pointers.
No ABI row can describe the harness form, so `pushConstantSizeFor()` returned 0
and the shim refused every launch.  The manifest counted them as mapped, because
by its own definition — "has a `spirvFile`" — they were: `PXG_SLANG_KERNEL_MAPPED`
stays at **258** here, and that number has been over-claiming these three all
along.

New shader `femContactSortBda.slang` ports all three line-for-line onto the BDA
ABI; the manifest is re-pointed at it (hand-edited, as always) and three ABI rows
land: `kRadixCopy` `{8,8,8,8}` shared by the two gathers, `kClampMaxValues`
`{8,8,8,4}`.  After the fix the trace shows **exactly one** refused kernel left in
the scene — `removeContactManagers_Stage5`, three times over the whole run, in
teardown.

**Honesty about impact: no scene number moves.**  Measured before and after with a
forced-fresh build, `clothselfcollide`, `clothcloth`, `clothdrape` and
`softbodycloth` are character-identical, and the softbody floors are bit-identical.
Every scene in this harness has at most one rigid actor generating FEM contacts, so
the by-rigid sort these kernels serve is trivial and a stale rank array gathers the
same answer as a correct one.  The evidence that the port is right is therefore the
fixture suite, not a scene: `testFemContactSort.cpp`, three fixtures, each compared
BIT-EXACT against a CPU transcription of the `.cu`.

**Mutation sweep: 21 mutants, 18 killed, 3 provably equivalent.**  The three
survivors are equivalent for stated reasons, and each reason is backed by a
DISCRIMINATING sibling mutant that dies:
- M4 / M10 (grid stride loses its `WorkgroupCount()` factor) survive because a
  *shrunk* stride still covers every index with the correct value — the extra
  writes are redundant and identical.  M19 / M20 *double* the stride instead,
  which tears real coverage gaps, and both die.
- M14 (the clamp's `value1` and `value2` pointers swapped) survives because all
  three slots receive the identical operation, so permuting them is a no-op.  M21
  writes `value2`'s result into `value1`'s buffer — asymmetric — and dies.
- M18 (`>` becomes `>=`) differs only when the counter already equals `maxValue`,
  and then the store writes back the value already there.

The first sweep reported four extra survivors and they were a FIXTURE defect, not
equivalence: at `numKeys = 700` against a 32 x 1024 = 32768-thread grid the
grid-stride loop never takes a second turn, so the stride was dead code the test
could not see.  The fixtures now run `numKeys = 35000` over a 40000 capacity, which
loops twice and leaves a sentinel tail.

### Finding 2 — the settle failure was an UNDAMPED MATERIAL, not an unconverged solve

With every kernel launching, all four floors still failed, bit-identically.  The
question the check actually asks is whether the solve CONVERGES, so the next
measurement was whether the oscillation grows or decays.  Tracing `PXG_SB_TRACE=1`
over the box floor, the mean-y span per 35-step window:

```
steps  55- 90: mean y span 0.5774 | sim y-extent span 0.2570
steps  90-125: mean y span 0.4796 | sim y-extent span 0.4314
steps 125-160: mean y span 0.2280 | sim y-extent span 0.3865
steps 160-200: mean y span 0.0582 | sim y-extent span 0.1064
```

Monotonically decaying.  The solve converges; it is simply nowhere near converged
after 200 steps.  The cause is in the scene, not the port: `buildSoftBodyScene`
created its material with

```cpp
/*elasticityDamping*/ stretchY > 1.0f ? 0.05f : 0.0f
```

so the damping was gated on the *stretch* knob and every floor case ran with
**zero constitutive damping**.  A cube dropped 3.5 m onto a hard contact with no
damping has only the contact solve and the implicit integrator to absorb the
impact energy, and 200 steps of that does not reach a 2e-3 bar.  The scene's own
settle assertion was unreachable by construction.  One line confirms the
mechanism: `PXG_SB_STRETCH=1.02` (which switches the damping on for an unrelated
reason) settles at `|dy| = 0.000813` on the same floor that fails at `0.015430`
without it.

The fix gives the damping to every case that has a floor —
`(stretchY > 1.0f) || !floorNone` — and leaves `PXG_SB_FLOOR=none` undamped, which
keeps the UNDAMPED branch of the tranche-6 solve covered and keeps the only case
that still asserts the closed-form free fall on exactly the material it always had.

**The two-tier `|dy|` bar is retired.**  It held the plane to 1e-4 and the GJK
floors to 2e-3 on the theory that contact density predicts convergence: the plane
branch writes one contact per penetrating corner (32 on this cube), the GJK/EPA
branch one witness pair per query (16).  Measured on a damped material the ordering
does not hold — sphere **0.000102** < plane **0.000153** < mesh **0.000769** < box
**0.000929** — so the split is removed rather than re-tuned to a number it would
only accidentally fit.  One bar, 2e-3, and all four numbers stay printed on every
run so a convergence that degrades within the bar is still visible.

The bar keeps its teeth: the worst passing floor sits 2.2x under it and the
undamped control sits 7.7x over it, a factor of ~17 between the passing and
failing states.

### The softbody floor matrix, re-measured

| floor | contacts | settled mean y | rest \|dy\| (bar 2e-3) |
|---|---|---|---|
| box (default) | 16 | 0.48955 | 0.000929 |
| plane | 32 | 0.48951 | 0.000153 |
| sphere | 16 | 0.47777 | 0.000102 |
| mesh | 22 | 0.48792 | 0.000769 |
| none (free fall) | 0 | −0.9867 | closed form 0.0006% off |

Every settled mean y is now BELOW 0.5 — the cube is compressed under its own
weight, which is what a resting elastic solid should do and what the undamped runs
never showed (they oscillated about a *stretched* state, y-extent 1.04–1.10 against
a rest length of 1.0).

### A note on determinism, since this tranche made the sort live

`softbodystack` and `softbodyattach` vary run to run (`|dy|` 0.000476–0.000605;
attach min y −0.1741/−0.1830).  Checked against a stashed working tree: they varied
at HEAD too (0.000579–0.000629), so this is the pre-existing atomic-append contact
ordering, not something P8.20 introduced.  `PXG_SCENE=softbody` on the box floor is
bit-identical across three runs.

### Verification

Forced-fresh `.spv`, suite through `tools/run_suite.sh` — **675/675 in both memory
modes** (674 → 675: the new `gpuslang.femContactSort` suite).  Rigid regression
green (box / sphere / capsule / capsulebox / convex / plane / stack / joint, plus
`particle`).  Every deformable scene green: `clothselfcollide` PGS and TGS,
`clothcloth`, `clothdrape` including `PXG_CLOTH_FLOOR=mesh`, `softbodyselfcollide`,
`softbodycloth`, `softbodystack`, `softbodyattach`, and all five `PXG_SCENE=softbody`
floor modes — the last of which had not been green in a reproducible way at any
commit this tranche could find.

## P8.19 — FEM-vs-FEM attachments: one soft volume hanging off another

`femAttachmentPrepareLaunch` + `sb_solveOutputSoftBodyAttachmentDeltaVLaunch` on the engine BDA
ABI. **Manifest 256 → 258.** New shader `softBodyFemAttachBda.slang`.

This is the last piece that lets one FEM volume hang from another with **no contact in the loop**.
It is the P8.3 rigid-attachment pair with a second deformable in place of the rigid, which makes
it SIMPLER: no velocity reader, no articulation branch, no rigid response term. Both sides are
position-based, and the whole constraint is

```
wTot0 = Σ bary0[i]² · invMass0[i]        wTot1 = Σ bary1[i]² · invMass1[i]
delta = (pos1 − pos0) / (wTot0 + wTot1)
softbody1 −= delta · b1w                 softbody0 += delta · b0w
```

**Neither kernel has a cross-lane operation**, so unlike every midphase port this one keeps the
CUDA `[numthreads(256,1,1)]` block verbatim and transcribes the strided indexing unchanged.

**The two struct layouts are the whole risk** and are named-offset constants in the shader with
`static_assert`s in the fixture. Two details are easy to get wrong and both have a mutant:
`PxgFEMFEMAttachmentConstraint` is a WARP-STRIDED block (every field an array of 32, indexed by
`workIndex & 31`, block `workIndex / 32`) whose two leading fields are unions; and `elemId1` is a
PxU32 **widened** into a PxU64 slot by the prepare and **narrowed** back by the solve, so the port
clears the upper word on write and reads only the low word.

**21 mutants, 20 killed.** The first sweep killed only 17; the four survivors were all one fixture
weakness — barycentric `(1,0,0,0)` on both sides. With equal barycentrics, swapping
`barycentric0`/`barycentric1` is invisible, and with only corner 0 ever weighted, neither the
per-corner `invMassBary > 0` gate nor the barycentric factor in `b0w` does anything observable.
The fixture now uses `(0.5, 0.25, 0.25, 0)` against `(0.1, 0.2, 0.3, 0.4)` with inverse masses
`(1, 2, 0, 4)` and `(0.5, 0, 1.5, 2)`, so **two corners carry zero weight for two DIFFERENT
reasons** — one has `invMass == 0`, the other has `barycentric == 0` despite `invMass == 4` — and
each kills one of those mutants. (The fourth, dropping the `b0w` barycentric factor, dies with
them.)

**One mutant survives and is provably equivalent.** Removing the `wTot > 0.0f` gate: `wTot` is
`Σ bary[i] · (bary[i] · invMass[i])`, so for the clamped-non-negative barycentrics the CUDA
guarantees, `wTot == 0` forces every `b*w[i]` to zero — and `updateTetPositionDelta` writes
nothing when the weight is zero. The division by zero produces a NaN that never reaches memory.

### The scene, and the three arrangements that did not work

`PXG_SCENE=softbodyattach`. Getting a scene that actually ISOLATES the attachment took three
tries, and each failure is worth recording because each looked like a pass:

| arrangement | with the port | control (kernels unmapped) | why it proved nothing |
|---|---|---|---|
| stacked flush, gap 0.10 | 1.0554 | **1.2653** | within contact distance — the softbody-vs-softbody narrowphase (P8.9) catches it |
| stacked, gap 0.25 | 0.9682 | **1.2340** | no contact at spawn, but the cube simply FALLS the 0.25 and lands on the pinned one |
| offset 2.0 sideways | −42.59 | — | no contact and nothing to land on, but a 2.0 initial error: a position projection closes it in ONE step (mean 1.80 → 0.71 on step 0) and the material blows up by step 90 |

**Anything directly above the pinned cube has something to land on, so no vertical arrangement can
work at all.** The answer is SIDE BY SIDE at the same height with a 0.20 face gap: far outside the
default contact offset, nothing beneath the free cube, and an initial constraint error of a
fraction of a unit instead of two.

```
with the port:      upper volume mean y 0.3000 ->   0.2633  (min 0.5017 ... -0.1741)
kernels unmapped:   upper volume mean y 0.3000 -> -44.0902  (both checks red)
```

The attachment points are chosen at RUNTIME, not hard-coded: the cubes' collision tetrahedra come
out of the cooker with unpredictable indices, so the builder reads both meshes back, takes the
pinned cube's largest-x tetrahedra and the free cube's smallest-x, and pairs them by proximity in
the facing (yz) plane, anchoring both at their centroids.

**Cone limits are ported but dormant on every scene**: an attachment created without a
`PxConeLimitedConstraint` carries −1 in all three parameters and `isConeLimitedEnabled` is false.
The branch is transcribed anyway — including its `PxSinCos`, the only transcendental in this
shader — and the fixture drives it directly, with a non-zero `constraintOffset` so the term that
only a non-zero offset can expose is covered too.

**Verification:** forced-fresh `.spv`, suite through `tools/run_suite.sh` — **674/674 in both
memory modes** (`gpuslang.softBodySim` 58 → 60 fixtures); `softbodyattach` green with the control
red; `softbodyselfcollide`, `softbodycloth` and `softbodystack` green.

### A PRE-EXISTING failure this tranche did not cause, and did not fix

`PXG_SCENE=softbody` fails its settle check on **all four floors**:

```
FAIL: the cube was caught but never settled: |dy| still 0.015430 over the last 50 steps
```

It is not P8.19's doing, and that is measured rather than argued: the scene creates no
attachments, so `getNbSoftBodySoftBodyAttachments()` is zero and neither new kernel is ever
launched — and stashing the whole working tree and running the scene at **HEAD (574c5a5)**
reproduces the identical |dy| of 0.015430. The other four softbody scenes
(`softbodyselfcollide`, `softbodycloth`, `softbodystack`, `softbodyattach`) all pass.

Two facts worth carrying into the investigation: the number is **identical across box, plane,
sphere and mesh floors**, which points away from any one contact-generation branch and at the
shared settle path; and the check that fires is a CONVERGENCE bar (`|dy| < 2e-3` for the GJK
floors, `1e-4` for the plane) rather than a position bar — the cube IS caught, it just keeps
jittering. The last recorded green run of this scene predates the cloth arc, so the regression
window is wide and wants a bisect.

## P8.18 — the cloth-vs-mesh settle: not a port bug, an ACTIVATION WINDOW

**No kernel was ported in this tranche; the manifest is unchanged at 256.**  P8.17 step 2 handed
over a chain that produced correct contacts and a sheet that still fell through, with the
cloth-vs-rigid TRIANGLE solve branch as the named suspect.  That suspicion was **wrong**, and
chasing it to the bottom turned `PXG_CLOTH_FLOOR=mesh` from a diagnostic into a full settle test:

```
mesh floor, before:  mean y 1.5000 -> -42.8902   (free fall, contacts generated, zero impulse)
mesh floor, after:   mean y 0.6000 ->   0.0496   (settles ON its 0.05 rest offset)
kernels unmapped:    mean y 0.6000 -> -43.7902   (the control, red)
```

### What the measurement said

A probe on `PxgFEMClothCore` right after `cloth_solveRigidClothCollisionLaunch`:

```
[solve] n=44 lambdaN=(0.000000 0.000000 0.000000)
        bary0=(0.333 0.333 0.333 0.000)   norm0=(-0.000 -1.000 -0.000 | 0.04183)
        raXn0.w=0.00000  fric0.w=0.00000  deltaPos[0]=(0 0 0 | 0.0)
```

Forty-four contacts reach the solve with the right barycentrics and the right normal, and every
λN is zero — and so is every reference count.  `raXn0.w = 0` and `fric0.w = 0` are both CORRECT
for a static floor (a static rigid contributes no response term), so the constraint block is fine.

The answer is one line, `deformableUtils.cuh:354`:

```cpp
const PxReal error = (initPen + relLinDelta.dot(normal)) * invDt;
CN = PxMax(maxPenBias, error) + normalVel;
const bool isActive = wasActive || CN < 0.0f;
```

`normalVel` is the **RIGID's** normal velocity, which is zero for a static floor.  `relLinDelta`
is the cloth's accumulated delta, zero at the first solve iteration of a step.  So
`CN = initPen * invDt`, and with a **zero rest offset** `initPen` is the raw separation: positive
for as long as the sheet is above the floor.  **The contact is generated and never activated.**

It would activate the moment the sheet penetrated — except that a cooked triangle is **ONE-SIDED**.
The midphase's front-face cull (`triangle1.triPlane.distance(clothCentroid) >= 0`,
`femClothClothMidPhase.cu:138`) drops the pair the instant the cloth triangle's centroid crosses
the plane.  So the window in which the contact both EXISTS and is ACTIVE is **empty**, and the
sheet sails through with contacts being generated the whole way down.

### Why the plane floor never showed this

`clothdrape`'s original floor is an infinite `PxPlaneGeometry`.  It has no midphase and no
front-face cull: `cloth_planeVertContactGenLaunch` keeps emitting contacts with a growing
NEGATIVE separation as each vertex sinks, so CN goes negative on its own and the constraint
activates.  That is why the scene has always settled without a rest offset, and why nothing in
the port had ever needed one.  `clothselfcollide` and `clothcloth` both DO set one (0.05 per side)
— for the same reason, discovered the same way, and recorded in their own entries.

### The fix, and what it is not

Two lines in the scene, applied ONLY to the mesh variant so the plane baseline is untouched:

```cpp
if (floorIsMesh) { shape->setContactOffset(0.1f); shape->setRestOffset(0.05f); }
```

**This is a scene fix, not a shader fix.**  Nothing in `femClothMeshMidphaseBda`,
`femClothMeshContactGenBda` or `femClothRigidContactBda` changed.  The triangle branch of the
cloth-vs-rigid solve — the P8.17 suspect — turns out to have been correct all along; it simply had
never been given an active constraint to work on.

**The drop height is also a real effect, and a separate one.**  Even with the rest offset the
sheet still tunnels from 1.50: the mesh pair's midphase runs roughly every sixth step, and at
~5 m/s the sheet crosses the whole 0.1 contact band between two consecutive invocations.  From
0.60 — and every height tested below it, down to 0.03 — it settles cleanly, so the mesh variant
drops from 0.60 and the plane variant keeps its 1.50.  Ordinary tunnelling, called out as such in
the scene rather than tuned away silently.

### Where it landed

`PXG_SCENE=clothdrape PXG_CLOTH_FLOOR=mesh` is now a full settle test: it must fall more than
0.40, converge, and come to rest within 0.03 of `floorY + 0.05` without any vertex going below
`floorY - 0.02`.  Measured **0.0496 mean / 0.0478 min**.  With the two manifest rows reverted to
`nullptr` it reads **-43.7902** and fails all three — the same A/B control `clothcloth` uses.

The engine probes used here were reverted; **no engine source is modified**.  They are the same
two recorded at the end of the P8.17 step 2 entry, plus one on
`PxgFEMClothCore::solveRigidContacts` dumping `lambdaNs`, `mRigidConstraintBuf[0]` and
`mDeltaPos[0..1]`.

**Verification:** suite through `tools/run_suite.sh` — **674/674 in both memory modes**; `cloth`,
`clothdrape` (plane, 1.50 baseline untouched), `clothdrape` (mesh, new), `clothhang`, `clothcloth`
and `clothselfcollide` all green.

## P8.17 step 2 — cloth-vs-COOKED-TRIANGLE-MESH: the contact generation, and where the impulse is lost

`cloth_meshContactGenLaunch` on the engine BDA ABI. **Manifest 255 → 256.** New shader
`femClothMeshContactGenBda.slang`. The kernel is ported, fixture-verified and demonstrably
producing correct contacts on a live scene — and the sheet still falls through, for a reason that
turns out to sit **downstream of this tranche entirely**. That investigation is the more valuable
half of this entry, so it is written up in full below.

**The core is a LIFT, not a new transcription.** `fcStaticTriCloth.slang` already ports
`staticTriangleVsClothCollision` for the descriptor-set engine (its mode 0), with
`testFcStaticTriCloth.cpp` covering the trim pipeline's sign-change bookkeeping, the in-place
insert branch, the `±0.01·cDistance` averaging band and the degenerate paths against a CPU
transcription. The clipping code here is that port verbatim; what is new is the BDA plumbing,
which is exactly what `fcStaticTriCloth`'s own header lists as "engine glue not ported". Two
copies of the clipping pipeline is a knowing cost: the descriptor-set shader is a unit-test
harness with a case-array interface and cannot be imported by a push-constant kernel.

**The arg pack is NOT `sb_meshContactGenLaunch`'s**, even though both are 192 bytes and both end
in an 80-byte `PxgFEMContactWriter`. The middle differs: `stackSizeBytes` sits at **@64 BEFORE**
the two stack pointers here and at **@80 AFTER** them there. Reading one with the other's layout
hands the kernel a byte count where a device pointer belongs — which is the first mutant below,
and it segfaults.

Beyond the pack, four things differ from the softbody sibling: there is no GJK/EPA (this is a
polygon clip); the transform runs the OTHER way (step 1 brings the cloth into the mesh's vertex
space, this brings the mesh triangle into the cloth's local frame — `vertex2Shape` MULTIPLIES);
the filter skip needs **all three** cloth vertices; and the write is
`writeRigidVsDeformableContact32`, which fills the barycentrics and the 32-bit
`contactSortedByX` lane while deliberately leaving the u64 `contactByX`/`tempContactByX` keys
alone.

**17 mutants, 17 killed** — the softbody field order, inverted cloth/mesh resolution, the `.z`/`.w`
blob size, `vertex2Shape` dividing, a reversed `transformInv` composition, the filter skip needing
any vertex instead of all three, `restDistance[0]` instead of `[cmIdx]`, the high word in the sort
lane, a vertex id in `pairInd1`, the isInCollision bit left set, the sub-warp loop truncated, the
u64 keys written, the deepest-point tolerance band removed, a flipped `fixNormal`, the stack
high-water mark in pairs instead of bytes, an inclusive ballot scan, and `meshTriangleIdx` ignored.

The fixture puts 40 pairs on the stack so they **straddle a warp boundary**, with a contact in
each warp, so both per-warp reservations are exercised and their slots must come out in warp
order. Its geometry is derived FROM the transform chain — each query triangle is built from the
mesh triangle's own cloth-local image, offset along its normal — so a shader whose chain differs
computes a different plane and a different separation.

### The scene: contacts are generated, and the impulse is lost downstream

`PXG_SCENE=clothdrape PXG_CLOTH_FLOOR=mesh` still ends with the sheet at **−44.31**, and the
instrumentation says why it is not this tranche's fault. Both kernels dispatch (6 times each over
the run), and a temporary probe in `PxgNarrowphaseCore::testSDKFemClothTrimesh` plus one at
`PxgFEMClothCore::prepRigidContactConstraint` shows:

```
[probe] clothTrimesh numTests=1 midphasePairs=64 contacts=43
[prep]  rigidContacts=43  sorted[0] pt=(-0.667 0.037 -0.667 | 0.000)
                                    np=(0.000 1.000 0.000 | 0.0374)
                                    bary=(0.333 0.333 0.333 0.000)
```

Sixty-four midphase pairs, forty-three contacts, and they arrive at the prepare **already
reordered** with the right normal, the right barycentrics and sane separations. The contact path
works end to end.

**And the sheet free-falls with an IDENTICAL total drop (44.3902) whether it is spawned 1.5 above
the floor, 0.25 above it, or 0.03 above it — INSIDE the 0.1 contact band.** A drop that does not
change when the sheet starts already in contact is not tunnelling and not a cadence problem: the
constraints produce **zero impulse**.

The suspect is specific. `writeRigidVsDeformableContact32` emits a **TRIANGLE** contact —
`pairInd1` is an encoded triangle and `barycentric.w == 0` — and every consumer branches on
exactly that: `cloth_rigidContactPrepareLaunch` (`FEMClothConstraintPrep.cu:130`),
`cloth_queryRigidClothContactReferenceCountLaunch` and `cloth_solveRigidClothCollisionLaunch` all
take a different path for `barycentric.w == 0` than for a vertex contact. **No scene has ever
exercised that branch**: `clothdrape`'s plane floor, `clothhang` and `clothselfcollide` all
produce VERTEX contacts (`barycentric = (0,0,0,1)`). `femClothRigidContactBda` carries the
triangle branch textually — `readClothVertexInvMasses` reads `mTriangleVertexIndices` under
`bc.w == 0.0f` — so this is not a missing branch but an unverified one, and the mass-splitting it
does through `mDeltaPos.w` is the part with the most room to be wrong.

**Where the investigation stopped, and what the next step is.** The next tranche should probe
`mRigidFEMAppliedForcesBuf` (the per-contact λN) right after `cloth_solveRigidClothCollisionLaunch`
on this scene: a λN of zero puts the fault in the prepare or the query's reference counting, a
non-zero λN that never reaches `mDeltaPos` puts it in the solve's scatter. The two probes used
above are the tool — they were removed from the engine sources before committing, and are
reproduced here so they can be re-applied:

* in `testSDKFemClothTrimesh`, after the contact-gen launch, `memcpyDtoH` of
  `gpuMidphasePairsNumOnDevice` and `totalNumCountsd`;
* in `prepRigidContactConstraint`, before the launch, `memcpyDtoH` of `totalContactCountsd` and
  element 0 of `contactsd` / `normalpensd` / `barycentricsd` / `contactInfosd`.

`PXG_CLOTH_SPAWNY` was added to the drape scene for this work: it lowers the drop so the sheet
arrives slowly, which is what ruled tunnelling out.

**Verification:** forced-fresh `.spv` for both P8.17 shaders, suite through `tools/run_suite.sh` —
**674/674 in both memory modes** (`gpuslang.femClothSim` 46 → 47 fixtures); `cloth`, `clothdrape`
(plane floor), `clothhang`, `clothcloth` and `clothselfcollide` all green. The engine probes were
reverted; the only engine-source change in this tranche is none.

## P8.17 step 1 — cloth-vs-COOKED-TRIANGLE-MESH: the midphase

`cloth_meshMidphaseGeneratePairsLaunch` on the engine BDA ABI. **Manifest 254 → 255.** New shader
`femClothMeshMidphaseBda.slang`. Step 2 (`cloth_meshContactGenLaunch`) is the next tranche — until
it lands the pairs this kernel produces have no consumer, and the scene below says so out loud
rather than pretending otherwise.

Both launches sit inside `testSDKFemClothTrimesh` (`PxgNarrowphaseCore.cpp:6534`) under `if (1)`,
so this is a live path — not one of the four cloth-VERTEX-vs-mesh/HF kernels P8.14 ruled dead.

**This is `softBodyMeshMidphaseBda`'s sibling** — same cooked blob, same top-4 extraction, same
grid — with a cloth TRIANGLE query instead of a softbody TETRAHEDRON. Four things change, and
every one is a plausible copy-paste from the sibling:

| | softbody sibling | this kernel |
|---|---|---|
| SAT | tet-vs-tri: four plane terms, no cross axes | tri-vs-tri: two plane terms **plus nine edge-cross axes**, each normalized |
| query gate | `isValidTet` + the `mTetMeshSurfaceHint` byte | none — every cloth triangle is a query |
| box test | unfattened bound, per-axis **displacement** vs `contactDist²` | **plain AABB overlap** against a **pre-fattened** bound |
| contact distance | world sum **÷ min\|scale\|** | the world sum, **undivided** |

The last two are the ones worth staring at. `TriangleLeafBoundMinMaxTraverser` is the name of
**two different structs in two different .cu files**, and they disagree about the box test.  And
`femClothClothMidPhase.cu:358` really does fatten a vertex-space bound by a raw world distance —
transcribed as written, because "fixing" it would change the emitted pair set on any scaled mesh.

**The fixture, and what the first version got wrong.** The expected pair set is computed by a CPU
transcription of the leaf predicate rather than written down, so a mutant anywhere in
`satIntersect` fails without the fixture having to anticipate which axis it broke. That paid for
itself immediately: the GPU matched the CPU reference on the first run and my **hand-written**
expectations were the thing that failed.

The reason is worth recording, because it is a property of this SAT rather than a mistake in the
port. `satIntersect`'s axis set is the two face normals plus the nine edge crosses — and **for two
PARALLEL triangles every one of those nine crosses is parallel to the common normal**. So the SAT
reports the plane gap for a pair of horizontal triangles *no matter how far apart they are in x*.
The first fixture put every horizontal triangle in one BVH leaf and expected each cloth triangle
to match only the one beneath it; instead each matched its neighbours' triangles too, at their
y-gap. The BVH boxes are the only thing that separates laterally-offset parallel triangles — which
is exactly their job in the engine. The fixture now gives each query slot its own leaf.

**18 mutants, 17 killed.** Five survived the first sweep and every one was a fixture weakness:

* **`rotateInv` → `rotate` on the world→local step.** The transform was 180 degrees about y, which
  **is its own inverse**. Changed to 90 degrees.
* **the separating axis left unnormalized.** The cross pair's axis was 4 units long, so dropping
  the `1/sqrt(magSq)` made the reported separation *larger* and the verdict did not move. The pair
  now has 0.5-length edges, so the axis is 0.25 long and the same mutant scales 0.25 down to
  0.0625 — from outside the band to inside it.
* **the two plane terms applied to the wrong triangle.** Each then measures a triangle against its
  own plane and returns 0 — but for parallel triangles the *cross* axes recover the true gap, so
  the answer was unchanged. Killed by a pair whose edges are all 0.03 long: every edge cross then
  has `magSq ≤ 8.1e-7`, below `satIntersect`'s `1e-5` floor, so all nine are skipped and the plane
  terms decide alone (0.28 correctly, 0.0 mutated).
* **the box test swapped for the softbody form.** Axis-aligned offsets cannot tell the two forms
  apart. Killed by a leaf box offset from the query bound by 0.15 in **both** x and y: the
  fattened-bound overlap accepts it, the unfattened displacement rejects it
  (0.15² + 0.15² = 0.045 > 0.20² = 0.04).
* **three others killed the process rather than a check** (the `.z`/`.w` node-array size, the
  `nbPrimitives` shift, and the inverted shape selection all read wild addresses). The first sweep
  reported them as SURVIVED because its kill detection only looked for the `[ FAIL ]` line and
  ignored the exit code. `tools/`-style sweeps now treat a nonzero return as a kill — a mutant that
  faults the process is killed just as surely as one that fails an assertion.

**One mutant survives, and it is provably equivalent.** Passing `contactDist` instead of
`min(bestDistance, contactDist)` as `satIntersect`'s tolerance changes only how early the routine
gives up. The value it returns on an early-out is always **greater than the tolerance**, and the
tolerance is `min(bestDistance, contactDist)`: if it was `bestDistance`, the value fails
`separation < bestDistance` and is not stored; if it was `contactDist`, the stored value is above
`contactDist` and fails the final band test. The outcome is identical for every input. Recorded
rather than chased.

**`PXG_CLOTH_FLOOR=mesh` — a diagnostic, not a settle test.** `PXG_SCENE=clothdrape` can now be
given a cooked 20 × 20 triangle floor instead of the infinite plane, the same floor
`PXG_SB_FLOOR=mesh` gives the softbody. With only the midphase ported the sheet still falls
straight through (mean y 1.50 → **−42.89**), so the scene checks only that the run is well-formed
and prints the numbers; the settle assertions switch back on when step 2 lands. What it *does*
show is that the kernel is reached: `PXG_LAUNCH_TRACE=1` counts **6 dispatches** of
`cloth_meshMidphaseGeneratePairsLaunch` over the run, where an unmapped kernel produces none (the
bridge refuses it before the trace point).

**Verification:** forced-fresh `.spv`, suite through `tools/run_suite.sh` — **674/674 in both
memory modes** (`gpuslang.femClothSim` 45 → 46 fixtures); `cloth`, `clothdrape`, `clothhang`,
`clothcloth` and `clothselfcollide` all green.

Next: step 2, `cloth_meshContactGenLaunch` (`femClothPrimitives.cu:3749`, launch
`PxgNarrowphaseCore.cpp:6669`) — a 15-argument pack ending in a `PxgFEMContactWriter`, which is
**not** `kSbMeshContactGen`'s layout.

## P8.16 — cloth-vs-DIFFERENT-cloth: the two kernels P8.14 scouted, and the scene that proves them

`cloth_differentClothCollisionVTLaunch` + `cloth_differentClothCollisionEELaunch`, on the engine
BDA ABI. **Manifest 252 → 254.** New shader `femClothDifferentClothBda.slang`, importing the
`pxgSlangClothMidphase` module P8.15 extracted for exactly this second consumer.

**The blob is NOT the self-collision one.** That was the first thing P8.14 had to establish and
it is worth repeating here, because the two kernels look like the same family:

| | self-collision | different-cloth |
|---|---|---|
| ABI row | `{4,8,8,8,8,4,8,8,8}` (72 B) | `{4,4,8,8,8,8,8,8,4,8,8,8}` (88 B) |
| where the cloth ids come from | `activeClothes[blockIdx.y]` | `cmInputs[blockIdx.y]` → two `PxgShape::particleOrSoftbodyId` |
| contact distance | `contactDistance[elementIndex] * 2` | `contactDistance[cacheRef0] + contactDistance[cacheRef1]` |
| `eDISABLE_SELF_COLLISION` | checked | not checked — that flag gates self-collision only |
| carries `toleranceLength` | no | yes (stored, never read) |

**The five differences, and what each costs if you transcribe the self path instead.** The two
traversers are the SAME C++ structs (`ClothTreeVertexTraverser` / `ClothTreeEdgeTraverser`), so
the leaf and box callbacks are the proven `femClothSelfCollisionBda` transcriptions. What differs
is everything around them:

1. **The pair is narrowphase-driven** (above).
2. **The contact distance is a SUM**, not one reference doubled.
3. **VT sweeps BOTH directions in one launch**: `globalWarpIdx < numVerts0 + numVerts1`, the
   first `numVerts0` warps querying cloth0's vertices against cloth1's tree and the rest the
   mirror. EE runs only `min(numTri0, numTri1)` queries with the SMALLER cloth (by
   triangles-with-active-edges) always driving — `isCloth0 = numTri0 <= numTri1`. Both use the
   branchless id select the CUDA writes verbatim (`mask * (id1 - id0) + id0`).
4. **Every `clothId0 != clothId1` guard is live and TRUE**, so the self path's three skips are
   all disabled: the VT vertex-is-a-corner-of-the-triangle test, the EE `t0_index > primitiveIndex`
   half-pair dedup, and the EE connected-edge test. The corollary is
   `markValidity(clothId0 != clothId1)` → **VALID**: a different-cloth pair is final at the
   midphase, where a self pair is re-checked in `cloth_clothContactPrepareLaunch`.
5. **The rest-pose filter is GONE.** The call sites pass NULL for both rest buffers and `0.0f`
   for `filterDistSq` (`femClothClothMidPhase.cu:1298-1300`, `:1370-1371`); `distSqInRestPose`
   returns `PX_MAX_REAL` for a null first argument (`:662-668`), so `tempRestDistSq > filterDistSq`
   is unconditionally true. Transcribing the self-collision lookup here would **dereference a null
   device pointer** — omitting the block is what the CUDA collapses to, not a deviation from it.
   This is the reason P8.15 deliberately left the rest-pose helpers out of the shared module.

**The fixtures, and the two survivors that made them sharper.** 18 mutants, 16 killed on the
first sweep. Both survivors were fixture weaknesses, not port defects, and both are the same
kind of mistake — a scene symmetric enough that the mutant lands back on the right answer:

* **`selMask` inverted** (the VT branchless select). It swaps which half of the query range is
  which cloth — but with the penetrating probe at vertex index 3, *both* arrangements cover index
  3, and because the select swaps `my`/`other` consistently the emitted ids come out **identical**.
  Fix: cloth 0 has five vertices and cloth 1 four, so moving the probe to index **4** makes the
  two halves cover different index sets, and the mutant drops the contact.
* **the EE query walking its OWN tree** instead of the other cloth's. Cloth 0's tree holds a
  single primitive, so the mutant still visits index 0 — and with the crossing triangle at
  cloth 1's index 0 it read exactly the right triangle. Fix: move the crossing triangle to
  primitive **1**, so the wrong tree finds the far triangle instead.

Both then died. The remaining 16 cover: the corner skip folded the self way, both halves of the
contact-distance sum (`2*cd[ref0]` and `2*cd[ref1]`), `markValidity` on both entry points, the VT
range truncated to `numVerts0`, the second half's `vertexIdx` not rebased, the
`updateContactPairs` early exit, `pairInd1` encoded with the wrong cloth id, the EE half-pair
dedup and connected-edge skip folded the self way, `isCloth0`'s comparison flipped, the
`t0_bounds` fattening dropped, the pair-type bit dropped, `auxInd0`/`auxInd1` swapped, and the
sub-warp loop truncated to one warp.

One deliberate note on the EE fixture's threshold: it is chosen against the two crossings'
**actual** `distSq` (0.006791 and 0.004851), not their geometric separation.
`closestPtLineLine` normalizes `b0`/`b1` by DIFFERENT edge lengths
(`deformableCollision.cuh:244`), so what it returns is not the true minimum distance between the
two segments. The first attempt picked the threshold from the geometry and found one crossing
where it expected two.

**`PXG_SCENE=clothcloth` — the scene, and a decisive control.** Two separate
`PxDeformableSurface` actors: a fully PINNED lower sheet acting as a cloth table, and a FREE
upper sheet dropped onto it from 0.30 above. Both are built from the **same recipe** — same
material, thickness, contact and rest offsets, same grid spacing — because P8.8 spent a whole
tranche chasing what turned out to be a mass ratio, and a shared recipe removes that confound by
construction. Self-collision is **disabled on both** (`eDISABLE_SELF_COLLISION`), which makes
`cloth_selfCollisionMidphase{VT,EE}` early-exit, and there is no rigid body underneath — so the
two new kernels are the only midphase that can see the pair.

```
with the port:      final upper sheet min y = 5.1000  (mean 5.1002), table y = 5.00
kernels unmapped:   final upper sheet min y = -38.9756 (mean -38.9756)
```

The control is the manifest rows reverted to `nullptr`, which makes the Vulkan bridge refuse the
dispatch. The sheet settles exactly on the contact rest distance (table + 2 × restOffset = 5.10)
with the port, and falls forty units through the table without it.

The downstream solve is unchanged: `cloth_clothContactPrepareLaunch` → query →
`cloth_solveClothClothDelta{VT,EE}`, the same chain the self path uses, ported at P7.2 tranche 2
— its header already recorded that "the different-cloth pairs flow through the identical generic
arithmetic", and this scene is the first thing to actually exercise that claim.

**Pre-existing and out of scope:** the scene prints `radixSortCopy` / `radixSortCopyBits` /
`clampMaxValues` launch failures from `PxgFEMClothCore.cpp`. `PXG_SCENE=clothselfcollide` prints
the same three, so they are not introduced here — they are unmapped kernels in the cloth contact
SORT, which no cloth scene has needed yet.

**Verification:** forced-fresh `.spv` (both `femClothDifferentClothBda` and
`femClothSelfCollisionBda`, since the shared module changed consumers), suite through
`tools/run_suite.sh` — **674/674 in both memory modes** (the two new fixtures live inside the
existing `gpuslang.femClothSim` ctest entry, which went 43 → 45); rigid
regression green; `clothcloth` green with the control red; `clothselfcollide` PGS min y **5.0999**
/ mean 5.1106 and TGS 5.1000 / 5.1104, both on baseline; `cloth`, `clothdrape`, `clothhang` green.

Next from the P8.14 ranking: cloth-vs-cooked-trimesh (`cloth_meshMidphaseGeneratePairsLaunch` +
`cloth_meshContactGenLaunch` — and its arg pack is **not** `kSbMeshContactGen`'s), then FEM-FEM
attachments, then `sb_heightfieldContactGenLaunch` (P8.11 step 2).

## P8.15 — `pxgSlangClothMidphase`: the shared cloth-midphase machinery, extracted before it got duplicated

Groundwork for cloth-vs-different-cloth, done as its own step so it can be proved
behaviour-preserving on its own.  **No kernel was mapped; the manifest is unchanged at 252.**

The two kernels P8.14 scouted are the self-collision midphase with two cloths, so they need the
same BV32 node readers, the same `closestPtPointTriangle` / `closestPtLineLine`, the same bounds
and edge-activity predicates, and the same `auxInd` bit layout that `femClothSelfCollisionBda`
already carries.  The choice was to copy them or to share them.

**This port has already paid for the copy once.**  `softBodyClothMidphaseBda`'s header records
it: a mutant that deleted an outer front-face gate SURVIVED, because a duplicated inner copy of
the same cull was still there.  Duplicated geometry does not merely age badly — it silently
absorbs mutants and makes the sweep report coverage it does not have.  So the helpers moved to
`pxgSlangClothMidphase.slang`, the same treatment `pxgSlangTetClosest` got at P8.8 step 4 when
the cloth-vertex contact generator became its second consumer.

**What went, and what deliberately did not:**

| moved | stayed |
|---|---|
| BV32 readers (`bvMin`/`bvMax`/`bvData`/`bvNbNodes`) and their offsets | `writeContactInfo` — it dereferences the importing shader's own push constants |
| `closestPtPointTriangle`, `closestPtLineLine` | `ballotGroup` — it owns groupshared state, and moving it would obscure where the barriers are |
| `boundsIntersect`, `isType0EdgeActive`, `pxEncodeClothIndex` | `distSqInRestPoseVT` / `...EE` |
| the `auxInd` bit layout, the barycentric thresholds, `WARP`, `STACK_SIZE`, `MIDPHASE_WARPS_PER_BLOCK` | |

**The rest-pose helpers staying put is a correctness decision, not tidiness.**  The
different-cloth call passes NULL for both rest buffers, and `distSqInRestPose` returns
`PX_MAX_REAL` for a null first argument (`femClothClothMidPhase.cu:662-668`), so that filter
collapses on the new path.  A shared rest-pose helper would invite exactly the null dereference
the collapse avoids — and on llvmpipe a speculated load through null is a fault whose location
moves with the groupshared layout, the failure mode this port has chased twice.

The module is PURE: no push constants, no groupshared, no side effects.
`femClothSelfCollisionBda` shrank 817 → 639 lines (185 deleted, 6 added).

**Proof that the extraction changed nothing**, which is the whole point of doing it separately:

* forced-fresh `.spv`, then the suite through `tools/run_suite.sh` — **674/674 in both memory
  modes**;
* `PXG_SCENE=clothselfcollide` on PGS: min y **5.0999**, mean 5.1106, against the recorded
  baseline of 5.0999 / 5.1105.  The last digit of the mean varies run to run because the
  deformable solves scatter through real float atomics and the summation order is not fixed —
  recorded at P8.9, and `min y` is the stable figure;
* the same scene on TGS (green since P8.12): 5.1000 / 5.1104;
* `cloth`, `clothdrape`, `clothhang` all still green.

One pre-existing slangc warning (`possible use of uninitialized variable` on the EE path's
`localEdgeIndices`) moved with its line number and is unchanged from HEAD — the construct is
byte-identical, and the array is written up to `intersectCount` and read only below it.

Next: `femClothDifferentClothBda` with both entry points, importing this module, with the four
differences P8.14 lists and the rest-pose block omitted.

## P8.14 scouting — cloth-vs-DIFFERENT-cloth is LIVE, and the survey's "dead" verdict was wrong

Scouting for the next tranche, `cloth_differentClothCollisionVTLaunch` +
`cloth_differentClothCollisionEELaunch`.  **No kernel was ported in this tranche** — what
follows is the reachability and porting groundwork, recorded because two of the findings would
otherwise have to be re-derived, and one of them reverses a verdict that would have caused the
pair to be skipped entirely.

**1. THE PAIR IS LIVE.  The empty narrowphase bucket entry is BY DESIGN, not dead code.**
The frontier survey flagged this group as skippable on the grounds that `testSDKFemClothCloth`
is an empty stub.  It IS an empty stub — and the engine says why, in the function itself
(`PxgNarrowphaseCore.cpp:6466`):

```cpp
void PxgGpuNarrowphaseCore::testSDKFemClothCloth(...)
{
    // FEMCloth-FEMCloth collision handled in PxgFEMClothCore.cpp
```

The bucket exists so the contact managers get REGISTERED; the midphase itself runs from the
SOLVER side, in `PxgFEMClothCore::differentClothCollision()` (`:733`, dispatch at `:785`), which
`prepareClothClothCollision` (`:1931`) calls unconditionally right beside `selfCollision()` —
and `selfCollision()` is already mapped and already green in `PXG_SCENE=clothselfcollide`, so
the enclosing function is demonstrably reached every step.  The only thing gating the two
kernels is `numTests > 0`, i.e. at least one cloth-vs-cloth contact-manager pair, and two
`PxDeformableSurface` actors route there unconditionally: `isCloth0 && isCloth1 ->
GPU_BUCKET_ID::eFemClothes` (`PxgNphaseImplementationContext.cpp:1136-1138`), no compatibility
test, no fallback.

**The general lesson, worth more than this pair:** an empty narrowphase bucket entry is NOT
evidence of dead code in this engine.  Three collision pairs are driven from the solver side
instead, and the bucket entry is a registration hook.  Reachability has to be traced from the
per-step entry through to the dispatch, not inferred from the bucket table.  (The genuinely dead
`cloth_clothMidphaseGeneratePairsLaunch` is a DIFFERENT kernel with no dispatch site at all —
the survey conflated the two.)

**2. THE TWO KERNELS ARE THE ALREADY-MAPPED SELF-COLLISION MIDPHASE WITH TWO CLOTHS.**  Same
`ClothTreeVertexTraverser` / `ClothTreeEdgeTraverser`, same output buffers
(`mVolumeContactOrVTContactInfoBuffer` / `mEEContactInfoBuffer`), same
`*updateContactPairs == 0` early exit, and the downstream solves
(`cloth_solveClothClothDeltaVTLaunch` / `...EELaunch`) are mapped and already cloth-id-generic.
Four differences, all consequences of there being two bodies:

| | self-collision (mapped) | different-cloth (to port) |
|---|---|---|
| which cloth | `activeClothes[blockIdx.y]` | `cmInputs[blockIdx.y]`, both shapes |
| query element | `globalWarpIdx < numVerts` | `globalWarpIdx < numVerts0 + numVerts1`, first half queries cloth 0 against cloth 1's tree, second half the reverse |
| contact distance | one body's, DOUBLED | the SUM of the two cache refs |
| shared-vertex skip | required | `clothId0 != clothId1 \| !(...)` — never applies |

**3. THE REST-POSE FILTER IS A NO-OP ON THIS PATH, and that is easy to port wrong.**  The
different-cloth call passes `NULL` for both rest-vertex buffers and `0.0f` for the filter
distance (`femClothClothMidPhase.cu:1298-1300`), and `distSqInRestPose` returns `PX_MAX_REAL`
when its first argument is null (`:662-668`).  So `tempRestDistSq > filterDistSq` is
unconditionally true and the whole block collapses.  Transcribing the self-collision version's
rest-pose lookup here would dereference a null pointer — on llvmpipe, with the speculated-load
behaviour this port has hit repeatedly, that is a fault whose location moves with the
groupshared layout.

**4. THE SCENE.**  Two `PxDeformableSurface` actors, one pinned and one dropped onto it, with
today's interpenetration as the control — the cheapest new scene in the port, and the pattern
`softbodystack` already established: both sheets from one recipe so the mass ratio cannot be the
confound it was in P8.8.

Nothing is claimed to work yet.  Manifest unchanged at 252; suite unchanged.

## P8.13 — the recurring one-off suite failure: what it is NOT, and how the next one gets caught

Five times across this port the per-tranche `ctest -j4` has failed EXACTLY ONCE, always in the
first parallel run after a forced-fresh `.spv` rebuild, and passed on every run after — with the
`PXG_DEVICE_V2=1` mode green throughout.  Each occurrence was noted and deferred.  It has now
recurred often enough to be worth an attempt, so this tranche is that attempt.

**IT DOES NOT REPRODUCE.  Ten deliberate cycles, none fired:**

| reproducer | cycles | result |
|---|---|---|
| forced-fresh `.spv` → rebuild → `ctest -j4` (the recipe step verbatim) | 6 | 674/674 every time |
| the same, PLUS the test binary RELINKED, `ctest -j8`, and a concurrent `build-sdk` compile loading the machine | 4 | 674/674 every time |

**What that rules out**, together with four things checked directly:

* not the forced-fresh rebuild by itself, and not the relink either;
* not parallel-test contention — doubling it to `-j8` under load changed nothing;
* not disk exhaustion: 30 GB free, and the two build trees are 52 MB and 69 MB;
* not a per-test timeout: no `TIMEOUT` property is set anywhere in the tests' CMakeLists, so
  ctest's 1500 s default applies, and the slowest test (`cvxCollide`) takes 168 s;
* not the on-disk pipeline cache: only `testP1PipelineCache` ever sets `pipelineCachePath`, and
  it uses its own file, so there is no cross-test race over a shared blob.

**WHY IT WAS NEVER DIAGNOSABLE, which turns out to be the real finding.**  ctest overwrites
`Testing/Temporary/LastTest.log` on the next invocation — and the next invocation was always the
re-run that confirmed the suite green.  Every one of the five occurrences destroyed its own
evidence, by the act of checking it.  I never even learned WHICH test failed; the `tail -3` that
caught the failure cut off the summary line above it.

**So the deliverable is not a fix, it is an instrument.**  `tools/run_suite.sh` replaces the
bare `ctest -j4` in the tranche recipe.  It runs both memory modes and, on any failure, archives
`LastTest.log`, the full ctest stdout, and a context snapshot (date, load average, disk, memory)
into a TIMESTAMPED directory before anything can overwrite them, then prints the failing test
names.  Per-mode directories, so a second mode's failure cannot destroy the first's evidence.

**The failure path is tested, not assumed.**  Removing one `.spv` and running the wrapper
produced exactly what it should:

```
run_suite.sh: FAILURE ARCHIVED -> build-gpuslang/suite-failures/20260802T135937Z-default
  failing tests:
      220 - gpuslang.femClothSim (Failed)
      299 - gpuslang.p4Manifest (Failed)
```

with `LastTest.log`, `ctest-stdout.txt` and `context.txt` in each of the two per-mode
directories.  An untested evidence-preservation path is worthless exactly when it is needed, and
this one had a real bug to catch: written naively, the v2 run's `LastTest.log` overwrites the
default run's before either is archived.

That run also confirmed a useful second signal: a missing `.spv` fails `p4Manifest` as well as
the test that uses it, because that fixture asserts every mapped manifest row resolves.

**Honest status: the transient is NOT fixed and NOT explained.**  What changed is that the next
occurrence will leave evidence instead of erasing it.  `suite-failures/` sits inside the
git-ignored build tree, so archives never reach a commit.

## P8.12 — MILESTONE: the TGS CLOTH SOLVER RUNS (`cloth_stepLaunch`, one kernel)

`cloth_stepLaunch` is mapped, and with it **every cloth scene passes on BOTH solvers**.

**This kernel's absence was why no cloth moved on TGS.**  On PGS the fall comes from
`cloth_preIntegrateLaunch` (mapped since P7.0), which does the gravity predict AND the position
integration in one shot at the top of the step.  On TGS that kernel only predicts; the POSITION
integration lives in `cloth_stepLaunch`, called once per solver substep from
`PxgFEMClothCore::step()` inside the `if (mIsTGS)` branch at `PxgFEMClothCore.cpp:1380`
(dispatch at `:1501`).  With `CLOTH_SIM_STEP` unmapped the bridge refused the launch and the
cloth simply never advanced.

**KEY EVIDENCE — the proof was a PRE-EXISTING RED SCENE.**  Measured before the port:

```
PXG_SCENE=clothdrape          (default solver = TGS)
  clothdrape: mean y 1.5000 -> 1.5000 (dropped 0.0000 over 3.00s)
  FAIL: cloth did not fall (drop=0.0000)
```

and after:

```
  clothdrape: mean y 1.5000 -> -0.0000 (dropped 1.5000 over 3.00s), rest |dy| = 0.000000
  [  OK  ] FEMCloth fell onto the floor and SETTLED under contact
```

| scene | TGS before | TGS after | PGS |
|---|---|---|---|
| `cloth` | red | **green** | green |
| `clothdrape` | red | **green** | green |
| `clothhang` | red | **green** | green |
| `clothselfcollide` | red | **green** | green |

That inverts the risk the last two tranches were caught by.  P8.10 and P8.11 both had a GREEN
fixture and a green mutation sweep over a shader that was wrong, and the scene found it.  Here
the scene was the pre-registered failing test from the start.

**The blob has a trap.**  `PxVec3 gravity` is TWELVE bytes, and `paramAlign(12)` gives 4, not
16 — so it sits at @20 UNPADDED, the three `bool`s pack at @32/@33/@34, and the next pointer
realigns to @40.  A 64-byte blob, ABI row `{8,8,4,12,1,1,1,8,8,8}`.  Padding the PxVec3 to 16
moves every pointer after it.

**What the kernel does, and what a settling scene cannot separate:**

| | |
|---|---|
| contact-pair bookkeeping | ONE thread of the dispatch publishes `updateContactPairs` and, on a forced update, zeroes the VT and EE contact counts — that is what makes the narrowphase re-run |
| gravity + damping | only when `externalForcesEveryTgsIterationEnabled`, gated on invMass != 0 and on `eDISABLE_GRAVITY`; damping is `1 - min(1, linearDamping*dt)` |
| integration | `delta = vel*dt`, added to the position AND the accumulated delta |
| displacement clamp | three regimes, measured against `mPrevPositionInRestOffset` rather than the substep start |
| previous positions | rest-offset copy written ALWAYS, contact-offset copy only on a forced update, with `.w` marked 1 |

**THE `min()` IN THE DAMPING IS THE POINT, NOT A GUARD.**  At `linearDamping * dt > 1` the
unclamped factor goes NEGATIVE and the damping REVERSES the velocity instead of killing it.  The
fixture pins it at damping 8, dt 0.25: the correct factor is 0 and the mutant's is −1, turning
−42 into +42.

**Three fixtures** (`femClothSimStep`, `...StepFlags`, `...StepDampingAndFlags`; 43 in the file),
every expected value computed by hand at dt = 0.25 and matched exactly.

**Mutation sweep: 13 of 14 killed** (forced-fresh `.spv` each time) — damping without the
`min()`; gravity applied to pinned vertices; `eDISABLE_GRAVITY` ignored; the clamp correction not
reaching the velocity; the accumulated delta not updated; the `isPrevPosSet` gate removed; the
`.w = 1` mark dropped; the mark applied to the rest-offset copy too; the contact-offset copy
written unconditionally; the contact counts not zeroed; the PxU8 store clobbering its whole word;
the negative-`maxLinearVelocity` regime using `maxVel*dt`; the clamp measured against the substep
start.

**ONE SURVIVES, and the arithmetic says why.**  Deleting the `mMaxLinearVelocity < PX_MAX_REAL`
early-out changes nothing measurable: at `PX_MAX_REAL` the cap becomes `PX_MAX_REAL * dt`, its
SQUARE overflows to infinity, and `displacementSq > inf` is false for every finite displacement.
The guard is an early-out, not a behaviour switch, and no fixture can separate the two without a
displacement of order 1e19.

**Three fixture bugs of mine, all found by running the sweep rather than by reading.**
(1) The `.w = 1` mark was invisible at inverse mass 1, because the mark and the mass are the same
number — vertex 2 now carries invMass 2 so the two are separable.  (2) The damping clamp needed
`damping * dt > 1` to bite; at 0.5 the clamped and unclamped factors agree.  (3) The negative-
`maxLinearVelocity` regime and the rest-offset reference both needed a vertex that actually
TRAVELS and whose reference is not its own start position; cloth C is built so the correct answer
(8.0), the wrong-regime answer (11.75) and the wrong-reference answer (6.0) are three distinct
numbers.

Manifest hand-edited 251 → 252.  Suite **674/674 in both memory modes**, rigid regression green,
and every other scene unchanged — `softbodyselfcollide` 0.2133, `softbodycloth` 0.8688,
`softbodystack` 0.9822, `particle` green, and the softbody floor matrix bit-identical at box
**0.49628** / plane **0.49753** / sphere **0.48436** / mesh **0.49666**.

*(The default-mode `ctest -j4` failed once immediately after the forced-fresh rebuild and passed
674/674 on the run that followed, with V2 green throughout.  This is the FIFTH one-off of that
shape, always in the first parallel run after a forced-fresh rebuild.  It has now recurred often
enough to be worth a deliberate diagnosis rather than another note.)*

## P8.11 step 1 — the softbody-vs-HEIGHTFIELD midphase

`sb_heightfieldMidphaseGeneratePairsLaunch` is mapped, the first of **two** for
softbody-vs-heightfield.  Step 2 (`sb_heightfieldContactGenLaunch`) is the remaining one, and it
is the trimesh path's step 2 with the triangle synthesised from heightfield samples instead of
read from a blob — same helper, same `flipOutputNormal = true` / `alternativeWriteMode = true`
(`softbodyPrimitives.cu:1136`), which P8.10 has now validated.

**There is no BVH on this path.**  Every other midphase in the port walks a BV32 tree; a
heightfield has none.  Its geometry IS its index: the tetrahedron's world AABB is pushed into
the heightfield's unscaled row/column space and the integer rectangle it lands on names the
candidate quads directly.  The cooked blob is flat too — `nbRows @0 | nbCols @4 | samples @8` —
and a sample is four bytes: `PxI16 height`, `PxBitAndByte materialIndex0`, `materialIndex1`.

**The material bytes carry a flag in bit 7** (`tessFlag`, which way the quad splits), and
`PxBitAndDataT::operator T()` masks it off.  Comparing the raw byte against `eHOLE = 127`
therefore treats a tessellated hole as solid.

**Fixture** `softBodySimHeightfieldMidphase` (58 in the file).  A 4 × 4 field with a hole in one
material slot of quad (0,0), a hole-with-the-flag in both slots of quad (0,1), a
solid-with-the-flag quad at (1,0), a mixed-sign quad at (1,1), and rows 2–3 raised to +50.  Five
pairs come out — one from quad (0,0), two each from (1,0) and (1,1).

**Mutation sweep: 5 of 7 killed** (forced-fresh `.spv` each time) — the tessFlag not masked off;
the `PxI16` height read as unsigned; `numTets` taken from `mNumTets` instead of the tetmesh blob
header; the triangle index without its material-slot bit; the surface-hint gate removed.

**TWO SURVIVE, and the arithmetic says why — both are properties of the kernel, not thin spots
that more geometry would fix.**

* **`getMinRow` clamping to `rows-1` instead of `rows-2`.**  Upstream's own rectangle guard gets
  there first: `!(lMin.x > nbRows - 1) || ...`, so a tetrahedron whose minimum row would exceed
  `rows-2` is either past `nbRows-1` and thrown out by the guard, or below it and clamped
  identically by both versions.  A tetrahedron placed at row 3.4 to catch it is simply rejected
  earlier — measured, 0 pairs.
* **Dropping `fattenFast(contactDist)`.**  The contact distance here is 0.002 in WORLD units,
  i.e. 0.001 of a row at rowScale 2 — three orders of magnitude below the quad size, so the
  fattened and unfattened bounds floor and ceil to the same integers.  Raising it enough to
  cross a boundary would spill every other tetrahedron into its neighbours' quads.  Step 2's
  fixture is where the contact distance has teeth.

**A third thing the fixture could not cover, for an upstream reason.**  Upstream un-scales the
local bound by `1/|scale|` and then swaps min and max when the scale is negative.  That is NOT
the same as dividing by the SIGNED scale: for `rowScale = -1` and a tetrahedron at row 0.2–0.6,
the shape-space bound is −0.6…−0.2, the division by `|scale|` leaves it there, and the swap
produces min = −0.2, max = −0.6 — an inverted interval whose clamps collapse to
`maxRow == minRow`, so nothing is emitted.  The first version of this fixture used
`rowScale = -1` and got zero pairs for exactly that reason.  The port transcribes the branch as
upstream has it and the fixture uses positive scales; a mutant deleting the swap survives.

**Two fixture bugs of mine, both found by running it.**  The tetrahedron meant to be rejected
"entirely below all four corner heights" sat under a quad with two corners at +50 and two at
−50 — it was straddled, and correctly kept; raising both rows fixed it.  And the signed-height
mutant survived the first corrected version, because for an all-negative quad the signed read
rejects through `con1` and the unsigned read through `con0` — both reject, indistinguishably.
Only a MIXED quad separates them, and quad (1,1) is now built that way.

Manifest hand-edited 250 → 251.  No scene changes — step 2 is still unmapped, so the engine
refuses it and the midphase output goes unread.  Suite **674/674 in both memory modes**, rigid
regression green, every deformable scene unchanged, and the softbody floor matrix bit-identical
at box **0.49628** / plane **0.49753** / sphere **0.48436** / mesh **0.49666**.

*(One transient: the default-mode `ctest -j4` failed once immediately after the forced-fresh
rebuild and passed 674/674 on the two runs that followed, with V2 green throughout.  This is the
fourth one-off of that shape in this port; noted, not diagnosed.)*

## P8.10 — MILESTONE: a FEM softbody SETTLES ON A COOKED TRIANGLE MESH (step 2 of 2)

`sb_meshContactGenLaunch` is mapped, and with it **softbody-vs-triangle-mesh is complete**.  The
contacts go into the RIGID contact stream through `PxgFEMContactWriter`, so everything after it
was already mapped by the softbody-vs-rigid tranches.

**KEY EVIDENCE — a FOURTH entry in the softbody floor matrix.**  `PXG_SB_FLOOR=mesh` builds a
20 × 20 cooked triangle-mesh quad in the plane y = 0 — the same floor HEIGHT as the box, the
plane and the sphere's north pole, which is why it belongs in this matrix rather than in a scene
of its own:

| floor | settle mean y | rest \|dy\| |
|---|---|---|
| box | 0.49628 | 0.000899 |
| plane | 0.49753 | 0.000002 |
| sphere | 0.48436 | 0.000575 |
| **mesh (new)** | **0.49666** | **0.000116** |

22 deformable-volume contacts from step 49, and the number lands between the box and the plane —
which is what a flat cooked floor should give.

**The push blob is BYTE-IDENTICAL to `sb_primitiveContactGenLaunch`'s**: the same fifteen
arguments in the same order and sizes, the same 192-byte blob, the same ABI row
`{4,8,8,8,8,8,8,8,8,8,4,8,4,8,80}`.  It is kept as its own row rather than aliased, so a future
divergence upstream shows up here.

### Two real port bugs, and the scene is what found them

The fixture passed, the mutation sweep was green, and the live scene then had the cube
**accelerate past free fall on first contact** — mean y −70.1 where an uncaught fall reaches
−50.8.  Both causes were arguments to the same helper that I had read past:

```
writeContactNoBarycentricFullWarp(writer, intersect, &closestPointA.x, &closestPointB.x,
                                  0, &normal.x, /*flipOutputNormal*/ true, restDist,
                                  pairId0, pairId1, materialIndex, /*alternativeWriteMode*/ true)
```

**1. `flipOutputNormal = true`.**  The hull path passes `false` (`softbodyPrimitives.cu:568`),
this one passes `true` (:889), so the xyz that reaches the solver is **−direction**.  The clouds
are the reason: on this path cloud A is the TETRAHEDRON and cloud B the TRIANGLE, so GJK's
direction runs deformable → rigid, the opposite of the hull path's rigid → deformable, and the
flip is what puts both back on the solver's convention.  The PENETRATION is computed from the
UNFLIPPED direction and is not negated with it.  Getting this wrong does not lose contacts — it
pushes the softbody INTO the floor, which is exactly what the trajectory showed.

**2. `alternativeWriteMode = true`.**  It selects a *different output buffer*:

| | hull path (`false`) | mesh path (`true`) |
|---|---|---|
| `contactByX[i]` | the full 64-bit node index | **untouched** |
| `tempContactByX[i]` | its low word | **untouched** |
| `contactSortedByX[i]` | untouched | **the low word** |
| `contactIndexSortedByX[i]` | `i` | `i` |

I had transcribed the hull path's pair, which leaves `contactSortedByX` stale and fills two
buffers this path's sort never reads.

**The fixture asserted the wrong behaviour for both**, which is the honest part of this: a green
fixture and a green mutation sweep certified a shader that pushed a softbody through the floor.
The scene is what caught it.  Both assertions are now the right way round, and the two new
mutants (`N1` normal not flipped, `N2` the hull path's write mode) both die.

### A third one, in the step-1 fixture: a mutant that survived

The first `softBodySimMeshContactGen` also let the multiply-vs-divide mutant through.  The
triangle spanned vertex x in [0, 1], and at that position it still covers the tetrahedron under
EITHER reading of the scale — the scale is 1 in y, so the vertical gap never moves, and any
horizontal overlap at all yields the same contact.  Moving the triangles to vertex x in
[1.0, 1.4] puts the two readings 1.4 apart in shape space, far outside the 0.05 contact
distance, and the mutant dies.  **A geometric fixture only discriminates where the geometry
does.**

**Mutation sweep: 12 of 12 killed** against the corrected shader (forced-fresh `.spv` each time)
— vertex2Shape divides instead of multiplying; the familiar winding cross product; the rigid id
from the softbody's cache ref; the material from the softbody shape; the `& 0x7fffffff` fold
dropped; the filter check removed; the stack demand reported clamped; no stack clamp;
restDistance indexed past cmIdx; the stored point taken from the tetrahedron witness; the normal
not flipped; the hull path's write mode.

Suite **674/674 in both memory modes**, rigid regression green, and every other scene unchanged
— `softbodyselfcollide` 0.2133, `softbodycloth` 0.8688, `softbodystack` 0.9822,
`clothselfcollide` min y 5.0999, and the other three floors bit-identical.  Manifest hand-edited
249 → 250.

## P8.10 step 1 — the softbody-vs-TRIANGLE-MESH midphase

`sb_meshMidphaseGeneratePairsLaunch` is mapped.  It is the first of **two** kernels for
softbody-vs-triangle-mesh — the pair that lets a FEM volume rest on cooked static geometry
rather than on a primitive.  Step 2 (`sb_meshContactGenLaunch`, GJK/EPA per pair) is the
remaining one; everything after it is already mapped, because the contacts go into the RIGID
contact stream through `PxgFEMContactWriter` and from there through
`fem_reorderRigidContactsLaunch` → `sb_other_contact_remap_to_simLaunch` →
`sb_rigidContactPrepareLaunch` → `sb_solveRigidSoftCollisionLaunch`.

**The traverser is the one `softBodyClothMidphaseBda` already carries**, and that is a measured
claim rather than a convenience.  Both call the same
`TriangleLeafBoundMinMaxTraverser<BackFaceCull>` (`softbodySoftbodyMidPhase.cu:117`); the cloth
path instantiates it `<false>` and this one `<true>`.  The template argument switches ONLY
`satIntersect`'s internal `plane.distance(tet.centroid) < 0 → PX_MAX_F32` early-out (:79-87),
and the traverser body already applies exactly that test itself at :148, for BOTH
instantiations.  So the two instantiations compute the same number at these call sites and the
traversal is transcribed once.

**What the trimesh lift actually adds**, and what the fixture therefore aims at:

| | cloth midphase | trimesh midphase |
|---|---|---|
| geometry source | three descriptor fields | a COOKED BLOB walked from `hullOrMeshPtr` |
| query tet frame | cloth local | cloth local **and then** mesh VERTEX space |
| contact distance | the plain sum | the sum **divided by** the smallest \|scale\| component |
| blob | 88 B | 80 B |

The blob walk (`readTriangleMesh`, `triangleMesh.cuh:120`) is
`header uint4 @0 | BV32DataPacked[header.w] @16 | float4 verts[header.x] | uint4 indices[header.y]`.
**`header.z` is `nbBv32TreeNodes` and `header.w` is `nbPackedNodes`** — different counts, and
only `.w` sizes the node array.  The fixture sets `.z` to a deliberately wrong 7 so a port that
uses it walks the vertex pointer seven nodes too far.

The transform is two steps and both matter for a scaled mesh: `transformInv` (world → mesh
local, `q.rotateInv(v - p)`) then `shape2Vertex` (local → vertex space,
`rot.rotateInv(rot.rotate(v) / scale)`, which DIVIDES).  Skipping the second silently works at
scale (1,1,1) and breaks everywhere else.

**Fixture** `softBodySimMeshMidphase` (56 in the file).  The mesh carries a non-uniform scale
(2, 1, 0.5) and a real 180° rotation about y, and the tetrahedra are placed in world through the
inverse of the full transform, so the gap ladder below is a ladder in vertex space:

| tet | vertex-space gap | correct band 0.40 | undivided 0.20 | ÷ max scale 0.10 |
|---|---|---|---|---|
| 0 | 0.20 | in | in | out |
| 1 | 0.12 | in | in | out |
| 2 | **0.28** | **in** | **out** | out |
| 3 | 0.04 | in | in | in |
| 4 | 0.45 | out | out | out |
| 5 | 0.02, surface hint 0 | — interior, always skipped | | |

Seven pairs come out: one each for tets 0, 2 and 3, and **four** for tet 1, which sees six
qualifying triangles and hits the `MaxContacts = 4` cap.  The stack under tet 1 runs FAR TO NEAR
so the last primitive in the leaf is one the cap KEEPS — ordered the other way the cap discards
it anyway and a `primStartIndex` mutant changes nothing observable.

**Mutation sweep: 9 of 9 killed** (forced-fresh `.spv` each time) — no scale division; divide by
max instead of min; node array sized by `header.z`; `shape2Vertex` skipped; the `transformInv`
rotation skipped; the front-face gate removed; the surface-hint gate removed; `MAX_CONTACTS`
raised to 8; `primStartIndex` shifted by 8 instead of 7.  The `header.z` mutant kills by
**SEGFAULT** rather than by a check — the vertex pointer leaves the buffer — which is recorded
as what it is rather than dressed up as a clean assertion.

**One fixture bug, mine.**  The first run emitted ZERO pairs and I had written a comment
claiming the mesh's mirroring x/z rotation flips the tet's handedness, so I had "compensated" by
swapping two corners.  It does not: the corners are given in vertex space and mapped to world
through the inverse transform, and the kernel undoes that rotation before `isValidTet`, so the
handedness it sees is the one written down.  The compensation is what produced the negative
volume (−0.16) that skipped every tetrahedron.  Reverting to the plain ordering fixed it.

No scene changes: `sb_meshContactGenLaunch` is still unmapped, so the engine refuses it and the
midphase output goes unread.  Suite **674/674 in both memory modes**, rigid regression green,
and every deformable scene unchanged — `softbodyselfcollide` 0.2133, `softbodycloth` 0.8688,
`softbodystack` 0.9822, `clothselfcollide` min y 5.0999, softbody floor matrix bit-identical at
box **0.49628** / plane **0.49753** / sphere **0.48436**.  Manifest hand-edited 248 → 249.

## P8.9 — MILESTONE: a FEM CUBE STACKS ON A FEM CUBE (softbody-vs-softbody, in ONE kernel)

`sb_sbMidphaseGeneratePairsLaunch` is mapped, and with it **softbody-vs-softbody is complete**.

Every other collision pair in this port needed a chain of four to seven kernels.  This one needs
**one**: `testSDKSoftbodies` (`PxgNarrowphaseCore.cpp:4399`) dispatches this kernel and nothing
else, then hands the result to `softbodyFemContactApplyCollisionToSimMeshMapping`, i.e. the
already-mapped `sb_fem_contact_remap_to_simLaunch`.  Constraint prepare and solve are
`sb_softbodyContactPrepareLaunch` and `sb_solveOutputSSDeltaVLaunch`, both mapped by the
SELF-collision tranche (P8.4) and both shared with it verbatim.  So one file closed the pair.

**KEY EVIDENCE — `PXG_SCENE=softbodystack`:** a 0.6-edge FEM-volume cube dropped from base
y = 1.5 onto a 1.0-edge FEM-volume cube resting on a plane.

| run | result |
|---|---|
| lower cube PRESENT | **CAUGHT, min y = 0.982** (lower top 1.00), settles, rest \|dy\| ≈ 0.002 |
| `PXG_SS_NOLOWER=1` (control) | falls clean through to the plane, **min y = 0.0000** |

Both cubes come out of the SAME recipe — the new `buildFemVolumeCube`, extracted from the
softbodycloth scene — so their per-vertex inverse masses are identical by construction.  That is
deliberate: P8.8 lost a tranche to a scene whose sheet was 192× heavier per vertex than the cube
it landed on, where a correct solve pushed the lighter body aside.  Two cubes from one recipe
cannot reproduce that.  Self-collision is off on both, so nothing the self-collision chain does
can be mistaken for the between-bodies chain being alive.

**The kernel.**  One warp per SURFACE vertex of softbody0 walks softbody1's BV32 tetrahedron
tree and keeps the single deepest surviving vertex-vs-tet contact.  The traverser body is the
same one `softBodySelfCollisionBda` already carries — the two were transcribed from adjacent
structs in the same `.cu` — and four things differ, all of them consequences of the query vertex
and the tree belonging to DIFFERENT bodies:

| | self-collision | softbody-vs-softbody |
|---|---|---|
| shared-vertex skip | required | absent (no shared vertices) |
| rest-pose re-check + stress gate | required | absent (only meaningful within one body) |
| contact distance | one body's, DOUBLED | the SUM of the two shapes' |
| emitted pair ids | one body twice | two different bodies |

The inside/outside decision is the same arithmetic in both despite looking different in the two
structs: upstream's self-collision negates the normal before the branch and tests
`v.dot(normal) > 0`; upstream's SS leaves it un-negated and tests `v.dot(normal) < 0`, then picks
`v.dot(normal) < -0.707f ? v : -normal`.  Substituting `normal -> -normal` turns each into the
other, including the tie-break.  The port keeps the self-collision FORM, because that form is
the one already covered by fixtures.

**`blockIdx.z` IS NOT A SPARE DIMENSION.**  The engine launches grid
(SB_SBMIDPHASE = 1024, numTests, **2**) and `blockIdx.z` is a `flip` flag: z = 0 tests shape0's
vertices against shape1's tree, z = 1 the other way round.  Upstream says why — "we do
bidirectional collision (A to B and B to A) ... to ensure we don't miss contacts when we have a
soft body with an insufficiently tessellated tetrahedron mesh."  Dropping it halves the contact
set and the coarser mesh sinks into the finer one.  The bridge forwards gridDimX/Y/Z verbatim,
so `flip = gid.z` costs nothing — but it has to be there, and the fixture puts each direction's
geometry in its own region of space so a missing z shows up as a **missing contact**, not as an
ambiguous change in a count.

**Fixture** `softBodySimSoftBodySoftBodyMidphase` (55 fixtures in the file now).  Four contacts
survive every gate — three on the flip = 0 pass and one on the flip = 1 pass — and the
contact-distance pair `a8`/`a9` straddles the sum from both sides so that *every* single-value
or doubled-value reading of it is killed:

| reading of the contact distance | a8 (0.040 out) | a9 (0.055 out) |
|---|---|---|
| **sum, 0.05** | kept | dropped |
| ref0 alone, 0.03 | dropped | dropped |
| ref1 alone, 0.02 | dropped | dropped |
| 2 × ref0, 0.06 | kept | **kept** |
| 2 × ref1, 0.04 | dropped | dropped |

**Mutation sweep: 8 of 8 killed** (forced-fresh `.spv` rebuild each time) — drop the
`blockIdx.z` flip; contact distance from ref0 alone; contact distance doubled; the wildcard
filter querying tet 0 instead of `PX_MAX_NB_DEFORMABLE_VOLUME_TET`; the surface-tet hint gate
removed; the cross-lane argmin fixed at lane 0; `pairInd0` carrying the vertex index instead of
`mSurfaceVertToTetRemap`; the tree taken from softbody0 instead of softbody1.

### What the scene found: two flush cubes cannot stack, and that is geometry, not a bug

The first version of the scene stacked two IDENTICAL 1.0 cubes and the upper one sank straight
through — with the new kernel dispatching 23 times over 40 steps and the remap right behind it.
A probe on the SS constraint block (since reverted) read out what the contacts actually were:

```
contacts=8  cross=8 self=0  pen=[ 0.0375 .. 0.0396]  n.y = 1.0000     <- first touch, correct
contacts=18 cross=18 self=0 pen=[-0.0009 .. 0.0025]  n.y = 0.0029     <- and then this
```

`cross=8`/`self=0` says the new kernel is emitting genuine A↔B pairs, so the port was doing its
job.  The normals are the problem: they go **horizontal**.  Two identical cubes stacked flush
have COPLANAR SIDE FACES, and every vertex on the upper cube's bottom rim lands exactly on one.
The narrowphase is a vertex-versus-closest-tet-SURFACE-point test, so at a tie between the side
face (distance ~0) and the top face (distance ~0) it takes the side, and a horizontal contact
with |pen| < 0.001 can never hold a cube up.

Making the upper cube 0.6 wide puts its whole bottom rim strictly inside the lower cube's 1 × 1
top face, the top face becomes unambiguously nearest, and the stack holds.  This is the same
reason the softbodycloth scene's sheet is smaller than the cube it lands on — recorded here
because it is the second time a degenerate SCENE looked exactly like a dead kernel, and the
tell both times was a probe on live data rather than more reading of the shader.

**One measured caveat on the numbers.**  `softbodystack` and `clothselfcollide` vary in their
last printed digit run to run (0.9822 / 0.9820, rest |dy| 0.002299 / 0.002152; mean y 5.1105 /
5.1106) because the deformable solves scatter through real float atomics and the summation order
is not fixed.  `min y` is stable to four decimals across three consecutive runs and every
assertion bar sits far from the variation.

Manifest hand-edited 247 → 248.  Suite **674/674 in both memory modes**, rigid regression green,
and every deformable scene unchanged — `softbodyselfcollide` 0.2133, `softbodycloth` 0.8688,
`clothselfcollide` min y 5.0999, softbody floor matrix bit-identical at box **0.49628** / plane
**0.49753** / sphere **0.48436**.

## P8.8 — MILESTONE: a FEM sheet is CAUGHT BY A FEM CUBE (the softbody-vs-CLOTH chain, 6 of 6)

Softbody-vs-cloth needs six kernels and **all six are on the engine BDA ABI**.  The chain
runs end to end: midphase in both directions, contact generation for triangles and vertices,
constraint prepare and solve.

**`PXG_SCENE=softbodycloth` is GREEN.**  A FEM-surface sheet dropped from y = 2 is caught by a
FEM-volume cube resting on a plane and settles on it at min y = **0.8688** (rest |dy| =
0.000034); the control run with the cube removed falls to y = 0.  See "The scene" below — and in
particular the mass-ratio finding, which is the reason this took two tranches instead of one.

| # | kernel | state |
|---|---|---|
| 1 | `sb_clothMidphaseGeneratePairsLaunch` | **mapped** (step 1) |
| 2 | `sb_clothContactGenLaunch` | **mapped** (step 2) |
| 3 | `sb_clothVertMidphaseGeneratePairsLaunch` | **mapped** (step 3) |
| 4 | `sb_clothVertContactGenLaunch` | **mapped** (step 4) |
| 5 | `sb_clothContactPrepareLaunch` | **mapped** (step 5) |
| 6 | `sb_solveOutputSCDeltaVLaunch` | **mapped** (step 6) |

**Step 1 — the midphase.**  One warp per query tetrahedron walks the CLOTH's BV32 triangle
tree and keeps the four closest triangles.  The traverser core
(`TriangleLeafBoundMinMaxTraverser`) was already transcribed once as the descriptor-set kernel
`ctTetTriMid`, which carries the derivation and a randomized fixture; the new work is the
driver — the shape pair found by TYPE, the tree at `mTriMeshData + sizeof(uint4)`, the
surface-hint gate, the world→cloth-space transform, the summed contact distance, and the
`uint4(cmIdx, tetIdx, primIndex, 0)` write format.

One thing was **dropped rather than copied**: `npTetTriMid` keeps `satIntersect`'s internal
back-face cull because it stands in for both template instantiations.  This call site is the
`<false>` one, where the cull lives at the caller.  Carrying both made the outer gate provably
redundant, and a mutant deleting it survived.

**Step 2 — the contact generator.**  Each midphase pair becomes a GJK/EPA query between a
tetrahedron and a triangle on the P8.6 driver.  Both barycentric sets are computed and both
gate the contact; the 4-bit `mTetMeshSurfaceHint` mask rejects contacts through interior
faces; the filter is two-level, and its vertex level is an **AND over all three** triangle
vertices.  The write format is the opposite of the rigid path's — point = the CLOTH witness,
normal stored NEGATED, `pairInd0` = cloth and `pairInd1` = softbody — and the solve chain
downstream depends on that order.

**Both fixtures predict rather than calibrate.**  A tetrahedron whose bottom face is
horizontal at height g above a triangle directly below it separates by exactly g, because
`satIntersect`'s triangle-plane term is identically zero and every side plane contributes a
negative number.  The contact fixture pins the sign convention by derivation: the GJK direction
runs from cloud A to cloud B (softbody → cloth), which is what makes `pen` positive for a
separated pair and the stored `-direction` point from the cloth up at the softbody.

**Seven fixture blind spots, all found by mutation and all closed in the fixture:**

| what survived | why | the fix |
|---|---|---|
| tet/primitive field swap (both fixtures) | tet index == triangle index | triangles laid out in REVERSE |
| `primStartIndex` shift | the leaf started at primitive 0 | a dummy leading primitive |
| `primStartIndex` shift, again | the primitive it drops off the end was one the top-4 cap discarded anyway | the stack runs far-to-near |
| the MaxContacts = 4 cap | no tet saw more than one triangle | a stack of five under one tet |
| the front-face gate | see the dropped duplicate above | one gate, plus a back-facing triangle |
| the vertex-level AND | all three vertices were filtered, where AND and OR agree | one tet has exactly ONE vertex filtered |
| the tet-vs-whole-cloth filter | nothing exercised it | a fifth tet killed by it |
| a single-sided contact distance | 0.10 + 0.20, and the larger alone carried the test | 0.16 + 0.14, neither reaching the gap |

Also recorded because it cost time: one of my own mutations was **ineffective rather than the
code being wrong** — flipping the MIDDLE conjunction of the vertex filter to `||` leaves
`((A0 && A1) || A2)`, which for a single filtered vertex evaluates identically.  The first
conjunction is the one that discriminates, and it dies.

**Surviving mutants, stated rather than hidden:** the triangle's 1e-6 inner sphere and the box
inner sphere of P8.7 are the same family — both reach only EPA's ~1e-4 convergence epsilon,
which no contact test can observe.  Both barycentric gates are defensive: a GJK witness lies on
its own hull by construction, so its coordinates are inside the simplex and the
`[-1e-5, 1+1e-5]` window only catches a degenerate solve.  `nbPrimitives = (data >> 1) & 63`
cannot be told from `& 31` by any harness that walks a leaf with one warp.  And
`intersectBoxFullWarp` is traverser core: a mutation on it dies in `gpuslang.npTetTriMid`
(verified, not assumed).

**Step 3 — the vertex-side midphase.**  The opposite direction from step 1: a cloth VERTEX
queries the SOFTBODY's tetrahedron tree, AABB against AABB.  Both directions are needed,
because a tet-vs-triangle test misses a cloth vertex that has poked into a tet's interior
without any of its triangles coming near a tet face.  There is no top-N cap here — every
surviving primitive is written through `pushOntoStackFullWarp`: one atomicAdd of the
population count, each qualifying lane storing at its scan offset, a full stack dropping the
WRITE but not the COUNT (the engine sizes its next allocation from that number).

Two details that are easy to get wrong and are pinned by their own sub-runs: the query volume
is a **CUBE** of half-width `cDistance`, not a sphere; and the broadphase early-out **SHRINKS**
the cloth's bounds by `(n-1) * contactDistance` before testing them, because the broadphase has
already inflated them — reading that as an inflation turns a skip into a no-op.

Its ABI row is the trap of this step: the pack is **88 bytes, exactly like step 1's**, but not
the same layout. Step 1 has a `PxU32` at @64 followed by a pointer, so @68 is dead padding;
step 3 carries a real `PxReal` there. The row is what tells them apart, and getting it wrong
shifts every pointer after it.

Ten mutants killed; **six needed the geometry sharpened first**, each a fixture change rather
than a shader change: a degenerate tet whose flat box the query still MEETS (placed elsewhere
the AABB test rejects first and `isValidTet` is untestable); a query cube touching a box
boundary EXACTLY, which needs contact distances that are exact binary fractions
(0.0625 + 0.1875) so the sum lands on the boundary rather than a hair past it; a vertex
reaching in the NEGATIVE direction; a scrambled vertex-to-tet mapping; two ADJACENT tets so one
ballot has two lanes set and the scan prefix matters; and a deliberately undersized stack.

**A process note, because it cost a full suite run.**  Two unrelated fixtures broke during
step 3 — not from the port but from an unguarded whole-file string replacement in the test
file, which rewrote `n2 == 4u` in fixtures that had nothing to do with this work.  The suite
caught it (the assertion message still printed its own literal, "got 4, expected 4", against a
condition that had become 6).  Edits to this 8000-line multi-fixture file need anchors that
include surrounding context, not bare substrings.

**Step 4 — the vertex contact generator.**  The consumer of step 3's pairs, and the simplest
kernel of the chain: one thread per (cloth vertex, tetrahedron) pair, no warp cooperation, no
GJK.  A point-vs-tetrahedron closest-point query answers it outright.

Its two branches are **different contacts, not two spellings of one**.  A vertex INSIDE the
tetrahedron takes the winning surface face's normal and a NEGATIVE penetration, with **no
contact-distance test at all** — inside is a contact at any depth.  A vertex OUTSIDE is
rejected beyond the contact distance, then takes a positive penetration and the
closest-point direction when that agrees with the face normal past 0.707, the face normal
otherwise.  So the SIGN of `pen` carries the inside/outside distinction downstream, and both
branches store the same point: the witness on the tet's surface.

The filter is a **three-way OR**, not step 2's two-level AND-chain — a vertex is a single
element, so there is nothing to take a conjunction over.  And `barycentric0` is the constant
`(0, 0, 0, 1)`: the cloth side is one vertex, so upstream stores a degenerate marker rather
than computing anything.

`closestPtTetrahedron` and its three companions moved out of `softBodySelfCollisionBda` into
a new module, **`pxgSlangTetClosest`**, rather than being copied — the self-collision fixture
re-verifies them for free, which is the point of extracting.

Ten mutants killed.  Four needed the geometry sharpened: a vertex INSIDE but FARTHER from the
face than the contact distance (the only case that shows the inside branch skipping the test);
a witness on a face CORNER rather than in its interior, so the closest-point direction and the
face normal actually differ and the 0.707 threshold becomes visible (predicted exactly:
pen 0.206155, normal −0.24253/0.97014); a stack deliberately smaller than the pair count, so
"report what was FOUND" can be told from "report what fitted"; and a leading dummy cloth vertex
so the vertex and tetrahedron indices differ and their order in the contact info is observable.

**One of my own labels was wrong, not the shader.**  The first draft called two vertices at
positive y "outside"; the tetrahedron sits ABOVE its y = 0 face, so they were inside, and the
inside branch correctly reported them without a distance test.  The shader was consistent
throughout — the third vertex matched its prediction exactly on the first run — and the fix was
to the fixture's geometry and its comments.

**Step 5 — the constraint prepare.**  It consumes BOTH contact shapes out of one stream, and
`clothBC.w` is the discriminator: `0` means the cloth element is a TRIANGLE (step 2), non-zero
means a VERTEX (step 4) — which is exactly why step 4 writes the degenerate marker
`(0, 0, 0, 1)`.  The two paths take their friction and their effective mass from different
places: material table via `mMaterialIndices` and a barycentric-weighted sum over three
vertices in one case, `mDynamicFrictions` and a single inverse mass in the other.  So `triInd`
is a triangle index in one case and a vertex index in the other, out of the same field.

**The normal is negated a SECOND time here.**  Steps 2 and 4 store `-direction`; this kernel
negates again, so the constraint carries the original direction.  Both negations have to be
present — dropping either flips the constraint and pushes the bodies together.

Two things deliberately NOT ported: the `clothPos` / `softBodyPos` computed at softBody.cu:1128
are dead stores, never read, and their cloth half indexes `mTriangleVertexIndices` with what is
a VERTEX index in the vertex case — benign in CUDA, a fault waiting to happen through a BDA
pointer.  Omitting them is behaviour-preserving and safer.

Every fixture input is a small exact binary fraction, so the host reference and the shader
agree **bit for bit** rather than within a tolerance.

Eleven mutants killed; two needed the fixture sharpened.  `mMaterialIndices` is `PxU16*` on
BOTH descriptors, and my first draft read the softbody's as `PxU32*` — the fixture did not
notice, because at an EVEN index the two happen to yield the same number; the contact now names
an ODD tet index, where they differ.  And with only two contacts `workIndex & 31 == workIndex`
and `workIndex / 32 == 0`, so neither half of the block-of-32 SoA addressing was exercised; a
34-contact sub-run puts the last two in block 1.

**One transient, recorded rather than explained away.**  A single parallel (`ctest -j4`) run in
each memory mode failed once, immediately after a full forced-fresh rebuild; four subsequent
runs across both modes passed 674/674, the serial run passed, and the log had been overwritten
before it could be read, so the failing test is unknown.  This is the third one-off of this
shape in the project (after `gpuslang.scan` and `npRigidSumBatches`) and it is noted, not
diagnosed.

**Step 6 — the solve.**  It solves the constraints step 5 prepared and scatters the positional
corrections onto both bodies.  The solver body is shared between the two branches; they differ
only in how many cloth vertices the correction is spread over:

    relDelta  = (cloth side, barycentrically weighted) - (softbody side, ditto)
    CN        = pen + dot(relDelta, normal)
    dLambdaN  = max(-CN * denomInv, -lambdaN)
    commonVec = dLambdaN * normal  [+ the friction term]

then `+w * bc * commonVec` on the cloth and `-w * bc * commonVec` on the softbody —
**opposite signs**, which is the whole point.

**The clamp against `-lambdaN` is what stops a contact PULLING.**  For a separated contact
`-CN * denomInv` is negative, and the clamp holds the accumulated impulse at zero; without it
the solver drags the bodies together.  No penetrating contact can show this, so the fixture
carries one deliberately separated contact whose lambda must come out exactly 0.

**The vertex path is not the triangle path with bc = 1.**  Its cloth correction has NO
barycentric factor and its atomic is guarded by the inverse mass ALONE, where the triangle path
also demands `bc > 1e-6`.  Both asymmetries are upstream's.

Every scatter is a real float atomic (`VK_EXT_shader_atomic_float`) and adds **1 to `.w`** — a
contribution count the averaging pass downstream divides by, so dropping it changes every
delta that follows.

Eight mutants killed.  Two needed the fixture sharpened: the lambda clamp (the separated
contact above), and the `.w` count, which needed a vertex receiving contributions from more
than one contact.

**One mutant is NOT killable, and the arithmetic says why.**  A degenerate constraint
`continue`s rather than returning, which differs only when a thread owns more than one contact
— i.e. `nbIterationsPerBlock > 1`.  Under the engine's fixed 1024 × 256 grid, which this port
hard-codes as every kernel in the chain does, that needs more than 262144 contacts.  Dispatching
fewer blocks does not help: the grid constant is compiled in, so a one-block dispatch still
computes `nbIterationsPerBlock = 1` and simply leaves contacts 256 and up unvisited (measured).
`continue` is kept as the faithful transcription; the mutant survives every fixture that fits
in memory, and that is recorded rather than papered over.

Manifest hand-edited 241 → 247.  Suite **674/674 in both memory modes** after each step, rigid
regression **16/16**, and the deformable scenes unchanged — `softbodyselfcollide` still at
0.2133, `clothselfcollide` at 5.0999, and the softbody floor matrix bit-identical at box
**0.49628** / plane **0.49753** / sphere **0.48436**.

### The scene: `PXG_SCENE=softbodycloth` — and what it found

A FEM-surface sheet (0.6 x 0.6) dropped from y = 2 onto a FEM-volume cube of edge 1 resting on
a plane, with `PXG_SC_NOSOFTBODY=1` as the control.  The sheet is smaller than the cube's top
face, so "caught" and "missed" are a full unit apart.

**BOTH RUNS PASS.**  The sheet is caught by the cube and settles at min y = **0.8688** (mean
0.9642, rest |dy| = 0.000034 over the last 37 of 150 steps, peak 32 deformable-volume contacts);
the control, with the cube removed and nothing else changed, falls clean through to the plane at
y = 0.0000.  The softbody-vs-cloth chain runs end to end on live data.

That took one more tranche than it should have, because the scene reached that state only after
a wrong hypothesis was measured out of the way.  What follows is the record.

What it established, measured rather than assumed:

* the pair IS classified into `GPU_BUCKET_ID::eSoftbodyFemCloth` — probed directly in
  `cndCaseTetraMesh` (`compat = 1`, `isRigid1 = 0`);
* **the chain dispatches end to end.**  Over 40 steps:
  `sb_clothVertMidphaseGeneratePairsLaunch` 16 launches, `sb_clothVertContactGenLaunch` 16,
  `sb_fem_contact_remap_to_simLaunch` 16, `sb_clothContactPrepareLaunch` 40,
  `sb_solveOutputSCDeltaVLaunch` 360 (once per solver iteration).  All report OK.

### The contact-count read-back — the answer, and where the gap actually is

Two temporary probes (one in `testSDKSoftbodyCloth` after the vertex contact-gen, one around the
`sb_clothContactPrepareLaunch` dispatch, both since reverted) measured the chain on live data:

| stage | measured |
|---|---|
| vertex midphase pairs | **170 – 231 per frame** |
| vertex contact gen | **29 – 48 contacts per frame** |
| `sb_fem_contact_remap_to_simLaunch` | runs, 16 frames of 40 |
| prepare: count in / out | **44 → 44**, `mMaxContacts` = 1048576, so nothing is clamped away |
| prepare: constraint 0 | normal ≈ (0, 1, 0), `velMultiplier` ≈ 0.1 (NON-zero), friction 0.5, barycentric summing to 1 |
| prepare: penetration over frames | **+0.0923 → −0.0117 → −0.0403 → −0.0602** |
| solve | 540 launches per 60 steps |

**So the narrowphase emits contacts, the remap runs, and the constraint build is CORRECT on live
data.**  The normal points up, the penetration tracks the sheet sinking into the cube frame by
frame, and `velMultiplier` is non-zero — which matters, because a zero there is exactly what
makes the solve skip a contact.

And the sheet kept sinking.  From that I concluded: **"the gap is between the solve WRITING its
deltas and those deltas reaching the vertices."**  That conclusion was wrong, and the next
section is how it was disproved.

### The delta read-back — the chain was never dead, and the SCENE was the bug

A third temporary probe (in `PxgSoftBodyCore::solveSCContactsOutputDelta()`, immediately after
the `sb_solveOutputSCDeltaVLaunch` dispatch, since reverted) read `cloth.mDeltaPos` back on
every solver iteration.  It killed the hypothesis on the first run:

| measured, per solver iteration | value |
|---|---|
| non-zero entries in `cloth.mDeltaPos` (of 25 verts) | **18 – 25** |
| longest correction | **0.005 – 0.011** |
| `lambdaN` over the nine iterations of one step | **0.0047 → 0.0197, monotonic** |
| `cloth_applyExternalDeltasLaunch` (the consumer) | **2320 launches over 40 steps** |
| cloth max y within one step | **0.96995 → 0.98867** — the correction visibly moves vertices |

The solve wrote, the apply pass consumed, and the vertices moved.  The deltas had been reaching
the vertices the whole time.

Two more numbers from the same probe named the real cause in one line:

```
clothW = 1        softbody sim vertex W = 192
```

Those are the two inverse masses the constraint weighs against each other.  The cube's
simulation vertices come out of the cooker at **invMass 192** — about 0.14 kg for the whole cube
— and the scene had handed the sheet **invMass 1**, i.e. 1 kg per vertex, **25 kg of cloth**.
The constraint splits its correction as `w_i / (w_cloth + Σ bc² w_softbody)`, measured here as
`velMultiplier` ≈ 0.097, so roughly **90 % of every impulse went into shoving the CUBE out of
the way and 10 % into holding the sheet**.  The solve was doing exactly the right thing with the
masses it was given: a 25 kg sheet dropped on 0.14 kg of foam goes through it.

Raising the sheet's inverse mass (new `PXG_SC_CLOTHW`, default **200**) catches it immediately:

| cloth invMass | outcome after 150 steps |
|---|---|
| 1 (the old scene) | falls through, min y **0.0000** |
| 50 | caught, min y **0.8647**, rest \|dy\| 0.000041 |
| 200 (new default) | caught, min y **0.8688**, rest \|dy\| 0.000034 |

**No shader, no ABI row and no engine source changed to make this pass** — the only edit was the
scene's own inverse mass.  Six kernels, six fixtures, and the thing that broke the integration
was a constant in the test harness.

**Two hypotheses were measured out of the way before this one, and both are worth recording as
wrong.**  First: "the impact is too fast for one frame's impulse budget."  Dropping the sheet
from 1.02 — two centimetres above the cube, arriving essentially at rest — sank just as surely,
which kills the speed explanation outright.  Second: "the deltas never reach the vertices", the
conclusion the previous section reached from correct measurements.  Every measurement behind it
was right; the inference from them was not.  A constraint that produces a small correction is
not the same as a constraint that produces none, and only the read-back could tell the two
apart.

**One correction to an earlier reading in this session.**  A 3-step launch trace showed no
vertex-path kernels and I took that to mean the narrowphase function was never reached.  It was
simply too short: the sheet had not yet fallen close enough for the broadphase.  At 25 steps the
kernels appear, at 60 they run throughout.  A trace that finds nothing is evidence about the
trace's window before it is evidence about the code.

**AND A FINDING THAT CHANGES WHAT STEPS 1 AND 2 ARE WORTH.**  Upstream hard-codes
`supportClothTriangleCollision = false` (`PxgNarrowphaseCore.cpp:4529`), so the TRIANGLE half of
the chain — `sb_clothMidphaseGeneratePairsLaunch` and `sb_clothContactGenLaunch`, P8.8 steps 1
and 2 — is **dead code in this engine version and can never run as shipped**.  Only the VERTEX
half is reachable.  That does not make those two ports wrong, and both are fixture-verified, but
no scene can exercise them without changing engine source.  Worth knowing before spending more
on that side.

## P8.7 — MILESTONE: the softbody settles on the ground BOX that was always there

`sb_primitiveContactGenLaunch` now handles **every** rigid type the CUDA does. `eBOX` and
`eCONVEXMESH` share one branch — to this code a box is a cooked eight-vertex hull — on top of
the GJK/EPA driver P8.6 extracted.

**The hull decode** (`softbodyPrimitives.cu:470-568`). Skip one `float4`
(`convexPtr = hullPtr + sizeof(float4)`), read the `uint4` header, take the vertex count out of
bits 8..15 (`getNbVerts = u16High(u32Low)`), read `float4 extents`, then walk the vertex array at
`+32`. Every vertex goes through `vertex2Shape` — rotate into the mesh-scale frame, scale
componentwise, rotate back — and then the rigid transform. `hullOrMeshOffset` carries
`size_t hullOrMeshPtr` verbatim and under this bridge a `CUdeviceptr` **is** a `VkDeviceAddress`,
so the two halves reassemble into a BDA with no translation.

**A hull folds in no radius.** That is sphere/capsule policy: their cloud is a point or a segment
and the radius rides in the contact distance, to be subtracted back out as `penOffset`. A hull
carries its thickness in its vertices, so `contactDistance = cDistance` unmodified and
`radius = 0`. Applying the sphere rule here is wrong by exactly one radius, in both directions.

`vertex2Shape` gets a **literal transcription of `PxQuat::rotateInv`** rather than the
algebraically-equivalent form the plane path has always used: it runs on every hull vertex and its
result picks a support vertex, where a last-bit difference is not a last-bit difference.

**The fixture predicts; it does not calibrate.** For a separated pair whose nearest feature is one
hull corner against the interior of the tet's `x = 0` face, the answer is the support function and
nothing else — the largest-x world corner, its projection onto `x = 0`, normal `+x`, separation
`-corner.x`. That is computed host-side before the dispatch and matched to 1e-3. Four pairs:

| pair | shape | configuration | expected | measured |
|---|---|---|---|---|
| 0 | BOX | generic rigid rotation, centre (-0.45, 0.6, 0.6) | gap 0.154364, witness vertex 0 | 0.154364 |
| 1 | BOX | identity, centre (-0.70, 0.6, 0.6) | gap 0.4 > cDistance 0.3 → **no contact** | none |
| 2 | CONVEXMESH | non-identity mesh-scale rotation, centre (-0.53, 0.6, 0.6) | gap 0.196020, witness vertex 7 | 0.196020 |
| 3 | BOX | axis-aligned, 0.05 through the `x = 0` face | depth **-0.05** via EPA | -0.050000 |

Pair 1 sits in the band between the true contact distance (0.3) and the wrongly-inflated one
(0.3 + scale.y = 0.5), so the no-inflation rule is observable rather than assumed. Pair 3 is the
only configuration that drives a hull cloud through EPA at all; only its depth is asserted, because
the overlap is a slab and any `(y, z)` in it is a valid witness pair.

**Two fixture blind spots, found by mutation and fixed in the fixture.** The first draft used a
unit cube and a mutant that shifted the vertex-array base by one `float4` **survived**: a cube is
symmetric enough that dropping a corner leaves the support in `+x` untouched, and reading the
extents `float4` as a vertex re-adds a corner the cube already had. The hull is now irregular and
ordered so pair 0's supporting vertex is index **0** and pair 2's is index **7** — the two ends of
the array, so a base one `float4` high loses one and one `float4` low (or a short vertex count)
loses the other. The fixture asserts both winner indices before it compares anything.

Killed: vertex base `+16` and `+48`, the header bit-field shift, the skipped `float4`, the
mesh-scale rotation, contact-distance inflation, penetration radius offset.

**Surviving and documented rather than papered over:** `inSphereRadius0` — `extents.w` times the
smallest axis scale for a box, times 1 for a convex mesh — reaches nothing but EPA's convergence
epsilon, `0.5 * (r0 + r1) * 2.25e-4`, order 1e-4. Removing the box scale factor entirely, or taking
the minimum over the wrong two axes, leaves every number the fixture can observe unchanged, pair 3
included. It is transcribed from `softbodyPrimitives.cu:513-519` and carried on trust; a test that
could see it would be an EPA convergence test, not a contact test.

### What the port invalidated, and how it reported itself

**The plane unit fixture SIGSEGV'd.** Its negative control was a box with *no hull at all* — valid
while every non-plane type was ignored, a null-pointer BDA load now. Its own comment had said the
GJK tranche "has to change this test consciously rather than silently"; it did, by faulting the
device. The control is now a real cooked cube parked 100 units away, still contributing nothing,
on geometric grounds this time.

**The scene had two invisible rigids under the softbody.** `buildSoftBodyScene` has always called
`addGroundBox` (half 50 x 0.5 x 50 at y = -0.5, top face exactly `y = 0`) and spawned a 1x1x1
dynamic box at the **origin** — described in the code as "far below", which it was only in the
sense that nothing could collide with it. Both became real. The falling cube started meeting the
dynamic box on the way down (contacts from step 41 instead of 49) and resting partly on top of it.

The dynamic body exists for one reason — `rigidSumInternalContactAndJointBatches2` null-derefs with
zero dynamic rigids — and that reason does not care where it stands, so it stands at `x = 10`. The
same two moves were needed in `softbodyselfcollide`, where the dynamic box sat inside the bottom
slab.

**The floor is now selected, not accumulated:** `PXG_SB_FLOOR=box|plane|sphere|none`, default
**box**. The alternatives replace the ground box (which sinks to `y = -50.5` to keep the dynamic
body fed) rather than joining it — a plane coplanar with the box's top face and a sphere tangent to
it would all generate contacts at the same height and the run would stop measuring the shape it
names. `PXG_SB_PLANE=1` and `PXG_SB_SPHERE=1` still work and mean the same floors.

The **free fall moved to `PXG_SB_FLOOR=none`**, where it is honest. It was only ever a free fall
because the floor was invisible.

**KEY EVIDENCE — the same cube, the same solve chain, four floors (`PXG_SCENE=softbody
PXG_SOLVER=pgs PXG_GPU_STEP=1`):**

| floor | contacts | settled mean y | rest \|dy\| |
|---|---|---|---|
| **box** (default, hull branch) | 16 | **0.49628** | 0.000899 |
| plane (plane branch) | 32 | 0.49753 | 0.000002 |
| sphere (hull branch) | 16 | 0.48436 | 0.000575 |
| none | 0 | free fall to -0.9867, monotonic | — |

The plane and sphere numbers are **bit-identical to their P8.5/P8.6 baselines**, which is the
control: the hull branch added a floor without perturbing the two that already worked.

**The settle bar now tracks the BRANCH, not the shape.** The two branches produce different contact
*densities*: the plane branch writes one contact **per penetrating corner**, up to four per
tetrahedron; the GJK/EPA branch writes one witness pair for the whole tet-vs-hull query. On this
cube that is 32 against a plane and 16 against a box, and the coarser set converges more slowly —
upstream's behaviour, not a port artefact. The plane keeps the tight 1e-4 bar so its own convergence
cannot quietly degrade to the GJK path's; box and sphere get 2e-3, far above both measured values
and far below anything still falling.

Suite **674/674 in both memory modes** (forced-fresh), rigid regression **16/16**, all deformable
and particle scenes green, `softbodyselfcollide` back to its recorded 0.2133 with the control run
still falling clean through to 0.0000.

## P8.6 — GJK/EPA: a softbody settles on a SPHERE

The softbody-vs-rigid contact path no longer stops at `ePLANE`. `sb_primitiveContactGenLaunch`
now handles **eSPHERE** and **eCAPSULE** through the real GJK + EPA driver; `eBOX` and
`eCONVEXMESH` still emit nothing, and need only the convex-hull decode on top of the same
machinery.

### Three modules, because squawk::gjk is one CUDA function

| module | from | was |
|---|---|---|
| `pxgSlangGjk` | `gjk.cuh:235-418` | inside `npGjk.slang` |
| `pxgSlangEpa` | `epa.cuh` | inside `npEpa.slang` |
| `pxgSlangTetCollide` | `convexNpCommon.h:212` | not ported |

`squawk::gjk` is called by every deformable-vs-convex path in the narrowphase. That is the
case the port's sharing rule names — one CUDA function used by both, share it or the copies
rot — and five hundred lines of simplex bookkeeping is the clearest instance of it. Nothing
but the vertex-cloud plumbing moved: `gVA[vbase + i]` became `a.v[i]` on a `GjkCloud` the
caller supplies. The leaf shaders are now their descriptor-set ABI plus a call, and their
unchanged ctests are what pin the extraction.

### The radius is not in the cloud

A sphere's cloud is its CENTRE, a capsule's its two segment endpoints. The radius is folded
into GJK's contact distance and subtracted back out of the penetration as `penOffset` — that
is how upstream keeps GJK on point sets rather than inflated hulls. Note also that
`scale.x` is a sphere's radius but a capsule's HALF-HEIGHT, while the radius used for
`penOffset` is `scale.y` for both. Upstream's asymmetry, transcribed rather than tidied.

### The money shot

Same cube, same solve chain, same thresholds — only the rigid's type changes:

| floor | cube settles at | rest \|dy\| |
|---|---|---|
| `PXG_SB_PLANE=1` | 0.4975 | 0.000002 |
| `PXG_SB_SPHERE=1` (radius 5 at y = -5) | **0.4844** | 0.000575 |

The sphere floor is the first non-plane rigid a softbody has ever touched in this port. The
convergence bar is shape-dependent and the two measurements say why: a sphere of radius 5 is
not flat under a cube of edge 1, so the contact set keeps shifting slightly as the cube beds
into the curvature. The plane keeps its tighter bar so its convergence cannot quietly degrade
to the sphere's.

### Three mistakes worth keeping

1. **A shadowed `out` parameter.** The mechanical extraction left `EpaOut o;` as a local
   inside `epaRun`, where `o` was now the out parameter — the caller's result was never
   written. `npEpa_driver` caught it, but only in the full suite: my targeted re-run used the
   regex `ctGjk|ctEpa`, which matches the shader-level targets and NOT the fixture suite
   `gpuslang.npEpa`.
2. **A fixture calibrated against broken output.** The `tetPrimitivesCollide2` fixture was
   written while EPA was silently returning nothing. It "passed" on a penetration depth of
   -0.2309401, which I explained as the unit tet's slant face. The explanation was plausible
   and wrong; the real depth is +0.2, the shallowest exit through one of three tied
   coordinate planes. **A number has to be predictable from the geometry BEFORE it is
   compared**, or the fixture pins nothing.
3. **`computeInnerSphere` is not the inscribed sphere.** Upstream takes the CENTROID and the
   smallest distance to the four face planes. I derived an incentre formula first; it gives a
   different number for any non-regular tetrahedron, and the value feeds EPA's epsilon.

A fourth, in the OLD scaffolding: the contact-gen kernel strides by
`SB_CG_GRID * MIDPHASE_WARPS_PER_BLOCK` = 1024, so a one-group dispatch covers only the first
TWO pairs. The plane fixture has exactly two and never noticed; the new one has three and its
third silently vanished until the fixture dispatched the engine's 512 blocks.

### Verification

| gate | result |
|---|---|
| GJK/EPA leaf tests | 14 / 14, including the HLSL cs_6_0 variants |
| target ctest (`gpuslang.softBodySim`) | pass, 47 fixtures |
| full suite, forced-fresh rebuild of every `.spv` | **674 / 674** |
| `PXG_SCENE=softbody` (no floor / plane / sphere) | all three green |

One `gpuslang.scan` SEGFAULT occurred on one forced-fresh run and did not reproduce (six
isolated runs clean, full suite green on re-run, 14 GB free). `scan` is a prefix-sum shader
untouched by this work. Same class as the `npRigidSumBatches` one-off already recorded above,
and it should be re-checked if either recurs.

## P8.5 — the co-rotated stress field, and the plasticity block settled

`sb_calculateStressLaunch` is mapped: one thread per COLLISION-mesh tetrahedron computes the
co-rotated Cauchy stress tensor and the scalar coefficient that goes with it. ABI row
`{8,8,8}`, 24-byte blob, 2D grid (x = tetrahedron blocks, y = the active bodies). Manifest
**MAPPED 240 -> 241**; softbody **30 -> 31 of 56**. It lives in `softBodySolveBda` because it
shares the flat-`PxMat33` rest-pose read and the descriptor walk with the rotation extractors
already there.

### Why this kernel, and why NOT the other three

The scalar coefficient is the input to the SELF-COLLISION stress gate ported in P8.4 tranche
a — and this kernel's own launch gate runs it when `eCOMPUTE_STRESS_TENSOR` is set **or**
self-collision is enabled. The field exists *for* that gate. Until it was mapped,
`mTetraStressCoefficient` held whatever the allocation contained, which is why
`softbodyselfcollide` had to set a tolerance no stress could exceed.

The three plasticity kernels are **deliberately** left unmapped, and the manifest now says so
at each entry rather than leaving a bare "unmapped":

| kernel | why not |
|---|---|
| `sb_plasticDeformLaunch` | body gated on a hard-coded `if (false)` (softBodyGM.cu:2844) |
| `sb_plasticDeformLaunch2` | same hard-coded `if (false)` (softBodyGM.cu:3013) |
| `sb_initPlasticDeformLaunch` | no call site anywhere in PhysX |

Porting them would map empty kernels and move the count without moving anything else. The
count is supposed to mean something.

### Mutation: 18 of 18

Two needed the fixture rebuilt before they died, and both are the same species of blind spot:
a fixture whose inputs are too symmetric to distinguish two orderings.

- with every rest pose an identity, `P * Qinv` and `Qinv * P` agree, so a **swapped
  deformation gradient** passed. Tet 1 now carries a non-symmetric shear;
- with one active body on a 1x1 grid, `gid.x` and `gid.y` are both 0, so the **body selector
  and the tetrahedron block** could be swapped. A second active body, sharing every input and
  differing only in where its results go, separates them.

The three fixture tets are chosen so the coefficient's definition is pinned by shape, not
only by the reference: tet 0 is undeformed (zero strain, zero coefficient), tet 1 is stretched
under a 30-degree rotation (the co-rotation must remove it), and tet 2 is stretched
**uniformly** — its diagonal differences cancel, so its coefficient is zero while its sigma is
not. A coefficient built from the strain magnitude instead of the diagonal differences passes
tet 0 and fails tet 2.

### The end-to-end evidence

`PXG_SCENE=softbodyselfcollide` now exposes the tolerance as `PXG_SB_STRESSTOL`, and the
three-way result is what makes the stress field observable outside a fixture:

| run | top slab min y |
|---|---|
| self-collision ON, gate off (default) | **0.2133** |
| self-collision ON, tolerance 0.001 | **0.0000** |
| self-collision OFF | **0.0000** |

A tight tolerance drops *every* self-contact, which can only happen if the coefficient carries
real values — the gate and the field are wired to each other through the engine, not through a
test harness. The default run is unchanged from before this kernel landed, byte for byte,
which is the other half of the statement: mapping it perturbed nothing.

### Verification

| gate | result |
|---|---|
| target ctest (`gpuslang.softBodySim`) | pass, 46 fixtures |
| mutation, forced-fresh `.spv` | 18 killed of 18 |
| full suite, default memory mode | **674 / 674**, 272 s |
| full suite, `PXG_DEVICE_V2=1` | **674 / 674**, 362 s |
| rigid regression, 8 scenes x 2 solvers | **16 / 16** |
| all seven deformable scenes + the self-collision control | unchanged, green |

## P8.4 tranche b — the self-collision chain closes, and a null pointer that hid in it

`sb_fem_contact_remap_to_simLaunch`, `sb_softbodyContactPrepareLaunch` and
`sb_solveOutputSSDeltaVLaunch{,TGS}` are mapped, so the contacts tranche a's midphase
produces are remapped to the simulation mesh, turned into constraints and solved. Manifest
**MAPPED 236 -> 240**; softbody **26 -> 30 of 56**.

| kernel | ABI row | blob |
|---|---|---|
| `sb_fem_contact_remap_to_simLaunch` | `{8,8,8,8,8,8,4}` | 56 B |
| `sb_softbodyContactPrepareLaunch` | `{8,8,8,8,8,8,8,8,8,4}` | 80 B |
| `sb_solveOutputSSDeltaVLaunch{,TGS}` | `{8,8,8,8,4,4,8,8}` | 56 B |

The remap joins its sibling in `softBodyContactRemapBda` because it shares every helper and
both llvmpipe workarounds; it adds a second y-grid (which half of the pair gets remapped),
an `isValidTet` gate and a `hasTet` branch that neutralises a contact with no valid
candidate. The prepare and solve are a new file: the symmetric twins of the
rigid-vs-softbody pair from P8.2 tranche d/e, where the same impulse goes into both delta
buffers with opposite signs and each side scales it by its own inverse masses.

### The remap's grid had to be decided, not transcribed

This kernel is launched from THREE call sites with THREE block shapes — (8192, 4) for
softbody-vs-softbody, (1024, 16) for self-collision and for cloth-vs-softbody. CUDA copes
because its stride is `gridDim.x * blockDim.y` and blockDim travels with the launch; the
bridge drops blockDim, so one SPIR-V entry point cannot carry three strides. Getting it
wrong is not a slowdown: the remap is **not idempotent** — a second visit reads an
already-remapped SIM index as though it were a collision index. So the port pins one grid
and refuses the groups above it. Coverage stays exact for every call site; only the
8192-block case loses parallelism it cannot use here.

### Mutation: 37 of 37

Four mutants needed the fixture sharpened before they died, and each gap is worth naming
because none was visible by inspection:

- `mSimVelocity_InvMass` and `mSimPosition_InvMass` carry the same inverse mass in the real
  engine, so a fixture whose two `.w` columns agree cannot tell that CUDA's PGS path takes
  its inverse masses from the **velocity** read. The columns now differ;
- `updateTetraPosDelta`'s `|bary| > 1e-6` gate needs a contact that actually weighs a corner
  at zero — a contact on a tet FACE;
- `deltaF`'s clamp at `-appliedForce.x` needs a **separating** contact with an existing
  force, or it never bites;
- `gtid.x & 31` and `workIndex & 31` are provably the same value here (every term of
  workIndex is either the thread index or a multiple of `blockDim.x` = 256). The
  CUDA-literal form is kept and the equivalence documented, so that mutant is expected to
  survive rather than left unnoticed.

### The bug 37 green mutants could not see

The first live self-collision scene SIGSEGVed inside the tranche-a midphase on frame one.
Root cause: **`mFilteringPairs` is NULL whenever the body has no filter pairs**.
`PxgSimulationCore` only assigns it inside its dirty-filter branch
(PxgSimulationCore.cpp:1446), so a body that never registered a filter arrives with a null
pointer and `mNumFilterPairs == 0` — the common case.

`findFilter`'s binary search is zero-trip for an empty list, which is exactly why this read
as safe. It is not, on llvmpipe: the first iteration's load is speculated, faults at address
0, and the value substituted **differs between lanes**. `findFilter` then stops being
warp-uniform, the caller's `continue` stops being group-uniform, and the workgroup barrier
after it is reached by only part of the group. That is why the crash moved with the
groupshared layout — `STACK_SIZE` 192 and 193 behaved differently — and vanished under
`PXG_SYNC_EACH`: the fault is a **barrier divergence, not a bad store**.

Four plausible hypotheses died on that ambiguity first (a stack overflow, a bad
`maxContacts`, the stress read, the write bound), each "confirmed" by a probe that merely
perturbed scheduling. What ended it was refusing to probe further and printing the
descriptor from the engine side: every other pointer was valid and `mFilteringPairs` was
`(nil)`. **The lesson is the method, not the pointer**: in this port a probe that changes
code shape is not evidence, and the cheapest real measurement beats the fourth clever guess.

The fix is the shape `softBodyContactRemapBda` already spells out for `prevContactCount`:
SELECT THE POINTER, NOT THE LOADED VALUE. A guard cannot work — the branch is already there
and gets if-converted — so the faulting ADDRESS is removed, substituting a base that is
never null and forcing the loop bound to 0 so the result stays what CUDA computes for an
empty list. The fixture had always passed a valid pairs buffer; it now dispatches the same
case with `mFilteringPairs` zeroed and requires the two filtered vertices to come back.

### PXG_SCENE=softbodyselfcollide

The scene that drives the chain end to end, and the one that found the null pointer. A
bracket — a cell grid with the middle carved out, leaving a bottom slab on a static plane, a
top slab cantilevered over it, and a connector column making it one body. The two slabs are
six cells apart along the tet graph at the free end, so their contact is a genuine self
collision rather than an adjacency the rest-pose gate would drop.

| run | top slab min y | rest \|dy\| |
|---|---|---|
| self-collision **ON** | 0.7500 -> **0.2133** | 0.003299 |
| self-collision **OFF** (`PXG_SB_NOSELF=1`) | 0.7500 -> **0.0000** | 0.000000 |

The bottom slab's surface is at y = 0.25 and the caught slab settles slightly below it —
not a failure to catch, but a soft body compressing under its own weight with the contact
resolved at a rest offset, the same reason `clothdrape` settles ON the contact rest distance
rather than exactly on the plane. The bar is 0.15, far from both numbers. The control half
is what makes the ON half mean anything: a bracket that simply never sagged that far would
pass the ON assertion on its own.

`PXG_SCENE=softbody` keeps `eDISABLE_SELF_COLLISION` set and its recorded free-fall numbers
unchanged.

### Verification

| gate | result |
|---|---|
| target ctest (`gpuslang.softBodySim`) | pass, 45 fixtures |
| mutation, forced-fresh `.spv` | 37 killed of 37 (tranche b) + the null-pairs mutant |
| full suite, default memory mode | **674 / 674**, 198 s |
| full suite, `PXG_DEVICE_V2=1` | **674 / 674**, 246 s |
| rigid regression, 8 scenes x 2 solvers | **16 / 16** |
| `PXG_SCENE=cloth` / `clothhang` / `clothdrape` | unchanged |
| `PXG_SCENE=clothselfcollide` | catches at min y = 5.0999 |
| `PXG_SCENE=softbody` / `particle` | unchanged |
| `PXG_SCENE=softbodyselfcollide` | catches at min y = 0.2133, settles |
| ...with `PXG_SB_NOSELF=1` | falls to 0.0000 |

**A note on the mutation harness**: it restores the pristine shader but does NOT rebuild, so
the last mutant's `.spv` stays on disk. One full-suite run was spent chasing a `softBodySim`
SEGFAULT that was simply the previous mutant still installed. Delete the `.spv` and rebuild
at the end of every sweep, the same forced-fresh rule the sweep itself uses.

## P8.4 tranche a — the softbody SELF-collision midphase

`sb_selfCollisionMidphaseGeneratePairsLaunch` is mapped. One warp per SURFACE vertex walks
its own softbody's BV32 tetrahedron BVH, keeps the single deepest surviving
vertex-vs-tetrahedron contact and appends it through an inlined `PxgSoftBodyContactWriter`.
Manifest **MAPPED 235 -> 236**; softbody **25 -> 26 of 56**.

| kernel | ABI row | blob |
|---|---|---|
| `sb_selfCollisionMidphaseGeneratePairsLaunch` | `{8,8,8,56}` | 80 B |

The traverser BODY is a lift of `npSbSelf.slang` (`ctSbSelfQuery`), the unit-tested
descriptor-set port of `SbMeshTreeTraverser` — the same relationship `softBodyContactRemapBda`
has to `npSbRemap`. The BV32 walk, the groupshared stack and the ballot/scan lowering come
from `softBodyContactGenBda` unchanged. What is new is the DRIVER.

### The grid: 2D *and* grid-striding, which no earlier kernel combined

    grid (SB_SBMIDPHASE = 1024, nbActiveSelfCollisionSoftbodies, 1), block (32, 2, 1)
    globalWarpIdx = threadIdx.y + blockIdx.x * blockDim.y      // [0, 2048)
    NbWarps       = blockDim.y * gridDim.x                     // 2048
    for (vertIdx = globalWarpIdx; vertIdx < numVerts; vertIdx += NbWarps)

The x-grid is FIXED at 1024 regardless of the vertex count, so the kernel strides. The bridge
forwards gridDimY (`femClothSelfCollisionBda` already relies on that, so `gid.y` is not new),
but it drops blockDim — hence the two warps per block are walked serially in a group-uniform
`sub` loop, and the stride stays the CUDA constant 2048 rather than anything derived from the
dispatch. A stride taken from the dispatch instead visits most vertices 1024 times; the
fixture dispatches the real engine grid so that mutant shows up as an exploded contact count.

`finalizeFullWarp`'s `minIndex(bestDist, FULL_MASK, sep)` lowers to a groupshared `float[32]`
plus a strict-`<` scan from lane 0. `minIndex` returns `lowestSetIndex(ballot(minV == v))`,
i.e. the LOWEST lane holding the minimum, and a strict-`<` scan picks the same lane — which
matters, because two tetrahedra at the same distance must not both write.

### Mutation: 18 killed of 20, and the two survivors are PROVEN redundant

The fixture drives the engine grid and aims one query vertex at each gate: the surface-vertex
hint, the surface-TET hint, `isValidTet`, the per-pair and whole-body filter lookups, the
rest-pose re-check, the stress tolerance, the cross-lane argmin, the `contactDistance * 2`,
the outside branch's 0.707 normal replacement, the `maxContacts` bound and the zeroing of
`pairInd0`'s high word. Two mutants survive, and neither is an untested line:

1. **the shared-vertex skip** (`tetIdx1.x == vertIdx || ...`). A vertex that is a corner of
   the candidate tet is at distance exactly **0** from that tet in the REST pose as well —
   `restP` IS one of the rest corners, the same array entry. The rest gate demands
   `!inside2 && newSqDist > filterDistanceSq`, and for a corner `inside2` is true (all four
   plane products are zero or a square) AND `newSqDist` is 0, which is never strictly greater
   than a non-negative filter distance. So the rest gate drops the contact whatever the skip
   does — unconditionally, for every mesh. Deleting the skip AND the rest gate together makes
   the vertex appear (4 contacts -> 6), which is what pins the diagnosis rather than leaving
   it an argument. Upstream's skip is a fast path, not a semantic gate.
2. **the outside branch's `dist < distance`**. Identical to the final `bestDist < distance`,
   and a rejected candidate can never win an argmin over smaller values. Deleting both
   together lets ten contacts through where four are correct.

Three fixture gaps were found by mutation rather than by inspection, and each needed a new
query vertex or a reordering:

- a FLAT tet makes every one of `isPointOutsideOfPlane4`'s four plane products identically
  zero, so the predicate returns true for any point — including in the rest pose, where it
  reads as "already penetrating at rest". With the rest tet flat too, the rest gate silently
  did `isValidTet`'s job. `isValidTet` is checked on the WORLD tet only, so giving the
  degenerate tet a real REST volume separates them;
- the deep and shallow candidate tets share one BV32 leaf so they land in different lanes.
  With the deep one at primitive 0 — lane 0 — "no argmin at all" produced the right answer by
  accident. Swapping them puts the winner in lane 1;
- a query vertex whose separation is PARALLEL to the winning face normal cannot see the
  `dot(normal, v) > 0.707 -> normal = v` replacement. A vertex just outside a tet EDGE, where
  the face normal is (0,0,-1) and the separation is (0,-0.217,-0.976), can.

### What this does NOT yet do

The rest of the self-collision chain — `sb_fem_contact_remap_to_simLaunch`,
`sb_softbodyContactPrepareLaunch` and `sb_solveOutputSSDeltaVLaunch{,TGS}` — is still
unmapped, so the contacts this kernel emits are counted and stored but never remapped to the
simulation mesh and never solved. That is a safe intermediate state (an unmapped kernel is a
no-op, so the contacts are simply ignored) and it is why no scene moves, but a softbody still
passes through itself until that chain lands.

No sdktest scene reached this kernel at the time of writing: `PXG_SCENE=softbody` sets
`eDISABLE_SELF_COLLISION`, so the midphase was never dispatched. Tranche b builds the
missing scene (`PXG_SCENE=softbodyselfcollide`), and the first frame of it found a NULL
`mFilteringPairs` that took the engine down — see that section. The mutation score below is
therefore 18 of 20 against the fixtures **as they stood**, and one more mutant exists that
none of them could reach.

### Verification

| gate | result |
|---|---|
| target ctest (`gpuslang.softBodySim`) | pass, 42 fixtures |
| mutation, forced-fresh `.spv` | 18 killed of 20, both survivors proven redundant above |
| full suite, default memory mode | **674 / 674**, 198 s |
| full suite, `PXG_DEVICE_V2=1` | **674 / 674**, 391 s |
| rigid regression, 8 scenes x 2 solvers | **16 / 16** |
| `PXG_SCENE=cloth` | dropped 4.9867 over 1.00 s |
| `PXG_SCENE=clothhang` | pinned row max \|dev\| 0.00000, rest \|dy\| 0.000000 |
| `PXG_SCENE=clothdrape` | settles, rest \|dy\| = 0.000000 |
| `PXG_SCENE=clothselfcollide` | catches at min y = 5.0999 (5.1000 expected) |
| `PXG_SCENE=softbody` | free fall 4.98675 vs 4.98675 expected, skin gap 0.00000 |
| `PXG_SCENE=particle` | dropped 19.7835 over 2.00 s |

Every scene is byte-for-byte what P8.3 recorded. It has to be: nothing dispatches this kernel
yet (see below), so a moved scene would mean the tranche had touched something it should not.

**A note on running the scenes**: they need `PXG_SOLVER=pgs` as well as `PXG_GPU_STEP=1`.
Without the solver variable the deformable scenes construct and then report a zero drop,
which looks exactly like a regression and is not one.

The recon for that chain is done, so the next tranche starts from data rather than from
re-reading the CUDA:

| kernel | CUDA | launch | ABI row | blob |
|---|---|---|---|---|
| `sb_fem_contact_remap_to_simLaunch` | softbodyPrimitives.cu:2277 | NarrowphaseCore.cpp:4166 | `{8,8,8,8,8,8,4}` | 56 B |
| `sb_softbodyContactPrepareLaunch` | softBody.cu:940 | SoftBodyCore.cpp:2547 | `{8,8,8,8,8,8,8,8,8,4}` | 80 B |
| `sb_solveOutputSSDeltaVLaunch{,TGS}` | softBody.cu:1424 / :1691 | SoftBodyCore.cpp:1636 | `{8,8,8,8,4,4,8,8}` | 56 B |

Two things in there are not routine:

- the remap is the two-grid twin of the already-ported `sb_other_contact_remap_to_simLaunch`
  (`pairIndex = blockIdx.y` selects which side of the pair gets remapped), so it inherits
  both llvmpipe workarounds from `softBodyContactRemapBda` — but its BLOCK SHAPE differs per
  call site: (1024, 16) for self-collision and cloth-vs-softbody, (8192, 4) for
  softbody-vs-softbody. One SPIR-V entry point cannot carry three strides, and the remap is
  NOT idempotent (a second pass would read an already-remapped id as a collision-tet id), so
  the port has to pin one grid and early-out above it. Coverage stays complete; only the
  parallelism of the 8192 case shrinks;
- `sb_solveOutputSSDeltaVLaunch` is the near-twin of `sb_solveRigidSoftCollisionLaunch` from
  tranche e, but symmetric: BOTH sides are softbody tets, so `computeTetraContact` and
  `updateTetraPosDelta` run twice with opposite `deltaPos`, and the friction coefficient is
  the mean of the two materials' `dynamicFriction` (`PxsDeformableVolumeMaterialData`, 32 B,
  `dynamicFriction` @8).

## P8.3 — the softbody-vs-rigid ATTACHMENT path

`sb_rigidAttachmentPrepareLaunch` and `sb_solveRigidSoftAttachmentLaunch{,TGS}` are mapped,
so a softbody can be pinned to a rigid body or to the world. Manifest **MAPPED 232 -> 235**;
softbody **22 -> 25 of 56**.

| kernel | ABI row | blob |
|---|---|---|
| `sb_rigidAttachmentPrepareLaunch` | `{8,8,8,8,4,8,8,8,8,8}` | 80 B |
| `sb_solveRigidSoftAttachmentLaunch{,TGS}` | `{8,8,4,8,8,8,8,4,8}` | 72 B |

Both rows are the cloth rows: the CUDA packs are position-for-position identical apart from
argument 0.

### One genuinely new piece of arithmetic

`getSoftBodyInvMass` (softBody.cuh:38) scales the blended inverse mass of a TET attachment by
the **length** of the barycentric vector — `baryMass * sqrt(bx²+by²+bz²+bw²)` — where the
cloth path simply takes the blended `.w`. For the fixture's (0.4, 0.2, 0.3, 0.1) that factor
is sqrt(0.30) = 0.5477, so dropping the square root gives every tet attachment the wrong
stiffness and every response term with it. It is the only arithmetic in the tranche that has
no cloth counterpart.

### The sharing rule, and why it cuts the OTHER way here

Tranche d moved `prepareFEMContacts` into a shared module because two copies of it would rot.
This tranche shares only the BYTE LAYOUTS (`pxgSlangFemRigidAttach`) and keeps the kernel
bodies duplicated, which is the opposite call for a reason worth stating once:

- **one CUDA function used by both** -> share it, or the copies drift apart silently;
- **two CUDA functions that happen to agree** -> keep two, or the port stops tracking them.

CUDA itself carries two near-identical attachment prepares (FEMClothConstraintPrep.cu:387 and
softBodyGM.cu:1822) and two solves. A single shared function would be a port-introduced
divergence from the source it transcribes; when NVIDIA changes one, two copies change with it.
`PxgFEMRigidAttachmentConstraint` and `PxgFEMRigidAttachment`, by contrast, are one struct
each written by one kernel and read by another — an ABI, and ABIs drift silently.

The extraction was measured, not assumed: three of the four cloth attachment entry points
compile to **byte-identical SPIR-V**, and the fourth — the prepare, the only caller of the
renamed vertex-type predicate — differs by **exactly one word with an identical opcode
histogram**, i.e. the `OpName` string and nothing else.

### Mutation: 12 of 12

Four rounds, and each round's survivor showed the previous assertion was too weak:

| mutant | killed by |
|---|---|
| drop the `getSoftBodyInvMass` square root | the tet slot's velocity multiplier, bit-exact |
| skip the quaternion rotate; flip the error sign | a NON-identity anchor rotation and translated frame |
| drop `vecGetNormalized` on the world axis | bit-exact `axis_angle` |
| don't zero `rigidDeltaVel`; don't record the attachment id | direct checks |
| drop the `inertiaScale` multiply, or the identity-inertia branch | a KINEMATIC anchor slot |
| solve: drop the barycentric scale; drop the reference count; treat a vertex as a tet | the per-vertex ratio and the (2,1,1,1) counts |
| solve: drop the negation on `deltaPos` | the CPU solve reference |
| solve: change the bias factor 0.5 -> 1.0 | the CPU solve reference |
| solve: "fix" the upstream quirk by reading raXn1/2's own error | the CPU solve reference |

**Three things the fixtures had to be rebuilt to see**, each found by mutation rather than by
inspection:

1. `inertiaScale` is only reachable through `invMass0 == 0 && !isStatic` — a KINEMATIC rigid.
   With static anchors only, the scale stays 1 and multiplying by it is invisible. Slot 2 of
   the prepare fixture is now a kinematic anchor, otherwise identical to slot 0, and the two
   must disagree.
2. Mirroring the anchor and asserting the delta flips sign does NOT catch a dropped negation:
   that flips both runs alike, so the product stays negative. Only an absolute reference does.
   The mirror check is kept because it pins the direction; it is not sufficient alone.
3. The bias factor and the upstream quirk are both invisible to signs and ratios, for the
   same reason — they scale or reshuffle everything alike. `refAttachSolvePGS` closes that.

### One narrow comparison exception, scoped and stated

The kinematic slot's identity-inertia row produces a component that is mathematically exactly
zero, and GPU and CPU disagree on that zero's **sign bit** — `-0` against `+0`. IEEE says the
two are equal and every consumer adds or multiplies the field, so the bit carries no
information. The fixture compares raw bits **except** that `v == 0.0f` maps both zeros to
`+0`, which is true only for the two zeros: every non-zero component stays bit-exact. Widening
this to a tolerance would give up what the NoContraction stamp buys, so it is not widened.
The CPU side is built with `-ffp-contract=off` (tests/CMakeLists.txt:1204), so this is not a
contraction difference.

### Deferred, as in the cloth port and for the same reasons

The ARTICULATION anchor branch; the CONE-LIMIT branch of the solve; and the DYNAMIC-rigid
reaction — the prepare zeroes `rigidDeltaVel` (which this port does) and the solve accumulates
into it (which it does not). A softbody pinned to a MOVING rigid needs that last half; pinned
to the world it does not.

### Verification

| gate | result |
|---|---|
| target ctest (`softBodySim*`) | pass |
| mutation, forced-fresh `.spv` | 12 killed of 12 |
| full suite, default memory mode | **674 / 674**, 170 s |
| full suite, `PXG_DEVICE_V2=1` | **674 / 674**, 243 s |
| rigid regression | 16 / 16 |
| `PXG_SCENE=cloth` | dropped 4.9867 over 1.00 s |
| `PXG_SCENE=clothhang` | hangs and drapes |
| `PXG_SCENE=clothdrape` | settles, rest \|dy\| = 0.000000 |
| `PXG_SCENE=clothselfcollide` | catches at min y = 5.0999 (5.1000 expected) |
| `PXG_SCENE=softbody` | falls, skin gap 0.00000 |
| `PXG_SCENE=particle` | dropped 19.7835 over 2.00 s |

No scene moved. That is the expected outcome — none of the six builds an attachment, so this
tranche is a pure addition and the regression gates are what it has to clear.

## P8.2 tranche e — MILESTONE: the softbody cube SETTLES on a plane

Chain kernels 6 and 7 are mapped, so the softbody-vs-rigid contact path runs end to end for
the first time. Manifest **MAPPED 228 -> 232** (query and solve, each in a PGS and a TGS
variant); softbody **18 -> 22 of 56**. All seven chain kernels are in.

| kernel | ABI row | blob |
|---|---|---|
| `sb_queryRigidSoftContactReferenceCountLaunch{,TGS}` | `{8,8,8,8,8,8,8,8,4,8,8}` | 88 B |
| `sb_solveRigidSoftCollisionLaunch{,TGS}` | `{8,8,8,8,8,8,8,8,8,8,8,4,8,8}` | 112 B |

### KEY EVIDENCE — the settle

```
step  49   caught, 32 contacts
step 100   y = 0.49443  (dy = -0.00002)
step 150+  y = 0.49753  (dy =  0.00000)   -- at rest
           sim y-extent 0.99749 = 0.25% compression under its own weight
CAUGHT by the plane and SETTLED: mean y 0.49753, rest |dy| over the last 50 steps = 0.000002
```

Settled at **y = 0.49753** against the ideal 0.5 for a unit cube resting on y=0; the
difference is elastic compression, not error. With the plane off the cube still free-falls to
exactly -0.98675, so the control is intact.

The scene default rises to **200 steps** when the plane is on. The cube is not caught until
step ~49 and does not come to rest until ~150, so the old 60-step default measured the
bounce. That misread cost a round: the first 90-step run looked like "catches but
oscillates" and was simply not converged.

### contactInitialize was NOT deformable-agnostic, contrary to the recon

The tranche-d recon predicted `contactInitialize` could be extracted unchanged. It could not:
the mass-splitting term is a 3-wide dot for a triangle and 4-wide for a tetrahedron, which is
exactly where CUDA templates `FEMCollision` on `PxVec3` / `PxVec4`. Rather than a float4
signature with a dummy `w` that every cloth caller must remember to zero -- and where a
garbage `w` of `inf` would poison the product -- that scalar is now a PARAMETER and each
caller computes its own width.

The cost is that byte-identity no longer holds for the three cloth entry points that call it:
they grow by 32 bytes, which is **exactly 3 extra `OpLoad` with every arithmetic opcode
unchanged** (opcode-histogram diff), i.e. data movement across a function boundary and
nothing else. `cloth_rigidContactPrepareLaunch` is still byte-identical, and the cloth
fixtures compare bit-exactly against a CPU reference and stay green.

### Mutation: 8 distinct mutants, 6 killed, 2 provably equivalent

The scene alone killed only 4. Three unit fixtures were built for the rest, and each exposed
that the previous assertion was too coarse:

| mutant | killed by |
|---|---|
| drop a reference-count bump | the scene: never settles, \|dy\| stays 0.0049 |
| don't mark the contact in-collision | the scene: falls through |
| drop the position-delta scatter | the scene: never settles |
| unsign the delta | the scene: explodes to -46220 |
| drop the mass-splitting multiply | `softBodySimRigidContactQuerySolve` with a FULL CPU reference |
| drop the applied-force write-back | the same fixture |
| delete a `bc > 1e-3` gate (x or w) | `softBodySimRigidContactStaysActive`, two weights at 0.0005 |
| — | — |
| ignore `wasActive` | EQUIVALENT |
| force the solve's `isActive` guard | EQUIVALENT |

**The two survivors are equivalent, verified at the source rather than assumed.**
`contactInitialize` computes `isActive = wasActive || (CN < 0)` itself and returns early on
`!isActive` with a zero lambda. So the outer `if (wasActive)` shortcut cannot change the
answer, and forcing the solve's `if (isActive)` writes back zeros and the unchanged
accumulated force. Both are upstream's shape and are kept as the shortcuts they are.

**Three mistakes worth recording, each caught by the next test rather than by inspection:**

1. The mass-split check was a RATIO between vertices. A dropped reference-count factor scales
   every vertex identically, so the ratio is unchanged and the mutant walked through. Killing
   it needed `refSolvePGS`, a full CPU transcription of the solve, and an absolute comparison.
2. The `bc > 1e-3` gate was untested because every barycentric weight in every fixture was
   above it. Two weights are now below, on two DIFFERENT components, since the four gates are
   four separate lines of code and testing one leaves the others free to be deleted.
3. The first solve fixture passed the SOLVER normal as +Y, which models a ceiling. The deltas
   pointed down, correctly, and the assertion was wrong. A floor gives -Y, because prepare
   negates the contact normal.

### Verification

```
suite, BOTH memory modes     674/674 (default and PXG_DEVICE_V2=1)
rigid regression             8 scenes x 2 solvers = 16/16
mutation                     6 of 8 killed, 2 provably equivalent
softbody (plane OFF)         4.0000 -> -0.98675, 0.0000% off the closed form
softbody (PXG_SB_PLANE=1)    caught step 49, settled y = 0.49753, rest |dy| = 0.000002
clothdrape                   mean y 1.5000 -> -0.0000; rest |dy| over the last 45 steps 0.000000
clothselfcollide             min y 5.0999 vs 5.1000 expected (mean 5.1105, shelf 5.00)
particle                     mean y 100.7500 -> 80.9665
cloth / clothhang            free fall and drape unchanged
```

Every scene value is byte-identical to its recorded baseline. The default softbody scene is
unchanged on purpose -- the plane is opt-in, so the free-fall control still measures exactly
what it measured before the solve kernels existed.

**One open point, recorded rather than dismissed.** On the FIRST default-mode suite run of
this tranche, `gpuslang.npRigidSumBatches` died with SEGFAULT -- one failure in six full-suite
runs this session. It is not reproducible: the suite rerun is 674/674, 40 targeted runs of
that test under concurrent suite load are clean, and memory was 14.5 GB free at the time.
The shader is also untouched by this work -- it does not import the new module and its `.spv`
is dated from an earlier phase. Treated as a pre-existing intermittent, not a regression, but
it is a real one-off SEGFAULT and should be re-checked if it recurs.

## P8.2 tranche d — chain kernel 5 lands, and prepareFEMContacts becomes shared

`sb_rigidContactPrepareLaunch` (softBody.cu:773, launch PxgSoftBodyCore.cpp:2239) is mapped
and tested. Manifest **MAPPED 227 -> 228**; softbody **17 -> 18 of 56**. Five of the seven
chain kernels are now in; two remain.

| kernel | ABI row | blob | state |
|---|---|---|---|
| `sb_rigidContactPrepareLaunch` | `{8,8,8,8,8,8,8,8,8,8,4,8,1}` | 104 B | **MAPPED** |

### prepareFEMContacts is now SHARED, and the extraction is provably a no-op

The 78-line constraint-block writer was file-local in `femClothRigidContactBda.slang`, and the
softbody kernel needs it byte for byte. It moved to a new module, `pxgSlangFemRigidContact`,
along with the `FRC_*` block offsets, the small vector helpers and the PxNodeIndex accessors;
both shaders import it and neither keeps a copy.

The risk of touching a green shader was retired by measurement rather than by argument: all
**five** cloth entry points compile to **byte-identical SPIR-V** before and after the
extraction (28936 / 37080 / 37080 / 40172 / 40172 bytes, `cmp` on the .spv). That is a
stronger statement than "the tests still pass", and it is cheap to re-run if the module is
ever touched again.

Deliberately NOT a tree-wide de-duplication: `crossVec` / `dotVec` / `mat33MulVec` /
`computeBasisVectors` are also defined locally in `femClothRigidAttachBda`,
`psContactSolveBda` and `particleSelfCollisionBda`, and the `CI_*` contact-info offsets in
`femClothClothContactBda` and `softBodyContactRemapBda`. Those shaders do not import the new
module and keep their own copies; folding them in is a larger change with no payoff here.

### What the kernel actually does differently from its cloth twin

The CUDA argument packs are position-for-position and type-for-type identical apart from
argument 0, so the ABI row is the cloth row. The body differs in exactly four places:
`PxgSoftBodyGpu` for `PxgFEMClothGpu`; the softbody id/tet packing; `mSimDeltaPos` (@128) and
`mSimTetIndices` (@80) for the cloth fields; and the barycentric projection, which is
UNCONDITIONAL and over FOUR weights. The cloth version branches on `barycentric.w == 0`
between a three-weight triangle blend and a direct vertex read, because a cloth contact can
name either; a softbody contact always names a tetrahedron, and the fourth weight is the
fourth vertex, not a discriminator.

`invDt` and `isTGS` sit in the push blob and are unread -- so are they in CUDA, where
brace-matching the whole 122-line `prepareFEMContacts` body (deformableUtils.cuh:132-253)
finds them on the signature line only. `sharedDesc` is likewise unread: the only thing CUDA
takes from it is `mBodySimBufferDeviceData` at :162, inside `if (rigidId.isArticulation())`,
and that branch stays DEFERRED as in the cloth port. The fixture passes a NULL sharedDesc to
keep that honest -- if the articulation branch is ever transcribed, the test faults rather
than quietly reading garbage.

### Mutation coverage: 7 of 7

Seven mutants, all killed -- but three of them only after the fixtures were strengthened, and
one needed a second fixture built specifically for it:

| mutant | killed by |
|---|---|
| drop the 4th barycentric weight | four DISTINCT per-vertex deltas, so a 3-weight blend lands elsewhere |
| don't negate the normal | a direct check against `-planeNormal.x` |
| use the raw penetration instead of `pen - rest` | a NON-zero rest distance in the contact |
| clear applied forces AFTER the tet guard | a skipped contact whose force slot must still be zeroed |
| pick the wrong tet vertex | distinct deltas again |
| `max` -> `min` on the pen-bias clamp | the two clamps made DIFFERENT (-5 vs -1e10); they were equal, so the choice was untestable |
| flatten `workIndex / 32`, or zero `workIndex & 31` | a second contact at index 33, which must land in block 1 slot 1 |
| `continue` -> `return` on the invalid-tet guard | a dedicated stride fixture (below) |

**The `continue`/`return` mutant deserves its own note**, because it is the one that shows a
fixture can be structurally blind rather than merely weak. With every thread handling one
contact, the two are indistinguishable -- there is nothing after the skip for the thread to
do. The difference needs `nbIterationsPerBlock = ceil(ceil(N/256)/1024) > 1`, i.e. **N >
262144**, which a real scene reaches easily (`maxDeformableVolumeContacts` defaults to 1M)
but a unit fixture cannot allocate for.

It does not have to. Only the *count* has to be large; the buffers only have to cover the
indices actually touched. With gridX = 1 the workIndex is `i * 256 + gtid`, so one group
touches 0..511 whatever N says. `softBodySimRigidContactPrepareStride` declares 262145
contacts, allocates 512, puts the invalid tet at index 0 and the real contact at index 256 --
thread 0 meets the guard on its first iteration and the contact on its second. With `return`
that contact is silently dropped, which in a large scene means contacts vanishing behind an
invalid one.

### Verification

```
suite, BOTH memory modes     674/674 (default and PXG_DEVICE_V2=1)
rigid regression             8 scenes x 2 solvers = 16/16
mutation                     7 of 7 killed
clothdrape                   mean y 1.5000 -> -0.0000; rest |dy| over the last 45 steps 0.000000
clothselfcollide             min y 5.0999 vs 5.1000 expected (mean 5.1106, shelf 5.00)
softbody                     coll mean y 4.0000 -> -0.9867, sim mean y identical, skin gap 0.00000
cloth / clothhang            free fall and drape unchanged
```

Every scene value is byte-identical to its recorded baseline. The softbody scene is
unchanged on purpose: kernel 5 prepares constraints that nothing consumes yet, so the cube
must still free-fall to exactly -0.98675. It does.

The CPU reference is a second, independent transcription of `prepareFEMContacts`;
`testFemClothSim.cpp` carries one too. Since the shader-side function is now shared, that
duplication is a CROSS-CHECK rather than a risk: two independently written references verify
one implementation, so a mistake in either surfaces as a failure instead of two tests
agreeing on a wrong answer.

## P8.2 — RECON for the last three chain kernels (5-7), before writing any of them

Read before starting tranche d. Everything below is first-hand from the sources, with
file:line; nothing is inferred from the port's own earlier notes, which have been wrong about
this chain before.

### Is landing kernels 5-7 sufficient? YES for the kernels — NO for the scene.

Traced `PxgSoftBodyCore::solve` (the PGS path, PxgSoftBodyCore.cpp:2812) and
`PxgSoftBodyCore::constraintPrep` (:2570) and cross-checked every `PxgKernelIds::` reference
in the traversed methods against the manifest. On the rigid-vs-softbody path exactly three
kernels are unmapped, and they are the expected three:

| function | line | kernel | state |
|---|---|---|---|
| `sortContacts` | 436 | radix sort ×4, clampMaxValues, `sb_reorderPSContacts` | all MAPPED |
| `prepRigidContactConstraint` | 2216 | `sb_rigidContactPrepareLaunch` | **UNMAPPED (5)** |
| `queryRigidContactReferenceCount` | 1743 | `sb_queryRigidSoftContactReferenceCountLaunch` | **UNMAPPED (6)** |
| `solveRSContactsOutputRigidDelta` | 1035 | `sb_solveRigidSoftCollisionLaunch` | **UNMAPPED (7)** |
| `applyExternalTetraDeltaGM` | 853 | `sb_gm_applyExternalDeltasLaunch` | MAPPED |
| `solveCorotationalFEM` | 1165 | solve-tetra partition, averageVerts, applyDeformationDeltas | MAPPED |

Two other unmapped kernels appear in traversed methods and are both **off** this path:
- `sb_gm_cp_solveTetrahedronsJacobiPartitionLaunch` — taken only when
  `maxPartitions > SB_PARTITION_LIMIT` (PxgSoftBodyCore.cpp:1185, "8th partition is for
  Jacobi-style update"). The 2x2x2 / 40-tet test cube does not reach eight partitions, which
  is why the existing softbody free-fall already works.
- `sb_softbodyContactPrepareLaunch` (SB_SS_CONTACTPREPARE, :2515) — soft-vs-SOFT (VT)
  contact prep. `constraintPrep` launches it unconditionally, but with a single softbody and
  no self-collision its contact count is zero.

**The scene is the real blocker.** `PXG_SCENE=softbody`'s floor is `addGroundBox`
(testGpuSdkStep.cpp:111-118) — a `PxBoxGeometry(50, 0.5, 50)` at y=-0.5. Kernel 2
(`sb_primitiveContactGenLaunch`) is **PLANE-path only**, so a box floor generates no softbody
contacts no matter how many of 5-7 land. The cloth drape scene already shows the fix: a
static `PxPlaneGeometry` floor built with `PxTransformFromPlaneEquation`
(testGpuSdkStep.cpp:640-641). The ground box must STAY — it exists so the rigid PGS island
loop has a live dynamic body (:1213-1218) — so the plane is an addition, not a replacement.

### The cost is far lower than the kernel count suggests: it is a LIFT

`femClothRigidContactBda.slang` already carries all three analogues, mapped and tested:
`cloth_rigidContactPrepareLaunch` (:364), `cloth_queryRigidClothContactReferenceCountLaunch`
(:768) and `cloth_solveRigidClothCollisionLaunch` (:893), plus the TGS twins, all at
`[numthreads(256,1,1)] // SB_UPDATEROTATION`.

The CUDA argument packs are **position-for-position and type-for-type identical** to the
softbody ones. The only differences in the whole signature set are argument 0
(`PxgSoftBody*` vs `PxgFEMCloth*`) and kernel 7's argument 12
(`PxsDeformableVolumeMaterialData*` vs `...Surface...`). So the ABI rows are already written
and already proven in `PxgVulkanKernelAbi.cpp`:

| kernel | ABI row | blob | CUDA |
|---|---|---|---|
| 5 `sb_rigidContactPrepareLaunch` | `{8,8,8,8,8,8,8,8,8,8,4,8,1}` | 104 B | softBody.cu:773, ~67 lines |
| 6 `sb_queryRigidSoftContactReferenceCountLaunch` | `{8,8,8,8,8,8,8,8,4,8,8}` | 88 B | :1944 (helper :1824), ~117 lines |
| 7 `sb_solveRigidSoftCollisionLaunch` | `{8,8,8,8,8,8,8,8,8,8,8,4,8,8}` | 112 B | :2062 (helper :1963), ~95 lines |

`PxgSolverSharedDesc<IterativeSolveData>` deserves a note, because it is templated where the
cloth ABI comment writes it unqualified, and it was flagged as a risk before being checked.
It is not one. It is a POINTER argument, so the ABI row is 8 bytes either way and the
88/112-byte blobs are safe regardless of the instantiation. And the mirror already covers the
derived part: `pxgSlangSolverDesc.slang` carries BOTH `PxgSolverSharedDescTGSGpu` and
`PxgSolverSharedDescPGSGpu` (base 96 + `iterativeData` 128 = 224 bytes, with `iterativeData`
flattened as `iter_*`), and `femClothRigidContactBda.slang:504` already has a
`readRigidVelocity` helper that reads `iter_solverBodyVelPool` through either variant and
selects on `isTGS` -- which is exactly the machinery kernels 6 and 7 need. That is the
PxgVelocityReader analogue, already ported.

**No new descriptor mirrors are needed.** All six rigid-side structs the trio touches are
already mirrored and used together by `femClothRigidContactBda.slang`: `PxgPrePrepDesc`,
`PxgConstraintPrepareDesc`, `PxgSolverSharedDescBase`, `PxgFemRigidConstraintBlock`,
`PxgSolverBodyData`, `PxgSolverTxIData`. `PxsDeformableVolumeMaterialData` is already
mirrored in `softBodySolveBda.slang`, and the rigid `PxsMaterialData` in `pxgSlangNpAbi.slang`.
This was the risk that could have sunk the tranche; it is not present.

### None of the three uses a warp primitive

Scanned all three bodies for `__ballot_sync`, `__shfl`, `warpReduction`, `lowestSetIndex`,
`__syncwarp`, `__shared__` and `threadIdx.y`: **zero hits**. They are flat per-contact loops
with no cross-lane cooperation, which makes them the easiest kernels left in this chain — the
opposite of the midphase and self-collision tranches. What they do use:

- **`blockDim`/`gridDim` arithmetic** (4 and 2 references each): the
  `nbBlocksRequired`/`nbIterationsPerBlock` idiom. The bridge drops blockDim, so it must be
  reproduced as constants — which the cloth port already does, with `RC_BLOCK`/`RC_GRID`
  matching its own `[numthreads]` (femClothRigidContactBda.slang:367-376). Copy that exactly;
  this is the grid-mismatch class that bit `femClothSelfCollisionBda`.
- **atomics, kernel 6 only**: four float `atomicAdd(&softbody.mSimDelta[tetId.*].w, 1.0f)`
  and one uint `atomicAdd(&rigidBodyReferenceCounts[id], 1)`. The float ones are exactly
  `pxgAtomicAddF4W` from P8.1, so the infrastructure is in place.
- **`primitiveConstraints[workIndex/32]`**: `PxgFemRigidConstraintBlock` is a 32-lane
  struct-of-arrays block, not one constraint. The cloth mirror already has the layout.
- **`return` on out-of-range, not `continue`** (softBody.cu:800). The cloth port matches. Note
  this is the EARLY-RETURN shape, which tranche c verified is NOT subject to the dropped
  loop-mask defect — but the two landmines from tranche c still apply to any new code here:
  an optionally-null pointer needs the POINTER selected, not the load guarded, and no block
  may both redefine a loop bound and store.

### PROBE RESULT: the plane path DOES feed a real engine contact — green light

The top-ranked risk below was that kernel 2's PLANE gate is proven only by a unit test, so
tranche e could end with three green kernels and a cube that still falls through. Probed
before writing any of them, and the answer is unambiguously yes.

`PXG_SB_PLANE=1` (new, default OFF) adds a static `PxPlaneGeometry` at y=0 to
`PXG_SCENE=softbody`, coplanar with the ground box's top face. Additive, not a replacement:
the ground box has to stay, because it is what keeps the rigid PGS island non-empty. The
step loop now reads back
`PxSimulationStatistics::gpuDynamicsMemoryConfigStatistics.deformableVolumeContacts` — the
pinned counter `PxgSoftBodyCore::checkBufferOverflows()` already maintains — and, with no
particles and self-collision off, softbody-vs-rigid is its only contributor.

```
plane OFF (control)   max contacts 0,   fall 4.0000 -> -0.98675
plane ON              max contacts 128, first at step 49
   step 49  mean y 0.52563   contacts 32     <- bottom face crosses y=0 HERE
   step 53  mean y -0.04662  contacts 96
   step 56  mean y -0.50442  contacts 128
   step 74  mean y -3.76625  contacts 128    (plateau: fully submerged)
                      fall 4.0000 -> -0.98675, 0.0000% off the closed form
```

Four things make this conclusive rather than suggestive:

1. **The onset is geometrically timed.** Contacts first appear at step 49, where the mean y is
   0.52563 — the cube's half-height is 0.5, so its bottom face crosses y=0 exactly there. A
   spurious or leaked count would not be timed to the geometry.
2. **It is not a buffer ceiling.** `maxDeformableVolumeContacts` defaults to 1×1024×1024, so
   128 is five orders of magnitude below the cap, and no overflow warning fires. The plateau
   at 128 is genuine saturation once the cube is fully through the plane. The 32-granularity
   (32 → 96 → 128) is the contact-gen writer reserving slots per warp, which is the expected
   shape.
3. **The control is intact and the fall is byte-identical.** With the plane on, the cube still
   falls to exactly -0.98675, 0.0000% off the semi-implicit-Euler closed form — the same value
   as with the plane off. That is the predicted outcome, and it is the proof that contacts are
   generated but not yet consumed: nothing between kernel 4 and the vertex update touches them.
   It also shows the plane perturbed nothing it should not have.
4. **The trace shows the chain stopping exactly where the manifest says it does.** With
   `PXG_LAUNCH_TRACE=1` over 55 steps: `sb_reorderPSContactsLaunch` dispatches 55 times and
   `sb_other_contact_remap_to_simLaunch` 15 times, while `sb_rigidContactPrepareLaunch`,
   `sb_queryRigidSoftContactReferenceCountLaunch` and `sb_solveRigidSoftCollisionLaunch` each
   fail their one-time wrangler lookup at init and are never dispatched.

The trace also shows `sb_solveRigidSoftAttachmentLaunch{,TGS}` missing. That is the ATTACHMENT
path (`solveRigidAttachment`, PxgSoftBodyCore.cpp:1790), not the contact path; the scene has
no attachments, so it is irrelevant to a settle, but it is a fourth softbody-vs-rigid kernel
that will be needed for attachment scenes later.

So tranche d/e will pay off, and the settle assertion in tranche e has a working contact
source to build on. `PXG_SB_PLANE` stays default-off until the solve kernels land, so the
recorded free-fall baseline remains the control; tranche e flips it on.

### Verification of the probe change

The probe touches only `sdktest/testGpuSdkStep.cpp`, which is not part of the ctest binary, so
the suite cannot be affected by it -- re-run anyway rather than argued away:

```
suite, BOTH memory modes     674/674 (default and PXG_DEVICE_V2=1)
rigid regression             8 scenes x 2 solvers = 16/16
clothdrape                   mean y 1.5000 -> -0.0000; rest |dy| over the last 45 steps 0.000000
clothselfcollide             settled ON the rest distance: min y 5.0999 vs 5.1000 expected
softbody                     coll mean y 4.0000 -> -0.9867, sim mean y identical, skin gap 0.00000
particle                     mean y 100.7500 -> 80.9665
cloth / clothhang            free fall and drape unchanged
```

Every scene value is byte-identical to the recorded baseline, which is the point: the plane is
default-off, and with it ON the fall is byte-identical too.

### Proposed tranche cuts

- **tranche d — kernel 5 alone.** Smallest (~67 lines), and 6 and 7 both consume its
  `PxgFemRigidConstraintBlock` output, so it must be right first. The body maps onto
  `cloth_rigidContactPrepareLaunch` almost line for line; what actually differs is small and
  enumerable: `PxgSoftBodyGpu` for `PxgFEMClothGpu`, `PXG_BITSHIFT_TET_ID` (20) /
  `PX_MAX_NB_DEFORMABLE_VOLUME_TET` for the cloth id shift and triangle mask (both already
  defined in softBodyContactRemapBda), `mSimDeltaPos` (@128) for `mAccumulatedDeltaPos`,
  `mSimTetIndices` (@80) for `mTriangleVertexIndices`, and an UNCONDITIONAL 4-weight
  barycentric projection where the cloth version branches on `barycentric.w == 0` between a
  3-weight blend and a direct vertex read. `mPenBiasClamp` is already mirrored (@324).
  The out-of-range action is `continue` on `tetInd == 0xfffff` and `return` on
  `workIndex >= tNumContacts` -- both, exactly as CUDA has them.

  **`invDt` and `isTGS` are dead parameters.** They are in the CUDA arg pack and therefore in
  the ABI row, but `prepareFEMContacts` never reads them: brace-matched over its whole
  122-line body (deformableUtils.cuh:132-253) they appear on the signature line only. The
  cloth port omitting them from its helper is correct, not a deferral. `bodySims` IS read, at
  :162, but only inside `if (rigidId.isArticulation())` -- the branch the cloth port already
  defers, and a static plane floor never takes it.

  **One decision to make first: `prepareFEMContacts` has to be shared, not copied.** It is 78
  lines (femClothRigidContactBda.slang:280-357) and the softbody kernel needs it byte-for-byte.
  Its dependencies are already half-shared: `PxgConstraintPrepareDescGpu` and
  `PxgPrePrepDescGpu` live in `pxgSlangPrepDesc.slang` and `nodeIndex` in `pxgSlangCommon.slang`,
  while `computeBasisVectors`, `nodeIsStaticBody`, `dotVec`, `crossVec` and the `FRC_*` field
  offsets are still file-local. Extract those plus the helper into a shared module (the
  precedent is `pxgSlangAtomicFloat.slang` from P8.1) and import it from both shaders. Copying
  instead would leave two 78-line constraint-block writers to keep in sync, which is exactly
  what rots. The risk of touching a green shader is covered cheaply: the cloth kernels are
  NoContraction-stamped with bit-exact fixtures, so an extraction that perturbs them by one
  bit fails immediately.

- **tranche e — kernels 6 and 7 together, plus the plane floor and the settle assertion.**
  They share one descriptor set and 7's correctness depends on 6's reference counts, so
  splitting them buys little and doubles the fixture work. The payoff — a softbody cube that
  actually settles — lands here, which is also where the scene change belongs.

  **It starts with the same extraction tranche d did, and the pieces are already identified.**
  `femClothRigidContactBda.slang` keeps four things file-local that the softbody kernels need
  verbatim, and one that they do not:

  | file-local in the cloth shader | lines | tranche e |
  |---|---|---|
  | `RigidState` + `readRigidVelocity` (:298) | ~55 | EXTRACT — the PxgVelocityReader analogue, PGS/TGS selection included |
  | `ContactResult` + `contactInitialize` (:364) | ~125 | EXTRACT — FEMCollision::initialize, operates on the constraint block and the rigid state, deformable-agnostic |
  | `getGlobalRigidBodyId` (:520) | ~12 | EXTRACT |
  | `readClothVertexInvMasses` (:480) | ~40 | LEAVE — cloth-specific; the softbody needs its own four-vertex tet reader |

  So roughly 190 more lines join `pxgSlangFemRigidContact`, and the softbody side writes only
  its own inverse-mass reader plus the two kernel bodies. The SPIR-V-identity check that
  guarded tranche d's extraction applies unchanged and is now a one-liner to re-run: compile
  each cloth entry point before and after and `cmp` the .spv.

  The four per-vertex reference-count atomics in kernel 6 —
  `atomicAdd(&softbody.mSimDelta[tetId.*].w, 1.0f)` guarded by `bc.* > 1e-3f &&
  deformableInvMasses.* != 0` — land exactly on `pxgAtomicAddF4W` from P8.1, and the rigid
  one is a plain uint atomic. No new atomic infrastructure is needed.

### What could still make this drag

Ranked, honestly. The risk that led this list -- that kernel 2's PLANE gate is proven only by
a unit test and might not feed a real engine contact -- has been RETIRED by the probe above.
What remains:
1. Kernel 6 at ~117 lines is the largest of the three and the only one with atomics; expect
   the mutation pass to find the most gaps there.
2. The plane path feeds contacts, but only 128 of them, saturating once the cube is fully
   submerged. Whether that set is RICH ENOUGH to hold a 27-vertex cube up -- rather than let
   it sink slowly -- is not something the probe can answer; it needs the solve. If tranche e
   produces a cube that sinks instead of settling, suspect the contact SET before suspecting
   the three kernels.

## P8.2 tranche c — BOTH chain kernels land, and two llvmpipe defects come out of it

Chain kernels 3 and 4 were lifted from `npSbRemap.slang`'s unit-tested descriptor-set ports
onto the BDA ABI, into the new `softBodyContactRemapBda.slang`.

| kernel | ABI row | state |
|---|---|---|
| `sb_reorderPSContactsLaunch` | ten pointers, 80 B | **MAPPED**, tested, green |
| `sb_other_contact_remap_to_simLaunch` | `{8,4,8,8,8,8,8}`, 56 B | **MAPPED**, tested, green |

Manifest **MAPPED 225 → 227**; softbody **15 → 17 of 56**.

### The reorder

A pure gather through `remapByRigid` into the sorted contact arrays. The fixture uses a
permutation with **no fixed points**, so an identity gather cannot pass, and asserts the
entries past the contact count keep their poison. Also documented: the CUDA iteration count
is `(total + blockDim*gridDim - 1) / blockDim * gridDim` — **missing parentheses**, so it
over-counts by roughly `gridDim²`. Harmless, because the loop returns on `index >= total`;
the port keeps a plain bounded stride loop rather than reproducing the slip.

### The remap: two driver defects stacked on top of each other

The remap was carried unmapped for a while behind a SIGSEGV, and the in-harness bisection
recorded in the previous revision of this document was **wrong in every one of its
exclusions** — it had ruled out the per-lane arrays, the dynamic indexing, the group size,
both geometry helpers and the loop trip count, and concluded "llvmpipe miscompiles this loop
nest". It does not. `tools/lvprepro/repro3` settled it in two runs, and there were two
independent defects, neither of which is specific to this kernel.

**1. A conditional load is if-converted across its own null test.** `prevContactCount` is
optional in CUDA (`prevContactCount ? *prevContactCount : 0`). Slang emits a real branch —
`OpSelectionMerge` with the `OpLoad` inside the taken side, confirmed with `spirv-dis` — and
the backend flattens it into an unconditional load plus a masked select, so the load runs on
a null address *before* the test:

```
mov   0x30(%rdx),%r10      ; r10 = prevContactCount == 0
vmovd (%r10),%xmm1         ; <-- faults, BEFORE the null test
neg   %r10 / sbb / kmovd   ; the "guard", evaluated afterwards
```

Speculating a load across a null test is illegal, and no arrangement of `if` / `?:` in the
source forbids it, because the branch is already there. The fix has to remove the faulting
*address*: **select the pointer, not the loaded value**, so one load reads something valid on
both paths (`totalContactCount`, dereferenced unconditionally a line earlier, so never null)
and a second select discards it. This is the stronger form of the speculated-load note in
`softBodyContactGenBda`, where a zero-trip loop hid the load; here an explicit ternary did not
stop it. **Any optionally-null device pointer in this port needs this shape, not a guard.**

This also explains why the bisection pointed the wrong way for so long. A read-only probe
survived while adding a single store faulted, which reads as "stores are broken here" — and
that is what put the blame on the arrays and the loop nest. But a read whose result is never
stored is **dead-code eliminated**: only the version that keeps the descriptor reads *live*
ever reaches the null load. Under DCE, "adding a store makes it fault" means "the reads
before it were being deleted", not "the store is at fault". Worth remembering as a
bisection rule, because it inverts the obvious reading.

With the pointer selected, the per-lane arrays compile and run — verified standalone at
range 1, 32, 33, 40 and 100 — so **the deviation reported last revision is withdrawn**. The
kernel now keeps CUDA's 32 per-lane registers and both tie-breaks the warp reduction encodes
are bit-exact. There is no remaining deviation in the reduction.

**2. A loop loses its per-lane exit mask when the bound is redefined next to a store.** This
one is SILENT — no fault, wrong results. Transcribing CUDA's contact-count clamp literally:

```slang
uint n = count[0];
if (n > max) { n = max; if (firstThread) count[0] = max; }   // MISCOMPILED
for (uint i = base; i < n; i += stride) { ... }
```

the whole first **subgroup** ran the loop once regardless of `n`. With 12 contacts declared
and 3 after the clamp, lanes 0..7 all wrote barycentrics — for contacts 3..7 that are out of
range — while every later lane correctly did nothing. The write-back itself was correct, so
nothing reported an error. Splitting the block two ways shows the trigger is the
**combination**: keeping the mutable `n` but lifting the store out is correct, and keeping the
nested store but computing `n` with `min` is correct. Only a block that both redefines the
enclosing loop's bound and stores is miscompiled. The kernel now uses a pure `min` plus a flat
conditional store, which removes both halves, so a later refactor cannot resurrect it by
undoing one.

**Sweep for both defects across the shader tree.** Defect 1: the only other guarded-pointer
sites (`npContactPrePrep.slang:970,973`, `broadphaseIncrementalHistogram.slang:190`) already
select the *pointer*, which is the safe form. Defect 2 needs the loop — an **early return** in
its place is unaffected, verified by patching this same kernel into that shape, which is why
`npSbRemap.slang:275,382` need no change despite looking identical.
`particleSelfCollisionBda.slang:305/320` does have the loop shape (an accumulator assigned in
a storing conditional, then used as a bound) and is **verified clean**; its pre-existing
fixture gave every lane the same trip count and so could not have seen the defect, and
`particleSelfCollisionDivergentCounts` now pins it with counts that actually differ across
lanes (2, 1, 1 in one subgroup).

### Mutation coverage

Ten mutants, nine killed. Four needed fixtures that did not exist: `softBodySimContactRemapToSim`
gained a non-null `prevContactCount` case (the null half is what faulted, so both halves are
now fenced), and the new `softBodySimContactRemapTieBreaks` supplies the geometry that makes
the reduction's tie-breaks observable — 54 sim tets across three collision tets, with
byte-identical duplicate tets placed 32 apart for a same-lane tie, 3 apart for a cross-lane
tie, and two overlapping containing candidates for the ballot. It also checks that the clamped
count **bounds the loop** rather than merely being written back, which is defect 2's regression
fence.

The one surviving mutant is `no-fallback-clamp`, and it **cannot** be killed: the clamp is
provably a no-op, because `computeTetBarycentric` already clamps every component into [0,1]
before returning (upstream's `deformableCollision.cuh:133` does the same, so this is CUDA's own
double clamp, not something the port introduced), and the only value that never passes through
that helper is the initial one, which is `float4(0,0,0,0)`. CUDA declares `float4 barycentric;`
**uninitialised**, so upstream's clamp does have a job on the `range == 0` path; initialising
instead is strictly better defined, and that is what makes the clamp redundant here rather than
merely unexercised. Kept anyway as a backstop.

### Where the chain stands

Seven kernels stand between a generated contact and a moving softbody vertex. Four are now
mapped; three remain:

| # | kernel | state |
|---|---|---|
| 1 | `sb_midphaseGeneratePairsLaunch` | mapped (tranche a) |
| 2 | `sb_primitiveContactGenLaunch` | mapped (tranche b) — **PLANE path only** |
| 3 | `sb_other_contact_remap_to_simLaunch` | mapped (this tranche) |
| 4 | `sb_reorderPSContactsLaunch` | mapped (this tranche) |
| 5 | `sb_rigidContactPrepareLaunch` | mapped (tranche d) |
| 6 | `sb_queryRigidSoftContactReferenceCountLaunch` | mapped (tranche e) |
| 7 | `sb_solveRigidSoftCollisionLaunch` | mapped (tranche e) |

So the softbody cube still does not settle, and cannot until 5-7 land — the contacts are now
correctly addressed and correctly ordered, but nothing consumes them yet. The GJK/EPA tranche
for the non-plane rigid types in kernel 2 stays separate; `softBodySimPrimitiveContactGenPlane`
pins that boundary with a BOX pair so it has to be changed consciously.

### Verification

Suite **674/674 in both memory modes**, rigid regression **8 scenes x 2 solvers = 16/16**, and
all six deformable/particle scenes stepped and byte-unchanged against their recorded baselines:

```
clothdrape        mean y 1.5000 -> -0.0000; rest |dy| over the last 45 steps = 0.000000
clothselfcollide  caught and settled ON the rest distance: min y 5.0999 vs 5.1000 expected
softbody          coll mean y 4.0000 -> -0.9867, sim mean y identical, skin gap 0.00000
particle          mean y 100.7500 -> 80.9665
cloth / clothhang free fall and drape unchanged
```

## P6.6 — MILESTONE: the PBD particle bag ALSO settles under the TGS solver (no longer tunnels)

The P6.5 settling bag was a **PGS** result; under **TGS** the same cloud TUNNELLED
(yEnd≈−9.7) because the TGS particle-solve path routes through three kernels that
were still unmapped. This tranche maps them, so the bag settles under BOTH solvers.
Three kernels; `PXG_SLANG_KERNEL_MAPPED` 177 → **180**, all in new file
`psContactSolveTgsBda.slang`, ported line-for-line from `particlesystem.cu`:
- `ps_stepParticlesLaunch` (`particlesystem.cu:120`, ABI `{8,8,4,4,1,1,1,12}`=40B) — the
  TGS substep PREDICTOR run at the top of each non-velocity iteration: advances the
  sorted position by `vel*dt` (from sorted-origin on the first iter, else sorted-pos +
  running deltaP) and accumulates the running deltaP. Launch `PxgParticleSystemCore.cpp:2460`.
- `ps_solvePCOutputParticleDeltaVTGSLaunch` (`:1690`, same 96B pack as the non-TGS twin) —
  the TGS twin of `ps_solvePCOutputParticleDeltaVLaunch`. DIFF vs non-TGS: the velocity read
  (`PxgVelocityReader::readVelocitiesTGS` unpacks lin/ang VELOCITY **and** lin/ang DELTA from
  the 3-float4-per-body TGS iter pool) and the solve core (`solvePCOutputDeltaVTGS`: folds
  `normalDelta + normalVel*projectionDt` / `tanDelta` into the separation, `projectionDt =
  velIter?0:dt`). For a static floor with zero body velocity/delta the folding collapses to
  the non-TGS result, so TGS and PGS agree on the money-shot contact. Launch `:1947`.
- `ps_finalizeParticlesLaunch` (`:5731`, ABI `{8,4,4,4}`=24B) — the TGS velocity-iteration
  FINALIZE: blends `deltaP*invTotalDt` into the sorted velocity when the stored velocity is
  larger (momentum conservation). Launch `PxgPBDParticleSystemCore.cpp:498`.
The contact PREPARE + deltaV ACCUMULATE stay shared with the non-TGS path (psContactSolveBda,
byte-untouched). All descriptor fields reused from existing mirrors; none added.

**KEY EVIDENCE — the TGS settling bag (`PXG_SOLVER=tgs`, cloud at y≈3.75):**
```
mean y: 3.75 → 3.38 (s15) → 2.40 (s30) → 0.85 (s45, accelerating) → 0.25000 (s60, caught) →
        0.25000 … 0.25000 (rest);  rest |dy| over the last 35 steps = 0.000000
[  OK  ] PBD particle bag fell, was caught by the floor, and SETTLED TO REST (the money shot)
```
Settles at **exactly y=0.25000** = the rest offset above the floor, byte-identical to the
PGS bag — NOT tunnelling to −9.7. The 3 kernels DISPATCH OK in the trace (step ×24, TGS-solve
×27, finalize ×3 over 3 steps). PGS bag STILL settles (unchanged, 3.75→0.25), and the DEFAULT
free-fall is byte-unchanged (100.75 → 80.9665) under BOTH solvers. New unit test
`gpuslang.psContactSolveTgs` (4 cases): a penetrating particle pushed OUT upward by the TGS
solve (bit-exact vs a CPU transcription of `solvePCOutputDeltaVTGS`), a moving NON-static body
that exercises the `readVelocitiesTGS` unpacking + the `normalDelta`/`normalVel*projectionDt`
folding (velIter true/false, bit-exact), and step + finalize in isolation. Full suite
**672/672** both memory modes; rigid regression 8×2 = **16/16**.

Deviations (flagged): the `isArticulation()` reaction branch is DEFERRED (dead; the
`artiCoreDesc`/`outArtiVelocity` reads the `PxgVelocityReader` ctor hoists are sunk INTO the
dead branch so a rigid/static contact never dereferences them — identical result for the
particle-vs-static-floor scenes); `ps_stepParticlesLaunch` indexes `particleSystems[blockIdx.y]`
directly (the `activeParticleSystems*` arg is ABI parity only); descriptor block-copy→direct
`Ptr<>` reads; PxU16 phase→material as 32-bit word+shift; no 32-lane-warp ops in any of the
three kernels. `psContactSolveTgsBda` is in `PXG_NOCONTRACT_SHADERS` (NoContraction stamped)
so the dot-product/sqrt-heavy TGS solve is bitwise vs the `-ffp-contract=off` CPU reference,
matching the `npPsSolvePCTGS` precedent.

**MANIFEST GOTCHA (found + fixed):** `PxgSlangKernelManifest.h` carries MANUAL overrides the
generator (`gen_kernel_manifest.py`) does NOT reproduce — notably the P6.1 BDA re-points
`radixSortDoubleCopy` / `radixSortDoubleCopyHigh32Bits` → `particleContactSortBda.spv` (the
generator's "first-definition-wins" instead picks the RWStructuredBuffer harness
`radixSortGather.spv`). Re-running the generator SILENTLY reverted those two rows, so the
engine dispatched a descriptor-bound shader with an unbound layout → GPU fault in the particle
sort (crashed every particle scene, both solvers, even free-fall). Fix: DO NOT regenerate — the
3 new rows + the count were hand-edited, leaving every other row (and its annotation) intact.
The rigid-side TGS twin `ps_solvePCOutputRigidDeltaVTGSLaunch` stays DEFERRED (inert for a
static floor, refuses harmlessly like the non-TGS rigid side under PGS).

## P6.5 — MILESTONE: a PBD particle bag falls, is caught by the floor, and SETTLES TO REST on Vulkan (the money shot)

The full PBD particle pipeline now produces a **settling granular bag** on the Slang/Vulkan
(llvmpipe) engine path — the culmination of the P6 marathon. A 4×4×4 particle cloud dropped
onto a static floor falls under gravity, is **caught by the particle-side primitive-contact
solve** (P6.4a generation → P6.4b-1 sort/organize → P6.4b-2 solve), and **comes to rest at
exactly `solidRestOffset` above the floor** — no tunnelling, no explosion, no crash.

Observed (PGS, cloud spawned at y≈3.75 onto the floor top at y=0):
```
mean y: 3.75 → 3.38 → 2.40 → 0.85 (falls, accelerating) → 0.25000 (caught) →
        0.25000 … 0.25000 (rest);  rest |dy| over the last 35 steps = 0.000000
[ OK ] PBD particle bag fell, was caught by the floor, and SETTLED TO REST (the money shot)
```
It settles at **exactly y=0.25000** = the rest offset (0.5·spacing) above the floor — the
particle-side solve alone suffices (the rigid-side reaction is inert for a static floor).

**Solver scope: this is a PGS result.** PGS routes the catch through the non-TGS
`ps_solvePCOutputParticleDeltaVLaunch` mapped in P6.4b-2. Under TGS the cloud still tunnels
(yEnd≈−9.7) because TGS routes through the `...TGSLaunch` twin (+ `ps_stepParticlesLaunch` /
`ps_finalizeParticlesLaunch`), which are the deferred next tranche — so the settling bag is
demonstrated on **`PXG_SOLVER=pgs`**.

This is a **pure scene/harness tranche** (no new kernels — the pipeline was completed by
P6.0–P6.4b-2). `testGpuSdkStep.cpp`: `PXG_PARTICLE_STARTY` lowers the cloud so it reaches the
floor within the run and switches the pass criterion from "monotonic free fall" to
**fall-then-rest** (drop > 0.5, rest |dy| over the final quarter < 0.01, settled near the floor,
finite); in bag mode the required rigid body is relocated far aside (x=20) so the pile never
touches it, and solver iterations are raised to 8/1. `PXG_SLANG_KERNEL_MAPPED` stays **177**
(no kernel change). The DEFAULT `PXG_SCENE=particle` free-fall is byte-unchanged (100.75 →
80.9665, monotonic) — the full gpuslang suite (671/671 both modes) and the 8-scene × 2-solver
rigid regression (16/16) are unaffected.

Deferred (out of this milestone): the rigid-side particle reaction (`ps_solvePCOutputRigidDeltaV*`,
`accumulateDeltaVRigid*` — needed only for a DYNAMIC floor), the TGS particle-solve twin, the
articulation reaction path, diffuse/cloth/fluid/inflatable particle features, and the
narrowphase-hygiene `removeContactManagers_Stage5`.

## P6.4b-2 — particle-side contact SOLVE on the engine ABI (the "catch" application)

Brought the particle-side primitive-contact SOLVE onto the BDA engine ABI — the kernels that
turn a particle-vs-shape contact into a corrective particle delta. Three kernels;
`PXG_SLANG_KERNEL_MAPPED` 174 → **177**:
- `ps_contactPrepareLaunch` (`particlesystem.cu:1238`, ABI `{8,8,8,8,8,8,8,4,1,8,8}`=80B) — prep the
  primitive contact constraint (effective mass / normal / bias) into the 3072-byte
  `PxgParticlePrimitiveConstraintBlock` SoA.
- `ps_solvePCOutputParticleDeltaVLaunch` (`:1531`, non-TGS, `{8,8,8,8,8,8,8,8,8,4,4,1,8}`=96B) —
  solve the constraint, particle side → per-contact deltaV.
- `ps_accumulateDeltaVParticleLaunch` (`:2828`, `{8,8,8,8,8,8}`=48B) — reduce per-contact deltaV
  over each particle's `[start,end)` range (P6.4b-1) into the particle's accumulated delta.
New file `psContactSolveBda.slang`; all descriptor fields reused from existing mirrors
(`PxgParticleSystemGpu`, the SoA constraint block walked by byte-offset `Ptr` views, plus the
solver-desc mirrors), none added.

**KEY EVIDENCE — a penetrating particle is pushed OUT:** unit test `gpuslang.psContactSolve` builds
one particle penetrating a static plane (normal (0,1,0), pen −0.1, range [0,1)), runs
prepare→solve→accumulate, and asserts `accumDeltaP = (0, +0.1, 0)` — UPWARD, fully resolving the
penetration — bit-exact vs a CPU transcription (plus relaxation-scaled and infinite-mass cases).
In the live scene the 3 kernels now **DISPATCH OK** (contactPrepare ×240, solve ×600, accumulate
×600 per run) and safely no-op on the zero-contact high cloud (uniform `workIndex<tNumContacts`
zero-trip guard) — this additionally proves the BDA/ABI wiring end-to-end. Free-fall byte-unchanged
(PGS 80.9665, TGS 9.89m). Suite **671/671** both memory modes. Rigid regression 8×2 = **16/16**.

Deviations (flagged): the `isArticulation()` reaction branch is structurally preserved but dead
(no write; the artiCoreDesc/outArtiVelocity reads sunk into the dead branch so a rigid/static
contact never dereferences them) — a byte-exact articulation mirror is a separate reaction tranche;
descriptor block-copy→direct `Ptr<>` reads; no 32-lane-warp ops in these kernels.

Remaining to the money shot: the rigid-side reaction (`ps_solvePCOutputRigidDeltaVLaunch`,
`accumulateDeltaVRigid*` — inert for a static floor) + the TGS solve twin + articulation path, and
**P6.5** (lower the cloud + rewrite the harness fall→rest so the settling bag becomes observable).

## P6.4b-1 — particle contact sort + reorder + range-find on the engine ABI

Brought the contact-**organization** half of the consume path onto the BDA engine ABI —
sort the generated primitive contacts by particle, reorder them, find each particle's
contact range. Five kernels (all previously `nullptr`); `PXG_SLANG_KERNEL_MAPPED` 169 → **174**:
- `radixSortMultiBlockLaunch` / `radixSortMultiCalculateRanksLaunch` (`RS_MULTIBLOCK` /
  `RS_CALCULATERANKS_MULTIBLOCK`, ABI `{8,4}`) — the **plain** contact-sort radix pair (a
  third variant distinct from WithCount/WithoutCount). It takes `PxgRadixSortBlockDesc*`
  (`: public PxgRadixSortDesc`, adds a `numKeys` **device pointer** at @48 — mirror
  `sizeof==56`, static_asserted) and reads the key count as `*desc.numKeys`. Added to
  `radixSortDesc.slang` reusing the proven scan cores.
- `ps_reorderPrimitiveContactsLaunch` — materializes the two sorted contact arrays from the ranks.
- `ps_findStartEndParticleFirstLaunch` + `...SecondLaunch` — per-particle `[start,end)` ranges + pair counts.
New file `psContactSortBda.slang`.

**Recovery note:** this tranche was reconstructed after a second container recycle killed the
in-flight agent — its written code survived on disk and was salvaged. A follow-up fix corrected
`ps_findStartEndParticleFirstLaunch`: the faithful CUDA port emits **within-block** offsets (each
of 32 blocks restarts its counter; pass 2 adds the per-block base), but the intended encoding for
the reconstructed test is a **global** running range index. Since `offsets`/`blockOffsets` are
internal intermediates (only pass 2's `rangeStart`/`rangeEnd`/`totalNumPairs` are consumed
downstream), pass 1 was folded onto one workgroup carrying a single global accumulator across all
512-contact chunks (blocks 1..31 uniform early-return) and pass 2 consumes the global offset
directly — the observable outputs are unchanged.

Verification: unit test `gpuslang.psContactSort` (plain-radix 8-pass bit-exact, reorder permutation,
findStartEnd ranges) passes both memory modes; suite **670/670** both modes. Trace: the plain radix
(×1920/frame), reorder (×120), and both findStartEnd (×120) all dispatch **OK** — the
"sortParticleContacts/sortParticles fail to launch" errors are gone. Free-fall byte-unchanged (PGS
80.9665, TGS 9.89m). Rigid regression 8×2 = **16/16** (the plain radix is a shared gpucommon kernel;
rigid path byte-unaffected).

P6.4b-2 frontier: the range-organized contacts feed the still-unmapped `ps_contactPrepareLaunch` →
`ps_solvePCOutputParticleDeltaV*` (+ rigid-side `ps_solvePCOutputRigidDeltaV*` /
`ps_accumulateDeltaVParticleLaunch` / `accumulateDeltaVRigid*`) solve chain; **P6.5** lowers the
cloud so the whole path dispatches and the bag settles.

## P6.4a-2 — particle-vs-primitive contact generator on the engine ABI

Brought the heavy `ps_primitivesCollisionLaunch` (`cudaParticleSystem.cu`, device core
`particlePrimitiveCollision`) onto the BDA engine ABI — the kernel that detects a particle
touching a static shape (plane/box/sphere/capsule) and writes the contacts. It's the
"catch" contact generator. New file `psContactGenBda.slang`; manifest count 168 → **169**.

The 15-arg launch pack (`PxgNarrowphaseCore.cpp:3225`) includes the `PxgParticleContactWriter`
**by value** — marshalled inline as the 72-byte tail of a 176-byte push blob (writer@104,
static_assert-pinned member offsets: particleContacts@104 … maxContacts@168). ABI row
`{1,4,4, 8×11, 72}`. Reused mirrors (all offset-validated in the test TU): `PxgParticleSystemGpu`/
`PxgParticleSimBufferGpu` (P6.0), `PxgContactManagerInput`(16)/`PxgShape`(48)/`PxsCachedTransform`(32)/
`PxgRigidFilterPair`(32) (transcribed, static_asserted), + the plane/sphere/capsule/box contact
core + `writeContact` staging fan-out reused from harness `psPrimCollide.slang`.

llvmpipe deviations (flagged): geometry switch keys the **real** `PxGeometryType` from
`PxgShape.type`; the caller's grid-stride ballot/warpScan compaction dropped (scheduling only —
contact set + total identical); per-warp `warpReduction<Max>` → full-**block** groupshared max
(block-uniform inner bound so ballot barriers are safe without a 32-lane subgroup); per-warp
`__ballot_sync`+lane-atomic kept as a 32-lane **logical** groupshared tile; every BDA `Ptr<>` load
behind a uniform-scalar guard (no hoist out of zero-trip cell loops).

**KEY EVIDENCE (no scene observable this tranche — the high cloud never reaches the floor so
`numTests=0` and the kernel doesn't dispatch):** unit test `gpuslang.psContactGen` builds ONE
particle at y=−0.1 penetrating a static plane y=0 with a filled grid, runs the kernel, and
asserts exactly **ONE** contact with normal **(0,1,0)** and penetration **−0.1**, bit-exact vs a
CPU transcription of the ePLANE branch (all six radix staging arrays + the counter correct). Both
memory modes. Suite **669/669** both modes. Free-fall byte-identical (PGS 80.9665, TGS 9.89m), no
new refused/error lines vs the P6.4a-1 baseline, no crash. Rigid regression 8×2 = **16/16**.

P6.4b frontier: the generated contacts feed the not-yet-mapped contact **sort → prep → one-way
solve** chain (`sortParticleContacts` radix staging → `ps_reorderPrimitiveContactsLaunch` →
`ps_contactPrepareLaunch` → `ps_findStartEndParticle*` → `ps_solvePCOutputParticleDeltaV*` →
`ps_accumulateDeltaVParticleLaunch`); **P6.5** lowers the cloud so the whole path dispatches and
the bag settles.

## P6.4a-1 — particle bound / cell-range kernels on the engine ABI

Toward a particle-vs-floor "catch", brought the 4 particle bound/cell-range
kernels onto the BDA engine ABI (the AABB bookkeeping that precedes contact gen;
the heavy `ps_primitivesCollisionLaunch` is the separate next tranche P6.4a-2):
- `ps_updateBoundFirstPassLaunch` (`{8,4,1,8}`) + `ps_updateBoundSecondPassLaunch`
  (`{8,4,8,4,8,8,4}`) — 2-pass particle-system world-AABB reduce feeding the
  broadphase (`PxgParticleSystemCore.cpp:856`/`:907`).
- `ps_primitivesBoundFirstPassLaunch` (`{4,8,8,8,8,8,8,8,8}`) +
  `ps_primitivesBoundSecondPassLaunch` (`{4,8,8,8}`) — per BP-found particle-shape
  pair, the grid-cell range to test (`PxgNarrowphaseCore.cpp:3151`/`:3190`).
New file `psBoundsBda.slang`. Reused mirrors: `PxgParticleSystemGpu` (P6.0, no
extension needed), `PxgContactManagerInput` (16B, from boxBoxNphase.slang) and
`PxgShape` (48B, type@40/particleOrSoftbodyId@44, from pxgSlangCommon.slang), both
pinned by `offsetof`/`static_assert` against the real engine structs. The 32-lane
`warpReduction` in updateBound lowered to a **portable full-block groupshared
min/max tree** (min/max of a finite float set is associative + rounding-free ⇒
bit-identical for any subgroup width — the llvmpipe guard).

**Trace: `ps_updateBound{First,Second}` dispatch OK ×120/frame**; free-fall
byte-unchanged (PGS 80.9665, TGS 9.89m). The `ps_primitivesBound` kernels are
mapped + unit-tested bit-exact but do NOT dispatch in the 2s free-fall window (the
cloud at y 81–100 never overlaps the ground box's AABB → `eParticlesystems` BP
bucket empty, numTests=0); they activate once P6.4a-2 contact-gen brings the
particle-vs-floor pairs online. Suite **668/668** both memory modes (new test
`gpuslang.psBounds`: AABB reduce + cell-range both bit-exact vs CPU).
`PXG_SLANG_KERNEL_MAPPED` 164 → **168**. Rigid regression 8×2 = **16/16** (reused
rigid-shared narrowphase mirrors byte-unaffected).

## P6.3 — particle self-collision + one-way solve on the engine ABI

Brought the particle-particle self-collision path onto the BDA engine ABI — the
3×3×3 grid neighbour search (over the P6.2 grid) + the one-way contact solve loop.
Four kernels, all previously `nullptr`:
- `ps_selfCollisionLaunch` (`particlesystem.cu:1145`, ABI `{8,4}`) — neighbour-search
  the grid cells via `mCellStart`/`mCellEnd`/`mSortedToUnsortedMapping`, emit one-way
  self-collision contacts + accumulate deltas.
- `ps_solveOneWayContactDeltaVLaunch` (`:1823`, ABI `{8,8,4,4,4,1}`) — solve the
  one-way contacts → `accumDeltaP`.
- `ps_applyDeltaLaunch` (`:3689`, ABI `{8,8,4}`) — SOR-average `mDelta` → `accumDeltaP`.
- `ps_updateParticleLaunch` (`:3010`, ABI `{8,8,4,1}`) — apply `accumDeltaP` + recompute
  the sorted velocity.
All reuse the P6.0 `PxgParticleSystemGpu` mirror unchanged (P6.3-touched offsets pinned
via `offsetof`: `mAccumDeltaP@152`, `mOneWayContactInfos@200`, `mOneWayForces@208`,
`mOneWayContactCount@232`, `mDelta@368`, `mData.mRestOffset@268`/`mSolidRestOffset@272`/
`mRelaxationFactor@300`, `mCommonData.mMaxNeighborhood@124`).

**Key llvmpipe deviation (verified equivalent):** the CUDA `SelfCollideCache`
`__shared__ uint[32][64]` is accessed ONLY as `[*][threadIdx.x]` (write @1130, read @1231)
— each thread exclusively owns its column, no cross-thread sharing — so it lowers to a
per-thread local `uint[32]` and the `__syncwarp()` drops out. This **removes the sole
32-lane-warp assumption**, making the kernel llvmpipe-safe, and is provably identical.

**Correctness is proven by the unit test, not the scene:** on the uniform free-fall cloud
all particles stay at rest spacing, so self-collision is a genuine no-op. The
`particleSelfCollisionSeparates` test constructs two particles overlapping at Δx=0.15
(< 2·solidRestOffset=0.2) with a filled grid, runs selfCollision → solveOneWay →
applyDelta → updateParticle, and confirms they are pushed apart to Δ=0.25 ≥ 0.2,
bit-exact vs a CPU transcription. The three solve kernels are also each unit-tested
bit-exact in isolation. **`PXG_SCENE=particle` trace: all 4 dispatch OK every frame**
(selfCollision ×120, the solve trio ×600 = 5 iters × 120 frames); **free-fall
byte-identical (PGS 80.9665, TGS 9.89m)**. Suite **668/668** both memory modes;
`PXG_SLANG_KERNEL_MAPPED` 160 → **164**. Rigid regression 8×2 = **16/16** (particle-only
kernels, rigid path byte-unaffected).

P6.4 frontier: the primitive/plane-contact chain (plain WithCount `radixSortMultiBlockLaunch`/
`CalculateRanks`, `ps_reorderPrimitiveContactsLaunch`, `ps_primitivesBound*`,
`ps_primitivesCollisionLaunch`, `ps_contactPrepareLaunch`, `ps_findStartEndParticle*`,
`ps_solvePCOutputParticleDeltaV*`, `ps_accumulateDeltaVParticleLaunch`) — the "catch" that
lets the cloud rest on a floor.

## P6.2 — particle grid build + reorder on the engine ABI (spatial hash → sort → cell start/end)

Brought the particle grid-build stage onto the BDA engine ABI — the spatial-hash /
radix-sort / reorder trio that is the hard prerequisite for particle self-collision.
Four kernels, all previously UNMAPPED (`nullptr` in the manifest → "Could not find
registered CUDA function" at runtime, so the engine's `updateGrid()` silently no-op'd):
- `radixSortMultiBlockLaunchWithoutCount` / `radixSortMultiCalculateRanksLaunchWithoutCount`
  (`radixSortImpl.cu:69`/`:81`, ABI `{8,4}`) — the previously un-authored WithoutCount
  radix twins. They turned out to be the already-proven WithCount BDA entrypoints
  (`radixSortDesc.slang`) with `numKeys` sourced from `desc.count` (offset-40 field of
  `PxgRadixSortDescGpu`) instead of a push param; bodies copied verbatim, WithCount
  entrypoints left byte-untouched.
- `ps_calculateHashLaunch` (`particlesystem.cu:803`, ABI `{8,8}`) — grid-cell hash per
  particle + identity sorted↔unsorted seed.
- `ps_reorderDataAndFindCellStartLaunch` (`particlesystem.cu:898`, ABI `{8,4}`,
  `__launch_bounds__(1024)`) — gather sorted arrays via the permutation + groupshared
  cell-boundary scan producing `mCellStart`/`mCellEnd`.
Both grid kernels REUSE the P6.0 `PxgParticleSystemGpu` mirror unchanged (every field
they touch was already present — pinned via `offsetof` static checks:
`mGridParticleHash@144`, `mCellStart@168`/`mCellEnd@176`, `mGridCellWidth@88`,
`mUnsortedToSortedMapping@48`/`mSortedToUnsortedMapping@56`). `calcGridHash`'s
`int & uint` wrap reproduced with `asuint(int) & (uint)`; the CUDA descriptor blockCopy
replaced by direct `Ptr<>` field loads (semantically identical).

**`PXG_SCENE=particle` trace: all 4 dispatch OK every frame** (`ps_calculateHashLaunch`
×120, `ps_reorderDataAndFindCellStartLaunch` ×120, the two WithoutCount radix ×960 each,
zero REFUSED). The grid is now correctly built on-device; **the free-fall is byte-identical
to the P6.1 baseline (PGS 80.9665, TGS drops 9.89m)** — the reorder→integrate round-trip
is an identity on the contact-free path, so the port is provably trajectory-neutral (grid
feeds self-collision, still unmapped). New unit test `gpuslang.particleGrid` (radix
WithoutCount 8-pass bit-exact; calculateHash vs CPU incl. negative-coord wrap; reorder +
cellStart/cellEnd) passes both modes. Suite **667/667** both memory modes (the lone -j4
`scan`/`npRigidSumBatches` SEGFAULTs are the known flakes; pass isolated).
`PXG_SLANG_KERNEL_MAPPED` 156 → **160**. Rigid regression 8×2 = **16/16** — the shared
`radixSortDesc` module change (WithCount untouched) leaves the rigid broadphase
byte-unaffected.

P6.3 frontier (now the blocking failures, reached on valid grid data): the plain
count-via-pointer `radixSortMultiBlockLaunch`/`radixSortMultiCalculateRanksLaunch`
(contact sort) + `ps_reorderPrimitiveContactsLaunch`, and upstream the self-collision
contact-gen/solve chain (`ps_selfCollisionLaunch`, `ps_primitivesBound*`,
`ps_primitivesCollisionLaunch`, `ps_contactPrepareLaunch`, `ps_solvePCOutputParticleDeltaVLaunch`,
`ps_findStartEndParticle*`, `ps_accumulateDeltaVParticleLaunch`, `ps_updateParticleLaunch`).

## P6.1 — particle contact-sort kernels on the engine ABI (stage-7 sort BDA-rewritten)

Brought the three particle contact-sort kernels onto the BDA engine ABI. They were
manifest-mapped but only to Phase-1 HARNESS shaders (resource-binding + scalar
push-constants: `radixSortGather.slang`'s `GatherParams{numKeys,lowBit,gridDimX}`
over `RWStructuredBuffer` bindings, `clampValues.slang`'s index scheme) which the
engine-ABI dispatcher cannot drive — so they refused. A *bloße* ABI row would not
have worked; each needed a real BDA rewrite whose push-constant IS the CUDA
argument pack of device pointers, exactly like the already-live
`radixSortMultiBlockLaunchWithCount` (`radixSortDesc.slang`). New shader
`particleContactSortBda.slang` (ported line-for-line from the .cu):
- `radixSortDoubleCopy` (`radixSortImpl.cu:187`, ABI `{8,8,8,8,8,8,8}` = 56B) —
  two rank-indexed 64-bit gathers; launched at `PxgParticleSystemCore.cpp:1268`.
- `radixSortDoubleCopyHigh32Bits` (`radixSortImpl.cu:139`, `{8,8,8,8,8,8,8}`) —
  the twin gather writing only the high word (`key.y`); launched at `:1207`.
- `clampMaxValue` (`MemCopyBalanced.cu:88`, ABI `{8,4}`) — in-place clamp of the
  contact counter to `mMaxContacts`; launched at `:1105`.
`numKeys` is a `const PxU32*` device pointer (read `*numKeys` via `Ptr<>`), not a
host scalar. Grid-stride uses `WorkgroupCount().x * 1024` (= `BlockDim::RADIX_SORT`,
matching `numthreads(1024)`), so the kernels are Vulkan-only and excluded from
`PXG_HLSL_ENTRIES`. The manifest rows were *re-pointed* to `particleContactSortBda.spv`
(the harness `radixSortGather.slang`/`clampValues.slang` stay for their existing
unit tests), so `PXG_SLANG_KERNEL_MAPPED` stays **156** (no new mapping, only a
re-point + new ABI rows).

**`PXG_SCENE=particle` trace: all three flip REFUSED→OK ×120/frame; the only
remaining refused particle launch is `removeContactManagers_Stage5` (×2, narrowphase
pair-mgmt off the bag path — deferred).** Free-fall byte-unchanged (PGS 19.78 m,
exit 0). New unit test `gpuslang.particleContactSort` bit-exact in both modes; suite
**666/666** both memory modes. Rigid regression: all 8 scenes × 2 solvers = 16/16
`exit 0` — the rigid-shared radix gather is byte-unaffected (those scenes don't
dispatch it, but re-verified, not assumed).

Scoping note: a 6-agent workflow mapped the full PBD settling pipeline. The grid-sort
machinery (`radixSortMultiBlock…WithCount`, `CalculateRanks`, `initializeRadixRanks`)
is already live; the roadmap to a settling bag is P6.2 grid build + reorder (incl.
authoring the un-ported `radixSort…WithoutCount` pair) → P6.3 self-collision → P6.4
plane/primitive contact → P6.5 settling-bag scene + fall-then-rest harness.

## P6.0 — MILESTONE: the first live particle frame — a PBD particle cloud free-falls on Vulkan

The first-ever deformable/particle frame on the Slang/Vulkan (llvmpipe) engine
path: a PBD particle cloud now free-falls under gravity and steps live, on both
solvers, `exit 0`, with the full suite and the entire rigid regression staying
green. This opens the Phase-2 runtime for the deformable/particle subsystems
(previously 0 kernels mapped; Phase-1 math was already ported + unit-tested).

Four `ps_*` kernels were brought up on the BDA engine ABI, ported line-for-line
from `gpusimulationcontroller/src/CUDA/particlesystem.cu`:
- `ps_updateUnsortedArrayLaunch` (ABI `{8,8}`) — the INGEST. Copies the user
  `PxParticleBuffer` into the internal `mUnsortedPositions_InvMass`. **Load-bearing
  discovery:** without it that array stays zero → `invMass==0` → `preIntegrate`'s
  `tPos.w!=0` guard skips every particle → no fall at all. The scoped "3 kernels"
  cannot produce an observable; this ingest kernel is the required 4th. Fires once
  (frame 1, buffer flags dirty); thereafter preIntegrate drives the fall.
- `ps_preIntegrateLaunch` (ABI `{8,8,12,4,1,1}`) — the gravity predict on the
  UNSORTED arrays: `vel = (vel + gravity*gravityScale*dt)*damping`,
  `pos += vel*dt*fwdScale` (fwdScale 0.5 TGS / 1.0 PGS). The free-fall driver.
- `ps_integrateLaunch` (ABI `{8,8,4,4}`) — post-solve writeback from the SORTED
  state. On the contact-free free-fall path the un-ported reorder never fills the
  sorted buffers (zero on fresh `vkAllocateMemory`) → `mSortedDeltaP==0` → locked →
  `pos4.w==0` guard → inert no-op that neither corrupts nor faults. A
  behaviour-preserving `sortedIndex < numParticles` bound guard hardens the one
  scatter write against the un-ported-reorder garbage case.
- `ps_updateUserBufferLaunch` (ABI `{8,8,4}`) — egress back to the user buffer so
  the host DtoH read observes the fall.

First-ever `PxgParticleSystem` BDA round-trip: `pxgSlangParticleDesc.slang` mirrors
`PxgParticleSystemGpu` (656 B, `mSortedDeltaP@160`/`mData@248`) +
`PxgParticleSimBufferGpu` (88 B, `mFlags@72`), validated three ways — SPIR-V
`ArrayStride 656`/`88`, `offsetof`/`static_assert` against the real engine
`PxgParticleSystem` compiled in the test TU, and a sentinel-filled device
round-trip per kernel (bit-exact).

**`PXG_SCENE=particle` free-fall (cloud from y≈100.75 over 2.0s, both solvers,
`exit 0`):** TGS drops 9.89 m (fwdScale 0.5), PGS drops 19.78 m ≈ true free fall
½·9.81·2² = 19.62 m — the drop ratio validates the `isTGS` bool marshalling.
`ps_updateUnsortedArrayLaunch` OK ×1, the other three OK ×120 each; the remaining
grid/sort/collision/solve launches refuse cleanly (non-fatal). A particle-only
scene SIGSEGVs the rigid solver (`rigidSumInternalContactAndJointBatches2`
null-derefs with zero dynamic rigids), so the scene keeps a proven box-on-ground
rigid config at y≈0.5 far below the cloud.

Suite 665/665 both memory modes (`gpuslang.particleSim` 5/5 bit-exact both modes);
`PXG_SLANG_KERNEL_MAPPED = 156`. Rigid regression unaffected: all 8 scenes ×
2 solvers = 16/16 `exit 0`, byte-for-byte identical.

## P5 Step 5 — MILESTONE COMPLETE: all eight common rigid scenes hold on BOTH solvers

Tranche 13 converted the last bound-buffer kernel in the joint path —
`jointConstraintBlockPrepareParallelLaunchTGS` — to the BDA engine ABI
(ABI `{8,8}` = `kTwoPtr`, mirroring the same-file contact-TGS twin and the
tranche-9 PGS joint twin; the joint port reads stepDt/stepInvDt/dt/invDt/
lengthScale from the shared desc and `erp = 0.5 * biasCoefficient` from the
prepare desc; `orthogonalize`/`preprocessRows` inertia reads plumbed through
`solverBodyTxIDataPool`).  The kernel was already mapped; only the ABI row was
missing (tranche 12 had deliberately left it out so the un-converted kernel
refused cleanly instead of crashing).

**`PXG_SCENE=joint PXG_SOLVER=tgs` now HOLDS at y=3.000, |v|=0.000, exit 0,
ZERO refused launches** (was 240 refusals/step + fall-through to y=-75).

**The common-rigid-scene milestone is COMPLETE: all eight scenes — box, stack
(3 dynamic boxes), sphere, plane, capsule, capsule-on-box, convex-mesh, and a
D6-jointed box — fall/hold and REST on the Slang/Vulkan (llvmpipe) GPU pipeline
under BOTH solvers (TGS + PGS): 16/16 scene×solver combinations, each `exit 0`
with ZERO refused kernel launches.**  Suite 664/664 both memory modes;
`PXG_SLANG_KERNEL_MAPPED = 152`.

Coverage now spans all four PhysX rigid primitives (box/sphere/capsule/plane),
convex meshes, primitive-vs-primitive + convex-convex GJK/EPA narrowphase,
box-plane, both TGS and PGS solver cores incl. static-block + mass-splitting
reference counts, and D6 joints — the full common-rigid-body pipeline.

Explicitly out of this milestone (documented deferrals): force-threshold
reporting (RADIXSORT_SINGLEBLOCK/CALCULATERANKS + the threshold group),
convex-vs-trimesh / heightfield contact pipelines, and all articulation /
deformable / particle paths (Phase-1 math ported + unit-tested, Phase-2 runtime
not started — 0/56 softbody, 0/65 particle, 0/57 cloth kernels mapped).

## P5 Step 5 — earlier: a jointed scene holds on the GPU (PGS); TGS joint-prep was the one remaining gap

Tranche 12 brought up the D6-joint data-prep on the engine ABI:
`constraint1DPrePrepLaunch` (the last unmapped joint kernel; ABI `{8,8}`,
thread-per-joint; rigid arm = `constraint1DPrePrep<false>` row construction
transcribed bit-exact, arti arm deferred) plus a discovered unmapped upstream
producer `updateJointsLaunch` (`{8}`, the CPU-pinned→GPU joint-pool scatter
without which the joint pools stay `0xFFFFFFFF`).  `PXG_SLANG_KERNEL_MAPPED`
150 → 152.  The harness now compiles two extension TUs (`ExtJoint.cpp` +
`ExtD6Joint.cpp` → `PxD6JointCreate`; clean, no serialization/PVD/metadata
drag).  New engine-ABI unit test `npD6JointPrep_engineKernel` (real
`PxgPrePrepDesc`/D6 joint data → bit-exact `Px1DConstraint` rows, incl. the
`ra = (raInner + pose0.p) − pose0.p` round-trip that is NOT algebraically
`raInner` under the CUDA `NoContraction` rounding).

`PXG_SCENE=joint`: a dynamic box suspended from a static world anchor by a
`PxD6Joint` (linear DOFs locked, spawned at y=3.5, pinned to y=3.0):
- **PGS: holds at y=3.000, |v|=0.0000, exit 0, ZERO refused launches.**
- TGS: PARTIAL — `jointConstraintBlockPrepareParallelLaunchTGS` is still in
  bound-buffer form (never converted to the BDA engine ABI, unlike its
  now-converted PGS twin; `constraintPrepTGS.slang:53-54` flags it "pending
  its own conversion").  It cleanly REFUSES (no crash) and the box falls
  through.  Converting it (mirror the contact-TGS twin: 11 bound buffers →
  `Ptr<>` desc reads + `orthogonalize`/`preprocessRows` + its unit test) is
  the next tranche.

Suite 664/664 both memory modes.

## P5 Step 5 — earlier milestone: the four rigid primitives + convex meshes rest on BOTH solvers

Tranche 11 assembled the CONVEX-HULL arm of collide() (`generateConvexContacts`,
cudaGJKEPA.cu L1458-1462) into `npCvxCollide.slang` (now 1707 lines): the T144
reference-frame setup, T139 calculateLocalNormal, T138 getPolygonIndexFromLocal-
Witness (generalised to both hulls), T145 polyClip + addContacts, T146 emit
tails, T137 schlock::reduce, and T140 addManifoldPoint (match→replace /
append<4 / full+matchless→5-candidate reduce) — transcribed from their verified
oracles, single-thread, into collide()'s else arm.  The stage2 gate is now the
exact `.cu` `if (transformsValid)` (L2616) — sphere, capsule AND convex-hull
type0 all run the assembled collide().  New intentional deviations
(if-conversion speculation hazards on llvmpipe — the null-vertex-source and
past-nbVerts blob reads are made dereferenceable and select-discarded) are
documented in-source with `.cu` line evidence.

`PXG_SCENE=convex` cooks an 8-vertex cube at runtime (immediate-mode
`physx::immediateCooking::createConvexMesh`, `buildGPUData=true`) and drops it
on the ground box → it **rests at y=0.5000, |v|=0.0000** through the real
GJK/EPA + polygon-clip contact path.

**The rigid milestone is met: all four PhysX rigid primitives (box, sphere,
capsule, plane) PLUS convex meshes, singly and stacked, fall and REST on the
Slang/Vulkan (llvmpipe) GPU pipeline under BOTH solvers (TGS and PGS) — 14/14
scene×solver combinations, each `exit 0` with ZERO refused kernel launches.**
Suite 664/664 both memory modes.  New `testCvxCollide` convex-hull fixtures
(face-touch / deep-EPA / rot-45 edge / replace-vs-reduce manifold / mutation
canary) on the engine ABI with a real cooked box hull.

Remaining (out of the common-rigid-scene milestone; documented deferrals):
jointed scenes (`constraint1DPrePrepLaunch` D6 data-prep — scoped, next),
force-threshold reporting, convex-vs-trimesh / heightfield contact pipelines,
and all articulation / deformable / particle paths.

## P5 Step 5 — earlier milestone: six scenes rest on BOTH solvers (TGS + PGS)

Tranche 10 completed the PGS path to scene-readiness: preIntegrationLaunch
(88-B pack, no skip-gravity bool), integrateCoreParallelLaunch, both PGS
constraint-prep entries ({8,8} desc packs), NEW solveStaticBlock +
propagateStaticSolverBodyVelocities (the static-ground path; {8,8,4,4,4,1} /
{8,8,4,4,4}), dmaBack pair, NEW constraint1DBlockPrePrepLaunch (launched with
the TOTAL batch count even in contact-only scenes — contact warps no-op like
the .cu) and clearFrictionPatchCounts.  10 new ABI rows;
`PXG_SLANG_KERNEL_MAPPED` 146 → 150; harness tests converted to shim+desc
form.  gpu-sdk-step gained `PXG_SOLVER=pgs`.

**All SIX scenes (box, stack, sphere, plane, capsule, capsulebox) now fall
and REST under BOTH solvers — 12/12 combinations, each `exit 0`, zero
refused launches.**  Suite 664/664 both memory modes.

Known PGS/TGS deferrals (documented, latent for the proven scenes): the
force-threshold reporting group (+ RADIXSORT_SINGLEBLOCK/CALCULATERANKS) only
launches with force-threshold pairs; `constraint1DPrePrepLaunch` (D6 joint
data prep) only with joints — a jointed scene is the next bring-up; all
articulation launches remain deferred.

## P5 Step 5 — earlier milestone: six scenes on TGS; collide() assembled; PGS core on the engine ABI

Tranche 8 (collide() assembly) closed the convex-convex frontier for
sphere/capsule type0: the new `npCvxCollide.slang` module (1170 lines) wires
the verified GJK-warm-start (T133) + EPA (T132) + post-GJK decision (T149) +
capsule face/EE contact gen (T136/T143) + prep/warm-seed/repack (T148)
transcriptions into a `public bool collide(...)` called from stage2Body,
gated on `transformsValid && (type0 == EGT_SPHERE || EGT_CAPSULE)` — the
convex-hull type0 arm keeps the byte-identical elision (its
`generateConvexContacts` driver is the remaining deferred piece).  Contact
points are written inside collide() straight into the manifold (localNormalPen
@0 / A @64 / B @128); the epilogue still stores only moreData words 9-12 and
warm words 13-15 stay read-never-stored (the .cu quirk, pinned by tests).
`PXG_SCENE=capsulebox` — capsule dropped onto the ground BOX, routed through
the real GJK/EPA chain — now **rests at y=0.5000, |v|=0.0000, exit 0, zero
refused launches**.  New `testCvxCollide` engine-ABI integration test (real
cooked box hull; touch/near/deep-EPA/distant/EE/sphere fixtures + mutation
canary; manifold word semantics pinned).

Tranche 9 (PGS core on the engine ABI) fixed the latent PGS gap: all six
mapped solverPGS kernels were harness-form with no ABI rows.  Converted
solveBlockPartition {8,8,4,4,1}, concludeBlocks/writebackBlocks/
writeBackBodies {8,8,4}, computeAverageSolverBodyVelocity/
propagateSolverBodyVelocity {8,8} (+ ZeroBodies PGS) to BDA, added
markActiveSlabPGS {8,8,4,4,8} (new map; PXG_SLANG_KERNEL_MAPPED 145→146),
PxgSolverSharedDescPGSGpu, and shim+desc tests.  PGS structural differences
honored: k*128 partition staging, 2-stripe deltaV pool, no elapsedTime/minPen,
real concludeBlocks work, (32,8,1) averaging blockDim.  Remaining for a PGS
scene: preIntegration, PGS constraint prep, integration, static-block pair,
dmaBack (tranche 10, in flight).

## P5 Step 5 — earlier milestone: five common rigid scenes rest on the GPU, zero refused launches

`gpu-sdk-step` grew a `PXG_SCENE`-dispatched scene table; **all five rigid
scenes now fall and REST on the Slang/Vulkan (llvmpipe) pipeline with ZERO
refused kernel launches**, each `exit 0`:

- `box`    — dynamic box on a static ground box (the Step-4 scene).
- `stack`  — three stacked dynamic boxes: `final y = 0.500 / 1.500 / 2.500`,
  `|v| = 0.000` (exercises dynamic-vs-dynamic box contacts + partitions).
- `sphere` — dynamic sphere on the ground box: rests at `y = 0.500`.
- `plane`  — dynamic box on a `PxPlaneGeometry`: rests at `y = 0.500`.
- `capsule`— dynamic capsule on its side on a `PxPlaneGeometry`: rests at
  `y = 0.500` (routes through the sphere/"primitive" narrowphase `planeCapsule`
  sub-case — the already-mapped `SPHERE_KERNEL_MAIN`, no new kernel).

Suite **666/666** both memory modes; `PXG_SLANG_KERNEL_MAPPED = 145`.
Deformables / particles / articulations remain explicitly out of scope for this
milestone.

**Next frontier — assemble `collide()` into convex-convex stage2.**  A `capsule`
(or `convex`) resting on a BOX ground routes through the convex-convex GJK
narrowphase (`CONVEX_CONVEX_KERNEL_EARLY_OUT` / `_MAIN` + `FINISH_CONTACTS_KERNEL`),
not the primitive kernel.  A `capsule`-on-box probe fell straight through the
ground (`y = -73.8`, no contacts) — **but a three-lens investigation
(wf_1b2513f5) proved the gap is a single intra-kernel elision, not a missing
kernel.**  Routing (`caseCapsuleBox` → `eConvex` bucket → `testSDKConvexConvexGjkEpaGpu`,
`PxgNphaseImplementationContext.cpp:1354/1441`), the box hull blob
(`layOutBoxHull`, host-produced, already live), stage1 refresh
(`npCvxStage1Refresh.slang`), `finishContactsKernel` (`npCvxFinishContacts.slang`),
and lost/found (`prepareLostFoundPairs_Stage1/2`) are ALL complete ports; every
kernel in the path is mapped and launches.  The one gap: the MAIN kernel's
`collide()` (GJK + EPA + `generateConvexContacts` + `schlock::reduce`) is elided —
`npCvxStage2Core.slang:177` hard-codes `bIntersection = false` and stores
`numManifoldContacts | 0x80000000` (the invalidated/no-touch bit), so every
convex-convex pair emits zero patches (`npCvxStage2Setup.slang:51-61` documents
this).  The `.cu` original `cudaGJKEPA.cu:2615-2632` calls `bIntersection =
collide(...)` (driver at `cudaGJKEPA.cu:1275`).

The verified collide() COMPONENTS are all ported (tranches 129/132/133/147/148/149)
and staged as `npCvxCoreGjk` / `npCvxCoreEpa` / `npCvxCoreCvxGen` /
`npCvxCollidePrep` / `npCvxCollideDecide` / `npCvxContactDecide` / `npGjk*` /
`npEpa*` / `npManifold*` / `npCvxCoreSupport`.  Remaining = wire them into a
`collide()` driver and call it at the `npCvxStage2Core.slang:177` site.  Ordered
plan:

- **T0** (throwaway smoke test — DONE, downstream CONFIRMED): injecting one
  fabricated upward manifold contact (box-local normal `(0,1,0)`, contactB
  `(0,0.5,0)`, `penetration.w = 0.5` so finishContacts' `sep = w - radius = 0`)
  + `bIntersection=true` at `npCvxStage2Core.slang:177` made the capsule-on-box
  scene **rest at y=0.5000, |v|=0.0000** (was falling through to −73.8).  This
  proves the ENTIRE convex-convex downstream — finishContacts patch atomics →
  solver → contact response → lost/found — is live on llvmpipe, and that the
  ONLY missing piece is stage2's contact CONTENT (the elided collide()).  The
  probe was reverted; the manifold frame it pinned down (shape1-local normal /
  contactB, `sep = w - radius`) is the target format for the real collide().
- **T1** support-mapping + `runGJK` over shape0 (implicit: capsule 2-vertex
  segment / sphere point / hull) vs shape1 (box hull blob).
- **T2** EPA (penetration depth + normal) for the overlapping/resting case.
- **T3** `generateConvexContacts` (face/edge clip → manifold points).
- **T4** wire `schlock::reduce` (internal to stage2 — no separate kernel).
- **T5** replace `bIntersection=false` with the collide() result + store
  `numManifoldContacts | (bIntersection ? 0 : 0x80000000)`; update the stub
  header.
- **Wrinkle**: stage2Body runs SERIALIZED ON LANE 0 (llvmpipe has no 32-wide
  subgroup), whereas the `.cu` collide() is warp-cooperative — the assembly must
  either serialize the GJK/EPA/contact-gen or route their warp reductions
  through the groupshared-reduction pattern used elsewhere.
- **Blocker note**: only capsule/sphere/box primitives are on this critical
  path (box hull already live; capsule/sphere are implicit, `hullOrMeshPtr==0` is
  harmless).  Arbitrary user `PxConvexMesh` operands additionally need the
  host-cooked hull (`addHull`/`layOutHull`) uploaded — CPU cooking, separate work.

Once collide() lands, capsule-on-box / convex-hull-on-box / convex-convex all
light up together, and a convex-hull ground scene can be added.

Landed since the Step-4 box-on-ground milestone (each: bit-exact / geometric
shim unit test on the real engine ABI + both-mode suite + engine-scene proof):
- tranche 1: `gpu-sdk-step` scene table (box/stack/sphere/plane builders,
  per-target fall→rest assertions); stacked-boxes scene rests.
- tranche 2: fixed the inverted `constraintType` encoding in the **PGS** solver
  (solveBlockPartition / concludeBlocks / writebackBlocks) — the same
  eCONTACT=1 vs legacy contact=0 bug fixed for TGS in Step 4, via
  `pxgIsContactType()`.
- tranche 3: `solveBlockUnified` on the engine ABI — the per-partition
  velocity-iteration rigid solve (`solveBlockPartitionTGSInternal`) the engine
  launches once per partition; stacked boxes solve cleanly.
- tranche 4: `sphereNphase_Kernel` (sphere narrowphase) on the engine ABI —
  sphere rests on the ground.
- tranche 5: `convexPlaneNphase_Kernel` (convex/box-vs-plane narrowphase) on
  the engine ABI — box rests on a plane.
- tranche 6: `markActiveSlabTGS` on the engine ABI — the per-partition
  active-slab marking (`precomputeReferenceCount`) that populates
  `solverEncodedReferenceCount` for mass-splitting reference counts.  Ported
  `checkActiveContactBlockTGS` (read-only "would this contact still push?"
  predicate; no ref-scaling, no velocity mutation) + the slab-bit `atomicOr`.
  Closes the **last non-fatal refused launch on the stack scene**: all four
  scenes now step with zero `fail to launch`.  Note the arg order differs from
  `solveBlockUnified` (minPen precedes elapsedTime) → ABI `{8,8,4,4,4,4,8}` =
  40 bytes; the `.cu` reads `mBodyCount`/`mArticulationCount` from
  `islandContextPool[0]` (mirrored exactly).
- tranche 7: `capsule`-on-plane scene — the capsule primitive now rests on the
  GPU via the already-mapped sphere/"primitive" narrowphase (`planeCapsule`).
  Also probed the convex-convex path (capsule-on-box) which exposed the
  next-frontier gap above (kernels launch, no contacts).  No new kernels.

## P5 Step 4 — MILESTONE ACHIEVED: the box falls AND RESTS on the GPU

As of tranche 19 (commit 0d547fa) `PXG_GPU_STEP=1 ./gpu-sdk-step` steps the
full unmodified PhysX SDK scene end-to-end on the Slang/Vulkan (llvmpipe)
pipeline: a dynamic box FALLS under gravity and RESTS on a static ground box
— `box start y=5.000, y after 1 step=4.997, final y=0.500, final |v|=0.000`,
`[ OK ] box fell + rested`, exit 0, no crash.  The entire per-frame
collision→solve loop now runs on the GPU: broadphase (incremental SAP) →
narrowphase (boxBoxNphase) → lost/found contact report → contact prep →
TGS solve (whole-island + static-contact solve) → integrate.  Suite 665/665
both memory modes; PXG_SLANG_KERNEL_MAPPED = 141.

Landed to close the contact→solve path since the free-fall milestone:
- tranche 17: incremental-SAP broadphase (markUpdated/incremental histograms/
  performIncrementalSAP) on the engine ABI — per-frame moved-shape sweep.
- tranche 18: updateFrictionPatches + prepareLostFoundPairs_Stage1/2 — the
  generated contact reaches the solver island.
- tranche 19: TGS static-contact solve (solveStaticBlockTGS +
  propagateStaticSolverBodyVelocitiesTGS + concludeBlocksTGS on engine ABI),
  plus two solver correctness fixes exposed by the first real contact:
  (a) constraintType encoding — solveWholeIslandTGS used the inverted legacy
  contact=0 convention; the engine writes eCONTACT=1 (eARTI_CONTACT=3), so a
  real contact was solved as a 1D joint and crashed; now discriminates with
  pxgIsContactType(); (b) llvmpipe zero-trip hoist — a static-only island
  (mNumPartitions==0) left the dynamic constraint loop zero-trip but a static
  contact had allocated iter_blockConstraintBatch, defeating the old guard;
  the scalar guard now also requires mNumPartitions!=0.

As of tranche 20 (commit 852abac) the box-on-ground scene steps with **ZERO
refused kernel launches**: `writebackBlocksTGS` (constraint-force /
friction-patch / threshold-stream writeback) was converted to the engine ABI
({8,8,4}, same shape as concludeBlocksTGS), closing the last harness-form
solver kernel.  The entire per-frame collision→solve→writeback loop for this
scene now runs cleanly on the GPU.  Suite 664/664 both modes.

## P5 Step 4 — earlier MILESTONE: the box falls on the GPU

As of tranche 12 (commit 250bb68) `PXG_GPU_STEP=1 ./gpu-sdk-step` shows the
box FALLING under gravity on the GPU: `box start y=5.000, y after 1 step
=4.997`, free-fall integration end-to-end, ZERO refused launches on the
rigid free-fall path, suite 679/679 both memory modes.  Landed since the
last anchor:
- tranche 10: TGS solver frame set (initStaticKinematics, ZeroBodiesTGS,
  initializeSolverVelocitiesTGS, computeAverage/propagate/writeBackBodiesTGS,
  integrateCoreParallelLaunchTGS).
- solveWholeIslandTGS crash fix: for a constraint-free island PhysX leaves
  the block-constraint pointers null; the llvmpipe optimizer speculatively
  hoisted the batches[] load out of the zero-trip loop and faulted.  Gated
  the solve loops on a non-null batch pointer (commit 707d763).  Debug seam
  PXG_SYNC_EACH (flush after each dispatch) added to the shim to pin a
  shader fault to the exact launch.
- tranche 11: simcontroller updateBodies/Shapes/Transform + PxgSimulationCore
  /PxgNewBodiesDesc mirrors (commit b0875db).
- tranche 12: preIntegrationLaunchTGS (the TGS gravity kernel) — shared
  preIntegrateBodyCompute helper + BDA entry; paramAlign/abiAlign fixed to
  the largest-power-of-two-divisor rule for the 12-byte by-value PxVec3
  gravity (commit 250bb68).

REMAINING to make the box REST on the ground (the contact pipeline; the box
currently falls THROUGH — final y=-73.8, |v|=39.2 — because no contact
constraint is generated).  The box-vs-box (dynamic box + static ground box)
contact path, from the engine trace's "Could not find registered CUDA
function" set (rigid-only; ignore cloth/particle/arti/bvh/sdf):
- `boxBoxNphase_Kernel` — UNMAPPED, no worktree yet.  Box-box primitives
  were ported in tranches 47-52 (npBoxBox*.slang helpers exist); needs the
  BDA entry + ABI row + mapping + engine-ABI test.
- contact/joint PREP set — staged UNVERIFIED in worktree
  `.claude/worktrees/agent-ac3f057108a97b50f`: constraintContactBlockPrePrep-
  Launch, constraint1DPrePrepLaunch, constraint1DBlockPrePrepLaunch,
  rigidSumInternalContactAndJointBatches1/2, contactConstraintBlockPrepare-
  ParallelLaunchTGS + pxgSlangPrepDesc.slang (PxgPrePrepDesc/PxgConstraint-
  PrepareDesc mirrors) + npPrepDescProbe + testPrepDescLayout.  Merge like
  the simcontroller worktree: git diff -> apply -> ABI rows -> retire HLSL ->
  reconcile PXG_SLANG_KERNEL_MAPPED -> suite both modes -> engine trace.
- contact STREAM/compress/finish set — staged UNVERIFIED in worktree
  `.claude/worktrees/agent-a734706e687bb0590`: npCompressContacts,
  npCompressContacts2, npCvxFinishContacts, npCvxStage1Refresh,
  npCvxStage2Setup + pxgSlangNpAbi.slang + npNpAbiProbe + testNpAbiLayout.
  (convexConvex path — needed for general convex; compress/finish are on the
  shared contact-stream path.)
- `clearFrictionPatchCounts` — UNMAPPED, needs port.
- Already mapped and working: solveWholeIslandTGS (solves the contact once
  batches exist), contact/jointConstraintBlockPrepareParallelLaunchTGS.
UPDATE (tranches 13-15 merged, commits 7d0c860/48e3e2f/5c44c1c): the contact
GENERATORS + prep + stream-compress are now all on the engine ABI —
rigidSum1/2, constraintContactBlockPrePrep, contactConstraintBlockPrepareTGS
(tranche 13); convexConvex stage1/2, finishContacts, compress1/2 (tranche 14);
boxBoxNphase_Kernel (tranche 15).  Suite 675/675 both modes, PXG_SLANG_KERNEL_
MAPPED = 136.  BUT the engine step still falls through and boxBoxNphase
launches ZERO times: the narrowphase DRIVER never dispatches.  Per-frame the
trace shows only updateTransformCacheAndBoundArray + the solver; broadphase
ran once (setup), narrowphase never.  So the true remaining frontier is the
whole NP DRIVER pipeline that turns a broadphase pair into a solved contact,
NOT the generators (which are ready):
- `clearFrictionPatchCounts` (NP per-frame reset) — unmapped
- `midphaseGeneratePairs` / `convexHeightFieldMidphase` (midphase pair gen) — unmapped
- `removeContactManagers_Stage1-4` (contact-manager lifecycle) — unmapped
- the per-frame broadphase re-run + contact-manager management + the NP
  dispatch driver (PxgNarrowphaseCore) — investigate why NP isn't dispatched
  at all (a single top-of-pipeline gate kernel may unblock the chain).
Next continuation: trace-driven — map the NP-driver entry kernels one tranche
at a time until boxBoxNphase actually launches and a contact constraint
reaches solveWholeIslandTGS; then the box rests.

Merge doctrine (proven 6x this session): worktree agent -> git diff HEAD >
patch -> git apply in main tree -> add reported ABI rows to bridge/
PxgVulkanKernelAbi.cpp -> retire reported PXG_HLSL_ENTRIES -> reconcile
PXG_SLANG_KERNEL_MAPPED -> rebuild both build trees -> suite both modes ->
gpu-sdk-step trace.  End state: box falls AND rests (finalY in [0.5,1.5],
|v|~0).

## P5 Step 4 — live bring-up state (continuation anchor)

Converted to the engine ABI (BDA + ABI row + shim-driven test, all green;
suite 681/681 both memory modes, PXG_SLANG_KERNEL_MAPPED = 127):
- the ENTIRE broadphase void** set (0 refused BP kernels in the engine trace):
  setup/foundation, radix desc pair, endPt/region/start-region/active-region/
  overlap histograms, createRegions, newBoundsPairs, reports.
- tranche 10: the TGS solver every-frame set — initStaticKinematics
  {4,4,8,8,8,8,8,8}, ZeroBodiesTGS/initializeSolverVelocitiesTGS/
  computeAverageSolverBodyVelocityTGS {8,8}, propagateAverageSolverBodyVelocityTGS
  {8,8,1,1,4}, writeBackBodiesTGS {8,8,4}, integrateCoreParallelLaunchTGS
  {4,8,8,8,8,8} — driven by byte-exact PxgSolverCoreDescGpu (truncated @288)/
  PxgSolverSharedDescTGSGpu/PxgIslandContextGpu mirrors (pxgSlangSolverDesc.slang,
  80-probe testSolverDescLayout).
Engine trace after tranche 10: solver launches all OK; y becomes NaN because
solveWholeIslandTGS is unmapped (solver outputs never written); refused set =
updateTransformCacheAndBoundArrayLaunch {8}, updateBodiesLaunch {8},
updateShapesLaunch {8,8,4}.
Infrastructure DONE: pinned-memory seam (p3ShimPinned), both launch forms
(sized + void** via PxgVulkanKernelAbi table), desc-mirror + sentinel-probe
pattern (bpDesc 57 probes, solverDesc 80 probes), PXG_LAUNCH_TRACE.

IN FLIGHT (5 worktree agents; at merge: apply diff, add reported ABI rows to
bridge/PxgVulkanKernelAbi.cpp, reconcile PXG_SLANG_KERNEL_MAPPED (127 + sum of
reported deltas), retire reported PXG_HLSL_ENTRIES rows, rebuild both build
trees, suite both modes, engine trace, commit):
1. solveWholeIslandTGS {8,8,4,1,4,4} — 256-thread single-block port; shared
   staging capped 944->640 (llvmpipe 32KB groupshared limit; host threshold in
   PxgTGSCudaSolverCore.cpp guarded by PXG_NO_CUDA_TOOLKIT); desc mirror
   extended past 288 with threshold counts + Dy::ErrorAccumulator tail.
2. simcontroller trio: updateTransformCacheAndBoundArray/updateBodies/
   updateShapes + PxgSimulationCoreDesc/PxgNewBodiesDesc mirrors
   (pxgSlangSimCoreDesc.slang).
3. NP pair management: prepareLostFoundPairs_Stage1 {8,8,8,4}/Stage2,
   removeContactManagers_Stage1-4 {8} (+Stage5 pair if not yet BDA),
   initializeManifolds {8,8,4,4} + PxgPairManagementData mirror.
4. contact prep: rigidSumInternalContactAndJointBatches1/2,
   constraintContactBlockPrePrepLaunch, contactConstraintBlockPrepareParallel-
   LaunchTGS + PxgPrePrepDesc/PxgConstraintPrepareDesc mirrors
   (pxgSlangPrepDesc.slang).
5. NP convex-convex core: convexConvexNphase_stage1Kernel {4,4,8,8,8,8,8},
   stage2Kernel {4,4,8,8,8,8,8,8}, finishContactsKernel (20 args),
   compressContactStage1/2 — raw-pointer ABIs, element-struct stride probes.

These five cover every kernel the box-falls-and-rests scene is known to need
(wrangler init warnings + trace analysis); anything still refused after the
next trace gets the same treatment.  AABB-manager aggregate set (spec group
aabbManager in scratchpad kernel-abi-specs.json) not needed for this scene.

Doctrine per kernel: BDA rewrite mirroring the .cu exactly (thread mapping,
null-branches), ABI row in PxgVulkanKernelAbi.cpp, shim-driven engine-ABI test
(EngineBlob pattern for multi-arg packs), mutation check, full suite both
modes BEFORE committing, gpu-sdk-step trace shows the kernel OK, commit.
End state: PXG_GPU_STEP=1 box falls + rests on GPU.
