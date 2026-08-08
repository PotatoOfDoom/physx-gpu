# Phase 2 runtime status

Progress tracker for the Vulkan production runtime (RUNTIME_PLAN.md P0–P7).
Phase 1 (bit-exact CUDA→Slang math port, 673 kernels) is closed; see
MIGRATION_STATUS.md.  Every item below is committed on
`claude/physx-cuda-slang-migration-mqbz7u` and verified against the standalone
gpuslang harness on llvmpipe (Vulkan 1.4) in BOTH memory modes — default
(host-visible) and `PXG_DEVICE_V2=1` (device-local + staged + timeline).

Test count: **697/697** green in both modes (`ctest -N` in `build-gpuslang`).
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

## P8.46c — convexTrimeshNarrowphase becomes an engine kernel

`convexTrimeshNarrowphase` (convexMesh.cu:621-759) on the engine BDA ABI, composed from the
six modules the preceding tranches built and verified one at a time: the stack carve, the
setup prologue, the hull-blob reader, the three SAT stages with their driver and face
selection, the clipper, and the emit tail (which pulls in the contact reduction).  Manifest
263 -> **264**.

### The three numbers that had to come out right

- The push-constant blob lands at **exactly 168 bytes** — the figure computed from the
  21-argument pack before a line of the shader existed.  It is not obvious: the `PxU32` at
  offset 120 forces eight bytes of padding before the four pointers that follow it.
- `spirv-val` is clean over the composed shader.
- `p4Manifest` passes, which checks the row, the entry-point name and the mapped counter.

168 bytes is under llvmpipe's 256-byte push range but **above Vulkan's guaranteed 128**, so
the kernel is portable only where the device reports more — and nothing in the runtime
enforces that cap.  Recorded here because no test can see it on this device.

### The first ABI row with a hostPtrMask

Argument 18, `maxTempMemRequirement`, is a pinned-host pointer, which is what P8.46a was
built for.  This is the first row in the table to set the bit.

### The dispatch shape, and the coverage it gives up

The CUDA is a grid-stride loop whose body performs a chained buffer-device-address load
(`hullOrMeshPtr` read out of the shape array and dereferenced) — the pattern P8.40 proved
segfaults llvmpipe.  **The obvious remedy is not available**: one workgroup per pair needs
the pair count, and that is a DEVICE value while the grid is computed on the host, with no
indirect dispatch reachable from the CUDA shim path.

So this takes the shape `convexTrimeshMidphaseBda` already ships green: one workgroup per
CUDA block, and a loop of CONSTANT trip count over that block's two warps.  That shader
performs a chained BDA load inside exactly this kind of loop and does not crash, which is
the evidence that the hazard is the grid-stride form and not loops as such.

**THE COVERAGE THIS GIVES UP IS REAL.**  The CUDA's grid-stride loop makes a second and
further pass when there are more pairs than grid slots; this shader makes exactly one.  The
ceiling is `2 * gridDim.x` pairs — for meshfloor, 8192 slots for at most two pairs, but a
scene dense enough to exceed it would lose contacts SILENTLY.  Closing it needs either
indirect dispatch in the shim or a host-side grid derived from the same device counter.

One warp per workgroup is likewise forced: the clipper and the reduction both require
`[numthreads(32,1,1)]` because their early-outs are warp-uniform but not group-uniform, and
llvmpipe hangs on a divergent barrier rather than erroring.  The CUDA's two warps per block
become two sequential iterations.

### The allocator pointers corrupt rather than fault

`maCompute` returns its seven addresses in ALLOCATION order, which is not the order of the
CUDA function's out-parameters, and every one of them points inside the same stack
allocation.  Writing `a.maxDepth` where `a.interm` belongs still lands in valid memory and
silently overwrites another section's records.  The six publishing stores are written by
NAME against the struct, never by position.  `a.pairs` is deliberately not published —
downstream consumers know that region as the stack base.

### What this does NOT yet do

The kernel is mapped and launchable, but `meshfloor` cannot step on it alone: four kernels of
the chain are still unmapped — `sortTriangleIndices`, `convexTrimeshPostProcess`,
`convexTrimeshCorrelate` and the finish-contacts kernel.  Until they exist the contacts this
kernel produces are never consumed, so the scene's behaviour is unchanged.  What HAS changed
is that its launch is no longer refused.

Suite **688/688 in both memory modes**, forced-fresh.  The count is unchanged from P8.46b:
this tranche adds an engine kernel, not a test suite, and there is no fixture that can drive
it — its inputs are six device-side blobs produced by kernels upstream of it.  What stands
behind it is the six module tranches, each swept and green on its own.

## P8.70 — the record structs the block fill reads from

The last ABI layer before `jcalc`'s record→block copy path can be written:
`PxgArticulationLink`, `Dy::ArticulationJointCore` and
`Dy::ArticulationJointCoreData`.  Plain arrays-of-structs this time, not SoA, so
only a stride and a field offset apply per access — but the fields themselves are
where the traps are.

**`PxgArticulationLink` is declared twice.**  Behind `#if !PX_CUDA_COMPILER`
(PxgArticulationLink.h:54-63) the host sees `PxVec3 initialAngVel; PxReal
penBiasClamp; PxVec3 initialLinVel; PxReal invMass;` and the device sees
`float4 initialAngVelXYZ_penBiasClamp; float4 initialLinVelXYZ_invMassW;` over
the same bytes.  `offsetof` can only see the host spelling, the kernels read the
device one, and the two agree only because 0/12/16/28 line up.  Generating the
constants from the host declaration and naming them after the device one is the
only way to get both halves right at once — reading either declaration by eye
gives you one of them.

**Not everything small is a word.**  `numPathToRoot` and `numChildren` are PxU16
sharing one word at 56/58; `disableGravity` and `retainsAccelerations` are single
bytes at 76/77; `dofIds`, `motion` and `invDofIds` are `PxU8[6]` runs at 520/526/
532 with `jCalcUpdateFrames` and `jointType` as single bytes at 538/539.  A word
read anywhere in there returns its neighbours.

**138 of 139 probed values were right on the first run.**  The one that was not is
worth keeping: `ArticulationJointCoreData::nbDof` sits at offset 4 in an 8-byte
struct, which reads like a `PxU32` — it is a **PxU8**
(DyFeatherstoneArticulationJointData.h:103).  The probe read a word and got
`0xEEEEEEED` against an expected `0xED`: the 0xEE fill plus the real byte.  A
generated offset is not a generated *width*, and the generator cannot catch that
because `offsetof` says nothing about size.  Only a functional probe against a
poisoned buffer does.

**Sweep: 24 of 24 killed** — every field moved onto its neighbour, all four
strides, both PxU16s, both single bytes, the per-axis array indexing, and the
`nbDof` byte read reverted to a word.  No survivors.

The probe now checks **139 values** across five block structs and three record
structs.  Suite **697/697** both memory modes; manifest unchanged at **270 of 501**.

## P8.69 — the joint-frame math, bit-exact against the engine itself

The first piece of real articulation **dynamics** in the port: `computeJointAxis`
(forwardDynamic2.cu:337) and `ArticulationJointCore::setJointFrame` /
`computeMotionMatrix` (DyArticulationJointCore.h:139-207), the head of `jcalc`.

It was taken first for one reason: both engine functions are `PX_CUDA_CALLABLE`,
so the fixture **calls the real engine implementation on the host** over the same
inputs and compares bit patterns.  There is no transcribed oracle to drift — the
strongest check available anywhere in this port, and it is what made the rest of
this entry possible.

**Bitwise, not tolerant.**  `rotateAndNormalize` is a hand-fused rotate-then-
normalize whose expression order is part of the result, and `getNormalized()` is
`m > 0 ? v * PxRecipSqrt(m) : 0` — a reciprocal square root with an explicit
zero-length guard.  The Slang side spells all of it out rather than reaching for
`normalize()` or `rsqrt`, and the module is registered in `PXG_NOCONTRACT_SHADERS`
so no multiply-add is fused behind the transcription's back.  13 joints across
every branch — prismatic, revolute, revolute-unwrapped, spherical at 1/2/3 dofs,
fix, a LOCKED axis (dofId 0xFF, the zero-axis input the normalize guard exists
for), and two cross-typed dof ids — all bit-identical.

### Five rewrites of correct code, and the measurement that ended it

The first run failed **55 of 242 values**.  So did the second, the third, the
fourth and the fifth — after a record-layout fix, a branch-free axis
construction, a switch from `out` array parameters to a returned struct, and
finally the removal of every array.  **Exactly 55 each time.**

The cause was none of those.  `PxArticulationJointType` is `eFIX = 0,
ePRISMATIC = 1, eREVOLUTE = 2, eREVOLUTE_UNWRAPPED = 3, eSPHERICAL = 4`, and the
Slang constants had been taken from the order the types appear in
`computeMotionMatrix`'s **switch** — off by one, with `eFIX` at the end instead of
the front.

What made that survive five attempts is the shape of the resulting failure.  Every
case that landed on a neighbouring branch which happens to compute the same thing
**passed**:

| case | host value | branch it hit | result |
|---|---|---|---|
| eREVOLUTE | 2 | the unwrapped label | same maths — PASS |
| eREVOLUTE_UNWRAPPED | 3 | the spherical label, dofs == 1 | same maths — PASS |
| eFIX | 0 | prismatic, with a zero axis | zero either way — PASS |
| ePRISMATIC | 1 | revolute — reads `.top`, not `.bottom` | zeros — FAIL |
| eSPHERICAL | 4 | fix — writes nothing | zeros — FAIL |

A subset of cases correct is the worst shape a bug can take, and every wrong
hypothesis it provoked was individually plausible.  What ended it was making the
shader **echo its own inputs back to the host**: one run showed `jointType = 1`
where the port expected `ePRISMATIC = 0`, and the whole pattern fell out at once.

*The rule, and it generalises past enums:* **when a fix does not move the failure
count, stop fixing and start measuring.** Five structurally different rewrites
produced the identical 55 — that constancy was the evidence, and it was on screen
from the second attempt.

*Also corrected here:* an intermediate version of this entry and of the shader
comment claimed that Slang `out`-array parameters and struct-of-array returns do
not survive lowering on this toolchain.  **That claim was wrong** — those forms
were rewritten while chasing the enum bug and neither was ever shown to be at
fault.  The array-free form is kept because it is what was in place when the real
cause surfaced, not because arrays are known broken.  A false platform rule left
in the record is worse than no rule.

**Sweep: 19 of 20 killed.**  Two first-round survivors were real gaps and were
closed by adding a rotational joint carrying a linear dof id and a fixed joint
carrying one — neither is a configuration the engine builds, but these are unit
tests of two functions, not of a scene, and the engine oracle handles both without
complaint.  The remaining survivor is `rsqrt` substituted for `1.0f / sqrt`: on
llvmpipe the two lower identically, so no fixture can separate them **here**.  The
spelling stays anyway — `rsqrt` is permitted to be an approximation, and the
constant is defence against a different driver, not against this one.

Suite **697/697** both memory modes.  Manifest unchanged at **270 of 501**: this
lands a shared helper, not a kernel.

## P8.68 — the link block, the last 29 KB of the map

`PxgArticulationBlockLinkData` is mapped: 47 fields across 29184 bytes, every one
generated from the engine header, probed at element [1] lane 3, and swept.  The
articulation block-layout map is now complete — `computeUnconstrainedVelocities1TLaunch`
and the `jcalc` chain are writable.

**Two irregular members, and both are the kind that a uniform-looking header hides:**

* **`mSpatialResponseMatrix` is a 6x6 block (9 float4 columns, 4608 bytes), not the
  7-column 3584-byte `PxgSpatialMatrixBlock` immediately before it.**  Two matrix
  blocks, adjacent, differently sized.  Reading the second at the first's stride is
  the natural mistake and it lands inside the same struct, so it corrupts quietly
  rather than faulting.  The probe reads column 8 — which does not exist in the
  7-column block — so the confusion cannot survive.
* **`mInvDofIds` is `PxU8[6][32]`, axis-major.**  Every neighbour in that byte run
  (`mDofs`, `mJointType`, `mDisableGravity`, `mRetainsAcceleration`) is a plain
  `PxU8[32]`, so the natural read returns axis 0 for every axis.  The probe reads
  axes 0, 2 and 5 with a distinct value per (axis, lane), which pins the row stride
  as well as the base.

**Sweep: 28 of 28 killed** — every field offset moved onto its neighbour, both
irregular members attacked at their specific weakness, and the struct stride tested
with and without the tail padding.  No survivors, so nothing here needs an excuse.

The probe now checks **106 values**.  Suite **696/696** both memory modes; manifest
still **270 of 501** — again, this maps no kernel.

## P8.67 — the articulation block-layout address map

`computeUnconstrainedVelocities1TLaunch` is a thin wrapper around a device
function that fills the **block-layout scratch**, and that scratch is the real
obstacle: `PxgArticulationBlockLinkData` alone is 29184 bytes, `PxgArticulationBlockDofData`
12800, `PxgArticulationBlockData` 7168.  Each holds thirty-two articulations
interleaved as structure-of-arrays, so every field access is three numbers —

```
block[warpIndex] . field [lane]
^ struct stride    ^ offset  ^ element stride
```

— and none is inferable from the others.  No articulation dynamics kernel can be
written until that map exists and is proven, so this tranche is the map, landed on
its own.  That is not a new precedent: `testSimCoreDescLayout`,
`testSolverDescLayout`, `testPrepDescLayout` and `testNpAbiLayout` are all standalone
layout proofs, and they are what made the kernel tranches after them safe.

**Every constant was GENERATED, not transcribed.**  A throwaway host program
emitted both the Slang constant block and the C++ `static_assert` list from the same
`sizeof`/`offsetof` expressions against the real engine headers.  Hand-copying fifty
offsets out of a 29 KB SoA header is precisely how P8.49a's BV32 stride bug
happened — shader and fixture carried the same wrong number and agreed with each
other instead of with the engine.  Generation removes that failure mode by
construction.

**Proven twice.**  Compile time: every Slang constant has a matching `static_assert`.
Run time: `artiBlockProbe` reads element **[1]** of a two-element array at lane **3**
and 57 values come back for a bitwise compare.  Both indices are deliberate — at
element [0] lane 0 the struct stride, the field offset and the lane stride all
collapse to the same address and the probe would prove nothing.  Element [0] is
filled with a *different* sentinel set rather than zeros, so a wrong stride reads
plausible wrong data instead of an obvious hole.

**Sweep: 40 of 40 killed.**  Of the first 37, thirty-five died.  The two survivors
were `PXG_AB_MAT33_COL1` and `PXG_AB_STB_P` — and they survived because **the probe
never read them**: `PxgMat33Block` and `PxgSpatialTransformBlock` occur only inside
`PxgArticulationBlockLinkData`, which this tranche does not map, so both were pinned
by size alone.  A constant nothing reads is a constant nothing can prove.  Both are
now probed standalone (with `mCol2` read as the `float[32]` it is, not the
`float4[32]` its neighbours are), and both mutants die, along with three new
sub-block mutants.  The earlier 35 were not re-run: the fixture only gained checks,
so a killed mutant cannot become invisible.

**What is deliberately NOT here.**  `PxgArticulationBlockLinkData` — the biggest of
the three and the one the link loops walk — is left for the next tranche.  Mapping
29 KB of fields in the same stretch as the map's own infrastructure would have made
both harder to review.

Suite **696/696** both memory modes.  Manifest unchanged at **270 of 501**: this
tranche maps no kernel, and saying otherwise would inflate the number that drives
the estimate.

## P8.66 — `dmaArticulationResidual`, and the address map the rest of the block needs

A small kernel chosen for what it forces rather than what it does.  One thread per
articulation reads the two residual accumulators out of the pool and writes them
into a pinned-host array the solver reads back on the CPU — but to do that it has to
address `PxgArticulationCoreDesc`, the 360-byte descriptor the **whole** articulation
solver block threads through every one of its kernels.  That address map now exists
and is pinned; every remaining articulation tranche reuses it.

**The three-hop index chain, transcribed and tested hop by hop:**

```
islandNodeIndices[tid + articulationOffset].index()  -> nodeIndex
bodySimPool[nodeIndex].articulationRemapId           -> poolIndex
articulations[poolIndex].{internal,contact}Residual  -> the payload
```

The dispatch index is a position in the island's articulation *range*, not an
articulation id.  The fixture uses offset 2, node indices 7/3/11 and pool slots
1/2/0, so no two hops agree and skipping any one of them is visible.

**The pinned-host argument.**  `errorPinnedHost` is
`PxPinnedArray<Dy::ErrorAccumulator>::begin()` cast straight to `CUdeviceptr`
(PxgTGSCudaSolverCore.cpp:1601) — legal under CUDA unified addressing, meaningless
here, so the launch-ABI row sets `hostPtrMask` bit 1 and the bridge substitutes the
device address.  The fixture allocates a **real** `memHostAlloc` buffer, passes the
CPU address, and reads the result back through the CPU mapping, so the mask bit is
exercised end to end rather than assumed.

**Sweep: 19 of 20 killed.**  The survivor is an invariance with a proof, not a gap:

* **the remap hop replaced by the dispatch index.**  Both the source record and the
  destination slot are indexed by the *same* `index`, and the engine's own assert
  (`index < nbArticulations`, forwardDynamic2.cu:3786) forces the remap to be a
  permutation of `[0, nbArticulations)`.  So over a full dispatch the correct code
  writes `{out[p] = pool[p] : p in [0,n)}` and so does the mutant — **byte-identical
  for every legal input**.  The remap carries no observable information *in this
  kernel*; the first kernel that can validate it is one whose output is indexed by
  the dispatch index while its input is indexed by the remap.  Recorded so the next
  tranche knows the hop is still unvalidated.

*A fixture bug worth keeping:* the first draft used pool index 4 with
`nbArticulations = 3`, which made one articulation's contact slot
(`index + nbArticulations`) collide with another's internal slot.  The test failed
and the **fixture** was wrong, not the shader — the engine invariant that keeps the
two output halves disjoint is exactly the assert above.  Reading the CUDA's asserts
is reading its preconditions.

**No scene reaches this kernel.**  Per-articulation residual reporting is off in the
harness, so `PXG_SCENE=articulation` never dispatches it (verified: zero occurrences
in the trace).  Its only validation is the fixture and the sweep, and that is stated
rather than papered over.

Suite **695/695** both memory modes; manifest **270 of 501**.

## P8.65 — MILESTONE: the articulation scene runs 240 steps

`newArticulationsLaunch` (updateBodiesAndShapes.cu:228) is mapped, and with it the
articulation scene stops crashing and starts *reporting*:

| | dispatches before it dies |
|---|---|
| P8.55 (articulation alone) | 52 — `initializeSolverVelocitiesTGS` |
| P8.60 (+ one companion rigid body) | 67 — `updateTransformCacheAndBoundArrayLaunch` |
| **P8.65 (pool filled)** | **16279 — it does not die** |

240 steps, exit 2 instead of SIGSEGV, and the engine names the six kernels it still
cannot launch: `artiContactConstraintPrepare`, `artiPropagateVelocity`,
`artiPropagateImpulses2`, `artiSolveInternalTendonAndMimicJointConstraints1T`,
`solveInternalConstraints`, `solveSelfConstraints`.  That is Phase 0's whole point:
the scene now names its own next kernel, which is how the softbody and
convex-trimesh chains were built.

The physics is of course wrong — the link snaps from y=3 to y=0 in one step and the
companion box hangs at y=3.73 instead of resting at 0.5 — because the six solver
kernels above are exactly the ones that would hold the joint together.  The scene
reports both as FAIL rather than pretending otherwise.

**What the kernel does.**  The host builds one 544-byte `PxgArticulation` record per
new articulation — device pointers into buffers it has just allocated
(PxgSimulationCore.cpp:884-933) — and stages them.  This kernel copies each record
into the persistent `mArticulationPool` at its own `data.index`, zeroes the
articulation's solver sleep data, and then fills every array those pointers name:
links, wake counters, link props, parents, children, both link transform arrays,
joint cores and joint data, spatial and fixed tendons (including the per-tendon
attachment / tendon-joint element copies reached *through* the tendon records),
mimic joint cores, the path-to-root table, the zeroed per-link state, and the five
dof arrays — each taken from `dofData` or zeroed depending on its dirty flag.

**Three transcription decisions, each load-bearing.**

*Every copy is a u32 copy.*  The CUDA walks each array with `warpCopy<T>` at a `T`
chosen for coalescing — uint4 for the aligned blocks, uint2 for spatial vectors,
uint for the 28-byte transforms.  `warpCopy` is an elementwise strided copy over a
destination that never overlaps its source, so the bytes that land in memory do not
depend on the element width used to walk them; every `totalSize` here is a multiple
of 4, and every value-fill is ZERO, which has the same bytes at every width.  The
width is a CUDA performance choice and nothing else.  This is bit-exact, not an
approximation — and it drops `warpCopy`'s 16-byte alignment assertions, which the
u32 form does not need.

*The shared-memory staging is gone, and it had to go.*  The CUDA stages the record
in shared memory, `__syncwarp()`s, copies shared -> pool, then reads the record's
pointer fields from the shared copy.  The source is never written, so reading those
fields straight from it is identical.  Keeping the staging would have been unsafe:
the workgroup is (32, 8, 1) and the whole body sits under
`globalWarpIndex < nbArticulations`, a guard that varies with `threadIdx.y` — so a
`GroupMemoryBarrierWithGroupSync()` inside it is a barrier in control flow that is
NOT uniform across the workgroup, and llvmpipe HANGS on those instead of erroring.
There is no cross-lane communication anywhere in this kernel, so no barrier is
needed at all.

*One engine quirk transcribed, not fixed.*  `numJointDofs` is a PxU16 that the CUDA
reads twice at different widths: `const PxU8 dofs` sizes the zero-fills, `const
PxU32 numDofs` sizes the copies from `dofData`.  Above 255 dofs (64 links x up to 6
is reachable) they disagree and the else-branches would zero only `dofs & 0xFF`
entries.  Kept exactly as written.

**Verification.**  `gpuslang.newArticulations` drives ten articulations across TWO
blocks of eight virtual warps.  The count is not decoration: with a single block the
`groupId.x * NA_WARPS_PER_BLOCK` term is always zero and a wrong warps-per-block
constant is invisible, so the two interesting articulations sit at warp indices 8
and 9.  They differ in every way that could hide a bug — pool slots 3 and 1 (neither
equals its warp index, and slots 0/2/12 must stay poisoned), source slices starting
at link 1 and link 5, one with a spatial tendon + a fixed tendon + two mimic joints
and one with none, and dirty flags chosen so that for each of the five dof arrays one
articulation copies while the other zero-fills.  Every destination is checked
byte-exactly against the source pool, and 64 poison bytes past each region catch
overruns.  Every layout constant is pinned to host-side `sizeof`/`offsetof` of the
real engine structs by `static_assert` — the P8.49a lesson: a fixture carrying its
own copy of a constant cannot detect that the constant is wrong.

**Sweep: 47 of 50 mutants killed.**  The three survivors, with proofs rather than
excuses:

* **the PxU8 dof truncation dropped.**  The intended control.  The fixture's dof
  counts are 5 and 2, so truncating to 8 bits changes nothing; separating the two
  reads needs an articulation with more than 255 dofs, i.e. 43+ links.  It survives
  by construction, and it is what shows the sweep is not simply rejecting every edit.
* **the copy stride lowered from 32 to 8.**  Lanes 0-7 still cover every index;
  lanes 8-31 rewrite the same values from the same source.  Coverage becomes a strict
  SUPERSET and the writes are idempotent, so the result is identical.  The constant
  matters in the *other* direction and that is tested: stride 64 loses elements and
  dies (both for the copy and, after the fixture grew A to six links so the largest
  zero-fill exceeds 32 words, for the fill).
* **warps-per-block lowered from 8 to 4.**  Same shape of invariance: block 1's base
  becomes 4 instead of 8, so articulations 4-7 are processed twice — idempotently —
  and 8 and 9 are still reached.  Raising it to 16 SKIPS articulations 8 and 9 and
  dies.

Two mutants survived the first sweep as genuine coverage gaps and the fixture was
changed to close them, not written off: the offset advance after the FORCES copy was
dead (nothing dirty followed it on either articulation) and the TARGET_POS/TARGET_VEL
bits were indistinguishable (one articulation had neither, the other had both).
Giving A `JOINT_TARGET_POS` but not `JOINT_TARGET_VEL` kills both.

Suite **694/694** in both memory modes (693 + `gpuslang.newArticulations`); manifest
**269 of 501**.  All twenty P8.54 scenes still green (`box stack plane sphere capsule
capsulebox convex joint meshfloor particle cloth clothhang clothdrape
clothselfcollide clothcloth softbody softbodyselfcollide softbodycloth softbodystack
softbodyattach`), so mapping this kernel changed nothing for the scenes that never
build an articulation.  `gpuslang.p4Wrangler` caught the audited mapped count still
reading 268 and was the reason it got bumped — that guard works.

### Correction to P8.64: `PXG_SYNC_EACH` already existed

P8.64 said it "ships the diagnostic".  It did not — `PXG_SYNC_EACH` has been in
`PxgCudaShim::launchKernel` since commit 5c44c1c, with a comment describing exactly
the attribution problem P8.61-P8.63 then spent three tranches failing to solve.
What P8.64 actually added was a second, redundant implementation of it one layer up
in `PxgVulkanCudaContext`; that duplicate is reverted here, and the `traceDispatch`
comment now points at the knob that exists.

The real lesson is worse than the one P8.64 drew, and worth keeping: the tool was
already there and documented. Nothing was missing but the decision to reach for it.

## P8.64 — the fault was never in dispatch 67; four tranches chased the wrong kernel

Two answers, and the second invalidates the question the first answered.

**Answer 1 — the two maps are the same object.**  `Sc::Scene::updateDynamicsPostPartitioning`
(ScPipeline.cpp:2119) passes `mAABBManager->getChangedAABBMgActorHandleMap()`, which
returns `Bp::AABBManagerBase::mChangedHandleMap` (BpAABBManagerBase.h:201), and
`PxgAABBManager : public Bp::AABBManagerBase`.  So the word count at
PxgSimulationCore.cpp:2177 and the buffer at `getChangedAABBMgrHandles()` are two
views of one bitmap, and **the P8.62 asymmetry does not exist**.  Measured live with
the articulation scene, to leave nothing to inference:

```
[dbg] preBpUpdate: alloc 4 bytes (words=1)
[dbg] wordCount(solve)=1 wordCount(bp)=1 updatedBytes=128 nbTotalShapes=4
```

One word, a 4-byte target buffer, a 128-byte source for `1 * 32` elements.  Every
index the shader forms is in bounds, exactly as P8.61 and P8.62 argued.

**Answer 2 — and that is because the faulting kernel is a different one.**
`launchKernel` only RECORDS a `vkCmdDispatch`, and `submit()` signals a timeline
semaphore instead of waiting on a fence, so a batch is still executing while the
trace prints its `OK` lines.  The trace's last line names the kernel recorded last,
**not** the kernel running when the process died.  Adding `PXG_SYNC_EACH=1` — drain
the stream after every launch — collapses that gap, and the trace ends:

```
[pxg-dispatch] sized updateTransformCacheAndBoundArrayLaunch ...     <- no OK line
```

The fault belongs to `updateTransformCacheAndBoundArrayLaunch`, dispatch **66**.
`updateChangedAABBMgrHandlesLaunch` was merely the next kernel recorded.

**The root cause, and it is not an engine bug this time.**  Dispatch 66 is the only
kernel in the trace with an articulation branch, and that branch chases a pointer
stored *inside* the articulation pool:

```slang
Ptr<PxgTransform28> linkBody2Worlds = Ptr<PxgTransform28>(
    artiLinkTransformArray(gArticulations, articulationId, PXG_ARTI_LINK_BODY2WORLDS_OFFSET));
```

`mArticulationPool` is filled by `newArticulationsLaunch`, which copies the staged
`PxgArticulation` records (PxgSimulationCore.cpp:935) into the pool.  That kernel is
**unmapped** — manifest row 27, `{ "newArticulationsLaunch", nullptr, nullptr }` — and
its launch result is `PX_UNUSED`'d (PxgSimulationCore.cpp:2391), so nothing is even
printed.  The pool therefore holds uninitialised memory, `linkBody2Worlds` is
whatever bytes were there, and the shader dereferences a wild address.  Consistent
with the fault: `rip` in an anonymous `r-xp` mapping (JIT'd shader code) on the
lavapipe queue thread, `si_addr` unreportable.

**So Phase 0 is not a workaround, it is the first port.**  The articulation scene
does not need an engine-side guard; it needs `newArticulationsLaunch` mapped.  That
is sub-chain 1, kernel 1 of the articulation plan — the scene named its own next
kernel, which is exactly how the softbody and convex-trimesh chains were built.

**What ships here** is the diagnostic that made this findable: `PXG_SYNC_EACH=1` in
`PxgVulkanCudaContext`, one submit-and-wait per launch, off by default and paired
with `PXG_LAUNCH_TRACE=1`.  The comment on `traceDispatch` that claimed a
pre-dispatch line "attributes a GPU fault to the exact launch" is corrected — it
cannot, on its own, and believing it did cost P8.61 through P8.63.

*Kept deliberately, because the shape of the error matters more than the error:*
those three tranches each reasoned carefully and each reached a sound conclusion
about the wrong kernel.  Nothing in the reasoning was going to catch that; only
changing the measurement was.  When a fault is attributed by "what printed last",
check what "last" means before trusting the name.

Suite **693/693 in both memory modes**, count unchanged — the new env knob is inert
when unset and there is no shader change to sweep.  One run of the `PXG_DEVICE_V2`
pass hit the **known intermittent `gpuslang.scan` segfault under `-j4`** (2 in 9 runs
as of P8.53, still unattributed, still open); it passed on immediate rerun.  Recorded
rather than hidden: it is the same open item, not a regression from this tranche.

## P8.63 — the asymmetry is one hop from resolved, and the hop is named

P8.62 flagged that PxgSimulationCore.cpp:2177 pairs the **AABB manager's** handle
buffer with the **simulation core's** word count.  Tracing both sides:

**The AABB manager's side is self-consistent** (PxgAABBManager.cpp:863-867):

```cpp
void PxgAABBManager::reallocateChangedAABBMgActorHandleMap(const PxU32 size)
{
    mChangedHandleMap.resizeAndClear(size);
    mChangedAABBMgrHandlesBuf.allocate(size * sizeof(PxU32), PX_FL);
}
```

One `size` drives both the map and the buffer, and `getChangedAABBMgrHandlesWordCount()`
just returns `mChangedHandleMap.getWordCount()`.  Buffer and count cannot diverge
from each other here.

**The simulation core's count is threaded down from above**, not read from the AABB
manager:

```
PxgContext::(…)              changedHandleMap  (a PxBitMapPinned&, parameter)
  -> PxgSimulationController::update(changedHandleMap)        :2793
    -> PxgSimulationCore::gpuMemDmaUp(…, changedHandleMap, …)  :2805
      -> bitMapWordCounts = changedHandleMap.getWordCount()    :2160
```

**So the whole question reduces to one hop:** is the `changedHandleMap` threaded
through `PxgContext` the *same object* as `PxgAABBManager::mChangedHandleMap`?

* If **yes** — there is no asymmetry at all, P8.62's concern dissolves, and the
  dispatch-67 fault is something else entirely.
* If **no** — two independently sized maps meet in one write, and an articulation
  making them diverge is the overrun.

That is a single grep up the call chain, at the caller of the `PxgContext` function
that owns `changedHandleMap` (PxgContext.cpp:611).  **Not run here**, and I would
rather name it exactly than guess at it: the two outcomes point at completely
different subsystems, so a coin-flip would send the next tranche the wrong way.

*Where the articulation frontier stands after five tranches on it:* the fatal
setup crash is understood and worked around (P8.60, an engine-side sizing bug that
CUDA survives only because a real GPU tolerates the OOB read); the scene now clears
the entire solver phase and names its unmapped kernels the normal way; and the one
remaining blocker is a single yes/no question with both answers mapped out.

## P8.62 — the READ is provably fine; the suspicion moves to the WRITE

P8.61 predicted a second instance of the P8.60 class — a buffer sized by a
rigid-only count and indexed by a total count.  **That prediction is wrong**, and
the allocation says so outright (PxgSimulationCore.cpp:2160-2163):

```cpp
const PxU32 bitMapWordCounts = changedHandleMap.getWordCount();
const PxU32 roundElement = bitMapWordCounts * 32;
mUpdatedBuffer.allocate(roundElement * sizeof(PxU32), PX_FL);
```

`mUpdated` is sized *exactly* `bitMapWordCounts * 32`, which is exactly the bound
the shader checks `i` against.  **The read is in bounds by construction.**  One
grep, hypothesis dead — which is the point of checking before digging.

**What the same passage exposes instead is an asymmetry in the WRITE.**  On the
GPU-broadphase path the descriptor is built as (:2177):

```cpp
constructDescriptor(..., aabbManager->getChangedAABBMgrHandles(), nbTotalShapes,
                    bitMapWordCounts);   // <- bitMapWordCounts is the SIM CORE's
```

so `mChangedAABBMgrHandles` is the **AABB manager's** buffer while
`mBitMapWordCounts` is the **simulation core's** word count
(`changedHandleMap.getWordCount()`).  The shader writes `changed[w]` for
`w < mBitMapWordCounts`.  Two different maps sizing the two halves of one write.

The CPU-broadphase branch immediately below does not have this shape — it allocates
`mChangedAABBMgrHandlesBuffer` at `bitMapWordCounts` itself, so buffer and bound
share a source.  And elsewhere the code is explicitly careful about exactly this
pairing (:3142-3143):

```cpp
mUpdatedActorDataDesc->mChangedAABBMgrHandles = aabbManager->getChangedAABBMgrHandles();
mUpdatedActorDataDesc->mBitMapWordCounts     = aabbManager->getChangedAABBMgrHandlesWordCount();
```

— the AABB manager's buffer paired with the AABB manager's count.  Line 2177 pairs
the AABB manager's buffer with the *other* map's count.

**Stated at the strength the evidence supports:** this is an asymmetry, not yet a
proven overrun.  It becomes the fault only if the simulation core's word count can
exceed the AABB manager's buffer, and an articulation adding link shapes is a
plausible way for the two maps to diverge — but that is unverified.  The check is
one host-side comparison of `changedHandleMap.getWordCount()` against
`aabbManager->getChangedAABBMgrHandlesWordCount()` with the articulation scene
loaded, and it is the next move.

*Record of a wrong turn, kept deliberately:* P8.61 reasoned by analogy from P8.60
and named the wrong buffer.  The analogy was reasonable and the conclusion was
false; what made that cheap was checking the allocation before writing any code.

## P8.61 — the dispatch-67 fault: the port is clean, so the data is not

Read the shader against the CUDA before digging further, and the result narrows the
search a lot: **`updateChangedAABBMgrHandlesLaunch` is correctly bounded against its
own descriptor.**

```slang
for (uint w = groupId.x; w < numWords; w += WorkgroupCount().x)   // w < numWords
    const uint i = w * 32u + lane;
    if (i < numElements && updated[i] != 0u)                      // i < numWords*32
    ...
    if (lane == 0) changed[w] = gAabbMgrWord;                     // w < numWords
```

Every index is derived from `mBitMapWordCounts` and checked against it.  The CUDA
(updateTransformAndBoundArray.cu:192) does the same arithmetic with a ballot
instead of a groupshared OR.  There is no unguarded index here of the kind P8.59
found in the solver, so **the fault is not in the translation — it is in the data
the descriptor hands it**: `mUpdated` or `mChangedAABBMgrHandles` null, or
`mBitMapWordCounts` garbage.

**Where to look, precisely.**  `PxgSimulationCore::constructDescriptor`
(PxgSimulationCore.cpp:612) fills both: `mUpdated = mUpdatedBuffer.getTypedPtr()`
and `mBitMapWordCounts = bitMapWordCounts`, the latter passed in by the caller.
The shader reads `bitMapWordCounts * 32` elements out of `mUpdatedBuffer`, so the
check is whether that buffer is sized for that many — **on the host, no device
readback needed**, exactly as P8.60's answer came from reading an `allocate()` call
rather than dumping memory.

**Why this is probably the same bug twice.**  P8.60's fault was a buffer sized by a
rigid-only count and indexed by a total count, exposed only because an articulation
inflates one and not the other.  An articulation adds link shapes, which get AABB
manager handles, which is exactly what `bitMapWordCounts` counts.  A second
instance of that class is the first thing to rule out — and if it is one, the fix is
again upstream in the engine, not in this port.

*Stopping the dig here deliberately:* the next step is a host-side size comparison,
which is a clean five-minute check with a clear yes/no, and it is better handed
over stated exactly than started and left half-answered.

## P8.60 — CONFIRMED: an articulation-only island empties `solverBodyReferences`

The read, not the write, was the fault — and the cause is an **array-sizing
mismatch that only an articulation-only island can produce**.

`mSolverBodyReferences` is allocated as `totalActiveBodyCount * numSlabs`
(PxgTGSCudaSolverCore.cpp:552) and memset to `INVALID`.  But
`initializeSolverVelocitiesTGS` indexes it as
`numSlabs * (bodyId - bodyOffset) + b` with `bodyId < numSolverBodies`.  Those are
two different counts.  My P8.55 scene had **no plain dynamic rigid body at all** —
just the articulation — so `totalActiveBodyCount` was zero, the allocation was
empty, and the shader read straight off the end of it.

**The experiment that proves it:** add one ordinary dynamic box beside the
articulation, change nothing else.

| | dispatches before the fault |
|---|---|
| articulation alone | **52** — dies in `initializeSolverVelocitiesTGS` |
| articulation + one rigid body | **67** — sails past it |

Fifteen dispatches further, through the entire solver phase, and the run now emits
ordinary `fail to launch kernel` messages for the unmapped `arti*` kernels
(`solveInternalConstraints`, `solveSelfConstraints`,
`artiSolveInternalTendonAndMimicJointConstraints1T`) — the graceful degradation
every other subsystem has, and exactly what the one-kernel-at-a-time route needs.

**The crash has MOVED, not gone.**  The process still dies, now at dispatch 67 in
`updateChangedAABBMgrHandlesLaunch`.  That is a different problem in a different
subsystem and is not claimed to be understood.

*(A caveat on my own earlier reading: one intermediate run of this experiment
appeared to exit cleanly.  It did not — `... | grep | head` had closed the pipe and
`$?` reported head's status, not the program's.  The scene has never yet completed.)*

**What this unblocks.**  P8.55 concluded that articulations "cannot be driven
forward the usual way until this fault is understood".  It is understood, and the
route is open: with a companion rigid body the articulation scene reaches the
articulation kernels and names them one at a time, which is how the softbody and
convex-trimesh chains were built.

**And a bug worth reporting upstream, separately from this port:** the sizing
mismatch is in the ENGINE, not in the Slang translation.  The CUDA reads the same
empty array with the same indices; it survives only because a real GPU tolerates
the out-of-bounds read where llvmpipe faults.  An articulation-only island on CUDA
is reading uninitialised memory too — it just does not crash.

## P8.59 — the pool bound lands; it does NOT fix the articulation crash

P8.58 deferred bounding `remappedBodyIndex` because the descriptor "carries no
`solverBodyVelPool` size field".  That was true and beside the point: the size is
**derivable exactly** from two values the shader already holds.
PxgTGSCudaSolverCore.cpp:412-423 allocates

```
totalBodiesSize = mirrorBodies + accumulationBodies + numOutputBodies
mSolverBodyOutputVelocityOffset = mirrorBodies + accumulationBodies
numOutputBodies = roundUp32(numSolverBodies) * 3
```

and PxgSolverCore.cpp:469 assigns that offset to `accumulatedBodyDeltaVOffset`.
So the pool's float4 count is exactly `accumulatedBodyDeltaVOffset +
roundUp32(numSolverBodies) * 3` — no descriptor change, no inferred layout:

```slang
const uint velPoolSize = outputOffset + (((allBodiesCount + 31u) & ~31u) * 3u);
...
if (remappedBodyIndex != INVALID && (remappedBodyIndex + 64u) < velPoolSize)
```

**A deliberate divergence, and the only one in this shader.**  Rejecting an
out-of-pool index cannot change a correct execution — a legitimate index is in the
pool by construction — so it only suppresses a write that is already undefined
behaviour, and turns a process kill into a survivable frame.

**It does not fix the articulation scene, and saying otherwise would be wrong.**
With the bound in place the scene still SIGSEGVs, and the launch trace still stops
at the same place:

```
[pxg-dispatch] sized initializeSolverVelocitiesTGS ...     <-- dispatch 52, still
```

So the faulting access is NOT the guarded write.  What is left unguarded in that
kernel is the READ that feeds it — `solverReferences[bodyIndex]`, with
`bodyIndex = numSlabs * (bodyId - bodyOffset) + b` — plus the two descriptor loads
and `velocities[bodyId]`.  The evidence now points at `desc.solverBodyReferences`
or `desc.outSolverVelocity` being **null or short** for an articulation-bearing
scene, which a guard on the value read *out* of that array can do nothing about.

The bound is kept regardless: it removes a real wild-write hazard that the CUDA
only survives because its arrays are always initialised, and it is exactly derived
rather than guessed.  Regression: suite **693/693** under `-j4`, and `box`,
`stack`, `joint`, `softbodystack`, `meshfloor` all still green.

Next: print the descriptor pointers at that dispatch and compare the articulation
scene against `joint`.  A null `solverBodyReferences` would explain everything and
is one memcpy to confirm.

## P8.58 — the one-line fix for the thing that actually cost three tranches

The articulation hunt burned P8.55, P8.56 and half of P8.57 on a wrong premise —
that the crash was in scene construction — and the cause was not subtle: the
harness's stdout is block-buffered when redirected, so every progress line it had
printed but not flushed died with the process.  PhysX's own error callback is
unbuffered, so what survived was all engine noise and no harness context, and the
crash appeared to happen three phases earlier than it did.

`main()` now line-buffers stdout:

```cpp
std::setvbuf(stdout, nullptr, _IOLBF, 0);
```

The articulation scene now locates its own crash with no external tooling:

```
creating GPU scene (eENABLE_GPU_DYNAMICS + eGPU broadphase, TGS)...
[  OK  ] GPU scene constructed: the Pxg* pipeline is live over Vulkan
[  OK  ] scene 'articulation' built: 1 dynamic body(ies) + ground on the GPU
stepping the GPU scene 'articulation' (240 x 0.0167s)...          <-- then SIGSEGV
```

`meshfloor`, `box` and `softbody` re-checked green.  This harness exists to drive
scenes that crash; it should never again be the reason a crash is mislocated.

**The shader bound is DEFERRED, and deliberately so.**  P8.57 proposed bounding
`remappedBodyIndex` in `initializeSolverVelocitiesTGS` so a missing upstream kernel
yields a diagnosable frame instead of a wild write.  That is still the right idea
and it is not done, for a concrete reason: the solver descriptor carries **no
`solverBodyVelPool` size field**, so an exact bound is not derivable from what the
shader can see.  A bound would have to be inferred from the pool layout —
`accumulatedBodyDeltaVOffset` plus the averaging region — and inferring a layout
inside a shader that twenty green scenes depend on, to guard a case that only
occurs when something upstream is already missing, is a bad trade made in a hurry.
Doing it properly means either adding the size to the descriptor mirror or
compiling a debug shader variant selected at load time.  Left open rather than
half-done.

## P8.57 — the articulation crash, fully diagnosed (and two more of my claims corrected)

**Correction 1: it is not scene construction.**  P8.55 and P8.56 both placed the
crash "during scene construction", because the harness printed `creating GPU
scene` and nothing after it.  Re-running under `stdbuf -oL` shows what was
actually happening:

```
  creating GPU scene (eENABLE_GPU_DYNAMICS + eGPU broadphase, TGS)...
  [  OK  ] GPU scene constructed: the Pxg* pipeline is live over Vulkan
  [  OK  ] scene 'articulation' built: 1 dynamic body(ies) + ground on the GPU
  stepping the GPU scene 'articulation' (240 x 0.0167s)...        <-- then SIGSEGV
```

The scene builds fine and the articulation is created fine.  The fault is in the
**first `simulate()`**.  Those three lines were in an unflushed stdout buffer and
died with the process; only PhysX's own error callback, which is unbuffered, got
through — which is exactly why the earlier logs looked like a setup crash.

**Correction 2: the "unmapped kernel upstream" story was right after all.**  P8.56
dismissed it because the trace showed no refused launch before the fault.  That
was bad reasoning: an unmapped kernel never reaches the dispatch path at all — the
wrangler hands back a null function at startup (those `Could not find registered
CUDA function` lines) and the engine skips it silently.  **Absence of a refusal in
the trace is not evidence that everything upstream ran.**

**The mechanism.**  `initializeSolverVelocitiesTGS` ends in an unchecked indirect
write (solverInitVelocitiesTGS.slang, mirroring solverMultiBlockTGS.cu:729):

```slang
const uint remappedBodyIndex = solverReferences[bodyIndex];
if (remappedBodyIndex != INVALID)
{
    bodyVelocities[remappedBodyIndex]        = vel0;   // <-- index straight out of memory
    bodyVelocities[remappedBodyIndex + 32u]  = vel1;
    bodyVelocities[remappedBodyIndex + 64u]  = vel2;
}
```

`solverReferences` is `PxgSolverCoreDesc::solverBodyReferences`, and the only thing
that ever writes it is the constraint-prep path (constraintPrepShared.cuh:145).
With an articulation in the scene that path runs through the **`arti*` constraint-prep
kernels, which are unmapped** — so for the articulation's bodies the array is never
written and holds allocator garbage.  Garbage is overwhelmingly unlikely to equal
`INVALID` (0xFFFFFFFF), so the guard passes and the shader performs a **wild write**
at an arbitrary 32-bit index.  That is the JIT-code segfault P8.56 located.

**Two separate things follow, and they are not the same size.**

1. The articulation frontier needs its constraint-prep kernels mapped.  That is the
   ~85-kernel job P8.54 already scoped, and this is simply the first place the
   absence bites.
2. **Independently: an unchecked index out of device memory turns a missing kernel
   into a process kill.**  Every other subsystem degrades to a wrong-but-alive
   frame; this one takes the process down, which is why articulations could not be
   driven forward the way softbody and convex-trimesh were.  The CUDA has the same
   unchecked write and gets away with it because the array is always initialised
   there.  Worth considering a debug-mode bound on `remappedBodyIndex` so a missing
   upstream kernel produces a diagnosable frame instead of a JIT segfault — a
   deliberate divergence, not a silent one, and it would have turned this
   three-tranche hunt into one line of output.

## P8.56 — the articulation fault is a MAPPED kernel, and P8.55 read it backwards

P8.55 called the articulation crash "a call through an unresolved function pointer,
not a bad dereference inside known code".  That was wrong, and the evidence that
settles it is one line:

```
FAULT PC 0x7fffecc14617 IS IN: 0x7fffecc10000  0x7fffecc1e000  0xe000  0x0  r-xp
```

The faulting address sits in an **anonymous executable mapping** — no file backing,
`r-xp` — which is llvmpipe's **JIT-compiled shader code**.  The absent caller frame
that P8.55 leaned on proves nothing: JIT frames carry no unwind information, so gdb
cannot walk out of them.  Two llvmpipe worker threads fault at the *identical* PC
while the main thread sits inside `libvulkan_lvp.so`, which is what two workers
running one shader looks like.

So this is a P8.40-class fault — a shader taking down the process on a bad
buffer-device-address access — not a shim gap.

**`PXG_LAUNCH_TRACE=1` names it exactly.**  The scene reaches dispatch **52**, and
the trace stops mid-dispatch:

```
[pxg-dispatch] sized ZeroBodiesTGS ...
[pxg-launch]   sized ZeroBodiesTGS OK
[pxg-dispatch] sized initializeSolverVelocitiesTGS ...      <-- never returns
```

`initializeSolverVelocitiesTGS` (solverMultiBlockTGS.cu:673) is **already mapped**
and runs green in all twenty other scenes.  All 51 dispatches before it returned OK
and **not one launch was refused** — so the "an unmapped articulation kernel
refused upstream and left a null behind" story is not supported by the trace
either.  Something about an articulation-bearing scene makes this specific solver
kernel read out of bounds.

**What is established:**

* the fault is inside JIT-compiled shader code, not PhysX host code;
* the shader is `initializeSolverVelocitiesTGS`, dispatch 52, during scene
  construction;
* no launch is refused before it;
* it is deterministic across five runs now (plain, redirected, gdb ×2, traced).

**What is not:** why that kernel goes out of bounds only when an articulation is
present.  It reads `outSolverVelocity[bodyId]` and `[bodyId + allBodiesCount]`
under a `bodyId < allBodiesCount` guard, so the suspects are the descriptor values
themselves — `numSolverBodies`, `numSlabs`, `accumulatedBodyDeltaVOffset`, or the
`outSolverVelocity` / `islandContextPool` pointers — one of which is presumably
null or absurd because the articulation path never populated it.

Next step: have the shim dump that dispatch's descriptor (it already has the push
blob) and compare the same fields against the `joint` scene, which is the closest
green neighbour.

## P8.55 — the articulation scene, and the wall it hits BEFORE any kernel runs

`PXG_SCENE=articulation` added to `gpu-sdk-step`: the smallest articulation that
exercises the feature — a reduced-coordinate articulation with a FIXED BASE and one
revolute link, spawned horizontal so gravity has maximum leverage, swinging down to
hang.  One inbound joint, one degree of freedom, no contacts, no self-collision, so
anything it refuses is attributable to the articulation pipeline itself.

**It takes a SIGSEGV on a worker thread**, deterministic across three runs (plain,
redirected, under gdb).  *(P8.55 and P8.56 both said "during scene construction".
That was wrong — see P8.57: the harness's own progress prints were sitting in an
unflushed stdout buffer, and the crash is actually in the FIRST `simulate()`.)*  The stack is a jump through a
bad pointer with no caller frame:

```
Thread 7 "gpu-sdk-step" received signal SIGSEGV
#0  0x00007fffecc14617 in ?? ()
#1  0x0000000000000000 in ?? ()
```

*(P8.56 shows this reading was WRONG — see below.  The address is inside JIT
code, and the missing caller frame is an artefact of JIT frames carrying no
unwind information, not evidence of an unresolved pointer.)*

**This matters for the roadmap P8.54 just wrote.**  The articulation frontier is
NOT simply "map ~85 kernels": something in the articulation setup path calls
through a pointer the Vulkan shim never populated, and it does so at
`addArticulation` time rather than at launch time.  Every other subsystem in this
port degrades gracefully when a kernel is missing — the wrangler reports
`fail to launch kernel` and the frame continues, which is exactly how the softbody
and convex-trimesh chains were driven forward one kernel at a time.  Articulations
do not get that far, so the usual method (run the scene, read the refused
launches, map the next kernel) **cannot start until this fault is understood**.

The scene is committed anyway, and deliberately: it reproduces the crash on demand
and it is the instrument the next tranche needs.  It is not wired into any green
list and claims nothing.

Next diagnostic step: build with symbols/`-g` for the GPU module and re-run under
gdb to name the call site, or bisect by disabling `eENABLE_GPU_DYNAMICS` for the
articulation scene to see whether the fault is in the GPU articulation module or in
the shared setup path.

## P8.54 — MILESTONE: every scene in the harness passes, and what is actually left

```
OK box   OK stack  OK plane            OK sphere      OK capsule
OK capsulebox      OK convex           OK joint       OK meshfloor
OK particle        OK cloth            OK clothhang   OK clothdrape
OK clothselfcollide                    OK clothcloth
OK softbody        OK softbodyselfcollide             OK softbodycloth
OK softbodystack   OK softbodyattach
```

**All twenty `gpu-sdk-step` scenes step on the Vulkan/llvmpipe pipeline** — every
rigid scene, the cooked triangle mesh, particles, all five cloth scenes and all
five softbody scenes.  There is no red scene left in the harness.

That is a good moment to stop and count what is NOT done, because "all scenes
pass" is a much weaker statement than it sounds.  The manifest holds **501 kernel
rows, 268 mapped and 233 unmapped** — the scenes pass because the unmapped
kernels are ones no scene reaches, not because the port is finished.

**The 233, by subsystem:**

| Count | Block | Note |
|---|---|---|
| ~85 | **articulations** | 69 `arti*` plus `computeUnconstrained*`, `updateBodies*`, `setupInternalConstraint*`, `accumulateDeltaVRigid*`, `getD6Joint*` — **both numbers are WRONG, see the correction below** |
| ~45 | isosurface + sparse grid (`iso_*`, `sg_*`) | fluid surface extraction, not simulation |
| 31 | particle variants (`ps_*`) | paths the one particle scene does not take |
| 22 | SDF (13) + BVH build (9) | SDF collision and its tree build |
| 17 | cloth variants | |
| 11 | heightfield | a whole collision shape type |
| 5 | convex-core | the newer contact-gen path |
| 7 | broadphase merge / frozen-unfrozen | |
| 2 | trimesh-vs-trimesh | |

*Correction (P8.65).* The "~85" and the "69 `arti*`" in that table are both
inflated by a substring match: `arti` also matches every **p·arti·cle** kernel, so
the particle rows were counted twice — once as particles and once as
articulations.  Counted properly the articulation block is **65 kernels** (49 core
plus 16 solver-side), and the particle block is 45, not 31.  The conclusion the
table draws is unaffected — articulations are still the largest coherent block and
still the next frontier — but the figure carried into planning should be 65.

**Articulations are the next frontier, and it is not close.**  They are the single
biggest coherent block, they are a headline PhysX feature — ragdolls, robots,
vehicles — and they are *completely* untested here: the harness has no
articulation scene at all, so not one of those ~85 kernels has ever been asked to
run.  Every other block is either a variant of something already working or a
feature no scene exercises.

The route is the one the softbody and convex-trimesh chains took: add the scene
first, run it, and let the refused launches name the work.

## P8.53 — a correction: the `gpuslang.scan` segfault is NOT reproducible

P8.52 recorded a `gpuslang.scan` SEGFAULT under `-j4` in default memory mode and
called it **reproducible**, on the evidence of two failing full runs.  That word
was wrong, and this entry exists to say so rather than quietly drop it.

Nine full `-j4` default-mode runs are now on record: the **first two failed, the
next seven all passed**, across every variation worth trying —

* a plain `ctest -j4`;
* `ctest -j4 --output-on-failure` (to capture the crash, which never came);
* four back-to-back `-j4` runs measuring a rate;
* two forced-fresh runs (`rm *.spv`, full shader rebuild, then `-j4`), because both
  original failures sat near a forced-fresh rebuild and that looked like a lead.

It is not the rebuild, and it is not something that reproduces on demand.  Two
mechanisms were checked and ruled out on the way: the box has **16 GB with 13.5 GB
free and no cgroup limit**, so it is not a global OOM; and the Vulkan pipeline
cache is only ever file-backed by `testP1PipelineCache`, which uses its own path —
the main test device leaves `pipelineCachePath` empty and touches no shared file,
so concurrent cache corruption is out.

**What is actually known**, and it is less than P8.52 implied: two crashes, both in
`gpuslang.scan`, both in default mode, both adjacent in time, never since.
`gpuslang.scan` is test #1 and unrelated to anything the convex-trimesh chain
touches.  Correctness is not in question — the suite passes serially, passes under
`-j4` seven times running, and passes in both memory modes.

The honest label is an **intermittent host-side flake of unknown cause, observed
twice in nine runs and not reproducible**.  It stays on the board rather than being
closed as fixed, because nothing was fixed.

## P8.52 — the engine-ABI fixture P8.51 shipped without, and a 46/47 sweep

P8.51 landed `cvxTriFinishBda` on the strength of the Phase-1 verbatim port plus a
live scene, and said so plainly: no engine-ABI fixture of its own, no mutation
sweep.  This closes that.

New suite `gpuslang.cvxTriFinish`, in the same file as the Phase-1 test and driven
against the SAME `refTrimeshOut` oracle — so a divergence is attributable to the
BDA plumbing and the warp emulation rather than to the algorithm.  220 cases with
real atomics on a real `PxgPatchAndContactCounters` and real writes into all three
streams.

**ONE PAIR PER DISPATCH, AND THAT IS NOT A SHORTCUT.**  The reservations are
`InterlockedAdd`s on a shared cursor, so with several pairs in flight the byte
offsets each pair receives depend on the order the atomics land — exactly what the
oracle cannot predict.  Every case therefore gets its own counters, streams and
output record as a per-case slice addressed through the push blob, and the
dispatches are queued and flushed together because they touch disjoint memory.

**Sweep 46/47**, after a first pass of 39/44 whose five survivors were worked
through one at a time rather than waved off:

| Was surviving | Closed by |
|---|---|
| the manifold stride 1200 vs 1216 | a SECOND pair per dispatch — a decoy with no manifolds, and a live-looking count planted where a 1200-stride reader would find its header.  The decoy's reservation then moves the counters, which the existing check already sees.  It also stops the constant-trip-2 warp loop from being assumed. |
| `setOverflowError` as a store instead of an atomic OR | seeding the error word with an unrelated bit.  With one pair per dispatch and a zero start, a single overflow sets one bit either way; the seed makes OR and store differ. |
| the patch-change flag reading the pre-rollback manifold count | a case where `prevPatches == nbManifolds` AND the patch reservation is forced to overflow.  The flag is then `prevPatches != 0` (true) with the rolled-back count and `prevPatches != nbManifolds` (false) without — the only input that separates them. |
| the cu:257 gate ignoring the overflow ballot | asserting an overflowing pair writes NOTHING rather than skipping its streams.  On a patch overflow the CONTACT and FORCE offsets keep their reserved values, so a kernel that ignored the ballot would fill those two streams — and the old comparison never looked. |
| the bounds check on the loop's second trip | dispatching every fifth case with an ODD pair count, so the second trip is genuinely out of range, and requiring the decoy's record to come back exactly as uploaded. |

Two more dispatch mutants were added while closing those and both die: covering one
warp per block instead of two, and dropping the sub-warp from the pair index.

**The single survivor is unreachable, not untested.**  Dropping the
`nbManifolds == 0` term from the gate cannot be observed: with no manifolds every
lane's `numContacts` is zero, so `hasContacts` is false across the warp and every
write sits behind it.  The body runs, reads valid memory and produces nothing.

Coverage is asserted rather than hoped for: all three overflow kinds, the
zero-manifold path, the no-reservation path that stores a wrapped
`base + 0xFFFFFFFF`, `insertAveragePoint`, `flip`, the sphere and capsule world-point
paths, exact-fit reservations, and the rollback case above.  The documented
intra-warp face-index write conflict is masked from comparison, as it is in the
Phase-1 test — it is a real unordered write in the CUDA, not something a port can
fix.

**Verification.**  `PXG_DEVICE_V2=1`: **693/693** under `-j4`, forced-fresh.
Default mode: **693/693** serially, and **693/693** under `-j4` — see the P8.53
correction below for why that sentence needed rewriting once.

## P8.51 — MILESTONE: a box falls and RESTS on a cooked triangle mesh, on Vulkan

```
[boxOnMesh] start y=5.000, y after 1 step=4.9973, final y=0.5000 (rest 0.500), |v|=0.0000
[  OK  ] gpu-sdk-step: scene 'meshfloor' stepped on the Vulkan GPU pipeline (all 1 body(ies) fell + rested)
```

`PXG_SCENE=meshfloor` under `PXG_GPU_STEP=1`: a dynamic box dropped from y=5 onto a
COOKED TRIANGLE MESH comes to rest at exactly its rest height with zero residual
velocity, with **zero refused and zero failed launches**.  This scene has been red
since P8.21.

**What it took to close it was one kernel.**  Running the scene after P8.50b showed
the whole chain launching except for a single repeated error —
`GPU convexTrimesh finishContacts fail to launch kernel!!` — so the remaining work
was exactly one map, not a hunt.

`convexTrimeshFinishContacts` (convexMeshOutput.cu:116-346) is now
`cvxTriFinishBda.slang`, manifest row 268, ABI row of twenty-one arguments
(`{8,8,8,8,8,8,4,8,8,8,8,1,8,8,8,8,8,8,4,4,4}`, 160 bytes — the same shape as the
convex-convex `finishContactsKernel`, one 1-byte bool included).  It turns a
multi-manifold into the contact patches, points and face indices the solver reads:
the reservations become real `InterlockedAdd`s on the shared
`PxgPatchAndContactCounters`, and the three streams become real writes.

Lifted from `npCvxTrimeshOut.slang`, the Phase-1 verbatim port of the same kernel,
with every quirk intact:

* the reservation is **lane 0's alone**, and `insertAveragePoint` inflates
  `totalNumContacts` on lane 0 ONLY — so the face-index region base is PER-LANE
  DIVERGENT and lane 0's slot can collide with another lane's.  That is a genuine
  unordered intra-warp write conflict in the CUDA; it is reproduced, not repaired.
  The average contacts themselves are never written — the slots are reserved and
  left stale, because `getAvgContactValue` is dead code.
* the overflow tests are a **chained else-if** over the full reservation sizes, so
  at most one error bit is ever set even when two limits are blown.
* a pair with manifolds but no contacts skips the reservation entirely, leaving
  the offsets at `0xFFFFFFFF`, yet still passes the gate and stores
  `base + 0xFFFFFFFF` as its output pointers — wrapped.  Nothing dereferences
  them because every stream write is behind `hasContacts`.
* `getWorldNormal` does NOT average despite its comment: it takes the manifold's
  FIRST contact normal, rotates it by the trimesh transform and normalizes.

One guard is NOT verbatim and is flagged in the shader: the patch write carries an
extra `patchByteOffset != 0xFFFFFFFF` test the CUDA does not have.  It is provably
unreachable — the reservation runs whenever `totalNumContacts != 0`, and reaching
that line needs `popcContacts > 0` — but it is kept because the alternative
failure mode is a WILD WRITE at `patchStream + 0xFFFFFFFF` rather than a wrong
value.

**What stands behind this kernel today, stated plainly:** the Phase-1 verbatim port
with its own fixture and sweep (`npCvxTrimeshOut`), plus the live scene above.  It
does NOT yet have an engine-ABI fixture of its own with a mutation sweep, the way
P8.48-P8.50b do.  That is the next tranche, not a claim being made here.

Suite **692/692 in both memory modes**, forced-fresh.  No regression: `meshfloor`,
`box`, `stack`, `plane`, `convex`, `joint`, `capsulebox` and `sphere` all fall and
rest on the Vulkan pipeline.

**The convex-vs-trimesh chain is complete**: setup prologue, midphase, SAT and
clipping, contact reduction, emit tail, the composed core kernel, the triangle-index
sort, the silhouette cull, the patch builder, and now the output stage.

## P8.50b — `convexTrimeshCorrelate`: the convex-trimesh chain reaches the solver

`convexTrimeshCorrelate` (convexMeshCorrelate.cu:348, core at :95) mapped as an
engine kernel — `cvxTriCorrelateBda.slang`, manifest row 267, ABI row of eleven
arguments (`{8,8,8,8,8,8,8,4,4,8,8}`, an 80-byte blob; the two scalars sit at @56
and @60 and the pointer after them realigns to @64, confirmed against `spirv-dis`).

This is the patch builder, and the last stage before contacts reach the solver.
Contact generation has by now produced, per contact manager, a run of triangles
each carrying a normal, a contact count and a slice of contact points.  This
kernel turns that into at most four PATCHES of at most six contacts: repeatedly
pick the deepest triangle still holding contacts, accept every triangle whose
normal agrees with it to within `PATCH_ACCEPTANCE_EPS`, pull all their contacts
into the warp, and reduce to six with the clustering `contactReduce` landed in
P8.50a.

**Coverage is exact.**  The host launches `ceil(numTests / 2)` blocks of two warps
and passes numTests as `numPairs` — no grid-stride loop, so one workgroup per CUDA
block with a constant-trip-2 loop covers every pair exactly once.  Nothing is
skipped and there is no ceiling to close later.

`[numthreads(32,1,1)]` is mandatory rather than preferred: `crContactReduce6Cluster`
carries barriers in warp-uniform control flow, and llvmpipe HANGS on a divergent
barrier.  The two warps of a CUDA block are serialised through one 32-thread group.

**Two places needed device-memory barriers, not groupshared ones.**
`purgeEmptyTriangles` compacts the depth and nic runs IN PLACE — on CUDA a warp's
loads all retire before its stores, but a 32-thread workgroup on llvmpipe gives no
such guarantee, so the record is read into registers, an `AllMemoryBarrier`
is taken, and only then written.  The same applies to the per-patch `tempCounts`
clear, which the next patch's deepest scan reads back.

**One deliberate deviation, stated rather than hidden.**  `purgeEmptyTriangles`
guards its depth load with `tI < nbToLoad` but its nic load with `nbToLoad` — the
block size, not the lane test (cu:70-71) — so lanes past the end read one record
beyond the run.  The value is PROVABLY discarded (their depth is FLT_MAX, so
`valid` is false), so this port clamps the index: the result is identical, and an
unclamped read would be an out-of-bounds BDA load that can take llvmpipe down
rather than return garbage.

Every documented quirk is ported verbatim: the FOUR-bit contact count purge stores
against the THREE-bit count the patch loop reads back (at exactly 8 a triangle
contributes zero contacts yet keeps a nonzero `tempCounts` forever, so
`remainingTriangles` never reaches zero and the loop runs out its four
iterations); the zero-triangle early-out writing `mNbManifolds` of the **pairIndex**
manifold while everything else writes the **cmIndex** one; `mNbContacts[patch]`
written even for empty patches.

New suite `gpuslang.cvxTriCorrelate`, in the same file as the Phase-1 test and
driven against the SAME `refCorrelate` oracle — the scalar shader proves the
algorithm, so a divergence here is attributable to the BDA plumbing and the warp
emulation and nothing else.  240 cases; the fixture rebases every per-case index
onto shared engine buffers (pair *i* owns depth/nic/tempCounts slots
`[i*64, i*64+64)` and contact-pool entries `[i*128, i*128+128)`, and the cbuf
VALUES are rebased too since they index the pool).  The two output slots are
modelled honestly: the engine cmIndex is set to the SAME slot as pairIndex when
the oracle's cmIndex is 0 — the aliasing the engine really hits — and to a
distinct slot otherwise, with every manifold word starting at `0xFFFFFFFF` so
"not written" is observable.

Coverage asserted, not hoped for: cached pairs, the zero-triangle early-out, the
cm1 split, a reduce actually running, multi-patch results, the 4-bit/3-bit quirk,
survivor runs spanning two 32-lane blocks, and BOTH depth ties — across lanes
(lowest lane wins the exact min) and within one lane (the first strided index wins
the strict `<`).  Those two are precisely what a scalar harness cannot reach.

**Sweep 35/37.**  Both survivors are unreachable rather than untested: firing the
reduce at `>= 6` instead of `> 6` calls a function that opens with
`countbits(allMask) <= 6 -> return allMask`, so the extra call is a no-op; and
writing `mRelativeTransform` unconditionally is indistinguishable because the
zero-triangle path has already returned, `remainingTriangles` is at least one, and
the patch loop therefore always runs at least once — `nbPatches == 0` never
happens.

A third mutant is worth recording for HOW it died: removing the reduce entirely
does not produce wrong contacts, it **hangs**.  With more than six contacts held
and nothing to cull them, `readMask = ~currentContactMask` empties out and
`c += countbits(readMask)` stops advancing.  The reduction is load-bearing for
TERMINATION, not only for manifold quality.

**With this the whole convex-vs-trimesh chain is mapped** — setup prologue,
midphase, SAT and clipping, contact reduction, emit tail, the composed core
kernel, the triangle-index sort, the silhouette cull, and now the patch builder.

Suite **692/692 in both memory modes**, forced-fresh.

## P8.50a — the SECOND contactReduce instantiation: clustering, as a real warp

`contactReduce<true, false, 6, false>` (convexMeshCorrelate.cu:277) added to
`pxgSlangContactReduce.slang` as `crContactReduce6Cluster`.  This is what
`convexTrimeshCorrelate` reaches, and it is the prerequisite for that kernel —
P8.44 covered only `<false, true, 5, true>`, the emit tail's instantiation.

Three differences from the one already there, and the third is a whole branch the
module did not have:

* **no dupe test** — coincident contacts survive into the anchor search;
* **no PCM-deepest step** — with six slots the fill stage reaches those points
  anyway;
* **clustering is ON** — instead of filling the remaining slots with the deepest
  unselected contacts, every contact is LABELLED with the anchor nearest to it in
  the normal plane, one representative is taken per non-empty cluster, and only
  then are the leftover slots filled.  The anchors are biased deeper by
  `clusterBias` so a cluster prefers the anchor that defined it, which is what
  stops the manifold flickering between near-equal contacts frame to frame.  The
  bias is a live host value here (PxgNarrowphaseCore.cpp:2718), not the zero the
  emit tail passes.

Driven against `refReduce` — the SAME C++ oracle the scalar port uses — so a
divergence is attributable to the clustering branch and the warp emulation, not
to the algorithm.  New suite entry `contactReduceWarpCluster6` inside
`gpuslang.contactReduce`: 12 constructed cases plus 512 random ones.

**Two structural facts were derived from the CUDA rather than assumed**, and both
shaped the fixture:

1. `popc(mask)` after the four anchors is ALWAYS 4 at this instantiation — each
   anchor is drawn from `allMask & ~mask`, so they land on four distinct lanes,
   and the candidate set cannot be exhausted because `popc(allMask) > 6`.  So
   `nbClusters` is always 4 and the fill loop always runs exactly twice.
2. a cluster is normally non-empty, because an anchor is zero distance from
   itself and the label test is a strict `<`.  The ONE way to empty one is **two
   anchors at the same position**: the later is zero away from the earlier, the
   earlier wins the strict test, and nothing labels itself into the later cluster.
   That cluster then costs a slot the fill loop — which counts from `nbClusters`,
   not from the points actually selected — never gives back, and the patch comes
   out with FIVE contacts instead of six.

Constructing (2) took real geometry work: `p1` is the furthest point from `p0` and
`p2`/`p3` are the extremes of the signed distance to the `p0-p1` line, so a
coincident pair — at distance zero from `p0`, sitting on the line — is normally
neither.  The construction that works puts every other point strictly on ONE side
of that line, which forces the argmin onto `p0`'s twin.  (First attempt failed:
the angular sweep ran past π at lane 31, putting that lane on the wrong side and
letting it win the argmin instead.)

**Sweep 28/32**, after a first pass of 15/28 whose gaps were all closed rather
than excused:

| Was surviving | Closed by |
|---|---|
| a five-point early-out bound; a strict `<=` | a six-lane case with coincident anchors — the early-out returns all six, the full path returns five |
| the label tie going to the LATER anchor | a lane equidistant from two anchors whose loser cannot be recovered by the fill, so the two answers stay apart |
| the empty-cluster guard dropped | holding lane 31 out of the set, so the `crLowestSetIndex(0) == -1` shift lands on a provably-wrong bit |
| `p1` allowed to collide with `p0` | all 32 contacts at one point — the only input where the distance-from-`p0` argmax has nothing to prefer |

Six more mutants had been SKIPPED because their anchor text is shared with
`crContactReduce5`; re-run against the cluster function's own text, five of them
die.

The four survivors have proofs, recorded in the module header: `label`'s initial
value is never read (the first anchor always assigns); reversing the edge
direction negates `d`, which merely swaps `f` and `g` and leaves the anchor SET
identical; measuring `d` from `p1` instead of `p0` differs by `dot(dir, p1 - p0)`,
which is exactly zero because `dir` is perpendicular to `p1 - p0` by construction;
and the fourth is the deliberate control (argmax is invariant under `sqrt`).

Suite **691/691 in both memory modes**, forced-fresh.  The kernel itself
(`convexTrimeshCorrelate`) follows in P8.50b.

## P8.49 — `convexTrimeshPostProcess`, the silhouette cull, on the engine ABI

`convexTrimeshPostProcess` (convexMeshPostProcess.cu:369, core at :182) mapped as
an engine kernel — `cvxTriPostProcessBda.slang`, manifest row 266, ABI row of
eight pointers (64-byte blob).

Contact generation emits contacts on every triangle it touches, including on edges
and vertices shared with a neighbour that also produced contacts.  Those duplicates
fight each other in the solver, so this kernel re-examines each pair flagged for a
second pass and, when the feature it sat on is already covered, CLEARS the contact
count.  The logic is a lift of `npCvxPostProcess.slang` (the Phase-1 verbatim port
of the same core): what changes is only where the bytes come from — fixture arrays
become `Ptr<>` reads through the engine's real pointers.  The heightfield half of
that module is a different kernel and is not part of this one.

**No loop at all, so the chained loads are free.**  The CUDA is a grid-stride loop
whose body chains through `gpuShapes[shapeID].hullOrMeshPtr` — the P8.40 pattern
that segfaults llvmpipe.  Avoiding it costs nothing here: the host launches
`max(8192, 4 * numTests)` blocks of 64 threads (PxgNarrowphaseCore.cpp:2667-2670),
so the first pass alone covers at least 524288 second-pass pairs.  One workgroup
per CUDA block and ONE PAIR PER THREAD reproduces that pass exactly, with no loop
for a chained load to sit in.  The ceiling is P8.46c's, with four orders of
magnitude more headroom.

New suite `gpuslang.cvxTriPostProcess`: 19 cases over a deliberately non-manifold
6-triangle mesh, each one thread, each pinning one decision, with the expected
clear/no-clear derived by hand from the CUDA and written beside it.  The fixture's
structure is doing specific work:

* CM0's ordered list holds T1 as a FACE contact and T2 as an EDGE contact and
  nothing else — that split is what separates the two searches, because the edge
  branch looks only for face contacts while the vertex branch accepts either.
* v4's reference list puts its only match LAST, so a loop stopping one entry early
  misses it.
* T5 exists solely because it is the only triangle boundary on `.x` and `.z` but
  not on `.y`, which separates eV0's `(.x, .z)` pair from an `(.x, .y)` misread.
* `nbContacts` is 1 wherever it can be: 1 is the only count a shift by 28 instead
  of 27 turns into zero.
* CM2 has `count == 0` with a live-looking word at its slice start, which is the
  only way to reach `findTriangle`'s range guard (the core's `assert` against a
  zero count is debug-only; the guard is the release path).
* the real cases start at second-pass index 64, behind 64 no-ops, so dropping the
  `gid.x * 64` term reaches none of them; pair indices are `100 + case`, never the
  thread index.

**Sweep 48/51.**  The first pass was 40/49 and every one of the six closeable
survivors got a fixture case rather than an excuse — the shift width, the
accumulated-refs off-by-one, the two adjacency-word swaps, the range guard and the
loop bound.  Two mutants were added while closing them (a coverage-LOSING block
size, and the `.z`-for-`.y` mirror), both killed.  The three that remain are
structural:

| Survivor | Why no fixture reaches it |
|---|---|
| `PP_THREADS_PER_BLOCK` lowered to 32 | `numthreads` stays 64, so a smaller stride makes coverage a strict SUPERSET — some pairs processed twice.  The kernel's only effect is `index &= 0x00FFFFFF`, which is IDEMPOTENT, so a second visit is invisible by construction.  The constant does matter the other way and that IS tested: raising it to 128 loses pairs 64-127 and dies. |
| `PP_EMASK` widened to `0xf8000000` | It differs from `0xe0000000` only in bits 27-28, and the low half of the same word is the pair index.  Reaching those bits needs a pair index ≥ 2^27 — 134 million stack entries — so the separating input cannot be allocated. |
| the searches bounded by the ROUNDED count | The entries past `count` are the radix sort's `0xFFFFFFFF` padding, which is ≥ every target the searches can form (`index \| flag`, flag in the top two bits, index far below 2^27).  Extending the range can neither add a match nor move the index binarySearch returns.  Equality would need `triangleIndex == 0xFFFFFFFF`, which the boundary branch short-circuits first. |

One fixture bug found and fixed on the way, worth recording because it looked like
a shader bug for one run: the 64 no-op entries were marked with `0x0F000000`, and
`0x0F000000 >> 27` is **1**, not 0 — bit 27 is inside the count field.  The no-ops
therefore ran the full path and cleared a slot.  The marker is now `0x07000000`.

Suite **691/691 in both memory modes**, forced-fresh.

## P8.49a — a wrong BV32 stride that a clean mutation sweep could not have caught

`pxgSlangCvxTriSetup.slang` walked the cooked mesh blob with a BV32 node stride of
**160**.  `sizeof(Gu::BV32DataPacked)` (GuBV32.h:88) is **1168** — mMin[32] float4 =
512, mMax[32] = 512, mData[32] = 128, mNbNodes + mDepth + padding[2] = 16.  Every
section past the tree — verts, indices, adjacencies, faceRemap — therefore sat at
the wrong address against real device memory, off by `1008 * nbPackedNodes` bytes.

Found while starting P8.49, because `npCvxPostProcess.slang` (a Phase-1 module
covering the same blob) carries 1168, and so does every other mesh-walking shader
in this tree: `convexTrimeshMidphaseBda` via `pxgSlangBv32`, `femClothRefitBda`,
`npBv32`, `npMidphaseCvx`, `npSbSelf`, `psMeshMid`, and six more.  The setup module
was the only one that disagreed.

**Why the P8.46b sweep came back 40/40 clean anyway.**  `testCvxTriSetup.cpp`
built its cooked-mesh blob with `static const uint32_t BV32_STRIDE = 160;` — the
same wrong number, typed independently.  Fixture and shader agreed with each
other and disagreed with the engine.  A mutation sweep cannot see that: mutating
the constant in the shader makes it disagree with the fixture, so the mutant dies
and the sweep reports the constant as *well covered*.  **A fixture that DEFINES a
layout cannot check it.**  This is a different failure mode from the survivors
P8.44-P8.48 catalogued: those were mutants no input could distinguish; this was a
mutant the fixture killed for the wrong reason.

**The fix makes the compiler the oracle.**  The fixture now takes the stride from
the engine header:

```cpp
#include "GuBV32.h"
static const uint32_t BV32_STRIDE = uint32_t(sizeof(physx::Gu::BV32DataPacked)); // 1168
static_assert(sizeof(physx::Gu::BV32DataPacked) == 1168, ...);
```

and `testNpAbiLayout.cpp` — whose stated job is layout proof — now pins every
number the convex-trimesh BDA shaders bake in, against the real headers:

| Pinned | Value |
|---|---|
| `sizeof(Gu::BV32DataPacked)` | 1168 |
| `sizeof(ConvexMeshPair)` | 56 |
| `offsetof(ConvexMeshPair, startIndex / count / cmIndex / roundedStartIndex / materialIndices)` | 28 / 32 / 36 / 40 / 48 |
| `sizeof(ConvexTriNormalAndIndex)`, `offsetof(…, index)` | 16, 12 |
| `sizeof(ConvexTriIntermediateData)` / `ConvexTriContacts` / `ConvexTriContact` | 4 / 4 / 16 |
| `calculateConvexMeshPairMemRequirement()` | 56 |

The last two 56s are **different numbers that happen to coincide** — one is the
ConvexMeshPair stride, the other the per-pair stack budget
(midphaseAllocate.cuh:93).  Both are now pinned separately and the coincidence is
called out in `convexTrimeshCoreBda.slang`, so changing one is never taken as
licence to change the other.

Verified the fix is load-bearing rather than cosmetic: with the corrected fixture,
reverting the shader to 160 now FAILS `gpuslang.cvxTriSetup`, as does 1152 (one
node's trailing fields).  Before the fixture change, 160 passed and 1168 failed —
exactly backwards.

Suite **690/690 in both memory modes**, forced-fresh.

## P8.48 — `sortTriangleIndices` on the engine ABI: the first convex-trimesh kernel with NO coverage ceiling

`sortTriangleIndices` (convexMeshPostProcess.cu:623) mapped as an engine kernel —
`sortTriIndicesBda.slang`, manifest row 265, ABI row `{ 8, 8, 4 }` (a 24-byte blob).
It is a thin wrapper: it hands each contact manager's run of `faceFlag | triangleIdx`
words to the warp radix sort landed in P8.47.

**The dispatch is EXACT, which P8.46c's was not.**  The core kernel of P8.46c had to
give up coverage: its CUDA is a grid-stride loop whose trip count is a DEVICE value,
so a host-computed grid can only cover the first pass.  This kernel has no grid-stride
loop at all.  The host launches `ceil(numTests / 2)` blocks of two warps and passes
`numTests` as `nbPairs` (PxgNarrowphaseCore.cpp:2649-2654), so one workgroup per CUDA
block with a constant-trip-2 loop over its warps reproduces the CUDA's coverage
*identically*.  No ceiling, nothing silently skipped, nothing to close later.

**The offset trap this tranche exists to pin.**  `ConvexMeshPair` (convexNpCommon.h:70-81)
carries running END offsets in its trailing comments, so reading them as field offsets
puts `count` on `cmIndex` and `roundedStartIndex` on `pad`.  P8.46b nearly took that
route.  Here both reads — `count` @32 and `roundedStartIndex` @40 — are pinned by a
fixture that gives @28/@36/@44 plausible-but-wrong values pointing at scratch regions,
so a misread is a mismatch and never an out-of-bounds write.

New suite: `gpuslang.sortTriIndices` (testSortTriIndices.cpp), one test, seven contact
managers whose regions are laid out in an order that is NOT pair order — a kernel that
derived the run base from the pair index instead of reading @40 fails it.

**The sweep: 15 of 22 mutants killed, and the seven survivors are all accounted for.**
Six are structural; the seventh is the deliberate control (hoisting a loop-invariant
chained load, which must survive or the sweep is just rejecting every edit).

| Survivor | Why no fixture can kill it |
|---|---|
| either early-out dropped, or both | `(0 + 3) / 4` and `(0xFFFFFFFF + 3) / 4` are BOTH zero — the second because the round-up wraps — so a skipped run reaches the sort with zero uint4s and every loop inside has zero trips.  The guard is defensive, not load-bearing.  Kept: the CUDA has it, and a change to the rounding would make it matter again. |
| temp half at `input + count * 4` instead of `input + nbUint4 * 16` | The two differ only when the count is not a multiple of four, and then the wrong window starts exactly at the FIRST PADDING WORD — never on a live key — and ends *earlier* than the right one, so it can neither corrupt a key nor overrun the region.  What it does clobber is the sanitised tail, and the scatter re-sanitises on every read (pxgSlangRadixWarp.slang:208), erasing the clobber before it is used. |
| `NP_TRIMESH_WARPS_PER_BLOCK` raised to 4 | The HOST fixes the grid at `ceil(nbPairs / 2)`.  A wider block only stretches each block's range of UNIQUE warp indices, so every pair is still covered exactly once and the extra indices fall out on the bounds check.  Equivalent modulo wasted workgroups. |
| the loop-top barrier dropped | Every groupshared access in the sort is already bracketed by its own barriers (rsPass:192, the scatter tail at :256/:266, rsReduceAdd at :94/:96).  Provably redundant today; kept because it is the only thing that stays correct if those internal barriers move. |

**One survivor WAS a fixture gap, and it is now closed.**  Dropping
`globalWarpIndex >= nbPairs` survived the first sweep: with `nbPairs = 7` the grid
reaches index 7, and reading one entry past the array returned zeros, which the
early-out swallowed.  The fixture now allocates the pair array with ONE ENTRY MORE
than `nbPairs` and makes it a live-looking decoy pointing at scratch — which is what
the engine's capacity-sized array really holds.  The mutant now sorts the decoy into
scratch and dies.

Suite **690/690 in both memory modes**, forced-fresh.  *Recorded honestly:* the first
forced-fresh default-mode run reported `npRigidSumBatches (SEGFAULT)`.  It passed 5/5
when re-run alone and the whole default-mode suite came back 690/690 on a straight
re-run with no code change — a `-j4` flake, almost certainly memory pressure while
`cvxCollide` (≈350 s of llvmpipe) runs alongside.  It is noted here rather than
dropped, because a segfault that appears once should be findable if it appears again.

## P8.47 — the warp radix sort, and three survivors with real proofs

`radixSortSingleWarpKeysOnly` (RadixSort.cuh:579-720) as a 32-lane virtual warp: eight
four-bit passes, least significant nibble first.  This is what `sortTriangleIndices` calls,
and it is the only piece of the convex-vs-trimesh chain with no Phase-1 Slang to lean on.

### The shift is the algorithm

Both the histogram and the scatter select a sub-field by shifting a 1 by an amount that is
DELIBERATELY out of range for every key that does not belong to the word being accumulated.
`radix - bit` is unsigned: 0 or 1 for the radixes the word owns (0..3 in the scatter), 2 or
more — or an enormous wrapped value when `radix < bit` — for every other.  Only PTX's
CLAMPING makes a foreign radix contribute nothing.  Every such shift goes through `rsShl`,
which returns zero explicitly, because SPIR-V leaves the case undefined and the plausible
wrong answer would add 1 to the word's low half and corrupt the histogram of a sort.

### The oracle is std::sort, and the padding is not the sentinel

Eight least-significant-first four-bit passes ARE an ascending unsigned sort, so the expected
output is simply the input sorted — including the tail `sanitizeKeys` forces to 0xFFFFFFFF.
The fixture writes its own padding as 0xCDCDCDCD, deliberately NOT the sentinel, so a missing
`sanitizeKeys` shows up as padding sorting into the MIDDLE rather than as a no-op.  Keys
carry bit 31 on purpose: a signed comparison would put the flagged entries first.

Twelve cases straddle every boundary the pass loops have — below one uint4, exactly one, one
past it, a full warp of keys, exactly one warp of uint4s, one past that, and two strided
passes — plus reversed, all-equal and already-sorted inputs.

### Sweep 34/37, and every survivor is proved rather than argued

Unlike P8.45's and P8.46b's first sweeps, none of these is a fixture gap.  Each is
equivalent for a reason that can be checked by reading the code rather than by adding a case:

- **M3**, moving the shift guard from `>= 32` to `>= 31`.  The guard would then also zero a
  shift of exactly 31 — which never occurs: every shift amount here is `rank << 3` or
  `(radix - bit) << 4`, so it is a multiple of 8 or of 16.  31 is unreachable.
- **M11**, letting `sanitizeKeys` admit a full uint4 (`goodVals <= 4`).  At `goodVals == 4`
  the body computes `badVals == 0` and all three of its tests are false, so it does nothing.
- **M32**, dropping the scatter's `accum != 0` guard.  When `accum` is zero all four ranks
  are 4 or more, so all four inner `rank < 4` tests fail and nothing is written.  The guard
  is a pure optimisation, exactly as in the CUDA.

That the survivors are all provable, rather than needing new cases, is itself the same signal
P8.46b gave: an integer-only stage with no float boundaries leaves no room for the
"needs exact equality at a threshold" survivors that the SAT and reduction tranches carry.

Suite **689/689 in both memory modes**.  The kernel `sortTriangleIndices` itself is not yet
mapped — this is the algorithm it delegates to, verified on its own first, which is the same
order P8.44 through P8.46 followed.

## P8.47 recon — the shift rule, restated correctly, before it is relied on twice

Reading `radixSortSingleWarpKeysOnly` (RadixSort.cuh:579-699) for the next tranche turned up
an error in P8.45's own reasoning, and correcting it is worth more than the port it was
found during.

### PTX CLAMPS, it does not mask

The emit tail guards `(1 << numContactsGenerated) - 1` because the count can be exactly 32.
The guard is right and the fixture proves it.  The JUSTIFICATION written next to it was not:
it said PTX `shl.b32` masks the shift amount to five bits.  That is the x86 rule, and it does
not even explain the behaviour it was invoked to explain — under masking `1 << 32` is 1 and
`(1 << 32) - 1` is 0, whereas the CUDA demonstrably lands on 0xFFFFFFFF.  PTX **clamps** the
shift to the register width, so `1 << 32` is ZERO and the subtraction gives 0xFFFFFFFF.

### Why the distinction is load-bearing rather than pedantic

`radixSortWarp` (RadixSort.cuh:135-170) packs two radix counts into each 32-bit accumulator
and selects the half with

```cuda
accum = (1u << ((radix.x - bit) << 4));
```

`radix - bit` is UNSIGNED.  For the two radixes this slot owns it is 0 or 1, giving `1 << 0`
and `1 << 16`.  For every other radix it is 2 or more — or, when `radix < bit`, an enormous
wrapped value — so the shift is 32 or more and CLAMPING makes the term zero: the radix
contributes nothing to a slot it does not belong to.  Under masking it would contribute 1 to
the slot's LOW half and quietly corrupt the histogram of a radix sort.

So the same construct appears twice in this chain with OPPOSITE intents — one wants
0xFFFFFFFF out of an over-wide shift, the other wants 0 — and both are correct only under
clamping.

**The rule for the port: a shift by the operand width or more must be written out
explicitly, because the CUDA relies on it producing zero.**  SPIR-V leaves it undefined, so
neither behaviour may be assumed.  This joins the two rules already standing — no chained
buffer-device-address load inside a grid-stride loop (P8.40), and no flag carried through a
float select (P8.43).

### What else the recon found in the sort

- **`sanitizeKeys` (RadixSort.cuh:110-129) is a FALLTHROUGH switch.**  `badVals == 3` sets
  `.y`, `.z` and `.w`; `badVals == 2` sets `.z` and `.w`; `badVals == 1` sets `.w` alone.
  Transcribed as a normal switch it would leave stale keys in the tail of the last uint4,
  and they would sort into the result as ordinary values.  It never touches `.x`, which is
  safe only because the caller's index arithmetic keeps `goodVals` at least 1.
- **The pass count's parity is load-bearing.**  The driver ping-pongs between the input and
  the temp buffer once per pass and runs EIGHT of them, so the sorted result ends up back in
  the ORIGINAL buffer.  That is why `sortTriangleIndices` can pass
  `orderedCvxTriInterm + startIndex` as the input and the same pointer plus `nbUint4` as the
  temp, and why the caller reads its answer from the input buffer afterwards.  An odd pass
  count would leave the answer in the scratch half.
### The padding question, answered — and the recon's own framing of it was wrong

The emit-stage recon left open whether uninitialised padding in `orderedCvxTriInterm` can
sort into the binary-searched prefix, and the note above framed it as two problems.  It is
one, and it is closed.  There are TWO regions of padding and they are not the same kind of
thing:

- **The rounding padding**, between a pair's `count` and its uint4-rounded count.  The sort
  DOES read it: `numUint4 = (count + 3) / 4`, so the last uint4 straddles the boundary.
  `sanitizeKeys` is exactly what covers it — it forces every lane past `count` in that last
  uint4 to 0xFFFFFFFF.  Those entries are then sorted like any other key and land at the END,
  because a real ordered word is `faceFlag | triangleIdx` with the flag in bit 31 and the
  index below 2^27, so at most 0x87FFFFFF — strictly less than the sentinel.  The consumer
  bounds itself by `count` (`findTriangle` and its two siblings pass the pair's own count to
  `binarySearch` and then re-check `index < count`), so it never reaches them.
- **The doubled half**, which the midphase's padded counter reserves with
  `((nbPairsPerCM + 3) & ~3) * 2`.  That is NOT padding at all — it is the radix sort's temp
  buffer, handed to `radixSortSingleWarpKeysOnly` as `input + nbUint4`.  It is fully written
  by the first pass before the second reads it, and the eight-pass parity leaves the answer
  back in the first half.  The recon called this "separate padding the doubled counter
  reserves", which mistook a scratch buffer for uninitialised data.

So nothing uninitialised reaches the prefix, and the port has to preserve two things for
that to keep holding: `sanitizeKeys`' fallthrough, and the sentinel being strictly greater
than any real key.  The second is worth an assertion in the fixture rather than a comment —
a mesh with a triangle index at 0x7FFFFFFF would make a real key collide with the sentinel,
which is the same 2^27-boundary family as the NI word's carry.

## P8.46b — the setup prologue, and a fixture made almost entirely of identities

`convexTrimeshNarrowphaseCore`'s prologue (convexMesh.cu:99-210): the shape and transform
resolve, the walk down the cooked mesh to the adjacency and face-remap sections, and the
triangle in convex-local space with its normal.  With this, every input
`convexTriangleContactGen` needs exists.

### The offset this almost got wrong

`ConvexMeshPair`'s declaration (convexNpCommon.h:70-81) carries trailing numbers that are
RUNNING END OFFSETS, not field offsets.  Read as field offsets they put `startIndex` at 32
and `roundedStartIndex` at 36 — which are `count` and `cmIndex`.  Both are small
non-negative integers, so the pair records would land at plausible places in the output
arrays and nothing would fault.  The real offsets are 28 and 40.  The fixture gives all four
fields different values so the confusion cannot pass.

### Three more things the transcription pins

- **The mesh header's `.z` word is not the node count** — `.w` is.  Mis-sizing the BV32 tree
  shifts the vertices, the indices, the adjacencies and the face remap all at once.  The
  fixture fills the tree with a recognisable pattern rather than zeros, so a mis-sized walk
  reports nonsense instead of zeros that could pass for data, and gives its two meshes
  different node counts so a wrong stride cannot accidentally agree between them.
- **The swap.**  Slot 0 must end up the convex and slot 1 the mesh, and the transform refs
  swap WITH the shape refs, before the transforms are read.
- **The winding flip.**  A negative scale DETERMINANT reverses the mesh winding, and the
  CUDA compensates with `writeIndex = lane ^ 3` — which for lanes 1 and 2 is exactly a swap
  of the last two vertices, lane 0 being guarded by `lane > 0`.  Without it the triangle
  normal points into the mesh and every contact is generated against the far side.

The transform composition runs the OTHER way round from the midphase's: this one is
convex-relative, `conj(q0)*q1` with `rotInv(q0, p1 - p0)`, where the midphase is
mesh-relative.

The module is warp-uniform throughout.  The CUDA spreads the work across lanes and syncs
three times; here the values are cheap enough that computing them redundantly costs less
than the barriers, and it removes three uniformity preconditions from the caller.

### Sweep 33/40, then 40/40 — and six of the seven survivors had ONE cause

Every survivor of the first sweep was a real fixture gap rather than an equivalence, and six
of them came from the same thing: **the fixture was made almost entirely of identities.**

- Every convex body transform was the identity, which hides THREE separate mistakes at once.
  The composition is `conj(q0)*q1` with `rotInv(q0, p1 - p0)`, so with `q0` the identity the
  conjugate is a no-op, the translation needs no rotating, and `quatRotateInv` cannot be
  told from `quatRotate`.
- Every mesh scale rotation was the identity, which degenerates `vertex2Shape` to a
  componentwise multiply and makes both quaternion rotations inside it invisible — that
  accounts for the seventh survivor too.
- The transform refs equalled the shape refs in every contact manager, so the contact
  distance — which is indexed by the TRANSFORM refs — read the same slots either way.
- The only negative scale had exactly ONE negative component, where "determinant < 0" and
  "any component < 0" agree.
- No triangle was degenerate, so `getNormalized` never took its zero branch.

The fixture now carries a rotating convex body, a rotated mesh scale, transform refs that
diverge from shape refs, a scale with TWO negative components (positive determinant, so NO
flip) and a degenerate triangle, each pinned by its own coverage assertion.  The second
sweep kills all forty.

**A clean sweep is a statement about the STAGE, not only about the fixture.**  This is the
first in the convex-vs-trimesh chain with no survivors, and the reason is that it is pure
address arithmetic and exact transforms: there are no float-boundary mutants of the kind
that survive in P8.44 and P8.45 because no fixture can construct exact equality at a
threshold, and no unreachable code of the kind P8.44 documented.

Suite **688/688 in both memory modes** — one suite more than P8.46a, the new `cvxTriSetup`.
Still no manifest entry and no ABI row; what remains for the engine kernel is the wiring
itself — the shader that composes the six
modules, the ABI row for the 168-byte blob, and the manifest row.

## P8.46a — the pinned-host pointer the engine kernel has to write to

A prerequisite for `convexTrimeshNarrowphase`, resolved before writing a line of it, because
the answer could have made the kernel's prologue unportable.

### The question

The kernel's prologue does an `atomicMax` on `maxTempMemRequirement` (convexMesh.cu:674).
That word is allocated in PINNED HOST memory (PxgNarrowphaseCore.cpp:223, via
`mMappedMemoryAllocators` → `cuMemHostAlloc`) and read on the CPU with NO COPY AT ALL
(:1644), where it drives the `collisionStackSize buffer overflow detected` warning.  Two
things had to be true for the port to keep that behaviour, and neither was established.

### One: atomics on pinned memory work

`p3ShimPinned` already proved a pinned allocation is device-writable and host-coherent
through its mapping — but only for ordinary STORES.  An atomic read-modify-write is a
separate promise.  `p3ShimPinnedAtomic` now drives 4096 invocations contending on one pinned
word with `InterlockedMax` and a second with `InterlockedAdd`: the maximum is exact over a
deliberately non-monotonic sequence, no increment is lost, and the CPU reads both through
the mapping without a DtoH.  So the prologue's atomic is neither a fault nor a silent no-op.

### Two: the pointer itself does not survive the trip — except here

Under CUDA's unified addressing a `cuMemHostAlloc` pointer IS a valid device pointer, which
is why PhysX hands such pointers straight to kernels.  This runtime backs pinned memory with
a host-visible Vulkan buffer (`PxgCudaShim::memHostAlloc`), and **the blob builders copy
argument bytes verbatim** — there is no translation anywhere on the launch path, in either
the sized or the `void**` form.  A pinned pointer argument would therefore reach the shader
as a CPU virtual address.

`PxgKernelAbiRow` now carries a `hostPtrMask`: bit *i* marks argument *i* as a pinned-host
pointer, and both blob builders substitute the buffer device address via
`hostGetDevicePointer` (which already resolves interior pointers).  A null pointer stays
null, matching the kernels' own null checks; a non-null pointer that does not resolve is a
REFUSED LAUNCH rather than a silent zero, because the alternative is a shader writing to
address 0.

**The mask is inert today and, on this platform, untestable.**  llvmpipe is a software
rasteriser living in the calling process's address space, and its buffer device addresses
ARE the CPU mappings: the test reports `host == dev == 139835543642112`.  An untranslated
pinned pointer therefore works here BY ACCIDENT, exactly as it does under CUDA, and no
fixture on this device can tell a translated argument from an untranslated one.  The test
says so in a printed note rather than asserting it in either direction — demanding the two
differ fails here, and demanding they match would bake a backend accident into the suite.
This is the same category as P8.45's write-order masking: a correctness requirement the
platform cannot demonstrate, named rather than argued away.

### Scope of the hazard

Four kernels take that pointer — the convex and sphere trimesh and heightfield cores
(PxgNarrowphaseCore.cpp:2019, :2319, :2625, :2927) — and ALL FOUR are still unmapped
(PxgSlangKernelManifest.h:88, :95, :96, :97).  So this is not a fix for a live fault; it is
the thing that has to exist before any of them can be brought up.  Every other pinned
allocation in the engine (`mBpDesc`, `mPairManagementData`, the patch/contact readback) is
staged through a copy rather than passed as a kernel argument, and is unaffected.

Suite **687/687 in both memory modes**, forced-fresh.  The change touches the launch path of
every kernel, so the full run in both modes is the point rather than a formality.

## P8.45 — the emit tail, and a mutant that cannot be killed on this platform

The tail of `convexTriangleContactGen` (convexTriangle.cuh:1769-1881) as one unit: the
reduction, the temp-stream reservation, the scatter, and the five per-pair records the rest
of the chain reads.  With this, everything between the hull blob and the engine kernel
exists.

### Why one unit and not the two the existing scalar shaders suggest

The tail looks like two stages, and there are two Phase-1 scalar shaders shaped that way.
That is the wrong seam.  The output half reads the RAW clipper staging slot 0, recovers the
triangle's other vertices by unconditional shuffle, and runs for EVERY pair INCLUDING SAT
failures, while the emit half runs only inside `if (isSATPassed)` and
`if (numContactsGenerated)`.  They are one control-flow unit over one warp's groupshared
state.

### Five things kept that look like mistakes

Each is documented in the module together with what "fixing" it would cost:

- `.index` is stored BEFORE the overflow test and the atomic reservation is never rolled
  back.  On overflow the index stays out of range and only the zeroed `nbOutputContacts`
  stops the post-process dereferencing it — the downstream filter is on the COUNT, not the
  depth, so reordering the store is not equivalent.
- `delayContacts &&= numContactsGenerated` ANDs the RAW clipper count, not the
  `nbOutputContacts` the overflow guard may have just zeroed.  Under temp-buffer pressure a
  port that used the output count would append fewer pairs to the second-pass list.
- The second-pass classifier reads raw staging slot 0 — the clipper's FIRST contact, which
  the reduction may well have discarded.
- `mask | triangleIdx` is a bitwise OR while the neighbouring NI word is an ADDITION.  The
  mixed spelling is load-bearing: a remap index at or above 2^27 CARRIES into the four-bit
  count field, and the fixture now has a case that proves it.
- `getTriangleActiveMask` returns 0 for a contact strictly inside the triangle, which the
  post-process turns into a silent kill.  The obviously-safe default would KEEP contacts
  PhysX drops.

### One deliberate deviation, and it is a narrowing

The CUDA writes `verts1` only on lanes < 3 and only inside the SAT branch, then shuffles it
UNCONDITIONALLY — so a SAT-failed pair shuffles a register that was never written.  Here the
three vertices are warp-uniform parameters: the same values wherever the CUDA's are defined,
defined everywhere the CUDA's are not.  Staging them under the branch instead would put a
barrier in non-uniform control flow, which llvmpipe hangs on rather than erroring.

### The atomics shaped the fixture

Reservation order across groups is not predictable, so the test asserts what is: the cursor
advances by the sum of EVERY reservation including the dropped ones (the CUDA never rolls
one back), each payload is checked at the index the GPU itself reported, the slices tile the
reserved range with no gap or overlap, and the second-pass list is compared as a multiset.
The three overflow cases run one pair per dispatch so `startIndex` is deterministic.

### Sweep 29/41, then 35/41

The first sweep left twelve survivors and — unlike P8.44's — SIX OF THEM WERE REAL FIXTURE
GAPS, each blind for a reason worth recording:

- every case carried the normal `(0, 0, 1)`, so the x and y words of the NI record were both
  zero and permuting them in the four-lane cooperative write was invisible;
- the deepest contact always sat below lane 16, which the default separation ramp happened
  to guarantee, so dropping the ONE butterfly stage that crosses between the warp halves
  changed nothing;
- every separation was negative, so the unselected lanes could never win the depth minimum
  and taking it over all lanes agreed with taking it over the selected ones;
- no triangle was degenerate, so the zero-area guard in the barycentrics was never taken;
- five contacts never included a duplicate, and at exactly five `>` and `>=` as the
  reduction gate agree unless the dupe cull has something to do;
- no remap index reached 2^27, so the ADD-versus-OR distinction in the NI word was invisible.

### The case that found a bug in the test rather than the shader

Adding the 2^27 remap index turned the test red — and the shader was right.  The expected
contact count was being recovered from the GPU's own NI word by masking and shifting, which
is circular, and it breaks on exactly the case that matters: the word is an ADDITION, so the
carry makes the read-back count one too high.  The expectation is now derived from the
reference; only the overflow VERDICT still comes from the reported index, because the
reservation order genuinely cannot be predicted.

### The six remaining survivors, and one that cannot be killed here

- **M5 and M6**, feeding the reduction unguarded staging and widening the live-lane test by
  one.  Both are covered by the argument the module already carries: the reduction
  neutralises out-of-set lanes before its butterflies and ANDs the set mask into every
  ballot, so their values can neither win an anchor nor set a cull bit that survives.
- **M34**, `+` instead of `|` in the ordered word.  Equivalent while the triangle index fits
  in 31 bits, which the 27-bit field guarantees.  (Its twin in the NI word is NOT equivalent,
  and is now killed.)
- **M28**, making the 0.97 region bound inclusive — needs exact float equality at the bound.
- **M24**, "the classifier reads the first EMITTED contact".  NOT FORMULABLE IN THIS MODULE:
  the raw slot is an explicit parameter and the classifier runs on lane 0, where the lane's
  own staged value IS slot 0.  The mistake is only available to the CALLER, so it belongs to
  the engine-kernel tranche, not here.  Recorded rather than argued away.
- **M20**, letting every lane scatter instead of only the selected ones.  This one is
  **provably unkillable on this platform**, and the proof is worth having.  The exclusive
  scan increments exactly at a selected lane, so the lanes that collide on slot `c+1` are the
  non-selected lanes above selected lane `j` plus the next selected lane `j'` — and `j'`, the
  one holding the CORRECT value, is always the HIGHEST lane in the group.  Any implementation
  that resolves same-address writes in ascending lane order therefore lands on the right
  value every time.  That is not equivalence; it is a platform-dependent masking, and no
  fixture on llvmpipe can see through it.

Suite **687/687 in both memory modes** — one suite more than P8.44, the new `cvxTriEmit`.
Still no manifest entry and no ABI row — the next tranche is `convexTrimeshNarrowphase` itself, where the chained buffer-device-address loads
of the setup stage finally do fall under the P8.40 rule.

## P8.44 — contactReduceShared on the virtual warp, and a fill loop that turns out to be dead

`contactReduceShared` (contactReduction.cuh:55-165) at the ONE instantiation the
convex-vs-triangle emit tail reaches — `contactReduce<false, true, 5, true>`
(convexTriangle.cuh:1789) — plus `maxIndex`/`minIndex` (nputils.cuh:137-163).

### Why port something that is already green

`contactReduce.slang` covers the same algorithm and its test drives 4096 random cases through
it, but it is a SCALAR harness: one thread walks all 32 contacts out of per-lane arrays.  That
proves the ALGORITHM and says nothing about the warp.  So the new module is driven against the
SAME C++ oracle, `refReduce`, which is what makes a divergence attributable to the warp
emulation rather than to the algorithm — and both tests now live in the same file so a case
added for one is available to the other.

### maxIndex is not the generic helper, and the difference is real

`pxgWarpReduceMaxWinnerF` (pxgSlangWarpInc.slang:313) resolves the winner as
`lowestSetIndex(ballot(mask, best == value))` — ONE mask in both the sync role and the
selection role.  The CUDA ballots over `FULL_MASK` and ANDs the RESULT with the set mask,
having neutralised out-of-set lanes with `-FLT_MAX` BEFORE the butterfly.  The equality is
tested against the lane's ORIGINAL value, so an out-of-set lane holding a value equal to the
in-set maximum sets its ballot bit and only the `& mask` removes it.  Random data essentially
never lands there; the fixture constructs it — lane 0 outside `allMask`, carrying the winning
separation.  That case kills the mutant.

### Self-contained on purpose

No `#include "pxgSlangWarpInc.slang"`.  The emit tail composes this module with
`pxgSlangConvexSat`, which does include it, and a Slang `import` is a module boundary: each
importer would carry its own copy of the warp library's groupshared scratch.
`pxgSlangPolyClip` solves it the same way, and the three primitives needed here — shuffle,
ballot, ffs — are forty lines.  Everything is prefixed `cr`/`gsCr`.

### Sweep 27/37, and the ten survivors are accounted for individually

Six are EQUIVALENT BY PROOF, not by argument:

- **M26 / M27**, negating the edge direction.  `d` flips sign, so the argmax and the argmin
  swap — and the two are ORed into the same `mask`, so the union is identical.
- **M28**, measuring the signed distance from the origin instead of from p0.  That adds
  `dot(dir, p0)`, which is warp-uniform, to every lane; an argmax is invariant under adding a
  constant.  (Exactly invariant in exact arithmetic; the fixture's near-tie tracker skips the
  cases where rounding could separate two previously equal values.)
- **M25**, squaring the distance instead of taking the magnitude.  `sqrt` is monotone on the
  non-negatives and maps equal inputs to equal outputs, so both the winner and the tie
  structure are preserved.
- **M17**, the early-out off by one the other way.  With exactly five candidates the full path
  selects four distinct anchors and then the PCM pick, i.e. all five — the same mask the
  early-out would have returned.
- **M31 / M32**, moving or removing the `popc(mask) == 4` gate on the PCM pick.  At
  `TClustering = false` the PCM pick and the fill loop use the SAME criterion, the raw
  separation, so it does not matter which of the two contributes the fifth point.  The gate
  only earns its keep with clustering, where the fill switches to the BIASED separation.

Two mutate code that cannot run:

- **M34 / M37**, the fill loop.  `i0` is drawn from `allMask`, `i1` from `allMask & ~mask`, `f`
  and `g` from the mask minus what came before — so the four anchors are DISTINCT BY
  CONSTRUCTION, and the sets are non-empty because the early-out already guaranteed more than
  five candidates.  Therefore `popc(mask) == 4` always holds here, the PCM pick always fires,
  `count` is always 5, and the fill loop's range `[5, 5)` is empty.  **The fill loop is dead
  code at TMaxPoints = 5.**  It is kept because `TMaxPoints` is a template parameter in the
  original and the scalar test does reach it at 6.  M35 and M36, which force the loop to run,
  are both killed — which is the empirical half of the same claim.

The last two need a coincidence no fixture can construct: **M12** wants `d` exactly equal to
`1e-8f` at the dupe gate.

### The case that killed the anchor mutant, and why the obvious version of it did not

"p1 may re-pick p0" survived the sweep, and the first case built for it did not kill it either
— which was the more useful result.  Making every contact project onto the same point in the
normal plane does drive the mutant down the wrong path: every distance is zero, the anchors
fall back to the lowest-lane rule, and a `p1` that does not exclude the anchors already chosen
re-picks p0 and ends up one anchor short.  The returned mask was NEVERTHELESS IDENTICAL,
because the fill loop picks the deepest remaining contacts and restored exactly the lane the
bad pick had lost.

The case only bites once DEPTH ORDER AND LANE ORDER ARE DECOUPLED — lane 3 made shallow so the
fill never reaches for it, the two deepest contacts parked out at lanes 10 and 20.  The
original then returns `{0,1,2,3,10}` and the mutant `{0,1,2,10,20}`.  It is a stack of contacts
along the normal, which is an ordinary thing for a clipper to produce.

That compensation is the same mechanism that keeps M31/M32 alive, and noticing it is what
turned the fill-loop survivors from "the fixture is thin here" into the reachability proof
above.

Suite **686/686 in both memory modes** — the SAME count as P8.43, deliberately.  `ctest` counts
suites, not tests, and `contactReduceWarp` was given no suite of its own: it lives in
testContactReduce.cpp beside the scalar port and runs under the existing
`gpuslang.contactReduce`.  A suite of its own would have matched by substring and run the test
twice.  Still no manifest entry and no ABI row; the next tranche is the emit tail itself
(convexTriangle.cuh:1769-1881), which this module exists to serve.

## P8.43 — a signed zero that llvmpipe folded away, and the joint that never broke

A forced-fresh rebuild — every `.spv` deleted first — turned `solverTGS_writebackBlocksAbi`
red with 16 mismatches, in a shader that had not been touched since P5 Step 4 tranche 20.
Every mismatching record was bit-identical except for ONE component:
`ConstraintWriteback::linearImpulse_broken.w`, where the GPU wrote `+0.0` and the reference
`-0.0`.

### Why that single bit is the whole flag

PhysX encodes "this joint broke" as the SIGN of a zero (`solverBlockTGS.cuh:1172`):

```cuda
writeback.linearImpulse_broken = make_float4(linVel.x, linVel.y, linVel.z, broken ? -0.0f : 0.0f);
```

The port transcribed that literally.  **Vulkan does not require signed-zero preservation** —
`SignedZeroInfNanPreserve` is an optional execution mode — so a compiler is free to treat the
two arms of `cond ? -0.0f : 0.0f` as equal and delete the select.  llvmpipe does exactly that:
`0 of 139` writeback records came back with the sign bit set where the oracle had `16`.

Selecting the BIT PATTERN and reinterpreting it survives, because the select is then an integer
one and no float rule applies:

```slang
float4(linVel, asfloat(broken != 0 ? 0x80000000u : 0u))
```

**The fold is not deterministic across shaders.**  `solverPGS.slang` carries the same three
lines with the same shape, and its 29 broken joints came back with their sign bits intact even
before the fix.  Two shaders, one source pattern, opposite outcomes — which is the argument for
spelling it defensively in both rather than only where it was observed to break.

### The rule this hands the port

**A bit pattern that carries a flag must never travel through a float select.**  Signed zeros
and NaN payloads are not values the compiler is obliged to keep.  P8.37 already met the other
half of this — it moves manifolds as `uint4` rather than `float4` so NaN canonicalisation
cannot touch them — and this is the same rule arriving from the arithmetic side.

### What made it invisible, and what now makes it loud

The PGS twin's fixture DOES break joints, and its test still passed with the bug present,
because on that shader the sign happened to survive.  Nothing in either test asserted that the
sign-bit path was exercised at all, so a fixture that never broke a joint would have read as
full coverage.  Both writeback tests now count the broken joints their oracle produces and
assert the count is non-zero — the same shape of coverage assertion the SAT tranches use.

### On when this broke

The shader source has not changed since the writeback tranche landed, and the failure appears
on the first forced-fresh build after a container restart.  Whether an older `.spv` on disk had
been compiled by a toolchain that kept the select, or the suite had been reporting green over a
stale binary, cannot be established after the fact — the old `.spv` files were deleted to force
the rebuild that exposed this.  What is certain is that the CURRENT toolchain folds it and the
current source no longer relies on it.

## P8.42 — the SAT driver and the face selection the clipper consumes

`convexMeshSAT` (`convexTriangle.cuh:868`), `delayGenerateContact` (:1402) and
`selectPolygonFeatureIndex` (:1420): the piece that turns a SAT verdict into the one or two
convex faces `polyClip` runs against.  With this, everything between the hull blob and the
already-ported clipper exists.

### Three things in the driver that are not visible from the call sequence

- **The barrier sits BEFORE the first early return.**  Stage 1 writes `convexPlaneN/D` and the
  CUDA's `__syncwarp()` at :877 is ahead of `if (!passedSAT) return false`.  `passedSAT` is
  warp-uniform, so every lane reaches it whatever stage 1 decided; moving it after the return
  would put a barrier in divergent control flow, which on llvmpipe hangs rather than errors.
- **The three seeds are the caller's** (:1649-1651), folded in here because
  `convexTriangleContactGen` is the only caller.  `minNormal` is seeded with `+PX_MAX_REAL` in
  ALL THREE components — not the zero vector the per-stage fixture uses.
- **Only a convex FACE axis can delay contacts.**  If the winner was the triangle normal or an
  edge cross, `delayGenerateContact` is false whatever the geometry.

### selectPolygonFeatureIndex, where every mistake is quiet

- `maxDpSq` is seeded with `minProj * minProj` — the REDUCED, warp-uniform best face
  projection.  An edge only displaces the face if it beats the face's own alignment, so the
  seed is the comparison's floor, not merely an initial value.
- the edge test is `sqDp > maxDpSq * enMagSq` but the store is `sqDp / enMagSq`.  Multiplying
  on one side and dividing on the other is how the CUDA compares normalised alignments without
  normalising `edgeN`.
- `maxDpSq2` starts at `-10.0f`, below any squared dot product, so the first neighbour of the
  winning face a lane meets always takes it.
- the second reduction is skipped entirely when an edge won, because the two faces then come
  from the edge itself.

### The fixture: a shared case list, and why the box was not enough

The case list is now shared by both tests in the file — `convexSatStages` checks the three SAT
stages one at a time, `convexSelectFace` checks the driver and the selection that runs on their
result — so a case added to sharpen one is automatically under the other's oracle.

The first sweep came back **12/24**, and the reason was one property of the fixture hull rather
than twelve separate gaps.  **Every box edge joins two PERPENDICULAR faces, so every box edge
normal has the same squared length, 2.**  With `enMagSq` constant, `sqDp > maxDpSq * enMagSq`
and `sqDp > maxDpSq / enMagSq` order the edges identically, the seed's scale stops mattering,
and four mutations become invisible at once.  Three cases against the 34-polygon drum — whose
edges join two side faces 11.25 degrees apart (`enMagSq` about 3.92) or a side face and a cap
(exactly 2) — killed six survivors between them.

Two more cases close a different gap.  Every stage-1 rejection in the fixture was ALSO a
stage-2 rejection, so a driver that ignored stage 1's verdict still returned "did not pass" and
read as correct.  A triangle out at x in [-3, -1.5] is rejected by the -x face while its own
normal axis still overlaps, and a large slanted triangle whose plane misses the box is rejected
by the triangle stage while every face and edge-cross axis overlaps.  `convexSatStages` now
asserts both rejections, which it previously tracked and threw away.

### What the driver leaves behind when the SAT fails is asserted too

That still left the stage-1 mutant alive, and the fix was not another case but a stronger
oracle.  `convexMeshSAT` returns by reference and does not undo a commit: a pair rejected by
stage 2 or 3 carries whatever stage 1 wrote, and ONLY a stage-1 rejection leaves the caller's
seeds untouched.  Asserting those outputs unconditionally — rather than only when the SAT
passed — is what makes "ran a stage it should have skipped" observable at all, and it killed
both the ignored-verdict mutant and the wrong-seed one.

### Sweep 21/24

The three survivors are argued:

- **S1**, dropping the driver's barrier — redundant on this platform because stage 1's last
  groupshared write is followed by `pxgAnySync`, whose emulation already carries a barrier that
  every lane reaches before the array is read.  The explicit one is kept because it is what the
  CUDA has and because that helper's internals are not a contract.
- **S10**, seeding the face argmax with 0 instead of `-PX_MAX_REAL` — for a CLOSED hull the
  face normals positively span R³, so some face always has a strictly positive dot with any
  direction and the seed can never survive the loop.
- **S18**, seeding the neighbour search with 0 instead of `-10.0f` — the quantity compared is a
  SQUARE, so the two seeds differ only when the best neighbour's alignment is exactly zero.

Suite **686/686 in both memory modes** — one test more than P8.41, the new `convexSelect`.
Still no manifest entry and no ABI row; the next tranche is the emit stage that turns the
clipped contacts into the engine's output buffers.

## P8.41 — satEdgeCrossDirection2, the third and last SAT stage

`satEdgeCrossDirection2` (`convexTriangle.cuh:575-696`) closes `convexMeshSAT`: for every
(convex edge, triangle edge) pair the cross product is a candidate separating axis.  It is the
only stage that READS what an earlier stage wrote — the shape-space plane normals stage 1 left
in groupshared are its gate — so getting it green also confirms P8.40 wrote them correctly.

### What the transcription keeps, and what it drops

- `_separation` starts from the **running best**, not `-PX_MAX_F32`, so an edge axis only wins
  if it strictly beats what the face and triangle stages already committed.
- The two hull side-tables are read with the CUDA's little-endian trick: an edge's two vertex
  indices as one `uint`, its two face indices as one `ushort`.  Both runs are 4-aligned from
  the blob base, so the packed loads are legal — the argument P8.39 made for its own readers.
- `magSquared` and `dp` are computed and never used in the CUDA, and `convexScale` /
  `convexScaleRot` are loaded and never used.  Dropped: dead code cannot move an output bit.
- `shapeNormal` is the RAW triangle normal.  The `shape2Vertex` version sits commented out in
  the CUDA at :595 and stays commented out here.
- The `__any_sync` early-out is INSIDE the iteration loop, so a warp that finds a separating
  axis leaves without scanning the rest.  The loop bound is warp-uniform, so the reduction
  inside it is too — the P8.38 rule about barriers in provably uniform control flow.

Per P8.40's rule the caller resolves the two section base addresses and passes them in; the
stage never dereferences an address it read out of memory.

### Three fixture cases, each built for one thing the stage does

Reaching stage 3 is not the same as exercising it.  The stage was reachable the moment it was
wired up and every value already agreed with the oracle — and none of that observed the half of
the stage that only runs when an edge axis actually wins.

- **"edge-cross axis wins"** — a triangle whose first edge crosses diagonally past the box
  corner at x=y=1.  Along the resulting axis, the xy diagonal, the hulls overlap by 0.141
  against the 1.0 the best face normal gives, so the edge axis takes the lead and the tail of
  the stage — the argmax across lanes, the sign choice between the two overlaps, the
  `EDGE_FEATURE_IDX` write — becomes observable.  THE GATE MATTERS HERE: the vertical edge
  nearest the crossing is REJECTED, because its two faces are +x and +y and the triangle normal
  has a positive dot with both.  The axis arrives from the three other vertical edges.  All
  three yield the identical vector, so their separations are bitwise equal and the lowest-lane
  rule decides — and the lowest of the three is lane 1, holding index 33, a value carried out
  of the SECOND pass of the strided loop.
- **"separated by an edge-cross axis only"** — all six face normals and the triangle normal
  overlap; only the same xy diagonal pulls the pair apart.  This is the only case that drives
  `isPassed = false`, so it is the only one that exercises the early-out inside the loop.  Its
  first separating index is 24, inside the FIRST pass, so the loop leaves without ever looking
  at indices 32..35.
- **"edge-cross with adjacency flags"** — the same geometry with the triangle's second edge
  marked non-convex and its third carrying a plain neighbour index.  Every other case leaves
  all three at BOUNDARY, so without it `csIsActiveEdge` is only ever asked the BOUNDARY
  question.  It also pins the pairing between an edge's own flag and the edge it spans: closing
  the triangle the other way round, `(vStart+2)%3`, yields the same three edge directions
  negated, and the stage tries both signs anyway, so under all-BOUNDARY adjacency that mistake
  is **invisible**.  With the flags set it is not — the winning diagonal is never formed.

### What the second memory mode caught

The "separated by an edge-cross axis only" case first went in with `probeIndex = 6` on a hull
that has six polygons — one past the end.  The probe reads `convexPlaneN/D` at that index, and
stage 1 only ever writes the first `nbPolygons` entries of that groupshared array, so the read
returned whatever the previous dispatch had left there.  Under the default allocator that
happened to be zero, which is exactly what the oracle expects for an unwritten entry, and the
case passed.  Under `PXG_DEVICE_V2=1` it was not zero and the case failed.

Nothing about the port was wrong; the fixture was.  It is the clearest payment yet for running
both memory modes on every tranche rather than one.  The probe index is now asserted to be
inside the hull before the blob is built, so the same mistake fails loudly and in both modes.

### Sweep 28/37, and what the survivors mean

M2, M3, M4 and M8 are P8.40's four stage-1 survivors, unchanged and already argued there.  The
five new ones:

- **M23**, closing the triangle edge the wrong way round, **survived the first sweep and was
  then killed** by the adjacency-flag case above.  That is the one survivor this tranche turned
  into a kill rather than an argument.
- **M17**, seeding the edge stage from `-PX_MAX_F32` instead of the running best — equivalent
  by the same algebra as M1: the stage commits iff `max over edges > running`, and that
  predicate is unchanged by the seed.  Only the value held by lanes that processed no edge
  moves, and those lanes never win the argmax when an edge does.
- **M20** (the gate keeps the faces pointing TOWARDS the triangle) and **M25** (an edge's second
  face index unpacked from the wrong byte, so the gate tests the first face twice) both survive
  for the same structural reason.  Write A for the edges with at least one face pointing away
  and B for those with at least one pointing towards; every edge has two faces, so A ∩ B is
  exactly the SILHOUETTE edges, and both spellings of the gate keep all of them.  The winning
  edge-cross axis is attained at a silhouette edge, so neither mutation can move it.  That is
  an argument rather than a proof, so it was checked: several thousand randomised triangles
  against both the box and the 34-polygon drum produced **no** configuration where either
  mutation changed an output bit.
- **M36**, striding the edge loop by one instead of by a warp — every edge index is still
  visited, just redundantly, and the lowest-lane rule then resolves to lane 0 holding the same
  best value.  Slower, not different.

One anomaly is recorded rather than smoothed over: **M1 came back KILLED on the first sweep and
SURVIVED on three targeted re-runs**, with six back-to-back runs of the unmutated fixture also
clean.  M1 is equivalent by construction — the caller passes exactly `-PX_MAX_F32`, which is
what the mutation substitutes — so the first result was the outlier.

### The P8.13 one-off fired twice here, and gave up its first real evidence

The recurring one-off suite failure described in P8.13 — always the first parallel run after a
forced-fresh `.spv` rebuild, always the default mode, never `PXG_DEVICE_V2=1`, never reproducible
— fired **twice** in this tranche, in 2 of 5 full default-mode runs.  Both times it was
`gpuslang.scan`, and this is the first occurrence in the whole port where the failing test is
even known: every earlier one destroyed its own log before it could be read.

**What the log shows is new.**  The record for the failed run contains the command, the start
time, and then `<end of output>` — **not one line of the test's own output, not even the backend
banner** that every run prints before the first check.  It dies before or during device
creation, not inside any kernel.  That is consistent with every negative result P8.13 collected
(it is not a kernel, not a timeout, not disk, not the pipeline cache) and it narrows the target
sharply for the next attempt.

It did not reproduce in three solo re-runs, in 24 synthetic 8-way concurrent starts of the same
test, or in eight repeats of the first 24 tests at `-j4`.  The M1 sweep anomaly above fits the
same shape — a one-off in a run that had just deleted the `.spv` and rebuilt.

One process note against myself: these runs used bare `ctest -j4` rather than
`tools/run_suite.sh`, which P8.13 added precisely so an occurrence archives its own evidence.
The second crash's log survived only because nothing happened to overwrite it in between.  The
final verification for this tranche was re-run through the wrapper.

Suite **685/685 in both memory modes**.  Still not an engine kernel: no manifest entry, no ABI
row.  `convexMeshSAT` now has all three of its stages, so the next tranche is the clipper hookup
and the emit stage of `convexTriangleContactGen`.

## P8.40 — the first two SAT stages, and the llvmpipe fault that held them up for four passes

`satConvexFaceNormals` (`convexTriangle.cuh:217`) and `satTriangleNormal` (:303), plus the
vertex-scaling prologue that feeds them, on a 32-lane virtual warp.  The oracle is a scalar
32-lane simulation, the same shape as P8.38's.  What made this tranche expensive was not the
port — it agreed with the oracle on every value early on — but a **segfault inside the
JIT-compiled shader** that took five passes to pin down.

### The fault, and the thing that finally cracked it

Four passes of bisecting DOWN from the port ruled out, each by a build-and-run: the early
return, groupshared indexing, blob alignment, the two global loads, the divergent `break`, the
whole strided loop, the `pxgSlangWarpInc` include, the module boundary itself (everything
inlined into one shader), `spirv-opt -O` over the binary, `__exported import`, `inout` struct
fields, and data dependence.  `spirv-val` was clean throughout and `LP_NUM_THREADS=1` reproduced
exactly.  A one-line flip *was* found in a scratch reproducer, and then the fix it suggested —
computing loop bounds in the caller — turned out not to fix the real module.  That prediction
had been written into this file before it was tested; the correction is the P8.40 entry's own
history.

What cracked it was the suggestion to **take the environment out of the picture entirely**: a
standalone Vulkan host, 130 lines of C, no PhysX, no test framework, nothing from this repo but
the SPIR-V.  It crashed there too — which exonerated the harness and, far more usefully, gave a
five-second edit-compile-run loop instead of a forty-second one.  Building UP in that loop
reached the trigger in about fifteen steps.

### The trigger

```slang
for (uint c = gid.x; c < gCs.numCases; c += WorkgroupCount().x)   // CRASHES
{
    const uint64_t hull = Ptr<uint64_t>(gCs.input + uint64_t(c) * 112UL)[0];
    const uint desc = Ptr<uint>(hull + 16UL)[0];
    if (gtid.x == 0u) Ptr<uint>(gCs.output + uint64_t(c) * 64UL)[0] = desc;
}
```

A **chained buffer-device-address load** — a 64-bit address read out of memory and then
dereferenced — inside a grid-stride loop.  Replace the loop with `const uint c = gid.x; if (c <
gCs.numCases)` and the identical body runs.  No groupshared, no barriers, no imports are
involved in the reproducer at all.  Reassembling the address from two 32-bit loads still
crashes, and so does a `while` form, so it is the loop plus the indirection rather than the
64-bit load itself.  The loop executes exactly ONE iteration in the failing run.

That also explains why `polyClipBda` never hit it despite the same loop shape: its per-pair
addresses are computed from the push constant directly.  It never dereferences an address it
read out of memory.  **This is the first tranche in the port that needed a chained BDA load
inside its dispatch loop, and it is the first one that crashed.**

The reproducer — `iso.slang`, `host.c`, `try.sh` — is parked in the session scratchpad under
`p840-iso/`.  It is small enough to file against slangc or Mesa as it stands.

### The rule this hands the port

**Do not put a chained BDA load inside a grid-stride loop.**  Dispatch one workgroup per item
and guard with `if (index < count)`.  The consumer already dispatched one group per case, so
this cost nothing; every future kernel that reads a pointer out of a descriptor struct is
subject to the same rule.

### The tranche itself

Sweep **11/16**, and the five survivors are argued rather than ignored:

- **M2, the `break` turned into a `continue` — equivalent, and it corrects the module's own
  comment.**  I had written that a `continue` would let a later polygon owned by the same lane
  win the argmax.  It cannot: if any lane sets `disjoint` the function returns false and the
  argmax and plane arrays are discarded wholesale, so the break only saves work.  The comment
  now says that.
- **M1**, seeding the running best with the incoming `separation` instead of `-PX_MAX_F32` —
  the caller passes `-PX_MAX_F32`, so the two are identical in this call order.
- **M4**, dropping the second half of the separation test — for a CLOSED hull, whenever
  `min0 > max1 + contactDist` fires for one face, `min1 > max0 + contactDist` fires for the
  roughly opposite one, so the pair is still rejected.  That is an argument, not a proof, and
  it is recorded as such.
- **M3 and M8**, turning `>` into `>=` and `<` into `<=` — both need exact float equality at a
  boundary that the fixture does not construct.

Three fixture cases were added to kill the earlier survivors: a **34-polygon drum**, so a lane
owns more than one face and the strided loop actually iterates (this killed both the `break`
mutant's cousin and the one that records the lane instead of the polygon index); a pair
separated only by the second half of the test; and a **tilted triangle whose own normal wins the
axis**, without which the sign of the committed normal is never observed.  Coverage assertions
now pin all three.

Suite **685/685 in both memory modes**.  Not an engine kernel yet — no manifest entry, no ABI
row.  `satEdgeCrossDirection2`, the third SAT stage, is the next tranche.

## P8.39 — the cooked convex-hull blob reader, plus a survey of what the core kernel actually needs

`pxgSlangConvexHull` transcribes the `ConvexScratch` accessors (`convexTriangle.cuh:55-136`) as
address arithmetic over a Buffer Device Address.  Every remaining kernel of the convex-vs-mesh
chain re-derives these offsets — the SAT phase walks planes and edge lists, the clipper walks a
face's vertex list, the post-process walks faces-by-vertices — and one wrong offset reads
plausible floats from the neighbouring section, which is the failure mode hardest to spot
downstream.  Same reasoning as P8.25's BV32 reader.

The blob, from `PxgShape::hullOrMeshPtr`:

    @0                      float4  centerOfMass
    @16                     uint4   (.x = the packed hull descriptor)
    @32                     float4  (not read by this chain)
    @48                     float4  vertices[nbVerts]
    +16*nbVerts             float4  planes[nbPolygons]
    +16*nbPolygons          uint    polyDescs[nbPolygons]
    +4*nbPolygons           uint16  verticesByEdges16[2*nbEdges]
    +4*nbEdges              uint8   facesByEdges8[2*nbEdges]
    +2*nbEdges              uint8   facesByVertices8[3*nbVerts]
    +3*nbVerts              uint8   vertexData8[...]

**The two packed words are not the same packing.**  Both are three fields in one uint, but the
hull descriptor's top half is a 16-bit edge count while a polygon descriptor's top half is a
16-bit vertex reference.  The CUDA returns `PxU8`/`PxU16` from those decoders, so the masks are
load-bearing rather than decorative: a hull with 300 polygons wraps, and this port wraps
identically.

**The oracle is the construction, not a formula.**  The fixture builds each blob by APPENDING
sections and recording where each landed; the expected offsets are those recorded positions.
Restating the shader's closed-form arithmetic in the test would be circular — the same slip in
both would cancel — whereas append-and-record derives the layout from the section sizes alone.
Every section is also filled with values encoding (section, index), so a wrong address is caught
twice: by the reported offset and by the value read through it.  Seven hull shapes, including
counts that leave every run past the planes unaligned, and the 8- and 16-bit field maxima.

**Sweep 20/21, and the survivor corrects something I wrote.**  M14 replaced the byte reader's
address rounding with index rounding — the spelling this port originally had.  I described that
as a bug the fixture would catch.  **It is not a value bug at all**: both spellings extract the
byte at `base + byteIndex`, because rounding the index down and shifting by its low two bits
lands on exactly the same byte as rounding the sum down and shifting by the sum's low two bits.
What differs is only the address the uint load is issued at, which in the index-rounding
spelling inherits the section base's misalignment.  Vulkan requires a scalar load to be
naturally aligned, so that is undefined behaviour — but llvmpipe services it, and **no fixture
on this platform can distinguish the two**.  It is recorded as an equivalent mutant, and the
absolute-address form is kept because it is the one that is actually defined.

### What the recon established for the tranches after this one

A read-only survey (six parallel readers plus a completeness critic) mapped the rest of the
chain.  The load-bearing corrections to my earlier notes:

- **`convexTrimeshNarrowphaseCore` is `convexMesh.cu:74-215`, not 74-470**, and it is pure
  per-warp setup with no algorithm-selecting branch: it fills a `ConvexMeshScratch` and
  tail-calls `convexTriangleContactGen` (`convexTriangle.cuh:1640-1882`).  The sphere and
  capsule cores that follow it in the file belong to a different kernel.
- **The core performs zero global writes.**  All seven per-pair outputs are written by
  `convexTriangleContactGen`.
- **The SAT is three staged sub-phases with early-out** — `satConvexFaceNormals` (:217),
  `satTriangleNormal` (:303), `satEdgeCrossDirection2` (:575) — behind `convexMeshSAT<true>`
  (:867).  `CONVEX_MAX_VERTICES_POLYGONS` is 64.
- **`ConvexTriNormalAndIndex`** is 16 B (normal @0, index @12) with `DeferredContactMask =
  0x80000000` and `NbContactsShift = 27`; `ConvexTriContacts` and `ConvexTriIntermediateData`
  are 4 B each; `ConvexTriContact` is a bare float4 (point in convex-local space, `.w` =
  separation).
- **The second-pass entry packs `pairIdx & 0x1fffffff` with a `VoronoiState` in bits 29-31**,
  which is exactly what P8.30's classifier produces.

Four upstream oddities were found and are being left alone deliberately, recorded here so a
future reader does not "fix" them into a behaviour difference: `convexMeshPostProcess.cu:363`
masks with `0x0ffffff` (24 bits) where `getTriangleIndex` uses 27; `(1 << numContactsGenerated)
- 1` at `convexTriangle.cuh:1781` shifts by the full word width when the contact count reaches
its 32 ceiling; `cvxTriContacts[off].index` is written before the overflow check that zeroes the
count; and the convex path does not swap `triAdjTrisIdx.x/.z` under a negative mesh-scale
determinant while the sphere and capsule paths do.

Suite **684/684 in both memory modes** (683 -> 684).  No engine-facing change: no manifest
entry, no ABI row.

## P8.38 — the polygon clipper, and two bugs a uniform fixture could not see

`polyClip` + `addContacts` (`convexTriangle.cuh:918-1058`) is the per-triangle core of rigid
convex-vs-mesh contact generation — P8.31 corrected the earlier claim that it was GJK/EPA, and
P8.34 established that it cannot be lifted by the P8.24 extract-and-prove-by-opcode pattern
because there the emission is fused with the control flow.  So `pxgSlangPolyClip` is a PORT,
with its own fixture and its own oracle.

**The oracle is an independent transcription**: a scalar C++ simulation with the shuffles as
array reads and the ballots as explicit loops over 32 lanes.  It shares the maths with the
shader but not the shape, which is what lets it catch a mistake in the virtual-warp machinery
rather than only in the geometry.  Agreement is **bitwise**, not within a tolerance — the
`NoContraction` stamp plus matching operand order makes that achievable, and it is the only
comparison that can treat a NaN as equal to the same NaN.

**One warp per group, and this is a hard requirement on every consumer.**  `polyClip` returns
early on `!__any_sync(out1)` and again on `!__any_sync(out0)`.  Those are warp-uniform but not
group-uniform, so with two warps per group one could leave while the other runs into a barrier
— and llvmpipe does not diagnose a divergent barrier, it hangs.  The CUDA kernel packs
`NP_TRIMESH_WARPS_PER_BLOCK = 2`; a consumer here dispatches twice as many groups instead,
which is legal because pairs are independent.

### The two bugs

**A race in the ballot primitive.**  Each reduction ends by reading the groupshared word after
its trailing barrier, and the next one opens by resetting that same word.  With no barrier in
front of the reset, lane 0 can clear the word before another lane has performed its read.
`addContacts` runs two ballots back to back, so the second clobbered the first: every lane read
an exclusive-scan base of 0, two contacts were written on top of each other at the same index —
and because `nbContacts` still counted two, **the count looked right and only the second slot
came back empty**.  The fix is a leading barrier in `pcReduceOr`/`pcReduceAnd`.

**A shuffle that ran off the end of its lane run.**  CUDA specifies that `__shfl_sync` takes a
source lane outside `[0, width-1]` modulo the width, and `polyClip` relies on it: the next-vertex
index `tI + 1 == nbVerts0 ? 0 : tI + 1` is 32 for lane 31 whenever `nbVerts0 < 32`.  In a
groupshared staging array that index is not lane 0 — it is one past the end of the x-component
run, i.e. lane 0's **y**.  Live for exactly one lane, and invisible until the fixture stopped
padding.

### The fixture-hole lesson: uniform padding hides more than one thing at a time

The first fixture filled every lane past `nbVerts0` with vertex 0 — which is also the wrap
target of the next-vertex index.  That single choice hid the shuffle bug *and* two mutants: one
that dropped the wrap entirely (it read the padding and got the right answer) and one that
stopped masking `in0`/`out0` to the three triangle planes (the padded triangle lanes all
collapsed to a zero-length edge, so the extra ballot bits were always zero).  Giving the padding
lanes **distinct** values — which cannot change the true result, since those lanes are masked
out of every path — killed both mutants and exposed the bug.

Sweep **17/24 first, 23/24 after sharpening the fixture**.  The five that needed new coverage:
a NaN vertex (the `!(sep > maxSep)` spelling differs from `sep <= maxSep` only for NaN, and
with finite geometry every path that could make one is already guarded to `PX_MAX_F32`); a
31-gon with a non-zero base so the capacity guard actually bites (5 + 31 candidates for 32
slots); a triangle vertex exactly on a poly0 edge, with another vertex sticking out so the
classification reaches `e1cross` instead of leaving at the first early-out; and the distinct
padding above.

**The one survivor, M24, is an equivalent mutant**: removing the SECOND early-out changes
nothing, because when every `out0` is zero, `e0cross = (in0 & out0p) | (in0p & out0)` is zero
for every lane, so `c` is zero, `a` is zero and the edge-edge loop never runs.  The FIRST
early-out is not like that — M23 dies — which is exactly the asymmetry the CUDA comment at
`:911` describes: the early-outs "are not actually an optimization, but have a functional
purpose in an edge case", and the case is two polygons exactly on top of one another, where
continuing past the first one emits the same points twice.

**Winding, recorded because it cost a debugging round.**  `polyClip` classifies with
`t < 0 -> inside` and builds its two edge-plane normals with the operands SWAPPED —
`cross(e0, axis)` for poly0 and `cross(axis, e1)` for the triangle.  With `axis = +z` that means
poly0 must be wound counter-clockwise and the triangle clockwise.  Getting both backwards does
not fail loudly: every case simply comes out with zero contacts through the edge-edge path
instead of three through the early-out.

An upstream landmine, left alone: with `nbVerts0 == 32` the CUDA's own `(1 << nbVerts0) - 1`
shifts by the full word width, which is undefined in C++ and in SPIR-V alike.  The fixture uses
31 rather than pinning a behaviour whose expected value would be a guess.

Not an engine kernel yet — this is the module plus its scaffolding, and the shader is absent
from the manifest.  `convexTrimeshNarrowphase` is what will import it.

Suite **683/683 in both memory modes** (682 -> 683).  No engine-facing change: no manifest
entry, no ABI row, so the trace is untouched.

## P8.37 — the contact-manager removal chain, all five stages at once

`removeContactManagers_Stage5_CvxTri` was the last REFUSED launch in the engine trace.  It is
**not** a one-kernel tranche.  `removeLostPairsGpuInternal` (`PxgNarrowphaseCore.cpp:8521-8553`)
launches five kernels back to back with the *same* argument pack, and stages 1-4 exist only to
build the run-sum buffer that stage 5 binary-searches.  Stages 1-4 were `nullptr` in the
manifest, so they failed the UNMAPPED way — `fromFunc()` fails and `launchKernel` returns
before `traceDispatch()`, so there is no trace line at all, only the engine's stderr — while
stage 5, being mapped to a bound-buffer Phase-1 shader, showed up as the loud REFUSED.
**Porting stage 5 alone would have made the trace clean and the result wrong**: it would have
binary-searched a run-sum nobody had filled in.  That asymmetry between the two failure modes
is exactly what makes a refused-launch count a bad completion metric on its own.

`npPairMgmtBda.slang` ports all six entry points (the five stages plus `_Stage5_CvxTri`).
ABI row `kPairMgmt {8,1}` — `(const PxgPairManagementData*, bool copyManifold)` — which
`pxgKernelAbiPushSize` rounds to a 16-byte blob.  Manifest 259 → **263** (stages 1-4 newly
mapped; the two stage-5 entries re-pointed off the harness shader).

**Everything else arrives through the struct pointer**, so the tranche stands or falls on
`PxgPairManagementData`'s layout.  The one offset that is not obvious is the last field: the
two `PxU32` at 72/76 do *not* let `mPersistentContactManagers` follow at 84 — it re-aligns to
8, lands at **80**, and the struct is **88** bytes, not 84.  `pairMgmtLayout` pins all twelve
offsets and the size with `offsetof` against the real header, and mutant M20 (reading the
pointer at 84) is killed by the chain test.

**Dimensions**: grid 32 × block 512, and here NUM_BLOCKS genuinely is not free — the engine
allocates the cross-block accumulator as exactly `sizeof(PxU32) * 32`
(`PxgNarrowphaseCore.cpp:8446`), and stages 3 and 4 must agree on the same partition.  The
CUDA blocks are 2D `(WARP_SIZE, blockDim/WARP_SIZE)` and index with
`threadIdx.x + WARP_SIZE*threadIdx.y`, which is just the linear thread id; `threadIdx.x & 15`
in stage 5 is therefore `linear & 15`, and the 1D port says so directly.

**Bit-exact moves, not typed moves.**  The six contact-manager arrays are copied as scalar
words: `PxsContactManagerOutput` is 48 bytes, so its odd elements are not 16-aligned, and the
P8.26 `PxBounds3` lesson applies.  The manifold moves as `uint4` rather than `float4` so a slot
holding a bit pattern that happens to be a signalling NaN comes back byte-identical instead of
canonicalised — the fixture fills half the rest distances with NaN payloads and the manifolds
with `0xCD…` words precisely to make a float-typed copy visible.

The fixture runs the five stages exactly as the engine does and compares against a CPU
transcription of `SparseRemove.cuh` — the run-sum itself plus all seven arrays — over removals
entirely inside the kept region, entirely in the tail (`nbToSwap == 0`, stage 5 must do
nothing), straddling, `N=1, K=1`, `copyManifold` false, and N past one scan block (512) and
past a whole grid's worth (512×32 = 16384) so the cross-block prefix actually carries.

**Sweep 18/20.**  Both survivors are equivalent mutants, and both for the same structural
reason — a cooperative loop whose work happens to fit in one pass:

- **M14, every lane moves the payload instead of lane 0.**  All 16 lanes of a swap share
  `dstIndex`/`srcIndex`, so they write identical bytes to identical addresses.  Distinct swaps
  never alias (dst is a hole in the kept region, src a survivor in the tail), so it is a benign
  duplicate write.
- **M16, the manifold walked with a 32-lane stride in the 16-lane variant.**
  `for (idx = lane; idx < 16; idx += stride)` with `lane < 16` runs exactly once for stride 16
  and stride 32 alike — `sizeof(PxgPersistentContactManifold)/sizeof(float4)` is 16, precisely
  the lane count, so the stride is unobservable.  It is observable in the `_CvxTri` variant
  (76 float4 over 32 lanes), which is why M17 — swapping the manifold size — dies.

Suite **682/682 in both memory modes** (681 → 682: the one new `pairMgmt` ctest entry, which
runs all three fixtures), rigid regression 8×2 = **16/16**.

**The engine trace is now at ZERO refused launches in every scene.**  P8.21's inventory of
"kernel is in the manifest, has valid SPIR-V, has no ABI row, so the shim refuses" is closed:
P8.20 took the radix-sort trio, P8.35 `initializeManifolds`, and this tranche the removal
chain.  What is left in `meshfloor` is the other failure mode entirely — five kernels that are
still `nullptr` in the manifest and so never reach the shim at all: `convexTrimeshCore`,
`sortTriangles`, `convexTrimeshPostProcess`, `convexTrimeshCorrelate` and
`convexTrimesh finishContacts`.  The default scene stays at zero of both.

## P8.36 — the grid/block transposition in P8.35, and why the fixture could not see it

P8.35 shipped `initializeManifolds` as **512 groups × 128 threads**.  The launch is
`launchKernel(fn, GridDims::INITIALIZE_MANIFOLDS, 1, 1, BlockDims::INITIALIZE_MANIFOLDS, 1, 1, …)`
(`PxgNarrowphaseCore.cpp:8193`) — grid first, then block — and those constants are **128** and
**512**.  I had them the other way round.

**Why nothing caught it.**  The shim ignores the launch's block dims entirely (the parameters
are unnamed in `PxgVulkanCudaContext.cpp:293`) and uses the workgroup size declared in the
SPIR-V, so the engine accepted the wrong shape silently.  And the kernel's output is
decomposition-independent — every record is written exactly once with the same value whatever
the block size — so all ten fixture cases passed at 128 threads, and would pass at any block
size that is a multiple of `nbThreadsPerElement`.

**But the block size is not a performance knob here.**  `nbPerBlock = blockDim.x /
nbThreadsPerElement` is an integer division that the very next line *divides by*:

    nbBlocksRequired = (nbTimesToReplicate + nbPerBlock - 1) / nbPerBlock

so the block must hold at least one whole record, i.e. `blockDim.x >= dataSize/16`.
`MaxStructureSize` is 4096 bytes → `nbThreadsPerElement` up to 256, which 512 covers and 128
does not.  A 128-thread block divides by zero for any record above 2048 bytes.

The two manifolds in the tree (`PxgPersistentContactManifold` 256 B,
`PxgPersistentContactMultiManifold` 1216 B) are both well under that, so the cliff was latent
rather than live — which is exactly why the fixture had to be extended to reach it.  A
`dataSize = 4096` case was added, and reverting the block size to 128 with it in place does
not merely fail, it **hangs** llvmpipe (confirmed: the whole ctest run was terminated at 90 s).
That case is now mutant **M13**, so the transposition cannot come back unnoticed.

Sweep re-run at the corrected dims: **10/13 killed**, the same three survivors as P8.35 and for
the same reasons (M5 and M7 output-equivalent, M11 unkillable on llvmpipe).  M6 flipped from
`KILLED(hang)` to `KILLED(rc=8)` — with 512 threads its bogus `nbThreadsPerElement` still lands
inside the guards, so it now produces wrong bytes instead of running away.  Suite 681/681 in
both memory modes.

**The general lesson**: a kernel whose *output* is decomposition-independent can still have a
launch geometry that is load-bearing, and no output comparison will tell you.  When a port
hard-codes a dimension the CUDA reads from `blockDim`/`gridDim`, the constant has to be checked
against the call site, not against the test.

## P8.35 — `initializeManifolds` on the engine ABI, and a mutation harness that ate its own shader

`initializeManifolds` (`pairManagement.cu:150`) was one of the two REFUSED launches P8.21's
inventory left in *every* scene.  Both were the P8.20 pattern: in the manifest, valid SPIR-V,
but pointing at a Phase-1 harness shader with bound buffers, so no ABI row is possible and
`pushConstantSizeFor()` returns 0 and the shim refuses.  `initManifoldsBda.slang` closes the
first of the two; only `removeContactManagers_Stage5_CvxTri` is left.

**What the kernel does** is replicate one `dataSize`-byte record `nbTimesToReplicate` times —
the engine stamps an empty manifold across a freshly grown manifold array
(`PxgNarrowphaseCore.cpp:8187`).  ABI row `kInitManifolds {8,8,4,4}` → 24-byte blob; grid
128 × block 512 (`PxgNarrowPhaseGridDims/BlockDims::INITIALIZE_MANIFOLDS`).
*(This tranche first landed with those two transposed — see P8.36, which is the
fix and the reason it was not merely cosmetic.)*

**The shared-memory transpose is the whole kernel, and it is easy to "simplify" away.**  The
source is not copied verbatim.  Lanes `0..nbThreadsPerElement-1` each read ONE source `float4`
and scatter its components into four *separate runs* of the groupshared array — all the x's,
then all the y's, then the z's, then the w's — and every destination lane gathers its element
back out with `threadReadIdx = tid % nbThreadsPerElement`.  A port written as
`dst[i] = src[i % n]` produces the same bytes only because scatter and gather are inverse;
break either half and the record comes out *transposed*, which does not look corrupt.  It is
transcribed literally so the pairing stays visible.

`MaxStructureSize` is 4096 **bytes**, so the groupshared array is 1024 floats and the scatter
writes `4*nbThreadsPerElement = dataSize/4` of them — the bound holds exactly when
`dataSize <= 4096`.  Upstream does not check it; neither does this.

`testInitManifolds.cpp`: 10 cases over dataSize 64/128/1216/2048 (1216 = a real
`PxgPersistentContactMultiManifold`, and the one size that does NOT divide 128), each with a
sentinel record past the end that must stay untouched.

**Mutation sweep 9/12 killed.  All three survivors are explained, and two of them are
genuinely equivalent mutants** — worth recording because "survivor ⇒ fixture hole" has been
the right reflex in every previous tranche and here it is wrong:

- **M5, the `maxThreadIdx` guard dropped — EQUIVALENT.**  For any `tid`, the write target is
  `Ptr<float4>(startPtr)[tid]` and the value is record float4 `tid % n`.  Since
  `tid = (tid/n)*n + (tid%n)`, index `tid` is *exactly* the correct slot for element
  `startBlockIdx + tid/n`, component `tid%n`.  So lanes past `maxThreadIdx` write the correct
  value to the correct address of an element another block also writes — a benign duplicate,
  not an overrun, because the `idx < nbTimesToReplicate` guard (M4, killed) still bounds it.
- **M7, the block-stride loop steps by 1 — EQUIVALENT.**  Every workgroup then does every
  block's work.  Same values to the same addresses; only the cost changes.
- **M11, the group barrier removed — not killable on llvmpipe.**  Lane `tid` reads slot
  `tid % n`, which was written by lane `tid % n ≤ tid`.  The dependency only ever points
  *backwards*, and llvmpipe runs a workgroup as an in-order sweep over 8-lane subgroups, so
  the producer has always already run.  The barrier stays: on hardware with genuinely
  concurrent subgroups the race is real.  This is a platform blind spot of the harness, not a
  fixture hole — no fixture can kill it here.

### The harness lesson: a `finally` is not enough

The sweep is run under `timeout`.  `timeout` sends **SIGTERM**, which kills the process
*before* the `finally` clause restores the shader — so a killed run leaves the tree MUTATED,
and successive killed runs **STACK** mutations.  Two timeouts left three mutations layered in
`initManifoldsBda.slang` at once (M1's scatter, M6's `/4u`, M2's swapped gather); the next run
then reported `SKIPPED(anchor ×0)` because the text it wanted to patch was already gone.
The `SKIPPED` was the only reason this was caught — a harness that silently tolerates a
missing anchor would have reported a clean sweep over a corrupt shader.

Sweep scripts now take a `shutil.copyfile` backup up front and restore it from `atexit` **and**
from explicit `SIGTERM`/`SIGINT` handlers.  The per-mutant ctest call also carries its own
`timeout=90` and counts a hang as a kill — the reference does not hang, so a mutant that does
is distinguished from the reference just as surely as one that returns wrong bytes (M6 was
caught exactly this way).

Forced-fresh `.spv`, suite **681/681 in both memory modes** (680 → 681).  Engine trace:
`initializeManifolds` no longer appears as REFUSED in any scene; the default scene is at
**zero** refused and zero failed launches, and `meshfloor` is down to one refused
(`removeContactManagers_Stage5_CvxTri`) plus the five still-unmapped convex-trimesh kernels
(`convexTrimeshCore`, `sortTriangles`, `convexTrimeshPostProcess`, `convexTrimeshCorrelate`,
`convexTrimesh finishContacts`).

## P8.34 scouting — the clipper is NOT extractable by the P8.24 pattern, and why

P8.33 named `npConvexPolyClip` as the right Phase-1 shader and predicted the
tranche shape: "extract into a module, prove the extraction inert by opcode
sequence, rather than port from the `.cu`".  **That prediction does not hold**,
and the reason is structural rather than a matter of effort.

### The measurement

`npConvexPolyClip.slang` has no separable clipper function.  The algorithm lives
inline in the entry point `ctPolyClip` (:137-291), and its contact EMISSION is
fused with its control flow at four points:

```
  :206   addContacts(wp, band & 7u, maxSep);     // phase 2 — poly1 in poly0
  :210   if (!anyOut1) { gCount[t] = wp.contactCount; return; }     // EARLY EXIT
  :250   addContacts(wp, flag2, maxSep);         // phase 3 — poly0 in poly1
  :254   if (!anyOut0) { gCount[t] = wp.contactCount; return; }     // EARLY EXIT
  :286   addContacts(wp, flag3, maxSep);         // phase 4 — edge-edge crossings
  :290   gCount[t] = wp.contactCount;
```

and `addContacts` itself (:114) writes straight into the bound
`RWStructuredBuffer gContacts` (:128).

### Why that forecloses the proof, not just complicates it

P8.24 and P8.25 could prove their extractions inert because what moved was PURE:
`tbIntersectTriangleBox` and the BV32 accessors take values and return values, so
the same source compiles to the same instruction stream on either side of the
move, and a disassembly diff settles it.

This clipper's output path is its IO.  The harness emits to a bound descriptor at
three interleaved points; the BDA consumer must emit to a device address derived
from `midphaseAllocate` (P8.29).  Those are different instructions by
construction, so **no refactor of this code can be opcode-identical**, and a
"proof" that reported identity would be measuring the wrong thing.

Any refactor that decoupled the emission — accumulating into the `Warp` struct and
flushing once at the end — would also change WHEN the `NUM_TMP` clamp bites
relative to the two early exits, which is observable whenever a clip produces more
than 32 contacts.

### What this changes for the next tranche

The per-triangle core is a **PORT with its own fixture**, not an extraction with a
proof.  Concretely that means: transcribe `polyClip` + `addContacts`
(`convexTriangle.cuh:918-1058`) onto the BDA emission path, and certify it the way
P8.27 certified the midphase — a CPU transcription, a golden-vector fixture and a
mutation sweep — rather than the way P8.24 certified the SAT.

Budget accordingly: P8.27's sweep needed four rounds and six of its first seven
survivors were holes in the fixture, on a kernel with a far simpler output path
than this one.

### Verification

Measurement only — no shader, manifest, ABI or scene changed.  Suite unchanged at
680/680.

## P8.33 scouting — the polygon clipper: the right shader is NOT the one named after it

The last piece before the core kernel is the polygon clipper.  There are **three**
Phase-1 shaders with clipper-shaped names, and the obvious one is the wrong one —
worth a note, because P8.31 already showed what a wrong pointer costs.

| shader | what it actually is |
|---|---|
| `npCvxPolyClip` | the **convex-vs-CONVEX** clipper from `cudaGJKEPA.cu`, used by `generateConvexContacts`.  Its own header says "distinct from the convex-vs-triangle clippers".  **Not this chain's.** |
| `npPolyClip4` | `polyClip4` from `femClothPrimitives.cu` — the **CLOTH** convex-vs-triangle clipper.  Also not this chain's. |
| **`npConvexPolyClip`** | "the RIGID convex-polygon vs triangle clipper `polyClip` (`convexTriangle.cuh:943-1058`) plus `addContacts` (`:918-940`)".  **This is the one.** |

And the stage after it is already transcribed too: `npCvxTriEmit` is
"the contact-reduce-and-emit stage of `convexTriangleContactGen`
(`convexTriangle.cuh:1769-1821`)".

So both halves of the per-triangle core exist in harness form, and the expected
tranche shape is the one P8.24 and P8.25 established — extract into a module,
prove the extraction inert by opcode sequence, rather than port from the `.cu`.

### The chain, in full

Certified so far:

| piece | tranche | result |
|---|---|---|
| PCM manifold gate | P8.23 | 21/21 mutants, six-dispatch fixture |
| triangle-box SAT | P8.24 | opcode-identical across extraction |
| BV32 reader | P8.25 | opcode-identical across five consumers |
| the midphase kernel | P8.27 | 15/16, wired, manifest 259 |
| stack carve-up | P8.29 | 12/12, exhaustive over the padding boundary |
| second-pass classifier | P8.30 | 12/12 |
| deferral predicate | P8.32 | 12/12 |

Remaining: extract `polyClip` + `addContacts` and the emit stage; write
`convexTrimeshNarrowphase` (21 args, 168-byte blob, verified to fit in P8.28);
then `sortTriangleIndices`, `convexTrimeshPostProcess`, `convexTrimeshCorrelate`
and `convexTrimeshFinishContacts`.  `PXG_SCENE=meshfloor` stays red until the box
stops falling through the floor at y = −73.8.

### Verification

Measurement only — no shader, manifest, ABI or scene changed.  Suite unchanged at
680/680.

## P8.32 — `pxgSlangDeferContact`: the predicate that decides whether to defer at all

Step 2 of P8.31's corrected decomposition, and the counterpart of the classifier
P8.30 certified.  P8.30 answers "WHICH feature did the contact land on"; this
answers the prior question, "should this pair be deferred to the second pass AT
ALL".  Seventeen lines of CUDA (`convexTriangle.cuh:1402`) with four details that
only literal transcription keeps.

### The four details

1. **The guard is an AND of two inequalities against NON-ADJACENT sentinels.**
   `TRI_FEATURE_IDX` is `0x7FFFFFFF` and `EDGE_FEATURE_IDX` is `0x80000000`
   (`triangle.cuh:32/33`).  As unsigned values they are adjacent, but they differ
   in the top bit, so no range check can stand in for the pair.  When the
   separating axis IS the triangle face or an edge cross, the pair is never
   deferred — only a CONVEX HULL FACE normal can defer.
2. **The triangle normal is negated first.**  `cosTheta` is measured against
   MINUS the stored triangle local normal.  Dropping the negation flips the
   predicate on every pair whose axes are within 45° of opposed, which is most of
   them.
3. **The comparison is strictly less.**  A pair exactly at the 45° threshold is
   NOT deferred.
4. **The threshold is the literal `0.707106781f`** — not a computed `sqrt(0.5)`,
   and not the `0.985f` sitting commented out one line below it in upstream.  That
   commented value is a tighter 10° bar someone tried; reinstating it would defer
   far more pairs and change the contact set on every mesh.

### 12 mutants, 12 killed — after a sweep-harness bug, not a shader bug

The first run reported 7/12 with **five SKIPPED**, and the cause was in the sweep,
not the code: the module header quotes the guard verbatim, so the anchor matched
twice — once in the comment and once in the code — and the harness refuses an
ambiguous anchor rather than guessing.  The same class of miss P8.17 hit with an
indentation mismatch.  Anchoring on the guard plus the line that follows it (which
the comment renders differently) disambiguates, and all five then die.

That the harness REFUSES rather than picking one is what made this recoverable.
A sweep that silently patched the first match would have mutated a comment,
reported SURVIVED, and left five real gaps looking like equivalences.

Killed: the missing negation, `<` becoming `<=` and being inverted, the 10°
threshold, each sentinel checked alone, the AND becoming an OR, the guard dropped,
a range check standing in for the pair, each sentinel off by one, and the default
return flipped.

### Verification

Forced-fresh `.spv`, **680/680 in both memory modes** (679 → 680).  No engine
kernel changed.

### Where the chain stands

Certified: PCM gate (P8.23, 21/21), triangle-box SAT (P8.24, opcode-identical),
BV32 reader (P8.25, opcode-identical over five consumers), the midphase kernel
itself (P8.27, 15/16, wired, manifest 259), the stack carve-up (P8.29, 12/12), the
second-pass classifier (P8.30, 12/12), and now the deferral predicate (12/12).

Remaining before the core kernel: the polygon clipper (`polyClip` plus
`selectPolygonFeatureIndex`), then the kernel wiring all of it, then the four
kernels after it.  `PXG_SCENE=meshfloor` stays red until those land.

## P8.31 scouting — a correction: the per-triangle core is POLYGON CLIPPING, not GJK/EPA

P8.28's scouting note described `convexTrimeshNarrowphase` as "running a GJK/EPA
query PER TRIANGLE" and made that the first item of its recommended decomposition.
**That is wrong**, and it is worth correcting before a tranche is spent on it.

Reading what `convexTriangleContactGen` (`convexTriangle.cuh:1640`) actually calls:

```
  selectPolygonFeatureIndex        polyClip
  getNbPolygons / getNbVerts       delayGenerateContact
  vertex2Shape                     PxPlane
```

No `gjk*`, no `epa*`, anywhere in the body.  The per-triangle algorithm is
SAT feature selection followed by **polygon clipping** — the same family as the
convex-vs-convex path's `npCvxPolyClip` / `npCvxTriEmit` / `npCvxTriFind` /
`npCvxTriPurge`, all of which Phase 1 already transcribed in harness form.

`pxgSlangGjk` and `pxgSlangEpa` do exist as shared modules — P8.6 extracted them
for the softbody-vs-rigid path — but they are **not** what this chain needs, and
starting from them would have produced a module with no consumer.

### The corrected decomposition

What is already certified for this chain, from the tranches so far:

| piece | tranche | result |
|---|---|---|
| PCM manifold gate | P8.23 | 21/21 mutants, six-dispatch fixture |
| triangle-box SAT | P8.24 | opcode-identical across extraction |
| BV32 reader | P8.25 | opcode-identical across five consumers |
| the midphase kernel | P8.27 | 15/16, wired, manifest 259 |
| stack carve-up | P8.29 | 12/12, exhaustive over the padding boundary |
| second-pass classifier | P8.30 | 12/12 |

What remains before the core kernel:

1. **The polygon clipper** — `polyClip` plus `selectPolygonFeatureIndex`, the real
   per-triangle core.  Phase 1 has it in harness form, so the likely shape is
   another extraction with an opcode-sequence proof (P8.24 / P8.25), not a port.
2. **`delayGenerateContact`** — the predicate that decides whether a contact is
   emitted now or deferred to the second pass.  It is the counterpart of the
   classifier P8.30 certified: P8.30 answers "which feature", this answers
   "defer at all".
3. The core kernel itself, wiring all of the above.

### Verification

Measurement only — no shader, manifest, ABI or scene changed.  Suite unchanged at
679/679.

## P8.30 — `pxgSlangTriFeature`: the second pass's classifier, and what its ORDER hides

Step 3 of P8.28's decomposition — the part of the convex-trimesh chain with no
precedent anywhere else in this port.

**What the second pass is.**  `convexTriangleContactGen` does not always emit its
contacts.  When `delayContacts` is set it appends the pair to a deferral list
(`convexTriangle.cuh:1849`):

```cpp
convTriSecondPairPass[atomicAdd(nbSecondPassPairs, 1)] = globalWarpIndex | mask;
```

`mask` records WHICH FEATURE of the triangle the contact landed on.  A contact on
a shared edge or vertex cannot be accepted or rejected from one triangle — the
neighbour decides — so the mask is the question the later pass answers.  It
occupies the top three bits (`n << 29`) and shares the word with the warp index,
which is why the enum starts at 1: zero means "interior hit, no feature".

### Two things the classifier's ORDER hides

The classifier (`convexTriangle.cuh:1562`) is one threshold, 0.97, and a strict
`>`, tested in a fixed sequence.  Both of the following are invisible in a
paraphrase and both are load-bearing:

- **The vertex tests come first and WIN.**  At `v = 0.98` both `v > 0.97` and
  `(v + u) > 0.97` hold — a point that close to a vertex is also close to both
  edges meeting there.  The order is what makes it a VERTEX.  Reordering silently
  reclassifies every near-vertex contact as an edge one, and the second pass then
  asks the wrong neighbour.
- **The pairings are not the obvious ones.**  Edge 0-1 is `v + u`, edge 0-2 is
  `u + w`, edge 1-2 is `w + v`.  With `v` the weight of vertex 1 and `w` of vertex
  2 each is the sum of the two vertices the edge JOINS — correct, but the naming
  invites `v + w` for edge 0-1, which is the OPPOSITE edge.

### 12 mutants, 12 killed — after one fixture gap

The sweep exercises exactly those two properties: reordering the vertex tests
after the edge tests, and re-pairing E01 as `v + w`, both die.  So do the
threshold, `>` becoming `>=`, swapped vertex codes, `u` computed as `v + w - 1`,
a shift of 28 instead of 29, and three ways of getting `barycentricCoordinates`
wrong.

One survivor needed a fixture fix, and it was the case a dense sweep structurally
cannot reach: **replacing the exact `totalArea == 0` pin with a plain divide.**
Upstream substitutes a ZERO reciprocal for a degenerate triangle, so `v = w = 0`
and the classifier reports V0; a plain divide yields NaN, every comparison is
false, and the answer silently becomes NONE.  No non-degenerate triangle can tell
the two apart, and the fixture had only one triangle — a good one.  Three
degenerate triangles (collinear, duplicated, coincident) now sit at the end of the
sweep and the mutant dies.

The sweep itself covers `(v, w)` on a 0.01 grid from **−0.08 to 1.08** — outside
the triangle as well as inside, because the classifier clamps nothing and a port
that added a clamp would look identical on the inside — plus ten named boundary
cases, and it asserts that **all seven outcomes actually occur**, so the sweep
cannot quietly prove less than it appears to.

### Verification

Forced-fresh `.spv`, **679/679 in both memory modes** (678 → 679).  No engine
kernel changed.

## P8.29 — `pxgSlangMidphaseAlloc`: the stack carve-up, certified exhaustively

Step 2 of the decomposition P8.28 recommended.  `midphaseAllocate`
(`midphaseAllocate.cuh:129`) is not a private detail of one kernel: the convex-
trimesh chain receives ONE flat collision stack, `convexTrimeshNarrowphase` carves
seven intermediate buffers out of it at runtime, and the five kernels after it
re-derive the same layout.  It is a contract, so it is extracted and certified on
its own — the same reason P8.23 took the PCM gate first.

`pxgSlangMidphaseAlloc.slang` owns the arithmetic; `midphaseAllocBda.slang` is a
dispatch wrapper so it can be tested before its first real consumer exists (the
scaffolding pattern `pcmManifoldBda` established).  Neither is in the manifest.

### The two things a paraphrase would lose

**The allocation order is NOT the argument order.**  The signature lists NI,
Contacts, MaxDepth, Interm, Ordered, SecondPass, Pairs.  The body allocates
Pairs, NI, Contacts, **Ordered**, MaxDepth, Interm, SecondPass — `ordered` moves
from fifth in the signature to fourth in memory, pushing `maxDepth` and `interm`
down behind it.  Following the signature puts three of the seven buffers at wrong
addresses, and every one of those addresses is still INSIDE the stack allocation,
so the corruption reads as bad contacts rather than as a fault.

**`ordered` is sized by the OTHER counter.**  Six of the seven use
`numPairsAfterMP = (unpadded + 3) & ~3`.  The ordered array alone uses
`numPaddedPairsAfterMP` — the separately-accumulated atomic the midphase bumps
with `((nbPairsPerCM + 3) & ~3) * 2`, already padded AND doubled because the radix
sort needs its temp space inline.  That is the whole reason
`midphasePairsNumPadded` exists as a second counter, a question P8.27 raised and
this answers.

### 12 mutants, 12 killed

Pure integer arithmetic is cheap to test exhaustively, so the fixture is
exhaustive where it matters: every unpadded count from **0 to 64** — the
`(n+3)&~3` rounding has a four-step period, so that covers every residue many
times and pins both boundaries (0 stays 0; 1..4 all round to 4) — plus ten
adversarial pairs including a padded counter SMALLER than the rounded unpadded
one and one far larger, which is what a port that "simplified away" the second
counter would produce.

The sweep needed no second round: `+3` changed to `+4` (off by one only at exact
multiples of four), the padding rounding down, `ordered` sized by the wrong
counter, `ordered` moved to its signature position, each of the four element
sizes, two pairs of buffers swapped, and a base multiplied instead of offset —
all twelve die.

### Verification

Forced-fresh `.spv`, **678/678 in both memory modes** (677 → 678: the new
`gpuslang.midphaseAlloc` suite).  No engine kernel changed, so no scene number can
move.

## P8.28 scouting — `convexTrimeshNarrowphase`: the blocker question, answered first

The next kernel in the chain is `convexTrimeshNarrowphase` (`convexMesh.cu:623`),
and it is a different order of magnitude from anything ported so far: ~550 lines
of warp code running a GJK/EPA query PER TRIANGLE, with **21 push-constant
arguments**.  This is measurement, not code — the facts the port needs before a
line is written, and the one that could have stopped it dead.

### The blocker: does a 21-argument pack fit in a push constant?

llvmpipe reports `maxPushConstantsSize = 256`.  Laid out at CUDA's natural
alignment — the same `paramAlign` rule the shim uses — the pack is:

```
  15 pointers            @0 .. @120
  stackSizeBytes  u32    @120
  ( 4 bytes of padding, because the next argument is a pointer )
  gpuTempContactStack    @128
  gpuTempContactIndex    @136
  mMaxConvexMeshTempMemory @144
  gpuStackShift          @152
  numTests        u32    @160
  -> 164 data bytes, rounded to the 8-alignment = 168-byte blob
```

**168 < 256: it fits**, with room to spare.  Worth establishing before writing
500 lines rather than after — a pack that overflowed would have forced a
descriptor-buffer redesign of the whole engine ABI for this one kernel.

### The structural surprise: seven arguments are DOUBLE pointers

`cvxTriNIPtr`, `cvxTriContactsPtr`, `cvxTriMaxDepthPtr`, `cvxTriIntermPtr`,
`orderedCvxTriIntermPtr`, `cvxTriSecondPassPairsPtr` and `pairsGPU` are declared
`T**`.  The kernel does not RECEIVE the sub-buffer addresses — it **computes them
itself**, by calling `midphaseAllocate` (`convexMesh.cu:670`) on one flat
`stackPtr` and then publishing the results back through those double pointers
(`:681`).  So the stack layout that P8.27's midphase wrote pairs into is carved
up here, at runtime, by the consumer.

For the BDA port that means each of the seven is a `Ptr<uint64_t>` — a device
address holding a device address — and `midphaseAllocate`'s arithmetic has to be
transcribed exactly, because the six downstream kernels index the same
sub-buffers by the same rule.  It also explains a detail P8.27 already had to get
right: `midphasePairsNumPadded` exists solely so this allocation can reserve the
radix sort's temp space.

### What Phase 1 already covers

The per-triangle geometry is largely transcribed already, in harness form:
`npConvexTri` (2 entry points), `npCvxTriEmit`, `npCvxTriFind`, `npCvxTriPurge`,
`npCvxTriWorldNormal`, `npCvxTriNpSetup`, plus the `npCvxCoreGjk` / `npCvxCoreEpa`
drivers.  The same situation P8.24 found for the SAT — so the tranche shape is
likely to repeat: extract the shared pieces into modules, prove each extraction
inert by opcode sequence, then write a thin BDA kernel that wires them.

### Recommended decomposition, and why

Not one tranche.  The three preceding tranches (P8.23 gate, P8.24 SAT, P8.25 BV32)
each certified one piece on its own, and P8.27 then wired them into a kernel whose
sweep killed 15 of 16 — with six of the first seven survivors turning out to be
holes in the FIXTURE, not the shader.  That ratio is the argument: for a kernel
this size, a single end-to-end fixture would have been the thing that was wrong,
and the failure would have looked like a passing test.

Order: (1) extract the GJK/EPA-per-triangle driver and certify it standalone;
(2) transcribe `midphaseAllocate` and certify the layout arithmetic against a CPU
model — it is pure integer arithmetic and cheap to test exhaustively;
(3) the second-pass mechanism (`nbSecondPassPairs`, `cvxTriSecondPassPairsPtr`),
which is the part with no precedent in this port; (4) the kernel itself.

### Verification

Measurement only — no shader, manifest, ABI or scene file changed in this commit.
Suite unchanged at 677/677.

## P8.27 — the convex-trimesh midphase is CERTIFIED and WIRED

P8.26 wrote the kernel and deliberately left it unmapped, because a wired-and-wrong
kernel fails as silently as an unwired one.  This tranche supplies what was
missing — three fixtures and a mutation sweep — and then wires it.
`PXG_SLANG_KERNEL_MAPPED` 258 → **259**; ABI row `kCvxTriMidphase`
`{4,4,8,8,8,8,8,8,8,8,8,8,8,8,4}`, a 112-byte blob.

Measured on `PXG_SCENE=meshfloor`: the chain's failed launches drop from **six to
four**.  `convexTrimeshMidphase` no longer appears; `convexTrimeshCore`,
`sortTriangles`, `convexTrimeshPostProcess` and `convexTrimeshCorrelate` remain.
The box still falls through the floor, and will until those land — this tranche
does not claim otherwise.

### 16 mutants, 15 killed — and six of the first seven survivors were the FIXTURE

The first sweep killed 9 of 16.  Every one of the seven survivors was diagnosed
rather than excused, and six turned out to be holes in the test:

- **the grid was too big for a leaf.**  The scene had 34 triangles in ONE BV32
  leaf, and a leaf is walked with one lane per primitive — so the two back-facing
  triangles at indices 32 and 33 were never *visited*, let alone culled.  A mutant
  that DELETED THE FRONT-FACE GATE survived on that alone.  Fixed by shrinking the
  grid to 18 triangles.
- **every vertex sat on an integer.**  The 0.04 contact distance never decided
  anything, so dropping it from the query box was unobservable.  Fixed with a
  triangle 0.02 above the box face — inside the 0.54 half-extent the contact
  distance produces, outside the 0.50 the box alone gives.
- **one leaf with a [-1000, 1000] box.**  A node box that CONTAINS the query box
  makes the overlap test agree with a swapped-operand version of itself, and with
  a single leaf the atomic base offset is always zero.  Two mutants — swapped
  `vMin`/`vMax`, and writing pairs at the bare lane offset — were both
  unobservable.  Fixed with THREE leaf children boxed tightly around their own
  primitive ranges, all smaller than the query box.
- **`deltaP` was 100.**  Any margin invalidates at that distance, so reading the
  hull extents from the CoM offset changed nothing.  Fixed by moving the cached
  case to `deltaP = 0.005`, which sits under the real `0.01` threshold and over
  the `0` a wrong-offset read produces.
- **one contact manager, already in the right order.**  The mesh/convex swap never
  fired.  Fixed with a second manager describing the same pair reversed; both must
  emit the identical set.

That last change immediately found a **bug in the fixture's own scaffolding**:
manager 1 invalidated while manager 0 stayed cached, because the manifold buffer
held one 1216-byte record and the kernel indexes it by `cmIdx`.  A single-record
buffer sends the second manager reading past the end.

After the six fixes, **15 of 16 die**.  The survivor, `nbPrimitives` not clamped
to 32, is genuinely equivalent: under `[numthreads(32,1,1)]` with a
`lane < nbPrimitives` guard, 34 and 32 activate exactly the same lanes.  The clamp
is defensive documentation, not a behaviour change, and this entry says so rather
than counting it as a kill.

Mutants that DID die on the first sweep are worth naming too, because they are the
ones the port could plausibly have got wrong: the `stackOffset`/`midphasePairsNum`
crossing (the drafting bug P8.26 caught by writing its comment), the gate ignored
(`fullContactGen = true`), the cached sentinel never written, the record carrying
the mesh transform instead of `aToB`, and `nbPairsPerCM` never accumulating.

### Verification

**677/677 in both memory modes** (676 → 677: the new `gpuslang.convexTrimeshMidphase`
suite).  Rigid regression green — box, stack, plane, sphere, convex, joint,
capsule, capsulebox, particle.  Every deformable scene green, and
`PXG_SCENE=softbody` on the MESH floor still at `|dy|` **0.000769**, bit-identical
to P8.20 and P8.25.

## P8.26 — `midphaseGeneratePairs` written: the kernel, and why it is NOT wired yet

The convex-vs-trimesh midphase is transcribed onto the engine ABI as
`convexTrimeshMidphaseBda.slang`.  It compiles, is contraction-stamped (241 ops),
and is built by CMake — and it is **deliberately absent from the manifest and
from the ABI table.**

### Why it is not enabled

P8.20's finding cuts both ways.  A kernel with no ABI row is REFUSED, silently
but harmlessly — the engine logs and steps on.  A kernel that is wired and WRONG
does not fail loudly either: it writes into `gpuIntermStack` and bumps two atomic
counters that five downstream kernels will eventually read.  Until this one has
a fixture, "mapped" is strictly worse than "unmapped", so it stays unmapped.  It
is step 1 of 6, banked and reviewable, not certified.

### What the kernel is made of

Almost nothing in it is new, which was the point of the previous three tranches:

| piece | tranche | certification |
|---|---|---|
| the PCM gate | P8.23 | 21/21 mutants, six-dispatch fixture |
| `Gu::intersectTriangleBox_RefImpl<true>` | P8.24 | opcode-sequence identical across extraction |
| the `BV32DataPacked` reader | P8.25 | opcode-sequence identical across five consumers |

What is genuinely new is the wiring, the query-box construction, the pair
emission with its overflow clamp, and the `ConvexMeshPair` record.

### One bug found by writing the comment rather than the code

The first draft bumped `midphasePairsNum` inside the traversal loop.  That is
wrong, and the mistake is easy to make because both counters are `PxU32*`
arguments sitting next to each other in the pack.  `MidphaseTreeTraverser` is
constructed with **`stackOffset`** as its counter (`convexMeshMidphase.cu:349`,
fifth constructor argument) and bumps it per LEAF; `midphasePairsNum` is a
different counter, bumped **once at the end** with this manager's total
(`:353`).  Crossing them writes pairs at the wrong slot AND hands the core kernel
a `startIndex` pointing into another manager's range — a corruption that would
have produced plausible contacts on a one-manager scene and garbage on any scene
with two.  Every current `meshfloor`-style scene has exactly one contact manager,
so no scene in this harness would have caught it.

### Decisions recorded in the shader, each with its reason

- **Stack depth 320, not 192.**  The deformable midphases use 192 because
  `femMidphaseScratch` declares `sBv32Nodes[192]`; the convex path has a separate
  `MidphaseScratch` declaring `sBv32Nodes[320]`.  Copying the deformable constant
  would corrupt traversal on a deep mesh.  (P8.22 raised this against the shipped
  deformable ports; checking it is what established the two-struct fact.)
- **`nbPrimitives` clamped to 32, bug for bug** — six-bit field, one lane per
  primitive, CUDA silently drops the rest.
- **`PxBounds3` (24 B) and `ConvexMeshPair` (56 B) read and written as scalars**,
  because both land 8-mod-16 on odd indices.  Upstream writes the pair record as
  a 14 x 4-byte scatter for exactly this reason.
- **`cmOutputs` is carried and never touched** — argument 10 of 15 exists in the
  CUDA signature and the blob must match byte for byte, but `midphaseCore` does
  not read or write it.

### Verification

**676/676 in both memory modes** — which for this tranche only establishes that
adding an unreferenced shader broke nothing.  The kernel itself has NO fixture
yet and NO mutation sweep, and this entry should not be read as saying otherwise.
Next: the fixture, then the ABI row and the manifest entry, then the remaining
five kernels of the chain.

## P8.25 — `pxgSlangBv32`: four copies of the BV32 reader become one

P8.24 extracted the triangle-box SAT because the convex-vs-trimesh midphase was
about to become its second consumer.  The same audit found a worse case one level
down: the `Gu::BV32DataPacked` reader — five constants and five accessors —
existed as **four byte-identical private copies**, in `pxgSlangClothMidphase`,
`softBodyMeshMidphaseBda`, `femClothMeshMidphaseBda` and
`softBodySoftBodyMidphaseBda`.  The convex midphase would have made it five.

`pxgSlangBv32.slang` now owns them.  `pxgSlangClothMidphase` re-exports the module
(`__exported import`) so its own consumers — `femClothSelfCollisionBda` and
`femClothDifferentClothBda` — keep reaching the names unchanged; the three BDA
midphases import it plainly.

### Proven inert across every consumer, not just the ones edited

The technique P8.24 established, applied to all five shaders that end up
containing this code — including the two that only see it through the cloth
module's re-export.  Disassembled before and after, the **full opcode sequence is
identical, position for position**, in every one:

| shader | instructions |
|---|---|
| `softBodyMeshMidphaseBda` | 2032 |
| `femClothMeshMidphaseBda` | 1944 |
| `softBodySoftBodyMidphaseBda` | 2216 |
| `femClothSelfCollisionBda` | 2538 |
| `femClothDifferentClothBda` | 2610 |

The two cloth shaders matter most here: nothing in them was edited, so an
identical sequence is what proves the `__exported import` re-export is
transparent rather than merely compiling.

### One thing the module records that no copy did

`mData`'s `nbPrimitives` field is **six bits wide — it can express 63** — while
`bv32Traversal.cuh` maps one lane per primitive and therefore silently drops
anything past 32.  A consumer that writes the obvious
`for (base = 0; base < nbPrimitives; base += LANES)` emits triangles CUDA never
emitted.  The clamp has to stay in the consumer, because it is a property of the
traversal's lane mapping rather than of the node format — but the fact now lives
next to the decoder instead of in one reviewer's head.

### Verification

Forced-fresh `.spv`, **676/676 in both memory modes** (unchanged: an extraction
adds no fixtures).  Scenes green: `clothselfcollide`, `clothcloth`,
`softbodystack`, `softbodycloth`, and `PXG_SCENE=softbody` on the MESH floor at
`|dy|` **0.000769** — bit-identical to the number P8.20 recorded, which is the
scene that actually walks a BV32 tree.

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
