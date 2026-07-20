# Phase 2 runtime status

Progress tracker for the Vulkan production runtime (RUNTIME_PLAN.md P0–P7).
Phase 1 (bit-exact CUDA→Slang math port, 673 kernels) is closed; see
MIGRATION_STATUS.md.  Every item below is committed on
`claude/physx-cuda-slang-migration-mqbz7u` and verified against the standalone
gpuslang harness on llvmpipe (Vulkan 1.4) in BOTH memory modes — default
(host-visible) and `PXG_DEVICE_V2=1` (device-local + staged + timeline).

Test count: **696/696** green in both modes.

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

## P5 Step 5 — MILESTONE: five common rigid scenes rest on the GPU, zero refused launches

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

- **T0** (throwaway smoke test): inject one fabricated manifold contact +
  `bIntersection=true` + clear the invalidate bit → prove the whole downstream
  (finishContacts patch atomics → solver → response) turns a manifold into a
  resting capsule.  Isolates the gap to purely collide() content.
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
