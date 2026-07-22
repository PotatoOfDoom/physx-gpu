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
