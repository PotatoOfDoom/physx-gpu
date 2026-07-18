# PhysX GPU on Vulkan — Runtime Plan (Phase 2)

Status: PLANNED.  Phase 1 (this repository's `MIGRATION_STATUS.md`) is complete:
all ~200 bit-exact-portable CUDA math cores are ported to Slang, verified
bitwise against CPU references (673 green tests), mutation-batteried and
adversarially reviewed.  This document plans Phase 2: the production runtime
that makes the full formerly-CUDA-accelerated PhysX pipeline run end-to-end on
non-NVIDIA GPUs.

The plan is grounded in three code surveys (July 2026) of: (A) the as-built
gpuslang device layer, (B) the CUDA orchestration layer to be replaced, and
(C) the quantified non-portable-construct inventory of the kernel side.


## 0. What the surveys established

**A. The gpuslang device layer** (`include/PxgSlangComputeDevice.h`, Vulkan +
DX12 backends, ~1.2k LOC) already provides: persistent pipelines and buffers,
multi-dispatch recording with inter-dispatch barriers submitted in a single
flush, push constants, opt-in shaderInt64/shaderFloat64.  It lacks (test-only
simplifications): non-blocking submission (flush = vkWaitForFences(UINT64_MAX)),
timeline semaphores, device-local memory + staging (everything host-visible,
one VkDeviceMemory per buffer), a sub-allocator, per-resource barriers (one
global memory barrier), descriptor reuse (fresh set per dispatch), indirect
dispatch, pipeline cache, queries.

**B. The CUDA orchestration layer** is a plain driver-API design — no CUDA
graphs, no cooperative/grid-sync launches, no managed memory, no multi-GPU, no
cuBLAS.  Its shape:
- A C-ABI factory bridge (`PxPhysXGpu.h`: `PxCreatePhysXGpu`,
  `PxCreateCudaContextManager`, module/function tables) loaded via dlopen;
  `ScScene` consumes only the `PxPhysXGpu` factory interface.
- `PxCudaContext` (~40 virtuals): mem alloc/free, pinned host alloc + mapped
  device pointers, module load/getFunction, streams (+priority), events,
  launchKernel, memcpy H/D/DtoD async, memset, error state.
- A flat 501-entry kernel registry (`PxgKernelNames.h` / `PxgKernelIndices.h`)
  resolved name→CUfunction by the KernelWrangler; ~569 host launch sites;
  per-module grid/block-dim enums.
- Streams: broadphase (highest prio), narrowphase (mid), solver ×2,
  simulation-controller/particles/softbody/cloth (low), synchronized by
  event edges (`synchronizeStreams`); readbacks at the BP (pair counts/lists),
  NP (`fetchNarrowPhaseResults`), and solver (`gpuMemDMAbackSolverData`)
  boundaries.
- Memory: buddy sub-allocator over big device roots with category tags and
  deferred frees (`PxgHeapMemoryAllocator`), RAII device buffers with
  grow-and-copy resize (`PxgCudaBuffer`, 128 B aligned), pinned bump/stack
  allocators for double-buffered contact/patch/force streams, paged linear
  scratch with per-step reset.
- The only special runtime feature: 3D texture objects created at runtime for
  SDF collision (`PxgGeometryManager`), plus optional graphics interop.

**C. The kernel-side non-portables** (what Phase 1 deliberately excluded) are
dominated by ONE structural assumption — warp == 32 (`PxgCommonDefines.h`,
545 WARP_SIZE sites, ~1900 warp-intrinsic call sites, 1168 FULL_MASK uses) —
but almost all of it flows through four primitive headers
(`gpucommon/src/CUDA/{reduction,shuffle,copy,atomic}.cuh` +
`gpunarrowphase/src/CUDA/warpHelpers.cuh`).  Also: 11 inline-PTX sites
(named barriers, a PTX scan, an L2 atomic, a cached vector load — all
hand-translatable), float atomicAdd scatter (needs VK_EXT_shader_atomic_float
or the already-present CAS fallback pattern), one tex3D trilinear site, two
genuine f64 spots, no cooperative groups (grid-wide coordination is already
multi-pass kernels + atomic counters), shared memory within Vulkan limits
except a couple of large midphase kernels to validate against
maxComputeSharedMemorySize.


## 1. Target device profile (decided up front)

The single highest-leverage decision: **require subgroup size 32**.

Required (hard):
- Vulkan 1.2+ (prefer 1.3), compute queue.
- `VK_EXT_subgroup_size_control` with `requiredSubgroupSize = 32` and
  `REQUIRE_FULL_SUBGROUPS` on every pipeline that uses subgroup ops.
- Subgroup features: basic, vote, arithmetic, ballot, shuffle (+relative).
- `shaderInt64`.
- `timelineSemaphore` (core 1.2).
- `bufferDeviceAddress` (core 1.2) — see §4 argument-marshalling decision.
- storage-buffer 8/16-bit access NOT required (all data is 32/64/128-bit).

Optional (feature-gated with fallbacks):
- `VK_EXT_shader_atomic_float` (`shaderBufferFloat32AtomicAdd`) — fallback:
  the CAS loop already used by CUDA for float min/max, applied to float add.
- `shaderFloat64` — needed by 2 kernel spots (robust orientation predicate in
  the BVH builder, one femCloth predicate) and the Phase-1 f64-spot shaders;
  fallback: double-double or accept-degraded variants on devices without it.
- Async compute (second queue family) — perf only, never correctness.

Compatibility consequence (documented, accepted): NVIDIA (subgroup 32),
AMD RDNA+ (wave32 native), Intel Xe+ (8-32, can require 32) are in;
pre-RDNA AMD GCN (wave64-only) is out of scope.  CI note: llvmpipe's
subgroup size must be probed in P0 — if it cannot honor required-32, CI runs
the primitive-emulation path (see §5 risk R1) while hardware runs native.

Deliverable P0: a `pxg-caps` probe tool + a committed compatibility matrix,
and a spike proving required-subgroup-size-32 subgroup scan/ballot/shuffle
works on at least one real driver and on the CI device.


## 2. Phase P1 — Device layer productization (gpuslang-backend v2)

Evolve `PxgSlangComputeDevice` from test harness to engine runtime.  All items
are additive; the 673 Phase-1 tests keep running on the same API.

1. **Submission model**: N-buffered command buffers + ONE timeline semaphore
   per queue.  `submit()` returns a timeline value; `wait(value)` /
   `poll(value)`; `flush()` becomes `submit()+wait()` for test compat.
   Removes the blocking-fence design.
2. **Memory**: device-local buffers with staging upload/download rings
   (host-visible ring persistently mapped); a buddy sub-allocator over large
   device allocations reproducing `PxgHeapMemoryAllocator` semantics
   (category/stat tags, deferred frees flushed at step boundaries,
   grow-and-copy resize, 128 B alignment); "pinned host" ≙ persistently
   mapped host-visible buffers with device addresses (mapped-memory parity).
3. **Barriers**: keep the global compute→compute barrier as the default
   (matches stream-ordered CUDA semantics) but add per-buffer
   `VkBufferMemoryBarrier2` for the readback/upload edges; transfer↔compute
   stage handling.
4. **Descriptors**: adopt buffer-device-address as the primary binding model
   (§4); retain the descriptor path only for the Phase-1 test harness.
5. **Pipelines**: `VkPipelineCache` persisted to disk; pipeline creation with
   required-subgroup-size flags; specialization constants for block-size
   variants (the per-module grid/block enums).
6. **Utilities**: `vkCmdFillBuffer`-based memset (D8/D16/D32 parity),
   buffer-to-buffer copies (DtoD parity), timestamp query pools +
   debug-utils labels for per-stage GPU timing.
7. **Queues/"streams"**: a `SubmissionContext` object per CUDA stream
   (command-buffer sequence + timeline points); event parity =
   cross-context timeline waits (`synchronizeStreams` ≙ wait on another
   context's timeline value).  P1 runs all contexts on ONE VkQueue in
   submission order (correctness-first serialization); multi-queue async
   compute is deferred to P6.
8. Indirect dispatch (`vkCmdDispatchIndirect`) plumbing — not required for
   parity (CUDA launches use host-known grids + in-kernel guards) but wired
   now because P6 uses it to eliminate count readbacks.

Verification: extend the Phase-1 test framework with a `v2` device mode and
re-run all 673 bit-exact tests through device-local memory + timeline
submission (proves the new paths don't disturb kernel results); add unit
tests for the sub-allocator (fragmentation/deferred-free/resize properties).


## 3. Phase P2 — Kernel completion (the non-portable remainder)

Goal: all 501 registry kernels exist as Slang entry points compiled to
SPIR-V, with a generated manifest (kernel-ID → module/entry/local-size).

1. **Port the primitive library first** (the leverage point): `reduction.cuh`,
   `warpHelpers.cuh`, `shuffle.cuh`, `copy.cuh`, `atomic.cuh` → a single
   `pxgSubgroup.slang` module implemented on KHR subgroup ops under the
   guaranteed-32 profile.  Mask policy: keep `uint` masks (valid at size 32;
   `subgroupBallot(x).x`), so downstream mask arithmetic ports unchanged.
   Verification: bit-level unit tests per primitive (scan/reduce/compaction/
   argmax/broadcast) against CPU references — these CAN be exact (integer) or
   order-fixed (float reductions have a defined shuffle-tree order at size
   32) — plus mutation batteries, continuing the Phase-1 doctrine.
2. **Hand-translate the 11 inline-PTX sites** (named `bar.sync N` → workgroup
   barrier; PTX shuffle-scan → subgroup scan; `red.global.add.f32` → plain
   atomicAdd; `ld.global.cg.v4` → plain load).
3. **Float-atomic module**: `atomicAddFloat`/`float4` on
   VK_EXT_shader_atomic_float when present, CAS-loop fallback otherwise
   (same entry points, specialization-constant selected).
4. **tex3D SDF module**: 3D sampled images + linear sampler replacing the
   `sdfCollision.cuh` tex3D reads; runtime image creation in the geometry
   manager equivalent (see P3).  Accept vendor-dependent interpolation-weight
   precision; add contact-stability regression scenes.
5. **Port the kernel BODIES** that Phase 1 scoped out (warp-driver loops,
   blockCopy staging, solver megakernels, launch glue) — these are
   compositions of already-verified math cores plus the new primitives.
   Testing here shifts from bit-exact to STAGE PARITY: capture inputs/outputs
   of the CUDA stage boundaries (or CPU-PhysX equivalents) and compare with
   tolerances; keep the Phase-1 tests as the bit-exact floor for the cores.
6. printf/assert: stripped in release; optionally routed to
   `GL_EXT_debug_printf` in debug builds.  `__launch_bounds__` → local_size +
   (dropped) occupancy hints.  Shared-memory audit for the two large midphase
   kernels against `maxComputeSharedMemorySize`.

Order of porting mirrors P4's bring-up order so each module becomes testable
as soon as its kernels exist.


## 4. Phase P3 — Orchestration shim (the CUDA-context replacement)

Goal: PhysX's CPU side runs UNMODIFIED against a Vulkan backend library.

1. **`PxVulkanContext`** implementing the `PxCudaContext` virtual surface on
   the v2 device layer: memAlloc/Free → sub-allocator; memHostAlloc +
   memHostGetDevicePointer → mapped host-visible buffers + device addresses;
   moduleLoadDataEx/moduleGetFunction → SPIR-V manifest lookup;
   streamCreate(WithPriority) → SubmissionContexts; event* → timeline points;
   launchKernel → pipeline dispatch; memcpy*/memset* → transfer ops;
   getLastError/abort-mode parity.
2. **Argument marshalling — the load-bearing decision**: CUDA kernels take
   parameter packs containing raw device pointers (often nested inside
   structs like `PxgSoftBody`).  Rewriting every kernel signature to binding
   tables would be invasive; instead use **buffer device addresses**: kernel
   params become a single push-constant/uniform blob whose pointer fields are
   `uint64_t` device addresses, and Slang kernels declare pointer-typed
   parameters (Slang supports BDA pointers).  This preserves the CUDA
   pointer-passing model nearly 1:1 — including pointers embedded in shared
   structs — and eliminates per-launch descriptor churn.  Fallback (if a
   target can't do BDA): generated per-kernel binding tables; kept off the
   critical path.
3. **Kernel wrangler**: replace fatbin registration with the P2 manifest;
   `getCuFunction(id)` → pipeline handle (created lazily, cached, warmed from
   the pipeline cache).  The 501-entry ID enum is reused verbatim.
4. **`PxVulkanContextManager`** implementing the `PxCudaContextManager`
   surface: device selection/properties (multiprocessorCount → compute-unit
   estimate, sharedMemPerBlock → maxComputeSharedMemorySize, …), context
   lock (host mutex parity), allocator callbacks, graphics-device field
   reserved for later interop.
5. **Bridge library `PhysXGpuVulkan`** exporting the same C ABI
   (`PxCreatePhysXGpu`, `PxCreateCudaContextManager`, table functions) so the
   module loader and `ScScene` wiring stay untouched; backend selected via
   library choice or an env/config switch.
6. Geometry-manager equivalent for runtime SDF images (replaces
   CUarray/CUtexObject lifecycle).

Verification: a "shim conformance" test suite that exercises the
PxCudaContext surface directly (alloc/copy/launch/event ordering semantics),
plus P4's module bring-up as the real proof.


## 5. Phase P4 — Module bring-up (dependency order, one milestone each)

Each milestone = kernels (P2) + orchestration (P3) for that module running
under real `ScScene` task flow, with a stage-parity harness and scene tests.

1. **gpucommon**: radix sort, scans, copy manager — GPU-vs-CPU exact
   (integer) tests; already largely covered by Phase-1 kernels.
2. **Broadphase (SAP + aggregates)**: found/lost pair sets equal to CPU
   broadphase on generated scenes (set equality, order-independent).
3. **Narrowphase**: convex-convex first, then trimesh/HF/SDF paths; contact
   parity vs CPU narrowphase with tolerances (position/normal/depth), patch
   counts, and the pinned contact-stream writeback path.
4. **Solver PGS → TGS**: rigid stacks/joints scenes; metrics: penetration
   depth, energy drift, joint error vs CPU solver envelopes; threshold/force
   streams readback parity.
5. **Articulation**: Featherstone scenes vs CPU articulation results.
6. **Simulation controller + deformables/particles**: softbody/cloth/particle
   snippets; the Phase-1 FEM cores guarantee the math, the harness checks the
   assembled per-step behavior (bounded drift, no NaN, stable contacts).

Exit criterion per milestone: the corresponding PhysX snippet(s) run
end-to-end on the Vulkan backend on at least two vendors + llvmpipe(*),
determinism-within-device across repeated runs where CUDA also guaranteed it
(atomic-order-dependent stages excluded, as on CUDA).


## 6. Phase P5 — Integration & validation

- Full `ScScene` scenes with all subsystems enabled; PhysX snippet gallery as
  the acceptance suite.
- Cross-vendor matrix: NVIDIA (Vulkan), AMD RDNA (wave32), Intel Xe,
  llvmpipe (CI).  Record per-stage GPU timings via the P1 query pools.
- Per-stage debug comparators (GPU stage vs CPU stage behind a flag) as the
  standing regression net.
- Documentation: RUNTIME_STATUS.md tracking per-module bring-up state,
  feature-gate matrix, known vendor deltas (SDF interpolation precision,
  atomic-float availability, f64 gating).

## 7. Phase P6 — Performance (after correctness)

- Async compute: map the CUDA stream priorities onto multiple VkQueues where
  available (BP high / NP mid / non-rigid low), overlapping like the CUDA
  event graph; timeline semaphores already express the edges.
- Indirect dispatch to remove the BP/NP count readbacks from the critical
  path (GPU-written workgroup counts), converting the two big CPU sync points
  into GPU-driven launches.
- Per-resource barrier narrowing on hot paths; descriptor/BDA micro-opt;
  pipeline-cache warmup; sub-allocator tuning per heap category; -O2/-O3
  shader compilation for the non-bit-exact kernels (the Phase-1 bit-exact set
  keeps -O0 + NoContraction only where parity tests require it — evaluate
  relaxing per kernel once stage-parity tests replace bit-exact needs).
- Perf target: within 1.5-2x of CUDA on comparable hardware initially;
  profile-guided closing of the gap (warp-synchronous tuning was NVIDIA-
  specific; subgroup-32 keeps most of it valid on RDNA/Xe).

## 8. Phase P7 — Stretch

- DX12 runtime parity (device backend exists; needs the same v2 features +
  DXIL compilation of the manifest).
- Graphics interop (shared images/buffers with the renderer).
- Optional wave64 compatibility layer for pre-RDNA GCN (primitive-library
  emulation at gl_SubgroupSize 64) if ever needed.


## 9. Risk register

| # | Risk | Impact | Mitigation |
|---|---|---|---|
| R1 | Required subgroup size 32 unavailable (llvmpipe CI, exotic drivers) | CI can't run native primitives | P0 probe; emulation build of the primitive library on shared memory (slow but correct) for CI-only |
| R2 | VK_EXT_shader_atomic_float missing on some targets | FEM/particle scatter slow-path | CAS-loop fallback (pattern already exists in CUDA source); specialization-constant switch |
| R3 | shaderFloat64 missing (iGPUs/mobile) | 2 kernel spots + f64-spot shaders | Feature-gate; double-double emulation for the orientation predicate; documented behavior delta |
| R4 | Shared-memory overflow on 16 KB-floor devices (2 large midphase kernels) | Kernel redesign needed on those devices | P2 audit; tiled variants if a real target hits it |
| R5 | BDA pointer model mismatch in Slang on some driver | Marshalling rework | Fallback binding-table generator (designed, not default) |
| R6 | SDF trilinear precision differs per vendor | Contact jitter deltas | Stage-parity tolerances + stability scenes; optional software-filter path for strict determinism |
| R7 | Perf gap vs CUDA from over-synchronization | Adoption risk | P6 async-compute + indirect-dispatch roadmap; per-stage timestamps from P1 onward |
| R8 | Scope: ~569 launch sites / 501 kernels of glue | Schedule risk | Bring-up strictly module-by-module (P4 order); math cores are done and stay guarded by the 673-test suite |

## 10. Verification doctrine (Phase-2 adaptation)

Phase 1's doctrine (bit-exact + mutation battery + adversarial review) carries
over WHERE it applies: the primitive library (P2.1) and all Phase-1 cores.
Above that level, bit-exactness is not the contract CUDA itself provided
(atomic order, texture filtering); Phase 2 verifies with: exact set/integer
parity where defined (sorts, scans, pair sets), toleranced stage parity vs
CPU PhysX elsewhere, behavioral scene metrics, same-device determinism where
CUDA guaranteed it, and per-phase adversarial reviews of the shim semantics
(stream/event ordering, allocator lifetimes, barrier sufficiency).

## 11. Suggested execution order & rough sizing

P0 (profile+spike): small.  P1 (device v2): medium — bounded, ~10-15 focused
PRs.  P2 (kernels): the bulk — primitive library first (small but critical),
then module-by-module glue porting interleaved with P4 bring-up; comparable
in tranche count to a third of Phase 1 (the math was the hard 70%).  P3
(shim): medium; mostly mechanical once BDA marshalling is proven by a spike
(recommended as part of P0).  P4-P5: per-module milestones as listed.  P6:
open-ended, driven by profiling.

First three concrete steps:
1. P0 spike: subgroup-size-32 probe + a BDA pointer-marshalling proof kernel
   (one CUDA-shaped struct with embedded device pointers dispatched via the
   v2 path) on llvmpipe and one real driver.
2. P1.1+P1.2: timeline submission + device-local/staging memory, re-running
   the full 673-test suite through the new paths.
3. P2.1: `pxgSubgroup.slang` primitive library with its bit-level battery.
