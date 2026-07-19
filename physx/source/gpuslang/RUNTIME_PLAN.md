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

### P0 results (executed; CI device = llvmpipe, LLVM 20.1.2, Vulkan 1.4)

Probe (`tools/pxgCaps.cpp`, target `pxg-caps`) matrix on the CI device:

| capability | llvmpipe |
|---|---|
| subgroup default / min / max | 8 / 8 / 8 (fixed) |
| requiredSubgroupSizeStages incl. compute | yes |
| computeFullSubgroups | yes |
| subgroup ops (basic/vote/arith/ballot/shuffle+rel) | yes |
| bufferDeviceAddress | yes |
| timelineSemaphore | yes |
| shaderInt64 / shaderFloat64 | yes / yes |
| shaderBufferInt64Atomics | yes |
| VK_EXT_shader_atomic_float (buffer+shared f32 add) | yes |
| vulkanMemoryModel | yes |
| maxComputeSharedMemorySize | 32 KiB |

Spike outcomes (tests `p0Subgroup`, `p0Bda`, both green):
- Wave invariants (prefix-sum/ballot/active-sum) hold at the default AND at a
  `requiredSubgroupSize` pipeline; out-of-window requests are rejected at
  pipeline creation.  The `createPipelineEx` mechanism is proven end-to-end.
- **Risk R1 is CONFIRMED for CI**: llvmpipe's subgroup window is [8,8] —
  required-32 is impossible there.  Decision locked in: the P2.1 primitive
  library ships TWO code paths selected at pipeline-build time — native
  subgroup ops under required-32 (hardware), and a shared-memory emulation of
  the warp-32 semantics (any subgroup size; used by CI and any <32 device).
  Both paths are bit-identical by construction (integer/exact-float ops).
- BDA argument marshalling works exactly as §4 assumes: a CUDA-shaped param
  block of raw `uint64_t` device addresses travels via push constants, the
  dispatch binds ZERO descriptors, and Slang `Ptr<float>(addr)` dereferences
  produce bitwise-correct results.  The P3 shim's launch model is validated.


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

### P1 progress

- **P1.1 DONE** — the Vulkan backend now runs N-buffered command buffers
  (4 frames in flight, per-frame descriptor pools recycled on slot reuse)
  retired through one timeline semaphore per queue.  `submit()` returns a
  monotonic ticket without waiting; `wait(ticket)`/`poll(ticket)`
  synchronize; `flush()` == `wait(submit())`.  Fence fallback retained for
  devices without timeline semaphores (blocking submit, ticket 0).
- **P1.2 memory path DONE** (sub-allocator still open) —
  `PxgSlangDeviceDesc.useDeviceLocalMemory` selects DEVICE_LOCAL storage
  buffers; all host<->device transfers stage through a persistently mapped
  8 MiB per-frame ring with per-buffer transfer<->compute barriers (P1.3's
  readback/upload edges).  The staged path is deliberately taken even when
  the memory type is host-visible (UMA/llvmpipe) so CI exercises the
  production path.  Env `PXG_DEVICE_V2=1` re-runs the ENTIRE bit-exact
  suite through device-local + staged + timeline mode; test `p1DeviceV2`
  additionally covers ring-crossing 20 MiB round-trips, offset semantics,
  ticket monotonicity, and a kernel dispatch on device-local storage on
  every normal suite run.
- **P1.2 sub-allocator DONE** — `PxgSlangHeapAllocator`
  (include/PxgSlangHeapAllocator.h): buddy allocator over device-buffer
  roots with `PxgHeapMemoryAllocator` semantics — 128 B minimum class and
  alignment, buddy coalescing, growth by whole roots, exceptional oversize
  allocations on dedicated buffers, deferred frees flushed at step
  boundaries, per-group statistics.  Unit test `p1HeapAlloc` covers the
  invariants (alignment/non-overlap via a shadow interval model over a
  4000-op randomized stress, coalescing recovery after full drain, root
  growth, deferred-free visibility, exceptional round-trip through real
  device memory) in both memory modes.
- **P1.1 follow-up (adversarial review)** — every timeline submission now
  waits on the previous ticket, so consecutive `submit()` batches execute
  and become visible in submission order (single-CUDA-stream semantics;
  Vulkan does NOT order same-queue submissions by itself, which made
  dependent dispatches split across batches a latent hazard on hardware
  that overlaps compute submissions).  `destroyBuffer`/`destroyPipeline`
  flush any open recording before waiting idle (staged uploads record
  copies into the open command buffer).  Regression test: chained BDA
  dispatches across two submits with no host wait.  Known tooling note:
  Khronos validation-layer syncval up to 1.3.275 does not model
  `vkWaitSemaphores` HOST waits and false-positives WAW on staging-ring
  reuse; a fence-based probe of the identical pattern is clean (layer
  limitation, fixed in newer layers).
- **P1.6 copy/fill utilities SHIPPED**: `copyBuffer` (`vkCmdCopyBuffer`,
  cuMemcpyDtoD parity) and `fillBuffer` (`vkCmdFillBuffer`, cuMemsetD32
  parity) recorded into the current stream with compute<->transfer
  barriers; the shim's `memcpyDtoD`/`memsetD32` now route through them
  instead of host-staging.  Base-class defaults host-stage so backends
  without a device path still work.  This surfaced and fixed a latent
  stream-ordering gap: host-visible `upload`/`download` now drain prior
  recorded/in-flight GPU work first (cuMemcpyHtoD/DtoH are stream-ordered)
  so a recorded fill/copy/dispatch cannot race a direct host map — cheap
  when idle.  Coverage: `p1DeviceV2` gains sub-range fill + offset copy
  checks (both memory modes); mutants caught (copy offset swap, fill
  offset dropped, download-without-flush regression).
- **P1.5 pipeline cache SHIPPED**: one `VkPipelineCache` per device, always
  used by `vkCreateComputePipelines`; when `PxgSlangDeviceDesc.pipelineCachePath`
  is set it is seeded from that blob at creation and written back at
  destruction, so shader/pipeline compilation is reused across runs.  A
  foreign or corrupt blob is rejected by the driver's header check and simply
  starts empty (verified).  Test `p1PipelineCache`: first device writes a
  header-valid blob, a second device seeded from it produces bitwise-identical
  kernel results, and a garbage blob neither crashes creation nor changes
  results; mutant caught (no-save fails the written-blob check).
- **Indirect dispatch SHIPPED** (P6 plumbing): `dispatchIndirect(pipeline,
  argBuffer, argOffset, ...)` reads the workgroup counts as a
  VkDispatchIndirectCommand from a device buffer (`vkCmdDispatchIndirect`),
  with a compute-write -> indirect-read barrier on the arg buffer; all
  buffers now carry `INDIRECT_BUFFER_BIT`.  Test `p6Indirect`: a producer
  kernel computes and writes the grid on device, the consumer is launched
  indirectly, and covering all elements proves the device-decided grid was
  honored — mutant caught (a fixed 1-group dispatch writes only 64/4096).
  (The indirect-read barrier's absence is not observable on llvmpipe, same
  class as the timeline syncval limitation; it is spec-correct for hardware.)
- **Adversarial review follow-up** (copy/fill + cache + indirect batches):
  review found no BLOCKER/MAJOR.  Applied fixes: host-visible `download`
  now issues a device->host domain barrier (`HOST_READ`) before the map,
  closing the visibility edge that HOST_COHERENT alone does not guarantee
  and that routing memset/DtoD through device transfers had widened;
  `savePipelineCache` removes a truncated blob on a short write; 4-byte
  alignment asserts on `fillBuffer`/`dispatchIndirect` offsets (the
  cuMemset/indirect-command contract).
- **Timestamp queries SHIPPED** (P1.6): a per-device `VkQueryPool`
  (TIMESTAMP), reset per recording batch, gated on the compute queue's
  `timestampValidBits` and a nonzero `timestampPeriod`.  `writeTimestamp()`
  records a bottom-of-pipe timestamp and returns its index;
  `getTimestampNanos(begin, end)` reads the pair after retirement and returns
  elapsed ns via masked modular subtraction ordered by query index (wrap-safe,
  argument-order-independent).  Test `p1Timestamps`: a dispatch bracketed by
  two timestamps yields a non-negative, finite ns count (~1.1 ms on llvmpipe),
  index monotonicity holds, symmetry holds, and out-of-range indices are
  rejected; mutants caught (index not advancing, reversed subtraction).
- Still open in P1/P6: specialization constants, SubmissionContexts
  (multi-stream; P1 serializes all work on one queue in submission order),
  async compute (second queue).


### P2.1 progress

- **Primitive library v1 SHIPPED** (`shaders/pxgSlangWarpInc.slang`,
  preprocessor-included so the path switch + groupshared scratch land in
  each kernel): CUDA-exact `pxgShfl{,Up,Down,Xor}Sync` (width-segmented,
  wrap/clamp semantics), `pxgBallotSync`/`pxgAnySync`/`pxgAllSync` (mask
  honored the CUDA way), rank-in-mask, count-and-broadcast, XOR-butterfly
  reductions (add/min/max/or/and, width-parameterized, winner-lane
  variant), inclusive/exclusive add-scans, and the `__ffs`/`lowestSetIndex`
  /`highestSetIndex` bit conventions including their zero cases.
  Two paths: shared-memory emulation (default; bit-exact at ANY subgroup
  size, workgroup-uniform call sites required — warp-divergent call sites
  must restructure when ported) and native subgroup ops behind
  `PXG_WARP_NATIVE=1` (requires required-32 pipelines; validated on
  hardware in P5 — llvmpipe cannot execute it at 32).
- **Battery** (`npWarpLib` / `testNpWarpLib.cpp`): 256 scripted tests x
  two tiles per workgroup (tile isolation) x 32 lanes, sweeping widths
  1..32, partial masks (0x3F contact patch, 4-lane nibbles, random),
  wrapped/negative shuffle indices and out-of-segment XORs; every output
  bitwise against a lockstep CPU model of the CUDA semantics.  Sensitivity
  proven by four caught mutants (shfl-up clamp off-by-one, ballot mask
  dropped, scan guard off-by-one, xor clamp removed); green in both
  memory modes.
- Survey grounding (agent inventory of the CUDA layer): FULL_MASK
  dominates (741 trivial vs a handful of live partial-mask sites: 6-lane
  0x3F in vector.cuh/particlesystem.cu, 4-lane nibble masks in
  cudaBox.cu, ballot-derived masks in reduction.cuh + loop-convergence
  masks in updateBodiesAndShapes.cu); ~90% of intrinsic call sites live
  OUTSIDE the five primitive headers, so kernel ports call the
  intrinsic-parity layer directly.  Two legacy non-sync `shfl.up.b32`
  inline-PTX sites (warpHelpers.cuh) map to `pxgShflUpSyncU`; float
  atomic min/max are already CAS loops in CUDA (atomic.cuh) and port
  directly.
- **P2.2 atomic layer SHIPPED** (same include): `pxgAtomicMinF/MaxF/AddF`
  as CAS loops via `InterlockedCompareExchange` — the min/max loops are a
  direct transcription of atomic.cuh's own CAS emulations (CUDA has no
  native float min/max atomic either), the add applies each contribution
  exactly once with unspecified ordering, exactly like CUDA's native
  float `atomicAdd`; `pxgAtomicOr64` reproduces the split 32-bit OR of
  `AtomicOr(PxU64)`.  64-bit bit conventions `pxgPopcll/pxgFfsll/pxgClzll`
  with exact zero-half/zero-value cases.  Battery `npAtomicLib`: 4096
  threads contending 64 float cells + 32 OR cells with order-invariant
  designs (min/max; integer-valued float sums that are exactly
  associative; ORs), all final cell values compared bitwise; data
  includes deliberate zero halves and zero pairs for the 64-bit branch
  coverage.  Seven mutants caught (min-as-max, CAS retry dropped, OR64
  index, ffsll/clzll offsets, clzll zero case); green in both memory
  modes.  Native `VK_EXT_shader_atomic_float` fast paths (and the
  `red.global.add.f32` PTX site) remain a P6 perf item — the CAS forms
  are the semantic baseline.
- **tex3D replacement SHIPPED**: `createTexture3D` (R32F volume, optimal
  tiling, staged upload with layout transitions), trilinear clamp-to-edge
  sampler, `createPipelineTex`/`dispatchTex` bind Texture3D+SamplerState
  pairs after the storage buffers (Slang emits them as separate sampled
  image + sampler descriptors).  Kernel-side convention: CUDA tex3D
  unnormalized coordinates normalized by inverse extents (exact for
  power-of-two dims).  Battery `npTex3d`: all voxel centers BITWISE
  (frac==0 degenerates the filter on any conformant implementation) +
  2048 arbitrary points incl. all six clamp faces against a double
  trilinear reference (llvmpipe worst abs error 3.9e-07; tolerance 1e-4,
  `PXG_TEX3D_TOL` override for fixed-point hardware filters in P5).
  Mutants caught: coordinate half-offset, clamp-as-repeat.  As planned,
  this path is a documented REPLACEMENT of CUDA tex3D (Vulkan does not
  pin filter weight precision), not a bit-exact port.
- **P2.1 native-path compile validation SHIPPED**: `npWarpNative.slang`
  (`PXG_WARP_NATIVE=1`) calls every native-lowering primitive so slangc +
  spirv-val + driver pipeline creation exercise the WaveShuffle /
  WaveActiveBallot lowerings the emulation battery cannot reach.  SPIR-V
  validates with GroupNonUniform/Shuffle/Ballot capabilities; test
  `warpNative` proves the driver accepts and dispatches the native module
  (and builds it at required-32 where the device allows).  Teeth: a broken
  native intrinsic fails the native compile while the emulation path still
  builds.  Hardware execution of the 32-lane semantics stays P5 (llvmpipe
  fixed at 8).
- Still open in P2.1/P2.2: native-path **hardware execution** (P5),
  block-scope collectives (blockReduction / threadBlockScanExclusive
  family) which need workgroup barriers and are ported with their kernels.

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


### P3 progress

- **Shim core SHIPPED** (`include/PxgCudaShim.h` + `src/PxgCudaShim.cpp`,
  compiled into `gpuslang-backend`, isolated from the PhysX build): the
  load-bearing middle of the CUDA-context replacement — the parts ScScene
  actually exercises — on the v2 device layer.  `PxgCudaShim` reproduces:
  - flat device-address allocation (`memAlloc`/`memFree` returning
    `CUdeviceptr`-parity `uint64_t` addresses from `vkGetBufferDeviceAddress`),
    with an interval map so ANY interior address resolves to (buffer, offset)
    — the CUDA flat-address-space lookup;
  - `memcpyHtoD`/`DtoH`/`DtoD` and `memsetD32` keyed on device addresses
    (with correct sub-range offsets; DtoD stages through host in P3.1, a
    `vkCmdCopyBuffer` fast path is a later device-layer follow-up);
  - `launchKernel(id, grid, paramBlob)` — a manifest (`PxgKernelDef`:
    spirv/entry/pushConstantSize/numTextures) drives lazy, cached pipeline
    creation; the parameter blob is forwarded verbatim as push constants and
    the dispatch binds ZERO storage descriptors (every buffer argument is a
    device address inside the blob, incl. addresses NESTED in aggregate
    structs — the PxgSoftBody pointer-passing model, 1:1).
- **Conformance test** (`p3Shim` / `testP3Shim.cpp`): drives the shim the
  way ScScene drives PxCudaContext — alloc, HtoD, memsetD32, build a nested
  device-address param blob, launchKernel by id, streamSync, DtoH — and
  checks bitwise AXPY (`npShimAxpy.slang`, three pointers nested in a
  struct), interior/one-past-end pointer resolution, sub-range offset
  HtoD/DtoH round-trips, DtoD copy, and base-vs-interior memFree parity.
  Mutants caught: interval end-inclusive, missing predecessor step, and
  dropped HtoD/DtoH offsets (the last exposed and fixed a real coverage gap
  — all copies had used base pointers).  Green in both memory modes.
- **Stream/event surface SHIPPED** (same shim): `streamCreate/Destroy`,
  `eventCreate/Destroy/Record/Synchronize/Query`, `streamWaitEvent`,
  `streamSynchronize(stream)` — the PxCudaContext stream*/event* parity.
  P3 serializes every stream on ONE VkQueue in submission order (the P1
  correctness-first design), so a stream is a logical ordering handle and
  an event is a timeline ticket captured at record time; cross-stream
  ordering is already satisfied by the submission-order serialization
  (true multi-queue async compute is P6).  `eventRecord` submits the
  recorded batch non-blocking and stores its ticket; `eventSynchronize` =
  `wait(ticket)`, `eventQuery` = `poll(ticket)`.  Test extends `p3Shim`:
  a launch synchronized through the event API (result visible + correct
  after `eventSynchronize`, `eventQuery` then complete), a second stream
  waiting on the event, and negative-handle rejection; mutants caught
  (record captures a zero ticket, eventSynchronize skips its handle guard).
- **P3.1 PxCudaContext bridge SHIPPED**: `bridge/PxgVulkanCudaContext.{h,cpp}`
  implements the REAL `physx::PxCudaContext` (~40 virtuals) forwarding onto the
  shim, built against the actual PhysX headers (foundation + cudamanager) with
  `PX_SUPPORT_GPU_PHYSX=1` but WITHOUT the CUDA toolkit.  `PxCudaTypes.h` makes
  this feasible: `CUdeviceptr` is a 64-bit int (= device address),
  `CUstream`/`CUevent`/`CUfunction`/`CUmodule` are opaque pointers reinterpreted
  as shim handles; `launchKernel` concatenates the `PxCudaKernelParam{data,size}`
  arguments into the push-constant blob exactly as CUDA's own impl extracts
  `.data` (`CudaContextManager.cpp:1165`).  Test `p3CudaContext` drives a real
  Slang kernel end-to-end through the abstract `PxCudaContext*`
  (memAlloc/memcpyHtoD/moduleGetFunction/launchKernel/streamSynchronize/
  memcpyDtoH), bitwise-correct in both memory modes; 3 mutants caught.
- Scope note (integration survey): a full drop-in replacement of the `PhysXGpu`
  DLL is NOT just the context — `PxgPhysXGpu.cpp` `static_cast`s the abstract
  `PxsKernelWranglerManager*`/heap manager to concrete `Pxg*` types, so the whole
  gpu-module suite (broadphase/narrowphase/solver/articulation/simcontroller)
  must be rebuilt against the Vulkan context plus a Vulkan kernel-wrangler.  P3.1
  de-risks the load-bearing seam; the module bring-up (P4) and the
  `PhysXGpuVulkan` C-ABI export layer (P3.4/P3.5) are the large remaining effort.
- **501-entry manifest SHIPPED**: `tools/gen_kernel_manifest.py` parses the
  live `PxgKernelNames.h` registry + the compiled Slang entries and emits the
  generated `include/PxgSlangKernelManifest.h` — kernel-ID (enum order) ->
  {cudaName, spirvFile, entry}, the index table the Vulkan kernel-wrangler
  will use.  Launch kernels are matched by exact CUDA-name == Slang-entry
  equality.  **Coverage: 123/501 registry launch kernels (24.6%)** already
  map to a ported Slang entry by exact name; the remaining 378 are either
  renamed helper-level ports P4 maps by hand, or not yet ported at the launch
  level.  Test `p4Manifest` cross-checks the manifest against the live
  registry header (via the X-macro) for exact count + order, validates every
  mapped module's SPIR-V magic, and reports coverage; 2 drift mutants caught.
- **P3.4 PxCudaContextManager wrapper SHIPPED**: `PxgVulkanCudaContextManager`
  (same bridge) implements the REAL `physx::PxCudaContextManager` — the object
  the scene receives via `PxSceneDesc::cudaContextManager`.  It owns an internal
  PxCudaContext, reports device properties (SM levels, shared-mem/thread limits,
  device name from the Slang device), makes the per-thread acquire/release a
  balanced no-op (a single Vulkan device is not a per-thread-current CUDA
  context), and forwards the 12 deprecated protected `*Internal` buffer helpers
  onto the context.  Test `p3CudaContextManager` runs a kernel end-to-end
  through the manager's context AND exercises the `allocDeviceBuffer`/`copyHToD`/
  `copyDToH` template helpers (which drive the `*Internal` virtuals) with an
  exact round-trip; both memory modes, 2 mutants caught.
- **P3.2 Vulkan kernel-wrangler SHIPPED**: `PxgVulkanKernelWrangler` (the
  engine's `KernelWrangler` replacement) turns a kernel id (manifest index)
  into the CUfunction the context launches, using the SAME encoding
  moduleGetFunction produces — so `ctx->launchKernel(wrangler.getFunction(id))`
  resolves back to manifest[id], the exact `getCuFunction(id) -> launchKernel`
  path the engine uses.  Test `p3KernelWrangler` launches a kernel BY ID
  end-to-end (bitwise correct), rejects unmapped/OOB ids, and — built over the
  generated 501-entry registry manifest — resolves exactly 123/501 kernel ids
  to a handle (ties the wrangler to the audited coverage); both modes, 2
  mutants caught.  Engine-faithful integration note: the engine's
  `CudaKernelWrangler.h` hard-includes `<cuda.h>` only for `CUfunction`
  (which `PxCudaTypes.h` defines identically when cuda.h is absent), so a
  one-line header de-coupling would let the real `KernelWrangler` build in the
  Vulkan config; that (plus the concrete `PxgCudaKernelWranglerManager`
  static_cast site) is the P4 engine-source step.
- **P3.1 bridge HARDENED (adversarial review outcome)**: a review over the full
  bridge (context + manager + wrangler) drove three fixes, each proved by a
  teeth-bearing test (`p3ContextHardening`, both memory modes, 4 mutants caught):
  1. **Multi-argument launch marshalling** now reproduces CUDA's natural-
     alignment `.param` layout (the reference hands the driver per-arg pointers
     and CUDA reads alignment from the compiled signature; `CudaContextManager.cpp:1165`).
     The previous tight concatenation misplaced any argument following a smaller
     one (e.g. a device pointer after a `PxU32`) — a garbage buffer-device
     address on the common multi-arg launch sites.  `launchKernel` now aligns
     each `PxCudaKernelParam` to its type alignment (`paramAlign`, from `.size`);
     single-descriptor-struct kernels stay an identity copy.  Kernel
     `npShimMisalign` (a uint64 value after a uint32) fails under tight packing,
     passes under natural alignment.
  2. **Abort mode** now matches `CudaCtx`: `getLastError()` reports
     `OUT_OF_MEMORY` while aborted (was reporting the last op result — a false-
     healthy read), every op no-ops in abort mode (returns the sticky result
     without touching the GPU, so a post-allocation-failure abort stops the
     cascade instead of launching against null device pointers), and clearing
     abort recovers `mLastResult` from the sticky OOM.
  3. **Name-resolution / launchability agreement**: `moduleGetFunction` now
     requires a compiled `spirvFile`, not just a reserved `entry` name, so it
     agrees with the wrangler's `isMapped()` gate — a not-yet-ported kernel
     resolves to NotFound rather than a launchable-but-empty handle.
- **P4 items surfaced by the review (design gaps, not bridge bugs)** — must be
  resolved during per-module bring-up, before real-scene verification:
  - **Mapped-host device pointers**: `memHostGetDevicePointer` is InvalidValue
    in the P3 single-device model (it now writes a null result defensively).
    The engine's pinned-host paths (`getMappedDevicePtr`, shape/geometry/AABB
    managers) assume a valid mapped device address and do NOT gate on
    `canMapHostMemory()`/`isIntegrated()` (both have zero engine consumers), so
    those call sites must be routed through explicit staged copies — or the
    bridge must back pinned host allocations with host-visible buffer-device-
    address memory (ReBAR / `VK_EXT_external_memory_host`).
  - **Sub-word memset**: `memsetD8`/`memsetD16` map onto 32-bit
    `vkCmdFillBuffer`, so a byte/short fill with an unaligned address or a count
    not a multiple of 4 is reported InvalidValue rather than mis-filled (the
    replicated-word math and word-count conversions are correct).  A functional
    sub-word fill site needs a small fill compute shader (or a staged head/tail
    copy) added to the device layer.
- Still open in P3/P4: the `PhysXGpuVulkan` C-ABI export library
  (`PxCreatePhysXGpu`/`PxCreateCudaContextManager`/
  `PxGetSuggestedCudaDeviceOrdinal`), the engine-source de-coupling of the
  concrete wrangler/heap types, filling the manifest's pushConstantSize +
  the 378 unmapped rows during per-module bring-up.

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
