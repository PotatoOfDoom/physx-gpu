# Phase 2 runtime status

Progress tracker for the Vulkan production runtime (RUNTIME_PLAN.md P0–P7).
Phase 1 (bit-exact CUDA→Slang math port, 673 kernels) is closed; see
MIGRATION_STATUS.md.  Every item below is committed on
`claude/physx-cuda-slang-migration-mqbz7u` and verified against the standalone
gpuslang harness on llvmpipe (Vulkan 1.4) in BOTH memory modes — default
(host-visible) and `PXG_DEVICE_V2=1` (device-local + staged + timeline).

Test count: **681/681** green in both modes.

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

## Empirical device profile (CI = llvmpipe, LLVM 20.1.2)

subgroup 8/8/8 fixed (required-32 impossible → emulation on CI); BDA yes;
timelineSemaphore yes; VK_EXT_shader_atomic_float (buffer+shared) yes;
shaderFloat64/Int64 yes; vulkanMemoryModel yes; 32 KiB shared.

## Open

- **P2 native path** — `PXG_WARP_NATIVE=1` compile validation + hardware
  execution at subgroup 32 (P5; CI cannot run it).
- **P1 remainder** — pipeline cache to disk, specialization constants,
  timestamp queries, multi-queue SubmissionContexts (P1 serializes on one
  queue), indirect dispatch (P6 count-readback elimination).
- **P3 tail / P4** — the real `PxCudaContext`/`PxCudaContextManager` virtual
  wrappers forwarding onto the shim, the generated 501-entry kernel manifest,
  and the `PhysXGpuVulkan` C-ABI bridge (`PxCreatePhysXGpu` /
  `PxCreateCudaContextManager`).  These need the PhysX headers in the build
  and move verification from harness batteries to real ScScene scenes — the
  qualitative shift from isolated tranches to full-engine integration.
- **P5** — cross-vendor validation (NVIDIA/AMD RDNA+/Intel Xe+), per-device
  tex3D filter tolerance calibration.
- **P6/P7** — async compute, indirect dispatch, DX12 backend, interop.

## Verification doctrine (unchanged from Phase 1)

Every device-layer or library change: focused test in both memory modes +
mutation battery proving the test has teeth + full-suite regression in both
modes before commit. Adversarial review for non-trivial synchronization code
(the P1.1 ordering gap and the P1.6 host-visible ordering gap were both found
this way / by the wiring, not by the happy path).
