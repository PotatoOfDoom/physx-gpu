# Phase 2 runtime status

Progress tracker for the Vulkan production runtime (RUNTIME_PLAN.md P0–P7).
Phase 1 (bit-exact CUDA→Slang math port, 673 kernels) is closed; see
MIGRATION_STATUS.md.  Every item below is committed on
`claude/physx-cuda-slang-migration-mqbz7u` and verified against the standalone
gpuslang harness on llvmpipe (Vulkan 1.4) in BOTH memory modes — default
(host-visible) and `PXG_DEVICE_V2=1` (device-local + staged + timeline).

Test count: **695/695** green in both modes.

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
| P4.4 broadphase seam | (this commit) | The FIRST concrete gpu subsystem over Vulkan: the REAL `PxgCudaBroadPhaseSap` (the type `createGpuBroadPhase` static_casts to) constructs on the Vulkan context via both prior seams, taking the context lock, allocating its ~40 device buffers + host-mapped descriptors through the buddy allocator, and creating its stream+event through the bridge — including the first direct-driver seam, `cuCtxGetStreamPriorityRange`, shimmed by `PxgVulkanDriverShim.cpp`. Needed the no-CUDA-toolkit header stand-ins (`nocuda/vector_types.h` etc., ABI-exact vs cuda.h) + a small `PxgCudaToolkitShim.h`, and `PXG_NO_CUDA_TOOLKIT` guards on 6 gpu headers. `p4Broadphase` constructs it, proves device bytes are charged to the eBROADPHASE heap group, and round-trips real data through the shared pool. |

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
  (`cuCtxGetStreamPriorityRange`).  Remaining: the other concrete subsystems (AABB
  manager, narrowphase, sim controller, dynamics) brought up the same way; the
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
