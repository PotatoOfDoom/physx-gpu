# physx-gpu — PhysX with Slang GPU kernels (CUDA migration)

This repository tracks the migration of the **NVIDIA PhysX 5.6.1** GPU
pipeline away from CUDA to **Slang** compute shaders with **Vulkan** and
**Direct3D 12** backends.

- `physx/` — the imported PhysX 5.6.1 open-source release (BSD-3-Clause),
  unmodified baseline including the original CUDA sources.
  See [UPSTREAM_README.md](UPSTREAM_README.md) and [LICENSE.md](LICENSE.md).
- `physx/source/gpuslang/` — the migration: Slang kernels, the
  Vulkan/D3D12 compute-device abstraction that replaces the CUDA driver
  layer, and a verification test suite that runs the ported kernels on a
  Vulkan device against CPU transcriptions of the CUDA originals.

Start here:

- [physx/source/gpuslang/README.md](physx/source/gpuslang/README.md) —
  architecture, build & test instructions
- [physx/source/gpuslang/MIGRATION_STATUS.md](physx/source/gpuslang/MIGRATION_STATUS.md) —
  per-kernel migration coverage (what is ported + verified, what remains)

## Quick start (verification suite)

```bash
# needs: C++17, CMake, Vulkan SDK + any Vulkan>=1.2 ICD (llvmpipe works), slangc
cmake -S physx/source/gpuslang/tests -B build-gpuslang -DCMAKE_BUILD_TYPE=Release
cmake --build build-gpuslang -j
ctest --test-dir build-gpuslang --output-on-failure
```

The suite runs GPU-less (Mesa llvmpipe) and on real GPUs alike.
