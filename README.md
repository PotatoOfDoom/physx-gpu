# physx-gpu — PhysX with Slang GPU kernels (CUDA migration)

This repository is a port of all PhysX GPU kernels to Slang for usage in Vulkan and DirectX. This is 100% vibeslopped but I actively tested the functionality and verified everything on my amd gpu.

Overview:

- `physx/` — the imported PhysX 5.12 open-source release (BSD-3-Clause),
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

### Or if you are using Visual Studio

1. Get the Vulkan SDK
2. start physx/generate_projects.bat
3. use preset 7 (vc18win64-vulkan)
4. open the visual studio solution physx/compiler/vc18win64-vulkan/PhysXSDK.slnx
5. Build the ALL_BUILD configuration
6. Wait a little bit
7. Go to physx/bin/win.x86_64.vc145.md/<YOUR_BUILD_PROFILE>/
8. Enjoy the snippet physx examples running on your gpu using vulkan.
9. Integrate this SDK into your project
10. Be happy that you saved so many claude tokens for your project

### Note
The Directx 12 part is untested. I only really verified the Vulkan part.