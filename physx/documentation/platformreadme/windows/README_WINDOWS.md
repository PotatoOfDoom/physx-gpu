# NVIDIA PhysX SDK for Windows

## Location of Binaries:

* SDK DLLs/LIBs: bin/
* PhysX Device DLLs: bin/

## Required packages to generate projects:

* Microsoft Windows 10 (or later)
* CMake, minimum version 3.21
* Python, minimum version 3.5
* NVIDIA CUDA Toolkit 12.8 (Not required for CPU only builds or for the Vulkan GPU presets below)

Compilers and C++ Standard:
  * Microsoft Visual Studio 2017, 2019, 2022, 2026 (64 bit)
  * Visual Studio 2026 uses the `vc18win64` / `vc18win64-cpu-only` presets and needs CMake 4.2 or newer
    (the "Visual Studio 18 2026" generator); the generation script stops with a message on an older CMake.
  * Tested with C++14 standard

## PhysX GPU Acceleration:

* Running GPU-accelerated simulations requires a CUDA toolkit 12.8 compatible display driver. The corresponding driver version can be found [here](https://docs.nvidia.com/cuda/cuda-toolkit-release-notes/index.html#cuda-major-component-versions__table-cuda-toolkit-driver-versions).
* Volta, CUDA ARCH 7.0 GPU or higher

## Generating solutions for Visual Studio:

* Solutions are generated through a script in the physx root directory: generate_projects.bat
* The generate_projects.bat script expects a preset name as a parameter; if a parameter is not provided, it lists the available presets.
* The generated solutions are in the folder compiler/"preset name".

## GPU acceleration without CUDA (Vulkan backend):

The `vc17win64-vulkan` and `vc18win64-vulkan` presets build `PhysXGpu_64.dll` from the Slang/SPIR-V port of the GPU kernels (source/gpuslang) over Vulkan compute, instead of from the CUDA kernels. No CUDA toolkit or NVIDIA driver is needed; any GPU with a Vulkan 1.3 driver can run it. The presets set the CMake switch `PX_GPU_BACKEND=VULKAN` (default: `CUDA`).

Status: the Windows build of this backend is implemented but **has not been compiled with MSVC yet**. It is built and run on Linux only; `gpuslang/tools/check_msvc_portability.py` (a ctest of the gpuslang suite) guards the MSVC-specific failure classes from there. Please report what a real Visual Studio build shows.

Required in addition to the packages above:
  * The LunarG Vulkan SDK 1.3 or newer (https://vulkan.lunarg.com). It sets `VULKAN_SDK` and ships `slangc.exe` in `%VULKAN_SDK%\Bin`.
  * slangc: the port is validated with Slang 2026.14.1; another version configures with a warning. To use a specific one, take a release from https://github.com/shader-slang/slang/releases and put it on `PATH`, set `SLANG_DIR` to the extracted directory, or pass `-DPXG_SLANGC=<path to slangc.exe>`.
  * Python 3 (the packman Python that generate_projects.bat uses is passed through) for the SPIR-V post-processing scripts.

Generating and building:
  * `generate_projects.bat vc18win64-vulkan` (or `vc17win64-vulkan`), then build `compiler/vc18win64-vulkan/PhysXSDK.sln`.
  * Output: `bin/win.x86_64.vc145.mt/<config>/PhysXGpu_64.dll` (vc143 for Visual Studio 2022) and next to it `PhysXGpuShaders\` with the compiled `.spv` files. The shaders are part of the GPU module: ship the directory together with the DLL. `INSTALL` copies both.

Shader lookup at run time, in this order:
  1. `%PXG_SHADER_DIR%\<file>` when `PXG_SHADER_DIR` is set (development builds);
  2. `PhysXGpuShaders\<file>` next to `PhysXGpu_64.dll`;
  3. `<file>` relative to the working directory.

Behaviour and limitations:
  * Without a Vulkan loader (`vulkan-1.dll`) `PhysXGpu_64.dll` does not load and the SDK runs on the CPU, as without a CUDA driver.
  * `PxGetPhysicsGpu()` returns NULL with this backend, so the skinning and isosurface snippets are not generated. See `source/gpuslang/MIGRATION_STATUS.md` for which GPU features are ported.
  * `PhysXGpu_64.dll` is loaded through the PhysX update loader (`PhysXUpdateLoader64.dll`) when one is installed with the NVIDIA PhysX System Software, which may substitute the driver's CUDA `PhysXGpu_64.dll`. To make sure the Vulkan DLL is used, load it before creating the foundation (`LoadLibrary` with its full path) or register its path with `PxSetPhysXGpuLoadHook`.
  * `PX_GENERATE_GPU_STATIC_LIBRARIES` and `PX_USE_NVTX` are not available with this backend.

