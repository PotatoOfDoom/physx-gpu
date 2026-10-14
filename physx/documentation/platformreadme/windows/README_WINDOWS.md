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

The `vc17win64-vulkan` and `vc18win64-vulkan` presets build `PhysXGpu_64.dll` from the Slang/SPIR-V port of the GPU kernels (source/gpuslang) over Vulkan compute, instead of from the CUDA kernels. No CUDA toolkit or NVIDIA driver is needed. Building needs the Vulkan 1.3 headers; at run time a device with a Vulkan 1.2 or newer driver is accepted. So far only Mesa llvmpipe (on Linux) has run it; a device whose subgroup-size or compute limits a kernel cannot meet refuses that kernel. The presets set the CMake switch `PX_GPU_BACKEND=VULKAN` (default: `CUDA`).

Status: the Windows build of this backend is implemented but **has not been compiled with MSVC yet**. It is built and run on Linux only; `gpuslang/tools/check_msvc_portability.py` (a ctest of the gpuslang suite) guards the MSVC-specific failure classes from there. Please report what a real Visual Studio build shows.

Required in addition to the packages above:
  * The LunarG Vulkan SDK 1.3 or newer (https://vulkan.lunarg.com), for the Vulkan headers and `vulkan-1.lib`. It sets `VULKAN_SDK`.
  * slangc 2026.14.1 or newer (the port is validated with 2026.14.1). Older releases, including the `slangc.exe` some Vulkan SDKs ship, fail on the shaders ("undefined identifier 'WorkgroupCount'"). Nothing to install by hand: when the slangc found is missing or older, the configure step downloads the Slang 2026.14.1 release from GitHub once into `compiler\_slang\` (SHA-256 checked) and uses it. Search order before that: `-DPXG_SLANGC=<path to slangc.exe>`, `%SLANG_DIR%\bin` and `%SLANG_DIR%`, `PATH`, `%VULKAN_SDK%\Bin`. Without network access, download https://github.com/shader-slang/slang/releases/tag/v2026.14.1 (`slang-2026.14.1-windows-x86_64.zip`) yourself and set `SLANG_DIR` to the extracted folder. `-DPXG_SLANG_DOWNLOAD=OFF` disables the download; `-DPXG_SLANGC_ALLOW_UNTESTED=ON` keeps an older slangc (expect the error above).
  * Python 3 (the packman Python that generate_projects.bat uses is passed through) for the SPIR-V post-processing scripts.

Generating and building:
  * `generate_projects.bat vc18win64-vulkan` (or `vc17win64-vulkan`), then build `compiler/vc18win64-vulkan/PhysXSDK.sln`.
  * Output: `bin/win.x86_64.vc145.mt/<config>/PhysXGpu_64.dll` (vc143 for Visual Studio 2022) and next to it `PhysXGpuShaders\` with the compiled `.spv` files. The shaders are part of the GPU module: ship the directory together with the DLL. `INSTALL` copies both.

Shader lookup at run time:
  * If `PXG_SHADER_DIR` is set and not empty (development builds), shaders are loaded only from `%PXG_SHADER_DIR%\<file>`, with no fallback.
  * Otherwise from `PhysXGpuShaders\<file>` next to `PhysXGpu_64.dll`, then `<file>` relative to the working directory.

Behaviour and limitations:
  * Without a Vulkan loader (`vulkan-1.dll`) `PhysXGpu_64.dll` does not load and the SDK runs on the CPU, as without a CUDA driver.
  * `PxGetPhysicsGpu()` returns NULL with this backend, so the skinning and isosurface snippets are not generated. See `source/gpuslang/MIGRATION_STATUS.md` for which GPU features are ported.
  * `PhysXGpu_64.dll` is loaded through the PhysX update loader (`PhysXUpdateLoader64.dll`) when one is installed with the NVIDIA PhysX System Software, which may substitute the driver's CUDA `PhysXGpu_64.dll`. To make sure the Vulkan DLL is used, load it with `LoadLibrary` and its full path before the first SDK call that loads the GPU module (`PxCreateCudaContextManager`, `PxGetSuggestedCudaDeviceOrdinal`, or creating a GPU scene); the SDK then uses the already loaded `PhysXGpu_64.dll` and does not ask the update loader. `PxSetPhysXGpuLoadHook` only changes the requested file name; with the update loader installed, the request still goes through it.
  * `PX_GENERATE_GPU_STATIC_LIBRARIES` and `PX_USE_NVTX` are not available with this backend.

