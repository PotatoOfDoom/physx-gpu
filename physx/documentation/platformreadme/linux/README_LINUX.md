# NVIDIA PhysX SDK for Linux

## Location of Binaries:

* SDK libraries: 
  * For Linux x86-64: bin/linux.x86_64
  * For Linux aarch64: bin/linux.aarch64

## Required packages to generate projects:

* CMake, minimum version 3.16
* Python, minimum version 3.5
* curl
* Clang for linux x86-64
* glibc, version 2.31 or higher (Note: GLIBC versions are typically not backwards compatible)
* NVIDIA CUDA Toolkit 12.8 (Not required for CPU only builds or for the Vulkan GPU presets below)

### Compilers and C++ Standard:
  * We support the following Ubuntu LTS releases and compilers:
    * For linux x86-64 (tested with Clang and GCC):
      * Ubuntu 20.04 LTS: Clang 10, GCC 9.4.0
      * Ubuntu 22.04 LTS: Clang 14, GCC 11.4.0
      * Ubuntu 24.04 LTS: Clang 18, GCC 13.2.0
    * For linux aarch64 (tested with GCC only):
      * Ubuntu 20.04 LTS: GCC 9.4.0
  * Tested with C++14 standard

## Generating Makefiles:

* Makefiles are generated through a script in the physx root directory: `generate_projects.sh`.
* If you have more than one version of the CUDA Toolkit installed, be sure to set the `CUDACXX` environment variable to the nvcc binary of the correct CUDA Toolkit. E.g. `export CUDACXX=/usr/local/cuda-12.8/bin/nvcc`.
* `generate_projects.sh` can be run either with a preset name as an argument or without arguments, in which case it will ask you to choose a preset from the list of available presets.
* A build directory is created for each PhysX build type (debug, checked, profile, release) at `compiler/{preset-name}-{physx-build-type}`. 

## Building SDK:

* Navigate to the build directory, e.g. `compiler/linux-clang-checked` (or run `make` with `-C compiler/linux-clang-checked`)
* Clean project: `make clean`
* Build project: `make`
* Install project: `make install`

Note:
Compile errors on unsupported compilers or platforms are frequently caused by additional warnings that are treated as errors by default. While we cannot offer support in this case, we recommend removing all occurrences of the `-Werror` flag in the file `physx/source/compiler/cmake/linux/CMakeLists.txt`.

## PhysX GPU Acceleration:

* Running GPU-accelerated simulations requires a CUDA toolkit 12.8 compatible display driver. The corresponding driver version can be found [here](https://docs.nvidia.com/cuda/cuda-toolkit-release-notes/index.html#cuda-major-component-versions__table-cuda-toolkit-driver-versions).
* Volta, CUDA ARCH 7.0 GPU or higher

## PhysX GPU Acceleration without CUDA (Vulkan backend):

* The `linux-gcc-vulkan` and `linux-clang-vulkan` presets build `libPhysXGpu_64.so` from the Slang/SPIR-V port of the GPU kernels (`source/gpuslang`) over Vulkan compute, instead of from the CUDA kernels (CMake switch `PX_GPU_BACKEND=VULKAN`; default `CUDA`). No CUDA toolkit, `nvcc` or `libcuda` is needed; any device with a Vulkan 1.3 driver runs it, including Mesa's llvmpipe on machines without a GPU.
* Additionally required: the Vulkan loader and headers 1.3 or newer (e.g. `libvulkan-dev`), `slangc` (validated with Slang 2026.14.1, from https://github.com/shader-slang/slang/releases or the LunarG Vulkan SDK; found on `PATH`, in `$SLANG_DIR/bin`, in `$VULKAN_SDK/bin`, or given with `-DPXG_SLANGC=`), and `python3`.
* Output: `bin/linux.x86_64/<config>/libPhysXGpu_64.so` and next to it `PhysXGpuShaders/` with the compiled `.spv` files, which belong to the module (`make install` copies both). At run time the shaders are looked up in `$PXG_SHADER_DIR` when that is set, else in `PhysXGpuShaders/` next to the library, else in the working directory.
* `PxGetPhysicsGpu()` returns NULL with this backend, so the skinning and isosurface snippets are not generated; `PX_GENERATE_GPU_STATIC_LIBRARIES` and `PX_USE_NVTX` are not available. See `source/gpuslang/README.md` and `source/gpuslang/MIGRATION_STATUS.md`.

## Required Packages for Building and Running PhysX Snippets:

* freeglut3
* libglu1
* libxdamage-dev
* libxmu6

## How to select the PhysX GPU Device:

* Set the environment variable PHYSX_GPU_DEVICE to the device ordinal on which GPU PhysX should run. Default is 0.
* Example: `export PHYSX_GPU_DEVICE="1"`
