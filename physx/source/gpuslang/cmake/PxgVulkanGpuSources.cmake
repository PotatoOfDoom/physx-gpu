## Redistribution and use in source and binary forms, with or without
## modification, are permitted provided that the following conditions
## are met:
##  * Redistributions of source code must retain the above copyright
##    notice, this list of conditions and the following disclaimer.
##  * Redistributions in binary form must reproduce the above copyright
##    notice, this list of conditions and the following disclaimer in the
##    documentation and/or other materials provided with the distribution.
##  * Neither the name of NVIDIA CORPORATION nor the names of its
##    contributors may be used to endorse or promote products derived
##    from this software without specific prior written permission.
##
## THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS ''AS IS'' AND ANY
## EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
## IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
## PURPOSE ARE DISCLAIMED.  IN NO EVENT SHALL THE COPYRIGHT OWNER OR
## CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
## EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
## PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
## PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY
## OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
## (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
## OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
##
## Copyright (c) 2008-2025 NVIDIA Corporation. All rights reserved.

# The source set of the Slang/Vulkan PhysXGpu, in ONE place.
#
# Two builds compile it: the official one (source/compiler/cmakegpuvulkan, which
# produces the shared PhysXGpu_64 module the SDK's loader opens) and the sdktest
# harness (gpuslang/sdktest, which links the same code statically into its GPU
# scene tests).  Before this file each named its own sources, and a file added to
# one would have been silently missing from the other -- the test would then
# certify a build nobody ships.
#
# pxg_vulkan_gpu_sources(<physx_source_dir>) sets, in the caller's scope:
#   PXG_VK_HOST_SOURCES     every gpu* host module .cpp (the CUDA/ device trees excluded)
#   PXG_VK_BRIDGE_SOURCES   the C-ABI PxPhysXGpu drop-in and its CUDA-context, kernel-table,
#                           driver and kernel-init shims, plus the engine KernelWrangler
#   PXG_VK_BACKEND_SOURCES  the Vulkan compute device and the CUDA-driver shim over it
#   PXG_VK_INCLUDE_DIRS     the gpu* include roots and the gpuslang include/bridge/nocuda dirs
#   PXG_VK_DEFINES          PX_PHYSX_GPU_EXPORTS;PXG_NO_CUDA_TOOLKIT
#
# PX_PHYSX_GPU_EXPORTS (not PX_PHYSX_GPU_STATIC) is what gates out the
# `#if !PX_PHYSX_GPU_EXPORTS` CUDA cubin force-linkage calls (init*Kernels*), which
# have no definition in the SPIR-V/Vulkan world.  src/dx12 is deliberately NOT part
# of the set: the D3D12 device has never been built.

include_guard(GLOBAL)

function(pxg_vulkan_gpu_sources src)
    set(gpuslang ${src}/gpuslang)

    # Each gpu* module's host .cpp files sit directly in <module>/src; the CUDA/
    # subdirectories hold the .cu device code this port replaces with Slang.
    file(GLOB_RECURSE host
        ${src}/gpucommon/src/*.cpp
        ${src}/gpubroadphase/src/*.cpp
        ${src}/gpunarrowphase/src/*.cpp
        ${src}/gpusimulationcontroller/src/*.cpp
        ${src}/gpusolver/src/*.cpp
        ${src}/gpuarticulation/src/*.cpp)
    list(FILTER host EXCLUDE REGEX "/CUDA/")
    list(FILTER host EXCLUDE REGEX "/windows/")
    list(SORT host)

    set(bridge
        ${gpuslang}/bridge/PhysXGpuVulkan.cpp
        ${gpuslang}/bridge/PxgModulePath.cpp
        ${gpuslang}/bridge/PxgVulkanCudaContext.cpp
        ${gpuslang}/bridge/PxgVulkanKernelAbi.cpp
        ${gpuslang}/bridge/PxgVulkanKernelTable.cpp
        ${gpuslang}/bridge/PxgVulkanDriverShim.cpp
        ${gpuslang}/bridge/PxgVulkanKernelInit.cpp
        # The engine KernelWrangler base the kernel manager derives from.
        ${src}/cudamanager/src/CudaKernelWrangler.cpp)

    set(backend
        ${gpuslang}/src/vulkan/PxgVulkanComputeDevice.cpp
        ${gpuslang}/src/PxgCudaShim.cpp)

    set(includes
        ${gpuslang}/include
        ${gpuslang}/bridge
        # Stand-ins for the CUDA toolkit headers the host code includes
        # (vector_types.h and friends).  Never put this on a target that also
        # sees the real toolkit.
        ${gpuslang}/nocuda
        ${src}/gpucommon/include
        ${src}/gpubroadphase/include
        ${src}/gpunarrowphase/include
        ${src}/gpusimulationcontroller/include
        ${src}/gpusolver/include
        ${src}/gpuarticulation/include
        ${src}/physxgpu/include)

    set(PXG_VK_HOST_SOURCES ${host} PARENT_SCOPE)
    set(PXG_VK_BRIDGE_SOURCES ${bridge} PARENT_SCOPE)
    set(PXG_VK_BACKEND_SOURCES ${backend} PARENT_SCOPE)
    set(PXG_VK_INCLUDE_DIRS ${includes} PARENT_SCOPE)
    set(PXG_VK_DEFINES PX_PHYSX_GPU_EXPORTS PXG_NO_CUDA_TOOLKIT PARENT_SCOPE)
endfunction()
