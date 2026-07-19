# RUNTIME_PLAN.md P5: full PhysX-SDK integration test.  Builds the CPU PhysX SDK
# from source (foundation + common + geomutils + lowlevel* + simulationcontroller
# + scenequery + Np core) as one static lib against our foundation recipe, with no
# packman and no CUDA toolkit, so a real PxScene can be created and stepped.  The
# GPU pipeline (our Vulkan bridge) is wired in a later milestone; this proves the
# unmodified engine SDK links + steps in the harness.
cmake_minimum_required(VERSION 3.16)
project(physx-sdktest CXX)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)
find_package(Threads REQUIRED)

get_filename_component(PXROOT "${CMAKE_CURRENT_SOURCE_DIR}/../../.." ABSOLUTE)
set(SRC ${PXROOT}/source)

# --- CPU SDK sources (globbed; exclude CUDA device code + windows platform) ---
file(GLOB_RECURSE SDK_SRC
    ${SRC}/foundation/*.cpp
    ${SRC}/task/src/*.cpp
    ${SRC}/pvd/src/*.cpp
    ${SRC}/common/src/*.cpp
    ${SRC}/geomutils/src/*.cpp
    ${SRC}/lowlevel/*.cpp
    ${SRC}/lowlevelaabb/src/*.cpp
    ${SRC}/lowleveldynamics/src/*.cpp
    ${SRC}/scenequery/src/*.cpp
    ${SRC}/simulationcontroller/src/*.cpp
    ${SRC}/physx/src/*.cpp)
list(FILTER SDK_SRC EXCLUDE REGEX "/CUDA/")
list(FILTER SDK_SRC EXCLUDE REGEX "/windows/")
list(FILTER SDK_SRC EXCLUDE REGEX "windows")
# PxGpu.cpp exports the app-facing CUDA C-ABI wrappers (PxCreateCudaContextManager
# et al.) that forward to the dlopen module loader.  Those are the SAME extern "C"
# symbols the Vulkan bridge (PhysXGpuVulkan.cpp) provides, so linking both into the
# GPU test would double-define them; the bridge is authoritative here.  Nothing in
# the SDK references these internally (engine GPU access goes through PxvGetPhysXGpu
# / the g_* pointers), so the CPU-only sdk-step is unaffected by the exclusion.
list(FILTER SDK_SRC EXCLUDE REGEX "/gpu/PxGpu.cpp")

add_library(physx-sdk STATIC ${SDK_SRC})
target_compile_definitions(physx-sdk PUBLIC PX_PHYSX_STATIC_LIB PXG_NO_CUDA_TOOLKIT PX_PUBLIC_RELEASE=1)
target_compile_options(physx-sdk PRIVATE -w -fno-strict-aliasing)
target_include_directories(physx-sdk PUBLIC
    ${PXROOT}/include
    ${PXROOT}/include/characterkinematic
    ${PXROOT}/include/collision
    ${PXROOT}/include/common
    ${PXROOT}/include/common/windows
    ${PXROOT}/include/cooking
    ${PXROOT}/include/cudamanager
    ${PXROOT}/include/extensions
    ${PXROOT}/include/filebuf
    ${PXROOT}/include/foundation
    ${PXROOT}/include/foundation/unix
    ${PXROOT}/include/foundation/unix/neon
    ${PXROOT}/include/foundation/unix/sse2
    ${PXROOT}/include/foundation/windows
    ${PXROOT}/include/geometry
    ${PXROOT}/include/geomutils
    ${PXROOT}/include/gpu
    ${PXROOT}/include/omnipvd
    ${PXROOT}/include/pvd
    ${PXROOT}/include/solver
    ${PXROOT}/include/task
    ${PXROOT}/include/vehicle2
    ${PXROOT}/include/vehicle2/braking
    ${PXROOT}/include/vehicle2/commands
    ${PXROOT}/include/vehicle2/drivetrain
    ${PXROOT}/include/vehicle2/physxActor
    ${PXROOT}/include/vehicle2/physxConstraints
    ${PXROOT}/include/vehicle2/physxRoadGeometry
    ${PXROOT}/include/vehicle2/pvd
    ${PXROOT}/include/vehicle2/rigidBody
    ${PXROOT}/include/vehicle2/roadGeometry
    ${PXROOT}/include/vehicle2/steering
    ${PXROOT}/include/vehicle2/suspension
    ${PXROOT}/include/vehicle2/tire
    ${PXROOT}/include/vehicle2/wheel
    ${SRC}/common/include/omnipvd
    ${SRC}/common/include/windows
    ${SRC}/common/src
    ${SRC}/compiler/windows/resource
    ${SRC}/cudamanager/include
    ${SRC}/filebuf/include
    ${SRC}/foundation
    ${SRC}/geomutils/include
    ${SRC}/geomutils/src
    ${SRC}/geomutils/src/ccd
    ${SRC}/geomutils/src/common
    ${SRC}/geomutils/src/contact
    ${SRC}/geomutils/src/convex
    ${SRC}/geomutils/src/cooking
    ${SRC}/geomutils/src/distance
    ${SRC}/geomutils/src/gjk
    ${SRC}/geomutils/src/hf
    ${SRC}/geomutils/src/intersection
    ${SRC}/geomutils/src/mesh
    ${SRC}/geomutils/src/pcm
    ${SRC}/geomutils/src/sweep
    ${SRC}/gpuarticulation/include
    ${SRC}/gpubroadphase/include
    ${SRC}/gpucommon/include
    ${SRC}/gpunarrowphase/include
    ${SRC}/gpusimulationcontroller/include
    ${SRC}/gpuslang/bridge
    ${SRC}/gpuslang/include
    ${SRC}/gpuslang/nocuda
    ${SRC}/gpuslang/tests
    ${SRC}/gpusolver/include
    ${SRC}/lowlevel/api/include
    ${SRC}/lowlevel/common/include/collision
    ${SRC}/lowlevel/common/include/pipeline
    ${SRC}/lowlevel/common/include/utils
    ${SRC}/lowlevel/software/include
    ${SRC}/lowlevelaabb/include
    ${SRC}/lowlevelaabb/src
    ${SRC}/lowleveldynamics/include
    ${SRC}/lowleveldynamics/shared
    ${SRC}/lowleveldynamics/src
    ${SRC}/physx/src
    ${SRC}/physx/src/omnipvd
    ${SRC}/physx/src/opensource/cudamanager/include
    ${SRC}/physxcharacterkinematic/src
    ${SRC}/physxcooking/src
    ${SRC}/physxextensions/src
    ${SRC}/physxextensions/src/omnipvd
    ${SRC}/physxextensions/src/serialization
    ${SRC}/physxextensions/src/serialization/Binary
    ${SRC}/physxextensions/src/serialization/File
    ${SRC}/physxextensions/src/serialization/Xml
    ${SRC}/physxextensions/src/tet
    ${SRC}/physxgpu/include
    ${SRC}/physxgpu/src
    ${SRC}/physxmetadata/core/include
    ${SRC}/physxmetadata/extensions/include
    ${SRC}/physxvehicle/src/pvd
    ${SRC}/pvd/include
    ${SRC}/pvd/src
    ${SRC}/scenequery/include
    ${SRC}/scenequery/src
    ${SRC}/simulationcontroller/include
    ${SRC}/simulationcontroller/src
    ${SRC}/common/include)

# --- the scene-step test ---
add_executable(sdk-step ${CMAKE_CURRENT_SOURCE_DIR}/testSdkStep.cpp)
target_link_libraries(sdk-step PRIVATE physx-sdk Threads::Threads ${CMAKE_DL_LIBS})
set_target_properties(sdk-step PROPERTIES CXX_STANDARD 17)

#=============================================================================#
# RUNTIME_PLAN.md P5 milestone B: the GPU pipeline linked against the CPU SDK.
# Builds every gpu* host module (broadphase / common / narrowphase / solver /
# simulationcontroller / articulation) no-CUDA-toolkit, plus the Vulkan bridge
# and gpuslang backend, then a GPU-enabled PxScene test that constructs the
# whole Pxg* pipeline through the bridge factories and steps it.
#=============================================================================#
find_package(Vulkan)
if(Vulkan_FOUND)
    set(GPUSLANG_ROOT ${SRC}/gpuslang)

    # gpuslang backend (Vulkan compute device + CUDA-driver shim over Vulkan).
    add_library(gpuslang-backend STATIC
        ${GPUSLANG_ROOT}/src/vulkan/PxgVulkanComputeDevice.cpp
        ${GPUSLANG_ROOT}/src/PxgCudaShim.cpp)
    set_target_properties(gpuslang-backend PROPERTIES POSITION_INDEPENDENT_CODE ON CXX_STANDARD 17)
    target_include_directories(gpuslang-backend PUBLIC ${GPUSLANG_ROOT}/include)
    target_link_libraries(gpuslang-backend PUBLIC Vulkan::Vulkan)

    # Every gpu* host module, minus the .cu-adjacent CUDA/ device trees.
    file(GLOB_RECURSE GPUHOST_SRC
        ${SRC}/gpucommon/src/*.cpp
        ${SRC}/gpubroadphase/src/*.cpp
        ${SRC}/gpunarrowphase/src/*.cpp
        ${SRC}/gpusimulationcontroller/src/*.cpp
        ${SRC}/gpusolver/src/*.cpp
        ${SRC}/gpuarticulation/src/*.cpp)
    list(FILTER GPUHOST_SRC EXCLUDE REGEX "/CUDA/")
    list(FILTER GPUHOST_SRC EXCLUDE REGEX "/windows/")

    add_library(physx-gpuhost STATIC
        ${GPUHOST_SRC}
        # The Vulkan bridge: the C-ABI PxPhysXGpu drop-in + its CUDA-context /
        # kernel-table / driver / kernel-init shims (replaces physxgpu + the real
        # CUDA CudaContextManager, which are deliberately excluded).
        ${GPUSLANG_ROOT}/bridge/PhysXGpuVulkan.cpp
        ${GPUSLANG_ROOT}/bridge/PxgVulkanCudaContext.cpp
        ${GPUSLANG_ROOT}/bridge/PxgVulkanKernelTable.cpp
        ${GPUSLANG_ROOT}/bridge/PxgVulkanDriverShim.cpp
        ${GPUSLANG_ROOT}/bridge/PxgVulkanKernelInit.cpp
        # The engine KernelWrangler base the manager derives from.
        ${SRC}/cudamanager/src/CudaKernelWrangler.cpp)
    set_target_properties(physx-gpuhost PROPERTIES POSITION_INDEPENDENT_CODE ON CXX_STANDARD 17)
    target_compile_options(physx-gpuhost PRIVATE -w -fno-strict-aliasing)
    # PX_PHYSX_GPU_EXPORTS (not STATIC_LIB): matches the proven drop-in .so recipe and
    # gates out the `#if !PX_PHYSX_GPU_EXPORTS` CUDA cubin force-linkage calls
    # (init*Kernels*), which have no definition in the SPIR-V/Vulkan world.
    target_compile_definitions(physx-gpuhost PRIVATE
        PX_PHYSX_GPU_EXPORTS PXG_NO_CUDA_TOOLKIT PX_PUBLIC_RELEASE=1)
    # physx-sdk's PUBLIC includes carry foundation/common/lowlevel/geomutils; add
    # the gpu-module include roots + the no-CUDA header stand-ins.
    target_include_directories(physx-gpuhost PRIVATE
        ${GPUSLANG_ROOT}/include ${GPUSLANG_ROOT}/bridge ${GPUSLANG_ROOT}/nocuda
        ${SRC}/gpucommon/include ${SRC}/gpubroadphase/include ${SRC}/gpunarrowphase/include
        ${SRC}/gpusimulationcontroller/include ${SRC}/gpusolver/include ${SRC}/gpuarticulation/include
        ${SRC}/physxgpu/include)
    target_link_libraries(physx-gpuhost PUBLIC physx-sdk gpuslang-backend)

    # The GPU-scene test.
    add_executable(gpu-sdk-step ${CMAKE_CURRENT_SOURCE_DIR}/testGpuSdkStep.cpp)
    target_compile_definitions(gpu-sdk-step PRIVATE PX_PHYSX_STATIC_LIB PXG_NO_CUDA_TOOLKIT PX_PUBLIC_RELEASE=1)
    target_include_directories(gpu-sdk-step PRIVATE
        ${GPUSLANG_ROOT}/include ${GPUSLANG_ROOT}/bridge ${GPUSLANG_ROOT}/nocuda
        ${SRC}/gpucommon/include ${SRC}/physxgpu/include)
    target_link_libraries(gpu-sdk-step PRIVATE physx-gpuhost physx-sdk gpuslang-backend Vulkan::Vulkan Threads::Threads ${CMAKE_DL_LIBS})
    set_target_properties(gpu-sdk-step PROPERTIES CXX_STANDARD 17)
endif()
