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

# PhysXGpu built from the Slang/SPIR-V port over Vulkan compute (PX_GPU_BACKEND=VULKAN).
#
# The counterpart of ../cmakegpu without the CUDA toolkit: ONE shared library,
# PhysXGpu_64.dll / libPhysXGpu_64.so, that exports the same C ABI the SDK's GPU
# loader (source/physx/src/gpu/PxPhysXGpuModuleLoader.cpp) resolves, so an
# application switches backends by swapping the module.  Next to it goes
# PhysXGpuShaders/ with the SPIR-V of every kernel the manifest maps; the module
# finds it relative to its own location (PXG_SHADER_DIR overrides).
#
# The directory scaffolding (configurations, flags, output dirs, postfixes, CRT)
# mirrors ../cmakegpu: those are directory-scoped and are NOT inherited from the
# sibling ../cmake tree.

cmake_minimum_required(VERSION 3.21)

PROJECT(PhysXGpu C CXX)

IF(DEFINED ENV{PM_winsdk_PATH} AND NOT "$ENV{PM_winsdk_PATH}" STREQUAL "" AND DEFINED ENV{PM_msvc_PATH} AND NOT "$ENV{PM_msvc_PATH}" STREQUAL "")
	SET(CMAKE_VS_SDK_INCLUDE_DIRECTORIES "$ENV{PM_winsdk_PATH}/include/ucrt;$ENV{PM_winsdk_PATH}/include/um;$ENV{PM_winsdk_PATH}/include/shared")
	SET(CMAKE_VS_SDK_LIBRARY_DIRECTORIES "$ENV{PM_winsdk_PATH}/lib/ucrt/x64;$ENV{PM_winsdk_PATH}/lib/um/x64;$ENV{VCToolsInstallDir}/lib/x64;$ENV{VCToolsInstallDir}/atlmfc/lib/x64")
ENDIF()

CMAKE_POLICY(SET CMP0057 NEW) # Enable IN_LIST

# This is required to be defined by external callers!
IF(NOT DEFINED PHYSX_ROOT_DIR)
	MESSAGE(FATAL_ERROR "PHYSX_ROOT_DIR variable wasn't set.")
ENDIF()

IF(NOT EXISTS ${PHYSX_ROOT_DIR})
	MESSAGE(FATAL_ERROR "PHYSX_ROOT_DIR variable was invalid.")
ENDIF()

IF(NOT TARGET_BUILD_PLATFORM STREQUAL "windows" AND NOT TARGET_BUILD_PLATFORM STREQUAL "linux")
	MESSAGE(FATAL_ERROR "PX_GPU_BACKEND=VULKAN supports the windows and linux platforms, not '${TARGET_BUILD_PLATFORM}'.")
ENDIF()

INCLUDE(NvidiaBuildOptions)

SET(CMAKE_POSITION_INDEPENDENT_CODE ON)

IF(CMAKE_CONFIGURATION_TYPES)
	SET(CMAKE_CONFIGURATION_TYPES debug checked profile release)
	SET(CMAKE_CONFIGURATION_TYPES "${CMAKE_CONFIGURATION_TYPES}" CACHE STRING
		"Reset config to what we need"
		FORCE)

	# Need to define these at least once.
	SET(CMAKE_SHARED_LINKER_FLAGS_CHECKED "/DEBUG")
	SET(CMAKE_SHARED_LINKER_FLAGS_PROFILE "/DEBUG")
	SET(CMAKE_EXE_LINKER_FLAGS_PROFILE "/DEBUG")
	SET(CMAKE_EXE_LINKER_FLAGS_CHECKED "/DEBUG")
ENDIF()

IF (NOT DEFINED PHYSX_CXX_FLAGS)
	MESSAGE(FATAL_ERROR "PhysXGpu uses the PhysX CXX flags, and they're not defined when they need to be.")
ENDIF()

# Get the CXX Flags from the Cached variables set by the PhysX CMakeLists
SET(CMAKE_CXX_FLAGS "${PHYSX_CXX_FLAGS}")

SET(CMAKE_CXX_FLAGS_DEBUG   ${PHYSX_CXX_FLAGS_DEBUG})
SET(CMAKE_CXX_FLAGS_CHECKED ${PHYSX_CXX_FLAGS_CHECKED})
SET(CMAKE_CXX_FLAGS_PROFILE ${PHYSX_CXX_FLAGS_PROFILE})
SET(CMAKE_CXX_FLAGS_RELEASE ${PHYSX_CXX_FLAGS_RELEASE})

IF(TARGET_BUILD_PLATFORM STREQUAL "windows")
	SET(CMAKE_C_FLAGS "${PHYSX_C_FLAGS}")

	SET(CMAKE_C_FLAGS_DEBUG   ${PHYSX_C_FLAGS_DEBUG})
	SET(CMAKE_C_FLAGS_CHECKED ${PHYSX_C_FLAGS_CHECKED})
	SET(CMAKE_C_FLAGS_PROFILE ${PHYSX_C_FLAGS_PROFILE})
	SET(CMAKE_C_FLAGS_RELEASE ${PHYSX_C_FLAGS_RELEASE})

	# Build PDBs for all configurations
	SET(CMAKE_SHARED_LINKER_FLAGS "/DEBUG /INCREMENTAL:NO")
	SET(CMAKE_SHARED_LINKER_FLAGS_DEBUG "/DEBUG /INCREMENTAL:NO")
	SET(CMAKE_SHARED_LINKER_FLAGS_CHECKED "/DEBUG /INCREMENTAL:NO /OPT:REF")
	SET(CMAKE_SHARED_LINKER_FLAGS_PROFILE "/DEBUG /INCREMENTAL:NO /OPT:REF")
	SET(CMAKE_SHARED_LINKER_FLAGS_RELEASE "/DEBUG /INCREMENTAL:NO /OPT:REF")
ENDIF()

SET(PHYSX_SOURCE_DIR ${PHYSX_ROOT_DIR}/source)
SET(GPUSLANG_ROOT ${PHYSX_SOURCE_DIR}/gpuslang)

#
# Toolchain: Vulkan (loader + headers >= 1.3), slangc, Python 3
#

FIND_PACKAGE(Vulkan REQUIRED)
# The compute device uses the Vulkan 1.3 core API (synchronization2 and friends).
# FindVulkan reports a version only from CMake 3.23 on, so read the header.
FIND_FILE(PXG_VULKAN_CORE_H vulkan/vulkan_core.h PATHS ${Vulkan_INCLUDE_DIRS} NO_DEFAULT_PATH)
IF(PXG_VULKAN_CORE_H)
	FILE(STRINGS ${PXG_VULKAN_CORE_H} _pxgVkComplete REGEX "^#define VK_HEADER_VERSION_COMPLETE ")
	IF(_pxgVkComplete MATCHES "VK_MAKE_API_VERSION\\(0, *([0-9]+), *([0-9]+),")
		IF(CMAKE_MATCH_1 LESS 1 OR (CMAKE_MATCH_1 EQUAL 1 AND CMAKE_MATCH_2 LESS 3))
			MESSAGE(FATAL_ERROR "PhysXGpu (Vulkan) needs the Vulkan 1.3 headers; ${PXG_VULKAN_CORE_H} is ${CMAKE_MATCH_1}.${CMAKE_MATCH_2}. Install a Vulkan SDK >= 1.3.")
		ENDIF()
	ENDIF()
ENDIF()

# The engine's SPIR-V, compiled with the verification suite's exact commands.
SET(PXG_ENGINE_SHADER_DIR ${CMAKE_CURRENT_BINARY_DIR}/shaders)
SET(SHADER_OUT_DIR ${PXG_ENGINE_SHADER_DIR})
INCLUDE(${GPUSLANG_ROOT}/cmake/PxgShaders.cmake)
INCLUDE(${GPUSLANG_ROOT}/cmake/PxgEngineShaders.cmake)
pxg_add_engine_shaders(${PXG_ENGINE_SHADER_DIR})

#
# Sources
#

INCLUDE(${GPUSLANG_ROOT}/cmake/PxgVulkanGpuSources.cmake)
pxg_vulkan_gpu_sources(${PHYSX_SOURCE_DIR})

# The PhysX code the CUDA PhysXGpu compiles into itself as well (it has its own
# foundation instance): one source of truth, read from ../cmakegpu.  Its platform
# file only sets variables; the common one also creates a target, so its source
# list is read rather than included.
SET(PROJECT_CMAKE_FILES_DIR source/compiler/cmakegpu)
INCLUDE(${PHYSX_ROOT_DIR}/${PROJECT_CMAKE_FILES_DIR}/${TARGET_BUILD_PLATFORM}/PhysXGpuDependencies.cmake)
FILE(STRINGS ${PHYSX_ROOT_DIR}/${PROJECT_CMAKE_FILES_DIR}/PhysXGpuDependencies.cmake _pxgDepLines
	REGEX "^[ \t]*\\\${PHYSX_SOURCE_DIR}/[^ \t]*\\.cpp[ \t]*$")
SET(PXG_VK_DEPENDENCY_SOURCES ${GPUDEPENDENCIES_PLATFORM_SOURCE})
FOREACH(_line ${_pxgDepLines})
	STRING(STRIP "${_line}" _line)
	STRING(REPLACE "\${PHYSX_SOURCE_DIR}" "${PHYSX_SOURCE_DIR}" _line "${_line}")
	LIST(APPEND PXG_VK_DEPENDENCY_SOURCES ${_line})
ENDFOREACH()
SET_PROPERTY(DIRECTORY APPEND PROPERTY CMAKE_CONFIGURE_DEPENDS
	${PHYSX_ROOT_DIR}/${PROJECT_CMAKE_FILES_DIR}/PhysXGpuDependencies.cmake)

SOURCE_GROUP("gpu" FILES ${PXG_VK_HOST_SOURCES})
SOURCE_GROUP("bridge" FILES ${PXG_VK_BRIDGE_SOURCES})
SOURCE_GROUP("backend" FILES ${PXG_VK_BACKEND_SOURCES})
SOURCE_GROUP("dependencies" FILES ${PXG_VK_DEPENDENCY_SOURCES})

IF(TARGET_BUILD_PLATFORM STREQUAL "windows")
	# PHYSXGPU_RESOURCE: the same version resource as the CUDA module.
	INCLUDE(${PHYSX_ROOT_DIR}/${PROJECT_CMAKE_FILES_DIR}/windows/PhysXGpuPlatformFiles.cmake)
	SOURCE_GROUP(resource FILES ${PHYSXGPU_RESOURCE})
ENDIF()

ADD_LIBRARY(PhysXGpu SHARED
	${PXG_VK_HOST_SOURCES}
	${PXG_VK_BRIDGE_SOURCES}
	${PXG_VK_BACKEND_SOURCES}
	${PXG_VK_DEPENDENCY_SOURCES}
	${PHYSXGPU_RESOURCE}
)

TARGET_INCLUDE_DIRECTORIES(PhysXGpu
	PRIVATE ${PXG_VK_INCLUDE_DIRS}
	PRIVATE ${PHYSX_ROOT_DIR}/include
	PRIVATE ${PHYSX_SOURCE_DIR}/common/include
	PRIVATE ${PHYSX_SOURCE_DIR}/common/src
	PRIVATE ${PHYSX_SOURCE_DIR}/geomutils/include
	PRIVATE ${PHYSX_SOURCE_DIR}/geomutils/src
	PRIVATE ${PHYSX_SOURCE_DIR}/geomutils/src/common
	PRIVATE ${PHYSX_SOURCE_DIR}/geomutils/src/contact
	PRIVATE ${PHYSX_SOURCE_DIR}/geomutils/src/convex
	PRIVATE ${PHYSX_SOURCE_DIR}/geomutils/src/hf
	PRIVATE ${PHYSX_SOURCE_DIR}/geomutils/src/mesh
	PRIVATE ${PHYSX_SOURCE_DIR}/geomutils/src/pcm
	PRIVATE ${PHYSX_SOURCE_DIR}/lowlevel/api/include
	PRIVATE ${PHYSX_SOURCE_DIR}/lowlevel/api/include/windows
	PRIVATE ${PHYSX_SOURCE_DIR}/lowlevel/common/include/pipeline
	PRIVATE ${PHYSX_SOURCE_DIR}/lowlevel/common/include/utils
	PRIVATE ${PHYSX_SOURCE_DIR}/lowlevel/software/include
	PRIVATE ${PHYSX_SOURCE_DIR}/lowlevelaabb/include
	PRIVATE ${PHYSX_SOURCE_DIR}/lowleveldynamics/include
	PRIVATE ${PHYSX_SOURCE_DIR}/lowleveldynamics/shared
	PRIVATE ${PHYSX_SOURCE_DIR}/lowleveldynamics/src
	PRIVATE ${PHYSX_SOURCE_DIR}/cudamanager/include
	PRIVATE ${PHYSX_SOURCE_DIR}/physx/src/opensource/cudamanager/include
	PRIVATE ${PHYSX_SOURCE_DIR}/task/include
)

IF(TARGET_BUILD_PLATFORM STREQUAL "windows")
	# PX_PHYSX_STATIC_LIB: the module compiles its own foundation, exactly like the
	# CUDA PhysXGpu (../cmakegpu/windows/PhysXGpu.cmake).
	SET(PHYSXGPU_COMPILE_DEFS
		${PHYSX_WINDOWS_COMPILE_DEFS};${PXG_VK_DEFINES};PX_PHYSX_STATIC_LIB

		$<$<CONFIG:debug>:${PHYSX_WINDOWS_DEBUG_COMPILE_DEFS};>
		$<$<CONFIG:checked>:${PHYSX_WINDOWS_CHECKED_COMPILE_DEFS};>
		$<$<CONFIG:profile>:${PHYSX_WINDOWS_PROFILE_COMPILE_DEFS};>
		$<$<CONFIG:release>:${PHYSX_WINDOWS_RELEASE_COMPILE_DEFS};>
	)
ELSE()
	SET(PHYSXGPU_COMPILE_DEFS
		${PHYSX_LINUX_COMPILE_DEFS};${PXG_VK_DEFINES}

		$<$<CONFIG:debug>:${PHYSX_LINUX_DEBUG_COMPILE_DEFS};>
		$<$<CONFIG:checked>:${PHYSX_LINUX_CHECKED_COMPILE_DEFS};>
		$<$<CONFIG:profile>:${PHYSX_LINUX_PROFILE_COMPILE_DEFS};>
		$<$<CONFIG:release>:${PHYSX_LINUX_RELEASE_COMPILE_DEFS};>
	)
ENDIF()

TARGET_COMPILE_DEFINITIONS(PhysXGpu
	PRIVATE ${PHYSXGPU_COMPILE_DEFS}
)

SET_TARGET_PROPERTIES(PhysXGpu PROPERTIES
	OUTPUT_NAME PhysXGpu
	POSITION_INDEPENDENT_CODE TRUE
	# The bridge defines cu* driver entry points of its own (the CUDA-driver shim);
	# they must bind inside the module, never to a libcuda loaded into the process.
	CXX_VISIBILITY_PRESET hidden
	VISIBILITY_INLINES_HIDDEN TRUE
	COMPILE_PDB_NAME_DEBUG "PhysXGpu${CMAKE_DEBUG_POSTFIX}"
	COMPILE_PDB_NAME_CHECKED "PhysXGpu${CMAKE_CHECKED_POSTFIX}"
	COMPILE_PDB_NAME_PROFILE "PhysXGpu${CMAKE_PROFILE_POSTFIX}"
	COMPILE_PDB_NAME_RELEASE "PhysXGpu${CMAKE_RELEASE_POSTFIX}"
)

IF(TARGET_BUILD_PLATFORM STREQUAL "windows")
	# The bridge and the Vulkan device use the standard library (fstream, mutex,
	# vector); PhysX's own flags carry no /EH.  Warnings stay on (/W4) but do not
	# fail the build: these sources have not been through MSVC's /W4 /WX yet.
	TARGET_COMPILE_OPTIONS(PhysXGpu PRIVATE /EHsc /WX-)
	# vulkan-1.dll is imported normally (no delay-load): when the Vulkan loader is
	# missing, loading PhysXGpu_64.dll fails and the SDK falls back to the CPU.
	TARGET_LINK_LIBRARIES(PhysXGpu PRIVATE Vulkan::Vulkan)
	SET_TARGET_PROPERTIES(PhysXGpu PROPERTIES LINK_FLAGS "/MAP")
ELSE()
	IF(CMAKE_CXX_COMPILER_ID MATCHES "Clang")
		# PhysX builds with clang's -Weverything -Werror.  The kernel-ABI tables and
		# the Vulkan device initialise structs by leading members (the rest
		# zero-initialised on purpose) and cast vkGetDeviceProcAddr results.
		SET_SOURCE_FILES_PROPERTIES(
			${GPUSLANG_ROOT}/bridge/PxgVulkanKernelAbi.cpp
			${GPUSLANG_ROOT}/src/vulkan/PxgVulkanComputeDevice.cpp
			PROPERTIES COMPILE_OPTIONS "-Wno-missing-field-initializers;-Wno-unknown-warning-option;-Wno-cast-function-type-strict")
	ENDIF()
	TARGET_LINK_LIBRARIES(PhysXGpu PRIVATE Vulkan::Vulkan ${CMAKE_DL_LIBS} pthread)
	# Resolve everything at link time, as the Windows linker does: an unresolved
	# symbol fails the build here instead of the module's dlopen at run time.
	TARGET_LINK_OPTIONS(PhysXGpu PRIVATE -Wl,--no-undefined)
ENDIF()

#
# Shaders next to the module: <lib dir>/<config>/PhysXGpuShaders/*.spv
#

# The library's own output directory (NvidiaBuildOptions: <lib dir>/<config>),
# spelled without $<TARGET_FILE_DIR:PhysXGpu> so PhysXGpu can depend on this target.
IF(NOT CMAKE_CONFIGURATION_TYPES)
	STRING(TOLOWER "${CMAKE_BUILD_TYPE}" _pxgBuildType)
	IF(NOT _pxgBuildType MATCHES "^(debug|checked|profile|release)$")
		MESSAGE(FATAL_ERROR "PhysXGpu (Vulkan): CMAKE_BUILD_TYPE must be debug, checked, profile or release (got '${CMAKE_BUILD_TYPE}'); generate_projects sets it.")
	ENDIF()
ENDIF()
# Lower-cased like the output directories (CMAKE_LIBRARY_OUTPUT_DIRECTORY_<CFG> are
# lower-case paths), so -DCMAKE_BUILD_TYPE=Release still puts the shaders next to the library.
SET(PXG_SHADER_BIN_DIR "${PX_OUTPUT_LIB_DIR}/${PX_ROOT_LIB_DIR}/$<LOWER_CASE:$<CONFIG>>/PhysXGpuShaders")
# copy_directory, not one copy_if_different with every file: 170 paths overflow
# cmd.exe's 8191-character command line on Windows.  The directory holds exactly the
# engine shaders (PXG_SHADER_ONLY); it is emptied first, so modules a newer manifest
# no longer maps (PhysX 5.12 dropped 8) do not linger next to the library.
ADD_CUSTOM_TARGET(PhysXGpuShaders ALL
	COMMAND ${CMAKE_COMMAND} -E rm -rf "${PXG_SHADER_BIN_DIR}"
	COMMAND ${CMAKE_COMMAND} -E make_directory "${PXG_SHADER_BIN_DIR}"
	COMMAND ${CMAKE_COMMAND} -E copy_directory "${PXG_ENGINE_SHADER_DIR}" "${PXG_SHADER_BIN_DIR}"
	DEPENDS ${PXG_ENGINE_SHADER_OUTPUTS}
	COMMENT "PhysXGpu (Vulkan): SPIR-V -> ${PXG_SHADER_BIN_DIR}"
	VERBATIM)
ADD_DEPENDENCIES(PhysXGpu PhysXGpuShaders)

SET_PROPERTY(TARGET PhysXGpu PROPERTY FOLDER "PhysX SDK/GPU")
SET_PROPERTY(TARGET PhysXGpuShaders PROPERTY FOLDER "PhysX SDK/GPU")

SET(PXG_INSTALL_LIB_DIR $<$<CONFIG:debug>:${PX_ROOT_LIB_DIR}/debug>$<$<CONFIG:release>:${PX_ROOT_LIB_DIR}/release>$<$<CONFIG:checked>:${PX_ROOT_LIB_DIR}/checked>$<$<CONFIG:profile>:${PX_ROOT_LIB_DIR}/profile>)

# Only the legacy PhysXSDK export set of the standalone build (compiler/public), as
# in ../cmakegpu: the FetchContent entry point (physx/CMakeLists.txt) defines
# PX_ENABLE_INSTALL and installs through its own export set.
IF(NOT DEFINED PX_ENABLE_INSTALL)
	INSTALL(
		TARGETS PhysXGpu
		EXPORT PhysXSDK
		DESTINATION ${PXG_INSTALL_LIB_DIR}
	)
	INSTALL(FILES ${PXG_ENGINE_SHADER_OUTPUTS} DESTINATION ${PXG_INSTALL_LIB_DIR}/PhysXGpuShaders)
	IF(TARGET_BUILD_PLATFORM STREQUAL "windows")
		INSTALL(FILES $<TARGET_PDB_FILE:PhysXGpu> DESTINATION ${PXG_INSTALL_LIB_DIR} OPTIONAL)
	ENDIF()
ENDIF()
