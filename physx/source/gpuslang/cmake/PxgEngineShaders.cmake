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

# The engine's SPIR-V: exactly the shader files the kernel manifest
# (include/PxgSlangKernelManifest.h) maps a CUDA kernel to, compiled by the same
# pxg_compile_shader calls as the verification suite (PxgShaderList.cmake).
#
# pxg_add_engine_shaders(<out_dir>) compiles them into <out_dir> and sets, in the
# caller's scope, PXG_ENGINE_SHADER_OUTPUTS (the .spv paths).  Configuration
# fails when the manifest names a shader file the list does not compile, or an
# entry point that file's call does not export -- the shipped module would then be
# missing a pipeline, which only shows up as a NULL pipeline at run time.

include_guard(GLOBAL)

function(pxg_add_engine_shaders out_dir)
    set(manifest ${GPUSLANG_ROOT}/include/PxgSlangKernelManifest.h)
    set_property(DIRECTORY APPEND PROPERTY CMAKE_CONFIGURE_DEPENDS ${manifest})
    # Rows: { "<cudaName>", "<file>.spv", "<entry>" }  (unmapped rows carry nullptr).
    # The whole text is matched, not file(STRINGS) lines: a CMake list does not split
    # at ';' inside an unclosed '[', and row comments such as "[start,end)" would
    # merge every later row into one element.
    file(READ ${manifest} text)
    set(rowRegex "{ \"[A-Za-z0-9_]+\", \"[A-Za-z0-9_]+\\.spv\", \"[A-Za-z0-9_]+\" }")
    string(REGEX MATCHALL "${rowRegex}" rows "${text}")
    # Cross-check: every ".spv\"" in the file must belong to a parsed row.
    string(REGEX MATCHALL "\\.spv\"" spvQuotes "${text}")
    list(LENGTH rows nRows)
    list(LENGTH spvQuotes nSpv)
    if(NOT nRows EQUAL nSpv)
        message(FATAL_ERROR "Parsed ${nRows} mapped rows from ${manifest} but it names a .spv ${nSpv} times; the row format changed?")
    endif()
    set(stems "")
    foreach(row ${rows})
        string(REGEX MATCH "\"([A-Za-z0-9_]+)\\.spv\", \"([A-Za-z0-9_]+)\"" _ "${row}")
        list(APPEND stems ${CMAKE_MATCH_1})
        list(APPEND PXG_MANIFEST_ENTRIES_${CMAKE_MATCH_1} ${CMAKE_MATCH_2})
    endforeach()
    list(REMOVE_DUPLICATES stems)
    list(LENGTH stems nStems)
    if(nStems EQUAL 0)
        message(FATAL_ERROR "No mapped kernels found in ${manifest}")
    endif()

    set(SHADER_OUT_DIR ${out_dir})
    file(MAKE_DIRECTORY ${SHADER_OUT_DIR})
    set(PXG_SHADER_ONLY ${stems})
    set(PXG_SHADER_OUTPUTS "")
    include(${GPUSLANG_ROOT}/cmake/PxgShaderList.cmake)

    foreach(stem ${stems})
        get_property(exported GLOBAL PROPERTY PXG_SHADER_ENTRIES_${stem})
        if(NOT ${SHADER_OUT_DIR}/${stem}.spv IN_LIST PXG_SHADER_OUTPUTS)
            message(FATAL_ERROR "The kernel manifest maps kernels to ${stem}.spv, but PxgShaderList.cmake has no pxg_compile_shader(${stem} ...)")
        endif()
        foreach(entry ${PXG_MANIFEST_ENTRIES_${stem}})
            if(NOT entry IN_LIST exported)
                message(FATAL_ERROR "The kernel manifest maps a kernel to ${stem}.spv:${entry}, but pxg_compile_shader(${stem} ...) does not export ${entry}")
            endif()
        endforeach()
    endforeach()
    message(STATUS "PhysXGpu (Vulkan): ${nStems} engine shaders for ${nRows} mapped kernels of the manifest")
    set(PXG_ENGINE_SHADER_OUTPUTS ${PXG_SHADER_OUTPUTS} PARENT_SCOPE)
endfunction()
