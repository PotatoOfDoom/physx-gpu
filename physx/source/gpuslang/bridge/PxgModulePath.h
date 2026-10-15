// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions
// are met:
//  * Redistributions of source code must retain the above copyright
//    notice, this list of conditions and the following disclaimer.
//  * Redistributions in binary form must reproduce the above copyright
//    notice, this list of conditions and the following disclaimer in the
//    documentation and/or other materials provided with the distribution.
//  * Neither the name of NVIDIA CORPORATION nor the names of its
//    contributors may be used to endorse or promote products derived
//    from this software without specific prior written permission.
//
// THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS ''AS IS'' AND ANY
// EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
// IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
// PURPOSE ARE DISCLAIMED.  IN NO EVENT SHALL THE COPYRIGHT OWNER OR
// CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
// EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
// PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
// PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY
// OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
// (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
// OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
//
// Copyright (c) 2008-2025 NVIDIA Corporation. All rights reserved.


#ifndef PXG_MODULE_PATH_H
#define PXG_MODULE_PATH_H

// File-system helpers for locating the compiled shaders next to the PhysXGpu
// module, kept in their own TU so the platform APIs (Win32 / dladdr) stay out of
// the bridge proper.  All strings are UTF-8; the Windows implementation converts
// to UTF-16 and uses the wide APIs, so install paths outside the ANSI code page work.

#include <cstdint>
#include <string>
#include <vector>

namespace physx
{
namespace gpuslang
{

// The directory of the binary this code is linked into (PhysXGpu_64.dll,
// libPhysXGpu_64.so, or an executable that links the bridge statically), with a
// trailing separator.  Empty when it cannot be determined.
std::string pxgModuleDirectory();

// The value of an environment variable; false when it is unset.
bool pxgGetEnv(const char* name, std::string& value);

// UTF-8 <-> UTF-16 (Windows paths), platform-neutral so Linux tests exercise them.
std::u16string pxgUtf8ToUtf16(const std::string& s);
std::string pxgUtf16ToUtf8(const std::u16string& w);

// Reads a whole file.  False when it cannot be opened or is empty.
bool pxgReadFile(const std::string& path, std::vector<uint8_t>& data);

// The per-user cache directory for this module, with a trailing separator:
// %LOCALAPPDATA%\PhysXGpuVulkan\ on Windows, $XDG_CACHE_HOME/PhysXGpuVulkan/ or
// ~/.cache/PhysXGpuVulkan/ elsewhere.  Not created; empty when no base is known.
std::string pxgUserCacheDirectory();

// Creates `dir` and its missing parents.  True when it exists afterwards.
bool pxgMakeDirectories(const std::string& dir);

// Writes `data` to `path` through a temporary file in the same directory that then
// replaces `path`, so a reader never sees a half-written file.  False on any failure
// (the temporary is removed).
bool pxgWriteFileReplace(const std::string& path, const std::vector<uint8_t>& data);

} // namespace gpuslang
} // namespace physx

#endif // PXG_MODULE_PATH_H
