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

#ifndef PXG_SLANG_ENV_H
#define PXG_SLANG_ENV_H

// Environment switches shared by the engine bridge (bridge/PhysXGpuVulkan.cpp) and
// gpuslang-tests, so the two cannot drift apart again.
//
// PXG_DEVICE_V2 selects the device memory mode (PxgSlangDeviceDesc::useDeviceLocalMemory):
//   unset or ""        -> device-local buffers + staged transfers (the default mode)
//   "1"                -> device-local buffers + staged transfers (same as the default)
//   "0"                -> host-visible buffers (the opt-out)
//   anything else      -> the default (device-local), with a one-time warning on stderr
//                         naming the value.
// The default used to be host-visible, with "1" opting into device-local.  It was flipped
// because device-local is the production path on discrete GPUs: with host-visible bulk
// buffers every kernel reads and writes system memory over PCIe unless the whole of VRAM
// is host-visible (resizable BAR), so the old default cost the most exactly on the
// hardware the module is for.  "0" therefore no longer means "the default"; it is the
// explicit opt-out that keeps the host-visible mode reachable (UMA/software devices,
// bisecting a mode-dependent result).  "1" stays accepted so existing scripts keep
// working.  Before this header the bridge took ANY set value (including "0") as
// device-local, while the suite took anything but "0" as device-local; both now follow
// this table.  An unrecognised value falls back to the default rather than guessing,
// and says so loudly.

#include <cstdio>
#include <cstdlib>
#include <cstring>

namespace physx
{
namespace gpuslang
{

enum class PxgDeviceV2Value
{
    eDEVICE_LOCAL,  // unset, "" or "1" (the default)
    eHOST_VISIBLE,  // "0"
    eINVALID        // anything else (treated as the default, device-local)
};

// Pure parse of a PXG_DEVICE_V2 value (nullptr = unset); no side effects.
inline PxgDeviceV2Value pxgParseDeviceV2(const char* value)
{
    if (value == nullptr || value[0] == '\0' || std::strcmp(value, "1") == 0)
        return PxgDeviceV2Value::eDEVICE_LOCAL;
    if (std::strcmp(value, "0") == 0)
        return PxgDeviceV2Value::eHOST_VISIBLE;
    return PxgDeviceV2Value::eINVALID;
}

// Reads PXG_DEVICE_V2 and returns whether device-local memory is requested.  An
// unrecognised value warns once per process and yields true (the default mode).
inline bool pxgDeviceV2FromEnv()
{
    const char* value = std::getenv("PXG_DEVICE_V2");
    const PxgDeviceV2Value v = pxgParseDeviceV2(value);
    if (v == PxgDeviceV2Value::eINVALID)
    {
        static bool warned = false;
        if (!warned)
        {
            warned = true;
            std::fprintf(stderr,
                         "[gpuslang] WARNING: PXG_DEVICE_V2='%s' is not recognised (use 0 for host-visible, "
                         "1 or unset for device-local); using the default device-local memory mode\n",
                         value);
        }
    }
    return v != PxgDeviceV2Value::eHOST_VISIBLE;
}

} // namespace gpuslang
} // namespace physx

#endif // PXG_SLANG_ENV_H
