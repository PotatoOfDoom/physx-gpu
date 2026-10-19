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

// artiCoreDescProbe.slang -- a test-only probe of the PxgArticulationCoreDesc
// address map (pxgSlangArticulationDesc.slang).  No engine kernel; no manifest row.

import pxgSlangCommon;
import pxgSlangArticulationDesc;

// ===========================================================================
// P8.82: the PxgArticulationCoreDesc address map, read back functionally
// ===========================================================================
//
// Until PhysX 5.6.1 this entry shared dmaArtiResidualBda.slang with
// dmaArticulationResidual, which exercised five of the descriptor's offsets and
// no more, so seventeen of the constants P8.82 added would otherwise have been
// carried by `static_assert` alone.  5.12 removed that kernel; the probe moved
// here unchanged (fixture: tests/testArtiCoreDescLayout.cpp).  That is not enough: a static_assert proves the offset
// matches the C++ header, but not that the SHADER reads at the offset it names
// -- a constant used at the wrong call site type-checks perfectly.
//
// This probe closes that by reading every mapped field of the descriptor
// through its named constant and writing the raw bits out in a fixed slot
// order.  The fixture fills each field with a distinct value through the real
// C++ struct member, so a swapped pair of constants (the pitches invite exactly
// that: MAX_MIMIC_JOINTS sits between MAX_DOFS and MAX_SPATIAL_TENDONS) puts
// the wrong number in the slot rather than merely reading a neighbour that
// happens to be zero.  One thread, no index chain: this measures addresses.
struct ArtiCoreDescProbeParams
{
    uint64_t scDesc; // PxgArticulationCoreDesc*
    uint64_t out;    // uint64_t[ACDP_COUNT]
};

[[vk::push_constant]]
ConstantBuffer<ArtiCoreDescProbeParams> gArtiCoreDescProbe;

static const uint ACDP_COUNT = 24;

[shader("compute")]
[numthreads(1, 1, 1)]
void artiCoreDescProbe(uint3 dispatchThreadId : SV_DispatchThreadID)
{
    if (dispatchThreadId.x != 0)
        return;

    const uint64_t d = gArtiCoreDescProbe.scDesc;
    Ptr<uint64_t> o = Ptr<uint64_t>(gArtiCoreDescProbe.out);
    uint i = 0;

    // the seven P8.66 fields, so the probe covers the whole map and not just
    // the part this tranche added
    o[i++] = artiPtrField(d, PXG_ARTI_CORE_BODY_SIM_POOL);
    o[i++] = artiPtrField(d, PXG_ARTI_CORE_SLEEP_DATA);
    o[i++] = artiPtrField(d, PXG_ARTI_CORE_ISLAND_NODE_INDICES);
    o[i++] = artiPtrField(d, PXG_ARTI_CORE_SOLVER_BODY_INDICES);
    o[i++] = artiPtrField(d, PXG_ARTI_CORE_ARTICULATIONS);
    o[i++] = uint64_t(artiU32Field(d, PXG_ARTI_CORE_ARTICULATION_OFFSET));
    o[i++] = uint64_t(artiU32Field(d, PXG_ARTI_CORE_NB_ARTICULATIONS));

    // invLengthScale is a PxReal; the raw bits go out so the comparison is
    // exact and does not depend on a float round-trip
    o[i++] = uint64_t(artiU32Field(d, PXG_ARTI_CORE_INV_LENGTH_SCALE));

    // the nine block-array bases the wrapper offsets per warp
    o[i++] = artiPtrField(d, PXG_ARTI_CORE_ARTICULATION_BLOCKS);
    o[i++] = artiPtrField(d, PXG_ARTI_CORE_LINK_BLOCKS);
    o[i++] = artiPtrField(d, PXG_ARTI_CORE_PATH_TO_ROOT_BITFIELD_BLOCKS);
    o[i++] = artiPtrField(d, PXG_ARTI_CORE_DOF_BLOCKS);
    o[i++] = artiPtrField(d, PXG_ARTI_CORE_SPATIAL_TENDON_BLOCKS);
    o[i++] = artiPtrField(d, PXG_ARTI_CORE_ATTACHMENT_BLOCKS);
    o[i++] = artiPtrField(d, PXG_ARTI_CORE_FIXED_TENDON_BLOCKS);
    o[i++] = artiPtrField(d, PXG_ARTI_CORE_TENDON_JOINT_BLOCKS);
    o[i++] = artiPtrField(d, PXG_ARTI_CORE_MIMIC_JOINT_BLOCKS);

    // the seven per-warp pitches
    o[i++] = uint64_t(artiU32Field(d, PXG_ARTI_CORE_MAX_LINKS));
    o[i++] = uint64_t(artiU32Field(d, PXG_ARTI_CORE_MAX_DOFS));
    o[i++] = uint64_t(artiU32Field(d, PXG_ARTI_CORE_MAX_MIMIC_JOINTS));
    o[i++] = uint64_t(artiU32Field(d, PXG_ARTI_CORE_MAX_SPATIAL_TENDONS));
    o[i++] = uint64_t(artiU32Field(d, PXG_ARTI_CORE_MAX_ATTACHMENTS));
    o[i++] = uint64_t(artiU32Field(d, PXG_ARTI_CORE_MAX_FIXED_TENDONS));
    o[i++] = uint64_t(artiU32Field(d, PXG_ARTI_CORE_MAX_TENDON_JOINTS));
}
