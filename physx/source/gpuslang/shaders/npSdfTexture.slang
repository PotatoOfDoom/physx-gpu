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

// npSdfTexture.slang -- P9.12c, the fixture entry points over pxgSlangSdfTexture.
//
// Two entries, because the tranche has two separable claims:
//   sdfTexProbe  -- the blob walk and initialize(): every decoded field written
//                   back so the host can compare it against the values it wrote
//                   into the blob.  This is what catches a wrong section stride
//                   or a dropped alignment, which would otherwise surface only
//                   as garbage samples with no indication of where they came from.
//   sdfTexSampleAt -- Sample() at a list of fine-grid coordinates, compared
//                   against Gu::SDF::decodeSparse on the host.

import pxgSlangSdfTexture;

// ONE push-constant block for BOTH entry points, and that is not a tidiness
// choice.  Two [[vk::push_constant]] blocks in one module are laid out in one
// shared range, so the second one sits at the first one's SIZE -- and a host that
// pushes each struct at offset 0 hands the second entry point its parameters
// shifted by 24 bytes.  The symptom is not a validation error (each entry point
// references only its own block, so the SPIR-V is legal); it is a garbage device
// address and a segfault with no output at all.
struct SdfParams
{
    uint64_t meshPtr;      // the cooked-mesh blob
    uint64_t outMesh;      // probe: 16 uints, the decoded PxgSdfMesh
    uint64_t outTex;       // probe: 24 uints, the initialised SparseSdfTexture
    uint64_t coords;       // sample: float4 per sample, xyz = FINE-grid coordinate
    uint64_t outValue;     // sample: float per sample
    uint numSamples;
}

[[vk::push_constant]]
ConstantBuffer<SdfParams> gP;

[shader("compute")]
[numthreads(1, 1, 1)]
void sdfTexProbe()
{
    PxgSdfMesh m;
    const bool hasSdf = sdfReadMesh(gP.meshPtr, m);

    Ptr<uint> om = Ptr<uint>(gP.outMesh);
    om[0] = hasSdf ? 1u : 0u;
    om[1] = m.sdfDims.x;
    om[2] = m.sdfDims.y;
    om[3] = m.sdfDims.z;
    om[4] = m.sdfDims.w;
    om[5] = asuint(m.meshLower.x);
    om[6] = asuint(m.meshLower.y);
    om[7] = asuint(m.meshLower.z);
    om[8] = asuint(m.spacing);
    om[9] = m.subgridSize;
    om[10] = asuint(m.subgridsMinSdfValue);
    om[11] = asuint(m.subgridsMaxSdfValue);
    om[12] = m.numStartSlots;
    om[13] = m.texObject;
    om[14] = m.texObjectSparse;
    // The start-slot ARRAY ADDRESS itself is not comparable across runs, but the
    // first slot it points at is -- and that is what proves the walk landed on
    // the right bytes rather than merely on plausible ones.
    om[15] = (hasSdf && m.numStartSlots != 0u) ? Ptr<uint>(m.subgridStarts)[0] : 0xDEADBEEFu;

    const SparseSdfTexture t = sdfTexInitialize(m);
    Ptr<uint> ot = Ptr<uint>(gP.outTex);
    ot[0] = asuint(t.sdfBoxLower.x);
    ot[1] = asuint(t.sdfBoxLower.y);
    ot[2] = asuint(t.sdfBoxLower.z);
    ot[3] = asuint(t.sdfDx);
    ot[4] = asuint(t.sdfBoxHigher.x);
    ot[5] = asuint(t.sdfBoxHigher.y);
    ot[6] = asuint(t.sdfBoxHigher.z);
    ot[7] = asuint(t.invSdfDx);
    ot[8] = t.texture;
    ot[9] = t.textureSubgrids;
    ot[10] = t.coarseSize.x;
    ot[11] = t.coarseSize.y;
    ot[12] = t.coarseSize.z;
    ot[13] = t.subgridSize;
    ot[14] = asuint(t.fineToCoarse);
    ot[15] = t.sdfDims.x;
    ot[16] = t.sdfDims.y;
    ot[17] = t.sdfDims.z;
    ot[18] = t.sdfDims.w;
    ot[19] = asuint(t.subgridsMinSdfValue);
    ot[20] = asuint(t.subgridsSdfValueRange);
    ot[21] = uint(t.subgridStartSlots & 0xFFFFFFFFUL);
    ot[22] = uint(t.subgridStartSlots >> 32);
    ot[23] = 0u;
}

[shader("compute")]
[numthreads(64, 1, 1)]
void sdfTexSampleAt(uint3 dtid : SV_DispatchThreadID)
{
    const uint i = dtid.x;
    if (i >= gP.numSamples)
        return;

    PxgSdfMesh m;
    // The blob walk is repeated PER THREAD, exactly as the CUDA repeats it per
    // block.  It is pure address arithmetic over a read-only blob, so the cost
    // is the point of comparison and not a correctness question.
    if (!sdfReadMesh(gP.meshPtr, m))
    {
        Ptr<float>(gP.outValue)[i] = -1234.5f;
        return;
    }
    const SparseSdfTexture t = sdfTexInitialize(m);
    const float3 f = Ptr<float4>(gP.coords)[i].xyz;
    Ptr<float>(gP.outValue)[i] = sdfTexSample(t, f);
}
