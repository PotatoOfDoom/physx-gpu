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

// npSdfParticle.slang -- P9.12f, the fixture entry over pxgSlangSdfParticle.
//
// One thread per (position, previousPosition) pair.  Everything the trace
// carries is written back; see the oracle note at the head of the module for
// why that is the test rather than a diagnostic.

import pxgSlangCommon;
import pxgSlangSdfTexture;
import pxgSlangSdfCollision;
import pxgSlangSdfParticle;

// 112 bytes.
struct SdfParticleParams
{
    uint64_t meshPtr;        //  0
    uint64_t positions;      //  8   float4, xyz = the PREDICTED position
    uint64_t positionsPrev;  // 16   float4, xyz = the ORIGIN position
    uint64_t outBuf;         // 24   64 floats per pair
    float4 aToBq;            // 32
    float4 scaleRot;         // 48
    float3 aToBp;            // 64
    float cullScale;         // 76
    float3 scale;            // 80
    float contactDist;       // 92
    float restDist;          // 96
    uint numPairs;           // 100
    uint enableCCD;          // 104
    uint pad0;               // 108
}

[[vk::push_constant]]
ConstantBuffer<SdfParticleParams> gP;

static const uint SDFP_STRIDE = 64u;

[shader("compute")]
[numthreads(64, 1, 1)]
void sdfParticleContactAt(uint3 dtid : SV_DispatchThreadID)
{
    const uint i = dtid.x;
    if (i >= gP.numPairs)
        return;

    PxgSdfMesh m;
    if (!sdfReadMesh(gP.meshPtr, m))
        return;
    const SparseSdfTexture tex = sdfTexInitialize(m);

    const float3 posWorld = Ptr<float4>(gP.positions)[i].xyz;
    const float3 posPrevWorld = Ptr<float4>(gP.positionsPrev)[i].xyz;

    float3 normal;
    float finalSep;
    SdfParticleTrace tr;
    const bool candidate = sdfParticleContactEx(tex, posWorld, posPrevWorld,
                                                gP.aToBp, gP.aToBq, gP.scale, gP.scaleRot,
                                                gP.cullScale, gP.contactDist, gP.restDist,
                                                gP.enableCCD != 0u, normal, finalSep, tr);

    Ptr<float> o = Ptr<float>(gP.outBuf) + i * SDFP_STRIDE;
    o[0] = tr.pos.x;        o[1] = tr.pos.y;        o[2] = tr.pos.z;
    o[3] = tr.posPrev.x;    o[4] = tr.posPrev.y;    o[5] = tr.posPrev.z;
    o[6] = tr.vtx.x;        o[7] = tr.vtx.y;        o[8] = tr.vtx.z;
    o[9] = tr.vtxPrev.x;    o[10] = tr.vtxPrev.y;   o[11] = tr.vtxPrev.z;
    o[12] = tr.startDistRaw;
    o[13] = tr.dirPrevRaw.x; o[14] = tr.dirPrevRaw.y; o[15] = tr.dirPrevRaw.z;
    o[16] = tr.distanceRaw;
    o[17] = tr.dirRaw.x;    o[18] = tr.dirRaw.y;    o[19] = tr.dirRaw.z;
    o[20] = asfloat(tr.entered);
    o[21] = tr.m;
    o[22] = tr.startDist;
    o[23] = tr.distance;
    o[24] = tr.dir.x;       o[25] = tr.dir.y;       o[26] = tr.dir.z;
    o[27] = tr.dirPrev.x;   o[28] = tr.dirPrev.y;   o[29] = tr.dirPrev.z;
    o[30] = asfloat(tr.branch);
    o[31] = tr.ccdD.x;      o[32] = tr.ccdD.y;      o[33] = tr.ccdD.z;
    o[34] = tr.ccdDisplacement;
    o[35] = tr.ccdPt.x;     o[36] = tr.ccdPt.y;     o[37] = tr.ccdPt.z;
    o[38] = tr.ccdVtx.x;    o[39] = tr.ccdVtx.y;    o[40] = tr.ccdVtx.z;
    o[41] = tr.ccdDistRaw;
    o[42] = tr.ccdDirRaw.x; o[43] = tr.ccdDirRaw.y; o[44] = tr.ccdDirRaw.z;
    o[45] = tr.ccdDistance;
    o[46] = tr.ccdDir.x;    o[47] = tr.ccdDir.y;    o[48] = tr.ccdDir.z;
    o[49] = normal.x;       o[50] = normal.y;       o[51] = normal.z;
    o[52] = finalSep;
    o[53] = asfloat(candidate ? 1u : 0u);
}
