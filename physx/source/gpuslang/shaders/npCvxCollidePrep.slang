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

// npCvxCollidePrep.slang
//
// Slang port of the collide() driver prep glue (cudaGJKEPA.cu L1294-1352 +
// runGJK's warm-start re-pack tail L796-817, prepareVertices from
// cudaNpCommon.h L315-343):
//   1. prepareVertices: vA[i] = aToB.transform(vertex2Shape(rawA[i], scale0, rot0))
//      (A overload takes the aToB transform), vB[i] = vertex2Shape(rawB[i], scale1,
//      rot1), where vertex2Shape(v) = rot.rotateInv(rot.rotate(v) * scale).
//   2. warm-start unpack/seed: numWarmStartPoints/warmStartAIndices/warmStartBIndices
//      live in lanes 13/14/15 of persistentContactManifoldMoreDataW; for
//      tI < numWarmStartPoints: aIndex = (wsA >> (tI*8)) & 0xFF, bIndex likewise,
//      cachedData.vertices[tI] = CachedVertex(aIndex, bIndex, vA[aIndex]-vB[bIndex])
//      with indices = 0x3f800000 | b<<8 | a; cachedData.size = numWarmStartPoints.
//   3. warm-start re-pack (runGJK tail): warmStartAIndices = OR-reduction over
//      lanes tI < cachedData.size of (a << (tI*8)) (likewise B); lanes 13/14/15
//      of persistentContactManifoldMoreDataW := size / wsA / wsB.
//   4. initialDir = aToB.p if |aToB.p|^2 > 0 else (1,0,0) (lazy-normalized inside
//      GJK, so emitted unnormalized).

float3 quatRotate(float4 q, float3 v)
{
    const float vx = 2.0f * v.x, vy = 2.0f * v.y, vz = 2.0f * v.z;
    const float w2 = q.w * q.w - 0.5f;
    const float d2 = q.x * vx + q.y * vy + q.z * vz;
    return float3(vx * w2 + (q.y * vz - q.z * vy) * q.w + q.x * d2,
                  vy * w2 + (q.z * vx - q.x * vz) * q.w + q.y * d2,
                  vz * w2 + (q.x * vy - q.y * vx) * q.w + q.z * d2);
}
float3 quatRotateInv(float4 q, float3 v)
{
    const float vx = 2.0f * v.x, vy = 2.0f * v.y, vz = 2.0f * v.z;
    const float w2 = q.w * q.w - 0.5f;
    const float d2 = q.x * vx + q.y * vy + q.z * vz;
    return float3(vx * w2 - (q.y * vz - q.z * vy) * q.w + q.x * d2,
                  vy * w2 - (q.z * vx - q.x * vz) * q.w + q.y * d2,
                  vz * w2 - (q.x * vy - q.y * vx) * q.w + q.z * d2);
}

float3 vertex2Shape(float3 v, float3 scale, float4 rot)
{
    return quatRotateInv(rot, quatRotate(rot, v) * scale);
}

struct CollidePrepIn
{
    float4 rawA[32];  // raw hull vertices A (.xyz)
    float4 rawB[32];  // raw hull vertices B (.xyz)
    float4 scale0;    // .xyz; .w = nbVertices0 (as float)
    float4 rot0;
    float4 scale1;    // .xyz; .w = nbVertices1 (as float)
    float4 rot1;
    float4 aToBq;
    float4 aToBp;     // .xyz translation
    uint wsLane13;    // persistentContactManifoldMoreDataW lane 13: numWarmStartPoints
    uint wsLane14;    // lane 14: warmStartAIndices (packed 4x8bit)
    uint wsLane15;    // lane 15: warmStartBIndices
    uint pad0;
}

struct CollidePrepOut
{
    float4 vA[32];       // prepared vertices A (.xyz, .w=0)
    float4 vB[32];       // prepared vertices B (.xyz, .w=0)
    float4 cachedV[4];   // seeded cachedData.vertices[j].v (.xyz), .w = 0
    uint cachedIdx[4];   // seeded cachedData.vertices[j].indices
    uint cachedSize;     // seeded cachedData.size
    uint rePack13;       // re-packed lane 13 (= cachedData.size)
    uint rePack14;       // re-packed lane 14 (warmStartAIndices)
    uint rePack15;       // re-packed lane 15 (warmStartBIndices)
    float4 initialDir;   // .xyz
}

struct CollidePrepParams { uint numTests; }

[[vk::push_constant]]
ConstantBuffer<CollidePrepParams> gParams;

RWStructuredBuffer<CollidePrepIn> gIn;    // binding 0
RWStructuredBuffer<CollidePrepOut> gOut;  // binding 1

[shader("compute")]
[numthreads(64, 1, 1)]
void cvxCollidePrep(uint3 dtid : SV_DispatchThreadID)
{
    const uint t = dtid.x;
    if (t >= gParams.numTests) return;
    const CollidePrepIn s = gIn[t];

    const uint nbV0 = uint(s.scale0.w);
    const uint nbV1 = uint(s.scale1.w);
    const float3 aToBp = s.aToBp.xyz;

    // 1. prepareVertices (A overload applies aToB.transform = q.rotate(v)+p)
    float3 vA[32], vB[32];
    for (uint i = 0u; i < 32u; ++i) { vA[i] = float3(0, 0, 0); vB[i] = float3(0, 0, 0); }
    for (uint i = 0u; i < nbV0; ++i)
        vA[i] = quatRotate(s.aToBq, vertex2Shape(s.rawA[i].xyz, s.scale0.xyz, s.rot0)) + aToBp;
    for (uint i = 0u; i < nbV1; ++i)
        vB[i] = vertex2Shape(s.rawB[i].xyz, s.scale1.xyz, s.rot1);

    // 2. warm-start unpack/seed from lanes 13/14/15
    const uint numWarmStartPoints = s.wsLane13;
    const uint wsA = s.wsLane14;
    const uint wsB = s.wsLane15;

    float3 cachedV[4];
    uint cachedIdx[4];
    for (uint j = 0u; j < 4u; ++j) { cachedV[j] = float3(0, 0, 0); cachedIdx[j] = 0u; }
    for (uint tI = 0u; tI < numWarmStartPoints; ++tI)
    {
        const uint aIndex = (wsA >> (tI << 3u)) & 0xFFu;
        const uint bIndex = (wsB >> (tI << 3u)) & 0xFFu;
        cachedV[tI] = vA[aIndex] - vB[bIndex];
        cachedIdx[tI] = 0x3f800000u | (bIndex << 8u) | aIndex; // encodeIndices(a, b)
    }
    const uint cachedSize = numWarmStartPoints;

    // 3. warm-start re-pack (runGJK tail): OR-reduction over lanes tI < size
    uint rePackA = 0u;
    uint rePackB = 0u;
    for (uint tI = 0u; tI < cachedSize; ++tI)
    {
        const uint a = cachedIdx[tI] & 0xFFu;         // CachedVertex::a()
        const uint b = (cachedIdx[tI] >> 8u) & 0xFFu; // CachedVertex::b()
        rePackA |= (a << (tI << 3u));
        rePackB |= (b << (tI << 3u));
    }

    // 4. initialDir (lazy-normalized inside GJK, emitted raw)
    float3 initialDir = float3(1, 0, 0);
    if (dot(aToBp, aToBp) > 0.0f)
        initialDir = aToBp;

    CollidePrepOut r;
    for (uint i = 0u; i < 32u; ++i) { r.vA[i] = float4(vA[i], 0.0f); r.vB[i] = float4(vB[i], 0.0f); }
    for (uint j = 0u; j < 4u; ++j) { r.cachedV[j] = float4(cachedV[j], 0.0f); r.cachedIdx[j] = cachedIdx[j]; }
    r.cachedSize = cachedSize;
    r.rePack13 = cachedSize;
    r.rePack14 = rePackA;
    r.rePack15 = rePackB;
    r.initialDir = float4(initialDir, 0.0f);
    gOut[t] = r;
}
