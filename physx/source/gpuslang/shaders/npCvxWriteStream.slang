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

// npCvxWriteStream.slang
//
// Slang port of one iteration of writeContactsToStream4threads (cudaGJKEPA.cu
// L1622-1875): 4 threads per pair, 8 pairs per warp iteration.  Per pair:
//   - shape load + flip = (type1 < type0), swapping types/scales/cacheRefs but
//     NOT the material indices; radius = scale0.y (post-flip) for sphere/capsule;
//     t1 = transformCache[cacheRef1 post-flip]; contactDist = cd[ref0] + cd[ref1]
//     (flip-invariant sum);
//   - manifold mNbContacts: clear the 0x80000000 invalidation bit in memory,
//     treat the count as 0 when the bit was set;
//   - group average normal: lane j holds normPen[j] (j < num, else 0); the
//     shfl_xor(2) + shfl_xor(1) butterfly gives (n0+n2)+(n1+n3); normalized by
//     1/sqrt when magSq >= 1e-5 else the STORING LANE's own normal (tidInGroup
//     0 stores, so the fallback is normPen[0].xyz); then t1.rotate, negated on
//     flip;
//   - world pointB = t1.transform(local B), .w = normPen.w - radius; kept iff
//     .w <= contactDist (lanes >= num hold FLT_MAX -> never kept);
//   - extra ballot: tidInGroup==0 && insertAveragePoint && numKept > 1;
//   - warp lane 0: totalContacts = popc(keep)+popc(extra); if nonzero,
//     atomically reserves 16*total contact bytes and 4*total force bytes; on
//     (offset + size*total) >= limit sets the overflow error flag and turns
//     that offset into 0xFFFFFFFF;
//   - per-lane compaction offset popc((keep|extra) & belowMask); the group's
//     average point (butterfly sum / numKept) is written FIRST by tidInGroup 0,
//     then each kept lane's pointB; when the contact reservation failed every
//     lane's numKept is forced to 0;
//   - tidInGroup 0: when numKept == 0 both output offsets become 0xFFFFFFFF;
//     cmOutputs.contactPoints/.contactForces get the (adjusted) byte offsets
//     (the force offset wraps in uint arithmetic when the force reservation
//     failed but contacts did not -- ported exactly); scratch
//     contactParams.nbContacts = numKept + (numKept > 1 && insertAveragePoint),
//     materialParams keep the UNSWAPPED material indices.
//
// One test = one warp iteration (8 pairs, 32 lanes serialized in one thread).
// The atomic counters are per-test slots (single warp -> serial reservation).
// Stream bytes are modelled as float4 slots relative to the initial counter.

static const float FLT_MAXV = 3.4028235e38f;
static const uint EGT_SPHERE = 0u;   // PxGeometryType::eSPHERE
static const uint EGT_CAPSULE = 2u;  // PxGeometryType::eCAPSULE (order: sphere, plane, capsule)
static const uint CONTACT_BUFFER_OVERFLOW = 1u;
static const uint FORCE_BUFFER_OVERFLOW = 2u;

float3 quatRotate(float4 q, float3 v)
{
    const float vx = 2.0f * v.x, vy = 2.0f * v.y, vz = 2.0f * v.z;
    const float w2 = q.w * q.w - 0.5f;
    const float d2 = q.x * vx + q.y * vy + q.z * vz;
    return float3(vx * w2 + (q.y * vz - q.z * vy) * q.w + q.x * d2,
                  vy * w2 + (q.z * vx - q.x * vz) * q.w + q.y * d2,
                  vz * w2 + (q.x * vy - q.y * vx) * q.w + q.z * d2);
}
float3 transformPoint(float4 q, float3 p, float3 v) { return quatRotate(q, v) + p; }

struct WriteStreamPair
{
    uint4 meta;        // x=type0 y=materialIndex0 z=type1 w=materialIndex1
    float4 scale0;     // .xyz
    float4 scale1;     // .xyz
    float4 tq[2];      // transformCache[i].transform.q
    float4 tp[2];      // transformCache[i].transform.p; .w = contactDistance[i]
    float4 normPen[4]; // manifold normal+pen
    float4 pointB[4];  // manifold local B points
    uint4 misc;        // x=raw mNbContacts (may carry 0x80000000) y=valid z,w pad
}

struct WriteStreamIn
{
    WriteStreamPair pairs[8];
    uint4 cfg;    // x=insertAveragePoint y=contactBytesLimit z=forceBytesLimit w pad
    uint4 counters; // x=initial contactsBytes y=initial forceAndIndiceBytes z,w pad
}

struct WriteStreamPairOut
{
    float4 avgNorm;   // scratch normx/y/z (.w = 0)
    uint4 outA;       // x=nbContacts (scratch contactParams) y=cmOut contactPoints offset
                      // z=cmOut contactForces offset w=mNbContacts after clear
    uint4 material;   // x=materialIndex0 y=materialIndex1 (scratch) z,w pad
}

struct WriteStreamOut
{
    float4 stream[40];        // contact stream slots relative to initial contactsBytes (max 32 kept + 8 average)
    uint4 counters;           // x=contactsBytes after y=forceAndIndiceBytes after z=overflow flags w pad
    WriteStreamPairOut pairs[8];
}

struct WriteStreamParams { uint numTests; }

[[vk::push_constant]]
ConstantBuffer<WriteStreamParams> gParams;

RWStructuredBuffer<WriteStreamIn> gIn;    // binding 0
RWStructuredBuffer<WriteStreamOut> gOut;  // binding 1

[shader("compute")]
[numthreads(64, 1, 1)]
void cvxWriteStream(uint3 dtid : SV_DispatchThreadID)
{
    const uint t = dtid.x;
    if (t >= gParams.numTests) return;
    const WriteStreamIn s = gIn[t];

    const bool insertAveragePoint = s.cfg.x != 0u;
    const uint contactBytesLimit = s.cfg.y;
    const uint forceBytesLimit = s.cfg.z;
    uint contactsBytes = s.counters.x;
    uint forceBytes = s.counters.y;
    uint overflowFlags = 0u;

    WriteStreamOut r;
    for (uint i = 0u; i < 40u; ++i) r.stream[i] = float4(0, 0, 0, 0);

    // ------------------------------------------------------------------
    // per-pair setup (the CUDA per-lane loads, serialized)
    // ------------------------------------------------------------------
    bool valid[8];
    bool flip[8];
    float radius[8];
    float contactDist[8];
    float4 t1q[8]; float3 t1p[8];
    uint num0[8];       // manifold count after invalidation handling
    uint mNbAfter[8];
    uint mat0[8], mat1[8];

    for (uint g = 0u; g < 8u; ++g)
    {
        valid[g] = s.pairs[g].misc.y != 0u;
        flip[g] = false;
        radius[g] = 0.0f;
        contactDist[g] = 0.0f;
        t1q[g] = float4(0, 0, 0, 1); t1p[g] = float3(0, 0, 0);
        mat0[g] = 0u; mat1[g] = 0u;

        if (valid[g])
        {
            uint type0 = s.pairs[g].meta.x;
            uint type1 = s.pairs[g].meta.z;
            mat0[g] = s.pairs[g].meta.y;
            mat1[g] = s.pairs[g].meta.w;
            float3 scale0 = s.pairs[g].scale0.xyz;
            float3 scale1 = s.pairs[g].scale1.xyz;
            uint ref0 = 0u, ref1 = 1u;

            flip[g] = type1 < type0;
            if (flip[g])
            {
                const uint tt = type0; type0 = type1; type1 = tt;
                const float3 ts = scale0; scale0 = scale1; scale1 = ts;
                const uint tr = ref0; ref0 = ref1; ref1 = tr;
            }

            if (type0 == EGT_SPHERE || type0 == EGT_CAPSULE)
                radius[g] = scale0.y;

            t1q[g] = s.pairs[g].tq[ref1];
            t1p[g] = s.pairs[g].tp[ref1].xyz;
            contactDist[g] = s.pairs[g].tp[ref0].w + s.pairs[g].tp[ref1].w;
        }

        // manifold count + invalidation-bit clear (all 4 lanes write the same value)
        num0[g] = 0u;
        mNbAfter[g] = s.pairs[g].misc.x;
        if (valid[g])
        {
            uint raw = s.pairs[g].misc.x;
            mNbAfter[g] = raw & ~0x80000000u;
            num0[g] = ((raw & 0x80000000u) != 0u) ? 0u : (raw & ~0x80000000u);
        }
    }

    // ------------------------------------------------------------------
    // per-lane normPen / pointB, group average normal
    // ------------------------------------------------------------------
    float4 normPenL[32];
    float4 pointBL[32];
    for (uint L = 0u; L < 32u; ++L)
    {
        const uint g = L / 4u, j = L % 4u;
        normPenL[L] = (j < num0[g]) ? s.pairs[g].normPen[j] : float4(0, 0, 0, 0);

        float4 pb = float4(FLT_MAXV, FLT_MAXV, FLT_MAXV, FLT_MAXV);
        if (j < num0[g])
        {
            const float sep = normPenL[L].w - radius[g];
            const float3 p = transformPoint(t1q[g], t1p[g], s.pairs[g].pointB[j].xyz);
            pb = float4(p, sep);
        }
        pointBL[L] = pb;
    }

    for (uint g = 0u; g < 8u; ++g)
    {
        // butterfly: (n0 + n2) + (n1 + n3)
        float3 sum = (normPenL[4u * g].xyz + normPenL[4u * g + 2u].xyz)
                   + (normPenL[4u * g + 1u].xyz + normPenL[4u * g + 3u].xyz);
        const float magSq = dot(sum, sum);
        float3 avgNorm;
        if (magSq >= 1e-5f)
            avgNorm = sum * (1.0f / sqrt(magSq)); // PxVec3::operator/= is reciprocal-multiply
        else
            avgNorm = normPenL[4u * g].xyz; // storing lane's own normal (tidInGroup 0)
        avgNorm = quatRotate(t1q[g], avgNorm);
        if (flip[g]) avgNorm = -avgNorm;
        r.pairs[g].avgNorm = float4(avgNorm, 0.0f);
    }

    // ------------------------------------------------------------------
    // keep / extra ballots, warp-wide reservation
    // ------------------------------------------------------------------
    uint keepContactW = 0u;
    for (uint L = 0u; L < 32u; ++L)
        if (pointBL[L].w <= contactDist[L / 4u]) keepContactW |= (1u << L);

    uint numKept[8];
    for (uint g = 0u; g < 8u; ++g)
        numKept[g] = countbits(keepContactW & (0xFu << (4u * g)));

    uint extraContactsW = 0u;
    for (uint g = 0u; g < 8u; ++g)
        if (insertAveragePoint && numKept[g] > 1u) extraContactsW |= (1u << (4u * g));

    uint contactByteOffset0 = 0xFFFFFFFFu;
    uint forceByteOffset0 = 0xFFFFFFFFu;
    {
        const uint totalContacts = countbits(keepContactW) + countbits(extraContactsW);
        if (totalContacts != 0u)
        {
            contactByteOffset0 = contactsBytes;
            contactsBytes += 16u * totalContacts; // sizeof(PxContact)
            forceByteOffset0 = forceBytes;
            forceBytes += 4u * totalContacts;     // sizeof(PxU32)

            if ((contactByteOffset0 + 16u * totalContacts) >= contactBytesLimit)
            {
                overflowFlags |= CONTACT_BUFFER_OVERFLOW;
                contactByteOffset0 = 0xFFFFFFFFu;
            }
            else if ((forceByteOffset0 + 4u * totalContacts) >= forceBytesLimit)
            {
                overflowFlags |= FORCE_BUFFER_OVERFLOW;
                forceByteOffset0 = 0xFFFFFFFFu;
                contactByteOffset0 = 0xFFFFFFFFu; // 5.12 (cudaGJKEPA.cu:1780)
            }
        }
    }

    // ------------------------------------------------------------------
    // per-lane compacted writes + per-pair outputs
    // ------------------------------------------------------------------
    const uint streamBase = s.counters.x;

    for (uint g = 0u; g < 8u; ++g)
    {
        uint numKeptG = numKept[g];
        uint cbLane0 = contactByteOffset0; // tidInGroup 0's adjusted offsets
        uint fbLane0 = forceByteOffset0;

        if (contactByteOffset0 != 0xFFFFFFFFu)
        {
            // group average point: CUDA sums lanes tidInGroup < numManifoldContacts,
            // where numManifoldContacts is the RE-derived kept count -- a lane-INDEX
            // test, not the kept bitmask, so an unkept low lane can contribute while
            // a kept high lane does not (ported exactly).  The shfl_xor(2)+shfl_xor(1)
            // butterfly associates (p0+p2)+(p1+p3).
            float4 pv[4];
            for (uint j = 0u; j < 4u; ++j)
                pv[j] = (j < numKeptG) ? pointBL[4u * g + j] : float4(0, 0, 0, 0);
            const float4 avgPt = (pv[0] + pv[2]) + (pv[1] + pv[3]);

            for (uint j = 0u; j < 4u; ++j)
            {
                const uint L = 4u * g + j;
                const uint mask = (1u << L) - 1u;
                const uint byteOffset = countbits(keepContactW & mask) + countbits(extraContactsW & mask);
                const uint cb = contactByteOffset0 + 16u * byteOffset;
                const uint fb = forceByteOffset0 + 4u * byteOffset;
                if (j == 0u) { cbLane0 = cb; fbLane0 = fb; }

                uint offset = 0u;
                if (insertAveragePoint && numKeptG > 1u)
                {
                    if (j == 0u)
                    {
                        // cutil operator/=(float4, float) is reciprocal-multiply
                        const float4 a = avgPt * (1.0f / float(numKeptG));
                        r.stream[(cb - streamBase) / 16u] = a; // avgPoint first
                        offset += 16u;
                    }
                }
                if (pointBL[L].w <= contactDist[g])
                    r.stream[(cb + offset - streamBase) / 16u] = pointBL[L];
            }
        }
        else
        {
            numKeptG = 0u;
        }

        // tidInGroup == 0 epilogue
        uint cbOut = cbLane0;
        uint fbOut = fbLane0;
        if (numKeptG == 0u)
        {
            cbOut = 0xFFFFFFFFu;
            fbOut = 0xFFFFFFFFu;
        }
        r.pairs[g].outA = uint4(0u, 0u, 0u, mNbAfter[g]);
        if (valid[g])
        {
            r.pairs[g].outA.y = cbOut;
            r.pairs[g].outA.z = fbOut;
        }
        r.pairs[g].outA.x = numKeptG + ((numKeptG > 1u && insertAveragePoint) ? 1u : 0u);
        r.pairs[g].material = uint4(mat0[g], mat1[g], 0u, 0u);
    }

    r.counters = uint4(contactsBytes, forceBytes, overflowFlags, 0u);
    gOut[t] = r;
}
