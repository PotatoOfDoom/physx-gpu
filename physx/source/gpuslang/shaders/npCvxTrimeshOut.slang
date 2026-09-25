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

// npCvxTrimeshOut.slang
//
// Slang port of convexTrimeshFinishContacts (convexMeshOutput.cu L116-346)
// with writeCompressedContact (L50-82): the multimanifold-to-stream output
// stage of the convex-trimesh pipeline.  One warp per pair, serialized.
// Quirks ported verbatim:
//   - lane layout: singleManifoldIndex = lane / 6, threadIndexInManifold =
//     lane % 6 (lanes 24..31 map to manifold >= 4 and never have contacts);
//   - the reservations happen on lane 0 with CHAINED else-if overflow tests
//     using the FULL reservation sizes (contrast: the convex-plane kernel
//     tests one element with independent ifs); rollback zeroes nbManifolds
//     AND totalNumContacts and rewrites HAS_NO_TOUCH; overflow also NULLs the
//     three output pointers (modeled as the 0xFFFFFFFE sentinel -- a CUDA
//     null pointer, distinct from a legitimate 0 byte offset);
//   - insertAveragePoint adds nbManifolds to totalNumContacts ON LANE 0 ONLY
//     (cu L190-191): the inflated count enters the reservation sizes, the
//     stored output->nbContacts and lane 0's faceIndex base, while lanes
//     1..31 keep the un-inflated popc -- the faceIndex region base
//     (forceOffset + totalNumContacts*4, cu L290) is therefore PER-LANE
//     divergent, and lane 0's face-index slot can collide with another
//     lane's (a genuine unordered intra-warp write conflict in CUDA; the
//     test masks conflicted words from comparison).  The average contacts
//     themselves are NEVER written by this kernel (reserved-but-stale slots;
//     getAvgContactValue at cu L84-98 is dead code);
//   - a pair with manifolds but zero real contacts and no average insertion
//     skips the reservation entirely, leaving the offsets 0xFFFFFFFF, yet
//     still passes the L257 gate (nbManifolds != 0, no overflow) and stores
//     output pointers base + 0xFFFFFFFF (wrapped -- ported exactly); the
//     hasContacts guards then suppress every stream write;
//   - flip = (type of shapeRef0 == eTRIANGLEMESH) swaps shape refs and
//     transform refs, and negates the patch normal;
//   - getWorldNormal (tranche-115 semantics) uses the manifold's FIRST
//     contact normal (not an average), rotated by the trimesh transform and
//     normalized (zero for lanes past numContacts);
//   - per-contact world point: eSPHERE -> sphereTransform.p - n*radius with
//     pen -= radius (radius = scale.y); eCAPSULE -> sphereTransform.
//     transform(pointA) - n*radius with pen -= radius; else trimesh
//     transform.transform(pointB); face index written from point.triIndex;
//   - writeCompressedContact fills the 64-byte PxContactPatch image: mass
//     modification (1,1,1,1), normal+restitution, dynamic/static friction,
//     damping, startContactIndex|nbContacts<<16|materialFlags<<24,
//     internalFlags(eHAS_FACE_INDICES=1)|materialIndex0<<16, materialIndex1;
//     combineMaterials is the tranche-43 port.

static const uint TOUCH_KNOWN = 3u;
static const uint HAS_TOUCH = 2u;
static const uint HAS_NO_TOUCH = 1u;
static const uint PATCH_OVF = 4u, CONTACT_OVF = 1u, FORCE_OVF = 2u;
static const uint NULL_SENTINEL = 0xFFFFFFFEu;
static const uint TYPE_SPHERE = 0u, TYPE_CAPSULE = 2u, TYPE_TRIANGLEMESH = 8u;

struct MaterialData
{
    float dynamicFriction, staticFriction, restitution, damping;
    uint flags, fricMode, restMode, dampMode;
}

struct TmoIn
{
    uint4 hdr;            // x = mNbManifolds, y = allflags word, z = insertAveragePoint, w pad
    uint4 nbC;            // mNbContacts[0..3]
    uint4 shapeInfo;      // x = typeA (shapeRef0 pre-flip), y = typeB, z = materialIndex0, w = materialIndex1
    float4 scaleA;        // shape A scale (y = radius when sphere/capsule)
    float4 scaleB;
    float4 tAq, tAp;      // transformCache[transformCacheRef0 pre-flip]
    float4 tBq, tBp;      // transformCache[transformCacheRef1 pre-flip]
    uint4 cfg;            // x = patchBytesLimit, y = contactBytesLimit, z = forceBytesLimit, w pad
    uint4 counters;       // x = initial patchesBytes, y = contactsBytes, z = forceAndIndiceBytes, w pad
    float4 contacts[72];  // mContacts[4][6]: (pA, pen), (pB, asfloat(triIndex)), (normal, 0)
}

struct TmoOut
{
    uint4 res;            // x = allflags after, y = touchChange, z = patchChange, w = output->nbContacts
    uint4 outputs;        // x = contactForces offset, y = contactPatches offset, z = contactPoints offset (NULL -> 0xFFFFFFFE), w = returnedEarly
    uint4 counters;       // x/y/z = counters after, w = overflow flags
    uint4 patches[16];    // 4 PxContactPatch images (4 uint4 each), zero when unwritten
    float4 cstream[24];   // contact float4 slots relative to contactByteOffset
    uint4 fstream[16];    // 64 words of the forceAndIndice region relative to forceAndIndiceByteOffset
}

struct TmoParams { uint numTests; }

[[vk::push_constant]]
ConstantBuffer<TmoParams> gParams;

RWStructuredBuffer<TmoIn> gIn;          // binding 0
RWStructuredBuffer<TmoOut> gOut;        // binding 1
RWStructuredBuffer<MaterialData> gMat;  // binding 2

float3 quatRotate(float4 q, float3 v)
{
    const float vx = 2.0f * v.x, vy = 2.0f * v.y, vz = 2.0f * v.z;
    const float w2 = q.w * q.w - 0.5f;
    const float d2 = q.x * vx + q.y * vy + q.z * vz;
    return float3(vx * w2 + (q.y * vz - q.z * vy) * q.w + q.x * d2,
                  vy * w2 + (q.z * vx - q.x * vz) * q.w + q.y * d2,
                  vz * w2 + (q.x * vy - q.y * vx) * q.w + q.z * d2);
}

float3 getNormalizedV(float3 v)
{
    const float m = dot(v, v);
    return m > 0.0f ? v * (1.0f / sqrt(m)) : float3(0, 0, 0);
}

float combineScalars(float a, float b, uint mode)
{
    if (mode == 0u) return 0.5f * (a + b);
    if (mode == 1u) return min(a, b);
    if (mode == 2u) return a * b;
    if (mode == 3u) return max(a, b);
    return 0.0f;
}

// tranche-43 combineMaterials port (PxsCombineMaterials)
void combineMaterialsFn(MaterialData m0, MaterialData m1,
                        out uint matFlags, out float staF, out float dynF, out float rest, out float damp)
{
    const float r0 = m0.restitution, r1 = m1.restitution;
    const bool c0 = r0 < 0.0f, c1 = r1 < 0.0f;
    const bool oneC = c0 != c1, bothC = c0 && c1;
    const bool a0 = (m0.flags & 16u) != 0u, a1 = (m1.flags & 16u) != 0u;
    const bool oneA = a0 != a1;

    if (bothC && oneA) rest = a0 ? r0 : r1;
    else
    {
        const uint mode = oneC ? 1u : max(m0.restMode, m1.restMode);
        const float flip = (bothC && mode == 2u) ? -1.0f : 1.0f;
        rest = flip * combineScalars(r0, r1, mode);
    }
    if (bothC && oneA) damp = a0 ? m0.damping : m1.damping;
    else
    {
        const uint mode = oneC ? 3u : max(m0.dampMode, m1.dampMode);
        damp = combineScalars(m0.damping, m1.damping, mode);
    }
    const uint cf = m0.flags | m1.flags;
    if ((cf & 1u) == 0u)
    {
        const uint mode = max(m0.fricMode, m1.fricMode);
        const float d = combineScalars(m0.dynamicFriction, m1.dynamicFriction, mode);
        const float s2 = combineScalars(m0.staticFriction, m1.staticFriction, mode);
        const float fd = max(d, 0.0f);
        dynF = fd; staF = (s2 - fd) >= 0.0f ? s2 : fd;
        matFlags = cf;
    }
    else { matFlags = cf | 2u; dynF = 0.0f; staF = 0.0f; }
}

[shader("compute")]
[numthreads(64, 1, 1)]
void cvxTrimeshOut(uint3 dtid : SV_DispatchThreadID)
{
    const uint t = dtid.x;
    if (t >= gParams.numTests) return;
    const TmoIn s = gIn[t];

    TmoOut r;
    r.res = uint4(0, 0, 0, 0);
    r.outputs = uint4(NULL_SENTINEL, NULL_SENTINEL, NULL_SENTINEL, 0);
    for (uint j = 0; j < 16; ++j) { r.patches[j] = uint4(0, 0, 0, 0); r.fstream[j] = uint4(0, 0, 0, 0); }
    for (uint j = 0; j < 24; ++j) r.cstream[j] = float4(0, 0, 0, 0);

    uint nbManifolds = s.hdr.x;                    // uniform pre-rollback value
    const uint allflags = s.hdr.y;
    const bool insertAveragePoint = s.hdr.z != 0u;

    // per-lane classification
    uint contactMask = 0u;
    for (uint L = 0; L < 32; ++L)
    {
        const uint mIdx = L / 6u;
        const uint tIdx = L % 6u;
        const uint nc = (mIdx >= nbManifolds) ? 0u : s.nbC[min(mIdx, 3u)];
        if (tIdx < nc) contactMask |= 1u << L;
    }
    const uint popcContacts = countbits(contactMask);

    const uint oldStatusFlags = (allflags >> 16u) & 0xFFu;
    uint statusFlags = oldStatusFlags & ~TOUCH_KNOWN;
    if (popcContacts != 0u) statusFlags |= HAS_TOUCH;
    else statusFlags |= HAS_NO_TOUCH;
    const uint prevPatches = (allflags & 0xFFFFu) >> 8u;

    uint patchesBytes = s.counters.x, contactsBytes = s.counters.y, forceBytes = s.counters.z;
    uint ovf = 0u;

    // ---- lane 0 block (cu L188-255) ----
    uint patchByteOffset = 0xFFFFFFFFu;
    uint contactByteOffset = 0xFFFFFFFFu;
    uint forceByteOffset = 0xFFFFFFFFu;
    bool overflow = false;
    uint totalNumContactsL0 = popcContacts;        // lane 0's private copy
    uint nbManifoldsL0 = nbManifolds;              // lane 0's private copy
    {
        const uint nbInsertAveragePoint = insertAveragePoint ? nbManifoldsL0 : 0u;
        totalNumContactsL0 = totalNumContactsL0 + nbInsertAveragePoint;

        if (totalNumContactsL0 != 0u)
        {
            patchByteOffset = patchesBytes;   patchesBytes += 64u * nbManifoldsL0;
            contactByteOffset = contactsBytes; contactsBytes += 16u * totalNumContactsL0;
            forceByteOffset = forceBytes;     forceBytes += 4u * totalNumContactsL0 * 2u;

            if ((uint64_t(patchByteOffset) + 64ULL * nbManifoldsL0) > uint64_t(s.cfg.x))
            {
                ovf |= PATCH_OVF; patchByteOffset = 0xFFFFFFFFu; overflow = true;
            }
            else if ((uint64_t(contactByteOffset) + 16ULL * totalNumContactsL0) > uint64_t(s.cfg.y))
            {
                ovf |= CONTACT_OVF; contactByteOffset = 0xFFFFFFFFu; overflow = true;
            }
            else if ((uint64_t(forceByteOffset) + 4ULL * totalNumContactsL0 * 2u) > uint64_t(s.cfg.z))
            {
                ovf |= FORCE_OVF; forceByteOffset = 0xFFFFFFFFu; overflow = true;
            }

            if (overflow)
            {
                nbManifoldsL0 = 0u; totalNumContactsL0 = 0u;
                statusFlags = (statusFlags & ~TOUCH_KNOWN) | HAS_NO_TOUCH;
            }
        }

        const bool previouslyHadTouch = (oldStatusFlags & HAS_TOUCH) != 0u;
        const bool prevTouchKnown = (oldStatusFlags & TOUCH_KNOWN) != 0u;
        const bool currentlyHasTouch = nbManifoldsL0 != 0u;
        r.res.y = ((previouslyHadTouch != currentlyHasTouch) || !prevTouchKnown) ? 1u : 0u;
        r.res.z = (prevPatches != nbManifoldsL0) ? 1u : 0u;
        r.res.x = (((prevPatches << 8u) | statusFlags) << 16u) | (nbManifoldsL0 << 8u);
        r.res.w = totalNumContactsL0;

        if (!overflow)
        {
            r.outputs.x = forceByteOffset;   // may be the wrapped 0xFFFFFFFF -- ported exactly
            r.outputs.y = patchByteOffset;
            r.outputs.z = contactByteOffset;
        }
        else
        {
            r.outputs.x = NULL_SENTINEL;     // CUDA stores literal null pointers
            r.outputs.y = NULL_SENTINEL;
            r.outputs.z = NULL_SENTINEL;
        }
    }
    r.counters = uint4(patchesBytes, contactsBytes, forceBytes, ovf);

    // ---- warp-wide early return (cu L257): any(overflow) || nbManifolds == 0
    // (nbManifolds was zeroed on lane 0 only; the uniform pre-rollback value
    // gates the other lanes, but any(overflow) already covers the rollback)
    if (overflow || nbManifolds == 0u)
    {
        r.outputs.w = 1u;
        gOut[t] = r;
        return;
    }

    // ---- post-gate body ----
    const bool flip = s.shapeInfo.x == TYPE_TRIANGLEMESH;
    const uint shape0Type = flip ? s.shapeInfo.y : s.shapeInfo.x;
    const float4 scale0 = flip ? s.scaleB : s.scaleA;
    // transforms: ref1 (post-flip) is the trimesh, ref0 the convex/sphere
    const float4 triQ = flip ? s.tAq : s.tBq;
    const float3 triP = (flip ? s.tAp : s.tBp).xyz;
    const float4 sphQ = flip ? s.tBq : s.tAq;
    const float3 sphP = (flip ? s.tBp : s.tAp).xyz;
    const uint materialIndex0 = s.shapeInfo.z, materialIndex1 = s.shapeInfo.w;

    // offsets shfl'd from lane 0 (cu L284-286)
    // totalNumContacts is NOT shfl'd: lane 0 keeps the inflated value, all
    // other lanes the plain popc (cu L191 vs L290)

    for (uint L = 0; L < 32; ++L)
    {
        const uint mIdx = L / 6u;
        const uint tIdx = L % 6u;
        const uint nc = (mIdx >= nbManifolds) ? 0u : s.nbC[min(mIdx, 3u)];
        const bool hasContacts = (contactMask & (1u << L)) != 0u;
        const uint writeIndex = countbits(contactMask & ((1u << L) - 1u));
        const uint totalLocal = (L == 0u) ? totalNumContactsL0 : popcContacts;

        // getWorldNormal: FIRST contact normal of this lane's manifold
        float3 nrm = float3(0, 0, 0);
        if (tIdx < nc && mIdx < 4u)
            nrm = s.contacts[(mIdx * 6u + 0u) * 3u + 2u].xyz;
        const float3 worldNormal = getNormalizedV(quatRotate(triQ, nrm));

        if (hasContacts)
        {
            const uint cBase = (mIdx * 6u + tIdx) * 3u;
            const float3 pointA = s.contacts[cBase + 0].xyz;
            float pen = s.contacts[cBase + 0].w;
            const float3 pointB = s.contacts[cBase + 1].xyz;
            const uint triIndex = asuint(s.contacts[cBase + 1].w);

            if (tIdx == 0u)
            {
                // writeCompressedContact (cu L50-82)
                uint matFlags; float staF, dynF, rest, damp;
                combineMaterialsFn(gMat[materialIndex0 & 0xFFFFu], gMat[materialIndex1 & 0xFFFFu], matFlags, staF, dynF, rest, damp);   // P9.88: cu:61 PxU16
                const float3 pn = flip ? -worldNormal : worldNormal;
                const uint pb = mIdx * 4u;
                r.patches[pb + 0] = uint4(asuint(1.0f), asuint(1.0f), asuint(1.0f), asuint(1.0f));
                r.patches[pb + 1] = uint4(asuint(pn.x), asuint(pn.y), asuint(pn.z), asuint(rest));
                r.patches[pb + 2] = uint4(asuint(dynF), asuint(staF), asuint(damp),
                                          (writeIndex & 0xFFFFu) | ((nc & 0xFFu) << 16u) | ((matFlags & 0xFFu) << 24u));
                r.patches[pb + 3] = uint4(1u /*eHAS_FACE_INDICES*/ | ((materialIndex0 & 0xFFFFu) << 16u),
                                          materialIndex1 & 0xFFFFu, 0u, 0u);
            }

            float3 worldPt;
            if (shape0Type == TYPE_SPHERE)
            {
                const float radius = scale0.y;
                pen = pen - radius;
                worldPt = sphP - worldNormal * radius;
            }
            else if (shape0Type == TYPE_CAPSULE)
            {
                const float radius = scale0.y;
                pen = pen - radius;
                worldPt = quatRotate(sphQ, pointA) + sphP - worldNormal * radius;
            }
            else
            {
                worldPt = quatRotate(triQ, pointB) + triP;
            }

            if (contactByteOffset != 0xFFFFFFFFu)
                r.cstream[writeIndex] = float4(worldPt, pen);

            if (forceByteOffset != 0xFFFFFFFFu)
            {
                // faceIndex = (forceOffset + totalLocal*4 bytes) as PxU32*, [writeIndex]
                const uint word = totalLocal + writeIndex;
                r.fstream[word / 4u][word & 3u] = triIndex;
            }
        }
    }

    gOut[t] = r;
}
