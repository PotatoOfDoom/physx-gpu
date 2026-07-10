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

// psHfCollide.slang
//
// Slang port of psHeightfieldMidphaseCollision — the particle-vs-heightfield
// midphase/contact core (gpunarrowphase/src/CUDA/particleSystemHFMidPhaseCG.cu),
// one PARTICLE per thread (no warp coupling):
//
// - the speculative CCD contact volume or the plain contact distance; the
//   world AABB pos +- radius transformed into heightfield space
//   (transformFast of the inverse pose) and divided per axis by |rowScale|,
//   heightScale (NOT abs — faithful) and |columnScale|, with min/max x resp.
//   z SWAPPED for negative row/column scales (transcribed faithfully, but
//   DEAD upstream: PxHeightFieldGeometry::isValid() rejects scales below
//   PX_MIN_HEIGHTFIELD_XZ_SCALE / _Y_SCALE, and for a hypothetical negative
//   scale the |scale| division followed by a bare swap leaves min > max, so
//   maxRow - minRow underflows — the branch is defensive, not functional);
// - the grid overlap early-outs (min.x > nbRows-1 etc., zero cell span);
// - the cell loop over (rows x columns x 2 materials): per cell the four
//   corner heights cull against the local bound's y range (con0 = maxy
//   below all four, con1 = miny above all four), holes
//   (PxHeightFieldMaterial::eHOLE = 127) skip, triangleIdx =
//   (vertexIndex << 1) | materialBit, getTriangle (heightfieldUtil.cuh:
//   tessellation-flag diagonal + negative-scale HANDEDNESS vertex swap +
//   hf2shapep scaling), then particleTriangleTest in heightfield shape
//   space with the normal rotated to world;
// - contacts accumulate into a 6-slot temp list (MaxStaticContactsPerMesh;
//   NO worst-contact replacement — the CUDA comments this as missing
//   upstream, faithfully preserved);
// - finalize QUIRK preserved literally: the static path atomicAdds the
//   one-way count, copies capped at MaxStaticContactsPerParticle = 12, then
//   PLAIN-STORES the count back to min(contactCounts, 12) — overwriting
//   whatever the atomicAdd accumulated.  With several heightfield meshes
//   (one kernel invocation each accumulating into the shared one-way
//   buffers) the LAST mesh's count wins: earlier meshes' slots stay
//   written but are no longer counted.  Because contactCounts <= 6 and the
//   plain store caps the next dispatch's base at 6, the endIndex
//   truncation at 12 never binds and the static remainder is always 0 —
//   the writer path below only ever fires for a DYNAMIC rigid (where
//   contactWritten stays 0 and src = 0 correctly emits the whole temp
//   list); it is suppressed for diffuse threads AFTER no reservation
//   (remaining = 0 short-circuits).
//
// Engine glue not ported (documented): LoadShapePair/transform-cache reads,
// the per-pair particle-system resolution and grid-stride particle batching
// (params carry poses, scales, flags, counts directly), and the driver's
// per-particle rigid filter early-out.

import pxgSlangCommon;

static const float PX_MAX_F32 = 3.4028235e38f;
static const uint MAX_MESH_CONTACTS = 6;      // PxgParticleContactInfo::MaxStaticContactsPerMesh
static const uint MAX_PARTICLE_CONTACTS = 12; // ...::MaxStaticContactsPerParticle
static const uint HOLE = 127;                 // PxHeightFieldMaterial::eHOLE
static const float PARTICLE_FORWARD_PROJECTION_STEP_SCALE_PGS = 1.0f;
static const float PARTICLE_FORWARD_PROJECTION_STEP_SCALE_TGS = 0.5f;
static const float PARTICLE_FORWARD_PROJECTION_STEP_SCALE_DIFFUSE = 1.0f;

struct PhParams
{
    uint numParticles;
    uint maxContacts; // writer capacity
    uint flags;       // 1 = enableCCD, 2 = isTGS, 4 = isDiffuse, 8 = rigid is static
    uint particleSystemId;
}

struct PhMesh
{
    float4 hfP;    // heightfield pose position
    float4 hfQ;    // heightfield pose quaternion
    float4 scale;  // x = rowScale, y = heightScale, z = columnScale, w = contactDist
    float4 misc;   // x = restDist
    uint4 dims;    // x = nbRows, y = nbCols
    uint4 rigid;   // xy = rigidId (u64)
}

// PxgParticlePrimitiveContact (32 bytes)
struct PsContact
{
    float4 normal_pen;
    uint2 rigidId;
    uint2 particleId;
}

[[vk::push_constant]]
ConstantBuffer<PhParams> gParams;

RWStructuredBuffer<PhMesh> gMesh;              // binding 0 (single element)
RWStructuredBuffer<uint> gSamples;             // binding 1 (packed PxHeightFieldSample)
RWStructuredBuffer<float4> gCurrentPos;        // binding 2 (world)
RWStructuredBuffer<float4> gPredictedPos;      // binding 3 (world)
RWStructuredBuffer<float4> gOneWayInfos;       // binding 4 (strided)
RWStructuredBuffer<uint2> gOneWayNodeIndices;  // binding 5 (strided)
RWStructuredBuffer<uint> gOneWayCounts;        // binding 6 (per particle)
RWStructuredBuffer<PsContact> gContacts;       // binding 7 (writer)
RWStructuredBuffer<uint2> gContactByRigid;     // binding 8
RWStructuredBuffer<uint> gTempContactByRigid;  // binding 9
RWStructuredBuffer<uint> gContactIdxByRigid;   // binding 10
RWStructuredBuffer<uint2> gContactSortedByParticle; // binding 11
RWStructuredBuffer<uint> gTempContactByParticle;    // binding 12
RWStructuredBuffer<uint> gContactIdxByParticle;     // binding 13
RWStructuredBuffer<uint> gContactCount;        // binding 14

// ---- heightfieldUtil.cuh helpers (npHeightfield/npHfPointQuery ports) -----
float getHeight(uint vertexIndex)
{
    const int h = int(gSamples[vertexIndex] << 16) >> 16;
    return float(h);
}

bool isZerothVertexShared(uint vertexIndex)
{
    return (gSamples[vertexIndex] & 0x800000u) != 0u;
}

// both material bytes are PxBitAndByte — the PxU8 read strips the high bit
// (materialIndex0's bit is the tessellation flag, materialIndex1's is a
// reserved flag), so both masks are 7-bit
uint materialIndex0(uint vertexIndex) { return (gSamples[vertexIndex] >> 16) & 0x7fu; }
uint materialIndex1(uint vertexIndex) { return (gSamples[vertexIndex] >> 24) & 0x7fu; }

float3 getVertex(uint vertexIndex, uint columns)
{
    const uint row = vertexIndex / columns;
    const uint col = vertexIndex % columns;
    return float3(float(row), getHeight(vertexIndex), float(col));
}

float3 hf2shapep(float3 v, float rowScale, float heightScale, float columnScale)
{
    return float3(v.x * rowScale, v.y * heightScale, v.z * columnScale);
}

void triVertexIndices(uint triangleIndex, uint columns, out uint i0, out uint i1, out uint i2)
{
    const uint cell = triangleIndex >> 1;
    const bool firstTri = (triangleIndex & 0x1u) == 0u;
    if (isZerothVertexShared(cell))
    {
        if (firstTri) { i0 = cell + columns; i1 = cell; i2 = cell + columns + 1u; }
        else          { i0 = cell + 1u; i1 = cell + columns + 1u; i2 = cell; }
    }
    else
    {
        if (firstTri) { i0 = cell; i1 = cell + 1u; i2 = cell + columns; }
        else          { i0 = cell + columns + 1u; i1 = cell + columns; i2 = cell + 1u; }
    }
}

// getTriangle: tessellation diagonal + negative-scale handedness swap
void hfGetTriangle(uint triangleIndex, float3 scale, uint columns,
                   out float3 v0, out float3 v1, out float3 v2)
{
    const float rowScale = scale.x;
    const float heightScale = scale.y;
    const float columnScale = scale.z;
    bool wrongHanded = false;
    if (columnScale < 0.0f) wrongHanded = !wrongHanded;
    if (rowScale < 0.0f) wrongHanded = !wrongHanded;

    uint t0, t1, t2;
    triVertexIndices(triangleIndex, columns, t0, t1, t2);

    const float3 vertex0 = getVertex(t0, columns);
    const float3 vertex1 = getVertex(wrongHanded ? t2 : t1, columns);
    const float3 vertex2 = getVertex(wrongHanded ? t1 : t2, columns);

    v0 = hf2shapep(vertex0, rowScale, heightScale, columnScale);
    v1 = hf2shapep(vertex1, rowScale, heightScale, columnScale);
    v2 = hf2shapep(vertex2, rowScale, heightScale, columnScale);
}

uint getMinRow(float x, uint rows)    { return uint(clamp(int(floor(x)), 0, int(rows) - 2)); }
uint getMaxRow(float x, uint rows)    { return uint(clamp(int(ceil(x)), 0, int(rows) - 1)); }
uint getMinColumn(float z, uint cols) { return uint(clamp(int(floor(z)), 0, int(cols) - 2)); }
uint getMaxColumn(float z, uint cols) { return uint(clamp(int(ceil(z)), 0, int(cols) - 1)); }

// ---- CCD volume + triangle test (tranche 87/92 copies) --------------------
float getParticleSpeculativeContactVolume(out float3 cVolumePos,
    float3 currentPos, float3 predictedPos, float contactDist, bool isDiffuse, bool isTGS)
{
    const float sScaleInvPGS = 1.0f / PARTICLE_FORWARD_PROJECTION_STEP_SCALE_PGS;
    const float sScaleInvTGS = 1.0f / PARTICLE_FORWARD_PROJECTION_STEP_SCALE_TGS;
    const float sScaleInvDiffuse = 1.0f / PARTICLE_FORWARD_PROJECTION_STEP_SCALE_DIFFUSE;
    float scaleInv;
    if (isDiffuse)
        scaleInv = sScaleInvDiffuse;
    else
        scaleInv = isTGS ? sScaleInvTGS : sScaleInvPGS;
    const float3 cVolumeOffset = (predictedPos - currentPos) * scaleInv * 0.5f;
    cVolumePos = currentPos + cVolumeOffset;
    return length(cVolumeOffset) + contactDist;
}

float3 closestPtPointTriangle2(float3 p, float3 a, float3 b, float3 c, float3 ab, float3 ac)
{
    const float3 ap = p - a;
    const float d1 = dot(ab, ap);
    const float d2 = dot(ac, ap);
    if (d1 <= 0.0f && d2 <= 0.0f) return a;
    const float3 bp = p - b;
    const float d3 = dot(ab, bp);
    const float d4 = dot(ac, bp);
    if (d3 >= 0.0f && d4 <= d3) return b;
    const float vc = d1 * d4 - d3 * d2;
    if (vc <= 0.0f && d1 >= 0.0f && d3 <= 0.0f) { const float v = d1 / (d1 - d3); return a + v * ab; }
    const float3 cp = p - c;
    const float d5 = dot(ab, cp);
    const float d6 = dot(ac, cp);
    if (d6 >= 0.0f && d5 <= d6) return c;
    const float vb = d5 * d2 - d1 * d6;
    if (vb <= 0.0f && d2 >= 0.0f && d6 <= 0.0f) { const float w = d2 / (d2 - d6); return a + w * ac; }
    const float va = d3 * d6 - d5 * d4;
    if (va <= 0.0f && (d4 - d3) >= 0.0f && (d5 - d6) >= 0.0f)
    { const float w = (d4 - d3) / ((d4 - d3) + (d5 - d6)); return b + w * (c - b); }
    const float denom = 1.0f / (va + vb + vc);
    return a + ab * (vb * denom) + ac * (vc * denom);
}

float distancePointTriangleSquared(float3 p, float3 a, float3 b, float3 c, out float3 closestP)
{
    const float3 ab = b - a;
    const float3 ac = c - a;
    closestP = closestPtPointTriangle2(p, a, b, c, ab, ac);
    const float3 v = p - closestP;
    return dot(v, v);
}

bool particleTriangleTest(inout float3 normal, inout float distance,
                          float3 particlePos, float3 cVolumePos, float cVolumeRadius,
                          float3 triV0, float3 triV1, float3 triV2, bool enableCCD)
{
    const float3 cn = cross(triV1 - triV0, triV2 - triV0);
    const float m = dot(cn, cn);
    const float3 triNormal = m > 0.0f ? cn * rsqrt(m) : float3(0, 0, 0);
    const float cVolumeRadiusSq = cVolumeRadius * cVolumeRadius;

    const float cVolumePosPlaneDist = dot(cVolumePos - triV0, triNormal);
    float3 closestPos;
    if (cVolumePosPlaneDist < cVolumeRadius)
    {
        float distSq = distancePointTriangleSquared(cVolumePos, triV0, triV1, triV2, closestPos);
        if (distSq <= cVolumeRadiusSq)
        {
            const float particlePosPlaneDist = dot(particlePos - triV0, triNormal);
            if (particlePosPlaneDist > 0.0f)
            {
                if (enableCCD)
                    distSq = distancePointTriangleSquared(particlePos, triV0, triV1, triV2, closestPos);
                const float3 distVec = particlePos - closestPos;
                normal = (distSq > 0.0f) ? distVec * rsqrt(distSq) : triNormal;
                distance = sqrt(distSq);
                return true;
            }
        }
    }
    return false;
}

// PxgParticleContactWriter::writeContact
void writeContact(uint index, float4 normalPen, uint2 compressedParticleIndex, uint2 rigidId)
{
    if (index < gParams.maxContacts)
    {
        gContactByRigid[index] = rigidId;
        gTempContactByRigid[index] = rigidId.x;
        gContactIdxByRigid[index] = index;
        gContactSortedByParticle[index] = compressedParticleIndex;
        gTempContactByParticle[index] = compressedParticleIndex.x;
        gContactIdxByParticle[index] = index;
        PsContact ct;
        ct.normal_pen = normalPen;
        ct.rigidId = rigidId;
        ct.particleId = compressedParticleIndex;
        gContacts[index] = ct;
    }
}

// ---------------------------------------------------------------------------
[shader("compute")]
[numthreads(256, 1, 1)]
void ctPsHfCollide(uint3 tid : SV_DispatchThreadID)
{
    const uint particleIndex = tid.x;
    const uint numParticles = gParams.numParticles;
    if (particleIndex >= numParticles) return;
    const bool enableCCD = (gParams.flags & 1) != 0;
    const bool isTGS = (gParams.flags & 2) != 0;
    const bool isDiffuse = (gParams.flags & 4) != 0;
    const bool rigidIsStatic = (gParams.flags & 8) != 0;

    const PhMesh mesh = gMesh[0];
    const float contactDist = mesh.scale.w;
    const float restDist = mesh.misc.x;
    const uint nbRows = mesh.dims.x;
    const uint nbCols = mesh.dims.y;

    const float3 currentPos = gCurrentPos[particleIndex].xyz;
    float3 cVolumePos;
    float cVolumeRadius;
    if (enableCCD)
    {
        const float3 predictedPos = gPredictedPos[particleIndex].xyz;
        cVolumeRadius = getParticleSpeculativeContactVolume(cVolumePos, currentPos, predictedPos,
                                                            contactDist, isDiffuse, isTGS);
    }
    else
    {
        cVolumePos = currentPos;
        cVolumeRadius = contactDist;
    }

    // world bound -> heightfield local bound (transformFast of the inverse:
    // rotate the 8 corners == rotateInv of the axis extents)
    const float3 wmin = cVolumePos - float3(cVolumeRadius, cVolumeRadius, cVolumeRadius);
    const float3 wmax = cVolumePos + float3(cVolumeRadius, cVolumeRadius, cVolumeRadius);
    // PxBounds3::transformFast(inverse(pose), bounds): center/extent form
    const float3 wc = (wmin + wmax) * 0.5f;
    const float3 we = (wmax - wmin) * 0.5f;
    const float3 lc = quatRotateInv(mesh.hfQ, wc - mesh.hfP.xyz);
    // |R^T| * extents via the rotation matrix rows of the inverse quat
    const float3 col0 = quatRotateInv(mesh.hfQ, float3(1, 0, 0));
    const float3 col1 = quatRotateInv(mesh.hfQ, float3(0, 1, 0));
    const float3 col2 = quatRotateInv(mesh.hfQ, float3(0, 0, 1));
    const float3 le = abs(col0) * we.x + abs(col1) * we.y + abs(col2) * we.z;
    float3 lmin = lc - le;
    float3 lmax = lc + le;

    const float oneOverHeightScale = 1.0f / mesh.scale.y;
    const float oneOverRowScale = 1.0f / abs(mesh.scale.x);
    const float oneOverColScale = 1.0f / abs(mesh.scale.z);

    lmin = float3(lmin.x * oneOverRowScale, lmin.y * oneOverHeightScale, lmin.z * oneOverColScale);
    lmax = float3(lmax.x * oneOverRowScale, lmax.y * oneOverHeightScale, lmax.z * oneOverColScale);

    if (mesh.scale.x < 0.0f) { const float t = lmin.x; lmin.x = lmax.x; lmax.x = t; }
    if (mesh.scale.z < 0.0f) { const float t = lmin.z; lmin.z = lmax.z; lmax.z = t; }

    if ((lmin.x > float(nbRows - 1)) || (lmin.z > float(nbCols - 1)) || (lmax.x < 0.0f) || (lmax.z < 0.0f))
        return;

    const uint minRow = getMinRow(lmin.x, nbRows);
    const uint maxRow = getMaxRow(lmax.x, nbRows);
    const uint minColumn = getMinColumn(lmin.z, nbCols);
    const uint maxColumn = getMaxColumn(lmax.z, nbCols);

    if ((2 * (maxColumn - minColumn) * (maxRow - minRow)) == 0)
        return;

    const float3 cVolumePosShapeSpace = quatRotateInv(mesh.hfQ, cVolumePos - mesh.hfP.xyz);
    const float3 particlePosShapeSpace = quatRotateInv(mesh.hfQ, currentPos - mesh.hfP.xyz);

    const float miny = lmin.y;
    const float maxy = lmax.y;
    const uint columnSpan = maxColumn - minColumn;

    float4 tempContacts[MAX_MESH_CONTACTS];
    uint contactCounts = 0;

    const uint totalNumProcessed = (maxRow - minRow) * columnSpan * 2;
    for (uint i = 0; i < totalNumProcessed; ++i)
    {
        const uint index = i / 2;
        const uint vertexIndex = (minRow + index / columnSpan) * nbCols + (minColumn + index % columnSpan);
        const float h0 = getHeight(vertexIndex);
        const float h1 = getHeight(vertexIndex + 1);
        const float h2 = getHeight(vertexIndex + nbCols);
        const float h3 = getHeight(vertexIndex + nbCols + 1);
        const bool con0 = maxy < h0 && maxy < h1 && maxy < h2 && maxy < h3;
        const bool con1 = miny > h0 && miny > h1 && miny > h2 && miny > h3;

        if (!(con0 || con1))
        {
            const bool isMaterial1 = (i & 1) != 0;
            const uint material = isMaterial1 ? materialIndex1(vertexIndex) : materialIndex0(vertexIndex);
            if (material != HOLE)
            {
                const uint triangleIdx = isMaterial1 ? ((vertexIndex << 1) + 1) : (vertexIndex << 1);

                float3 triV0, triV1, triV2;
                hfGetTriangle(triangleIdx, mesh.scale.xyz, nbCols, triV0, triV1, triV2);

                float3 normalShapeSpace = float3(0, 0, 0);
                float distance = 0.0f;
                const bool intersect = particleTriangleTest(normalShapeSpace, distance,
                    particlePosShapeSpace, cVolumePosShapeSpace, cVolumeRadius, triV0, triV1, triV2, enableCCD);

                // NOTE: no worst-contact replacement here (missing upstream)
                if (intersect && contactCounts < MAX_MESH_CONTACTS)
                {
                    const float3 normal = quatRotate(mesh.hfQ, normalShapeSpace);
                    tempContacts[contactCounts++] = float4(normal, distance - restDist);
                }
            }
        }
    }

    contactCounts = min(contactCounts, MAX_MESH_CONTACTS);
    uint contactWritten = 0;
    if (rigidIsStatic)
    {
        if (contactCounts != 0)
        {
            uint contactIndex;
            InterlockedAdd(gOneWayCounts[particleIndex], contactCounts, contactIndex);
            const uint endIndex = min(contactIndex + contactCounts, MAX_PARTICLE_CONTACTS);
            for (uint i2 = contactIndex, src = 0; i2 < endIndex; ++i2, ++src)
            {
                const uint outputIndex = particleIndex + i2 * numParticles;
                gOneWayInfos[outputIndex] = tempContacts[src];
                gOneWayNodeIndices[outputIndex] = mesh.rigid.xy;
            }
        }
        contactWritten = min(contactCounts, MAX_PARTICLE_CONTACTS);
        // faithful quirk: a PLAIN store overwrites the atomicAdd result
        gOneWayCounts[particleIndex] = contactWritten;
    }

    const uint remaining = contactCounts - contactWritten;
    if (remaining > 0 && !isDiffuse)
    {
        const uint2 compressedParticleIndex = uint2(particleIndex, gParams.particleSystemId);
        uint contactStartIndex;
        InterlockedAdd(gContactCount[0], remaining, contactStartIndex);
        const uint contactEndIndex = contactStartIndex + remaining;
        // src starts at 0 as upstream; only reachable with a dynamic rigid
        // (static remainder is always 0 since MaxStaticContactsPerMesh <=
        // MaxStaticContactsPerParticle), so the full temp list is emitted
        for (uint contactIndex = contactStartIndex, src = 0; contactIndex < contactEndIndex; ++contactIndex, ++src)
        {
            const float4 contact = tempContacts[src];
            writeContact(contactIndex, contact, compressedParticleIndex, mesh.rigid.xy);
        }
    }
}
