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

// npCvxFaceContacts.slang
//
// Slang port of generatedFaceContacts() from
// gpunarrowphase/src/CUDA/cudaGJKEPA.cu (~L1099).  For each of the two GJK
// witness vertices vA[0], vA[1] it casts a vertex-space ray
// (shape2Vertex(vA_i), shape2Vertex(gjkDir)) through the convex polyhedron via
// intersectRayPolyhedron; if the ray enters the polyhedron within
// contactDistance it emits one contact:
//   va         = aToB.transformInv(vA_i)         (in the other body's frame)
//   vb         = vA_i - dir * tEnter             (dir = -gjkDir)
//   normalPen  = (dir, tEnter)
// The two contacts fill slots 0..1 (slot = prior contact count).
//
// intersectRayPolyhedron (tranche 135) and shape2Vertex are inlined for a
// self-contained shader.

static const float PXG_MAX_F32 = 3.4028235e38f;
static const uint WARP_SIZE = 32u;

// mirrors PxQuat::rotate()
float3 quatRotate(float4 q, float3 v)
{
    const float vx = 2.0f * v.x;
    const float vy = 2.0f * v.y;
    const float vz = 2.0f * v.z;
    const float w2 = q.w * q.w - 0.5f;
    const float dot2 = (q.x * vx + q.y * vy + q.z * vz);
    return float3(
        (vx * w2 + (q.y * vz - q.z * vy) * q.w + q.x * dot2),
        (vy * w2 + (q.z * vx - q.x * vz) * q.w + q.y * dot2),
        (vz * w2 + (q.x * vy - q.y * vx) * q.w + q.z * dot2));
}

// mirrors PxQuat::rotateInv()
float3 quatRotateInv(float4 q, float3 v)
{
    const float vx = 2.0f * v.x;
    const float vy = 2.0f * v.y;
    const float vz = 2.0f * v.z;
    const float w2 = q.w * q.w - 0.5f;
    const float dot2 = (q.x * vx + q.y * vy + q.z * vz);
    return float3(
        (vx * w2 - (q.y * vz - q.z * vy) * q.w + q.x * dot2),
        (vy * w2 - (q.z * vx - q.x * vz) * q.w + q.y * dot2),
        (vz * w2 - (q.x * vy - q.y * vx) * q.w + q.z * dot2));
}

// shape2Vertex: shape space -> convex vertex space
float3 shape2Vertex(float3 v, float3 scale, float4 rot)
{
    return quatRotateInv(rot, quatRotate(rot, v) / scale);
}

// PxTransform::transformInv(PxVec3) = q.rotateInv(input - p)
float3 transformInvPoint(float4 q, float3 p, float3 v)
{
    return quatRotateInv(q, v - p);
}

struct FaceContactsIn
{
    float4 vA0;      // GJK witness vertex 0 (shape space of body A); .w=nbPolygons
    float4 vA1;      // GJK witness vertex 1
    float4 gjkDir;   // gjkOutput.direction; .w = contactDistance
    float4 scale1;   // convex (mesh) scale
    float4 rot1;     // convex (mesh) scale rotation quaternion
    float4 aToBq;    // aToB rotation quaternion
    float4 aToBp;    // aToB translation (.xyz)
}

// per contact: va (.xyz), vb (.xyz), normalPen (.xyz normal, .w pen); meta.x=count
struct FaceContactsOut
{
    float4 va0, vb0, np0;
    float4 va1, vb1, np1;
    float4 meta;
}

struct FaceContactsParams { uint numTests; uint maxPolys; }

[[vk::push_constant]]
ConstantBuffer<FaceContactsParams> gParams;

RWStructuredBuffer<FaceContactsIn> gIn;      // binding 0
RWStructuredBuffer<float4>         gPlanes;  // binding 1 : maxPolys planes per test
RWStructuredBuffer<FaceContactsOut> gOut;    // binding 2

// serialised intersectRayPolyhedron (see npCvxRayPoly.slang for the derivation)
bool intersectRayPolyhedron(float3 a, float3 dir, uint testIdx, uint nbPolygons,
                            out float tEnter, out float tExit)
{
    const float eps = 1e-7f;
    float tFirst = 0.0f;
    float tLast = PXG_MAX_F32;
    tEnter = 0.0f;
    tExit = 0.0f;

    const uint base = testIdx * gParams.maxPolys;

    for (uint i = 0u; i < nbPolygons; i += WARP_SIZE)
    {
        bool blockSep = false;
        float newFirst = tFirst;
        float newLast = tLast;

        for (uint lane = 0u; lane < WARP_SIZE; ++lane)
        {
            const uint k = i + lane;
            if (k >= nbPolygons)
                continue;

            const float4 plane = gPlanes[base + k];
            const float3 n = float3(plane.x, plane.y, plane.z);
            const float d = plane.w;

            const float denominator = dot(n, dir);
            const float distToPlane = dot(n, a) + d;

            if (abs(denominator) < eps)
            {
                if (distToPlane > 0.0f)
                    blockSep = true;
            }
            else
            {
                const float tTemp = -distToPlane / denominator;
                const bool con = 0.0f > denominator;
                const bool con0 = tTemp > tFirst;
                const bool con1 = tLast > tTemp;
                if (con && con0)
                    newFirst = max(newFirst, tTemp);
                if (con1 && !con)
                    newLast = min(newLast, tTemp);
            }
        }

        if (blockSep)
            return false;

        tFirst = newFirst;
        tLast = newLast;

        if (tFirst > tLast)
            return false;
    }

    tEnter = tFirst;
    tExit = tLast;
    return true;
}

[shader("compute")]
[numthreads(64, 1, 1)]
void cvxFaceContacts(uint3 dtid : SV_DispatchThreadID)
{
    const uint t = dtid.x;
    if (t >= gParams.numTests) return;
    const FaceContactsIn s = gIn[t];

    const float3 vA0 = s.vA0.xyz;
    const float3 vA1 = s.vA1.xyz;
    const uint nbPolygons = uint(s.vA0.w);
    const float3 gjkDir = s.gjkDir.xyz;
    const float inflatedRadius = s.gjkDir.w;   // contactDistance
    const float3 scale1 = s.scale1.xyz;
    const float4 rot1 = s.rot1;
    const float4 aToBq = s.aToBq;
    const float3 aToBp = s.aToBp.xyz;

    float tEnter = 0.0f, tExit = 0.0f;
    const float3 dir = -gjkDir;

    const float3 vertexSpaceDir = shape2Vertex(-dir, scale1, rot1);
    const float3 p0 = shape2Vertex(vA0, scale1, rot1);
    const float3 p1 = shape2Vertex(vA1, scale1, rot1);

    FaceContactsOut r;
    r.va0 = float4(0, 0, 0, 0); r.vb0 = float4(0, 0, 0, 0); r.np0 = float4(0, 0, 0, 0);
    r.va1 = float4(0, 0, 0, 0); r.vb1 = float4(0, 0, 0, 0); r.np1 = float4(0, 0, 0, 0);

    uint numContacts = 0u;

    if (intersectRayPolyhedron(p0, vertexSpaceDir, t, nbPolygons, tEnter, tExit) && inflatedRadius >= tEnter)
    {
        const float3 va = transformInvPoint(aToBq, aToBp, vA0);
        const float3 vb = vA0 - dir * tEnter;
        const float4 vaf = float4(va, 0.0f);
        const float4 vbf = float4(vb, 0.0f);
        const float4 npf = float4(dir.x, dir.y, dir.z, tEnter);
        // slot = numContacts (0 here)
        r.va0 = vaf; r.vb0 = vbf; r.np0 = npf;
        numContacts++;
    }

    if (intersectRayPolyhedron(p1, vertexSpaceDir, t, nbPolygons, tEnter, tExit) && inflatedRadius >= tEnter)
    {
        const float3 va = transformInvPoint(aToBq, aToBp, vA1);
        const float3 vb = vA1 - dir * tEnter;
        const float4 vaf = float4(va, 0.0f);
        const float4 vbf = float4(vb, 0.0f);
        const float4 npf = float4(dir.x, dir.y, dir.z, tEnter);
        // slot = numContacts (0 if first missed, else 1)
        if (numContacts == 0u) { r.va0 = vaf; r.vb0 = vbf; r.np0 = npf; }
        else                   { r.va1 = vaf; r.vb1 = vbf; r.np1 = npf; }
        numContacts++;
    }

    r.meta = float4(float(numContacts), 0, 0, 0);
    gOut[t] = r;
}
