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

// npCvxCvxEE.slang
//
// Slang port of the convex-convex edge-edge contact generator generateEE() from
// gpunarrowphase/src/CUDA/cudaGJKEPA.cu (line ~1144).  Unlike the capsule/convex-
// triangle generateEE (npCapTriangle.slang / npCvxMeshNp.slang), this one:
//   - gates on the *signed* distance directly against inflatedRadius (not the
//     squared distance against a squared radius);
//   - clips the reference edge parameter tValue to an *expanded* span
//     [-expandedRatio, 1+expandedRatio] (expandedRatio = 0.005);
//   - emits localPointA transformed back through aToB.transformInv, localPointB in
//     the same frame, and a float4 (normal, signedDist) local-normal-penetration.
//
// The reference edge is (a,b); the colliding edge is (p,q); normal is the contact
// normal.  Returns whether a contact was produced.

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

// PxTransform::transformInv(PxVec3) = q.rotateInv(input - p)
float3 transformInvPoint(float4 q, float3 p, float3 v)
{
    return quatRotateInv(q, v - p);
}

struct GenEEIn
{
    float4 p, q;        // colliding edge endpoints
    float4 a, b;        // reference edge endpoints
    float4 normal;      // contact normal
    float4 aToBq;       // aToB rotation quaternion (x,y,z,w)
    float4 aToBp;       // aToB translation (.xyz); .w = inflatedRadius
}

// c*.xyz point, c*.w signed-distance/penetration; meta.x = hit (1/0)
struct OutRes { float4 localA, localB, normalPen, meta; }

struct CvxCvxEEParams { uint numTests; }

[[vk::push_constant]]
ConstantBuffer<CvxCvxEEParams> gParams;

RWStructuredBuffer<GenEEIn> gEEIn;   // binding 0
RWStructuredBuffer<OutRes>  gEEOut;  // binding 1

// generateEE: convex-convex edge-edge contact.
// Returns true and fills glocalA/glocalB/gnormalPen when a contact is produced.
bool generateEE(float3 p, float3 q, float3 normal, float3 a, float3 b,
                float4 aToBq, float3 aToBp, float inflatedRadius,
                out float3 glocalA, out float3 glocalB, out float4 gnormalPen)
{
    glocalA = float3(0, 0, 0);
    glocalB = float3(0, 0, 0);
    gnormalPen = float4(0, 0, 0, 0);

    const float expandedRatio = 0.005f;
    const float3 ab = b - a;
    const float3 n = cross(ab, normal);
    const float d = dot(n, a);
    const float np = dot(n, p);
    const float nq = dot(n, q);
    const float signP = np - d;
    const float signQ = nq - d;
    const float temp = signP * signQ;
    if (temp > 0.0f)
        return false;

    // if colliding edge (p,q) and plane are parallel return no collision
    const float3 pq = q - p;
    const float npq = dot(n, pq);
    if (npq == 0.0f)
        return false;

    // intersect point in the segment pq with plane n.(x - a)
    const float segTValue = (d - np) / npq;
    const float3 localPointA = pq * segTValue + p;

    // ab, localPointA and normal are coplanar -> 2D segment/segment intersection:
    // build a normal perpendicular to the ray localPointA + normal*t, intersect.
    const float3 perNormal = cross(normal, pq);
    const float3 ap = localPointA - a;
    const float nom = dot(perNormal, ap);
    const float denom = dot(perNormal, ab);

    const float tValue = nom / denom;

    const float maxT = 1.0f + expandedRatio;
    const float minT = -expandedRatio;
    if (tValue > maxT || minT > tValue)
        return false;

    const float3 v = ap - ab * tValue;
    const float signedDist = dot(v, normal);
    if (inflatedRadius >= signedDist)
    {
        glocalB = localPointA - v;
        glocalA = transformInvPoint(aToBq, aToBp, localPointA);
        gnormalPen = float4(normal.x, normal.y, normal.z, signedDist);
        return true;
    }

    return false;
}

[shader("compute")]
[numthreads(256, 1, 1)]
void cvxCvxGenEE(uint3 dtid : SV_DispatchThreadID)
{
    const uint t = dtid.x;
    if (t >= gParams.numTests) return;
    const GenEEIn s = gEEIn[t];
    const float3 p = s.p.xyz, q = s.q.xyz, a = s.a.xyz, b = s.b.xyz;
    const float3 normal = s.normal.xyz;
    const float4 aToBq = s.aToBq;
    const float3 aToBp = s.aToBp.xyz;
    const float inflatedRadius = s.aToBp.w;

    float3 lA, lB;
    float4 npen;
    const bool hit = generateEE(p, q, normal, a, b, aToBq, aToBp, inflatedRadius, lA, lB, npen);

    OutRes r;
    r.localA = float4(lA, 0.0f);
    r.localB = float4(lB, 0.0f);
    r.normalPen = npen;
    r.meta = float4(hit ? 1.0f : 0.0f, 0, 0, 0);
    gEEOut[t] = r;
}
