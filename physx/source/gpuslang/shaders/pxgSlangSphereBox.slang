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

// pxgSlangSphereBox.slang -- P9.15a.  The point/sphere-vs-box closest-feature
// query, LIFTED rather than rewritten.
//
// gpunarrowphase/src/CUDA/sphereCollision.cuh:371, the PxVec3-origin overload of
// `spherebox`.  This body was already ported and swept in Phase 1 as
// fcStaticTriCloth.slang's `sphereboxNormalPen` (mode 1 of that shader's battery
// pins the inside-box membership boundaries, the min-face tie chain and the
// inflation gate).  P9.15a needs the SAME query for the engine-ABI kernel
// cloth_boxVertexContactGenLaunch, so the function moves here and both callers
// import it.  Transcribing it a second time is exactly the duplication this port
// exists to avoid, and the Phase-1 battery keeps verifying the lifted copy.
//
// The two importers compile it under different FMA policies -- femClothContactGenBda
// is in PXG_NOCONTRACT_SHADERS and fcStaticTriCloth is not.  That is safe in this
// direction: the Phase-1 battery passes WITH contraction allowed, and nocontract is
// the stricter setting.
//
// ---------------------------------------------------------------------------
// TRAPS
// ---------------------------------------------------------------------------
// SB1 -- THE MIN-FACE TIE CHAIN IS ORDERED AND USES `<=`.  The body seeds
//     localNorm/dist from the Z face, then `if (dx <= dy && dx <= dz)` takes X,
//     `else if (dy <= dz)` takes Y.  With `<=` a tie goes to the EARLIER axis
//     (x beats y and z; y beats z); with `<` it goes to Z, the seed.  A cube with
//     the point on its diagonal hits all three ties at once.
//
// SB2 -- THE NORMAL'S SIGN COMES FROM THE CLAMPED POINT `p`, not from
//     sphereCenter.  Inside the box the clamp is the identity so the two agree,
//     and that is the ONLY branch that reads the sign -- but the CUDA spells it
//     with `p` and so does this port.
//
// SB3 -- `dist` IS NEGATED: `dist = -distToSurface.<axis>`, and distToSurface is
//     `boxExtents - |p|`, i.e. non-negative inside.  So the inside branch reports
//     a NEGATIVE penetration, deeper the further in.  Dropping the negation
//     reports depth as separation and the solver pushes the wrong way.
//
// SB4 -- the outside branch computes `len = 1.0f / rsqrt(lengthSq)`, NOT
//     `sqrt(lengthSq)`.  On CUDA `PxRecipSqrt` is `intrinsics::recipSqrt`, and the
//     round trip through the reciprocal is not the same float as a direct sqrt.
//     Kept as the engine spells it.
//
// SB5 -- the gate is `sqInflatedSum > lengthSq`, STRICTLY greater: a point exactly
//     at the inflated distance produces NO contact.
//
// SB6 -- the normal is rotated OUT of box space with the box's own quaternion
//     (`quatRotate`), while the centre came IN through `quatRotateInv`.  Swapping
//     the two is a silent transpose that only shows on a non-identity rotation.
//
// SB7 -- `boxExtents` is `PxgShape::scale.scale`, the box's HALF-extents.  The
//     caller passes it straight through; there is no halving anywhere in the chain.

// The quaternion helpers are PRIVATE (no `public`), so importing this module does
// not put a second `quatRotate` into a translation unit that already has one --
// femClothContactGenBda transcribes its own.  The spelling is copied verbatim from
// pxgSlangCommon so the lifted body keeps the arithmetic the Phase-1 battery
// verified, operation for operation.
float3 sbQuatRotate(float4 q, float3 v)
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

float3 sbQuatRotateInv(float4 q, float3 v)
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

// Returns 1 and fills outNormalPen (world-space normal, penetration in .w) when
// the inflated sphere touches the box; 0 otherwise, leaving outNormalPen zeroed.
public uint sphereboxNormalPen(float3 sphereOrigin, float4 boxQ, float3 boxP, float sphereRadius,
                               float3 boxExtents, float contactDist, out float4 outNormalPen)
{
    outNormalPen = float4(0, 0, 0, 0);
    const float3 sphereCenter = sbQuatRotateInv(boxQ, sphereOrigin - boxP);   // SB6
    const float3 nBoxExtents = -boxExtents;

    const float inflatedSum = sphereRadius + contactDist;
    const float sqInflatedSum = inflatedSum * inflatedSum;

    const float x = clamp(sphereCenter.x, nBoxExtents.x, boxExtents.x);
    const float y = clamp(sphereCenter.y, nBoxExtents.y, boxExtents.y);
    const float z = clamp(sphereCenter.z, nBoxExtents.z, boxExtents.z);
    const float3 p = float3(x, y, z);
    const float3 v = sphereCenter - p;
    const float lengthSq = dot(v, v);

    if (sqInflatedSum > lengthSq)   // SB5: strictly greater
    {
        const bool bX = boxExtents.x >= abs(sphereCenter.x);
        const bool bY = boxExtents.y >= abs(sphereCenter.y);
        const bool bZ = boxExtents.z >= abs(sphereCenter.z);
        const bool bInsideBox = bX && bY && bZ;
        if (bInsideBox)
        {
            const float3 absP = float3(abs(p.x), abs(p.y), abs(p.z));   // SB2
            const float3 distToSurface = boxExtents - absP;

            // SB1: seeded from Z, then the two `<=` tests walk X, then Y.
            float3 localNorm = p.z >= 0.0f ? float3(0, 0, 1) : float3(0, 0, -1);
            float dist = -distToSurface.z;                              // SB3
            if (distToSurface.x <= distToSurface.y && distToSurface.x <= distToSurface.z)
            {
                localNorm = p.x >= 0.0f ? float3(1, 0, 0) : float3(-1, 0, 0);
                dist = -distToSurface.x;
            }
            else if (distToSurface.y <= distToSurface.z)
            {
                localNorm = p.y >= 0.0f ? float3(0, 1, 0) : float3(0, -1, 0);
                dist = -distToSurface.y;
            }

            const float3 normal = sbQuatRotate(boxQ, localNorm);
            const float penetration = dist - sphereRadius;
            outNormalPen = float4(normal, penetration);
        }
        else
        {
            // SB4: 1/rsqrt, not sqrt.
            const float recipLength = rsqrt(lengthSq);
            const float len = 1.0f / recipLength;
            const float3 locNorm = v * recipLength;
            const float penetration = len - sphereRadius;
            const float3 normal = sbQuatRotate(boxQ, locNorm);
            outNormalPen = float4(normal, penetration);
        }
        return 1;
    }
    return 0;
}
