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

// pxgTriTriBodyRef.h -- the host mirror of the mesh-mesh kernel's per-thread
// CONTACT BODY (pxgSlangTriTriBody): convexMeshMidphase.cu:1089-1195 transcribed
// line for line over the engine's own PxVec3 / PxQuat / PxTransform / PxMath and
// the shared __device__ mirrors in pxgTriTriGeomRef.h.
//
// P9.83b-8 (4/n) moved these here VERBATIM from testNpTriTriBody.cpp, where the
// body fixture had them, because the kernel fixture replays the same body.  The SDF
// call in the middle of each pass is not mirrored: its outputs (sep, dir, point)
// are inputs here -- the body fixture chooses them to reach every branch, the
// kernel fixture takes them from the certified device entry
// npSdfCollision:sdfCollTriangle.

#ifndef PXG_TRI_TRI_BODY_REF_H
#define PXG_TRI_TRI_BODY_REF_H

#include "pxgTriTriGeomRef.h"

#include <cstdint>
#include <vector>

namespace pxgtest
{
namespace tritribodyref
{
using namespace pxgtest::tritrigeomref;

static const float PX_MAX_F32_ = 3.402823466e+38f;

// The CUDA's `Contact` (:753-758), with pos set by the SDF call.
struct RefContact
{
    PxVec3 pos = PxVec3(0.f);
    PxReal separation = PX_MAX_F32_;
    PxVec3 normal = PxVec3(0.f);
    bool candidateContact = false;
};

// ---- convexMeshMidphase.cu, transcribed ---------------------------------------

// :1091-1094
inline void refPass0Vertices(const std::vector<U4>& inds, const std::vector<F4>& verts,
                      const PxVec3& scale0, const PxQuat& rot0, const PxTransform& aToB, uint32_t ind, uint32_t subInd,
                      const PxVec3& scale1, const PxQuat& rot1, PxVec3& v0, PxVec3& v1, PxVec3& v2)
{
    refGetTriangleVertices(inds, verts, scale0, rot0, aToB, ind, subInd, v0, v1, v2);
    v0 = refShape2Vertex(v0, scale1, rot1);
    v1 = refShape2Vertex(v1, scale1, rot1);
    v2 = refShape2Vertex(v2, scale1, rot1);
}

// :1101-1154
inline void refPass0Post(const PxVec3& v0, const PxVec3& v1, const PxVec3& v2, PxReal sep, PxVec3 dir,
                  const PxVec3& boxLo, const PxVec3& boxHi, const PxVec3& scale1, const PxQuat& rot1,
                  bool mesh0HasSdf, PxReal cullScale, PxReal contactDistance, PxReal refinementRatioSquared,
                  RefContact& contact, bool& needsRefinement, bool& usedDir2)
{
    usedDir2 = false;
    needsRefinement = false;
    if (!mesh0HasSdf)
    {
        const PxReal triRadius2 = refTriangleRadiusSquared(v0, v1, v2);
        const PxReal sdfRadius2 = refSdfRadiusSquared(boxLo, boxHi);
        needsRefinement = triRadius2 * refinementRatioSquared > sdfRadius2;
    }

    contact.pos = refVertex2Shape(contact.pos, scale1, rot1);

    if (sep < cullScale)
    {
        dir = refShape2Vertex(dir, scale1, rot1);   // vertex2ShapeNormalVector

        PxReal m = dir.magnitudeSquared();

        if (!mesh0HasSdf)
        {
            PxVec3 n = (v1 - v0).cross(v2 - v0);
            PxVec3 sdfBoxCenter = 0.5f * (boxHi + boxLo);
            PxReal triangleNormalSign = -physx::PxSign((sdfBoxCenter - v0).dot(n));
            PxVec3 dir2 = triangleNormalSign * refShape2Vertex(n, scale1, rot1);
            if (dir2.magnitudeSquared() > 1e-8f)
            {
                dir2.normalize();
                dir = dir2 * physx::PxSqrt(m);
                usedDir2 = true;
            }
        }

        if (m > 0.0f)
        {
            m = 1.0f / physx::PxSqrt(m);
            sep = sep * m;
            dir = dir * m;
        }

        contact.candidateContact = sep < contactDistance;

        if (contact.candidateContact)
        {
            contact.separation = sep;
            contact.normal = -dir;
        }
    }
}

// :1158-1167
inline void refPass1Vertices(const std::vector<U4>& inds, const std::vector<F4>& verts,
                      const PxVec3& scale1, const PxQuat& rot1, const PxTransform& aToB, uint32_t ind,
                      const PxVec3& scale0, const PxQuat& rot0, PxVec3& v0, PxVec3& v1, PxVec3& v2)
{
    const U4 t = inds[ind];
    const PxVec3 s0 = refVertex2Shape(v3(verts[t.x]), scale1, rot1);
    const PxVec3 s1 = refVertex2Shape(v3(verts[t.y]), scale1, rot1);
    const PxVec3 s2 = refVertex2Shape(v3(verts[t.z]), scale1, rot1);
    v0 = refShape2Vertex(aToB.transformInv(s0), scale0, rot0);
    v1 = refShape2Vertex(aToB.transformInv(s1), scale0, rot0);
    v2 = refShape2Vertex(aToB.transformInv(s2), scale0, rot0);
}

// :1170-1193
inline void refPass1Post(PxReal sep, PxVec3 dir, const PxVec3& scale0, const PxQuat& rot0, const PxTransform& aToB,
                  PxReal cullScale, PxReal contactDistance, RefContact& contact)
{
    contact.pos = aToB.transform(refVertex2Shape(contact.pos, scale0, rot0));

    if (sep < cullScale)
    {
        dir = refShape2Vertex(dir, scale0, rot0);

        PxReal m = dir.magnitudeSquared();
        if (m > 0.0f)
        {
            m = 1.0f / physx::PxSqrt(m);
            sep = sep * m;
            dir = dir * m;
        }

        contact.candidateContact = sep < contactDistance;

        if (contact.candidateContact)
        {
            contact.separation = sep;
            contact.normal = aToB.rotate(dir);
        }
    }
}

} // namespace tritribodyref
} // namespace pxgtest

#endif // PXG_TRI_TRI_BODY_REF_H
