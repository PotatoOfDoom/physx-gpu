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

// pxgTriTriGeomRef.h -- the host mirrors of the mesh-mesh kernel's __device__
// geometry, shared by the P9.83b-8 fixtures (1/n geometry, 2/n contact body).
//
// Only what is __device__ in the engine is transcribed here, each next to its
// CUDA line; everything below it (PxQuat::rotate, PxTransform, PxMax, PxSign,
// Gu::getSubTriangleEncoded) is the ENGINE'S OWN host-compilable code.  One copy
// per function, for the reason pxgTriTriRef.h gives: a second copy would be a
// second thing to keep in step with the CUDA, and a fixture whose oracle drifted
// from its sibling's would certify two different ports.

#ifndef PXG_TRI_TRI_GEOM_REF_H
#define PXG_TRI_TRI_GEOM_REF_H

#include "foundation/PxMath.h"
#include "foundation/PxQuat.h"
#include "foundation/PxTransform.h"
#include "foundation/PxVec3.h"
#include "GuTriangleRefinement.h"

#include <cstdint>
#include <vector>

namespace pxgtest
{
namespace tritrigeomref
{
using physx::PxQuat;
using physx::PxReal;
using physx::PxTransform;
using physx::PxVec3;

struct F4 { float x, y, z, w; };
struct U4 { uint32_t x, y, z, w; };

inline PxVec3 v3(const F4& f) { return PxVec3(f.x, f.y, f.z); }
inline PxQuat q4(const F4& f) { return PxQuat(f.x, f.y, f.z, f.w); }

// cudaNpCommon.h:270-281
inline PxVec3 refVertex2Shape(const PxVec3& v, const PxVec3& scale, const PxQuat& rotation)
{
    PxVec3 temp = rotation.rotate(v);

    temp.x *= scale.x;
    temp.y *= scale.y;
    temp.z *= scale.z;

    return rotation.rotateInv(temp);
}

// cudaNpCommon.h:284-296 -- and vertex2ShapeNormalVector (:301-304) is this.
inline PxVec3 refShape2Vertex(const PxVec3& v, const PxVec3& scale, const PxQuat& rotation)
{
    PxVec3 temp = rotation.rotate(v);

    temp.x /= scale.x;
    temp.y /= scale.y;
    temp.z /= scale.z;

    return rotation.rotateInv(temp);
}

// sdfCollision.cuh:521-537 (the uint4* overload forwarding to the uint4& one)
inline void refGetTriangleVertices(const std::vector<U4>& meshInd, const std::vector<F4>& meshVerts,
                                   const PxVec3& scale, const PxQuat& scaleRot, const PxTransform& t,
                                   uint32_t triangleIndex, uint32_t encodedSubIndex,
                                   PxVec3& a, PxVec3& b, PxVec3& c)
{
    const U4 inds = meshInd[triangleIndex];
    a = t.transform(refVertex2Shape(v3(meshVerts[inds.x]), scale, scaleRot));
    b = t.transform(refVertex2Shape(v3(meshVerts[inds.y]), scale, scaleRot));
    c = t.transform(refVertex2Shape(v3(meshVerts[inds.z]), scale, scaleRot));

    if (encodedSubIndex > 0)
        physx::Gu::getSubTriangleEncoded(a, b, c, encodedSubIndex);
}

// sdfCollision.cuh:555-558
inline PxVec3 refGetVertex(const std::vector<F4>& meshVerts, const PxVec3& scale, const PxQuat& scaleRot,
                           const PxTransform& t, uint32_t vertexIndex)
{
    return t.transform(refVertex2Shape(v3(meshVerts[vertexIndex]), scale, scaleRot));
}

// sdfCollision.cuh:729-734.  The CUDA's `center` is computed and never read; it
// has no side effect, so it is left out here as it is in the port.
inline PxReal refTriangleRadiusSquared(const PxVec3& v0, const PxVec3& v1, const PxVec3& v2)
{
    return physx::PxMax(physx::PxMax((v1 - v0).magnitudeSquared(), (v2 - v1).magnitudeSquared()),
                        (v0 - v2).magnitudeSquared());
}

// sdfCollision.cuh:736-739
inline PxReal refSdfRadiusSquared(const PxVec3& lo, const PxVec3& hi) { return (hi - lo).magnitudeSquared(); }

} // namespace tritrigeomref
} // namespace pxgtest

#endif // PXG_TRI_TRI_GEOM_REF_H
