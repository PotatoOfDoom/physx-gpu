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

// pxgArtiRefHelpers.h
//
// The three articulation helpers that are `static __device__` in
// forwardDynamic2.cu and therefore cannot be called from the host, mirrored ONCE
// for every fixture that needs them (P8.86's algebra table and P8.88's
// leaf-to-root sweep).
//
// Only the STRUCTURE is transcribed: every operation inside is the engine's own
// `PxMat33` / `PxVec3` operator, so the arithmetic is still the engine's.  That
// distinction matters because the structure is exactly what P8.86's trap 5 is
// about — `translateInertia` forms `bl` and `br` from the OLD `topLeft` /
// `topRight` before overwriting `topLeft`, and never touches `topRight`.
//
// A second copy of these five lines per fixture is precisely the duplication this
// port exists to avoid, which is why they live in a header rather than being
// pasted twice.

#ifndef PXG_ARTI_REF_HELPERS_H
#define PXG_ARTI_REF_HELPERS_H

#include "foundation/PxMat33.h"
#include "foundation/PxVec3.h"
#include "CmSpatialVector.h"
#include "DyFeatherstoneArticulationUtils.h"

namespace pxgtest
{
// constructSkewSymmetricMatrix (forwardDynamic2.cu:1561).  A permutation of r's
// components with three sign flips -- no arithmetic, so nothing to round.  `r` is
// a float4 in the engine because the caller has one; w is ignored.
inline physx::PxMat33 refSkewSymmetric(float x, float y, float z)
{
    using physx::PxVec3;
    return physx::PxMat33(PxVec3(0.f, z, -y), PxVec3(-z, 0.f, x), PxVec3(y, -x, 0.f));
}

// translateInertia (forwardDynamic2.cu:1546).  `bl` and `br` come from the OLD
// topLeft / topRight; topRight is never written; bottomLeft is symmetrised last.
inline void refTranslateInertia(const physx::PxMat33& sTod, physx::Dy::SpatialMatrix& inertia)
{
    const physx::PxMat33 dTos = sTod.getTranspose();

    physx::PxMat33 bl = sTod * inertia.topLeft + inertia.bottomLeft;
    physx::PxMat33 br = sTod * inertia.topRight + inertia.getBottomRight();

    inertia.topLeft = inertia.topLeft + inertia.topRight * dTos;
    inertia.bottomLeft = bl + br * dTos;

    inertia.bottomLeft = (inertia.bottomLeft + inertia.bottomLeft.getTranspose()) * 0.5f;
}

// translateSpatialVectorInPlace (forwardDynamic2.cu:57).  The bottom half picks up
// `offset x top`; the top half is untouched.
inline void refTranslateSpatialVectorInPlace(const physx::PxVec3& offset,
                                             physx::Cm::UnAlignedSpatialVector& vec)
{
    vec.bottom += offset.cross(vec.top);
}

}  // namespace pxgtest

#endif
