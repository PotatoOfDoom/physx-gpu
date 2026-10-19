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

// pxgDySpatialMatrixRef.h
//
// 5.12 compiles Dy::SpatialMatrix's operator+ / operator- and the by-value
// constructSpatialMatrix forms only under PX_CUDA_COMPILER (the host gets
// SIMD out-parameter forms instead), and comments out operator-() /
// operator*(PxReal) / operator*(SpatialMatrix) / getTranspose as unused
// (DyFeatherstoneArticulationUtils.h).  The kernels use the CUDA forms, so
// the host oracles call these transcriptions of exactly those bodies; the
// PxMat33 / PxVec3 arithmetic inside is still the engine's.

#ifndef PXG_DY_SPATIAL_MATRIX_REF_H
#define PXG_DY_SPATIAL_MATRIX_REF_H

#include "foundation/PxMat33.h"
#include "CmSpatialVector.h"
#include "DyFeatherstoneArticulationUtils.h"

namespace pxgtest
{

// SpatialMatrix::operator +(const SpatialMatrix&) const (#if PX_CUDA_COMPILER)
inline physx::Dy::SpatialMatrix dySmAdd(const physx::Dy::SpatialMatrix& a,
                                        const physx::Dy::SpatialMatrix& s)
{
    const physx::PxMat33 newTopLeft = a.topLeft + s.topLeft;
    const physx::PxMat33 newTopRight = a.topRight + s.topRight;
    const physx::PxMat33 newBottomLeft = a.bottomLeft + s.bottomLeft;
    return physx::Dy::SpatialMatrix(newTopLeft, newTopRight, newBottomLeft);
}

// SpatialMatrix::operator -(const SpatialMatrix&) const (#if PX_CUDA_COMPILER)
inline physx::Dy::SpatialMatrix dySmSub(const physx::Dy::SpatialMatrix& a,
                                        const physx::Dy::SpatialMatrix& s)
{
    const physx::PxMat33 newTopLeft = a.topLeft - s.topLeft;
    const physx::PxMat33 newTopRight = a.topRight - s.topRight;
    const physx::PxMat33 newBottomLeft = a.bottomLeft - s.bottomLeft;
    return physx::Dy::SpatialMatrix(newTopLeft, newTopRight, newBottomLeft);
}

// operator-() -- commented out as unused in 5.12, body kept there verbatim
inline physx::Dy::SpatialMatrix dySmNeg(const physx::Dy::SpatialMatrix& a)
{
    const physx::PxMat33 newTopLeft = -a.topLeft;
    const physx::PxMat33 newTopRight = -a.topRight;
    const physx::PxMat33 newBottomLeft = -a.bottomLeft;
    return physx::Dy::SpatialMatrix(newTopLeft, newTopRight, newBottomLeft);
}

// operator *(const PxReal&) -- commented out as unused in 5.12
inline physx::Dy::SpatialMatrix dySmScale(const physx::Dy::SpatialMatrix& a, const physx::PxReal& s)
{
    const physx::PxMat33 newTopLeft = a.topLeft * s;
    const physx::PxMat33 newTopRight = a.topRight * s;
    const physx::PxMat33 newBottomLeft = a.bottomLeft * s;
    return physx::Dy::SpatialMatrix(newTopLeft, newTopRight, newBottomLeft);
}

// operator *(const SpatialMatrix&) -- commented out as unused in 5.12
inline physx::Dy::SpatialMatrix dySmMul(const physx::Dy::SpatialMatrix& a,
                                        const physx::Dy::SpatialMatrix& s)
{
    const physx::PxMat33 sBottomRight = s.topLeft.getTranspose();
    const physx::PxMat33 bottomRight = a.topLeft.getTranspose();

    const physx::PxMat33 newTopLeft = a.topLeft * s.topLeft + a.topRight * s.bottomLeft;
    const physx::PxMat33 newTopRight = a.topLeft * s.topRight + a.topRight * sBottomRight;
    const physx::PxMat33 newBottomLeft = a.bottomLeft * s.topLeft + bottomRight * s.bottomLeft;
    return physx::Dy::SpatialMatrix(newTopLeft, newTopRight, newBottomLeft);
}

// getTranspose() -- commented out as unused in 5.12
inline physx::Dy::SpatialMatrix dySmGetTranspose(const physx::Dy::SpatialMatrix& a)
{
    const physx::PxMat33 newTopLeft = a.topLeft.getTranspose();
    const physx::PxMat33 newTopRight = a.bottomLeft.getTranspose();
    const physx::PxMat33 newBottomLeft = a.topRight.getTranspose();
    return physx::Dy::SpatialMatrix(newTopLeft, newTopRight, newBottomLeft);
}

// constructSpatialMatrix(const Cm::SpatialVectorF&, const Cm::SpatialVectorF&)
// (#if PX_CUDA_COMPILER, by value)
inline physx::Dy::SpatialMatrix dySmConstruct(const physx::Cm::SpatialVectorF& Is,
                                              const physx::Cm::SpatialVectorF& stI)
{
    using physx::PxVec3;
    using physx::PxMat33;
    //construct top left
    const PxVec3 tLeftC0 = Is.top * stI.top.x;
    const PxVec3 tLeftC1 = Is.top * stI.top.y;
    const PxVec3 tLeftC2 = Is.top * stI.top.z;
    const PxMat33 topLeft(tLeftC0, tLeftC1, tLeftC2);

    //construct top right
    const PxVec3 tRightC0 = Is.top * stI.bottom.x;
    const PxVec3 tRightC1 = Is.top * stI.bottom.y;
    const PxVec3 tRightC2 = Is.top * stI.bottom.z;
    const PxMat33 topRight(tRightC0, tRightC1, tRightC2);

    //construct bottom left
    const PxVec3 bLeftC0 = Is.bottom * stI.top.x;
    const PxVec3 bLeftC1 = Is.bottom * stI.top.y;
    const PxVec3 bLeftC2 = Is.bottom * stI.top.z;
    const PxMat33 bottomLeft(bLeftC0, bLeftC1, bLeftC2);

    return physx::Dy::SpatialMatrix(topLeft, topRight, bottomLeft);
}

// template <typename SpatialVector> constructSpatialMatrix(const SpatialVector* columns)
// (#if PX_CUDA_COMPILER)
template <typename SpatialVector>
inline physx::Dy::SpatialMatrix dySmConstructColumns(const SpatialVector* columns)
{
    const physx::PxMat33 topLeft(columns[0].top, columns[1].top, columns[2].top);
    const physx::PxMat33 bottomLeft(columns[0].bottom, columns[1].bottom, columns[2].bottom);
    const physx::PxMat33 topRight(columns[3].top, columns[4].top, columns[5].top);
    return physx::Dy::SpatialMatrix(topLeft, topRight, bottomLeft);
}

} // namespace pxgtest

#endif
