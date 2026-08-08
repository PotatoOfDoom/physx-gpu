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

// npArtiBlockProbe.slang
//
// Functional proof of the articulation block-layout address map
// (pxgSlangArticulationBlocks.slang).  The host fills the REAL engine
// PxgArticulationBlockData / PxgArticulationBlockDofData with distinct
// sentinels, uploads the raw bytes, and this kernel reads every mapped field
// back through the mirror for a bitwise compare.
//
// It reads element **[1]** of a two-element array at lane **3**, so all three
// numbers in a block access are independently wrong-able:
//
//   * a wrong STRUCT STRIDE reads element [0]'s bytes (which the host fills
//     with a different sentinel set, not with zeros);
//   * a wrong FIELD OFFSET reads a neighbouring field;
//   * a wrong LANE STRIDE reads another articulation's slot.
//
// A probe at element [0] lane 0 would have every one of those collapse to the
// same address and prove nothing.

import pxgSlangArticulationBlocks;

struct ArtiBlockProbeParams
{
    uint64_t blockData; // PxgArticulationBlockData[2]
    uint64_t dofData;   // PxgArticulationBlockDofData[2]
    // PxgMat33Block and PxgSpatialTransformBlock appear only inside
    // PxgArticulationBlockLinkData, which this tranche does not map yet -- but
    // their field offsets are in the module, so they are probed standalone.  A
    // constant nothing reads is a constant nothing can prove: the first sweep
    // had PXG_AB_MAT33_COL1 and PXG_AB_STB_P survive for exactly that reason.
    uint64_t mat33;     // PxgMat33Block[2]
    uint64_t transform; // PxgSpatialTransformBlock[2]
    uint64_t out;       // uint64_t[]
};

[[vk::push_constant]]
ConstantBuffer<ArtiBlockProbeParams> gArtiBlockProbe;

static const uint AP_ELEM = 1u;  // the SECOND element -- pins the struct stride
static const uint AP_LANE = 3u;  // a non-zero lane -- pins the element stride

[shader("compute")]
[numthreads(1, 1, 1)]
void artiBlockProbe()
{
    Ptr<uint64_t> o = Ptr<uint64_t>(gArtiBlockProbe.out);
    uint i = 0;

    // ---- the sub-block and struct sizes, so a drifting stride is caught even
    //      where no field read happens to cross it ----
    o[i++] = uint64_t(PXG_AB_SPATIAL_VECTOR_BLOCK_SIZE);
    o[i++] = uint64_t(PXG_AB_MAT33_BLOCK_SIZE);
    o[i++] = uint64_t(PXG_AB_SPATIAL_MATRIX_BLOCK_SIZE);
    o[i++] = uint64_t(PXG_AB_SPATIAL_TRANSFORM_BLOCK_SIZE);
    o[i++] = uint64_t(PXG_AB_BLOCK_SIZE);
    o[i++] = uint64_t(PXG_AB_DOF_SIZE);

    // ---- PxgArticulationBlockData[1], lane 3 ----
    const uint64_t b = gArtiBlockProbe.blockData + uint64_t(AP_ELEM) * PXG_AB_BLOCK_SIZE;

    o[i++] = uint64_t(abU32(b, PXG_AB_mFlags, AP_LANE));
    o[i++] = uint64_t(abU32(b, PXG_AB_mNumLinks, AP_LANE));
    o[i++] = uint64_t(abU32(b, PXG_AB_mNumSpatialTendons, AP_LANE));
    o[i++] = uint64_t(abU32(b, PXG_AB_mNumFixedTendons, AP_LANE));
    o[i++] = uint64_t(abU32(b, PXG_AB_mNumMimicJoints, AP_LANE));
    o[i++] = uint64_t(abU32(b, PXG_AB_mTotalDofs, AP_LANE));
    o[i++] = uint64_t(abU32(b, PXG_AB_mArticulationIndex, AP_LANE));
    o[i++] = uint64_t(asuint(abF32(b, PXG_AB_mSleepThreshold, AP_LANE)));
    o[i++] = uint64_t(abU32(b, PXG_AB_mLinkWithDeferredImpulse, AP_LANE));
    // a device POINTER field: 8 bytes per lane, not 4 -- an element stride bug
    // here reads half of two neighbouring pointers
    o[i++] = abU64(b, PXG_AB_mMotionVelocitiesPtr, AP_LANE);
    // a PxU8[32] field
    o[i++] = uint64_t(abU8(b, PXG_AB_mStateDirty, AP_LANE));
    // the two per-block scalars living in the tail padding: NO lane index
    o[i++] = uint64_t(abBlockScalar(b, PXG_AB_mTotalSelfConstraintCount));
    o[i++] = uint64_t(abBlockScalar(b, PXG_AB_mSelfConstraintOffset));

    // float4 field
    {
        const float4 com = abF4(b, PXG_AB_mCOM_TotalInvMassW, AP_LANE);
        o[i++] = uint64_t(asuint(com.x));
        o[i++] = uint64_t(asuint(com.y));
        o[i++] = uint64_t(asuint(com.z));
        o[i++] = uint64_t(asuint(com.w));
    }
    // the leading PxgSpatialMatrixBlock: column 5 of 7, so a column-stride bug
    // is separable from a base-offset bug
    {
        const float4 c5 = abF4(b, PXG_AB_mInvSpatialArticulatedInertia + 5u * 512u, AP_LANE);
        o[i++] = uint64_t(asuint(c5.x));
        o[i++] = uint64_t(asuint(c5.w));
    }
    // two PxgSpatialVectorBlock fields, read through the load helper
    {
        float3 top, bottom;
        abLoadSpatialVector(b, PXG_AB_mCommonLinkDeltaVelocity, AP_LANE, top, bottom);
        o[i++] = uint64_t(asuint(top.x));
        o[i++] = uint64_t(asuint(top.z));
        o[i++] = uint64_t(asuint(bottom.x));  // the float4's .w component
        o[i++] = uint64_t(asuint(bottom.z));  // the float2's .y component

        abLoadSpatialVector(b, PXG_AB_mRootDeferredZ, AP_LANE, top, bottom);
        o[i++] = uint64_t(asuint(top.y));
        o[i++] = uint64_t(asuint(bottom.y));
    }

    // ---- PxgArticulationBlockDofData[1], lane 3 ----
    const uint64_t d = gArtiBlockProbe.dofData + uint64_t(AP_ELEM) * PXG_AB_DOF_SIZE;

    o[i++] = uint64_t(asuint(abF32(d, PXG_AB_DOF_mJointPositions, AP_LANE)));
    o[i++] = uint64_t(asuint(abF32(d, PXG_AB_DOF_mJointVelocities, AP_LANE)));
    o[i++] = uint64_t(asuint(abF32(d, PXG_AB_DOF_mJointUnconstrainedVelocities, AP_LANE)));
    o[i++] = uint64_t(asuint(abF32(d, PXG_AB_DOF_mPosJointVelocities, AP_LANE)));
    o[i++] = uint64_t(asuint(abF32(d, PXG_AB_DOF_mJointAccel, AP_LANE)));
    o[i++] = uint64_t(asuint(abF32(d, PXG_AB_DOF_mQstZ, AP_LANE)));
    o[i++] = uint64_t(asuint(abF32(d, PXG_AB_DOF_mQstZIcInternal, AP_LANE)));
    o[i++] = uint64_t(asuint(abF32(d, PXG_AB_DOF_mDeferredQstZ, AP_LANE)));
    o[i++] = uint64_t(asuint(abF32(d, PXG_AB_DOF_mTmpQstZ, AP_LANE)));
    o[i++] = uint64_t(asuint(abF32(d, PXG_AB_DOF_mInvStIsT_x, AP_LANE)));
    o[i++] = uint64_t(asuint(abF32(d, PXG_AB_DOF_mInvStIsT_y, AP_LANE)));
    o[i++] = uint64_t(asuint(abF32(d, PXG_AB_DOF_mInvStIsT_z, AP_LANE)));
    o[i++] = uint64_t(abU8(d, PXG_AB_DOF_mDofIds, AP_LANE));
    o[i++] = uint64_t(abU8(d, PXG_AB_DOF_mMotion, AP_LANE));

    // the five PxgSpatialVectorBlock members, one component each -- enough to
    // separate all five base offsets from one another
    {
        float3 top, bottom;
        abLoadSpatialVector(d, PXG_AB_DOF_mWorldMotionMatrix, AP_LANE, top, bottom);
        o[i++] = uint64_t(asuint(top.x));
        abLoadSpatialVector(d, PXG_AB_DOF_mLocalMotionMatrix, AP_LANE, top, bottom);
        o[i++] = uint64_t(asuint(top.x));
        abLoadSpatialVector(d, PXG_AB_DOF_mIsInvDW, AP_LANE, top, bottom);
        o[i++] = uint64_t(asuint(top.x));
        abLoadSpatialVector(d, PXG_AB_DOF_mIsW, AP_LANE, top, bottom);
        o[i++] = uint64_t(asuint(top.x));
        abLoadSpatialVector(d, PXG_AB_DOF_mJointAxis, AP_LANE, top, bottom);
        o[i++] = uint64_t(asuint(top.x));
        o[i++] = uint64_t(asuint(bottom.z));
    }

    // the first word of the embedded constraint block, which pins its offset
    o[i++] = uint64_t(abU32(d, PXG_AB_DOF_mConstraintData, AP_LANE));

    // ---- PxgMat33Block[1] and PxgSpatialTransformBlock[1], lane 3 ----
    const uint64_t m = gArtiBlockProbe.mat33 + uint64_t(AP_ELEM) * PXG_AB_MAT33_BLOCK_SIZE;
    o[i++] = uint64_t(asuint(abF4(m, PXG_AB_MAT33_COL0, AP_LANE).x));
    o[i++] = uint64_t(asuint(abF4(m, PXG_AB_MAT33_COL1, AP_LANE).y));
    // mCol2 is a float[32], not a float4[32] -- the one asymmetric field in the
    // three sub-blocks, and a float4 read of it would silently span four lanes
    o[i++] = uint64_t(asuint(abF32(m, PXG_AB_MAT33_COL2, AP_LANE)));

    const uint64_t t = gArtiBlockProbe.transform +
                       uint64_t(AP_ELEM) * PXG_AB_SPATIAL_TRANSFORM_BLOCK_SIZE;
    o[i++] = uint64_t(asuint(abF4(t, PXG_AB_STB_Q, AP_LANE).z));
    o[i++] = uint64_t(asuint(abF4(t, PXG_AB_STB_P, AP_LANE).x));
}
