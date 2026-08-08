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

// P8.67: layout proof for the articulation BLOCK-LAYOUT address map
// (pxgSlangArticulationBlocks.slang).
//
// PxgArticulationBlockData and PxgArticulationBlockDofData hold thirty-two
// articulations interleaved as structure-of-arrays, so every field access is
// three numbers -- struct stride, field offset, lane stride -- and each is
// independently part of the ABI.  This test proves all three, twice over:
//
//   * COMPILE TIME: every constant in the Slang module is re-derived here from
//     host-side sizeof/offsetof of the real engine structs.  The list below was
//     GENERATED from those headers, not transcribed, for the same reason the
//     Slang constants were: fifty hand-copied offsets out of a 29 KB SoA header
//     is how P8.49a's BV32 stride bug happened, where shader and fixture
//     carried the same wrong number and agreed with each other.
//
//   * RUN TIME: the artiBlockProbe kernel reads element [1] of a two-element
//     array at lane 3 and the values come back for a bitwise compare.  Element
//     [0] is filled with a DIFFERENT sentinel set rather than zeros, so a wrong
//     struct stride reads plausible-looking wrong data instead of an obvious
//     zero; and lane 3 with distinct per-lane values means a wrong element
//     stride lands on another articulation's slot.
//
// No kernel consumes this map yet.  It is landed on its own because it is the
// prerequisite for the whole unconstrained-dynamics chain, and because this
// port already treats a layout proof as a deliverable (testSimCoreDescLayout,
// testSolverDescLayout, testPrepDescLayout, testNpAbiLayout) -- those are what
// made the kernel tranches that followed them safe.

#include "pxgTestFramework.h"
#include "PxgCudaShim.h"

#include "PxArticulationTendonData.h"
#include "PxgArticulation.h"

#include <cstring>
#include <vector>

namespace
{
using namespace pxgtest;
using physx::gpuslang::PxgCudaShim;
using physx::gpuslang::PxgKernelDef;
using physx::gpuslang::PxgDevicePtr;
using physx::gpuslang::PxgResult;

static std::vector<uint8_t> artiBlockLoader(const char* file) { return loadShaderBlob(file); }

// ---------------------------------------------------------------------------
// Compile-time pins -- GENERATED from the engine headers, one per Slang constant
// ---------------------------------------------------------------------------
namespace pin
{
using namespace physx;
static_assert(sizeof(PxgSpatialVectorBlock) == 768, "PXG_AB_SPATIAL_VECTOR_BLOCK_SIZE");
static_assert(offsetof(PxgSpatialVectorBlock, mTopxyz_bx) == 0, "PXG_AB_SVB_TOPXYZ_BX");
static_assert(offsetof(PxgSpatialVectorBlock, mbyz) == 512, "PXG_AB_SVB_BYZ");
static_assert(sizeof(PxgMat33Block) == 1152, "PXG_AB_MAT33_BLOCK_SIZE");
static_assert(offsetof(PxgMat33Block, mCol0) == 0, "PXG_AB_MAT33_COL0");
static_assert(offsetof(PxgMat33Block, mCol1) == 512, "PXG_AB_MAT33_COL1");
static_assert(offsetof(PxgMat33Block, mCol2) == 1024, "PXG_AB_MAT33_COL2");
static_assert(sizeof(PxgSpatialMatrixBlock) == 3584, "PXG_AB_SPATIAL_MATRIX_BLOCK_SIZE");
static_assert(sizeof(PxgSpatialTransformBlock) == 1024, "PXG_AB_SPATIAL_TRANSFORM_BLOCK_SIZE");
static_assert(offsetof(PxgSpatialTransformBlock, q) == 0, "PXG_AB_STB_Q");
static_assert(offsetof(PxgSpatialTransformBlock, p) == 512, "PXG_AB_STB_P");

static_assert(sizeof(PxgArticulationBlockData) == 7168, "PXG_AB_BLOCK_SIZE");
static_assert(offsetof(PxgArticulationBlockData, mInvSpatialArticulatedInertia) == 0, "PXG_AB_mInvSpatialArticulatedInertia");
static_assert(offsetof(PxgArticulationBlockData, mFlags) == 3584, "PXG_AB_mFlags");
static_assert(offsetof(PxgArticulationBlockData, mNumLinks) == 3712, "PXG_AB_mNumLinks");
static_assert(offsetof(PxgArticulationBlockData, mNumSpatialTendons) == 3840, "PXG_AB_mNumSpatialTendons");
static_assert(offsetof(PxgArticulationBlockData, mNumFixedTendons) == 3968, "PXG_AB_mNumFixedTendons");
static_assert(offsetof(PxgArticulationBlockData, mNumMimicJoints) == 4096, "PXG_AB_mNumMimicJoints");
static_assert(offsetof(PxgArticulationBlockData, mTotalDofs) == 4224, "PXG_AB_mTotalDofs");
static_assert(offsetof(PxgArticulationBlockData, mArticulationIndex) == 4352, "PXG_AB_mArticulationIndex");
static_assert(offsetof(PxgArticulationBlockData, mSleepThreshold) == 4480, "PXG_AB_mSleepThreshold");
static_assert(offsetof(PxgArticulationBlockData, mCommonLinkDeltaVelocity) == 4608, "PXG_AB_mCommonLinkDeltaVelocity");
static_assert(offsetof(PxgArticulationBlockData, mLinkWithDeferredImpulse) == 5376, "PXG_AB_mLinkWithDeferredImpulse");
static_assert(offsetof(PxgArticulationBlockData, mMotionVelocitiesPtr) == 5504, "PXG_AB_mMotionVelocitiesPtr");
static_assert(offsetof(PxgArticulationBlockData, mRootDeferredZ) == 5760, "PXG_AB_mRootDeferredZ");
static_assert(offsetof(PxgArticulationBlockData, mCOM_TotalInvMassW) == 6528, "PXG_AB_mCOM_TotalInvMassW");
static_assert(offsetof(PxgArticulationBlockData, mStateDirty) == 7040, "PXG_AB_mStateDirty");
static_assert(offsetof(PxgArticulationBlockData, mTotalSelfConstraintCount) == 7072, "PXG_AB_mTotalSelfConstraintCount");
static_assert(offsetof(PxgArticulationBlockData, mSelfConstraintOffset) == 7076, "PXG_AB_mSelfConstraintOffset");

static_assert(sizeof(PxgArticulationBlockDofData) == 12800, "PXG_AB_DOF_SIZE");
static_assert(offsetof(PxgArticulationBlockDofData, mJointPositions) == 0, "PXG_AB_DOF_mJointPositions");
static_assert(offsetof(PxgArticulationBlockDofData, mJointVelocities) == 128, "PXG_AB_DOF_mJointVelocities");
static_assert(offsetof(PxgArticulationBlockDofData, mJointUnconstrainedVelocities) == 256, "PXG_AB_DOF_mJointUnconstrainedVelocities");
static_assert(offsetof(PxgArticulationBlockDofData, mPosJointVelocities) == 384, "PXG_AB_DOF_mPosJointVelocities");
static_assert(offsetof(PxgArticulationBlockDofData, mJointAccel) == 512, "PXG_AB_DOF_mJointAccel");
static_assert(offsetof(PxgArticulationBlockDofData, mQstZ) == 640, "PXG_AB_DOF_mQstZ");
static_assert(offsetof(PxgArticulationBlockDofData, mQstZIcInternal) == 768, "PXG_AB_DOF_mQstZIcInternal");
static_assert(offsetof(PxgArticulationBlockDofData, mDeferredQstZ) == 896, "PXG_AB_DOF_mDeferredQstZ");
static_assert(offsetof(PxgArticulationBlockDofData, mTmpQstZ) == 1024, "PXG_AB_DOF_mTmpQstZ");
static_assert(offsetof(PxgArticulationBlockDofData, mWorldMotionMatrix) == 1152, "PXG_AB_DOF_mWorldMotionMatrix");
static_assert(offsetof(PxgArticulationBlockDofData, mLocalMotionMatrix) == 1920, "PXG_AB_DOF_mLocalMotionMatrix");
static_assert(offsetof(PxgArticulationBlockDofData, mIsInvDW) == 2688, "PXG_AB_DOF_mIsInvDW");
static_assert(offsetof(PxgArticulationBlockDofData, mIsW) == 3456, "PXG_AB_DOF_mIsW");
static_assert(offsetof(PxgArticulationBlockDofData, mJointAxis) == 4224, "PXG_AB_DOF_mJointAxis");
static_assert(offsetof(PxgArticulationBlockDofData, mInvStIsT_x) == 4992, "PXG_AB_DOF_mInvStIsT_x");
static_assert(offsetof(PxgArticulationBlockDofData, mInvStIsT_y) == 5120, "PXG_AB_DOF_mInvStIsT_y");
static_assert(offsetof(PxgArticulationBlockDofData, mInvStIsT_z) == 5248, "PXG_AB_DOF_mInvStIsT_z");
static_assert(offsetof(PxgArticulationBlockDofData, mDofIds) == 5376, "PXG_AB_DOF_mDofIds");
static_assert(offsetof(PxgArticulationBlockDofData, mMotion) == 5408, "PXG_AB_DOF_mMotion");
static_assert(offsetof(PxgArticulationBlockDofData, mConstraintData) == 5440, "PXG_AB_DOF_mConstraintData");
} // namespace pin

// distinct sentinels: element index and lane both enter every value, so no two
// slots in the whole array can collide
static uint32_t sentU32(uint32_t elem, uint32_t lane, uint32_t field)
{
    return 0x51000000u + elem * 0x00410000u + lane * 0x00000700u + field;
}
static float sentF32(uint32_t elem, uint32_t lane, uint32_t field)
{
    return 1.5f + float(elem) * 64.0f + float(lane) * 2.0f + float(field) * 0.03125f;
}
static uint64_t bits(float f) { uint32_t u; std::memcpy(&u, &f, 4); return u; }

PXG_TEST(artiBlockLayout)
{
    using namespace physx;

    PxgSlangComputeDevice* dev = testDevice();
    const PxgKernelDef manifest[] = {
        { "npArtiBlockProbe.spv", "artiBlockProbe", 40u, 0 }, // { u64 x5 }
    };
    PxgCudaShim shim(dev, manifest, 1, &artiBlockLoader);

    const uint32_t ELEM = 1;   // the element the probe reads
    const uint32_t LANE = 3;   // the lane the probe reads

    // ---- fill the REAL engine structs, both elements, every lane ----
    std::vector<PxgArticulationBlockData> blocks(2);
    std::memset(blocks.data(), 0xEE, blocks.size() * sizeof(PxgArticulationBlockData));
    std::vector<PxgArticulationBlockDofData> dofs(2);
    std::memset(dofs.data(), 0xEE, dofs.size() * sizeof(PxgArticulationBlockDofData));
    // standalone sub-blocks: PxgMat33Block and PxgSpatialTransformBlock live
    // inside PxgArticulationBlockLinkData, which is not mapped yet, so they are
    // probed on their own rather than left unread (see the probe's note)
    std::vector<PxgMat33Block> mat33(2);
    std::memset(mat33.data(), 0xEE, mat33.size() * sizeof(PxgMat33Block));
    std::vector<PxgSpatialTransformBlock> xform(2);
    std::memset(xform.data(), 0xEE, xform.size() * sizeof(PxgSpatialTransformBlock));
    for (uint32_t e = 0; e < 2; ++e)
        for (uint32_t l = 0; l < 32; ++l)
        {
            mat33[e].mCol0[l] = make_float4(sentF32(e, l, 200), sentF32(e, l, 201),
                                            sentF32(e, l, 202), sentF32(e, l, 203));
            mat33[e].mCol1[l] = make_float4(sentF32(e, l, 204), sentF32(e, l, 205),
                                            sentF32(e, l, 206), sentF32(e, l, 207));
            mat33[e].mCol2[l] = sentF32(e, l, 208);
            xform[e].q[l] = make_float4(sentF32(e, l, 210), sentF32(e, l, 211),
                                        sentF32(e, l, 212), sentF32(e, l, 213));
            xform[e].p[l] = make_float4(sentF32(e, l, 214), sentF32(e, l, 215),
                                        sentF32(e, l, 216), sentF32(e, l, 217));
        }

    for (uint32_t e = 0; e < 2; ++e)
    {
        PxgArticulationBlockData& B = blocks[e];
        for (uint32_t l = 0; l < 32; ++l)
        {
            uint32_t f = 0;
            B.mFlags[l] = sentU32(e, l, f++);
            B.mNumLinks[l] = sentU32(e, l, f++);
            B.mNumSpatialTendons[l] = sentU32(e, l, f++);
            B.mNumFixedTendons[l] = sentU32(e, l, f++);
            B.mNumMimicJoints[l] = sentU32(e, l, f++);
            B.mTotalDofs[l] = sentU32(e, l, f++);
            B.mArticulationIndex[l] = sentU32(e, l, f++);
            B.mSleepThreshold[l] = sentF32(e, l, f++);
            B.mLinkWithDeferredImpulse[l] = sentU32(e, l, f++);
            B.mMotionVelocitiesPtr[l] = reinterpret_cast<Cm::UnAlignedSpatialVector*>(
                0xC000000000000000ull + uint64_t(e) * 0x1000000ull + uint64_t(l) * 0x10000ull);
            B.mStateDirty[l] = PxU8(0x40u + e * 16u + l);
            B.mCOM_TotalInvMassW[l] = make_float4(sentF32(e, l, 20), sentF32(e, l, 21),
                                                  sentF32(e, l, 22), sentF32(e, l, 23));
            for (uint32_t c = 0; c < 7; ++c)
                B.mInvSpatialArticulatedInertia.columns[c][l] =
                    make_float4(sentF32(e, l, 30 + c * 4), sentF32(e, l, 31 + c * 4),
                                sentF32(e, l, 32 + c * 4), sentF32(e, l, 33 + c * 4));
            B.mCommonLinkDeltaVelocity.mTopxyz_bx[l] =
                make_float4(sentF32(e, l, 70), sentF32(e, l, 71), sentF32(e, l, 72), sentF32(e, l, 73));
            B.mCommonLinkDeltaVelocity.mbyz[l] = make_float2(sentF32(e, l, 74), sentF32(e, l, 75));
            B.mRootDeferredZ.mTopxyz_bx[l] =
                make_float4(sentF32(e, l, 80), sentF32(e, l, 81), sentF32(e, l, 82), sentF32(e, l, 83));
            B.mRootDeferredZ.mbyz[l] = make_float2(sentF32(e, l, 84), sentF32(e, l, 85));
        }
        B.mTotalSelfConstraintCount = sentU32(e, 0, 90);
        B.mSelfConstraintOffset = sentU32(e, 0, 91);

        PxgArticulationBlockDofData& D = dofs[e];
        for (uint32_t l = 0; l < 32; ++l)
        {
            uint32_t f = 100;
            D.mJointPositions[l] = sentF32(e, l, f++);
            D.mJointVelocities[l] = sentF32(e, l, f++);
            D.mJointUnconstrainedVelocities[l] = sentF32(e, l, f++);
            D.mPosJointVelocities[l] = sentF32(e, l, f++);
            D.mJointAccel[l] = sentF32(e, l, f++);
            D.mQstZ[l] = sentF32(e, l, f++);
            D.mQstZIcInternal[l] = sentF32(e, l, f++);
            D.mDeferredQstZ[l] = sentF32(e, l, f++);
            D.mTmpQstZ[l] = sentF32(e, l, f++);
            D.mInvStIsT_x[l] = sentF32(e, l, f++);
            D.mInvStIsT_y[l] = sentF32(e, l, f++);
            D.mInvStIsT_z[l] = sentF32(e, l, f++);
            D.mDofIds[l] = PxU8(0x10u + e * 8u + l % 8u);
            D.mMotion[l] = PxU8(0x90u + e * 8u + l % 8u);
            PxgSpatialVectorBlock* svb[5] = { &D.mWorldMotionMatrix, &D.mLocalMotionMatrix,
                                              &D.mIsInvDW, &D.mIsW, &D.mJointAxis };
            for (uint32_t k = 0; k < 5; ++k)
            {
                svb[k]->mTopxyz_bx[l] = make_float4(sentF32(e, l, 130 + k * 6), sentF32(e, l, 131 + k * 6),
                                                    sentF32(e, l, 132 + k * 6), sentF32(e, l, 133 + k * 6));
                svb[k]->mbyz[l] = make_float2(sentF32(e, l, 134 + k * 6), sentF32(e, l, 135 + k * 6));
            }
            reinterpret_cast<PxU32*>(&D.mConstraintData)[l] = sentU32(e, l, 199);
        }
    }

    // ---- the expected values, in the probe's emit order ----
    std::vector<uint64_t> expect;
    expect.push_back(sizeof(PxgSpatialVectorBlock));
    expect.push_back(sizeof(PxgMat33Block));
    expect.push_back(sizeof(PxgSpatialMatrixBlock));
    expect.push_back(sizeof(PxgSpatialTransformBlock));
    expect.push_back(sizeof(PxgArticulationBlockData));
    expect.push_back(sizeof(PxgArticulationBlockDofData));

    const PxgArticulationBlockData& B = blocks[ELEM];
    expect.push_back(B.mFlags[LANE]);
    expect.push_back(B.mNumLinks[LANE]);
    expect.push_back(B.mNumSpatialTendons[LANE]);
    expect.push_back(B.mNumFixedTendons[LANE]);
    expect.push_back(B.mNumMimicJoints[LANE]);
    expect.push_back(B.mTotalDofs[LANE]);
    expect.push_back(B.mArticulationIndex[LANE]);
    expect.push_back(bits(B.mSleepThreshold[LANE]));
    expect.push_back(B.mLinkWithDeferredImpulse[LANE]);
    expect.push_back(uint64_t(B.mMotionVelocitiesPtr[LANE]));
    expect.push_back(B.mStateDirty[LANE]);
    expect.push_back(B.mTotalSelfConstraintCount);
    expect.push_back(B.mSelfConstraintOffset);
    expect.push_back(bits(B.mCOM_TotalInvMassW[LANE].x));
    expect.push_back(bits(B.mCOM_TotalInvMassW[LANE].y));
    expect.push_back(bits(B.mCOM_TotalInvMassW[LANE].z));
    expect.push_back(bits(B.mCOM_TotalInvMassW[LANE].w));
    expect.push_back(bits(B.mInvSpatialArticulatedInertia.columns[5][LANE].x));
    expect.push_back(bits(B.mInvSpatialArticulatedInertia.columns[5][LANE].w));
    expect.push_back(bits(B.mCommonLinkDeltaVelocity.mTopxyz_bx[LANE].x));
    expect.push_back(bits(B.mCommonLinkDeltaVelocity.mTopxyz_bx[LANE].z));
    expect.push_back(bits(B.mCommonLinkDeltaVelocity.mTopxyz_bx[LANE].w));
    expect.push_back(bits(B.mCommonLinkDeltaVelocity.mbyz[LANE].y));
    expect.push_back(bits(B.mRootDeferredZ.mTopxyz_bx[LANE].y));
    expect.push_back(bits(B.mRootDeferredZ.mbyz[LANE].x));

    const PxgArticulationBlockDofData& D = dofs[ELEM];
    expect.push_back(bits(D.mJointPositions[LANE]));
    expect.push_back(bits(D.mJointVelocities[LANE]));
    expect.push_back(bits(D.mJointUnconstrainedVelocities[LANE]));
    expect.push_back(bits(D.mPosJointVelocities[LANE]));
    expect.push_back(bits(D.mJointAccel[LANE]));
    expect.push_back(bits(D.mQstZ[LANE]));
    expect.push_back(bits(D.mQstZIcInternal[LANE]));
    expect.push_back(bits(D.mDeferredQstZ[LANE]));
    expect.push_back(bits(D.mTmpQstZ[LANE]));
    expect.push_back(bits(D.mInvStIsT_x[LANE]));
    expect.push_back(bits(D.mInvStIsT_y[LANE]));
    expect.push_back(bits(D.mInvStIsT_z[LANE]));
    expect.push_back(D.mDofIds[LANE]);
    expect.push_back(D.mMotion[LANE]);
    expect.push_back(bits(D.mWorldMotionMatrix.mTopxyz_bx[LANE].x));
    expect.push_back(bits(D.mLocalMotionMatrix.mTopxyz_bx[LANE].x));
    expect.push_back(bits(D.mIsInvDW.mTopxyz_bx[LANE].x));
    expect.push_back(bits(D.mIsW.mTopxyz_bx[LANE].x));
    expect.push_back(bits(D.mJointAxis.mTopxyz_bx[LANE].x));
    expect.push_back(bits(D.mJointAxis.mbyz[LANE].y));
    expect.push_back(reinterpret_cast<const PxU32*>(&D.mConstraintData)[LANE]);

    expect.push_back(bits(mat33[ELEM].mCol0[LANE].x));
    expect.push_back(bits(mat33[ELEM].mCol1[LANE].y));
    expect.push_back(bits(mat33[ELEM].mCol2[LANE]));
    expect.push_back(bits(xform[ELEM].q[LANE].z));
    expect.push_back(bits(xform[ELEM].p[LANE].x));

    // ---- upload, dispatch, compare ----
    PxgDevicePtr dBlocks = 0, dDofs = 0, dMat33 = 0, dXform = 0, dOut = 0;
    shim.memAlloc(&dBlocks, blocks.size() * sizeof(PxgArticulationBlockData));
    shim.memAlloc(&dDofs, dofs.size() * sizeof(PxgArticulationBlockDofData));
    shim.memAlloc(&dMat33, mat33.size() * sizeof(PxgMat33Block));
    shim.memAlloc(&dXform, xform.size() * sizeof(PxgSpatialTransformBlock));
    shim.memAlloc(&dOut, expect.size() * 8);
    if (!dBlocks || !dDofs || !dMat33 || !dXform || !dOut)
    {
        std::printf("  NOTE: no BDA -> skip\n");
        return;
    }
    shim.memcpyHtoD(dBlocks, blocks.data(), blocks.size() * sizeof(PxgArticulationBlockData));
    shim.memcpyHtoD(dDofs, dofs.data(), dofs.size() * sizeof(PxgArticulationBlockDofData));
    shim.memcpyHtoD(dMat33, mat33.data(), mat33.size() * sizeof(PxgMat33Block));
    shim.memcpyHtoD(dXform, xform.data(), xform.size() * sizeof(PxgSpatialTransformBlock));
    shim.memsetD32(dOut, 0, uint32_t(expect.size() * 2));

    struct Params { uint64_t blockData, dofData, mat33, transform, out; } prm = {
        uint64_t(dBlocks), uint64_t(dDofs), uint64_t(dMat33), uint64_t(dXform), uint64_t(dOut)
    };
    PXG_CHECK_MSG(shim.launchKernel(0, 1, 1, 1, &prm, sizeof(prm)) == PxgResult::Success,
                  "launch artiBlockProbe");
    shim.streamSynchronize();

    std::vector<uint64_t> got(expect.size(), 0);
    shim.memcpyDtoH(got.data(), dOut, got.size() * 8);

    size_t bad = 0;
    for (size_t i = 0; i < expect.size(); ++i)
        if (got[i] != expect[i])
        {
            if (bad < 12)
                std::printf("  value %zu: got 0x%016llX want 0x%016llX\n", i,
                            (unsigned long long)got[i], (unsigned long long)expect[i]);
            ++bad;
        }

    PXG_CHECK_MSG(bad == 0, "%zu of %zu probed values wrong", bad, expect.size());
    std::printf("  articulation block layout: %zu values, element [%u] lane %u, "
                "struct stride + field offset + lane stride all pinned\n",
                expect.size(), ELEM, LANE);
}

} // namespace
