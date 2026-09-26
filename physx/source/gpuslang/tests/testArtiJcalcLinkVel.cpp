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

// P8.81: the fused computeRelativeTransformC2P + computeLinkVelocities loop.
//
// THE ORACLE IS BACK.  Every tranche since P8.69 has been data movement against a
// transcribed expectation; this one has real numerics, and
// Cm::UnAlignedSpatialVector::rotate is PX_CUDA_CALLABLE, so the expectation below
// is computed by CALLING THE ENGINE -- its rotate, its cross, its PxMin/PxAbs -- on
// the host and comparing bit patterns.  Nothing about the maths is transcribed.
//
// FOUR BEHAVIOURS THE FIXTURE IS SHAPED TO CATCH:
//
//  1. TWO PASSES.  The clamp ratio is the minimum over EVERY dof and only then
//     applied.  The dof that supplies the minimum is deliberately the LAST one, so
//     a port that fuses the passes scales the earlier dofs by 1 and fails.
//  2. THE CURSOR.  dofs are walked by a running cursor driven by the link block's
//     mDofs, not indexed by a joint offset.  Link dof counts are 2, 1, 3 -- all
//     different, and none equal to its linkID.
//  3. THE ROOT'S PRE-TRANSFORM IS A RAW float4 COPY, p.w included, while every
//     other link goes through storeSpatialTransform which writes p.w = 0.  The
//     fixture puts a NON-ZERO value in every accumulated pose's p.w: the root's
//     must survive into mPreTransform, the others' must become 0.  A port that
//     unifies the two paths fails on one or the other.
//  4. rootPreMotionVelocity is written through the RECORD, not the block.
//  5. (P9.86c) THE CLAMP IS PxMin<float> = selectMin, `a < b ? a : b`
//     (forwardDynamic2.cu:995, PxUnixMathIntrinsics.h:116-119): on a +-0 tie the
//     SECOND operand survives, so PxMin(-0, +0) = +0.  GLSL FMin as llvmpipe runs it
//     gives -0 there.  Lanes 5-7 carry +-0 velocity limits (setMaxJointVelocity is
//     unvalidated; forwardDynamic2.cu:538/717 copy it verbatim), so pass one meets
//     the tie, and the sign of the resulting zero ratio shows in EVERY stored joint
//     velocity (jVel * ratio).  Mutant twins of the clamp must change the image.
//
// The link loop is strictly sequential -- link 3's parent is link 2, whose motion
// velocity was written one iteration earlier -- so the host expectation replays it
// in the same order, and the tree (parents 0,0,1,2) has both a branch at the root
// and a two-deep chain.

#include "pxgTestFramework.h"
#include "PxgCudaShim.h"

#include "PxArticulationTendonData.h"
#include "PxgArticulation.h"
#include "PxgArticulationLink.h"

#include <cstring>
#include <vector>

namespace
{
using namespace pxgtest;
using physx::gpuslang::PxgCudaShim;
using physx::gpuslang::PxgKernelDef;
using physx::gpuslang::PxgDevicePtr;
using physx::gpuslang::PxgResult;

static std::vector<uint8_t> jcalcVelLoader(const char* file) { return loadShaderBlob(file); }

static const uint32_t LANES = 8;       // 5 generic + 3 signed-zero-limit lanes (note 5)
static const uint32_t NB_LINKS = 4;
static const uint32_t DOF_BLOCKS = 7;   // 6 used + 1 spare; distinct from LANES/NB_LINKS
static const uint32_t GUARD = 128;

// P9.86c twins of the pass-one clamp.  0 = the CUDA: PxMin<float> = selectMin
// (forwardDynamic2.cu:995, PxUnixMathIntrinsics.h:116-119), b survives a +-0 tie and a
// NaN.  1 = GLSL FMin as llvmpipe runs it (F4: -0 wins a +-0 tie in either order, a NaN
// operand is ignored).  2 = a min that keeps the FIRST operand on a tie.
static int gRatioMinVariant = 0;
static uint32_t ratioBitsOf(float f) { uint32_t u; std::memcpy(&u, &f, 4); return u; }
static float ratioMin(float a, float b)
{
    if (gRatioMinVariant == 1)
    {
        if (b != b) return a;
        if (a != a) return b;
        if (a == 0.0f && b == 0.0f)
            return ((ratioBitsOf(a) | ratioBitsOf(b)) >> 31) ? -0.0f : 0.0f;
        return (a < b) ? a : b;
    }
    if (gRatioMinVariant == 2)
        return (a <= b) ? a : b;
    return physx::PxMin(a, b);
}

PXG_TEST(artiJcalcLinkVel)
{
    using namespace physx;

    PxgSlangComputeDevice* dev = testDevice();
    const PxgKernelDef manifest[] = {
        { "npArtiJointFrame.spv", "artiJcalcLinkVel", 32u, 0 }, // { u64 x3, u32 x2 }
    };
    PxgCudaShim shim(dev, manifest, 1, &jcalcVelLoader);

    const uint32_t parents[NB_LINKS] = { 0u, 0u, 1u, 2u };
    const uint32_t dofCount[NB_LINKS] = { 0u, 2u, 1u, 3u };
    const uint32_t USED_DOFS = 6;

    auto fv = [](uint32_t L, uint32_t i, uint32_t f) {
        return 0.125f * float(L + 1) + 0.0625f * float(i + 1) + 0.03125f * float(f);
    };

    // ---- inputs ----
    struct LinkIn { PxQuat q; PxVec3 p; float pw; Cm::UnAlignedSpatialVector mv; };
    std::vector<LinkIn> linkIn(size_t(LANES) * NB_LINKS);
    struct DofIn { Cm::UnAlignedSpatialVector lmm; float jVel; float maxJVel; };
    std::vector<DofIn> dofIn(size_t(LANES) * DOF_BLOCKS);

    for (uint32_t L = 0; L < LANES; ++L)
    {
        for (uint32_t l = 0; l < NB_LINKS; ++l)
        {
            LinkIn& in = linkIn[size_t(L) * NB_LINKS + l];
            in.q = PxQuat(fv(L, l, 1), fv(L, l, 2), fv(L, l, 3), 0.9f).getNormalized();
            // NOT a linear ramp across the components.  With p = (fv(4), fv(5),
            // fv(6)) scaled equally, the per-component offsets are the same constant
            // for every link, so rw = p_child - p_parent came out with rw.y == rw.z
            // and "mRw_y and mRw_z swapped" survived the first sweep.  Different
            // multipliers and a quadratic term break the cancellation.
            in.p = PxVec3(fv(L, l, 4) * 8.0f,
                          fv(L, l, 5) * 3.0f + 0.75f * float(l * l),
                          fv(L, l, 6) * 11.0f - 0.4f * float(l));
            // NON-ZERO: only the root's copy of this must survive (see note 3)
            in.pw = 7.5f + float(L) + 0.25f * float(l);
            in.mv.top = PxVec3(fv(L, l, 7), fv(L, l, 8), fv(L, l, 9));
            in.mv.bottom = PxVec3(fv(L, l, 10), fv(L, l, 11), fv(L, l, 12));
        }
        for (uint32_t d = 0; d < DOF_BLOCKS; ++d)
        {
            DofIn& in = dofIn[size_t(L) * DOF_BLOCKS + d];
            in.lmm.top = PxVec3(fv(L, d, 20), fv(L, d, 21), fv(L, d, 22));
            in.lmm.bottom = PxVec3(fv(L, d, 23), fv(L, d, 24), fv(L, d, 25));
            in.jVel = 1.0f + 0.5f * float(d) + 0.125f * float(L);
            // the LAST used dof carries the tightest limit, so it is the one that
            // sets the ratio -- a fused single pass would miss it
            in.maxJVel = (d == USED_DOFS - 1) ? (0.5f * in.jVel) : (100.0f + float(d));
        }
        // One dof with zero velocity AND a zero limit, which the clamp must SKIP
        // rather than divide by: unguarded that is 0/0 = NaN, and what min() does
        // with a NaN operand is a driver decision, not a language guarantee.  With
        // a non-zero limit the guard is provably a no-op (k/0 = +inf and min
        // ignores it), so this is the only input that can discriminate it at all.
        dofIn[size_t(L) * DOF_BLOCKS + 1].jVel = 0.0f;
        dofIn[size_t(L) * DOF_BLOCKS + 1].maxJVel = 0.0f;
    }

    // Signed-zero velocity limits (note 5).  setMaxJointVelocity is unvalidated and
    // forwardDynamic2.cu:538/717 copy the limit verbatim; with jVel != 0 the quotient
    // is a zero carrying the limit's sign, so pass one meets PxMin's +-0 tie.
    const uint32_t ZL_NEGPOS = 5, ZL_POSNEG = 6, ZL_MID = 7;
    auto dofAt = [&](uint32_t L, uint32_t d) -> DofIn& { return dofIn[size_t(L) * DOF_BLOCKS + d]; };
    dofAt(ZL_NEGPOS, 0).maxJVel = -0.0f;                  // ratio 1 -> -0
    dofAt(ZL_NEGPOS, USED_DOFS - 1).maxJVel = 0.0f;       // PxMin(-0, +0) = +0
    dofAt(ZL_POSNEG, 0).maxJVel = 0.0f;                   // ratio 1 -> +0
    dofAt(ZL_POSNEG, USED_DOFS - 1).maxJVel = -0.0f;      // PxMin(+0, -0) = -0
    dofAt(ZL_MID, 3).jVel = -dofAt(ZL_MID, 3).jVel;       // |jVel| exercised
    dofAt(ZL_MID, 3).maxJVel = -0.0f;                     // -0 picked up mid-way
    dofAt(ZL_MID, USED_DOFS - 1).jVel = -dofAt(ZL_MID, USED_DOFS - 1).jVel;
    dofAt(ZL_MID, USED_DOFS - 1).maxJVel = 0.0f;          // PxMin(-0, +0) = +0

    const size_t LINK_BLOCK = sizeof(PxgArticulationBlockLinkData);
    const size_t DOF_BLOCK = sizeof(PxgArticulationBlockDofData);
    const size_t LINK_REGION = size_t(NB_LINKS) * LINK_BLOCK + GUARD;
    const size_t DOF_REGION = size_t(DOF_BLOCKS) * DOF_BLOCK + GUARD;
    const size_t RPMV_REGION = size_t(LANES) * sizeof(Cm::UnAlignedSpatialVector) + GUARD;

    PxgDevicePtr dArtis = 0, dLink = 0, dDof = 0, dRPMV = 0;
    shim.memAlloc(&dArtis, size_t(LANES) * sizeof(PxgArticulation));
    shim.memAlloc(&dLink, LINK_REGION);
    shim.memAlloc(&dDof, DOF_REGION);
    shim.memAlloc(&dRPMV, RPMV_REGION);
    if (!dArtis || !dLink || !dDof || !dRPMV)
    {
        std::printf("  NOTE: no BDA -> skip\n");
        return;
    }

    std::vector<PxgArticulation> recs(LANES);
    std::memset(recs.data(), 0xEE, recs.size() * sizeof(PxgArticulation));
    for (uint32_t L = 0; L < LANES; ++L)
        recs[L].rootPreMotionVelocity = reinterpret_cast<Cm::UnAlignedSpatialVector*>(
            dRPMV + PxgDevicePtr(size_t(L) * sizeof(Cm::UnAlignedSpatialVector)));
    shim.memcpyHtoD(dArtis, recs.data(), recs.size() * sizeof(PxgArticulation));

    // ---- build the poisoned block images and write the inputs into them ----
    std::vector<uint8_t> linkImg(LINK_REGION, 0xAB), dofImg(DOF_REGION, 0xAB);
    auto putF = [](std::vector<uint8_t>& img, size_t off, float v) {
        std::memcpy(img.data() + off, &v, 4);
    };
    auto putU32 = [](std::vector<uint8_t>& img, size_t off, uint32_t v) {
        std::memcpy(img.data() + off, &v, 4);
    };
    auto putU8 = [](std::vector<uint8_t>& img, size_t off, uint8_t v) { img[off] = v; };
    auto putSV = [&](std::vector<uint8_t>& img, size_t base, size_t field, uint32_t L,
                     const Cm::UnAlignedSpatialVector& v) {
        putF(img, base + field + 0 + size_t(L) * 16 + 0, v.top.x);
        putF(img, base + field + 0 + size_t(L) * 16 + 4, v.top.y);
        putF(img, base + field + 0 + size_t(L) * 16 + 8, v.top.z);
        putF(img, base + field + 0 + size_t(L) * 16 + 12, v.bottom.x);
        putF(img, base + field + 512 + size_t(L) * 8 + 0, v.bottom.y);
        putF(img, base + field + 512 + size_t(L) * 8 + 4, v.bottom.z);
    };

    typedef PxgArticulationBlockLinkData LK;
    typedef PxgArticulationBlockDofData DF;
    typedef PxgArticulationInternalConstraintData IC;
    const size_t CD = offsetof(DF, mConstraintData);

    for (uint32_t L = 0; L < LANES; ++L)
    {
        for (uint32_t l = 0; l < NB_LINKS; ++l)
        {
            const LinkIn& in = linkIn[size_t(L) * NB_LINKS + l];
            const size_t blk = size_t(l) * LINK_BLOCK;
            const size_t ap = blk + offsetof(LK, mAccumulatedPose);
            putF(linkImg, ap + 0 + size_t(L) * 16 + 0, in.q.x);
            putF(linkImg, ap + 0 + size_t(L) * 16 + 4, in.q.y);
            putF(linkImg, ap + 0 + size_t(L) * 16 + 8, in.q.z);
            putF(linkImg, ap + 0 + size_t(L) * 16 + 12, in.q.w);
            putF(linkImg, ap + 512 + size_t(L) * 16 + 0, in.p.x);
            putF(linkImg, ap + 512 + size_t(L) * 16 + 4, in.p.y);
            putF(linkImg, ap + 512 + size_t(L) * 16 + 8, in.p.z);
            putF(linkImg, ap + 512 + size_t(L) * 16 + 12, in.pw);   // non-zero on purpose
            putSV(linkImg, blk, offsetof(LK, mMotionVelocity), L, in.mv);
            putU8(linkImg, blk + offsetof(LK, mDofs) + L, uint8_t(dofCount[l]));
            putU32(linkImg, blk + offsetof(LK, mParents) + size_t(L) * 4, parents[l]);
        }
        for (uint32_t d = 0; d < DOF_BLOCKS; ++d)
        {
            const DofIn& in = dofIn[size_t(L) * DOF_BLOCKS + d];
            const size_t db = size_t(d) * DOF_BLOCK;
            putSV(dofImg, db, offsetof(DF, mLocalMotionMatrix), L, in.lmm);
            putF(dofImg, db + offsetof(DF, mJointVelocities) + size_t(L) * 4, in.jVel);
            putF(dofImg, db + CD + offsetof(IC, mMaxJointVelocity) + size_t(L) * 4, in.maxJVel);
        }
    }

    std::vector<uint8_t> rpmvImg(RPMV_REGION, 0xAB);
    shim.memcpyHtoD(dLink, linkImg.data(), linkImg.size());
    shim.memcpyHtoD(dDof, dofImg.data(), dofImg.size());
    shim.memcpyHtoD(dRPMV, rpmvImg.data(), rpmvImg.size());

    struct Params
    {
        uint64_t artis, linkBlocks, dofBlocks;
        uint32_t numLinks, nbArticulations;
    } prm = { uint64_t(dArtis), uint64_t(dLink), uint64_t(dDof), NB_LINKS, LANES };

    PXG_CHECK_MSG(shim.launchKernel(0, 1, 1, 1, &prm, sizeof(prm)) == PxgResult::Success,
                  "launch artiJcalcLinkVel");
    shim.streamSynchronize();

    // ---- the ORACLE: the engine's own arithmetic, on the host ----
    // Built as a function of the clamp variant (note 5): 0 is the CUDA and is what the
    // device must match; 1 and 2 are mutant twins that must produce a different image.
    uint32_t negPosTie = 0, posNegTie = 0;   // coverage, counted for variant 0 only
    std::vector<uint32_t> ratioBits(LANES, 0xFFFFFFFFu);
    auto buildExpected = [&](int variant, std::vector<uint8_t>& wLink, std::vector<uint8_t>& wDof,
                             std::vector<uint8_t>& wRPMV) {
        wLink = linkImg;
        wDof = dofImg;
        wRPMV = rpmvImg;
        gRatioMinVariant = variant;

        for (uint32_t L = 0; L < LANES; ++L)
        {
            const LinkIn& root = linkIn[size_t(L) * NB_LINKS];

            // the root's pre-transform: a RAW copy, p.w included
            const size_t rpt = offsetof(LK, mPreTransform);
            putF(wLink, rpt + 0 + size_t(L) * 16 + 0, root.q.x);
            putF(wLink, rpt + 0 + size_t(L) * 16 + 4, root.q.y);
            putF(wLink, rpt + 0 + size_t(L) * 16 + 8, root.q.z);
            putF(wLink, rpt + 0 + size_t(L) * 16 + 12, root.q.w);
            putF(wLink, rpt + 512 + size_t(L) * 16 + 0, root.p.x);
            putF(wLink, rpt + 512 + size_t(L) * 16 + 4, root.p.y);
            putF(wLink, rpt + 512 + size_t(L) * 16 + 8, root.p.z);
            putF(wLink, rpt + 512 + size_t(L) * 16 + 12, root.pw);

            // pass one: the clamp ratio, over every dof, engine PxMin/PxAbs
            PxReal ratio = 1.0f;
            {
                uint32_t cursor = 0;
                for (uint32_t l = 1; l < NB_LINKS; ++l)
                    for (uint32_t ind = 0; ind < dofCount[l]; ++ind)
                    {
                        const DofIn& in = dofIn[size_t(L) * DOF_BLOCKS + cursor];
                        if (in.jVel != 0.0f)
                        {
                            const PxReal r = in.maxJVel / PxAbs(in.jVel);
                            if (variant == 0 && ratio == 0.0f && r == 0.0f)
                            {
                                const uint32_t sa = ratioBitsOf(ratio) >> 31, sb = ratioBitsOf(r) >> 31;
                                if (sa && !sb) ++negPosTie;   // (-0,+0): select keeps +0, FMin gives -0
                                if (!sa && sb) ++posNegTie;   // (+0,-0): select keeps -0, keep-first gives +0
                            }
                            ratio = ratioMin(ratio, r);   // variant 0: the engine's PxMin
                        }
                        ++cursor;
                    }
            }
            if (variant == 0)
                ratioBits[L] = ratioBitsOf(ratio);

            // the root's pre-motion velocity, through the record
            {
                const size_t off = size_t(L) * sizeof(Cm::UnAlignedSpatialVector);
                putF(wRPMV, off + 0, root.mv.top.x);
                putF(wRPMV, off + 4, root.mv.top.y);
                putF(wRPMV, off + 8, root.mv.top.z);
                putF(wRPMV, off + 12, root.mv.bottom.x);
                putF(wRPMV, off + 16, root.mv.bottom.y);
                putF(wRPMV, off + 20, root.mv.bottom.z);
            }

            // pass two, replayed in order -- link 3 reads link 2's freshly written velocity
            std::vector<Cm::UnAlignedSpatialVector> mv(NB_LINKS);
            mv[0] = root.mv;
            uint32_t cursor = 0;
            for (uint32_t l = 1; l < NB_LINKS; ++l)
            {
                const LinkIn& in = linkIn[size_t(L) * NB_LINKS + l];
                const LinkIn& pin = linkIn[size_t(L) * NB_LINKS + parents[l]];
                const size_t blk = size_t(l) * LINK_BLOCK;

                const PxTransform body2World(in.p, in.q);
                Cm::UnAlignedSpatialVector v = mv[parents[l]];

                // the link's pre-transform goes through storeSpatialTransform: p.w = 0
                const size_t pt = blk + offsetof(LK, mPreTransform);
                putF(wLink, pt + 0 + size_t(L) * 16 + 0, in.q.x);
                putF(wLink, pt + 0 + size_t(L) * 16 + 4, in.q.y);
                putF(wLink, pt + 0 + size_t(L) * 16 + 8, in.q.z);
                putF(wLink, pt + 0 + size_t(L) * 16 + 12, in.q.w);
                putF(wLink, pt + 512 + size_t(L) * 16 + 0, in.p.x);
                putF(wLink, pt + 512 + size_t(L) * 16 + 4, in.p.y);
                putF(wLink, pt + 512 + size_t(L) * 16 + 8, in.p.z);
                putF(wLink, pt + 512 + size_t(L) * 16 + 12, 0.0f);   // NOT in.pw

                const PxVec3 rw = in.p - pin.p;
                putF(wLink, blk + offsetof(LK, mRw_x) + size_t(L) * 4, rw.x);
                putF(wLink, blk + offsetof(LK, mRw_y) + size_t(L) * 4, rw.y);
                putF(wLink, blk + offsetof(LK, mRw_z) + size_t(L) * 4, rw.z);

                v.bottom += (-rw).cross(v.top);   // translateSpatialVectorInPlace

                for (uint32_t ind = 0; ind < dofCount[l]; ++ind)
                {
                    const DofIn& din = dofIn[size_t(L) * DOF_BLOCKS + cursor];
                    const size_t db = size_t(cursor) * DOF_BLOCK;

                    // THE ENGINE'S OWN rotate
                    const Cm::UnAlignedSpatialVector worldCol = din.lmm.rotate(body2World);
                    const PxReal jVel = din.jVel * ratio;
                    v += worldCol * jVel;

                    putF(wDof, db + offsetof(DF, mJointVelocities) + size_t(L) * 4, jVel);
                    putSV(wDof, db, offsetof(DF, mWorldMotionMatrix), L, worldCol);
                    ++cursor;
                }

                putSV(wLink, blk, offsetof(LK, mMotionVelocity), L, v);
                mv[l] = v;
            }
        }
        gRatioMinVariant = 0;
    };

    std::vector<uint8_t> wLink, wDof, wRPMV;
    buildExpected(0, wLink, wDof, wRPMV);
    PXG_CHECK_MSG(negPosTie == 2 && posNegTie == 1,
                  "zero-limit lanes: %u (-0,+0) and %u (+0,-0) PxMin ties, want 2 and 1",
                  negPosTie, posNegTie);
    PXG_CHECK_MSG(ratioBits[ZL_NEGPOS] == 0x00000000u && ratioBits[ZL_POSNEG] == 0x80000000u &&
                      ratioBits[ZL_MID] == 0x00000000u,
                  "zero-limit ratios %08x %08x %08x, want +0 -0 +0", ratioBits[ZL_NEGPOS],
                  ratioBits[ZL_POSNEG], ratioBits[ZL_MID]);
    for (int v = 1; v <= 2; ++v)
    {
        std::vector<uint8_t> tl, td, tr;
        buildExpected(v, tl, td, tr);
        PXG_CHECK_MSG(td != wDof,
                      "ratio twin %d gives the same dof image -- the zero-limit lanes do not "
                      "observe PxMin's +-0 tie", v);
    }

    // ---- compare ----
    size_t bad = 0;
    auto region = [&](const char* what, PxgDevicePtr p, const std::vector<uint8_t>& want,
                      size_t stride) {
        std::vector<uint8_t> got(want.size(), 0);
        shim.memcpyDtoH(got.data(), p, got.size());
        size_t local = 0;
        for (size_t i = 0; i < want.size(); ++i)
        {
            if (got[i] != want[i])
            {
                if (bad < 12)
                    std::printf("  %s +%zu (block %zu, +%zu): got 0x%02X want 0x%02X\n",
                                what, i, i / stride, i % stride, got[i], want[i]);
                ++bad;
                if (++local >= 6)
                    return;
            }
        }
    };
    region("linkBlocks", dLink, wLink, LINK_BLOCK);
    region("dofBlocks", dDof, wDof, DOF_BLOCK);
    region("rootPreMotionVelocity", dRPMV, wRPMV, sizeof(Cm::UnAlignedSpatialVector));

    shim.memFree(dArtis);
    shim.memFree(dLink);
    shim.memFree(dDof);
    shim.memFree(dRPMV);

    PXG_CHECK_MSG(bad == 0, "%zu byte(s) wrong", bad);
    std::printf("  jcalc link velocities: 8 articulations (3 with +-0 velocity limits: %u "
                "(-0,+0) and %u (+0,-0) PxMin ties), 4 links (branch + chain), dofs "
                "2/1/3 -- bit-exact against the engine's own rotate and cross; the clamp "
                "ratio set by the last dof; the root's pre-transform p.w preserved and the "
                "links' zeroed\n", negPosTie, posNegTie);
}

} // namespace
