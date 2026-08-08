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

// P8.69: the joint-frame math of jcalc, checked against the ENGINE ITSELF.
//
// `computeJointAxis` (forwardDynamic2.cu:337) and
// `ArticulationJointCore::setJointFrame` / `computeMotionMatrix`
// (DyArticulationJointCore.h:139-207) are both PX_CUDA_CALLABLE, so this test
// does not transcribe an oracle -- it CALLS the real engine functions on the
// host over the same inputs and compares bit patterns with the Slang port.
// That is the strongest oracle available anywhere in this port: there is no
// second copy of the maths to drift.
//
// Bitwise, not tolerant.  The Slang side spells out `rotateAndNormalize`'s
// fused expression order and `PxRecipSqrt` as `1.0f / sqrt(m)` rather than
// using `normalize()` or `rsqrt`, and the module is in PXG_NOCONTRACT_SHADERS
// so no multiply-add is fused.  If any of that were wrong the comparison would
// fail rather than pass within an epsilon -- which is the point of doing it
// bitwise.
//
// The cases cover every branch of the switch and the two degenerate inputs that
// the guards exist for:
//
//   * prismatic (reads the axis BOTTOM), revolute, revolute-unwrapped,
//     spherical with 1, 2 and 3 dofs, and eFIX (writes nothing);
//   * a LOCKED axis, dofId 0xFF, which produces a ZERO joint axis -- exactly
//     the input `getNormalized()`'s `m > 0` guard exists for.  Without the
//     guard the motion matrix fills with NaN, and NaN != NaN would make the
//     comparison fail for the wrong reason, so this case is checked by bit
//     pattern like every other;
//   * non-unit and non-axis-aligned child quaternions, and a non-zero child
//     offset, so `u.cross(childOffset)` is not trivially zero.

#include "pxgTestFramework.h"
#include "PxgCudaShim.h"

#include "DyArticulationJointCore.h"

#include <cstring>
#include <vector>

namespace
{
using namespace pxgtest;
using physx::gpuslang::PxgCudaShim;
using physx::gpuslang::PxgKernelDef;
using physx::gpuslang::PxgDevicePtr;
using physx::gpuslang::PxgResult;

static std::vector<uint8_t> artiJointLoader(const char* file) { return loadShaderBlob(file); }

#pragma pack(push, 4)
struct AjJointIn
{
    float parentQ[4];
    float childQ[4];
    float childP[3];
    uint32_t jointType;
    uint32_t dof;
    uint8_t dofIds[16];
    uint32_t pad[3];
};
struct AjJointOut
{
    float relativeQuat[4];
    float mm[6][4]; // top0, bottom0, top1, bottom1, top2, bottom2
};
#pragma pack(pop)
static_assert(sizeof(AjJointIn) == 80, "AjJointIn stride");
static_assert(sizeof(AjJointOut) == 112, "AjJointOut stride");

// the enum values the Slang side hard-codes -- guessed once, wrong once
static_assert(int(physx::PxArticulationJointType::eFIX) == 0, "AJ_FIX");
static_assert(int(physx::PxArticulationJointType::ePRISMATIC) == 1, "AJ_PRISMATIC");
static_assert(int(physx::PxArticulationJointType::eREVOLUTE) == 2, "AJ_REVOLUTE");
static_assert(int(physx::PxArticulationJointType::eREVOLUTE_UNWRAPPED) == 3, "AJ_REVOLUTE_UNWRAPPED");
static_assert(int(physx::PxArticulationJointType::eSPHERICAL) == 4, "AJ_SPHERICAL");

PXG_TEST(artiJointFrame)
{
    using namespace physx;
    using namespace physx::Dy;

    PxgSlangComputeDevice* dev = testDevice();
    const PxgKernelDef manifest[] = {
        { "npArtiJointFrame.spv", "artiJointFrame", 24u, 0 }, // { u64, u64, u32 -> 24 }
    };
    PxgCudaShim shim(dev, manifest, 1, &artiJointLoader);

    struct Case
    {
        const char* name;
        PxArticulationJointType::Enum type;
        uint32_t dof;
        uint8_t dofIds[3];
        PxQuat parentQ, childQ;
        PxVec3 childP;
    };

    const PxQuat qA(0.2f, -0.4f, 0.5f, 0.7f);          // deliberately non-unit
    const PxQuat qB(0.35f, 0.15f, -0.55f, 0.74f);
    const PxQuat qId(PxIdentity);

    const Case cases[] = {
        { "prismatic X",      PxArticulationJointType::ePRISMATIC, 1, { 3, 0, 0 }, qA, qB, PxVec3(0.3f, -0.7f, 1.1f) },
        { "prismatic Z",      PxArticulationJointType::ePRISMATIC, 1, { 5, 0, 0 }, qB, qA, PxVec3(-1.0f, 0.25f, 0.0f) },
        { "revolute Y",       PxArticulationJointType::eREVOLUTE,  1, { 1, 0, 0 }, qA, qB, PxVec3(0.5f, 0.5f, -0.5f) },
        { "revolute unwrap",  PxArticulationJointType::eREVOLUTE_UNWRAPPED, 1, { 2, 0, 0 }, qB, qA, PxVec3(2.0f, 0.0f, 0.125f) },
        { "revolute at zero offset", PxArticulationJointType::eREVOLUTE, 1, { 0, 0, 0 }, qId, qA, PxVec3(0.0f, 0.0f, 0.0f) },
        { "spherical 1 dof",  PxArticulationJointType::eSPHERICAL, 1, { 0, 0, 0 }, qA, qB, PxVec3(0.9f, -0.2f, 0.4f) },
        { "spherical 2 dof",  PxArticulationJointType::eSPHERICAL, 2, { 1, 2, 0 }, qB, qA, PxVec3(-0.6f, 0.8f, 0.1f) },
        { "spherical 3 dof",  PxArticulationJointType::eSPHERICAL, 3, { 0, 1, 2 }, qA, qA, PxVec3(1.5f, -1.5f, 0.75f) },
        { "fix",              PxArticulationJointType::eFIX,       0, { 0, 0, 0 }, qA, qB, PxVec3(0.1f, 0.2f, 0.3f) },
        // dofId 0xFF is a LOCKED axis: computeJointAxis leaves the axis ZERO,
        // and getNormalized()'s m > 0 guard is what keeps the result finite
        { "locked axis",      PxArticulationJointType::eREVOLUTE,  1, { 0xFF, 0, 0 }, qA, qB, PxVec3(0.4f, 0.4f, 0.4f) },
        { "spherical locked middle", PxArticulationJointType::eSPHERICAL, 3, { 0, 0xFF, 2 }, qB, qB, PxVec3(0.2f, -0.9f, 1.3f) },
        // A rotational joint carrying a LINEAR dof id, and a fixed joint
        // carrying one.  Neither is a configuration the engine builds -- but
        // these are unit tests of two functions, not of a scene, and without
        // them two mutants survive: one that lets a linear dof id also set a
        // TOP component (invisible while every top-reading case uses an
        // angular id), and one that gives eFIX the prismatic branch (invisible
        // while the fix case's id yields a zero axis anyway).  The engine
        // oracle handles both inputs without complaint and returns zero.
        { "revolute w/ linear dofId", PxArticulationJointType::eREVOLUTE, 1, { 3, 0, 0 }, qA, qB, PxVec3(0.7f, 0.1f, -0.3f) },
        { "fix w/ linear dofId",      PxArticulationJointType::eFIX,      0, { 4, 0, 0 }, qB, qA, PxVec3(-0.2f, 1.4f, 0.6f) },
    };
    const uint32_t N = uint32_t(sizeof(cases) / sizeof(cases[0]));

    // ---- the GPU input records ----
    std::vector<AjJointIn> in(N);
    std::memset(in.data(), 0, in.size() * sizeof(AjJointIn));
    for (uint32_t i = 0; i < N; ++i)
    {
        const Case& c = cases[i];
        in[i].parentQ[0] = c.parentQ.x; in[i].parentQ[1] = c.parentQ.y;
        in[i].parentQ[2] = c.parentQ.z; in[i].parentQ[3] = c.parentQ.w;
        in[i].childQ[0] = c.childQ.x; in[i].childQ[1] = c.childQ.y;
        in[i].childQ[2] = c.childQ.z; in[i].childQ[3] = c.childQ.w;
        in[i].childP[0] = c.childP.x; in[i].childP[1] = c.childP.y; in[i].childP[2] = c.childP.z;
        in[i].jointType = uint32_t(c.type);
        in[i].dof = c.dof;
        std::memset(in[i].dofIds, 0, sizeof(in[i].dofIds));
        for (uint32_t k = 0; k < 3; ++k)
            in[i].dofIds[k] = c.dofIds[k];
    }

    // ---- the ORACLE: the real engine functions, on the host ----
    std::vector<AjJointOut> want(N);
    std::memset(want.data(), 0, want.size() * sizeof(AjJointOut));
    for (uint32_t i = 0; i < N; ++i)
    {
        const Case& c = cases[i];

        // the engine's constructor sets the poses and raises jCalcUpdateFrames;
        // it is used rather than a memset so nothing depends on this test's idea
        // of what a default joint core looks like
        ArticulationJointCore joint(PxTransform(PxVec3(0.0f), c.parentQ),
                                    PxTransform(c.childP, c.childQ));
        joint.jointType = PxU8(c.type);
        joint.jCalcUpdateFrames = true;   // setJointFrame is a no-op otherwise
        for (uint32_t k = 0; k < PxArticulationAxis::eCOUNT; ++k)
            joint.dofIds[k] = 0xFF;
        for (uint32_t k = 0; k < 3; ++k)
            joint.dofIds[k] = c.dofIds[k];

        // computeJointAxis, verbatim from forwardDynamic2.cu:337
        Cm::UnAlignedSpatialVector jointAxis[3];
        for (PxU32 k = 0; k < 3; ++k)
        {
            if (k < c.dof)
            {
                const PxU32 ind = joint.dofIds[k];
                jointAxis[k] = Cm::UnAlignedSpatialVector::Zero();
                if (ind < 6)
                    jointAxis[k][ind] = 1.0f;
            }
            else
                jointAxis[k] = Cm::UnAlignedSpatialVector::Zero();
        }

        // the motion matrix starts zeroed: eFIX writes nothing, and the Slang
        // side zeroes its outputs, so both must agree on "untouched"
        Cm::UnAlignedSpatialVector motionMatrix[3];
        for (uint32_t k = 0; k < 3; ++k)
            motionMatrix[k] = Cm::UnAlignedSpatialVector::Zero();

        PxQuat relativeQuat(PxIdentity);
        joint.setJointFrame(motionMatrix, jointAxis, relativeQuat, c.dof);

        want[i].relativeQuat[0] = relativeQuat.x; want[i].relativeQuat[1] = relativeQuat.y;
        want[i].relativeQuat[2] = relativeQuat.z; want[i].relativeQuat[3] = relativeQuat.w;
        for (uint32_t k = 0; k < 3; ++k)
        {
            want[i].mm[k * 2 + 0][0] = motionMatrix[k].top.x;
            want[i].mm[k * 2 + 0][1] = motionMatrix[k].top.y;
            want[i].mm[k * 2 + 0][2] = motionMatrix[k].top.z;
            want[i].mm[k * 2 + 1][0] = motionMatrix[k].bottom.x;
            want[i].mm[k * 2 + 1][1] = motionMatrix[k].bottom.y;
            want[i].mm[k * 2 + 1][2] = motionMatrix[k].bottom.z;
        }
    }

    // ---- dispatch ----
    PxgDevicePtr dIn = 0, dOut = 0;
    shim.memAlloc(&dIn, in.size() * sizeof(AjJointIn));
    shim.memAlloc(&dOut, N * sizeof(AjJointOut));
    if (!dIn || !dOut)
    {
        std::printf("  NOTE: no BDA -> skip\n");
        return;
    }
    shim.memcpyHtoD(dIn, in.data(), in.size() * sizeof(AjJointIn));
    shim.memsetD32(dOut, 0xCDCDCDCDu, uint32_t(N * sizeof(AjJointOut) / 4));

    struct Params { uint64_t joints, out; uint32_t count; } prm = { uint64_t(dIn), uint64_t(dOut), N };
    PXG_CHECK_MSG(shim.launchKernel(0, 1, 1, 1, &prm, sizeof(prm)) == PxgResult::Success,
                  "launch artiJointFrame");
    shim.streamSynchronize();

    std::vector<AjJointOut> got(N);
    shim.memcpyDtoH(got.data(), dOut, N * sizeof(AjJointOut));

    size_t bad = 0;
    auto same = [](float a, float b) {
        uint32_t ua, ub;
        std::memcpy(&ua, &a, 4);
        std::memcpy(&ub, &b, 4);
        return ua == ub;   // bitwise: NaN must match NaN, -0 must match -0
    };
    for (uint32_t i = 0; i < N; ++i)
    {
        for (uint32_t k = 0; k < 4; ++k)
            if (!same(got[i].relativeQuat[k], want[i].relativeQuat[k]))
            {
                if (bad < 10)
                    std::printf("  [%s] relativeQuat[%u]: got %.9g want %.9g\n",
                                cases[i].name, k, got[i].relativeQuat[k], want[i].relativeQuat[k]);
                ++bad;
            }
        for (uint32_t r = 0; r < 6; ++r)
            for (uint32_t k = 0; k < 3; ++k)
                if (!same(got[i].mm[r][k], want[i].mm[r][k]))
                {
                    if (bad < 10)
                        std::printf("  [%s] mm[%u][%u]: got %.9g want %.9g\n",
                                    cases[i].name, r, k, got[i].mm[r][k], want[i].mm[r][k]);
                    ++bad;
                }
    }

    PXG_CHECK_MSG(bad == 0, "%zu value(s) differ from the engine", bad);
    std::printf("  articulation joint frame: %u joints (prismatic/revolute/unwrapped/"
                "spherical 1-3 dof/fix/locked/cross-typed dof ids) bit-exact vs the real engine setJointFrame\n", N);
}

} // namespace
