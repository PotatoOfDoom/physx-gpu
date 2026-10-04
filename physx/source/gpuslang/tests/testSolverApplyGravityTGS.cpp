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

// Verification of solverApplyGravityTGS.slang: applyTGSSubstepGravity and its
// applyAcceleration helper, on the ENGINE ABI (P9.95).
//
// The kernel is launched through PxgCudaShim exactly as PxgTGSCudaSolverCore.cpp:1354-1374
// does: the pack {const PxgSolverCoreDesc*, const PxgSolverSharedDesc<IterativeSolveDataTGS>*,
// PxVec3 gravity BY VALUE} = 8 + 8 + 12 -> 32 bytes, and the UNSCALED engine grid
// ceil(32*(mBodyCount + mBodyStartIndex)/256) -- the bridge drops blockDim (32, 8) and the
// shader's numthreads(256) is one CUDA block.  Every count, offset and pool pointer comes
// from REAL PxgSolverCoreDesc / PxgIslandContext / PxgSolverSharedDesc structs (desc
// offsets pinned by testSolverDescLayout), and the per-body inputs are REAL PxgBodySim /
// PxgSolverBodyData / PxgSolverTxIData records.
//
// The oracle is the CUDA (solverMultiBlockTGS.cu:1575-1606 applyAcceleration, :1608-1683
// the kernel) copied as host C++ over PxVec3 / PxMat33 (the REAL getInverse() and
// operator*), with the thread grid emulated block x threadIdx.y x threadIdx.x and
// nextPowerOfTwo / isPowerOfTwo copied from solverBlock.cuh:62-75.  This file is built
// with -ffp-contract=off, so every comparison is BITWISE: the whole velocity pool, a
// sentinel guard region on each side of it, and every input buffer (which must come back
// unchanged).
//
// Three oracle twins prove that the directed cases observe what they claim to:
//   gIdentityMulVariant        1 = "return v" for the singular PxMat33(PxIdentity) product (P9.86c)
//   gDetDotVariant             1 = determinant dot summed x + (y + z) (P9.90, llvmpipe OpDot)
//   gDisableGravityFixVariant  1 = the CPU semantics (DyTGSDynamics.cpp:1490-1493: no gravity
//                                  under disableGravity) instead of the CUDA quirk

#include "pxgTestFramework.h"
#include "PxgCudaShim.h"

#include "PxNodeIndex.h" // completes PxNodeIndex for the PxgNarrowphaseCore.h chain
#include "PxgSolverCoreDesc.h"
#include "PxgIslandContext.h"
#include "PxgSolverBody.h"
#include "PxgBodySim.h"
#include "PxRigidDynamic.h"
#include "foundation/PxMat33.h"
#include "foundation/PxVec3.h"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <limits>
#include <random>
#include <vector>

namespace
{
using namespace pxgtest;
using namespace physx;
using physx::gpuslang::PxgCudaShim;
using physx::gpuslang::PxgKernelDef;
using physx::gpuslang::PxgDevicePtr;
using physx::gpuslang::PxgResult;

// ---- layout pins (the shader reads these through pxgSlangCommon / pxgSlangSolverDesc) ----
static_assert(offsetof(PxgBodySim, sleepAngVelAccXYZ_accelScaleW) == 80, "PxgBodySim accelScale float4");
static_assert(offsetof(PxgBodySim, lockFlags) == 184, "PxgBodySim lockFlags");
static_assert(offsetof(PxgBodySim, disableGravity) == 186, "PxgBodySim disableGravity");
static_assert(offsetof(PxgBodySim, externalLinearAcceleration) == 192, "PxgBodySim externalLinearAcceleration");
static_assert(offsetof(PxgBodySim, externalAngularAcceleration) == 208, "PxgBodySim externalAngularAcceleration");
static_assert(sizeof(PxgBodySim) == 224, "PxgBodySim stride");
static_assert(sizeof(PxgSolverBodyData) == 96, "PxgSolverBodyData stride");
static_assert(offsetof(PxgSolverTxIData, sqrtInvInertia) == 28, "PxgSolverTxIData sqrtInvInertia");
static_assert(sizeof(PxgSolverTxIData) == 64, "PxgSolverTxIData stride");
static_assert(sizeof(PxgSolverReferences) == 4, "PxgSolverReferences stride");

// The host pack PxgTGSCudaSolverCore.cpp:1362-1367 under the bridge's natural-alignment
// rule: ptr @0, ptr @8, PxVec3 @16 (12 bytes), rounded to the 8-byte max alignment.
struct ApplyGravityPrm
{
    uint64_t solverCoreDesc;
    uint64_t sharedDesc;
    PxVec3 gravity;
    uint32_t pad;
};
static_assert(sizeof(ApplyGravityPrm) == 32, "applyTGSSubstepGravity blob = 32 (kTwoPtrVec3)");
static_assert(offsetof(ApplyGravityPrm, gravity) == 16, "gravity @16");

struct F4 { float x, y, z, w; };
static_assert(sizeof(F4) == 16, "velocity pool element");

const uint32_t INVALID = 0xffffffffu;
const uint32_t GUARD = 64;            // float4 guard elements on each side of the velocity pool
const float GUARD_SENT = -12345.678f; // guard fill

uint32_t f2u(float f) { uint32_t u; std::memcpy(&u, &f, 4); return u; }
float u2f(uint32_t u) { float f; std::memcpy(&f, &u, 4); return f; }

int gIdentityMulVariant = 0;
int gDetDotVariant = 0;
int gDisableGravityFixVariant = 0;
uint32_t gSingularCalls = 0;

// solverBlock.cuh:62-70, literally
uint32_t nextPowerOfTwo(uint32_t x)
{
    x |= (x >> 1);
    x |= (x >> 2);
    x |= (x >> 4);
    x |= (x >> 8);
    x |= (x >> 16);
    return x + 1;
}

// solverBlock.cuh:72-75
bool isPowerOfTwo(uint32_t x)
{
    return (x & (x - 1)) == 0;
}

// inertia.sqrtInvInertia.getInverse() * v (solverMultiBlockTGS.cu:1592).  With every twin
// off this is the REAL PxMat33 code (PxMat33.h:215-247 getInverse/getDeterminant, :344-347
// transform); the hand copy below exists only to host the twins.
PxVec3 inverseTimes(const PxMat33& m, const PxVec3& v)
{
    if (m.getDeterminant() == 0.0f)
        ++gSingularCalls;
    if (gIdentityMulVariant == 0 && gDetDotVariant == 0)
        return m.getInverse() * v;

    const PxVec3 cr = m.column1.cross(m.column2);
    const float det = gDetDotVariant == 1 ? m.column0.x * cr.x + (m.column0.y * cr.y + m.column0.z * cr.z)
                                          : (m.column0.x * cr.x + m.column0.y * cr.y) + m.column0.z * cr.z;
    if (det != 0.0f)
    {
        const PxMat33& c = m;
        const float invDet = 1.0f / det;
        PxMat33 inverse;
        inverse.column0.x = invDet * (c.column1.y * c.column2.z - c.column2.y * c.column1.z);
        inverse.column0.y = invDet * -(c.column0.y * c.column2.z - c.column2.y * c.column0.z);
        inverse.column0.z = invDet * (c.column0.y * c.column1.z - c.column0.z * c.column1.y);
        inverse.column1.x = invDet * -(c.column1.x * c.column2.z - c.column1.z * c.column2.x);
        inverse.column1.y = invDet * (c.column0.x * c.column2.z - c.column0.z * c.column2.x);
        inverse.column1.z = invDet * -(c.column0.x * c.column1.z - c.column0.z * c.column1.x);
        inverse.column2.x = invDet * (c.column1.x * c.column2.y - c.column1.y * c.column2.x);
        inverse.column2.y = invDet * -(c.column0.x * c.column2.y - c.column0.y * c.column2.x);
        inverse.column2.z = invDet * (c.column0.x * c.column1.y - c.column1.x * c.column0.y);
        return inverse * v;
    }
    if (gIdentityMulVariant == 1)
        return v;
    return PxMat33(PxIdentity) * v;
}

// solverMultiBlockTGS.cu:1575-1606, literally (disableGravity is read only by the
// gDisableGravityFixVariant twin; the CUDA never reads it here).
void applyAcceleration(PxU32 bodyIndex, F4* bodyVelocities, const PxVec3& gravityTimesDt, const PxReal stepDt,
                       const PxU16 disableGravity, const float4& linearAcceleration, const float4& angularAcceleration,
                       const bool hasLinearAcceleration, const bool hasAngularAcceleration, const float accelScale,
                       const PxU16 lockFlags, const PxgSolverTxIData& inertia, PxU32 angVelIndexOffset)
{
    (void)hasLinearAcceleration;
    const bool cpuFix = gDisableGravityFixVariant == 1 && disableGravity != 0;
    F4 v = bodyVelocities[bodyIndex];
    v.x += (cpuFix ? 0.0f : accelScale * gravityTimesDt.x) + linearAcceleration.x * stepDt;
    v.y += (cpuFix ? 0.0f : accelScale * gravityTimesDt.y) + linearAcceleration.y * stepDt;
    v.z += (cpuFix ? 0.0f : accelScale * gravityTimesDt.z) + linearAcceleration.z * stepDt;

    v.x = lockFlags & PxRigidDynamicLockFlag::eLOCK_LINEAR_X ? 0.f : v.x;
    v.y = lockFlags & PxRigidDynamicLockFlag::eLOCK_LINEAR_Y ? 0.f : v.y;
    v.z = lockFlags & PxRigidDynamicLockFlag::eLOCK_LINEAR_Z ? 0.f : v.z;

    if (hasAngularAcceleration)
    {
        F4 a = bodyVelocities[bodyIndex + angVelIndexOffset];

        PxVec3 angAcc = inverseTimes(inertia.sqrtInvInertia, PxVec3(angularAcceleration.x, angularAcceleration.y, angularAcceleration.z));

        v.w += angAcc.x * stepDt;
        a.x += angAcc.y * stepDt;
        a.y += angAcc.z * stepDt;

        v.w = lockFlags & PxRigidDynamicLockFlag::eLOCK_ANGULAR_X ? 0.f : v.w;
        a.x = lockFlags & PxRigidDynamicLockFlag::eLOCK_ANGULAR_Y ? 0.f : a.x;
        a.y = lockFlags & PxRigidDynamicLockFlag::eLOCK_ANGULAR_Z ? 0.f : a.y;

        bodyVelocities[bodyIndex + angVelIndexOffset] = a;
    }

    bodyVelocities[bodyIndex] = v;
}

// One fixture case: everything the kernel reads.
struct GravityCase
{
    uint32_t allBodiesCount = 0, bodyOffset = 0, numSlabs = 0, outputOffset = 0;
    PxVec3 gravity = PxVec3(0.0f);
    float stepDt = 0.0f, dt = 0.0f;
    std::vector<PxgBodySim> bodySim;          // indexed by node index (longer than allBodiesCount)
    std::vector<PxgSolverBodyData> bodyData;  // allBodiesCount
    std::vector<PxgSolverTxIData> txI;        // EXACTLY allBodiesCount
    std::vector<uint32_t> refs;               // numSlabs * (allBodiesCount - bodyOffset)
    std::vector<F4> pool;                     // the solver body velocity pool
};

uint32_t engineGrid(const GravityCase& c)
{
    // PxgTGSCudaSolverCore.cpp:1356-1358: nbThreadsRequired = 32 * (mBodyCount + mBodyStartIndex),
    // nbBlocksRequired = ceil(nbThreadsRequired / 256); the bridge dispatches it unscaled.
    const uint32_t mBodyCount = c.allBodiesCount - c.bodyOffset;
    return (32u * (mBodyCount + c.bodyOffset) + 255u) / 256u;
}

// solverMultiBlockTGS.cu:1608-1683 over the emulated grid (blockIdx.x, threadIdx.y, threadIdx.x).
std::vector<F4> oracle(const GravityCase& c)
{
    std::vector<F4> pool = c.pool;
    F4* bodyVelocities = pool.data();
    const PxReal stepDt = c.stepDt;
    const PxgSolverReferences* solverReferences = reinterpret_cast<const PxgSolverReferences*>(c.refs.data());
    const PxU32 bodyOffset = c.bodyOffset;
    const PxU32 numSlabs = c.numSlabs;
    const PxU32 allBodiesCount = c.allBodiesCount;
    const PxgSolverTxIData* iData = c.txI.data();
    const uint32_t nbBlocks = engineGrid(c);

    for (uint32_t blockIdx = 0; blockIdx < nbBlocks; ++blockIdx)
        for (uint32_t threadIdxY = 0; threadIdxY < 8; ++threadIdxY)
            for (uint32_t threadIdxX = 0; threadIdxX < 32; ++threadIdxX)
            {
                const PxU32 warpIndex = blockIdx * 8 + threadIdxY;
                const PxU32 numThreadsPerBody = PxMin(isPowerOfTwo(numSlabs) ? numSlabs : nextPowerOfTwo(numSlabs), 32u);
                const PxU32 numBodiesPerWarp = 32 / numThreadsPerBody;
                const PxU32 threadIdxInWorkUnit = threadIdxX & (numThreadsPerBody - 1);
                const PxU32 warpBodyStartIndex = warpIndex * numBodiesPerWarp;
                const PxU32 bodyInWarp = threadIdxX / numThreadsPerBody;
                const PxU32 bodyId = warpBodyStartIndex + bodyInWarp;
                const PxU32 outputOffset = c.outputOffset;
                PxVec3 gravityTimesDt = stepDt * c.gravity;

                if (bodyId < allBodiesCount)
                {
                    if (bodyId >= bodyOffset)
                    {
                        const PxgSolverBodyData& data = c.bodyData[bodyId];
                        const PxU32 nodeIndex = data.islandNodeIndex.index();
                        const PxgBodySim& bodySim = c.bodySim[nodeIndex];
                        const PxU16 disableGravity = bodySim.disableGravity;
                        const float4 linearAcceleration = bodySim.externalLinearAcceleration;
                        const float4 angularAcceleration = bodySim.externalAngularAcceleration;
                        const bool hasLinearAcceleration = linearAcceleration.x != 0.0f || linearAcceleration.y != 0.0f || linearAcceleration.z != 0.0f;
                        const bool hasAngularAcceleration = angularAcceleration.x != 0.0f || angularAcceleration.y != 0.0f || angularAcceleration.z != 0.0f;
                        PxgSolverTxIData iData0;
                        std::memset(&iData0, 0, sizeof(iData0));
                        if (hasAngularAcceleration)
                            iData0 = iData[bodyId /*+ bodyOffset*/];

                        if (!disableGravity || hasLinearAcceleration || hasAngularAcceleration)
                        {
                            float accelScale = bodySim.sleepAngVelAccXYZ_accelScaleW.w;
                            PxU16 lockFlags = bodySim.lockFlags;
                            for (uint32_t b = threadIdxInWorkUnit; b < numSlabs; b += numThreadsPerBody)
                            {
                                const uint32_t bodyIndex = numSlabs * (bodyId - bodyOffset) + b;
                                PxgSolverReferences reference = solverReferences[bodyIndex];
                                const PxU32 remappedBodyIndex = reference.mRemappedBodyIndex;
                                if (remappedBodyIndex != 0xFFFFFFFF)
                                    applyAcceleration(remappedBodyIndex, bodyVelocities, gravityTimesDt, stepDt, disableGravity, linearAcceleration,
                                                      angularAcceleration, hasLinearAcceleration, hasAngularAcceleration, accelScale, lockFlags, iData0, 32);
                            }
                            if (threadIdxInWorkUnit == 0)
                                applyAcceleration(outputOffset + bodyId, bodyVelocities, gravityTimesDt, stepDt, disableGravity, linearAcceleration,
                                                  angularAcceleration, hasLinearAcceleration, hasAngularAcceleration, accelScale, lockFlags, iData0, allBodiesCount);
                        }
                    }
                }
            }
    return pool;
}

static std::vector<uint8_t> applyGravityLoader(const char* file) { return loadShaderBlob(file); }

template <typename T>
bool sameBytes(const std::vector<T>& a, const std::vector<T>& b)
{
    return a.size() == b.size() && (a.empty() || std::memcmp(a.data(), b.data(), a.size() * sizeof(T)) == 0);
}

// Launch on the engine ABI and return the pool (empty on a launch failure / no BDA).
// Checks the guards and that every input buffer comes back unchanged.
std::vector<F4> runGpu(const GravityCase& c, const char* tag)
{
    PxgSlangComputeDevice* dev = testDevice();
    const PxgKernelDef manifest[] = { { "solverApplyGravityTGS.spv", "applyTGSSubstepGravity", 32u, 0 } };
    PxgCudaShim shim(dev, manifest, 1, &applyGravityLoader);

    // the decoy bodySim pool behind PxgSolverSharedDesc::mBodySimBufferDeviceData (@32):
    // the kernel reads the CORE desc's pool.  In the engine both point at the same buffer,
    // so a kernel reading the shared desc's pool is equivalent there; only the decoy
    // (gravity disabled, no acceleration -> the gate stays shut) can tell them apart.
    std::vector<PxgBodySim> decoy(c.bodySim.size());
    std::memset(decoy.data(), 0, decoy.size() * sizeof(PxgBodySim));
    for (PxgBodySim& s : decoy) s.disableGravity = 1;

    std::vector<F4> poolAll(c.pool.size() + 2 * GUARD, F4{ GUARD_SENT, GUARD_SENT, GUARD_SENT, GUARD_SENT });
    std::memcpy(poolAll.data() + GUARD, c.pool.data(), c.pool.size() * sizeof(F4));

    PxgDevicePtr dSim = 0, dDecoy = 0, dData = 0, dTxI = 0, dRefs = 0, dPool = 0, dIsland = 0, dDesc = 0, dShared = 0;
    shim.memAlloc(&dSim, c.bodySim.size() * sizeof(PxgBodySim));
    shim.memAlloc(&dDecoy, decoy.size() * sizeof(PxgBodySim));
    shim.memAlloc(&dData, c.bodyData.size() * sizeof(PxgSolverBodyData));
    shim.memAlloc(&dTxI, c.txI.size() * sizeof(PxgSolverTxIData));
    shim.memAlloc(&dRefs, (c.refs.empty() ? 1 : c.refs.size()) * 4);
    shim.memAlloc(&dPool, poolAll.size() * sizeof(F4));
    if (!dSim || !dDecoy || !dData || !dTxI || !dRefs || !dPool)
    {
        std::printf("  NOTE: no BDA -> skip\n");
        return {};
    }
    shim.memcpyHtoD(dSim, c.bodySim.data(), c.bodySim.size() * sizeof(PxgBodySim));
    shim.memcpyHtoD(dDecoy, decoy.data(), decoy.size() * sizeof(PxgBodySim));
    shim.memcpyHtoD(dData, c.bodyData.data(), c.bodyData.size() * sizeof(PxgSolverBodyData));
    shim.memcpyHtoD(dTxI, c.txI.data(), c.txI.size() * sizeof(PxgSolverTxIData));
    if (!c.refs.empty()) shim.memcpyHtoD(dRefs, c.refs.data(), c.refs.size() * 4);
    shim.memcpyHtoD(dPool, poolAll.data(), poolAll.size() * sizeof(F4));

    // the REAL engine structs
    PxgIslandContext island;
    std::memset(&island, 0, sizeof(island));
    island.mBodyStartIndex = c.bodyOffset;
    island.mBodyCount = c.allBodiesCount - c.bodyOffset;
    shim.memAlloc(&dIsland, sizeof(island));
    shim.memcpyHtoD(dIsland, &island, sizeof(island));

    PxgSolverCoreDesc desc;
    std::memset(&desc, 0, sizeof(desc));
    desc.islandContextPool = reinterpret_cast<PxgIslandContext*>(dIsland);
    desc.solverBodyReferences = reinterpret_cast<PxgSolverReferences*>(dRefs);
    desc.solverBodyDataPool = reinterpret_cast<PxgSolverBodyData*>(dData);
    desc.solverBodyTxIDataPool = reinterpret_cast<PxgSolverTxIData*>(dTxI);
    desc.mBodySimBufferDeviceData = reinterpret_cast<PxgBodySim*>(dSim);
    desc.numSlabs = c.numSlabs;
    desc.numSolverBodies = c.allBodiesCount;
    desc.accumulatedBodyDeltaVOffset = c.outputOffset;
    shim.memAlloc(&dDesc, sizeof(desc));
    shim.memcpyHtoD(dDesc, &desc, sizeof(desc));

    PxgSolverSharedDesc<IterativeSolveDataTGS> shared;
    std::memset(&shared, 0, sizeof(shared));
    shared.iterativeData.solverBodyVelPool = reinterpret_cast<float4*>(dPool + GUARD * sizeof(F4));
    shared.mBodySimBufferDeviceData = reinterpret_cast<PxgBodySim*>(dDecoy);
    shared.dt = c.dt;
    shared.stepDt = c.stepDt;
    shim.memAlloc(&dShared, sizeof(shared));
    shim.memcpyHtoD(dShared, &shared, sizeof(shared));

    ApplyGravityPrm prm;
    std::memset(&prm, 0, sizeof(prm));
    prm.solverCoreDesc = dDesc;
    prm.sharedDesc = dShared;
    prm.gravity = c.gravity;
    const bool launched = shim.launchKernel(0, engineGrid(c), 1, 1, &prm, sizeof(prm)) == PxgResult::Success;
    PXG_CHECK_MSG(launched, "%s: launch applyTGSSubstepGravity", tag);
    shim.streamSynchronize();

    std::vector<F4> gotAll(poolAll.size());
    shim.memcpyDtoH(gotAll.data(), dPool, gotAll.size() * sizeof(F4));

    // guards: the sentinel must survive bit for bit on both sides
    uint32_t guardHits = 0;
    for (uint32_t g = 0; g < GUARD; ++g)
        for (const F4* e : { &gotAll[g], &gotAll[GUARD + c.pool.size() + g] })
            guardHits += f2u(e->x) != f2u(GUARD_SENT) || f2u(e->y) != f2u(GUARD_SENT) ||
                         f2u(e->z) != f2u(GUARD_SENT) || f2u(e->w) != f2u(GUARD_SENT);
    PXG_CHECK_MSG(guardHits == 0, "%s: %u guard element(s) around the velocity pool written", tag, guardHits);

    // every input buffer must come back unchanged
    std::vector<PxgBodySim> simBack(c.bodySim.size()), decoyBack(decoy.size());
    std::vector<PxgSolverBodyData> dataBack(c.bodyData.size());
    std::vector<PxgSolverTxIData> txiBack(c.txI.size());
    std::vector<uint32_t> refsBack(c.refs.size());
    shim.memcpyDtoH(simBack.data(), dSim, simBack.size() * sizeof(PxgBodySim));
    shim.memcpyDtoH(decoyBack.data(), dDecoy, decoyBack.size() * sizeof(PxgBodySim));
    shim.memcpyDtoH(dataBack.data(), dData, dataBack.size() * sizeof(PxgSolverBodyData));
    shim.memcpyDtoH(txiBack.data(), dTxI, txiBack.size() * sizeof(PxgSolverTxIData));
    if (!refsBack.empty()) shim.memcpyDtoH(refsBack.data(), dRefs, refsBack.size() * 4);
    PxgIslandContext islandBack;
    PxgSolverCoreDesc descBack;
    PxgSolverSharedDesc<IterativeSolveDataTGS> sharedBack;
    shim.memcpyDtoH(&islandBack, dIsland, sizeof(islandBack));
    shim.memcpyDtoH(&descBack, dDesc, sizeof(descBack));
    shim.memcpyDtoH(&sharedBack, dShared, sizeof(sharedBack));
    const bool inputsKept = sameBytes(simBack, c.bodySim) && sameBytes(decoyBack, decoy) && sameBytes(dataBack, c.bodyData) &&
                            sameBytes(txiBack, c.txI) && sameBytes(refsBack, c.refs) &&
                            std::memcmp(&islandBack, &island, sizeof(island)) == 0 &&
                            std::memcmp(&descBack, &desc, sizeof(desc)) == 0 &&
                            std::memcmp(&sharedBack, &shared, sizeof(shared)) == 0;
    PXG_CHECK_MSG(inputsKept, "%s: an input buffer was modified", tag);

    for (PxgDevicePtr p : { dSim, dDecoy, dData, dTxI, dRefs, dPool, dIsland, dDesc, dShared })
        shim.memFree(p);

    return std::vector<F4>(gotAll.begin() + GUARD, gotAll.begin() + GUARD + c.pool.size());
}

uint32_t bitwiseMismatches(const std::vector<F4>& got, const std::vector<F4>& want, const char* tag)
{
    uint32_t mm = 0;
    for (size_t e = 0; e < want.size(); ++e)
    {
        const float* a = &got[e].x;
        const float* r = &want[e].x;
        for (int l = 0; l < 4; ++l)
            if (f2u(a[l]) != f2u(r[l]))
            {
                if (mm < 3)
                    std::printf("  %s: elem %zu lane %d got 0x%08x want 0x%08x\n", tag, e, l, f2u(a[l]), f2u(r[l]));
                ++mm;
            }
    }
    return mm;
}

uint32_t lanesDiffering(const std::vector<F4>& a, const std::vector<F4>& b)
{
    uint32_t n = 0;
    for (size_t e = 0; e < a.size(); ++e)
    {
        const float* x = &a[e].x;
        const float* y = &b[e].x;
        for (int l = 0; l < 4; ++l) n += f2u(x[l]) != f2u(y[l]);
    }
    return n;
}

void setSim(PxgBodySim& s, uint32_t disableGravity, uint32_t lockFlags, float accelScale, const F4& lin, const F4& ang)
{
    std::memset(&s, 0, sizeof(s));
    s.disableGravity = PxU16(disableGravity);
    s.lockFlags = PxU16(lockFlags);
    s.sleepAngVelAccXYZ_accelScaleW.x = 0.125f; // neighbours of the read lane, never read
    s.sleepAngVelAccXYZ_accelScaleW.y = -0.25f;
    s.sleepAngVelAccXYZ_accelScaleW.z = 0.5f;
    s.sleepAngVelAccXYZ_accelScaleW.w = accelScale;
    s.externalLinearAcceleration.x = lin.x; s.externalLinearAcceleration.y = lin.y;
    s.externalLinearAcceleration.z = lin.z; s.externalLinearAcceleration.w = lin.w;
    s.externalAngularAcceleration.x = ang.x; s.externalAngularAcceleration.y = ang.y;
    s.externalAngularAcceleration.z = ang.z; s.externalAngularAcceleration.w = ang.w;
}

void setTxI(PxgSolverTxIData& t, const PxMat33& m, float seed)
{
    std::memset(&t, 0, sizeof(t));
    t.deltaBody2World = PxTransform(PxVec3(seed, -seed, 2.0f * seed), PxQuat(PxIdentity)); // never read
    t.sqrtInvInertia = m;
}

// A randomised case on the engine layout:
//   * node indices are a permutation (nodeIndex != bodyId) with a nonzero mLinkID high word;
//   * every body below bodyOffset still has a live (gravity-on) bodySim record;
//   * the 3-bit gate (disableGravity in {0, 1, 0x8000}) x hasLin x hasAng is cycled;
//   * lockFlags run through all 64 low values, every other one with junk in bits 6..15;
//   * accelScale in {1, 0.75, 0.25+eps, 0, -0};
//   * sqrtInvInertia distinct per index and invertible, the TxI pool exactly allBodiesCount long;
//   * ~15% INVALID references; remap targets 96 apart so the [r, r+32] pairs never alias;
//   * the accumulation copies (outputOffset + bodyId, + allBodiesCount) sit above them.
GravityCase makeRandomCase(uint32_t seed, uint32_t allBodiesCount, uint32_t bodyOffset, uint32_t numSlabs)
{
    std::mt19937 rng(seed);
    auto rf = [&](float lo, float hi) { return lo + (hi - lo) * std::uniform_real_distribution<float>(0.0f, 1.0f)(rng); };

    GravityCase c;
    c.allBodiesCount = allBodiesCount;
    c.bodyOffset = bodyOffset;
    c.numSlabs = numSlabs;
    c.gravity = PxVec3(-9.81f, 0.37f, 1.3e-3f);
    c.stepDt = 1.0f / 240.0f;
    c.dt = 1.0f / 60.0f;

    const uint32_t nodeCount = allBodiesCount + 37;
    std::vector<uint32_t> perm(nodeCount);
    for (uint32_t i = 0; i < nodeCount; ++i) perm[i] = i;
    std::shuffle(perm.begin(), perm.end(), rng);

    const float accelScales[5] = { 1.0f, 0.75f, u2f(f2u(0.25f) + 1u), 0.0f, -0.0f };
    const uint32_t disableGravities[3] = { 0u, 1u, 0x8000u };

    c.bodySim.resize(nodeCount);
    for (uint32_t n = 0; n < nodeCount; ++n) // unused nodes: live decoys
        setSim(c.bodySim[n], 0u, 0u, 1.0f, F4{ 7.5f, 7.5f, 7.5f, 0.0f }, F4{ 3.5f, 3.5f, 3.5f, 0.0f });

    c.bodyData.resize(allBodiesCount);
    c.txI.resize(allBodiesCount);
    for (uint32_t bodyId = 0; bodyId < allBodiesCount; ++bodyId)
    {
        PxgSolverBodyData& d = c.bodyData[bodyId];
        std::memset(&d, 0, sizeof(d));
        d.islandNodeIndex = PxNodeIndex(perm[bodyId], bodyId % 5 + 1);

        const uint32_t k = bodyId;
        const uint32_t gate = (k / 3) % 4;
        const bool wantLin = (gate & 1) != 0, wantAng = (gate & 2) != 0;
        F4 lin{ 0, 0, 0, 0 }, ang{ 0, 0, 0, 0 };
        if (wantLin)
        {
            lin = F4{ rf(-3, 3), rf(-3, 3), rf(-3, 3), rf(-1, 1) };
            if (k % 7 == 1) lin.y = 0.0f;
            if (k % 7 == 2) lin.z = -0.0f;
        }
        if (wantAng)
        {
            ang = F4{ rf(-2, 2), rf(-2, 2), rf(-2, 2), rf(-1, 1) };
            if (k % 5 == 1) ang.x = -0.0f;
            if (k % 5 == 3) ang.y = 0.0f;
        }
        const uint32_t lock = (k % 64) | ((k / 64) % 2 ? 0xFFC0u : 0u);
        setSim(c.bodySim[perm[bodyId]], disableGravities[k % 3], lock, accelScales[k % 5], lin, ang);

        PxMat33 m;
        if (k % 4 == 0) // diagonal (the common case)
            m = PxMat33(PxVec3(rf(0.2f, 2), 0, 0), PxVec3(0, rf(0.2f, 2), 0), PxVec3(0, 0, rf(0.2f, 2)));
        else            // diagonally dominant -> invertible
            m = PxMat33(PxVec3(rf(3, 5), rf(-1, 1), rf(-1, 1)), PxVec3(rf(-1, 1), rf(3, 5), rf(-1, 1)),
                        PxVec3(rf(-1, 1), rf(-1, 1), rf(3, 5)));
        setTxI(c.txI[bodyId], m, float(bodyId));
    }

    const uint32_t nPairs = numSlabs * (allBodiesCount - bodyOffset);
    c.refs.resize(nPairs);
    for (uint32_t i = 0; i < nPairs; ++i)
        c.refs[i] = (rng() % 100 < 15) ? INVALID : 96u * i;
    c.outputOffset = 96u * nPairs + 128u;
    c.pool.resize(c.outputOffset + 2u * allBodiesCount + 64u);
    for (F4& e : c.pool)
    {
        e = F4{ rf(-5, 5), rf(-5, 5), rf(-5, 5), rf(-5, 5) };
        if (rng() % 20 == 0) e.x = -0.0f;
        if (rng() % 20 == 0) e.w = 0.0f;
    }
    return c;
}

// The slots a body's update can touch: its slab copies and its accumulation copies.
std::vector<uint32_t> slotsOf(const GravityCase& c, uint32_t bodyId)
{
    std::vector<uint32_t> s;
    for (uint32_t b = 0; b < c.numSlabs; ++b)
    {
        const uint32_t r = c.refs[c.numSlabs * (bodyId - c.bodyOffset) + b];
        if (r != INVALID) { s.push_back(r); s.push_back(r + 32); }
    }
    s.push_back(c.outputOffset + bodyId);
    s.push_back(c.outputOffset + bodyId + c.allBodiesCount);
    return s;
}

void checkCase(const GravityCase& c, const char* tag)
{
    const std::vector<F4> want = oracle(c);
    const std::vector<F4> got = runGpu(c, tag);
    if (got.empty()) return;
    const uint32_t mm = bitwiseMismatches(got, want, tag);
    PXG_CHECK_MSG(mm == 0, "%s: %u bitwise lane mismatches", tag, mm);
}

PXG_TEST(solverApplyGravityTGS_engineAbi)
{
    GravityCase c = makeRandomCase(95010, 203, 5, 4);

    // directed lane (a): gravity disabled, a NEGATIVE x-only linear acceleration -- the
    // gate is open through hasLinearAcceleration alone
    setSim(c.bodySim[c.bodyData[7].islandNodeIndex.index()], 1u, 0u, 1.0f, F4{ -1.375f, 0.0f, -0.0f, 0.0f }, F4{ 0, 0, 0, 0 });
    // directed lane (b): gravity disabled, a z-only angular acceleration
    setSim(c.bodySim[c.bodyData[8].islandNodeIndex.index()], 1u, 0u, 0.75f, F4{ 0, 0, 0, 0 }, F4{ 0.0f, -0.0f, 2.5f, 0.0f });
    // directed lane (c): diagonal sqrtInvInertia, ang = (1.25, -0, +0), ANG_Y unlocked, pool a.x = -0.
    // getInverse()'s column0.y = invDet * -(c0y*c2z - c2y*c0z) = invDet * -(+0) = -0 and
    // column2.y = -0, so angAcc.y = (-0*1.25 + i1y*-0) + -0*+0 = -0 and a.x stays -0; the
    // spelling invDet * (c2y*c0z - c0y*c2z) gives +0 there.
    setSim(c.bodySim[c.bodyData[9].islandNodeIndex.index()], 0u, 0u, 1.0f, F4{ 0.5f, 0, 0, 0 }, F4{ 1.25f, -0.0f, 0.0f, 0.0f });
    setTxI(c.txI[9], PxMat33(PxVec3(0.3f, 0, 0), PxVec3(0, 0.5f, 0), PxVec3(0, 0, 0.7f)), 9.0f);
    for (uint32_t b = 0; b < c.numSlabs; ++b)
        c.refs[c.numSlabs * (9 - c.bodyOffset) + b] = 96u * (c.numSlabs * (9 - c.bodyOffset) + b);
    for (uint32_t b = 0; b < c.numSlabs; ++b)
        c.pool[c.refs[c.numSlabs * (9 - c.bodyOffset) + b] + 32].x = -0.0f;
    c.pool[c.outputOffset + 9 + c.allBodiesCount].x = -0.0f;
    // directed lanes (d)/(e): PxMat33::getInverse() (PxMat33.h:219) inverts for ANY det != 0,
    // so a NEGATIVE determinant inverts, and a NaN determinant (NaN != 0 is true) takes the
    // inverse branch too and poisons every angular lane.  A `det > 0` spelling sends both to
    // the identity product instead.  Gravity on, unlocked, ang = (1, 1, 1).
    // (d) sqrtInvInertia = diag(-0.3, 0.5, 0.7): det = -0.105
    setSim(c.bodySim[c.bodyData[10].islandNodeIndex.index()], 0u, 0u, 1.0f, F4{ 0, 0, 0, 0 }, F4{ 1.0f, 1.0f, 1.0f, 0.0f });
    setTxI(c.txI[10], PxMat33(PxVec3(-0.3f, 0, 0), PxVec3(0, 0.5f, 0), PxVec3(0, 0, 0.7f)), 10.0f);
    // (e) sqrtInvInertia = diag(0.3, 0.5, 0.7) with column1.x = NaN: det = NaN
    setSim(c.bodySim[c.bodyData[11].islandNodeIndex.index()], 0u, 0u, 1.0f, F4{ 0, 0, 0, 0 }, F4{ 1.0f, 1.0f, 1.0f, 0.0f });
    setTxI(c.txI[11], PxMat33(PxVec3(0.3f, 0, 0), PxVec3(std::numeric_limits<float>::quiet_NaN(), 0.5f, 0), PxVec3(0, 0, 0.7f)), 11.0f);

    const std::vector<F4> want = oracle(c);
    // the directed lanes must be live in the oracle
    const std::vector<F4>& init = c.pool;
    PXG_CHECK_MSG(f2u(want[c.outputOffset + 7].x) != f2u(init[c.outputOffset + 7].x), "lane (a) not updated by the oracle");
    PXG_CHECK_MSG(f2u(want[c.outputOffset + 8 + c.allBodiesCount].y) != f2u(init[c.outputOffset + 8 + c.allBodiesCount].y),
                  "lane (b) not updated by the oracle");
    PXG_CHECK_MSG(f2u(want[c.outputOffset + 9 + c.allBodiesCount].x) == 0x80000000u, "lane (c): expected a.x = -0");
    // (d): the inverse's x lane is 1/-0.3 * 1 < 0 (the identity product would add +stepDt)
    PXG_CHECK_MSG(want[c.outputOffset + 10].w < init[c.outputOffset + 10].w, "lane (d): negative-det inverse not taken by the oracle");
    // (e): the NaN determinant takes the inverse branch -> NaN angular lanes (identity: finite)
    PXG_CHECK_MSG(std::isnan(want[c.outputOffset + 11].w) && std::isnan(want[c.outputOffset + 11 + c.allBodiesCount].x) &&
                      std::isnan(want[c.outputOffset + 11 + c.allBodiesCount].y),
                  "lane (e): NaN-det inverse not taken by the oracle");

    const std::vector<F4> got = runGpu(c, "engineAbi");
    if (got.empty()) return;
    const uint32_t mm = bitwiseMismatches(got, want, "engineAbi");
    PXG_CHECK_MSG(mm == 0, "applyTGSSubstepGravity engine ABI: %u bitwise lane mismatches", mm);
}

// numSlabs across the work-unit geometry, each at the unscaled engine grid: 1 (32 bodies per
// warp), 5 and 12 (non-powers of two -> 8 and 16 lanes per body), 32, and 33 (32 lanes, the
// strided slab loop makes two passes).
PXG_TEST(solverApplyGravityTGS_engineSlabCoverage)
{
    const uint32_t slabs[] = { 1, 2, 4, 5, 12, 32, 33 };
    for (uint32_t i = 0; i < sizeof(slabs) / sizeof(slabs[0]); ++i)
    {
        GravityCase c = makeRandomCase(95020 + i, 75, 3, slabs[i]);
        char tag[64];
        std::snprintf(tag, sizeof(tag), "slabCoverage(numSlabs=%u)", slabs[i]);
        checkCase(c, tag);
    }
}

// Signed zeros, NaN and inf in the acceleration lanes: {-0, +0} lanes leave hasLin/hasAng
// false (-0 != 0 is false), so a gravity-disabled body is untouched and a gravity-enabled one
// gets gravity only; a NaN lane opens the gate (NaN != 0) and propagates; an inf angular lane
// turns the inverse product's other lanes into NaN.
PXG_TEST(solverApplyGravityTGS_engineAccelZeroNaN)
{
    GravityCase c = makeRandomCase(95030, 40, 2, 2);
    const float qnan = std::numeric_limits<float>::quiet_NaN();
    const float inf = std::numeric_limits<float>::infinity();
    struct Lane { uint32_t body, dg; F4 lin, ang; };
    const Lane lanes[] = {
        { 2, 1u, F4{ -0.0f, 0.0f, -0.0f, 5.0f }, F4{ 0.0f, -0.0f, -0.0f, 5.0f } },   // gate shut: untouched
        { 3, 0u, F4{ -0.0f, -0.0f, 0.0f, 0.0f }, F4{ -0.0f, 0.0f, 0.0f, 0.0f } },    // gravity only, no angular
        { 4, 1u, F4{ qnan, 0.0f, 0.0f, 0.0f }, F4{ 0, 0, 0, 0 } },                   // NaN opens the gate
        { 5, 1u, F4{ 0, 0, 0, 0 }, F4{ 0.0f, qnan, 0.0f, 0.0f } },                   // NaN angular lane
        { 6, 0u, F4{ inf, -inf, 0.0f, 0.0f }, F4{ 0, 0, 0, 0 } },                    // +-inf linear
        { 7, 1u, F4{ 0, 0, 0, 0 }, F4{ -inf, 0.0f, 0.0f, 0.0f } },                   // inf angular
    };
    for (const Lane& l : lanes)
        setSim(c.bodySim[c.bodyData[l.body].islandNodeIndex.index()], l.dg, 0u, 1.0f, l.lin, l.ang);

    const std::vector<F4> want = oracle(c);
    for (uint32_t s : slotsOf(c, 2))
        PXG_CHECK_MSG(std::memcmp(&want[s], &c.pool[s], sizeof(F4)) == 0, "zero lanes: body 2 slot %u must be untouched", s);
    PXG_CHECK_MSG(std::isnan(want[c.outputOffset + 4].x), "NaN lane must propagate");
    checkCase(c, "engineAccelZeroNaN");
}

// UPSTREAM QUIRK, pinned: the gate reads disableGravity (solverMultiBlockTGS.cu:1656) but
// applyAcceleration never does (:1575-1583), so a gravity-disabled body with an external
// acceleration also receives accelScale*gravity*stepDt.  The CPU twin zeroes it
// (DyTGSDynamics.cpp:1490-1493).  The port reproduces the CUDA.
PXG_TEST(solverApplyGravityTGS_engineDisableGravityQuirk)
{
    GravityCase c = makeRandomCase(95040, 24, 1, 2);
    for (uint32_t bodyId = 1; bodyId < c.allBodiesCount; ++bodyId) // all bodies lock-free here
        c.bodySim[c.bodyData[bodyId].islandNodeIndex.index()].lockFlags = 0;
    setSim(c.bodySim[c.bodyData[1].islandNodeIndex.index()], 1u, 0u, 1.0f, F4{ 0.75f, 0, 0, 0 }, F4{ 0, 0, 0, 0 });  // (a) lin only
    setSim(c.bodySim[c.bodyData[2].islandNodeIndex.index()], 1u, 0u, 1.0f, F4{ 0, 0, 0, 0 }, F4{ 0, 0.5f, 0, 0 });   // (b) ang only
    setSim(c.bodySim[c.bodyData[3].islandNodeIndex.index()], 1u, 0u, 1.0f, F4{ 0, 0, 0, 0 }, F4{ 0, 0, 0, 0 });      // (c) neither

    const std::vector<F4> want = oracle(c);
    gDisableGravityFixVariant = 1;
    const std::vector<F4> fix = oracle(c);
    gDisableGravityFixVariant = 0;

    // (a)/(b): the CUDA adds gravity, so the CPU-semantics twin must differ in their y lanes
    const uint32_t accA = c.outputOffset + 1, accB = c.outputOffset + 2;
    PXG_CHECK_MSG(f2u(want[accA].y) != f2u(fix[accA].y) && f2u(want[accB].y) != f2u(fix[accB].y),
                  "disableGravity twin does not change bodies (a)/(b) -- the case does not observe the quirk");
    PXG_CHECK_MSG(lanesDiffering(want, fix) > 0, "disableGravity twin changes no lane");
    for (uint32_t s : slotsOf(c, 3))
        PXG_CHECK_MSG(std::memcmp(&want[s], &c.pool[s], sizeof(F4)) == 0, "(c): body 3 slot %u must be untouched", s);

    const std::vector<F4> got = runGpu(c, "engineDisableGravityQuirk");
    if (got.empty()) return;
    const uint32_t mm = bitwiseMismatches(got, want, "engineDisableGravityQuirk");
    PXG_CHECK_MSG(mm == 0, "disableGravity quirk: %u bitwise lane mismatches (%u lanes would differ under the CPU fix)",
                  mm, lanesDiffering(want, fix));
    PXG_CHECK_MSG(lanesDiffering(got, fix) > 0, "GPU matches the CPU-semantics twin, not the CUDA");
}

// P9.86c: singular sqrt-inverse-inertia -> getInverse() returns PxMat33(PxIdentity)
// (solverMultiBlockTGS.cu:1592, PxMat33.h:240/:346).  Lane i of the identity product is
// (1*v_i + 0*v_j) + 0*v_k, so a -0 lane next to a +0/positive lane becomes +0.  Three bodies
// with one zero inverse-inertia axis at a 180-degree orientation and a direct-GPU-API torque
// (setRigidDynamicTorque, updateBodiesAndShapes.cu:1269-1276) carry a -0 angular-acceleration
// lane; their momocity lane is -0 (sqrtInertia * angVel of the same body).  The CUDA stores
// -0 + (+0*dt) = +0; the old port shortcut stores -0.  Compared BITWISE.
PXG_TEST(solverApplyGravityTGS_applySingularInertiaZeroSign)
{
    const float s5 = std::sqrt(5.f);
    GravityCase c;
    c.allBodiesCount = 4; c.bodyOffset = 1; c.numSlabs = 2; c.outputOffset = 64;
    c.gravity = PxVec3(0.f, -9.81f, 0.3f);
    c.stepDt = 0.016f; c.dt = 0.064f;
    const uint32_t nodeIndex[4] = { 3, 2, 1, 0 };
    c.bodySim.resize(4);
    c.bodyData.resize(4);
    c.txI.resize(4);
    for (uint32_t i = 0; i < 4; ++i)
    {
        std::memset(&c.bodyData[i], 0, sizeof(PxgSolverBodyData));
        c.bodyData[i].islandNodeIndex = PxNodeIndex(nodeIndex[i]);
    }
    // body 0: below bodyOffset, skipped
    setTxI(c.txI[0], PxMat33(PxIdentity), 0.0f);
    setSim(c.bodySim[nodeIndex[0]], 0u, 0u, 1.f, F4{ 1, 1, 1, 0 }, F4{ 1, 1, 1, 0 });
    const uint32_t lockLin = PxRigidDynamicLockFlag::eLOCK_LINEAR_X | PxRigidDynamicLockFlag::eLOCK_LINEAR_Y |
                             PxRigidDynamicLockFlag::eLOCK_LINEAR_Z; // no angular lock
    // A: invInertia (0,0,5), q = (0,0,-1,0), torque (-1,0,-1), angVel (-1,0,-1)
    setTxI(c.txI[1], PxMat33(PxVec3(0.f, -0.f, 0.f), PxVec3(-0.f, 0.f, -0.f), PxVec3(0.f, -0.f, s5)), 1.0f);
    setSim(c.bodySim[nodeIndex[1]], 1u, lockLin, 1.f, F4{ 0, 0, 0, 0 }, F4{ -0.f, 0.f, -5.f, 0 });
    // B: invInertia (5,0,0), q = (-1,0,0,0), torque (-1,-1,+0), angVel (-1,-1,0)
    setTxI(c.txI[2], PxMat33(PxVec3(s5, 0.f, -0.f), PxVec3(0.f, 0.f, -0.f), PxVec3(-0.f, -0.f, 0.f)), 2.0f);
    setSim(c.bodySim[nodeIndex[2]], 1u, lockLin, 1.f, F4{ 0, 0, 0, 0 }, F4{ -5.f, -0.f, 0.f, 0 });
    // C: invInertia (0,5,0), q = (0,-1,0,0), torque (1,-1,-1), angVel (1,-1,-1)
    setTxI(c.txI[3], PxMat33(PxVec3(0.f, -0.f, -0.f), PxVec3(-0.f, s5, 0.f), PxVec3(-0.f, 0.f, 0.f)), 3.0f);
    setSim(c.bodySim[nodeIndex[3]], 1u, lockLin, 1.f, F4{ 0, 0, 0, 0 }, F4{ 0.f, -5.f, -0.f, 0 });

    c.refs = { 0, 4, 8, 12, 16, 20 };
    c.pool.resize(80);
    for (uint32_t e = 0; e < 80; ++e) c.pool[e] = F4{ 100.f + e, 200.f + e, 300.f + e, 400.f + e };
    // body 1 (A): linear 0,4,65 (w = -0); angular 32,36,69
    for (uint32_t e : { 0u, 4u, 65u }) c.pool[e] = F4{ 0.f, 0.f, 0.f, -0.f };
    for (uint32_t e : { 32u, 36u, 69u }) c.pool[e] = F4{ 0.f, 0.f, 0.25f, -0.5f };
    // body 2 (B): linear 8,12,66; angular 40,44,70 (x = -0)
    for (uint32_t e : { 8u, 12u, 66u }) c.pool[e] = F4{ 0.f, 0.f, 0.f, 0.f };
    for (uint32_t e : { 40u, 44u, 70u }) c.pool[e] = F4{ -0.f, 0.f, 0.25f, -0.5f };
    // body 3 (C): linear 16,20,67; angular 48,52,71 (y = -0)
    for (uint32_t e : { 16u, 20u, 67u }) c.pool[e] = F4{ 0.f, 0.f, 0.f, 0.f };
    for (uint32_t e : { 48u, 52u, 71u }) c.pool[e] = F4{ 0.f, -0.f, 0.25f, -0.5f };

    gSingularCalls = 0;
    const std::vector<F4> want = oracle(c);
    const uint32_t singular = gSingularCalls;
    gIdentityMulVariant = 1;
    const std::vector<F4> twin = oracle(c);
    gIdentityMulVariant = 0;
    gSingularCalls = 0;

    PXG_CHECK_MSG(singular == 9, "singular identity branch taken %u times, expected 9", singular);
    uint32_t zsA = 0, zsB = 0, zsC = 0;
    for (uint32_t e : { 0u, 4u, 65u }) zsA += (f2u(want[e].w) == 0u && f2u(twin[e].w) == 0x80000000u);
    for (uint32_t e : { 40u, 44u, 70u }) zsB += (f2u(want[e].x) == 0u && f2u(twin[e].x) == 0x80000000u);
    for (uint32_t e : { 48u, 52u, 71u }) zsC += (f2u(want[e].y) == 0u && f2u(twin[e].y) == 0x80000000u);
    PXG_CHECK_MSG(zsA == 3 && zsB == 3 && zsC == 3,
                  "mutant twin (identity shortcut) no longer changes lanes A.w/B.x/C.y (%u/%u/%u of 3) -- the case does not observe it",
                  zsA, zsB, zsC);
    PXG_CHECK_MSG(lanesDiffering(want, twin) == 9, "mutant twin differs in %u lanes, expected 9", lanesDiffering(want, twin));

    const std::vector<F4> got = runGpu(c, "singularZeroSign");
    if (got.empty()) return;
    const uint32_t mm = bitwiseMismatches(got, want, "singularZeroSign");
    PXG_CHECK_MSG(mm == 0, "applyTGSSubstepGravity singular zero-sign: %u bitwise lane mismatches", mm);
}

} // namespace

// P9.90: NoContraction stamp.  getInverse()'s determinant is column0.dot(column1.cross(column2))
// (PxMat33.h:247), a PxVec3 dot (x*x' + y*y') + z*z'; unstamped, llvmpipe evaluates an OpDot
// as x + (y + z) and the reciprocal feeds every angular-acceleration lane.  256 bodies with
// generally oriented, non-diagonal sqrt-inverse-inertia matrices and non-round accelerations; the
// whole velocity buffer is compared BITWISE against the -ffp-contract=off oracle.  An oracle twin
// with the determinant summed x + (y + z) must change angular lanes, so the case observes the order.
PXG_TEST(solverApplyGravityTGS_p90DotOrder)
{
    GravityCase c = makeRandomCase(9090, 256, 0, 2);
    std::mt19937 rng(9091);
    auto rf = [&](float lo, float hi) { return lo + (hi - lo) * std::uniform_real_distribution<float>(0.f, 1.f)(rng); };
    c.gravity = PxVec3(0.137f, -9.8127f, 0.2913f);
    for (uint32_t i = 0; i < c.allBodiesCount; ++i)
    {
        setSim(c.bodySim[c.bodyData[i].islandNodeIndex.index()], i % 5 == 0 ? 1u : 0u, 0u, rf(0.7f, 1.3f),
               F4{ rf(-3, 3), rf(-3, 3), rf(-3, 3), 0 }, F4{ rf(-2, 2), rf(-2, 2), rf(-2, 2), 0 });
        // R * diag(d) * R^T with R from a random unit quaternion (double, then rounded)
        double q[4]; double n = 0;
        for (double& x : q) { x = rf(-1, 1); n += x * x; }
        n = std::sqrt(n); for (double& x : q) x /= n;
        const double x = q[0], y = q[1], z = q[2], w = q[3];
        const double R[3][3] = { { 1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w) },
                                 { 2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w) },
                                 { 2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y) } };
        const double d[3] = { rf(0.3f, 3.f), rf(0.3f, 3.f), rf(0.3f, 3.f) };
        float M[3][3];
        for (int r = 0; r < 3; ++r)
            for (int k = 0; k < 3; ++k)
                M[r][k] = float(R[r][0] * d[0] * R[k][0] + R[r][1] * d[1] * R[k][1] + R[r][2] * d[2] * R[k][2]);
        setTxI(c.txI[i], PxMat33(PxVec3(M[0][0], M[1][0], M[2][0]), PxVec3(M[0][1], M[1][1], M[2][1]),
                                 PxVec3(M[0][2], M[1][2], M[2][2])), float(i));
    }
    // small initial velocities so a 1-ulp determinant difference survives the += into them
    for (F4& e : c.pool) e = F4{ rf(-0.01f, 0.01f), rf(-0.01f, 0.01f), rf(-0.01f, 0.01f), rf(-0.01f, 0.01f) };

    const std::vector<F4> want = oracle(c);
    gDetDotVariant = 1;
    const std::vector<F4> twin = oracle(c);
    gDetDotVariant = 0;
    const uint32_t twinLanes = lanesDiffering(want, twin);
    PXG_CHECK_MSG(twinLanes >= 64, "determinant-dot twin x+(y+z) changes only %u lanes -- the case does not observe the order", twinLanes);

    const std::vector<F4> got = runGpu(c, "p90DotOrder");
    if (got.empty()) return;
    uint32_t mm = 0, eqTwin = 0;
    for (size_t e = 0; e < want.size(); ++e)
    {
        const float* a = &got[e].x; const float* r = &want[e].x; const float* t = &twin[e].x;
        for (int l = 0; l < 4; ++l)
            if (f2u(a[l]) != f2u(r[l]))
            {
                if (mm == 0)
                    std::printf("  first mismatch: elem %zu lane %d got 0x%08x want 0x%08x\n", e, l, f2u(a[l]), f2u(r[l]));
                ++mm;
                eqTwin += f2u(a[l]) == f2u(t[l]);
            }
    }
    PXG_CHECK_MSG(mm == 0, "applyTGSSubstepGravity p90 dot order: %u bitwise lane mismatches (%u equal the x+(y+z) twin; twin changes %u)",
                  mm, eqTwin, twinLanes);
}
