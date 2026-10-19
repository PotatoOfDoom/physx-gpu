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

// pxgParticleSolveRef.h -- the host oracle of the PhysX 5.12 particle/rigid contact
// solve, transcribed from the CUDA (NOT from the Slang port):
//
//   gpusimulationcontroller/src/CUDA/deformableAndParticleUtils.cuh
//       PxgDbContactPair, PxgDbSolveOutput::computeDelta, getGlobalRigidBodyId,
//       rigidIsRefCounted, PxgRigidPart (readBodyProperties, readNodelessStatic,
//       readContactPrep, writeContactDeltas), readParticleLinDelta,
//       PxgDeformablePart<PxVec3>::readParticle / writeParticle,
//       particleAdhesionImpulse, solveRigidDbContact<PxVec3, withAdhesion>
//   gpunarrowphase/src/CUDA/dataReadWriteHelper.cuh   PxgVelocityReader
//   lowlevel/software/include/PxsMaterialCombiner.h   combineScalars
//
// Every expression keeps the CUDA association order (PxVec3::dot is (x + y) + z);
// the test target builds with -ffp-contract=off, so nothing here is contracted.
// PxMax / PxMin on floats are intrinsics::selectMax / selectMin.

#ifndef PXG_PARTICLE_SOLVE_REF_H
#define PXG_PARTICLE_SOLVE_REF_H

#include <cmath>
#include <cstdint>
#include <cstring>

namespace pxgpsref
{

struct V3 { float x, y, z; };
struct F2 { float x, y; };
struct F4 { float x, y, z, w; };

inline V3 v3(float x, float y, float z) { V3 r = { x, y, z }; return r; }
inline V3 xyz(const F4& f) { return v3(f.x, f.y, f.z); }
inline V3 add(V3 a, V3 b) { return v3(a.x + b.x, a.y + b.y, a.z + b.z); }
inline V3 sub(V3 a, V3 b) { return v3(a.x - b.x, a.y - b.y, a.z - b.z); }
inline V3 mul(V3 a, float s) { return v3(a.x * s, a.y * s, a.z * s); }
inline V3 neg(V3 a) { return v3(-a.x, -a.y, -a.z); }
inline float dot(V3 a, V3 b) { return a.x * b.x + a.y * b.y + a.z * b.z; }
inline V3 cross(V3 a, V3 b) { return v3(a.y * b.z - a.z * b.y, a.z * b.x - a.x * b.z, a.x * b.y - a.y * b.x); }
inline float selectMax(float a, float b) { return a > b ? a : b; }
inline float selectMin(float a, float b) { return a < b ? a : b; }

// PxVec3::normalize: one reciprocal, three multiplies; a zero vector is untouched.
inline float normalize(V3& v)
{
    const float m = std::sqrt(dot(v, v));
    if (m > 0.0f)
    {
        const float r = 1.0f / m;
        v = v3(v.x * r, v.y * r, v.z * r);
    }
    return m;
}

// PxComputeBasisVectors (PxMathUtils.h).
inline void computeBasisVectors(V3 dir, V3& right, V3& up)
{
    if (std::fabs(dir.y) <= 0.9999f)
    {
        right = v3(dir.z, 0.0f, -dir.x);
        normalize(right);
        up = v3(dir.y * right.z, dir.z * right.x - dir.x * right.z, -dir.y * right.x);
    }
    else
    {
        right = v3(1.0f, 0.0f, 0.0f);
        up = v3(0.0f, dir.z, -dir.y);
        normalize(up);
    }
}

// PxMat33 (column major) * PxVec3.
inline V3 mat33Mul(const float m[9], V3 v)
{
    return v3(m[0] * v.x + m[3] * v.y + m[6] * v.z,
              m[1] * v.x + m[4] * v.y + m[7] * v.z,
              m[2] * v.x + m[5] * v.y + m[8] * v.z);
}

// PxSpatialMatrix (column[i][j] at e[i*6+j]) * Cm::UnAlignedSpatialVector(top, bottom):
// result[i] = 0.f + column[0][i]*s[0] + ... + column[5][i]*s[5].
inline void spatialMul(const float e[36], V3 top, V3 bottom, V3& outTop, V3& outBottom)
{
    const float s[6] = { top.x, top.y, top.z, bottom.x, bottom.y, bottom.z };
    float r[6];
    for (int i = 0; i < 6; ++i)
    {
        float acc = 0.0f + e[0 * 6 + i] * s[0];
        for (int j = 1; j < 6; ++j)
            acc = acc + e[j * 6 + i] * s[j];
        r[i] = acc;
    }
    outTop = v3(r[0], r[1], r[2]);
    outBottom = v3(r[3], r[4], r[5]);
}

// combineScalars (PxsMaterialCombiner.h), PxCombineMode eAVERAGE 0 / eMIN 1 / eMULTIPLY 2 / eMAX 3.
inline float combineScalars(float a, float b, int mode)
{
    switch (mode)
    {
    case 0: return 0.5f * (a + b);
    case 1: return selectMin(a, b);
    case 2: return a * b;
    case 3: return selectMax(a, b);
    default: return 0.0f;
    }
}

// PxNodeIndex as its packed PxU64: low = mID, high = mLinkID (bit 0 = articulation).
inline bool nodeIsStatic(uint64_t n) { return uint32_t(n) == 0xffffffffu; }
inline uint32_t nodeIndex(uint64_t n) { return uint32_t(n); }
inline bool nodeIsArticulation(uint64_t n) { return (uint32_t(n >> 32) & 1u) != 0u; }
inline uint32_t nodeLinkId(uint64_t n) { return uint32_t(n >> 32) >> 1; }
inline uint64_t makeRigidNode(uint32_t index) { return uint64_t(index); }
inline uint64_t makeArticulationNode(uint32_t index, uint32_t linkId)
{
    return uint64_t(index) | (uint64_t((linkId << 1) | 1u) << 32);
}
inline uint64_t staticNode() { return uint64_t(0xffffffffu); }

// PxgParticleRef (PxgSimulationCoreDesc.h): [system:30 @34][flags:2][index:32].
inline uint64_t particleRef(uint32_t systemId, uint32_t index) { return (uint64_t(systemId) << 34) | uint64_t(index); }
inline uint32_t particleSystemIdOf(uint64_t ref) { return uint32_t(ref >> 34); }
inline uint32_t particleIndexOf(uint64_t ref) { return uint32_t(ref); }

// ---------------------------------------------------------------------------
// PxgRigidParticleContactBlock lane, as the prepare writes it.
// ---------------------------------------------------------------------------
struct BlockLane
{
    F4 raXn_resp, raXnF0_resp, raXnF1_resp, normal_errorW, fricTan0_invMass0;
    float penBiasClampRigid;
};

// ---------------------------------------------------------------------------
// The velocity reader.
// ---------------------------------------------------------------------------
struct VelocityReader
{
    const uint32_t* solverBodyIndices;
    const F4* artiLinkVelocities;
    uint32_t maxLinks, nbArticulations, artiOffset;
    uint32_t numSolverBodies;
    const F4* solverBodyDeltaVel;   // velPool + accumulatedBodyDeltaVOffset
    const F4* initialVel;

    void readPGS(uint64_t rigidId, V3& linVel, V3& angVel, V3& linDelta, V3& angDelta) const
    {
        const uint32_t idx = nodeIsStatic(rigidId) ? 0u : solverBodyIndices[nodeIndex(rigidId)];
        if (nodeIsArticulation(rigidId))
        {
            const F4* vels = &artiLinkVelocities[idx * maxLinks];
            const uint32_t linkID = nodeLinkId(rigidId);
            linVel = xyz(vels[linkID]);
            angVel = xyz(vels[linkID + artiOffset]);
        }
        else
        {
            linVel = add(xyz(initialVel[idx]), xyz(solverBodyDeltaVel[idx]));
            angVel = add(xyz(initialVel[idx + numSolverBodies]), xyz(solverBodyDeltaVel[idx + numSolverBodies]));
        }
        linDelta = v3(0.0f, 0.0f, 0.0f);
        angDelta = v3(0.0f, 0.0f, 0.0f);
    }

    void readTGS(uint64_t rigidId, V3& linVel, V3& angVel, V3& linDelta, V3& angDelta) const
    {
        const uint32_t idx = nodeIsStatic(rigidId) ? 0u : solverBodyIndices[nodeIndex(rigidId)];
        if (nodeIsArticulation(rigidId))
        {
            const F4* vels = &artiLinkVelocities[idx * maxLinks];
            const uint32_t linkID = nodeLinkId(rigidId);
            linVel = xyz(vels[linkID]);
            angVel = xyz(vels[linkID + artiOffset]);
            linDelta = xyz(vels[linkID + artiOffset + artiOffset]);
            angDelta = xyz(vels[linkID + artiOffset + artiOffset + artiOffset]);
        }
        else
        {
            const F4 vel0 = solverBodyDeltaVel[idx];
            const F4 vel1 = solverBodyDeltaVel[idx + numSolverBodies];
            const F4 vel2 = solverBodyDeltaVel[idx + numSolverBodies + numSolverBodies];
            linVel = v3(vel0.x, vel0.y, vel0.z);
            angVel = v3(vel0.w, vel1.x, vel1.y);
            linDelta = v3(vel1.z, vel1.w, vel2.x);
            angDelta = v3(vel2.y, vel2.z, vel2.w);
        }
    }
};

inline int globalRigidBodyId(const uint32_t* solverBodyIndices, uint64_t rigidId, uint32_t numSolverBodies,
                             uint32_t maxLinksPerArticulation)
{
    if (nodeIsStatic(rigidId))
        return -1;
    const uint32_t solverBodyIdx = solverBodyIndices[nodeIndex(rigidId)];
    return nodeIsArticulation(rigidId)
        ? int(numSolverBodies + solverBodyIdx * maxLinksPerArticulation + nodeLinkId(rigidId))
        : int(solverBodyIdx);
}

inline bool rigidIsRefCounted(int globalRigidBodyId, float rigidInvMass)
{
    return globalRigidBodyId != -1 && rigidInvMass != 0.0f;
}

// ---------------------------------------------------------------------------
// The solve parts.
// ---------------------------------------------------------------------------
struct ContactPair
{
    V3 normal = { 0.0f, 0.0f, 0.0f };
    V3 tangent0 = { 0.0f, 0.0f, 0.0f };
    float initPen = 0.0f;
    float maxPenBiasClamp = 0.0f;

    void readRigidParticle(const BlockLane& b, float particlePenBiasClamp)
    {
        normal = xyz(b.normal_errorW);
        initPen = b.normal_errorW.w;
        tangent0 = xyz(b.fricTan0_invMass0);
        maxPenBiasClamp = selectMax(b.penBiasClampRigid, particlePenBiasClamp);
    }
};

struct SolveOut
{
    V3 tangent = { 0.0f, 0.0f, 0.0f };
    V3 raXt = { 0.0f, 0.0f, 0.0f };
    float deltaLambdaN = 0.0f;
    float deltaLambdaT = 0.0f;
    float accumulatedDeltaLambdaN = 0.0f;

    float computeDelta(const ContactPair& cp, V3& delta, float dt) const
    {
        delta = mul(neg(add(mul(cp.normal, deltaLambdaN), mul(tangent, deltaLambdaT))), dt);
        return accumulatedDeltaLambdaN;
    }
};

struct RigidPart
{
    uint32_t referenceCount = 1;
    float friction = 0.0f;
    int fricCombineMode = 3;
    V3 raXn = { 0, 0, 0 }, raXnF0 = { 0, 0, 0 }, raXnF1 = { 0, 0, 0 };
    float raXnResp = 0.0f, raXnF0Resp = 0.0f, raXnF1Resp = 0.0f, invMass = 0.0f;
    V3 linVel = { 0, 0, 0 }, angVel = { 0, 0, 0 }, linDelta = { 0, 0, 0 }, angDelta = { 0, 0, 0 };

    // The particle paths pass a NULL rigid material.
    void readBodyProperties(int gid, float rigidInvMass, const uint32_t* refCounts)
    {
        referenceCount = 1;
        if (refCounts && rigidIsRefCounted(gid, rigidInvMass))
            referenceCount = refCounts[gid];
        friction = 0.0f;
        fricCombineMode = 3;
    }

    void readNodelessStatic()
    {
        referenceCount = 1;
        friction = 0.0f;
        fricCombineMode = 3;
        raXn = raXnF0 = raXnF1 = v3(0.0f, 0.0f, 0.0f);
        raXnResp = raXnF0Resp = raXnF1Resp = 0.0f;
        invMass = 0.0f;
        linVel = angVel = linDelta = angDelta = v3(0.0f, 0.0f, 0.0f);
    }

    void readContactPrep(const BlockLane& b)
    {
        raXn = xyz(b.raXn_resp);
        raXnResp = b.raXn_resp.w;
        raXnF0 = xyz(b.raXnF0_resp);
        raXnF0Resp = b.raXnF0_resp.w;
        raXnF1 = xyz(b.raXnF1_resp);
        raXnF1Resp = b.raXnF1_resp.w;
        invMass = b.fricTan0_invMass0.w;
    }

    void readVelocity(const VelocityReader& r, uint64_t rigidId, bool isTGS)
    {
        if (isTGS)
            r.readTGS(rigidId, linVel, angVel, linDelta, angDelta);
        else
            r.readPGS(rigidId, linVel, angVel, linDelta, angDelta);
    }

    void writeContactDeltas(F4* rigidDeltaVel, uint64_t rigidId, const ContactPair& cp, const SolveOut& so,
                            uint32_t linIdx, uint32_t angIdx) const
    {
        V3 dLin = v3(0.0f, 0.0f, 0.0f), dAng = v3(0.0f, 0.0f, 0.0f);
        float count = 0.0f;
        if (invMass != 0.0f && !nodeIsStatic(rigidId))
        {
            count = 1.0f;
            const float refCount = float(referenceCount);
            dAng = mul(add(mul(raXn, so.deltaLambdaN), mul(so.raXt, so.deltaLambdaT)), refCount);
            dLin = mul(mul(add(mul(cp.normal, so.deltaLambdaN), mul(so.tangent, so.deltaLambdaT)), invMass), refCount);
        }
        rigidDeltaVel[linIdx] = F4{ dLin.x, dLin.y, dLin.z, count };
        rigidDeltaVel[angIdx] = F4{ dAng.x, dAng.y, dAng.z, 0.0f };
    }
};

struct ParticleView
{
    // The fields readParticle consumes, for one particle.
    F4 sortedDeltaP;
    F4 sortedVelocity;
    F4 accumDeltaP;
    float restOffset;
    float friction, adhesion, adhesionRadiusScale;
};

struct DeformablePart
{
    float friction = 0.0f, adhesion = 0.0f, adhesionRadius = 0.0f;
    V3 linDelta = { 0, 0, 0 };
    V3 posDelta = { 0, 0, 0 };
    V3 vertexInvMasses = { 0, 0, 0 };
    V3 bc = { 0, 0, 0 };

    void readParticle(const ParticleView& p, bool hasMaterial, bool inflateByRefCount, float dt, bool isVelocityIteration)
    {
        bc = v3(1.0f, 0.0f, 0.0f);
        const float refCount = inflateByRefCount ? selectMax(p.accumDeltaP.w, 1.0f) : 1.0f;
        vertexInvMasses = v3(p.sortedDeltaP.w * refCount, 0.0f, 0.0f);
        if (hasMaterial)
        {
            friction = p.friction;
            adhesion = p.adhesion;
            adhesionRadius = p.restOffset * (p.adhesionRadiusScale - 1.0f);
        }
        linDelta = isVelocityIteration ? mul(xyz(p.sortedVelocity), dt) : xyz(p.sortedDeltaP);
        posDelta = xyz(p.sortedDeltaP);
    }

    F4 writeParticle(V3 delta, const SolveOut& so) const
    {
        const V3 s = mul(delta, vertexInvMasses.x);
        const float applied = (so.deltaLambdaN != 0.0f || so.deltaLambdaT != 0.0f) ? 1.0f : 0.0f;
        return F4{ s.x, s.y, s.z, applied };
    }
};

inline float particleAdhesionImpulse(float separation, float adhesion, float adhesionRadius, float velMultiplier,
                                     float invDt, bool isVelocityIteration)
{
    if (isVelocityIteration || adhesion <= 0.f || adhesionRadius <= 0.f)
        return 0.f;
    const float t = 1.f - separation / adhesionRadius;
    return -selectMax(0.f, adhesion * (t * t * t * t) * selectMax(0.f, separation) * velMultiplier * invDt);
}

inline bool solveRigidDbContact(const RigidPart& rigid, const DeformablePart& db, const ContactPair& cp, SolveOut& so,
                                float appliedForceRef, float appliedTanLambdaRef, float dt, bool wasActive,
                                bool checkOnlyActivity, bool isVelocityIteration, bool withAdhesion)
{
    so.accumulatedDeltaLambdaN = appliedForceRef;
    const float appliedTanRef = appliedTanLambdaRef;
    const float threshold = 1.0e-14f;
    const float invDt = 1.0f / dt;

    const float separation = cp.initPen + dot(rigid.linDelta, cp.normal) + dot(rigid.angDelta, rigid.raXn)
                             - dot(db.posDelta, cp.normal);

    const V3 rigidLinDelta = isVelocityIteration ? v3(0.0f, 0.0f, 0.0f) : rigid.linDelta;
    const V3 rigidAngDelta = isVelocityIteration ? v3(0.0f, 0.0f, 0.0f) : rigid.angDelta;
    const V3 relLinDelta = sub(rigidLinDelta, db.linDelta);
    const float normalVel = dot(rigid.linVel, cp.normal) + dot(rigid.angVel, rigid.raXn);

    float CN;
    if (isVelocityIteration)
        CN = normalVel + (selectMax(0.0f, separation) + dot(relLinDelta, cp.normal)) * invDt;
    else
        CN = normalVel + selectMax(cp.maxPenBiasClamp, separation * invDt);

    const bool isActive = wasActive || CN < 0.0f;
    if (checkOnlyActivity || !isActive)
        return isActive;

    // bc.multiply(bc).dot(vertexInvMasses)
    const V3 bc2 = v3(db.bc.x * db.bc.x, db.bc.y * db.bc.y, db.bc.z * db.bc.z);
    const float deformableInvMass_massSplitting = dot(bc2, db.vertexInvMasses);

    const float rigidRefCount = float(rigid.referenceCount);
    const float unitResponse = rigidRefCount * rigid.raXnResp + deformableInvMass_massSplitting;
    const float invDenom = (unitResponse > 0.0f) ? (1.0f / unitResponse) : 0.0f;

    if (withAdhesion)
    {
        const float adhesionImpulse =
            particleAdhesionImpulse(separation, db.adhesion, db.adhesionRadius, invDenom, invDt, isVelocityIteration);
        const float repulsionIncrement = selectMax(-CN * invDenom - adhesionImpulse, -so.accumulatedDeltaLambdaN);
        so.deltaLambdaN = repulsionIncrement + adhesionImpulse;
        so.accumulatedDeltaLambdaN += repulsionIncrement;
    }
    else
    {
        so.deltaLambdaN = selectMax(-CN * invDenom, -so.accumulatedDeltaLambdaN);
        so.accumulatedDeltaLambdaN += so.deltaLambdaN;
    }

    const V3 fric0 = cp.tangent0;
    const V3 fric1 = cross(cp.normal, fric0);

    const float tanVel0 = dot(rigid.linVel, fric0) + dot(rigid.angVel, rigid.raXnF0);
    const float tanVel1 = dot(rigid.linVel, fric1) + dot(rigid.angVel, rigid.raXnF1);

    const float CT0 = (dot(fric0, relLinDelta) + dot(rigidAngDelta, rigid.raXnF0)) * invDt + tanVel0;
    const float CT1 = (dot(fric1, relLinDelta) + dot(rigidAngDelta, rigid.raXnF1)) * invDt + tanVel1;
    const V3 relTanDelta = add(mul(fric0, CT0), mul(fric1, CT1));
    const float tanMagSq = dot(relTanDelta, relTanDelta);

    if (tanMagSq > threshold)
    {
        const float CT = std::sqrt(tanMagSq);
        const float invTanMag = 1.0f / CT;
        so.tangent = mul(relTanDelta, invTanMag);

        const float frac0 = dot(so.tangent, fric0);
        const float frac1 = dot(so.tangent, fric1);
        so.raXt = add(mul(rigid.raXnF0, frac0), mul(rigid.raXnF1, frac1));

        const float unitResponseT0 = rigidRefCount * rigid.raXnF0Resp + deformableInvMass_massSplitting;
        const float invTanDenom0 = (unitResponseT0 > 0.0f) ? (1.0f / unitResponseT0) : 0.0f;
        const float unitResponseT1 = rigidRefCount * rigid.raXnF1Resp + deformableInvMass_massSplitting;
        const float invTanDenom1 = (unitResponseT1 > 0.0f) ? (1.0f / unitResponseT1) : 0.0f;

        const float deltaLambdaT0 = CT0 * invTanDenom0;
        const float deltaLambdaT1 = CT1 * invTanDenom1;

        const float combinedFriction = combineScalars(rigid.friction, db.friction, rigid.fricCombineMode);

        const float requiredLambdaTNeg = -std::sqrt(deltaLambdaT0 * deltaLambdaT0 + deltaLambdaT1 * deltaLambdaT1);
        const float frictionLimit = combinedFriction * so.accumulatedDeltaLambdaN;
        const float targetAccumLambdaT = selectMax(appliedTanRef + requiredLambdaTNeg, -frictionLimit);
        so.deltaLambdaT = targetAccumLambdaT - appliedTanRef;
    }

    return so.deltaLambdaN != 0.0f || so.deltaLambdaT != 0.0f;
}

inline bool bitsEqual(float a, float b) { uint32_t x, y; std::memcpy(&x, &a, 4); std::memcpy(&y, &b, 4); return x == y; }
inline bool bitsEqual(const F4& a, const F4& b) { return std::memcmp(&a, &b, sizeof(F4)) == 0; }
inline bool bitsEqual(const F2& a, const F2& b) { return std::memcmp(&a, &b, sizeof(F2)) == 0; }

} // namespace pxgpsref

#endif // PXG_PARTICLE_SOLVE_REF_H
