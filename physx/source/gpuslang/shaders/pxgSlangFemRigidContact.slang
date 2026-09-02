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

// pxgSlangFemRigidContact — the deformable-vs-rigid CONSTRAINT PREPARE, shared by the
// cloth and softbody contact paths.
//
// WHY THIS MODULE EXISTS.  `prepareFEMContacts` (deformableUtils.cuh:132) is one CUDA
// function that both cloth and softbody contact prepare call.  It was first ported inside
// femClothRigidContactBda.slang as a file-local function; sb_rigidContactPrepareLaunch
// needs it BYTE-FOR-BYTE, so it was extracted here rather than copied.  Two 78-line
// constraint-block writers that must stay in sync is exactly the kind of duplication that
// rots -- and the block layout below is the ABI the solve kernels read back, so a drift
// between two copies would be silent until a scene misbehaved.
//
// Extracted with it: only what the function needs -- the PxgFemRigidConstraintBlock field
// offsets, the small vector helpers and the PxNodeIndex accessors.  Deliberately NOT a
// tree-wide de-duplication pass: crossVec / dotVec / mat33MulVec / computeBasisVectors are
// also defined locally in femClothRigidAttachBda, psContactSolveBda and
// particleSelfCollisionBda, and the PxgFemOtherContactInfo (CI_*) offsets in
// femClothClothContactBda and softBodyContactRemapBda.  Those shaders do NOT import this
// module and keep their own copies; folding them in would be a much larger change with no
// payoff for the kernel that motivated this one.  Importers must therefore not also define
// these names locally.
//
// ARTICULATION BRANCH DEFERRED, as in the original port.  CUDA's prepareFEMContacts has an
// `if (rigidId.isArticulation())` anchor branch that reads
// sharedDesc->mBodySimBufferDeviceData (deformableUtils.cuh:162) and the articulation link
// transforms.  Neither is reachable for a rigid or static collider, which is every case the
// port exercises, so the else path is the only one transcribed and `sharedDesc` is not a
// parameter here.  A scene with an articulation collider needs that branch first.
//
// `invDt` and `isTGS` are NOT parameters, and that is not an omission: CUDA takes both but
// never reads them.  Brace-matched over the whole 122-line body (deformableUtils.cuh:132-253)
// they appear on the signature line only.  They stay in the callers' push blobs because the
// ABI row mirrors the CUDA arg pack, and are simply unread there too.
module pxgSlangFemRigidContact;

import pxgSlangPrepDesc;   // PxgPrePrepDescGpu, PxgConstraintPrepareDescGpu
import pxgSlangCommon;     // PxgSolverBodyData, PxgSolverTxIData
import pxgSlangSolverDesc; // PxgSolverCoreDescGpu, PxgSolverSharedDesc{PGS,TGS}Gpu

// FEMCollision::initialize friction threshold (deformableUtils.cuh:292).
public static const float FEMC_THRESHOLD = 1.0e-14f;

// ---------------------------------------------------------------------------
// Byte layout of the block-of-32 PxgFemRigidConstraintBlock (PxgFEMCore.h:167).
// 32 constraints per block (SoA warp-packed), indexed [offset]; stride 3200.
// ---------------------------------------------------------------------------
public static const uint FRC_raXn_resp        = 0u;      // float4[32]
public static const uint FRC_raXnF0_resp      = 512u;    // float4[32]
public static const uint FRC_raXnF1_resp      = 1024u;   // float4[32]
public static const uint FRC_normal_errorW    = 1536u;   // float4[32]
public static const uint FRC_fricTan0_invMass = 2048u;   // float4[32]
public static const uint FRC_barycentric      = 2560u;   // float4[32]
public static const uint FRC_maxPenBias       = 3072u;   // float[32]
public static const uint FRC_STRIDE           = 3200u;

// ---------------------------------------------------------------------------
// Small vector helpers (byte-for-byte the CUDA foundation math).
// ---------------------------------------------------------------------------
public float3 crossVec(float3 a, float3 b)
{
    return float3(a.y * b.z - a.z * b.y, a.z * b.x - a.x * b.z, a.x * b.y - a.y * b.x);
}
public float dotVec(float3 a, float3 b) { return a.x * b.x + a.y * b.y + a.z * b.z; }

// PxMat33 (column-major sqrtInvInertia[9]) * PxVec3.
public float3 mat33MulVec(float m[9], float3 v)
{
    return float3(m[0] * v.x + m[3] * v.y + m[6] * v.z,
                  m[1] * v.x + m[4] * v.y + m[7] * v.z,
                  m[2] * v.x + m[5] * v.y + m[8] * v.z);
}

// PxComputeBasisVectors (PxMathUtils.h:297): two orthonormal tangents to `dir`.
// right.normalize() is a true divide by the magnitude (PxVec3::normalize(), PxVec3.h:295).
public void computeBasisVectors(float3 dir, out float3 right, out float3 up)
{
    // P9.45: these two normalisations used to be THREE DIVIDES by the same scalar.  The
    // engine writes `right.normalize()`, and PxVec3::normalize (PxVec3.h:293) scales through
    // operator/= (PxVec3.h:260), which is ONE reciprocal and three multiplies -- the P9.36
    // defect class, in the componentwise spelling that the `gpuslang.vectorDivide` checker
    // cannot see (it matches an OpFDiv whose RESULT TYPE is a vector; three scalar OpFDivs
    // look like ordinary arithmetic).  Measured: the two forms differ in at least one
    // component for most normals -- the single tilted normal in runRigidContact happened to
    // be one where they agree, which is why the fixture was green for eight tranches.
    // femClothSimRigidContactDynamic* now spreads six normals across the same code and
    // caught it.
    if (abs(dir.y) <= 0.9999f)
    {
        right = float3(dir.z, 0.0f, -dir.x);
        const float m = sqrt(right.x * right.x + right.y * right.y + right.z * right.z);
        if (m > 0.0f)
        {
            const float r = 1.0f / m;
            right = float3(right.x * r, right.y * r, right.z * r);
        }
        up = float3(dir.y * right.z, dir.z * right.x - dir.x * right.z, -dir.y * right.x);
    }
    else
    {
        right = float3(1.0f, 0.0f, 0.0f);
        up = float3(0.0f, dir.z, -dir.y);
        const float m = sqrt(up.x * up.x + up.y * up.y + up.z * up.z);
        if (m > 0.0f)
        {
            const float r = 1.0f / m;
            up = float3(up.x * r, up.y * r, up.z * r);
        }
    }
}

// PxNodeIndex (PxNodeIndex.h:49) packed in the low/high 32 bits of a PxU64.
public bool nodeIsStaticBody(uint64_t ind) { return uint(ind & 0xffffffffUL) == 0xffffffffu; }
public uint nodeIndex(uint64_t ind) { return uint(ind & 0xffffffffUL); }
public bool nodeIsArticulation(uint64_t ind) { return (uint(ind >> 32) & 1u) != 0u; }

// ===========================================================================
// prepareFEMContacts (deformableUtils.cuh:132) — the rigid / static (non-articulation)
// path.  Writes the block-of-32 PxgFemRigidConstraintBlock slot [offset] at cBase.
// ===========================================================================
public void prepareFEMContacts(uint64_t cBase, uint offset, float3 normal,
                               float3 p, float pen, float3 delta, uint64_t rigidId,
                               float4 barycentric, uint64_t prepDescPtr,
                               uint64_t prePrepDescPtr, float penBiasClampFEM)
{
    Ptr<PxgConstraintPrepareDescGpu> prepDesc = Ptr<PxgConstraintPrepareDescGpu>(prepDescPtr);
    const uint64_t bodyFramesBase = prepDesc[0].body2WorldPool;
    const uint64_t solverBodyDataBase = prepDesc[0].solverBodyDataPool;
    const uint64_t solverTxIDataBase = prepDesc[0].solverBodyTxIDataPool;

    // Select two tangent vectors to the normal (PxComputeBasisVectors).
    float3 t0, t1;
    computeBasisVectors(normal, t0, t1);

    // ARTICULATION anchor branch DEFERRED (see header): a static floor / sphere is never
    // an articulation, so the else (rigid / static) path is taken.  solverBodyIdx == 0 for
    // a static body (solverBodyIndices untouched).
    const bool isStatic = nodeIsStaticBody(rigidId);
    uint idx = 0u;
    if (!isStatic)
        idx = Ptr<uint>(Ptr<PxgPrePrepDescGpu>(prePrepDescPtr)[0].solverBodyIndices)[nodeIndex(rigidId)];

    PxgSolverTxIData txI = Ptr<PxgSolverTxIData>(solverTxIDataBase)[idx];
    float invSqrtInertia[9];
    for (uint k = 0; k < 9u; ++k)
        invSqrtInertia[k] = txI.sqrtInvInertia[k];

    const PxgSolverBodyData sbd = Ptr<PxgSolverBodyData>(solverBodyDataBase)[idx];
    const float4 linVel_invMass0 = sbd.initialLinVelXYZ_invMassW;
    const float penBiasClampRigid = sbd.initialAngVelXYZ_penBiasClamp.w;
    const float invMass0 = linVel_invMass0.w;

    // both static and kinematic objects have invMass == 0; a static body is never kinematic.
    const bool isKinematic = (invMass0 == 0.0f) && (!isStatic);

    const float4 bfP = Ptr<float4>(bodyFramesBase + uint64_t(idx) * 32u + 16u)[0];
    const float3 bodyFrame0p = float3(bfP.x, bfP.y, bfP.z);

    const float3 ra = float3(p.x - bodyFrame0p.x, p.y - bodyFrame0p.y, p.z - bodyFrame0p.z);
    const float3 raXn = crossVec(ra, normal);
    const float3 raXF0 = crossVec(ra, t0);
    const float3 raXF1 = crossVec(ra, t1);

    const float3 raXnSqrtInertia = mat33MulVec(invSqrtInertia, raXn);
    const float resp0 = dotVec(raXnSqrtInertia, raXnSqrtInertia) + invMass0;

    const float3 raXF0SqrtInertia = mat33MulVec(invSqrtInertia, raXF0);
    const float3 raXF1SqrtInertia = mat33MulVec(invSqrtInertia, raXF1);

    const float respF0 = dotVec(raXF0SqrtInertia, raXF0SqrtInertia) + invMass0;
    const float respF1 = dotVec(raXF1SqrtInertia, raXF1SqrtInertia) + invMass0;

    float4 raXn_resp;
    float4 raXnF0_resp;
    float4 raXnF1_resp;
    if (isKinematic)
    {
        raXn_resp = float4(raXn.x, raXn.y, raXn.z, resp0);
        raXnF0_resp = float4(raXF0.x, raXF0.y, raXF0.z, respF0);
        raXnF1_resp = float4(raXF1.x, raXF1.y, raXF1.z, respF1);
    }
    else
    {
        raXn_resp = float4(raXnSqrtInertia.x, raXnSqrtInertia.y, raXnSqrtInertia.z, resp0);
        raXnF0_resp = float4(raXF0SqrtInertia.x, raXF0SqrtInertia.y, raXF0SqrtInertia.z, respF0);
        raXnF1_resp = float4(raXF1SqrtInertia.x, raXF1SqrtInertia.y, raXF1SqrtInertia.z, respF1);
    }

    const float maxPenBias = max(penBiasClampRigid, penBiasClampFEM);
    const float error = pen + dotVec(delta, normal);

    Ptr<float4>(cBase + FRC_normal_errorW)[offset] = float4(normal.x, normal.y, normal.z, error);
    Ptr<float4>(cBase + FRC_raXn_resp)[offset] = raXn_resp;
    Ptr<float4>(cBase + FRC_raXnF0_resp)[offset] = raXnF0_resp;
    Ptr<float4>(cBase + FRC_raXnF1_resp)[offset] = raXnF1_resp;
    Ptr<float4>(cBase + FRC_fricTan0_invMass)[offset] = float4(t0.x, t0.y, t0.z, invMass0);
    Ptr<float>(cBase + FRC_maxPenBias)[offset] = maxPenBias;
    Ptr<float4>(cBase + FRC_barycentric)[offset] = barycentric;
}

// ---------------------------------------------------------------------------
// PxgVelocityReader (dataReadWriteHelper.cuh:70) — non-articulation rigid path.
// ARTICULATION branch DEFERRED (see header).  Returns lin/ang velocity (+ lin/ang delta
// for TGS) of the rigid body; a static body reads solverBodyIdx 0 (zero for a floor).
// ---------------------------------------------------------------------------
public struct RigidState
{
    float3 linVel;
    float3 angVel;
    float3 linDelta;
    float3 angDelta;
}

public RigidState readRigidVelocity(uint64_t rigidId, bool isTGS, uint64_t solverCoreDescPtr,
                             uint64_t sharedDescPtr, uint64_t prePrepDescPtr)
{
    Ptr<PxgSolverCoreDescGpu> solverCoreDesc = Ptr<PxgSolverCoreDescGpu>(solverCoreDescPtr);
    const uint numSolverBodies = solverCoreDesc[0].numSolverBodies;
    const uint accOffset = solverCoreDesc[0].accumulatedBodyDeltaVOffset;

    const bool isStatic = nodeIsStaticBody(rigidId);
    uint solverBodyIdx = 0u;
    if (!isStatic)
        solverBodyIdx = Ptr<uint>(Ptr<PxgPrePrepDescGpu>(prePrepDescPtr)[0].solverBodyIndices)[nodeIndex(rigidId)];

    RigidState s;
    s.linVel = float3(0.0f, 0.0f, 0.0f);
    s.angVel = float3(0.0f, 0.0f, 0.0f);
    s.linDelta = float3(0.0f, 0.0f, 0.0f);
    s.angDelta = float3(0.0f, 0.0f, 0.0f);

    if (isTGS)
    {
        Ptr<float4> pool = Ptr<float4>(Ptr<PxgSolverSharedDescTGSGpu>(sharedDescPtr)[0].iter_solverBodyVelPool);
        const float4 v0 = pool[accOffset + solverBodyIdx];
        const float4 v1 = pool[accOffset + solverBodyIdx + numSolverBodies];
        const float4 v2 = pool[accOffset + solverBodyIdx + numSolverBodies + numSolverBodies];
        s.linVel = float3(v0.x, v0.y, v0.z);
        s.angVel = float3(v0.w, v1.x, v1.y);
        s.linDelta = float3(v1.z, v1.w, v2.x);
        s.angDelta = float3(v2.y, v2.z, v2.w);
    }
    else
    {
        Ptr<float4> initialVel = Ptr<float4>(solverCoreDesc[0].outSolverVelocity);
        Ptr<float4> pool = Ptr<float4>(Ptr<PxgSolverSharedDescPGSGpu>(sharedDescPtr)[0].iter_solverBodyVelPool);
        const float4 iv0 = initialVel[solverBodyIdx];
        const float4 iv1 = initialVel[solverBodyIdx + numSolverBodies];
        const float4 dv0 = pool[accOffset + solverBodyIdx];
        const float4 dv1 = pool[accOffset + solverBodyIdx + numSolverBodies];
        s.linVel = float3(iv0.x + dv0.x, iv0.y + dv0.y, iv0.z + dv0.z);
        s.angVel = float3(iv1.x + dv1.x, iv1.y + dv1.y, iv1.z + dv1.z);
    }
    return s;
}

// ---------------------------------------------------------------------------
// FEMCollision::initialize (deformableUtils.cuh:280) — deformable vs rigid, static /
// null-material path.  Returns isActive; on the full path also fills deltaLambdaN,
// deltaLambdaT, tangent (deltaLambdaT == 0 with the deferred null-material friction).
// ---------------------------------------------------------------------------
public struct ContactResult
{
    public bool isActive;
    public float deltaLambdaN;
    public float deltaLambdaT;
    public float3 normal;
    public float3 tangent;
    // P9.45: the two angular rows computeRigidChange needs.  raXn mirrors the CUDA member
    // assigned at deformableUtils.cuh:306 -- BEFORE both early returns, so it is live even
    // on the checkOnlyActivity path; raXt mirrors the member assigned only inside the
    // tangent branch (:394), and stays at its zero default otherwise.  For a DYNAMIC
    // non-kinematic body raXn_resp.xyz holds raXn * sqrtInvInertia, not raXn, exactly as
    // prepareFEMContacts wrote it -- the downstream accumulate folds the second factor.
    public float3 raXn;
    public float3 raXt;
    public float accumulatedDeltaLambdaN;
}

// THE MASS-SPLITTING TERM IS A PARAMETER, not computed here, and that is what makes this
// function deformable-agnostic.  CUDA templates FEMCollision on the vector type -- PxVec3
// for a cloth triangle, PxVec4 for a softbody tetrahedron -- and the ONLY place the width
// shows through is `bc.multiply(bc).dot(invMasses)`.  Passing that scalar in lets one
// implementation serve both, instead of a float4 signature with a dummy w that every cloth
// caller would have to remember to zero (and where a garbage w of inf would poison the
// product).  Callers compute their own 3- or 4-wide dot.
public ContactResult contactInitialize(uint64_t cBase, uint offset, uint64_t rigidId, bool isTGS,
                                float appliedForceRef, float dt, float3 bcVec,
                                float3 deformableLinDelta, float deformableInvMass_massSplitting,
                                uint rigidBodyReferenceCount, bool wasActive, bool checkOnlyActivity,
                                uint64_t solverCoreDescPtr, uint64_t sharedDescPtr, uint64_t prePrepDescPtr)
{
    ContactResult r;
    r.isActive = false;
    r.deltaLambdaN = 0.0f;
    r.deltaLambdaT = 0.0f;
    r.tangent = float3(0.0f, 0.0f, 0.0f);
    r.raXn = float3(0.0f, 0.0f, 0.0f);
    r.raXt = float3(0.0f, 0.0f, 0.0f);
    r.accumulatedDeltaLambdaN = appliedForceRef;

    const float invDt = 1.0f / dt;

    const float4 raXn_resp = Ptr<float4>(cBase + FRC_raXn_resp)[offset];
    const float4 normal_biasW = Ptr<float4>(cBase + FRC_normal_errorW)[offset];
    const float4 raXnF0_resp = Ptr<float4>(cBase + FRC_raXnF0_resp)[offset];
    const float4 raXnF1_resp = Ptr<float4>(cBase + FRC_raXnF1_resp)[offset];
    const float maxPenBias = Ptr<float>(cBase + FRC_maxPenBias)[offset];
    const float4 fricTan0_invMass0 = Ptr<float4>(cBase + FRC_fricTan0_invMass)[offset];

    const float3 normal = float3(normal_biasW.x, normal_biasW.y, normal_biasW.z);
    r.normal = normal;
    const float3 fric0 = float3(fricTan0_invMass0.x, fricTan0_invMass0.y, fricTan0_invMass0.z);
    const float3 fric1 = crossVec(normal, fric0);
    const float initPen = normal_biasW.w;
    const float3 raXn = float3(raXn_resp.x, raXn_resp.y, raXn_resp.z);
    r.raXn = raXn;   // :306, before both early returns

    const RigidState st = readRigidVelocity(rigidId, isTGS, solverCoreDescPtr, sharedDescPtr, prePrepDescPtr);
    const float3 linVel = st.linVel;
    const float3 angVel = st.angVel;

    float CN;
    float3 relLinDelta;
    float3 angDelta;
    if (isTGS)
    {
        const float normalVel = dotVec(linVel, normal) + dotVec(angVel, raXn);
        relLinDelta = float3(st.linDelta.x - deformableLinDelta.x, st.linDelta.y - deformableLinDelta.y,
                             st.linDelta.z - deformableLinDelta.z);
        angDelta = st.angDelta;
        const float error = (initPen + dotVec(relLinDelta, normal) + dotVec(angDelta, raXn)) * invDt;
        const float errorBiased = max(maxPenBias, error);
        CN = errorBiased + normalVel;
    }
    else
    {
        const float normalVel = dotVec(linVel, normal) + dotVec(angVel, raXn);
        relLinDelta = float3(-deformableLinDelta.x, -deformableLinDelta.y, -deformableLinDelta.z);
        angDelta = float3(0.0f, 0.0f, 0.0f);
        const float error = (initPen + dotVec(relLinDelta, normal)) * invDt;
        const float errorBiased = max(maxPenBias, error);
        CN = errorBiased + normalVel;
    }

    r.isActive = wasActive || (CN < 0.0f);

    if (checkOnlyActivity)
        return r;

    if (!r.isActive)
        return r;

    const float rigidRefCount = float(rigidBodyReferenceCount);
    const float unitResponse = rigidRefCount * raXn_resp.w + deformableInvMass_massSplitting;
    const float invDenom = (unitResponse > 0.0f) ? (1.0f / unitResponse) : 0.0f;

    r.deltaLambdaN = max(-CN * invDenom, -r.accumulatedDeltaLambdaN);
    r.accumulatedDeltaLambdaN += r.deltaLambdaN;

    // Friction constraint in the tangent direction (null-material path -> combinedFriction 0).
    const float3 raXnF0 = float3(raXnF0_resp.x, raXnF0_resp.y, raXnF0_resp.z);
    const float3 raXnF1 = float3(raXnF1_resp.x, raXnF1_resp.y, raXnF1_resp.z);

    const float tanVel0 = dotVec(linVel, fric0) + dotVec(angVel, raXnF0);
    const float tanVel1 = dotVec(linVel, fric1) + dotVec(angVel, raXnF1);

    const float CT0 = (dotVec(fric0, relLinDelta) + dotVec(angDelta, raXnF0)) * invDt + tanVel0;
    const float CT1 = (dotVec(fric1, relLinDelta) + dotVec(angDelta, raXnF1)) * invDt + tanVel1;
    const float3 relTanDelta = float3(CT0 * fric0.x + CT1 * fric1.x, CT0 * fric0.y + CT1 * fric1.y,
                                      CT0 * fric0.z + CT1 * fric1.z);
    const float tanMagSq = dotVec(relTanDelta, relTanDelta);

    if (tanMagSq > FEMC_THRESHOLD)
    {
        const float CT = sqrt(tanMagSq);
        const float invTanMag = 1.0f / CT;
        r.tangent = float3(relTanDelta.x * invTanMag, relTanDelta.y * invTanMag, relTanDelta.z * invTanMag);

        // P9.45: raXt = frac0 * raXnF0 + frac1 * raXnF1 (:391-394).  Previously omitted
        // because only computeRigidChange reads it and that was deferred.
        const float frac0 = dotVec(r.tangent, fric0);
        const float frac1 = dotVec(r.tangent, fric1);
        r.raXt = float3(frac0 * raXnF0.x + frac1 * raXnF1.x,
                        frac0 * raXnF0.y + frac1 * raXnF1.y,
                        frac0 * raXnF0.z + frac1 * raXnF1.z);

        const float unitResponseT0 = rigidRefCount * raXnF0_resp.w + deformableInvMass_massSplitting;
        const float invTanDenom0 = (unitResponseT0 > 0.0f) ? (1.0f / unitResponseT0) : 0.0f;
        const float unitResponseT1 = rigidRefCount * raXnF1_resp.w + deformableInvMass_massSplitting;
        const float invTanDenom1 = (unitResponseT1 > 0.0f) ? (1.0f / unitResponseT1) : 0.0f;

        const float deltaLambdaT0 = CT0 * invTanDenom0;
        const float deltaLambdaT1 = CT1 * invTanDenom1;

        const float dLT = sqrt(deltaLambdaT0 * deltaLambdaT0 + deltaLambdaT1 * deltaLambdaT1);
        // getCombinedFriction() == 0 (null-material path, DEFERRED): deltaLambdaT == -0.
        const float combinedFriction = 0.0f;
        r.deltaLambdaT = -min(dLT, combinedFriction * abs(r.deltaLambdaN));
    }

    return r;
}

// ---------------------------------------------------------------------------
// FEMCollision::computeRigidChange (deformableUtils.cuh:418-429) — P9.45.
//
//   deltaAngVel0 = isArticulation ? (raXn*dLN + raXt*dLT)
//                                 : (raXn*dLN + raXt*dLT) * refCount;
//   deltaLinVel0 = (normal*dLN + tangent*dLT) * rigidInvMass * refCount;
//
// The reference count multiplies the SUM, not each term, and the linear row multiplies
// by rigidInvMass FIRST and the count second -- both are PxVec3 operator* chains
// evaluated left to right, and with NoContraction on they are three separate rounds per
// component.  Width-independent: it touches no barycentric, so one implementation serves
// the cloth triangle (PxVec3) and the softbody tetrahedron (PxVec4) alike.
// ---------------------------------------------------------------------------
public struct RigidChange
{
    public float3 deltaLinVel0;
    public float3 deltaAngVel0;
}

public RigidChange computeRigidChange(ContactResult r, bool isArticulation, float rigidInvMass,
                                      uint rigidBodyReferenceCount)
{
    const float rigidRefCount = float(rigidBodyReferenceCount);

    const float3 angBase = float3(r.raXn.x * r.deltaLambdaN + r.raXt.x * r.deltaLambdaT,
                                  r.raXn.y * r.deltaLambdaN + r.raXt.y * r.deltaLambdaT,
                                  r.raXn.z * r.deltaLambdaN + r.raXt.z * r.deltaLambdaT);

    RigidChange o;
    o.deltaAngVel0 = isArticulation
                   ? angBase
                   : float3(angBase.x * rigidRefCount, angBase.y * rigidRefCount, angBase.z * rigidRefCount);

    const float3 linBase = float3(r.normal.x * r.deltaLambdaN + r.tangent.x * r.deltaLambdaT,
                                  r.normal.y * r.deltaLambdaN + r.tangent.y * r.deltaLambdaT,
                                  r.normal.z * r.deltaLambdaN + r.tangent.z * r.deltaLambdaT);
    o.deltaLinVel0 = float3((linBase.x * rigidInvMass) * rigidRefCount,
                            (linBase.y * rigidInvMass) * rigidRefCount,
                            (linBase.z * rigidInvMass) * rigidRefCount);
    return o;
}

public int getGlobalRigidBodyId(uint64_t rigidId, uint64_t prePrepDescPtr, uint numSolverBodies)
{
    if (nodeIsStaticBody(rigidId))
        return -1;
    const uint solverBodyIdx = Ptr<uint>(Ptr<PxgPrePrepDescGpu>(prePrepDescPtr)[0].solverBodyIndices)[nodeIndex(rigidId)];
    return nodeIsArticulation(rigidId) ? int(numSolverBodies + solverBodyIdx) : int(solverBodyIdx);
}
