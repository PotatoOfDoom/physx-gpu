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

// pxgSlangArticulationDynamic.slang
//
// P8.90: the Featherstone/Mirtich impulse-response pair — a spatial impulse
// applied to a child link propagated UP to its parent, and an acceleration (or
// velocity) propagated DOWN from a parent to its child.  Both are what
// `computeMassMatrix1T` calls six times per link to build the 6x6 response
// matrix, and both reappear throughout the solver.
//
// Originals: `propagateImpulseW_everythingPreLoaded` and
// `propagateAccelerationW_everythingPreLoaded`, articulationDynamic.cuh:167 /
// :206.  Mirtich p141 (impulses/velocities) and p121 (forces/accelerations).
//
// ===========================================================================
// WHY THIS IS A LIFT AND NOT A NEW PORT
// ===========================================================================
//
// Phase 1 already ported both bodies, as `shaders/npArtiPropagate.slang`
// (`apImpulse`) and `shaders/npArtiPropagateAccel.slang` (`apAccel`), each with
// its own fixture and its traps already written down.  The bodies below are
// LIFTED from those — same expressions in the same order — rather than derived a
// second time from the CUDA, which is the duplication this port exists to avoid.
//
// The Phase-1 pair stays where it is rather than being rewired to call this
// module, and that is deliberate: both are in `PXG_HLSL_ENTRIES`, so they must
// keep lowering to cs_6_0, and this module imports the BDA block layer, which
// does not lower.  What keeps the two copies honest is that **this module's
// fixture compares against the ENGINE** — `translateSpatialVector` is
// `PX_CUDA_CALLABLE` (DyFeatherstoneArticulation.h:828), as are `innerProduct`,
// `cross` and `dot` — where the Phase-1 fixtures compare against a host
// transcription.  So the lift is checked against a stronger reference than the
// thing it was lifted from.
//
// ===========================================================================
// THE TRAPS, INHERITED FROM THE PHASE-1 HEADERS
// ===========================================================================
//
//  1. **`innerProduct` is the DUAL product**, `bottom.dot(top) + top.dot(bottom)`
//     — not `dot`.  Both directions use it, and both would still produce
//     plausible numbers with the Euclidean one.
//  2. **`stZ` is taken against `Z`, not against the running `temp`.**  The
//     accumulator is updated in the same loop, so using it would be the natural
//     mistake and silently changes the result from the second dof onward.
//  3. **The acceleration direction NEGATES the offset**:
//     `translateSpatialVector(-parentToChild, ...)`.  The impulse direction does
//     not negate `childToParent`.  Two functions, two directions, one sign.
//  4. **The acceleration direction is TWO PASSES.**  Every `tJointDelta` is
//     formed from the *pre-accumulation* `pDeltaV`; only the second pass mutates
//     it.  Fusing them feeds each dof a `pDeltaV` the earlier dofs have already
//     moved.
//  5. `jDelta` is a FULL three-lane dot over `tJointDelta`, including the lanes
//     at or beyond `dofCount` — which are zero, so the mask is implicit rather
//     than written.  A port that bounds that dot by `dofCount` is right today and
//     wrong the moment `tJointDelta` is seeded with anything but zero.
//
// `translateSpatialVector` itself is one line and leaves the TOP half alone:
// `(top, bottom + offset x top)`.

import pxgSlangCommon;
import pxgSlangArticulationBlocks;
import pxgSlangSpatialMatrix;

// Dy::FeatherstoneArticulation::translateSpatialVector
// (DyFeatherstoneArticulation.h:828).  The top half is untouched.
public AbSpatialVector adTranslateSpatialVector(float3 offset, AbSpatialVector v)
{
    return asmSVec(v.top, v.bottom + asmCross(offset, v.top));
}

// propagateImpulseW_everythingPreLoaded (articulationDynamic.cuh:167).
//
// `qstZ` ACCUMULATES — the caller's value is added to, not replaced.  In
// `computeMassMatrix1T` the caller seeds it with a literal `{0, 0, 0}` and the
// engine's C++ performs the add, which is the P8.84 signed-zero shape; that is
// the caller's fixture problem, not this function's, and the caller's note says
// so.
public AbSpatialVector adPropagateImpulseW(AbSpatialVector isInvD[3], float3 childToParent,
                                           AbSpatialVector motionMatrix[3], AbSpatialVector Z,
                                           uint dofCount, inout float qstZ[3])
{
    AbSpatialVector temp = Z;

    for (uint ind = 0u; ind < 3u; ++ind)
    {
        if (ind < dofCount)
        {
            // against Z, NOT against temp -- see trap 2
            const float stZ = -asmInnerProduct(motionMatrix[ind], Z);
            qstZ[ind] = qstZ[ind] + stZ;
            temp = asmSVecAdd(temp, asmSVecScale(isInvD[ind], stZ));
        }
    }

    // into the parent's frame
    return adTranslateSpatialVector(childToParent, temp);
}

// propagateImpulseW<WriteQstZ> (articulationDynamic.cuh:129) — the TEMPLATE
// variant, which differs from propagateImpulseW_everythingPreLoaded above in
// two ways that both matter to the bit pattern:
//
//  1. it applies an optional JOINT impulse/force to each dof of the inbound
//     joint: `jf = jointForce ? jointForce[ind] * jointForceMultiplier : 0.0f`
//     and `stZ = jf - sas.innerProduct(Z)`.  The NULL path computes
//     `0.0f - dot`, which differs from the preloaded variant's `-dot` at
//     exactly dot == +0 (`0 - (+0) = +0` where `-(+0) = -0`) — so this is NOT
//     a forwarder around adPropagateImpulseW; the two spellings coexist in the
//     engine and they coexist here.  `hasJf` reproduces the NULL-pointer test.
//  2. the qstZ accumulation target is the caller's choice (the template
//     parameter): callers seed `qstZ` from mDeferredQstZ / mTmpQstZ / a local
//     and write it back, exactly as P8.94b/c already do for the accel side.
public AbSpatialVector adPropagateImpulseWJf(AbSpatialVector isInvD[3], float3 childToParent,
                                             AbSpatialVector motionMatrix[3], AbSpatialVector Z,
                                             uint dofCount, inout float qstZ[3],
                                             float jf[3], float jfMult, bool hasJf)
{
    AbSpatialVector temp = Z;

    for (uint ind = 0u; ind < 3u; ++ind)
    {
        if (ind < dofCount)
        {
            const float jfv = hasJf ? jf[ind] * jfMult : 0.0f;
            // against Z, NOT against temp
            const float stZ = jfv - asmInnerProduct(motionMatrix[ind], Z);
            qstZ[ind] = qstZ[ind] + stZ;
            temp = asmSVecAdd(temp, asmSVecScale(isInvD[ind], stZ));
        }
    }

    return adTranslateSpatialVector(childToParent, temp);
}

// propagateAccelerationW_everythingPreLoaded (articulationDynamic.cuh:206).
//
// `invStISW[ind]` is a ROW of the inverse joint-space inertia -- the same three
// floats `computeIs`' spherical path wrote to `mInvStIsT_{x,y,z}` for dof `ind`.
// P8.94b needs the per-dof `jDelta` this function computes and discards -- the
// `_updateJointDofVels` variant (articulationDynamic.cuh:329) writes
// `mJointVelocities[ind] += jDelta` with it.  Rather than transcribe the body a
// second time, the body lives here with `jDelta` as an out parameter and the
// original signature is a two-line forwarder, so P8.90's fixtures keep driving
// exactly the code P8.94b drives.
public AbSpatialVector adPropagateAccelerationW(float3 parentToChild, float3 invStISW[3],
                                                AbSpatialVector motionMatrix[3],
                                                AbSpatialVector parentLinkAccelerationW,
                                                uint dofCount, AbSpatialVector IsW[3],
                                                float QMinusSTZ[3])
{
    float jDeltaOut[3];
    return adPropagateAccelerationWEx(parentToChild, invStISW, motionMatrix,
                                      parentLinkAccelerationW, dofCount, IsW, QMinusSTZ,
                                      jDeltaOut);
}

public AbSpatialVector adPropagateAccelerationWEx(float3 parentToChild, float3 invStISW[3],
                                                  AbSpatialVector motionMatrix[3],
                                                  AbSpatialVector parentLinkAccelerationW,
                                                  uint dofCount, AbSpatialVector IsW[3],
                                                  float QMinusSTZ[3], out float jDeltaOut[3])
{
    // NEGATED offset -- see trap 3
    AbSpatialVector pDeltaV = adTranslateSpatialVector(-parentToChild, parentLinkAccelerationW);

    // pass one: every entry formed from the PRE-accumulation pDeltaV
    float tJointDelta[3] = { 0.0f, 0.0f, 0.0f };
    for (uint ind = 0u; ind < 3u; ++ind)
    {
        if (ind < dofCount)
        {
            const float temp = asmInnerProduct(IsW[ind], pDeltaV);
            tJointDelta[ind] = QMinusSTZ[ind] - temp;
        }
    }

    // pass two: only this one mutates pDeltaV
    jDeltaOut = { 0.0f, 0.0f, 0.0f };
    for (uint ind = 0u; ind < 3u; ++ind)
    {
        if (ind < dofCount)
        {
            const float3 iStIsTi = invStISW[ind];

            // the full three-lane dot -- lanes at or beyond dofCount are zero
            const float jDelta = iStIsTi.x * tJointDelta[0]
                               + iStIsTi.y * tJointDelta[1]
                               + iStIsTi.z * tJointDelta[2];

            jDeltaOut[ind] = jDelta;
            pDeltaV = asmSVecAdd(pDeltaV, asmSVecScale(motionMatrix[ind], jDelta));
        }
    }

    return pDeltaV;
}

// ===========================================================================
// The implicit-drive scalar family (DyCpuGpuArticulation.h), LIFTED from
// npArtiDriveParams.slang in P8.97a so setupInternalConstraints and the
// Phase-1 fixture drive the same bodies.  computeImplicitDriveParamsForceDrive
// (L74-90), computeImplicitDriveParamsAccelerationDrive (L93-109),
// computeDriveImpulse (L184-195), computeFrictionImpulse (L165-173).  The
// eFORCE/eACCELERATION dispatch of computeImplicitDriveParams (L127) is a
// select at the caller.  ArticulationImplicitDriveDesc = (targetVelPlusInitial
// Bias, biasCoefficient, velMultiplier, impulseMultiplier, targetPosBias).
// ===========================================================================

public struct AdDriveDesc { public float tvBias; public float biasCoef; public float velMul; public float impMul; public float posBias; }

public AdDriveDesc adForceDrive(float stiffness, float damping, float dt, float simDt,
                                float unitResponse, float geomError, float targetVelocity, bool isTGS)
{
    const float a = dt * (dt * stiffness + damping);
    const float b = dt * (damping * targetVelocity);
    const float x = unitResponse > 0.0f ? 1.0f / (1.0f + a * unitResponse) : 0.0f;
    const float initialBias = geomError - (simDt - dt) * targetVelocity;
    const float biasCoefficient = stiffness * x * dt;
    AdDriveDesc d;
    d.tvBias = (x * b) + biasCoefficient * initialBias;
    d.velMul = -x * a;
    d.biasCoef = biasCoefficient;
    d.impMul = isTGS ? 1.0f : (1.0f - x);
    d.posBias = isTGS ? (biasCoefficient * targetVelocity) : 0.0f;
    return d;
}

public AdDriveDesc adAccelDrive(float stiffness, float damping, float dt, float simDt,
                                float recipUnitResponse, float geomError, float targetVelocity, bool isTGS)
{
    const float a = dt * (dt * stiffness + damping);
    const float b = dt * (damping * targetVelocity);
    const float x = 1.0f / (1.0f + a);
    const float initialBias = geomError - (simDt - dt) * targetVelocity;
    const float biasCoefficient = stiffness * x * recipUnitResponse * dt;
    AdDriveDesc d;
    d.tvBias = (x * b * recipUnitResponse) + biasCoefficient * initialBias;
    d.velMul = -x * a * recipUnitResponse;
    d.biasCoef = biasCoefficient;
    d.impMul = isTGS ? 1.0f : (1.0f - x);
    d.posBias = isTGS ? (biasCoefficient * targetVelocity) : 0.0f;
    return d;
}

public float adDriveImpulse(float accum, float jointVel, float jointDeltaPos, float elapsed, AdDriveDesc d)
{
    return accum * d.impMul
         + jointVel * d.velMul
         + d.tvBias
         - jointDeltaPos * d.biasCoef
         + elapsed * d.posBias;
}

public float adFrictionImpulse(float fImp, float staticF, float dynamicF, float viscous, float jointVel)
{
    if (abs(fImp) > staticF)
    {
        const float lo = -dynamicF - viscous * abs(jointVel);
        const float hi = dynamicF + viscous * abs(jointVel);
        // PxClamp(v, lo, hi) = PxMin(hi, PxMax(lo, v)); PxMax(a,b)=(a>b)?a:b, PxMin(a,b)=(a<b)?a:b
        const float mx = (lo > fImp) ? lo : fImp;
        fImp = (hi < mx) ? hi : mx;
    }
    return fImp;
}

// ===========================================================================
// P8.100b lifts.  The drive-impulse performance-envelope clamp
// (DyCpuGpuArticulation.h:197-331) moves here VERBATIM from
// npArtiClampDriveImp.slang (which now imports and forwards — one body, two
// users), and computeLimitImpulse (:352-455) is transcribed fresh with the
// ENGINE's host-callable original as the fixture oracle.
// ===========================================================================

static const float AD_MAX_F32 = asfloat(0x7f7fffffu);
static const float AD_EPS_F32 = asfloat(0x34000000u);

public float adPMin(float a, float b) { return (a < b) ? a : b; }
public float adPMax(float a, float b) { return (a > b) ? a : b; }
public float adPClamp(float v, float lo, float hi) { return adPMin(hi, adPMax(lo, v)); }

// computeBoundsForNegativeImpulse (L197-228); returns (bound1, bound2)
float2 adBoundsNeg(float velDeviation, float maxv, float denom1, float denom2, float impulse)
{
    if (abs(denom1) < AD_EPS_F32)
        return float2(-AD_MAX_F32, AD_MAX_F32);
    if (abs(denom2) < AD_EPS_F32)
    {
        const float bound1 = (velDeviation - maxv) / denom1;
        float bound2;
        if (abs(velDeviation + maxv) < AD_EPS_F32)
            bound2 = AD_MAX_F32;
        else
            bound2 = (velDeviation + maxv) > 0.0f ? AD_MAX_F32 : -AD_MAX_F32;
        return float2(bound1, adPMin(0.0f, bound2));
    }
    const float bound1 = (velDeviation - maxv) / denom1;
    const float bound2 = (velDeviation + maxv) / denom2;
    if (denom2 >= 0.0f)
        return float2(adPMax(bound1, impulse), adPMin(0.0f, bound2));
    return float2(adPMax(adPMax(bound1, bound2), impulse), 0.0f);
}

// computeBoundsForPositiveImpulse (L231-266)
float2 adBoundsPos(float velDeviation, float maxv, float denom1, float denom2, float impulse)
{
    if (abs(denom2) < AD_EPS_F32)
        return float2(-AD_MAX_F32, AD_MAX_F32);
    if (abs(denom1) < AD_EPS_F32)
    {
        const float bound2 = (velDeviation + maxv) / denom2;
        float bound1;
        if (abs(velDeviation - maxv) < AD_EPS_F32)
            bound1 = -AD_MAX_F32;
        else
            bound1 = (velDeviation - maxv) > 0.0f ? AD_MAX_F32 : -AD_MAX_F32;
        return float2(adPMax(0.0f, bound1), bound2);
    }
    const float bound1 = (velDeviation - maxv) / denom1;
    const float bound2 = (velDeviation + maxv) / denom2;
    if (denom1 >= 0.0f)
        return float2(adPMax(0.0f, bound1), adPMin(bound2, impulse));
    return float2(0.0f, adPMin(adPMin(bound1, bound2), impulse));
}

// clampDriveImpulse (L279-331)
public float adClampDriveImpulse(float jointVel0, float driveImpulse0, float driveImpulse,
                                 float response, float maxJointVel, float maxImpulse,
                                 float speedImpulseGradient, float vdr)
{
    const float velDeviation = driveImpulse0 * response - jointVel0;
    float2 bounds1, bounds2;

    if (driveImpulse > 0.0f)
    {
        float denom1 = response - speedImpulseGradient;
        float denom2 = response + speedImpulseGradient;
        bounds1 = adBoundsPos(velDeviation, maxJointVel, denom1, denom2, driveImpulse);
        if (vdr == 0.0f)
            bounds2 = float2(-maxImpulse, maxImpulse);
        else
        {
            denom1 = response - 1.0f / vdr;
            denom2 = response + 1.0f / vdr;
            bounds2 = adBoundsPos(velDeviation, maxImpulse / vdr, denom1, denom2, driveImpulse);
        }
    }
    else
    {
        float denom1 = response + speedImpulseGradient;
        float denom2 = response - speedImpulseGradient;
        bounds1 = adBoundsNeg(velDeviation, maxJointVel, denom1, denom2, driveImpulse);
        if (vdr == 0.0f)
            bounds2 = float2(-maxImpulse, maxImpulse);
        else
        {
            denom1 = response + 1.0f / vdr;
            denom2 = response - 1.0f / vdr;
            bounds2 = adBoundsNeg(velDeviation, maxImpulse / vdr, denom1, denom2, driveImpulse);
        }
    }

    const float lowerBound = adPMax(bounds1.x, bounds2.x);
    const float upperBound = adPMin(bounds1.y, bounds2.y);
    if (lowerBound > upperBound)
        return 0.0f;
    return adPClamp(driveImpulse, lowerBound, upperBound);
}

// computeLimitImpulse (DyCpuGpuArticulation.h:352-455) — the joint-limit
// impulse with the future-violation lookahead, the repulsive-only clamps and
// the pull-back branch when no limit is active.  The impulse WRITEBACK is
// conditional (newLow/newHigh) — an unconditional write of the unchanged
// value would be byte-identical; kept faithful anyway.
public float adLimitImpulse(float dt, float recipDt, bool isVelIter,
                            float response, float recipResponse, float erp,
                            float errorLow, float errorHigh, float jointPDelta,
                            inout float lowImpulse_, inout float highImpulse_,
                            inout float jointV_)
{
    float jointV = jointV_;
    float lowImpulse = lowImpulse_;
    float highImpulse = highImpulse_;

    const float futureDeltaJointP = jointPDelta + jointV * dt;

    const float currErrLow = errorLow + jointPDelta;
    const float nextErrLow = errorLow + futureDeltaJointP;
    const float currErrHigh = errorHigh - jointPDelta;
    const float nextErrHigh = errorHigh - futureDeltaJointP;

    bool limited = false;
    bool newLow = false;
    bool newHigh = false;

    const float tolerance = 0.0f;

    float deltaF = 0.0f;
    if (currErrLow < tolerance || nextErrLow < tolerance)
    {
        float newJointV = jointV;
        limited = true;
        if (currErrLow < tolerance)
        {
            if (!isVelIter)
                newJointV = -currErrLow * recipDt * erp;
        }
        else
        {
            newJointV = -currErrLow * recipDt;
        }

        const float deltaV = newJointV - jointV;
        deltaF = adPMax(lowImpulse + deltaV * recipResponse, 0.0f) - lowImpulse;
        lowImpulse += deltaF;
        newLow = true;
    }
    else if (currErrHigh < tolerance || nextErrHigh < tolerance)
    {
        float newJointV = jointV;
        limited = true;
        if (currErrHigh < tolerance)
        {
            if (!isVelIter)
                newJointV = currErrHigh * recipDt * erp;
        }
        else
        {
            newJointV = currErrHigh * recipDt;
        }

        const float deltaV = newJointV - jointV;
        deltaF = adPMin(highImpulse + deltaV * recipResponse, 0.0f) - highImpulse;
        highImpulse += deltaF;
        newHigh = true;
    }

    if (!limited)
    {
        const float impulseForZeroVel = -jointV * recipResponse;
        if (jointV > 0.0f)
        {
            deltaF = adPMax(impulseForZeroVel, -lowImpulse);
            lowImpulse += deltaF;
            newLow = true;
        }
        else
        {
            deltaF = adPMin(impulseForZeroVel, -highImpulse);
            highImpulse += deltaF;
            newHigh = true;
        }
    }

    jointV += deltaF * response;

    if (newLow)
        lowImpulse_ = lowImpulse;
    if (newHigh)
        highImpulse_ = highImpulse;
    jointV_ = jointV;

    return deltaF;
}
