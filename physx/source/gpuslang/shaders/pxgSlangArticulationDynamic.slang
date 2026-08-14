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
