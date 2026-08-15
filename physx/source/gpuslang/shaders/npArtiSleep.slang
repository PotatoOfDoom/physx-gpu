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


// npArtiSleep.slang
//
// P8.95a: `sleepCheck1T` (forwardDynamic2.cu:2706) -- the per-link wake-counter
// and sleep-energy update that `updateBodiesLaunch_Part2` runs last.  The
// kernel itself is P8.95b; this module carries the one genuinely NEW body it
// needs, with its own fixture and sweep, so the assembly stays composition.
//
// No transcendental anywhere: quaternion rotateInv, dot products and a divide.
// The oracle is therefore BIT-EXACT again, the first articulation tranche
// since P8.93 that can say so.
//
// ===========================================================================
// THE TRAPS
// ===========================================================================
//
//  1. **The energy is only examined when `wc < resetTime/2 || wc < dt`** -- two
//     conditions, and the second matters exactly when dt exceeds half the
//     reset time.
//  2. **The inertia reciprocal guard yields ONE, not zero**:
//     `x > 0 ? 1/x : 1.f` -- unlike every inverse in P8.84, a degenerate
//     component contributes the ANGULAR VELOCITY SQUARED ITSELF, not nothing.
//  3. **Only the angular accumulator is rotated into the BODY frame**
//     (`q.rotateInv`); the linear one accumulates in world space untouched.
//  4. **`invMass == 0` becomes 1** -- a static link still accumulates energy.
//  5. **The reset writes `factor * 0.5 * resetTime`** with
//     `factor = threshold == 0 ? 2 : min(E / threshold, 2)` -- and the
//     accumulators are zeroed on the SPOT, before the store.
//  6. **The decrement happens only when the reset did not** (`alreadyUpdateWC`),
//     and clamps at zero.
//  7. **`wc == 0` zeroes the accumulators again** -- a separate zeroing from
//     trap 5's, reachable without it.
//  8. **The accumulator stores write w = 0**, not the loaded w.

import pxgSlangCommon;
import pxgSlangArticulationDesc;

static const float ASL_WAKE_COUNTER_RESET_TIME = 20.0f * 0.02f;

// sleepCheck1T (forwardDynamic2.cu:2706).  `sleepData` is the RECORD-side
// PxgArticulationLinkSleepData for this link; `wakeCounter` is in/out.
public float aslSleepCheck1T(uint64_t sleepData, inout float wakeCounter, float4 body2WorldQ,
                             float3 linearMotionVel, float3 angularMotionVel, float dt,
                             float sleepThreshold, float4 inverseInertiaXYZ_invMassW)
{
    Ptr<float4> sd = Ptr<float4>(sleepData);
    const float4 sleepLinVelAccXYZ = sd[0];   // PXG_ARTI_SLEEPDATA_LINVEL
    const float4 sleepAngVelAccXYZ = sd[1];   // PXG_ARTI_SLEEPDATA_ANGVEL

    float wc = wakeCounter;

    bool alreadyUpdateWC = false;
    float3 sleepLinVelAcc = float3(0.0f, 0.0f, 0.0f);
    float3 sleepAngVelAcc = float3(0.0f, 0.0f, 0.0f);

    // trap 1: TWO conditions
    if (wc < ASL_WAKE_COUNTER_RESET_TIME * 0.5f || wc < dt)
    {
        // trap 2: the degenerate reciprocal is ONE
        const float3 inertia =
            float3(inverseInertiaXYZ_invMassW.x > 0.0f ? 1.0f / inverseInertiaXYZ_invMassW.x
                                                       : 1.0f,
                   inverseInertiaXYZ_invMassW.y > 0.0f ? 1.0f / inverseInertiaXYZ_invMassW.y
                                                       : 1.0f,
                   inverseInertiaXYZ_invMassW.z > 0.0f ? 1.0f / inverseInertiaXYZ_invMassW.z
                                                       : 1.0f);

        sleepLinVelAcc = linearMotionVel;
        // trap 3: only the ANGULAR half changes frame
        // pxgSlangCommon's quatRotateInv IS PxQuat::rotateInv's spelling to the
        // bit (P8.95a first carried a private copy; one shared body, not two)
        sleepAngVelAcc = quatRotateInv(body2WorldQ, angularMotionVel);

        sleepLinVelAcc.x += sleepLinVelAccXYZ.x;
        sleepLinVelAcc.y += sleepLinVelAccXYZ.y;
        sleepLinVelAcc.z += sleepLinVelAccXYZ.z;

        sleepAngVelAcc.x += sleepAngVelAccXYZ.x;
        sleepAngVelAcc.y += sleepAngVelAccXYZ.y;
        sleepAngVelAcc.z += sleepAngVelAccXYZ.z;

        // trap 4
        float invMass = inverseInertiaXYZ_invMassW.w;
        if (invMass == 0.0f)
            invMass = 1.0f;

        const float angular = dot(sleepAngVelAcc * sleepAngVelAcc, inertia) * invMass;
        const float linear = dot(sleepLinVelAcc, sleepLinVelAcc);
        const float normalizedEnergy = 0.5f * (angular + linear);

        if (normalizedEnergy >= sleepThreshold)
        {
            // trap 5
            sleepLinVelAcc = float3(0.0f, 0.0f, 0.0f);
            sleepAngVelAcc = float3(0.0f, 0.0f, 0.0f);
            const float factor = sleepThreshold == 0.0f
                                     ? 2.0f
                                     : min(normalizedEnergy / sleepThreshold, 2.0f);
            wc = factor * 0.5f * ASL_WAKE_COUNTER_RESET_TIME;
            alreadyUpdateWC = true;
        }
    }

    // trap 6
    if (!alreadyUpdateWC)
        wc = max(wc - dt, 0.0f);

    // trap 7
    const bool wakeCounterZero = (wc == 0.0f);
    if (wakeCounterZero)
    {
        sleepLinVelAcc = float3(0.0f, 0.0f, 0.0f);
        sleepAngVelAcc = float3(0.0f, 0.0f, 0.0f);
    }

    wakeCounter = wc;
    // trap 8: w = 0 on both stores
    sd[0] = float4(sleepLinVelAcc.x, sleepLinVelAcc.y, sleepLinVelAcc.z, 0.0f);
    sd[1] = float4(sleepAngVelAcc.x, sleepAngVelAcc.y, sleepAngVelAcc.z, 0.0f);

    return wc;
}

// ---------------------------------------------------------------------------
// The probe entry: one case per thread, everything through BDA.
// in:  float[nbCases * 16] (sleepLin4, sleepAng4, q4, wc, dt, threshold, pad)
//      + float[nbCases * 8] (linVel3, angVel3 pad2) + float4 invInertia per case
// laid out as three arrays for clarity.
// out: float[nbCases * 12]: sleepLin4, sleepAng4, wc, ret, pad, pad
// ---------------------------------------------------------------------------
struct AslParams
{
    uint64_t inA;    // @0  float[nbCases*16]
    uint64_t inB;    // @8  float[nbCases*12] (linVel3 angVel3 invInertia4 pad2)
    uint64_t outF;   // @16 float[nbCases*12]
    uint nbCases;    // @24
    uint pad0;       // @28 -> 32
};

[[vk::push_constant]]
ConstantBuffer<AslParams> gAsl;

[shader("compute")]
[numthreads(64, 1, 1)]
void artiSleepCheck(uint3 dispatchThreadId : SV_DispatchThreadID)
{
    const uint tid = dispatchThreadId.x;
    if (tid >= gAsl.nbCases)
        return;

    Ptr<float> a = Ptr<float>(gAsl.inA + uint64_t(tid) * 16u * 4u);
    Ptr<float> b = Ptr<float>(gAsl.inB + uint64_t(tid) * 12u * 4u);
    Ptr<float> o = Ptr<float>(gAsl.outF + uint64_t(tid) * 12u * 4u);

    // the sleep data is written IN PLACE, as in the kernel: stage it into the
    // output buffer first and hand the function that address
    o[0] = a[0]; o[1] = a[1]; o[2] = a[2]; o[3] = a[3];
    o[4] = a[4]; o[5] = a[5]; o[6] = a[6]; o[7] = a[7];

    float wc = a[8];
    const float dt = a[9];
    const float threshold = a[10];
    const float4 q = float4(a[12], a[13], a[14], a[15]);

    const float ret = aslSleepCheck1T(gAsl.outF + uint64_t(tid) * 12u * 4u, wc, q,
                                      float3(b[0], b[1], b[2]), float3(b[3], b[4], b[5]), dt,
                                      threshold, float4(b[6], b[7], b[8], b[9]));
    o[8] = wc;
    o[9] = ret;
    o[10] = 0.0f;
    o[11] = 0.0f;
}
