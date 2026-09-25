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

// npMaterialCombine.slang
//
// Slang port of PxsCombineMaterials (physx/.../PxsMaterialCombiner.h), the
// material-combining core called by insertIntoPatchStream (contactPatchUtils.cuh)
// when a narrow-phase contact patch is written.  Given two material records it
// produces the combined static/dynamic friction, restitution, damping and
// material flags, honouring the per-property combine modes, the compliant
// (negative-restitution) special cases and the friction-disable path.  The atomic
// patch/point stream allocation and the PxContactPatch struct write are deferred
// to the narrow-phase integration.
//
// PxCombineMode: eAVERAGE=0, eMIN=1, eMULTIPLY=2, eMAX=3.
// PxMaterialFlag: eDISABLE_FRICTION=1, eDISABLE_STRONG_FRICTION=2,
//                 eCOMPLIANT_ACCELERATION_SPRING=16.

struct MaterialData
{
    float dynamicFriction;
    float staticFriction;
    float restitution;
    float damping;
    uint flags;
    uint fricCombineMode;
    uint restCombineMode;
    uint dampingCombineMode;
}

struct MatParams { uint numPairs; }

[[vk::push_constant]]
ConstantBuffer<MatParams> gParams;

RWStructuredBuffer<MaterialData> gMaterials; // binding 0
RWStructuredBuffer<uint2> gPairs;            // binding 1  (materialIndex0, materialIndex1)
RWStructuredBuffer<float4> gOutFriction;     // binding 2  (staticFriction, dynamicFriction, restitution, damping)
RWStructuredBuffer<uint> gOutFlags;          // binding 3

static const uint MODE_AVERAGE = 0u, MODE_MIN = 1u, MODE_MULTIPLY = 2u, MODE_MAX = 3u;
static const uint FLAG_DISABLE_FRICTION = 1u, FLAG_DISABLE_STRONG_FRICTION = 2u, FLAG_COMPLIANT_ACC_SPRING = 16u;

// mirrors combineScalars / PxsCombinePxReal
float combineScalars(float a, float b, uint mode)
{
    if (mode == MODE_AVERAGE) return 0.5f * (a + b);
    if (mode == MODE_MIN) return min(a, b);
    if (mode == MODE_MULTIPLY) return a * b;
    if (mode == MODE_MAX) return max(a, b);
    return 0.0f;
}

[shader("compute")]
[numthreads(256, 1, 1)]
void combineMaterials(uint3 dtid : SV_DispatchThreadID)
{
    const uint i = dtid.x;
    if (i >= gParams.numPairs) return;

    const uint2 pair = gPairs[i];
    // P9.88: combineMaterials takes PxU16 index0/index1 (materialCombiner.cuh:35).
    const MaterialData m0 = gMaterials[pair.x & 0xFFFFu];
    const MaterialData m1 = gMaterials[pair.y & 0xFFFFu];

    const float r0 = m0.restitution;
    const float r1 = m1.restitution;
    const bool compliant0 = r0 < 0.0f;
    const bool compliant1 = r1 < 0.0f;
    const bool exactlyOneCompliant = compliant0 != compliant1;
    const bool bothCompliant = compliant0 && compliant1;
    const bool compliantAcc0 = (m0.flags & FLAG_COMPLIANT_ACC_SPRING) != 0u;
    const bool compliantAcc1 = (m1.flags & FLAG_COMPLIANT_ACC_SPRING) != 0u;
    const bool exactlyOneAccCompliant = compliantAcc0 != compliantAcc1;

    // combine restitution
    float combinedRestitution;
    if (bothCompliant && exactlyOneAccCompliant)
    {
        combinedRestitution = compliantAcc0 ? r0 : r1;
    }
    else
    {
        const uint combineMode = exactlyOneCompliant ? MODE_MIN : max(m0.restCombineMode, m1.restCombineMode);
        const float flipSign = (bothCompliant && (combineMode == MODE_MULTIPLY)) ? -1.0f : 1.0f;
        combinedRestitution = flipSign * combineScalars(r0, r1, combineMode);
    }

    // combine damping
    float combinedDamping;
    {
        const float d0 = m0.damping;
        const float d1 = m1.damping;
        if (bothCompliant && exactlyOneAccCompliant)
        {
            combinedDamping = compliantAcc0 ? d0 : d1;
        }
        else
        {
            const uint combineMode = exactlyOneCompliant ? MODE_MAX : max(m0.dampingCombineMode, m1.dampingCombineMode);
            combinedDamping = combineScalars(d0, d1, combineMode);
        }
    }

    // combine isotropic friction
    float combinedDynamicFriction;
    float combinedStaticFriction;
    uint combinedMaterialFlags;
    {
        const uint combineFlags = m0.flags | m1.flags;
        if ((combineFlags & FLAG_DISABLE_FRICTION) == 0u)
        {
            const uint frictionCombineMode = max(m0.fricCombineMode, m1.fricCombineMode);
            const float dynFriction = combineScalars(m0.dynamicFriction, m1.dynamicFriction, frictionCombineMode);
            const float staFriction = combineScalars(m0.staticFriction, m1.staticFriction, frictionCombineMode);
            const float fDynFriction = max(dynFriction, 0.0f);
            const float fStaFriction = (staFriction - fDynFriction) >= 0.0f ? staFriction : fDynFriction;
            combinedDynamicFriction = fDynFriction;
            combinedStaticFriction = fStaFriction;
            combinedMaterialFlags = combineFlags;
        }
        else
        {
            combinedMaterialFlags = combineFlags | FLAG_DISABLE_STRONG_FRICTION;
            combinedDynamicFriction = 0.0f;
            combinedStaticFriction = 0.0f;
        }
    }

    gOutFriction[i] = float4(combinedStaticFriction, combinedDynamicFriction, combinedRestitution, combinedDamping);
    gOutFlags[i] = combinedMaterialFlags;
}
