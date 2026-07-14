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

// npContactPrePrep.slang
//
// Slang port of the per-lane contact-patch packing core of
// createSolverContactConstraintDescsFromPatchWithSharedMem
// (gpusolver/src/CUDA/constraintBlockPrePrep.cu L167-253) plus the
// per-lane contact-point emission (L304-319; flat-equivalent to the
// 4-float4-per-pass shared-mem chunking because chunks are 4 float4s and
// the modifiable stride is 2, so pairs never straddle a chunk; also
// matches the commented-out reference form at L410-422) and the
// mWriteback = forceIndex + contactStartIndex store (L1044):
//   - perPointFriction = internalFlags & eHAS_TARGET_VELOCITY(16) ||
//     wuFlags & eDISABLE_STRONG_FRICTION(1<<2);
//   - the eFORCE_NO_RESPONSE(4) gate; inside it: the modifiable stride
//     (eCOMPRESSED_MODIFIED_CONTACT(128) -> contact pointer advanced by
//     startContactIndex a SECOND time, stride 2), numContacts/
//     contactStartIndex, the eDOMINANCE_0/1 (1<<14 / 1<<15) zeroing of the
//     mass-modification lanes, normal_restitution/damping passthrough,
//     dynamic/staticFriction, disableFriction = materialFlags &
//     eDISABLE_FRICTION(1) and isAccelerationSpring = materialFlags &
//     eCOMPLIANT_ACCELERATION_SPRING(16);
//   - solver flags: ePER_POINT_FRICTION(2), eHAS_FORCE_THRESHOLDS(1) from
//     wuFlags & eFORCE_THRESHOLD(1<<9), eDISABLE_FRICTION(4),
//     eCOMPLIANT_ACCELERATION_SPRING(8) (PxgSolverFlags.h L34-54);
//   - the PxgMaterialContactData image (PxgConstraintPrep.h L85-94):
//     (restDistance, staticFriction, dynamicFriction,
//      mNumContacts | mSolverFlags<<8 | prevFrictionPatchCount<<16);
//   - contact emission: numContacts pairs of (point_separationW,
//     targetVel_maxImpulseW), the latter (0,0,0,PX_MAX_REAL) unless
//     modifiable.
// ALL outputs bitwise.  CUDA reads UNINITIALIZED dynamic/staticFriction +
// mSolverFlags when gContactPatch is NULL (the data image is still stored),
// and NEVER writes data.pad, so byte 15 of the stored image is stack
// garbage in every case -- the port zero-initializes those lanes/bytes;
// deterministic-and-documented replacement of UB.  CUDA also leaves the prepData lanes (invMass, normal,
// damping) UNWRITTEN outside the response gate; the port zeroes them
// (consumers read them only when mNumContacts > 0).

static const float FLT_MAXQ = 3.402823466e+38f;
static const uint MAX_PTS = 20u;

struct CpIn
{
    uint4 meta;    // hasPatch, wuFlags, internalFlags, materialFlags
    float4 mass;   // mMassModification: invMassScale0, invInertiaScale0, invMassScale1, invInertiaScale1
    float4 nr;     // normal.xyz, restitution
    float4 mat;    // dynamicFriction, staticFriction, damping, restDistance
    uint4 cnt;     // startContactIndex, nbContacts, prevFrictionPatchCount, forceIndex
    float4 pts[MAX_PTS];  // compressed contact stream (base, pre-offset)
}

struct CpOut
{
    float4 invMassOut;   // dominance-zeroed mass modification
    float4 nrOut;        // normal_restitutionW
    float4 dampDataOut;  // x = damping, y = asfloat(writeback), zw = 0
    float4 dataOut;      // PxgMaterialContactData image
    uint4 counts;        // numContacts, contactStartIndex, stride, perPointFriction
    float4 pointsOut[12];  // per contact: (point_separationW, targetVel_maxImpulseW)
}

struct CpParams { uint numTests; }

[[vk::push_constant]]
ConstantBuffer<CpParams> gParams;

RWStructuredBuffer<CpIn> gIn;    // binding 0
RWStructuredBuffer<CpOut> gOut;  // binding 1

[shader("compute")]
[numthreads(64, 1, 1)]
void cppPack(uint3 dtid : SV_DispatchThreadID)
{
    const uint tid = dtid.x;
    if (tid >= gParams.numTests) return;
    const CpIn s = gIn[tid];

    const uint hasPatch = s.meta.x;
    const uint wuFlags = s.meta.y;
    const uint internalFlags = s.meta.z;
    const uint materialFlags = s.meta.w;

    uint numContacts = 0u;
    uint stride = 1u;
    uint contactStartIndex = 0u;
    uint ptOffset = 0u;

    CpOut o;
    o.invMassOut = float4(0, 0, 0, 0);
    o.nrOut = float4(0, 0, 0, 0);
    // CUDA: dynamicFriction/staticFriction/mSolverFlags are uninitialized when
    // !hasPatch or the response gate skips; the port pins them to zero
    float dynamicFriction = 0.0f, staticFriction = 0.0f;
    uint solverFlags = 0u;
    float damping = 0.0f;
    bool perPointFriction = false;

    if (hasPatch != 0u)
    {
        perPointFriction = (internalFlags & 16u) != 0u ||    // eHAS_TARGET_VELOCITY
                           (wuFlags & (1u << 2u)) != 0u;     // eDISABLE_STRONG_FRICTION
        bool disableFriction = false;
        bool isAccelerationSpring = false;

        if ((internalFlags & 4u) == 0u)                      // eFORCE_NO_RESPONSE
        {
            const bool isModifiable = (internalFlags & 128u) != 0u;  // eCOMPRESSED_MODIFIED_CONTACT

            ptOffset = s.cnt.x;                              // contactPoints += startContactIndex
            if (isModifiable)
            {
                ptOffset += s.cnt.x;                         // += startContactIndex again
                stride = 2u;
            }

            numContacts = s.cnt.y;
            contactStartIndex = s.cnt.x;

            float4 lin0X_ang0Y_lin1Z_ang1W = s.mass;
            if ((wuFlags & (1u << 14u)) != 0u)               // eDOMINANCE_0
            {
                lin0X_ang0Y_lin1Z_ang1W.x = 0.0f;
                lin0X_ang0Y_lin1Z_ang1W.y = 0.0f;
            }
            if ((wuFlags & (1u << 15u)) != 0u)               // eDOMINANCE_1
            {
                lin0X_ang0Y_lin1Z_ang1W.z = 0.0f;
                lin0X_ang0Y_lin1Z_ang1W.w = 0.0f;
            }
            o.invMassOut = lin0X_ang0Y_lin1Z_ang1W;

            o.nrOut = s.nr;
            damping = s.mat.z;

            dynamicFriction = s.mat.x;
            staticFriction = s.mat.y;
            disableFriction = (materialFlags & 1u) != 0u;    // PxMaterialFlag::eDISABLE_FRICTION
            isAccelerationSpring = (materialFlags & 16u) != 0u;  // eCOMPLIANT_ACCELERATION_SPRING
        }

        if (perPointFriction)
            solverFlags |= 2u;                               // ePER_POINT_FRICTION
        if ((wuFlags & (1u << 9u)) != 0u)                    // eFORCE_THRESHOLD
            solverFlags |= 1u;                               // eHAS_FORCE_THRESHOLDS
        if (disableFriction)
            solverFlags |= 4u;                               // eDISABLE_FRICTION
        if (isAccelerationSpring)
            solverFlags |= 8u;                               // eCOMPLIANT_ACCELERATION_SPRING
    }

    // PxgMaterialContactData image: restDistance, staticFriction,
    // dynamicFriction, mNumContacts | mSolverFlags<<8 | prevFriction<<16
    const uint packed = (numContacts & 0xFFu) | (solverFlags << 8u) | ((s.cnt.z & 0xFFu) << 16u);
    o.dataOut = float4(s.mat.w, staticFriction, dynamicFriction, asfloat(packed));

    // n.mWriteback = forceIndex + contactStartIndex (uint wrap allowed:
    // forceIndex defaults to 0xFFFFFFFF when no force buffer exists)
    o.dampDataOut = float4(damping, asfloat(s.cnt.w + contactStartIndex), 0.0f, 0.0f);
    o.counts = uint4(numContacts, contactStartIndex, stride, perPointFriction ? 1u : 0u);

    // contact emission (flat-equivalent form of the shared-mem chunk loop)
    for (uint j = 0u; j < 12u; ++j)
        o.pointsOut[j] = float4(0, 0, 0, 0);
    const uint count = numContacts * stride;
    uint emit = 0u;
    for (uint a = 0u; a < count; a += stride)
    {
        o.pointsOut[emit * 2u + 0u] = s.pts[ptOffset + a];
        if (stride != 1u)
            o.pointsOut[emit * 2u + 1u] = s.pts[ptOffset + a + 1u];
        else
            o.pointsOut[emit * 2u + 1u] = float4(0.0f, 0.0f, 0.0f, FLT_MAXQ);
        emit += 1u;
    }

    gOut[tid] = o;
}
