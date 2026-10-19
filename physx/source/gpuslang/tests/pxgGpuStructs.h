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

#ifndef PXG_TESTS_GPU_STRUCTS_H
#define PXG_TESTS_GPU_STRUCTS_H

// Standalone C++ mirrors of the PhysX GPU data structures used by the ported
// kernels, with static_asserts pinning the byte layout that the Slang side
// (pxgSlangCommon.slang) declares.  These intentionally avoid including the
// PhysX headers so the verification suite builds without the full SDK build
// (the CUDA originals pull in <vector_types.h> from the CUDA toolkit).
//
// Layout ground truth: PxgBodySim.h, PxgSolverBody.h, PxgShapeSim.h,
// PxgConvexConvexShape.h, PxsTransformCache.h, AlignedTransform.h.

#include <cstdint>
#include <cstring>

namespace pxgtest
{

struct Float4
{
    float x, y, z, w;
};

struct Float3
{
    float x, y, z;
};

struct Quat
{
    float x, y, z, w;
};

// PxTransform: 28 bytes, { PxQuat q; PxVec3 p; }
struct Transform28
{
    Quat q;
    Float3 p;
};
static_assert(sizeof(Transform28) == 28, "PxTransform mirror must be 28 bytes");

// PxAlignedTransform: 32 bytes, { PxAlignedQuat q; float4 p; }
struct AlignedTransform
{
    Float4 q;
    Float4 p;
};
static_assert(sizeof(AlignedTransform) == 32, "PxAlignedTransform mirror must be 32 bytes");

// PxgBodySim: 224 bytes
struct BodySim
{
    Float4 linearVelocityXYZ_inverseMassW;                                                 // 0
    Float4 angularVelocityXYZ_maxPenBiasW;                                                 // 16
    Float4 maxLinearVelocitySqX_maxAngularVelocitySqY_linearDampingZ_angularDampingW;      // 32
    Float4 inverseInertiaXYZ_contactReportThresholdW;                                      // 48
    Float4 sleepLinVelAccXYZ_freezeCountW;                                                 // 64
    Float4 sleepAngVelAccXYZ_accelScaleW;                                                  // 80
    Float4 freezeThresholdX_wakeCounterY_sleepThresholdZ_bodySimIndex;                     // 96
    AlignedTransform body2World;                                                           // 112
    AlignedTransform body2Actor_maxImpulseW;                                               // 144
    uint32_t articulationRemapId;                                                          // 176
    uint32_t internalFlags;                                                                // 180
    uint16_t lockFlags;                                                                    // 184
    uint16_t disableGravity;                                                               // 186
    float offsetSlop;                                                                      // 188
    Float4 externalLinearAcceleration;                                                     // 192
    Float4 externalAngularAcceleration;                                                    // 208
};
static_assert(sizeof(BodySim) == 224, "PxgBodySim mirror must be 224 bytes");
static_assert(offsetof(BodySim, body2World) == 112, "PxgBodySim::body2World offset");
static_assert(offsetof(BodySim, externalLinearAcceleration) == 192, "PxgBodySim::externalLinearAcceleration offset");

// PxgSolverBodyData: 96 bytes (16-byte aligned in C++)
struct SolverBodyData
{
    Float4 initialAngVelXYZ_penBiasClamp; // 0
    Float4 initialLinVelXYZ_invMassW;     // 16
    AlignedTransform body2World;          // 32
    uint64_t islandNodeIndex;             // 64 (PxNodeIndex: low32 = mID, high32 = mLinkID)
    float reportThreshold;                // 72
    float maxImpulse;                     // 76
    uint32_t flags;                       // 80
    float offsetSlop;                     // 84
    uint32_t pad[2];                      // 88
};
static_assert(sizeof(SolverBodyData) == 96, "PxgSolverBodyData mirror must be 96 bytes");

// PxgSolverBodySleepData: 8 bytes
struct SolverBodySleepData
{
    float wakeCounter;
    uint32_t internalFlags;
};
static_assert(sizeof(SolverBodySleepData) == 8, "PxgSolverBodySleepData mirror must be 8 bytes");

// PxgSolverTxIData: 64 bytes { PxTransform deltaBody2World; PxMat33 sqrtInvInertia; }
struct SolverTxIData
{
    Transform28 deltaBody2World; // 0
    float sqrtInvInertia[9];     // 28 (column major)
};
static_assert(sizeof(SolverTxIData) == 64, "PxgSolverTxIData mirror must be 64 bytes");

// PxgShapeSim: 72 bytes
struct ShapeSim
{
    Transform28 transform;    // 0
    Float3 localBoundsMin;    // 28
    Float3 localBoundsMax;    // 40
    uint32_t pad0;            // 52
    uint64_t bodySimIndex;    // 56 (PxNodeIndex)
    uint32_t hullDataIndex;   // 64
    uint16_t shapeFlags;      // 68
    uint16_t shapeType;       // 70
};
static_assert(sizeof(ShapeSim) == 72, "PxgShapeSim mirror must be 72 bytes");
static_assert(offsetof(ShapeSim, bodySimIndex) == 56, "PxgShapeSim::bodySimIndex offset");

// PxgShape: 48 bytes; hullOrMeshPtr repurposed as byte offset into the
// geometry blob buffer (see updateBounds.slang header comment)
struct Shape
{
    Float3 scale;             // 0  (PxMeshScale::scale)
    Quat scaleRotation;       // 12 (PxMeshScale::rotation)
    uint32_t materialIndex;   // 28
    uint64_t hullOrMeshOffset;// 32 (was size_t hullOrMeshPtr)
    uint32_t type;            // 40
    uint32_t particleOrSoftbodyId; // 44
};
static_assert(sizeof(Shape) == 48, "PxgShape mirror must be 48 bytes");
static_assert(offsetof(Shape, hullOrMeshOffset) == 32, "PxgShape::hullOrMeshOffset offset");

// PxsCachedTransform: 32 bytes
struct CachedTransform
{
    Transform28 transform; // 0
    uint32_t flags;        // 28
};
static_assert(sizeof(CachedTransform) == 32, "PxsCachedTransform mirror must be 32 bytes");

// PxBounds3: 24 bytes
struct Bounds3
{
    Float3 minimum;
    Float3 maximum;
};
static_assert(sizeof(Bounds3) == 24, "PxBounds3 mirror must be 24 bytes");

// ---- flag constants (PxsRigidBody.h / PxRigidDynamic.h / PxShape.h) ----
enum BodyInternalFlags : uint32_t
{
    eFROZEN = 1u << 0,
    eFREEZE_THIS_FRAME = 1u << 1,
    eUNFREEZE_THIS_FRAME = 1u << 2,
    eACTIVATE_THIS_FRAME = 1u << 3,
    eDEACTIVATE_THIS_FRAME = 1u << 4,
    // 5.12: eDISABLE_GRAVITY_GPU (1 << 5) is gone; the *_GPU names below are
    // PxsRigidBody::eSPECULATIVE_CCD_GPU / eENABLE_GYROSCOPIC_GPU / eRETAIN_ACCELERATION_GPU
    eSPECULATIVE_CCD = 1u << 6,
    eENABLE_GYROSCOPIC = 1u << 7,
    eRETAIN_ACCELERATION = 1u << 8,
    eFIRST_BODY_COPY_GPU = 1u << 9,
    eVELOCITY_COPY_GPU = 1u << 10,
    eGPU_FLAGS = eSPECULATIVE_CCD | eENABLE_GYROSCOPIC | eRETAIN_ACCELERATION | eFIRST_BODY_COPY_GPU |
                 eVELOCITY_COPY_GPU,
    eFREE_FLAGS = 0x1Fu << 11   // eFREE_FLAG_1..5
};

enum LockFlags : uint32_t
{
    eLOCK_LINEAR_X = 1u << 0,
    eLOCK_LINEAR_Y = 1u << 1,
    eLOCK_LINEAR_Z = 1u << 2,
    eLOCK_ANGULAR_X = 1u << 3,
    eLOCK_ANGULAR_Y = 1u << 4,
    eLOCK_ANGULAR_Z = 1u << 5
};

enum RigidBodyFlags : uint32_t
{
    eRB_ENABLE_SPECULATIVE_CCD = 1u << 3,
    eRB_ENABLE_GYROSCOPIC_FORCES = 1u << 6
};

enum ShapeFlags : uint32_t
{
    eSIMULATION_SHAPE = 1u << 0,
    eSCENE_QUERY_SHAPE = 1u << 1,
    eTRIGGER_SHAPE = 1u << 2
};

enum GeometryType : uint16_t
{
    eSPHERE = 0,
    ePLANE = 1,
    eCAPSULE = 2,
    eBOX = 3,
    eCONVEXCORE = 4,
    eCONVEXMESH = 5
};

// DySleepingConfigulation.h
static const float PXD_FREEZE_INTERVAL = 1.5f;
static const float PXD_FREEZE_TOLERANCE = 0.25f;
static const float PXD_SLEEP_DAMPING = 0.5f;
static const float PXD_FREEZE_SCALE = 0.1f;

static const uint32_t PX_INVALID_NODE = 0xFFFFFFFFu;

inline uint64_t makeNodeIndex(uint32_t id, uint32_t linkId = 0, bool isArticulation = false)
{
    const uint32_t linkWord = (linkId << 1) | (isArticulation ? 1u : 0u);
    return (uint64_t(linkWord) << 32) | uint64_t(id);
}

} // namespace pxgtest

#endif // PXG_TESTS_GPU_STRUCTS_H
