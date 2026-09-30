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

// pxgSlangCommon.slang
//
// Shared math types, GPU data-structure layouts and constants for the
// Slang port of the PhysX GPU (formerly CUDA) kernels.
//
// The struct layouts below mirror the C++ structs used by the original
// CUDA pipeline byte-for-byte (std430 storage-buffer layout).  Structs
// that contain embedded device pointers in the CUDA code (e.g.
// PxgShape::hullOrMeshPtr) reinterpret the pointer field as a byte
// offset into a bound storage buffer, since descriptor-based APIs
// (Vulkan / D3D12) do not support raw device pointers in the same way.

module pxgSlangCommon;

// ---------------------------------------------------------------------------
// Constants (mirrors PxsRigidBody.h internal flags)
// ---------------------------------------------------------------------------
public static const uint PXG_BODY_FROZEN                = 1u << 0;
public static const uint PXG_BODY_FREEZE_THIS_FRAME     = 1u << 1;
public static const uint PXG_BODY_UNFREEZE_THIS_FRAME   = 1u << 2;
public static const uint PXG_BODY_ACTIVATE_THIS_FRAME   = 1u << 3;
public static const uint PXG_BODY_DEACTIVATE_THIS_FRAME = 1u << 4;
public static const uint PXG_BODY_DISABLE_GRAVITY_GPU   = 1u << 5;
public static const uint PXG_BODY_SPECULATIVE_CCD       = 1u << 6;
public static const uint PXG_BODY_ENABLE_GYROSCOPIC     = 1u << 7;
public static const uint PXG_BODY_RETAIN_ACCELERATION   = 1u << 8;

// PxRigidBodyFlag (public API flags written to PxgSolverBodyData::flags)
public static const uint PXG_RB_FLAG_ENABLE_SPECULATIVE_CCD  = 1u << 3; // PxRigidBodyFlag::eENABLE_SPECULATIVE_CCD
public static const uint PXG_RB_FLAG_ENABLE_GYROSCOPIC       = 1u << 6; // PxRigidBodyFlag::eENABLE_GYROSCOPIC_FORCES

// PxRigidDynamicLockFlag
public static const uint PXG_LOCK_LINEAR_X  = 1u << 0;
public static const uint PXG_LOCK_LINEAR_Y  = 1u << 1;
public static const uint PXG_LOCK_LINEAR_Z  = 1u << 2;
public static const uint PXG_LOCK_ANGULAR_X = 1u << 3;
public static const uint PXG_LOCK_ANGULAR_Y = 1u << 4;
public static const uint PXG_LOCK_ANGULAR_Z = 1u << 5;

// DySleepingConfigulation.h
public static const float PXD_FREEZE_INTERVAL  = 1.5f;
public static const float PXD_FREEZE_TOLERANCE = 0.25f;
public static const float PXD_SLEEP_DAMPING    = 0.5f;
public static const float PXD_FREEZE_SCALE     = 0.1f;

// PxGeometryType (subset handled by the bounds kernel)
public static const uint PXG_GEOM_SPHERE     = 0;
public static const uint PXG_GEOM_PLANE      = 1;
public static const uint PXG_GEOM_CAPSULE    = 2;
public static const uint PXG_GEOM_BOX        = 3;
public static const uint PXG_GEOM_CONVEXCORE = 4;
public static const uint PXG_GEOM_CONVEXMESH = 5;

// PxShapeFlag
public static const uint PXG_SHAPE_SIMULATION  = 1u << 0;
public static const uint PXG_SHAPE_SCENE_QUERY = 1u << 1;
public static const uint PXG_SHAPE_TRIGGER     = 1u << 2;

// PxsTransformFlag
public static const uint PXG_TRANSFORM_CACHE_FROZEN = 1u << 0;

public static const uint  PXG_INVALID_NODE = 0xFFFFFFFFu;
public static const float PXG_MAX_F32      = 3.4028234663852885981170418348452e+38f;

// ---------------------------------------------------------------------------
// Quaternion helpers (mirrors PxQuat / PxAlignedQuat semantics)
// Quaternions are stored (x, y, z, w).
// ---------------------------------------------------------------------------
public float4 quatMul(float4 a, float4 b)
{
    return float4(
        a.w * b.x + a.x * b.w + a.y * b.z - a.z * b.y,
        a.w * b.y + a.y * b.w + a.z * b.x - a.x * b.z,
        a.w * b.z + a.z * b.w + a.x * b.y - a.y * b.x,
        a.w * b.w - a.x * b.x - a.y * b.y - a.z * b.z);
}

public float3 quatRotate(float4 q, float3 v)
{
    // mirrors PxQuat::rotate()
    const float vx = 2.0f * v.x;
    const float vy = 2.0f * v.y;
    const float vz = 2.0f * v.z;
    const float w2 = q.w * q.w - 0.5f;
    const float dot2 = (q.x * vx + q.y * vy + q.z * vz);
    return float3(
        (vx * w2 + (q.y * vz - q.z * vy) * q.w + q.x * dot2),
        (vy * w2 + (q.z * vx - q.x * vz) * q.w + q.y * dot2),
        (vz * w2 + (q.x * vy - q.y * vx) * q.w + q.z * dot2));
}

public float3 quatRotateInv(float4 q, float3 v)
{
    // mirrors PxQuat::rotateInv()
    const float vx = 2.0f * v.x;
    const float vy = 2.0f * v.y;
    const float vz = 2.0f * v.z;
    const float w2 = q.w * q.w - 0.5f;
    const float dot2 = (q.x * vx + q.y * vy + q.z * vz);
    return float3(
        (vx * w2 - (q.y * vz - q.z * vy) * q.w + q.x * dot2),
        (vy * w2 - (q.z * vx - q.x * vz) * q.w + q.y * dot2),
        (vz * w2 - (q.x * vy - q.y * vx) * q.w + q.z * dot2));
}

public float4 quatNormalize(float4 q)
{
    const float s = 1.0f / sqrt(dot(q, q));
    return q * s;
}

public float4 quatConjugate(float4 q)
{
    return float4(-q.x, -q.y, -q.z, q.w);
}

public float3 quatBasisVector0(float4 q)
{
    // mirrors PxQuat::getBasisVector0()
    const float x2 = q.x * 2.0f;
    const float w2 = q.w * 2.0f;
    return float3((q.w * w2) - 1.0f + q.x * x2, (q.z * w2) + q.y * x2, (-q.y * w2) + q.z * x2);
}

// Column-major 3x3 rotation matrix built from a quaternion.
// col0/col1/col2 mirror PxMat33(PxQuat) columns.
public struct PxgMat33
{
    public float3 col0;
    public float3 col1;
    public float3 col2;
}

public PxgMat33 mat33FromQuat(float4 q)
{
    const float x = q.x, y = q.y, z = q.z, w = q.w;
    const float x2 = x + x, y2 = y + y, z2 = z + z;
    const float xx = x2 * x, yy = y2 * y, zz = z2 * z;
    const float xy = x2 * y, xz = x2 * z, xw = x2 * w;
    const float yz = y2 * z, yw = y2 * w, zw = z2 * w;

    PxgMat33 m;
    m.col0 = float3(1.0f - yy - zz, xy + zw, xz - yw);
    m.col1 = float3(xy - zw, 1.0f - xx - zz, yz + xw);
    m.col2 = float3(xz + yw, yz - xw, 1.0f - xx - yy);
    return m;
}

public float3 mat33Transform(PxgMat33 m, float3 v)
{
    return m.col0 * v.x + m.col1 * v.y + m.col2 * v.z;
}

public PxgMat33 mat33MulMat33(PxgMat33 a, PxgMat33 b)
{
    PxgMat33 r;
    r.col0 = mat33Transform(a, b.col0);
    r.col1 = mat33Transform(a, b.col1);
    r.col2 = mat33Transform(a, b.col2);
    return r;
}

// M(row, col) accessor equivalent used by transformInertiaTensor below.
float mat33At(PxgMat33 m, uint row, uint col)
{
    const float3 c = col == 0 ? m.col0 : (col == 1 ? m.col1 : m.col2);
    return row == 0 ? c.x : (row == 1 ? c.y : c.z);
}

// Symmetric 3x3 stored as 9 floats, column major (matches PxMat33 memory).
public void mat33Store(PxgMat33 m, out float outM[9])
{
    outM[0] = m.col0.x; outM[1] = m.col0.y; outM[2] = m.col0.z;
    outM[3] = m.col1.x; outM[4] = m.col1.y; outM[5] = m.col1.z;
    outM[6] = m.col2.x; outM[7] = m.col2.y; outM[8] = m.col2.z;
}

public PxgMat33 mat33Load(float inM[9])
{
    PxgMat33 m;
    m.col0 = float3(inM[0], inM[1], inM[2]);
    m.col1 = float3(inM[3], inM[4], inM[5]);
    m.col2 = float3(inM[6], inM[7], inM[8]);
    return m;
}

// mirrors transformInertiaTensor() in preIntegration.cuh:
// computes M * diag(invD) * M^T
public PxgMat33 transformInertiaTensor(float3 invD, PxgMat33 M)
{
    const float axx = invD.x * mat33At(M, 0, 0), axy = invD.x * mat33At(M, 1, 0), axz = invD.x * mat33At(M, 2, 0);
    const float byx = invD.y * mat33At(M, 0, 1), byy = invD.y * mat33At(M, 1, 1), byz = invD.y * mat33At(M, 2, 1);
    const float czx = invD.z * mat33At(M, 0, 2), czy = invD.z * mat33At(M, 1, 2), czz = invD.z * mat33At(M, 2, 2);

    const float m00 = axx * mat33At(M, 0, 0) + byx * mat33At(M, 0, 1) + czx * mat33At(M, 0, 2);
    const float m11 = axy * mat33At(M, 1, 0) + byy * mat33At(M, 1, 1) + czy * mat33At(M, 1, 2);
    const float m22 = axz * mat33At(M, 2, 0) + byz * mat33At(M, 2, 1) + czz * mat33At(M, 2, 2);

    const float m01 = axx * mat33At(M, 1, 0) + byx * mat33At(M, 1, 1) + czx * mat33At(M, 1, 2);
    const float m02 = axx * mat33At(M, 2, 0) + byx * mat33At(M, 2, 1) + czx * mat33At(M, 2, 2);
    const float m12 = axy * mat33At(M, 2, 0) + byy * mat33At(M, 2, 1) + czy * mat33At(M, 2, 2);

    PxgMat33 r;
    r.col0 = float3(m00, m01, m02);
    r.col1 = float3(m01, m11, m12);
    r.col2 = float3(m02, m12, m22);
    return r;
}

// ---------------------------------------------------------------------------
// Transforms
// ---------------------------------------------------------------------------

// PxTransform: 28 bytes, tightly packed { PxQuat q; PxVec3 p; }
// NOTE: field order in PxTransform is (q, p) with q first in PhysX 5.
// Stored as scalars to guarantee std430 packing identical to C++.
public struct PxgTransform28
{
    public float qx, qy, qz, qw;
    public float px, py, pz;

    public float4 q() { return float4(qx, qy, qz, qw); }
    public float3 p() { return float3(px, py, pz); }
}

public PxgTransform28 makeTransform28(float4 q, float3 p)
{
    PxgTransform28 t;
    t.qx = q.x; t.qy = q.y; t.qz = q.z; t.qw = q.w;
    t.px = p.x; t.py = p.y; t.pz = p.z;
    return t;
}

// this = this * other (mirrors PxTransform::transform(PxTransform); note the
// PhysX original does not renormalize the quaternion)
public PxgTransform28 transformTransform(PxgTransform28 a, PxgTransform28 b)
{
    const float4 q = quatMul(a.q(), b.q());
    const float3 p = a.p() + quatRotate(a.q(), b.p());
    return makeTransform28(q, p);
}

// mirrors PxTransform::transformInv(PxTransform) (PxTransform.h:162-169) LITERALLY:
//     qinv = q.getConjugate(); (qinv.rotate(src.p - p), qinv * src.q)
// P9.94a: NOT q.rotateInv(src.p - p), which this helper spelled until P9.94a.  The two
// give the same bits for every non-zero component, but an exact-zero component can
// take the other sign (the conjugate's cross term (-y)*vz - (-z)*vy is +0 where
// rotateInv's -(y*vz - z*vy) is -0).  All five callers mirror this very CUDA call.
public PxgTransform28 transformInvTransform(PxgTransform28 a, PxgTransform28 b)
{
    const float4 qinv = quatConjugate(a.q());
    const float4 q = quatMul(qinv, b.q());
    const float3 p = quatRotate(qinv, b.p() - a.p());
    return makeTransform28(q, p);
}

// PxAlignedTransform: 32 bytes { PxAlignedQuat q (float4); float4 p; }
public struct PxgAlignedTransform
{
    public float4 q;
    public float4 p;
}

// ---------------------------------------------------------------------------
// GPU data structures (byte-identical mirrors of the C++/CUDA layouts)
// ---------------------------------------------------------------------------

// mirrors PxgBodySim (gpusimulationcontroller/include/PxgBodySim.h), 224 bytes
public struct PxgBodySim
{
    public float4 linearVelocityXYZ_inverseMassW;                                   // 0
    public float4 angularVelocityXYZ_maxPenBiasW;                                   // 16
    public float4 maxLinearVelocitySqX_maxAngularVelocitySqY_linearDampingZ_angularDampingW; // 32
    public float4 inverseInertiaXYZ_contactReportThresholdW;                        // 48
    public float4 sleepLinVelAccXYZ_freezeCountW;                                   // 64
    public float4 sleepAngVelAccXYZ_accelScaleW;                                    // 80
    public float4 freezeThresholdX_wakeCounterY_sleepThresholdZ_bodySimIndex;       // 96
    public PxgAlignedTransform body2World;                                          // 112
    public PxgAlignedTransform body2Actor_maxImpulseW;                              // 144
    public uint articulationRemapId;                                                // 176
    public uint internalFlags;                                                      // 180
    public uint lockFlags_disableGravity;   // packed PxU16 lockFlags | PxU16 disableGravity << 16
    public float offsetSlop;                                                        // 188
    public float4 externalLinearAcceleration;                                       // 192
    public float4 externalAngularAcceleration;                                      // 208
}                                                                                   // = 224

// mirrors PxgSolverBodyData (gpusolver/include/PxgSolverBody.h), 96 bytes
public struct PxgSolverBodyData
{
    public float4 initialAngVelXYZ_penBiasClamp;    // 0
    public float4 initialLinVelXYZ_invMassW;        // 16
    public PxgAlignedTransform body2World;          // 32
    public uint2 islandNodeIndex;                   // 64 (PxNodeIndex, PxU64: low = mID, high = mLinkID)
    public float reportThreshold;                   // 72
    public float maxImpulse;                        // 76
    public uint flags;                              // 80
    public float offsetSlop;                        // 84
    public uint2 pad;                               // 88 (struct is 16-byte aligned in C++)
}                                                   // = 96

// mirrors PxgSolverBodySleepData
public struct PxgSolverBodySleepData
{
    public float wakeCounter;
    public uint internalFlags;
}

// mirrors PxgSolverTxIData: { PxTransform deltaBody2World; PxMat33 sqrtInvInertia; } = 64 bytes
public struct PxgSolverTxIData
{
    public PxgTransform28 deltaBody2World; // 0..28
    public float sqrtInvInertia[9];        // 28..64, column major
}

// mirrors PxgShapeSim (gpusimulationcontroller/include/PxgShapeSim.h), 72 bytes.
// NOTE: only float4/uint2/scalar members are used in buffer-visible structs so
// that the std430 layout matches the tightly-packed C++ layout exactly
// (float3 would be padded to 16 bytes under std430).
public struct PxgShapeSim
{
    public PxgTransform28 mTransform;   // 0
    public float lbMinX, lbMinY, lbMinZ; // 28 (mLocalBounds.minimum)
    public float lbMaxX, lbMaxY, lbMaxZ; // 40 (mLocalBounds.maximum)
    public uint pad0;                   // 52
    public uint2 mBodySimIndex;         // 56 (PxNodeIndex)
    public uint mHullDataIndex;         // 64
    public uint mShapeFlags_mShapeType; // 68: PxU16 flags | PxU16 type << 16

    public float3 localBoundsMin() { return float3(lbMinX, lbMinY, lbMinZ); }
    public float3 localBoundsMax() { return float3(lbMaxX, lbMaxY, lbMaxZ); }
}                                       // = 72

// mirrors PxgShape (gpunarrowphase/include/PxgConvexConvexShape.h), 48 bytes.
// hullOrMeshPtr held a raw device pointer under CUDA; in the Slang port it
// is a byte offset into the bound geometry blob buffer.
public struct PxgShape
{
    public float scaleX, scaleY, scaleZ; // 0  (PxMeshScale::scale)
    public float scaleRotX, scaleRotY, scaleRotZ, scaleRotW; // 12 (PxMeshScale::rotation)
    public uint materialIndex;           // 28
    public uint2 hullOrMeshOffset;       // 32 (was size_t hullOrMeshPtr)
    public uint type;                    // 40
    public uint particleOrSoftbodyId;    // 44

    public float3 scale() { return float3(scaleX, scaleY, scaleZ); }
    public float4 scaleRotation() { return float4(scaleRotX, scaleRotY, scaleRotZ, scaleRotW); }
}                                        // = 48

// mirrors PxsCachedTransform, 32 bytes
public struct PxsCachedTransform
{
    public PxgTransform28 transform; // 0..28
    public uint flags;               // 28..32
}

// PxNodeIndex helpers (PxU64 union: low word mID, high word mLinkID)
public uint nodeIndexIndex(uint2 n)             { return n.x; }
public bool nodeIndexIsStaticBody(uint2 n)      { return n.x == PXG_INVALID_NODE; }
public bool nodeIndexIsArticulation(uint2 n)    { return (n.y & 1u) != 0u; }
public uint nodeIndexArticulationLinkId(uint2 n){ return n.y >> 1; }

// fsel(a, b, c): b if a >= 0, else c (mirrors physx::intrinsics::fsel)
public float fsel(float a, float b, float c)
{
    return a >= 0.0f ? b : c;
}
