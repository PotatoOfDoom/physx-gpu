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

// pxgSlangNpAbi.slang
//
// P5 Step 4 (BDA engine ABI): byte-exact physical-storage mirrors of the
// element structs the rigid convex-convex narrowphase kernels index through
// RAW DEVICE POINTERS on the engine launch path.  These are Ptr<T> targets,
// so their physical layout (offsets AND stride) must equal what the engine
// writes; testNpAbiLayout.cpp proves every struct empirically (sentinel fill
// of the REAL engine structs, probe via npNpAbiProbe.slang, bitwise compare,
// element-[1] stride proof, mutation check).
//
//   PxgContactManagerInput   gpunarrowphase/include/PxgContactManager.h  (16 B)
//   PxgShape                 gpunarrowphase/include/PxgConvexConvexShape.h (48 B)
//   PxsCachedTransform       lowlevel/software/include/PxsTransformCache.h (32 B)
//   PxgPersistentContactManifold  gpunarrowphase/include/PxgPersistentContactManifold.h (256 B)
//   PxsContactManagerOutput  lowlevel/software/include/PxsContactManagerState.h (48 B)
//   PxsMaterialData          lowlevel/api/include/PxsMaterialCore.h (24 B)
//   PxgPatchAndContactCounters  gpunarrowphase/include/cudaNpCommon.h (16 B)
//   PxGpuContactPair         include/PxContact.h (80 B)
//
// Sub-word fields (PxU8/PxU16 packs) are mirrored as whole uint words and
// decoded/encoded with shifts, exactly as the CUDA reinterpret_casts do.

module pxgSlangNpAbi;

// mirrors PxgContactManagerInput, 16 B / stride 16.
public struct PxgContactManagerInputGpu
{
    public uint shapeRef0;          // 0
    public uint shapeRef1;          // 4
    public uint transformCacheRef0; // 8
    public uint transformCacheRef1; // 12
}

// mirrors PxgShape, 48 B / stride 48.  hullOrMeshPtr is the raw device
// pointer to the cooked convex-hull blob (dereferenced via Ptr<> here,
// unlike the bound-buffer ports which modelled it as a byte offset).
public struct PxgShapeGpu
{
    public float scaleX, scaleY, scaleZ;                     // 0..8   PxMeshScale::scale
    public float scaleRotX, scaleRotY, scaleRotZ, scaleRotW; // 12..24 PxMeshScale::rotation
    public uint materialIndex;                               // 28
    public uint64_t hullOrMeshPtr;                           // 32
    public uint type;                                        // 40
    public uint particleOrSoftbodyId;                        // 44
}

// mirrors PxsCachedTransform, 32 B / stride 32 (PxTransform = q then p).
public struct PxsCachedTransformGpu
{
    public float qx, qy, qz, qw; // 0..12
    public float px, py, pz;     // 16..24
    public uint flags;           // 28
}

// mirrors PxgPersistentContactManifold, 256 B / stride 256.  The CUDA
// kernels view the tail (offset 192) both as uint4[4] and PxU32[16]
// ("moreData"); word 12 is mNbContacts.
public struct PxgPersistentContactManifoldGpu
{
    public float4 localNormalPen[4]; // 0    mLocalNormal_PenW
    public float4 localContactA[4];  // 64   mLocalContactA
    public float4 localContactB[4];  // 128  mLocalContactB
    public float4 relativePos;       // 192  mRelativePos      (moreData words 0-3)
    public float4 quatA;             // 208  mQuatA            (words 4-7)
    public float4 quatB;             // 224  mQuatB            (words 8-11)
    public uint nbContacts;          // 240  mNbContacts       (word 12)
    public uint nbWarmStartPoints;   // 244                    (word 13)
    public uint warmStartA;          // 248                    (word 14)
    public uint warmStartB;          // 252                    (word 15)
}

// mirrors PxsContactManagerOutput, 48 B / stride 48.
//   allflagsWord  = allflagsStart | nbPatches<<8 | statusFlag<<16 | prevPatches<<24
//   nbContactsFlags = nbContacts | flags<<16
public struct PxsContactManagerOutputGpu
{
    public uint64_t contactPatches;  // 0
    public uint64_t contactPoints;   // 8
    public uint64_t contactForces;   // 16
    public uint64_t frictionPatches; // 24
    public uint allflagsWord;        // 32
    public uint nbContactsFlags;     // 36
    public uint pad0;                // 40
    public uint pad1;                // 44
}

// mirrors PxsMaterialData, 24 B / stride 24.
//   flagsModes0 = flags(u16) | fricCombineMode<<16 | restCombineMode<<24
//   modes1      = dampingCombineMode | (3 pad bytes)<<8
public struct PxsMaterialDataGpu
{
    public float dynamicFriction; // 0
    public float staticFriction;  // 4
    public float restitution;     // 8
    public float damping;         // 12
    public uint flagsModes0;      // 16
    public uint modes1;           // 20
}

// mirrors PxgPatchAndContactCounters, 16 B (atomically reserved stream cursors).
public struct PxgPatchAndContactCountersGpu
{
    public uint patchesBytes;        // 0
    public uint contactsBytes;       // 4
    public uint forceAndIndiceBytes; // 8
    public uint overflowError;       // 12
}

// mirrors PxGpuContactPair, 80 B / stride 80 (PxNodeIndex = one u64).
//   nbContactsPatches = nbContacts | nbPatches<<16 (both PxU16, written together)
public struct PxGpuContactPairGpu
{
    public uint64_t contactPatches;  // 0
    public uint64_t contactPoints;   // 8
    public uint64_t contactForces;   // 16
    public uint64_t frictionPatches; // 24
    public uint transformCacheRef0;  // 32
    public uint transformCacheRef1;  // 36
    public uint64_t nodeIndex0;      // 40
    public uint64_t nodeIndex1;      // 48
    public uint64_t actor0;          // 56
    public uint64_t actor1;          // 64
    public uint nbContactsPatches;   // 72
    public uint pad0;                // 76
}

// ---------------------------------------------------------------------------
// Shared quaternion/transform math (bit-identical to the verified bound-buffer
// ports; PxQuat::rotate / rotateInv term order, PxQuat operator*).
// ---------------------------------------------------------------------------
public float3 npAbiQuatRotate(float4 q, float3 v)
{
    const float vx = 2.0f * v.x, vy = 2.0f * v.y, vz = 2.0f * v.z;
    const float w2 = q.w * q.w - 0.5f;
    const float d2 = q.x * vx + q.y * vy + q.z * vz;
    return float3(vx * w2 + (q.y * vz - q.z * vy) * q.w + q.x * d2,
                  vy * w2 + (q.z * vx - q.x * vz) * q.w + q.y * d2,
                  vz * w2 + (q.x * vy - q.y * vx) * q.w + q.z * d2);
}

public float3 npAbiQuatRotateInv(float4 q, float3 v)
{
    const float vx = 2.0f * v.x, vy = 2.0f * v.y, vz = 2.0f * v.z;
    const float w2 = q.w * q.w - 0.5f;
    const float d2 = q.x * vx + q.y * vy + q.z * vz;
    return float3(vx * w2 - (q.y * vz - q.z * vy) * q.w + q.x * d2,
                  vy * w2 - (q.z * vx - q.x * vz) * q.w + q.y * d2,
                  vz * w2 - (q.x * vy - q.y * vx) * q.w + q.z * d2);
}

// PxQuat operator* (this = a, arg = b)
public float4 npAbiQuatMul(float4 a, float4 b)
{
    return float4(a.w * b.x + b.w * a.x + a.y * b.z - a.z * b.y,
                  a.w * b.y + b.w * a.y + a.z * b.x - a.x * b.z,
                  a.w * b.z + b.w * a.z + a.x * b.y - a.y * b.x,
                  a.w * b.w - a.x * b.x - a.y * b.y - a.z * b.z);
}

// calculatePCMConvexMargin (margin = min(minExtent * 0.25, tolerance * 0.05))
public float npAbiPcmConvexMargin(float3 extents, float toleranceLength)
{
    const float mn = min(min(extents.x, extents.y), extents.z);
    return min(mn * 0.25f, toleranceLength * 0.05f);
}

// PxTransform::isValid() = p.isFinite() && q.isFinite() && q.isUnit()
// (PxQuat::isUnit tolerance 1e-3).  -ffast-math is off for these kernels,
// so NaN comparisons behave IEEE.
public bool npAbiIsFinite(float f) { return !isnan(f) && !isinf(f); }

public bool npAbiTransformIsValid(float4 q, float3 p)
{
    const bool pFinite = npAbiIsFinite(p.x) && npAbiIsFinite(p.y) && npAbiIsFinite(p.z);
    const bool qFinite = npAbiIsFinite(q.x) && npAbiIsFinite(q.y) && npAbiIsFinite(q.z) && npAbiIsFinite(q.w);
    const float mag = sqrt(q.x * q.x + q.y * q.y + q.z * q.z + q.w * q.w);
    return pFinite && qFinite && (abs(mag - 1.0f) < 1e-3f);
}
