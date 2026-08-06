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

// pxgSlangCvxTriSetup.slang
//
// `convexTrimeshNarrowphaseCore`'s PROLOGUE (convexMesh.cu:99-210): everything
// that has to be true before `convexTriangleContactGen` can run on a pair.  It
// resolves the two shapes, composes the mesh-to-convex transform, locates the
// cooked hull and the cooked mesh, and produces the triangle in convex-local
// space together with its normal, its adjacency word and its CPU face index.
//
// WHAT MAKES THIS ITS OWN TRANCHE.  Everything here is address arithmetic over
// blobs whose layout is nowhere written down except in the reader that walks
// them, and a wrong offset produces plausible floats from the neighbouring
// section — the same failure mode that made the hull-blob reader (P8.39) and the
// BV32 reader (P8.25) their own tranches.  The cooked mesh in particular has TWO
// sections this port has never read before: the per-triangle adjacency words and
// the face-remap table, both of which sit past the BV32 tree, the vertices and
// the indices, so every earlier count has to be right for them to be right.
//
// THE COOKED MESH (readTriangleMesh, triangleMesh.cuh:120-150):
//
//   @0                                  uint4  (numVerts, numTris, _, nbPackedNodes)
//   @16                                 BV32DataPacked[nbPackedNodes]
//   +BV32_STRIDE*nbPackedNodes          float4 verts[numVerts]
//   +16*numVerts                        uint4  indices[numTris]
//   +16*numTris                         uint4  adjacencies[numTris]
//   +16*numTris                         uint32 faceRemap[numTris]
//
// The `.z` word of the header is NOT a node count — it is `nbAdjVertsTotal` in
// one spelling of the reader and `numBv32TreeNodes` in another, and NEITHER sizes
// the node array.  `.w` does.  Getting that wrong shifts every section after the
// tree, which is why it is called out here as well as in the midphase shader.
//
// THE SCALE FLIP.  A triangle mesh with a NEGATIVE scale determinant has its
// winding reversed, and the CUDA compensates by writing vertices 1 and 2 to
// swapped slots (`writeIndex = threadIdx.x ^ 3` for lanes 1 and 2, convexMesh.cu:186).
// `1 ^ 3 == 2` and `2 ^ 3 == 1`, so it is exactly a swap of the last two; lane 0
// is left alone because `threadIdx.x > 0` guards it.  Without this the triangle
// normal points into the mesh and every contact is generated against the far side.

// (no imports: the setup stage is pure address arithmetic over the two blobs)

static const uint CT_WARP = 32u;
// sizeof(Gu::BV32DataPacked) (GuBV32.h:88): mMin[32] float4 = 512, mMax[32] = 512,
// mData[32] uint = 128, mNbNodes + mDepth + padding[2] = 16.  1168, and every
// other mesh-walking shader in this tree already carries that number.
//
// THIS READ 160 UNTIL P8.49, AND THE FIXTURE READ 160 TOO.  They agreed with each
// other and disagreed with the engine, so every section past the tree — verts,
// indices, adjacencies, faceRemap — sat at the wrong address against real device
// memory, and the P8.46b sweep came back 40/40 clean anyway.  That is the failure
// mode of a fixture that DEFINES the layout it is meant to check: mutating the
// constant moves shader and oracle together.  The fixture now takes the stride
// from the engine header (`sizeof(physx::Gu::BV32DataPacked)`) with a
// static_assert, so the compiler is the oracle and this can only drift again if
// the engine struct itself changes.
static const uint CT_BV32_STRIDE = 1168u;

public static const uint CT_GEOM_TRIANGLEMESH = 4u;   // PxGeometryType::eTRIANGLEMESH

// PxgConvexConvexShape.h, 48 B — the same mirror the midphase shader carries.
public struct CtShape
{
    public float scaleX, scaleY, scaleZ;                     // @0
    public float scaleRotX, scaleRotY, scaleRotZ, scaleRotW; // @12
    public uint materialIndex;                               // @28
    public uint2 hullOrMeshOffset;                           // @32
    public uint type;                                        // @40
    public uint particleOrSoftbodyId;                        // @44
}

public struct CtContactManagerInput
{
    public uint shapeRef0, shapeRef1, transformCacheRef0, transformCacheRef1;
}

// ---------------------------------------------------------------------------
// quaternion / scale helpers (PxQuat, PxMeshScale)
// ---------------------------------------------------------------------------
public float3 ctQuatRotate(float4 q, float3 v)
{
    const float3 u = float3(q.x, q.y, q.z);
    const float w = q.w;
    return v * (w * w - dot(u, u)) + u * (dot(v, u) * 2.0f) + cross(u, v) * (w * 2.0f);
}

public float3 ctQuatRotateInv(float4 q, float3 v)
{
    const float3 u = float3(q.x, q.y, q.z);
    const float w = q.w;
    return v * (w * w - dot(u, u)) + u * (dot(v, u) * 2.0f) - cross(u, v) * (w * 2.0f);
}

public float4 ctQuatMul(float4 a, float4 b)
{
    return float4(a.w * b.x + b.w * a.x + a.y * b.z - a.z * b.y,
                  a.w * b.y + b.w * a.y + a.z * b.x - a.x * b.z,
                  a.w * b.z + b.w * a.z + a.x * b.y - a.y * b.x,
                  a.w * b.w - a.x * b.x - a.y * b.y - a.z * b.z);
}

public float4 ctQuatConjugate(float4 q) { return float4(-q.x, -q.y, -q.z, q.w); }

public float3 ctVertex2Shape(float3 v, float3 scale, float4 rot)
{
    return ctQuatRotateInv(rot, ctQuatRotate(rot, v) * scale);
}

// PxVec3::getNormalized() — the ZERO VECTOR for a zero-magnitude input, not NaN.
public float3 ctNormalized(float3 v)
{
    const float m = sqrt(dot(v, v));
    return m > 0.0f ? v / m : float3(0.0f, 0.0f, 0.0f);
}

// ---------------------------------------------------------------------------
// what the prologue produces
// ---------------------------------------------------------------------------
public struct CtSetup
{
    public uint64_t hull;             // the cooked convex hull blob
    public float3 convexScale;
    public float4 convexScaleRot;
    public float contactDist;
    public float3 triLocVerts0, triLocVerts1, triLocVerts2;
    public float3 triangleLocNormal;
    public uint3 triAdjTrisIdx;
    public uint remapCpuTriangleIdx;
    public uint convexTriPairOffset;
    public uint convexTriPairOffsetPadded;
}

// The whole prologue, warp-uniform: every lane computes the same values.  The
// CUDA spreads the work over lanes and syncs three times; here each value is
// cheap enough that computing it redundantly costs less than the barriers would,
// and it removes three uniformity preconditions from the caller.
//
// `curPair` is the midphase's pair record: .x = contact-manager index,
// .y = triangle index, .z = the pair's offset within that contact manager.
public CtSetup ctSetup(uint64_t cmInputs, uint64_t transformCache, uint64_t contactDistance,
                       uint64_t gpuShapes, uint64_t cvxTrimeshPair, uint4 curPair)
{
    CtSetup s;

    const uint cmIdx = curPair.x;
    const uint triangleIdx = curPair.y;
    const uint testOffset = curPair.z;

    const CtContactManagerInput cin = Ptr<CtContactManagerInput>(cmInputs)[cmIdx];
    uint transformCacheRef0 = cin.transformCacheRef0;
    uint transformCacheRef1 = cin.transformCacheRef1;
    uint shapeRef0 = cin.shapeRef0;
    uint shapeRef1 = cin.shapeRef1;

    Ptr<CtShape> shapes = Ptr<CtShape>(gpuShapes);

    // Slot 0 must be the convex and slot 1 the mesh.  The transform refs swap with
    // the shape refs, and the CUDA swaps them BEFORE reading the transforms.
    if (shapes[shapeRef0].type == CT_GEOM_TRIANGLEMESH)
    {
        const uint tc = transformCacheRef0; transformCacheRef0 = transformCacheRef1; transformCacheRef1 = tc;
        const uint sr = shapeRef0;          shapeRef0 = shapeRef1;                   shapeRef1 = sr;
    }

    const CtShape convexShape = shapes[shapeRef0];
    const CtShape trimeshShape = shapes[shapeRef1];

    s.convexScale = float3(convexShape.scaleX, convexShape.scaleY, convexShape.scaleZ);
    s.convexScaleRot = float4(convexShape.scaleRotX, convexShape.scaleRotY,
                              convexShape.scaleRotZ, convexShape.scaleRotW);
    const float3 triScale = float3(trimeshShape.scaleX, trimeshShape.scaleY, trimeshShape.scaleZ);
    const float4 triScaleRot = float4(trimeshShape.scaleRotX, trimeshShape.scaleRotY,
                                      trimeshShape.scaleRotZ, trimeshShape.scaleRotW);

    // PxsCachedTransform, 32 B: PxQuat q @0, PxVec3 p @16.
    const uint64_t xf0 = transformCache + uint64_t(transformCacheRef0) * 32UL;
    const uint64_t xf1 = transformCache + uint64_t(transformCacheRef1) * 32UL;
    const float4 q0 = Ptr<float4>(xf0)[0];
    const float3 p0 = float3(Ptr<float>(xf0 + 16UL)[0], Ptr<float>(xf0 + 20UL)[0], Ptr<float>(xf0 + 24UL)[0]);
    const float4 q1 = Ptr<float4>(xf1)[0];
    const float3 p1 = float3(Ptr<float>(xf1 + 16UL)[0], Ptr<float>(xf1 + 20UL)[0], Ptr<float>(xf1 + 24UL)[0]);

    // convexTransform.transformInv(trimeshTransform): q = conj(q0)*q1,
    // p = rotInv(q0, p1 - p0).  NOTE THE DIRECTION -- the midphase composes the
    // OTHER way round (mesh-relative), this one is convex-relative
    // (convexMesh.cu:141).
    const float4 toCvxQ = ctQuatMul(ctQuatConjugate(q0), q1);
    const float3 toCvxP = ctQuatRotateInv(q0, p1 - p0);

    s.contactDist = Ptr<float>(contactDistance)[transformCacheRef0]
                  + Ptr<float>(contactDistance)[transformCacheRef1];

    s.hull = uint64_t(convexShape.hullOrMeshOffset.x) | (uint64_t(convexShape.hullOrMeshOffset.y) << 32);

    // ---- the cooked mesh ---------------------------------------------------
    const uint64_t meshPtr = uint64_t(trimeshShape.hullOrMeshOffset.x)
                           | (uint64_t(trimeshShape.hullOrMeshOffset.y) << 32);
    const uint4 meshHeader = Ptr<uint4>(meshPtr)[0];
    const uint numVerts = meshHeader.x;
    const uint numTris = meshHeader.y;
    const uint nbPackedNodes = meshHeader.w;   // .z is NOT the node count

    const uint64_t vertsAddr = meshPtr + 16UL + uint64_t(nbPackedNodes) * uint64_t(CT_BV32_STRIDE);
    const uint64_t indicesAddr = vertsAddr + uint64_t(numVerts) * 16UL;
    const uint64_t adjAddr = indicesAddr + uint64_t(numTris) * 16UL;
    const uint64_t faceRemapAddr = adjAddr + uint64_t(numTris) * 16UL;

    const uint4 adj = Ptr<uint4>(adjAddr)[triangleIdx];
    s.triAdjTrisIdx = uint3(adj.x, adj.y, adj.z);
    s.remapCpuTriangleIdx = Ptr<uint>(faceRemapAddr)[triangleIdx];

    // ---- the triangle in convex-local space --------------------------------
    const uint4 triVertIdx = Ptr<uint4>(indicesAddr)[triangleIdx];
    const bool flipNormal = (triScale.x * triScale.y * triScale.z) < 0.0f;

    float3 loc[3];
    for (uint i = 0u; i < 3u; ++i)
    {
        const uint vi = (i == 0u) ? triVertIdx.x : ((i == 1u) ? triVertIdx.y : triVertIdx.z);
        const float3 v = Ptr<float4>(vertsAddr)[vi].xyz;
        const float3 scaled = ctVertex2Shape(v, triScale, triScaleRot);
        // trimeshToConvexTransform.transform(v)
        loc[i] = ctQuatRotate(toCvxQ, scaled) + toCvxP;
    }

    // A negative scale determinant reverses the winding; the CUDA compensates by
    // writing lanes 1 and 2 to swapped slots (`writeIndex = lane ^ 3`).  Lane 0 is
    // never moved.
    s.triLocVerts0 = loc[0];
    s.triLocVerts1 = flipNormal ? loc[2] : loc[1];
    s.triLocVerts2 = flipNormal ? loc[1] : loc[2];

    s.triangleLocNormal = ctNormalized(cross(s.triLocVerts1 - s.triLocVerts0,
                                             s.triLocVerts2 - s.triLocVerts0));

    // ---- where this pair's records go --------------------------------------
    // ConvexMeshPair (convexNpCommon.h:70-81), 56-byte stride:
    //
    //   @0  PxTransform aToB (PxQuat 16 + PxVec3 12 = 28)
    //   @28 int startIndex
    //   @32 int count
    //   @36 int cmIndex
    //   @40 int roundedStartIndex
    //   @44 int pad
    //   @48 uint2 materialIndices
    //
    // THE TRAILING NUMBERS IN THAT DECLARATION ARE RUNNING END OFFSETS, NOT FIELD
    // OFFSETS.  Reading them as field offsets puts startIndex at 32 and
    // roundedStartIndex at 36 — which is `count` and `cmIndex`.  Both are small
    // non-negative integers, so the pair records would land at plausible-looking
    // places in the output arrays and nothing would fault.
    const uint64_t pairAddr = cvxTrimeshPair + uint64_t(cmIdx) * 56UL;
    s.convexTriPairOffset = Ptr<uint>(pairAddr + 28UL)[0] + testOffset;
    s.convexTriPairOffsetPadded = Ptr<uint>(pairAddr + 40UL)[0] + testOffset;

    return s;
}
