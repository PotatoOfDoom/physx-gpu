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

// P9.57: this said 4, which is eCONVEXCORE.  PxGeometryType (PxGeometry.h:48-64)
// is eSPHERE 0, ePLANE 1, eCAPSULE 2, eBOX 3, eCONVEXCORE 4, eCONVEXMESH 5,
// ePARTICLESYSTEM 6, eTETRAHEDRONMESH 7, eTRIANGLEMESH 8, eHEIGHTFIELD 9 -- and
// eCONVEXCORE is a 5.5-era addition that shifted every member above it by one.
// With 4 the swap below NEVER fired for a genuinely mesh-first pair, so the
// MESH shape's blob would have been read as a convex hull; and it fired for an
// eCONVEXCORE shape in slot 0, which has no such blob either.  The fixture
// carried the same 4, which is why nothing saw it -- see the note six lines up
// about the BV32 stride, which is the identical failure mode.  The tenth static
// checker (tools/check_geom_type_constants.py) now reads this value against the
// engine header, for this constant and the fifty-odd others like it.
public static const uint CT_GEOM_TRIANGLEMESH = 8u;   // PxGeometryType::eTRIANGLEMESH

// P9.51b.  ctCommon takes the geometry type that, IN SLOT 0, means "the operands
// arrived the other way round -- swap them".  The trimesh path passes
// CT_GEOM_TRIANGLEMESH because convexMesh.cu:135 swaps on exactly that test.  The
// HEIGHTFIELD path passes CT_NO_SWAP, because convexHeightfield.cu does not swap
// at all: it reads slot 0 as the convex unconditionally (:211, :214).  A type no
// shape can carry says that in the call rather than in a comment.
public static const uint CT_NO_SWAP = 0xFFFFFFFFu;

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
    // P9.36: getNormalized is `v * PxRecipSqrt(magnitudeSquared())` (PxVec3.h:286)
    // with PxRecipSqrt == 1.0f/sqrtf (PxMath.h:161) -- ONE reciprocal, three
    // multiplies.  Never a per-component divide.
    const float m = sqrt(dot(v, v));
    return m > 0.0f ? v * (1.0f / m) : float3(0.0f, 0.0f, 0.0f);
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

// ---------------------------------------------------------------------------
// THE HALF BOTH MESH PATHS SHARE (P9.51b)
// ---------------------------------------------------------------------------
// Everything up to and including the pair offsets is identical for the cooked
// TRIANGLE MESH and for the HEIGHTFIELD: the same operand resolution, the same
// convex-relative transform, the same contact distance, the same hull pointer,
// the same ConvexMeshPair record.  What differs is only where the TRIANGLE comes
// from -- an index buffer, or arithmetic on a grid -- and whether slot 0 and slot
// 1 may arrive swapped.
//
// This was split out rather than copied.  Duplicating ~80 lines into the
// heightfield core would put two copies of the operand resolution in the tree,
// which is the exact shape P9.21 spent a tranche removing for find(); and the
// half that would have been copied includes the running-end-offset trap
// documented at the bottom of this function, which is precisely the kind of thing
// that gets fixed in one copy and not the other.
//
// WHAT THE SPLIT COSTS, MEASURED.  "The tests still pass" is not evidence that a
// refactor was value-preserving, so the compiled output was compared instead:
// both modules that call ctSetup grew by EXACTLY 1596 bytes / 83 instructions
// (convexTrimeshCoreBda 84072 -> 85668, cvxTriSetupBda 15792 -> 17388), and
// cvxTriPostProcessBda -- which imports this module but never calls ctSetup --
// is byte-identical.  The identical delta in two otherwise unrelated modules is
// the point: the whole change lives in this module's contribution.
//
// The delta is structural, and only structural.  Slang does NOT inline ctCommon;
// it emits it as its own function (+1 OpFunction/OpFunctionEnd/OpTypeFunction/
// OpLabel/OpReturnValue, +1 OpFunctionCall, +7 OpFunctionParameter) returning a
// struct through a local (+1 OpTypeStruct/OpTypePointer/OpVariable, +11 OpStore,
// +15 OpCompositeExtract, +12 OpAccessChain -- the fields written here and read
// back in the caller).  Every arithmetic opcode is unchanged: no OpFAdd, OpFMul,
// OpFSub, OpFDiv, OpDot or GLSL.std.450 call moved by one, and the constant pool
// is untouched down to the literal counts (35 -> 35 literals over 29 -> 29
// distinct values here; 78 -> 78 over 58 -> 58 in the core).  OpDecorate is
// likewise unchanged, so the NoContraction pass sees the same op set.  The extra
// stores are a struct round-trip through Function storage that any driver's
// mem2reg removes before FMA formation -- and the trimesh fixtures stay bit-exact
// against their CPU mirrors, which a contraction change would have broken first.
public struct CtCommon
{
    public uint64_t hull;             // the cooked convex hull blob
    public uint64_t meshPtr;          // the mesh / heightfield blob
    public float3 convexScale;
    public float4 convexScaleRot;
    public float3 meshScale;
    public float4 meshScaleRot;
    public float4 toCvxQ;             // mesh -> convex, per convexMesh.cu:141
    public float3 toCvxP;
    public float contactDist;
    public uint triangleIdx;
    public uint convexTriPairOffset;
    public uint convexTriPairOffsetPadded;
}

// Warp-uniform: every lane computes the same values.  The CUDA spreads the work
// over lanes and syncs three times; here each value is cheap enough that
// computing it redundantly costs less than the barriers would, and it removes
// three uniformity preconditions from the caller.
//
// `curPair` is the midphase's pair record: .x = contact-manager index,
// .y = triangle index, .z = the pair's offset within that contact manager.
//
// `swapWhenSlot0Type` is the geometry type that, in slot 0, means the operands
// arrived reversed.  Pass CT_NO_SWAP for a path whose .cu does not swap.
public CtCommon ctCommon(uint64_t cmInputs, uint64_t transformCache, uint64_t contactDistance,
                         uint64_t gpuShapes, uint64_t cvxTrimeshPair, uint4 curPair,
                         uint swapWhenSlot0Type)
{
    CtCommon s;

    const uint cmIdx = curPair.x;
    const uint triangleIdx = curPair.y;
    const uint testOffset = curPair.z;
    s.triangleIdx = triangleIdx;

    const CtContactManagerInput cin = Ptr<CtContactManagerInput>(cmInputs)[cmIdx];
    uint transformCacheRef0 = cin.transformCacheRef0;
    uint transformCacheRef1 = cin.transformCacheRef1;
    uint shapeRef0 = cin.shapeRef0;
    uint shapeRef1 = cin.shapeRef1;

    Ptr<CtShape> shapes = Ptr<CtShape>(gpuShapes);

    // Slot 0 must be the convex and slot 1 the mesh.  The transform refs swap with
    // the shape refs, and the CUDA swaps them BEFORE reading the transforms.
    // CT_NO_SWAP matches no shape, so a path that passes it never swaps.
    if (shapes[shapeRef0].type == swapWhenSlot0Type)
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

    s.toCvxQ = toCvxQ;
    s.toCvxP = toCvxP;
    s.meshScale = triScale;
    s.meshScaleRot = triScaleRot;

    s.contactDist = Ptr<float>(contactDistance)[transformCacheRef0]
                  + Ptr<float>(contactDistance)[transformCacheRef1];

    s.hull = uint64_t(convexShape.hullOrMeshOffset.x) | (uint64_t(convexShape.hullOrMeshOffset.y) << 32);

    s.meshPtr = uint64_t(trimeshShape.hullOrMeshOffset.x)
              | (uint64_t(trimeshShape.hullOrMeshOffset.y) << 32);
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
    // roundedStartIndex at 36 -- which is `count` and `cmIndex`.  Both are small
    // non-negative integers, so the pair records would land at plausible-looking
    // places in the output arrays and nothing would fault.
    const uint64_t pairAddr = cvxTrimeshPair + uint64_t(cmIdx) * 56UL;
    s.convexTriPairOffset = Ptr<uint>(pairAddr + 28UL)[0] + testOffset;
    s.convexTriPairOffsetPadded = Ptr<uint>(pairAddr + 40UL)[0] + testOffset;

    return s;
}

// ---------------------------------------------------------------------------
// THE COOKED-TRIANGLE-MESH PATH: the shared half, plus this path's triangle.
// ---------------------------------------------------------------------------
public CtSetup ctSetup(uint64_t cmInputs, uint64_t transformCache, uint64_t contactDistance,
                       uint64_t gpuShapes, uint64_t cvxTrimeshPair, uint4 curPair)
{
    const CtCommon c = ctCommon(cmInputs, transformCache, contactDistance, gpuShapes,
                                cvxTrimeshPair, curPair, CT_GEOM_TRIANGLEMESH);

    CtSetup s;
    s.hull = c.hull;
    s.convexScale = c.convexScale;
    s.convexScaleRot = c.convexScaleRot;
    s.contactDist = c.contactDist;
    s.convexTriPairOffset = c.convexTriPairOffset;
    s.convexTriPairOffsetPadded = c.convexTriPairOffsetPadded;

    const uint triangleIdx = c.triangleIdx;
    const uint64_t meshPtr = c.meshPtr;
    const float3 triScale = c.meshScale;
    const float4 triScaleRot = c.meshScaleRot;
    const float4 toCvxQ = c.toCvxQ;
    const float3 toCvxP = c.toCvxP;

    // ---- the cooked mesh ---------------------------------------------------
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

    return s;
}
