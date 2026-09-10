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

// pxgSlangCvxCoreMake.slang -- Gu::makeConvexShape (convexCoreCollision.cu:114-169),
// the PxgShape -> Gu::ConvexShape conversion every convexCore*Nphase kernel starts
// with.  P9.69b: one definition, included by cvxCorePlaneNpBda and cvxCoreCvxNpBda.
//
// Requires the includer to have imported pxgSlangNpAbi (PxgShapeGpu) and
// pxgSlangCvxCoreShape (ShapeRef, CORE_*).  An include and not a module for the
// reason measured in pxgSlangCvxCoreCvx's header.
//
// ===========================================================================
// THE CORE TYPE IS SMUGGLED THROUGH hullOrMeshPtr
// ===========================================================================
//
//     convex.coreType = Gu::ConvexCore::Type::Enum(shape.hullOrMeshPtr);
//     memcpy(convex.coreData, &shape.scale.scale.x, MAX_CORE_SIZE);
//     convex.margin = shape.scale.rotation.w;
//
// The 64-bit hull POINTER field carries a small ENUM, the core's parameters live in
// the PxMeshScale's scale bytes, and the margin in the scale rotation's w.  A port
// that reads hullOrMeshPtr as an address here dereferences a number between 0 and 5.
//
// ===========================================================================
// THE ePOINTS ARM, AND WHY IT NEEDS AN ADDRESS OUT-PARAMETER
// ===========================================================================
//
//     const float4* ptr = (const float4*)shape.hullOrMeshPtr;
//     core.points    = ptr + 3;                                    // +48 bytes
//     core.numPoints = (((const int4*)(ptr + 1))->x >> 8) & 0xff;  // word at +16
//     core.stride    = sizeof(float4);
//     core.S = shape.scale.scale;  core.R = shape.scale.rotation;  margin = 0;
//
// numPoints is up to 255.  Until P9.69b the shared shape layer took the points as a
// `float4 pool[12]` BY VALUE, so this arm could not be written at all -- which is
// why cvxCorePlaneNpBda documents its slot 1 as never being eCONVEXMESH rather than
// handling it.  The layer now reads them through Ptr<float4>, and the address comes
// back through `ptsAddr`.  core[0].x/y still carry the hull pointer because
// supportLocalQ uses that word pair as its 64-bit NULL guard (review 164).
bool makeConvexShapeCvxCore(PxgShapeGpu s, float4 poseQ, float3 poseP,
                            out ShapeRef sh, out float margin, out uint64_t ptsAddr)
{
    sh.type = 0u;
    sh.core[0] = uint4(0u, 0u, 0u, 0u);
    sh.core[1] = uint4(0u, 0u, 0u, 0u);
    sh.core[2] = uint4(0u, 0u, 0u, 0u);
    sh.poseQ = poseQ;
    sh.poseP = poseP;
    margin = 0.0f;
    ptsAddr = 0UL;

    if (s.type == EGT_CONVEXCORE)
    {
        sh.type = uint(s.hullOrMeshPtr);
        // MAX_CORE_SIZE IS SIX FLOATS (PxConvexCoreGeometry.h:264), so the memcpy
        // copies words 0..5 and NOTHING ELSE.  P9.67c wrote scaleRotW -- the MARGIN --
        // into word 6, which the CUDA never copies there and which the verified
        // npCvxCoreSetup transcription leaves at 0.  Word 6 is only read by the
        // CORE_POINTS layout, which eCONVEXCORE never takes, so the difference was
        // unobservable; it is still a difference, and P9.69b removes it.
        sh.core[0] = uint4(asuint(s.scaleX), asuint(s.scaleY), asuint(s.scaleZ),
                           asuint(s.scaleRotX));
        sh.core[1] = uint4(asuint(s.scaleRotY), asuint(s.scaleRotZ), 0u, 0u);
        margin = s.scaleRotW;
        return true;
    }
    if (s.type == EGT_SPHERE)
    {
        sh.type = CORE_POINT;
        margin = s.scaleX;
        return true;
    }
    if (s.type == EGT_CAPSULE)
    {
        sh.type = CORE_SEGMENT;
        sh.core[0].x = asuint(s.scaleX * 2.0f);   // SegmentCore::length
        margin = s.scaleY;
        return true;
    }
    if (s.type == EGT_BOX)
    {
        sh.type = CORE_BOX;
        sh.core[0] = uint4(asuint(s.scaleX * 2.0f), asuint(s.scaleY * 2.0f),
                           asuint(s.scaleZ * 2.0f), 0u);   // BoxCore::extents
        margin = 0.0f;
        return true;
    }
    if (s.type == EGT_CONVEXMESH)
    {
        sh.type = CORE_POINTS;
        // PointsCore: { const float4* points; PxVec3 S; PxQuat R; PxU8 numPoints,
        // stride; } -- words 0/1 the pointer, 2..4 S, 5..8 R, 9 the packed counts.
        const uint64_t hull = s.hullOrMeshPtr;
        ptsAddr = hull + 48UL;                     // ptr + 3 float4s
        sh.core[0].x = uint(hull & 0xFFFFFFFFUL);  // the NULL guard reads this pair
        sh.core[0].y = uint(hull >> 32);
        sh.core[0].z = asuint(s.scaleX);           // S
        sh.core[0].w = asuint(s.scaleY);
        sh.core[1].x = asuint(s.scaleZ);
        sh.core[1].y = asuint(s.scaleRotX);        // R
        sh.core[1].z = asuint(s.scaleRotY);
        sh.core[1].w = asuint(s.scaleRotZ);
        sh.core[2].x = asuint(s.scaleRotW);
        const uint nbVerts = (Ptr<uint>(hull + 16UL)[0] >> 8u) & 0xFFu;
        sh.core[2].y = nbVerts | (16u << 8u);      // numPoints | stride << 8
        margin = 0.0f;
        return true;
    }
    return false;
}
