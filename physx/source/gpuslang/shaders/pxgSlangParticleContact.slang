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

// pxgSlangParticleContact.slang -- P9.12g, the plumbing every particle-vs-shape
// midphase needs and that three of them had grown their own copy of.
//
// WHY THIS EXISTS NOW.  psMeshMidphaseBda's own comment (P9.10) said it:
//
//     "This is a THIRD copy in the corpus ... Extracting all three is its own
//      tranche; duplicating a fourth time would not be."
//
// ps_sdfMeshCollisonLaunch is the fourth caller, and it needs BOTH the contact
// writer and the rigid element filter, so it would have made copy four of the
// writer and copy two of the filter.  This is that tranche.
//
// WHAT MOVED, and from where:
//   LoadShapePair<A, B>                     psMeshMidphaseBda, psHfMidphaseBda
//   PxgContactManagerInput                  both
// (The 5.6.1 writer and rigid filter that also lived here are gone, see below.)
//
// WHAT DID NOT MOVE, deliberately: psContactGenBda and femClothContactGenBda
// carry their own filter variants over PRIVATE query structs, and neither is a
// module.  Folding those in is a change to two shaders this tranche has no
// fixture for; they are left alone and named here so the next person knows the
// extraction is still partial.
//
// THE ONE GENERALISATION.  LoadShapePair is a template on TWO geometry types in
// the CUDA, and the two callers do not agree on which one it tests: the mesh
// midphase tests the TRIANGLE MESH (particleSystemMeshMidphase.cu:403) while the
// heightfield midphase tests the PARTICLE SYSTEM
// (particleSystemHFMidPhaseCG.cu:293).  Rather than pick one and bury the
// difference, psLoadShapePair takes the tested type as an argument and returns
// the matching half first.  The choice therefore stays visible at each call
// site, which is where it is a trap.

import pxgSlangCommon;
import pxgSlangParticleDesc;

// PxgContactManagerInput, gpunarrowphase/include/PxgContactManager.h (16 B).
public struct PxgContactManagerInput
{
    public uint shapeRef0, shapeRef1, transformCacheRef0, transformCacheRef1;
}

// PhysX 5.12: the contact record and its writer moved to pxgSlangDeformableWriters
// (PxgRigidParticleContactWriter, SoA PxgDeformablePairInfo + float4 normalPen), and the
// particle-rigid element filter this module carried (find / findRange over
// PxParticleRigidFilterPair) was removed with particle rigids in 5.9.  The three
// importers build a PxgRigidParticleContactWriterGpu from their push blocks and key
// the particle with the 64-bit PxgEncodeParticleRef.

// ---------------------------------------------------------------------------
// LoadShapePair<A, B>.  `testedType` is A: the pair's slot 0 is checked against
// it, and whichever half matches comes back FIRST.  See the note at the head of
// this file about why that is a parameter and not a convention.
// ---------------------------------------------------------------------------
public void psLoadShapePair(uint64_t cmInputs, uint64_t gpuShapes, uint cmIdx, uint testedType,
                            out PxgShape matchShape, out uint matchCacheRef,
                            out PxgShape otherShape, out uint otherCacheRef)
{
    const PxgContactManagerInput cin = Ptr<PxgContactManagerInput>(cmInputs)[cmIdx];
    Ptr<PxgShape> shapes = Ptr<PxgShape>(gpuShapes);
    uint sRefA = cin.shapeRef0, cRefA = cin.transformCacheRef0;
    uint sRefB = cin.shapeRef1, cRefB = cin.transformCacheRef1;
    if (shapes[cin.shapeRef0].type != testedType)
    {
        sRefA = cin.shapeRef1; cRefA = cin.transformCacheRef1;
        sRefB = cin.shapeRef0; cRefB = cin.transformCacheRef0;
    }
    matchShape = shapes[sRefA]; matchCacheRef = cRefA;
    otherShape = shapes[sRefB]; otherCacheRef = cRefB;
}
