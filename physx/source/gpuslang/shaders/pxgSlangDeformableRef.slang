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

// pxgSlangDeformableRef -- the PhysX 5.12 participant encoding shared by every
// deformable and particle producer (narrowphase) and consumer (prep / solve).
//
// Transcribed from gpusimulationcontroller/include/PxgSimulationCoreDesc.h (5.12):
// PxgDeformableRef, PxgParticleRef, PxgEncode* / PxgGet* / PxgDeformablePairInfo
// accessors, PxgComparePairInfo, plus the byte layouts of the pair-carrying
// records (PxgDeformablePairInfo, PxgDeformableAttachment) and of the block-of-32
// constraint records of PxgDeformableConstraints.h and PxgParticleSystem.h.
//
//     [deformable:30][active:1][invalid:1][type:3][localEdge:2][featureId:27]
//        63..34         bit 33    bit 32   31..29    28..27         26..0
//     [systemId:30  ][active:1][invalid:1][        particleIndex:32         ]
//
// The two layouts keep the flags in the same bits, so the pair accessors read ref1
// without knowing which kind it holds.  ref0 may be a rigid PxNodeIndex, whose bit
// 32 is isArticulation(), so only ref1 is ever probed for the flags.
//
// Every value is a plain uint64_t.  The constants are built with uint64_t(...)
// casts rather than UL literals: the corpus has a measured case of Slang folding a
// UL-suffixed complement in 32 bits (softBodySolveBda.slang), and the flags live
// exactly at the 32-bit boundary where that would bite.  testDeformableRef512
// pins every function here against the real 5.12 header functions on the host.
module pxgSlangDeformableRef;

// ---------------------------------------------------------------------------
// PxgDeformableRef
// ---------------------------------------------------------------------------
// PxgDeformableRef::Enum
public static const uint PXG_DREF_eVERTEX = 0u;
public static const uint PXG_DREF_eTRIANGLE = 1u;
public static const uint PXG_DREF_eTETRAHEDRON = 2u;
public static const uint PXG_DREF_eTRIANGLE_EDGE = 3u;

public static const uint PXG_DREF_TYPE_BITS = 3u;
public static const uint PXG_DREF_LOCAL_EDGE_BITS = 2u;
public static const uint PXG_DREF_LOCAL_EDGE_MASK = 3u;
public static const uint PXG_DREF_FEATURE_ID_BITS = 27u;           // 32 - TYPE_BITS - LOCAL_EDGE_BITS
public static const uint PXG_DREF_FEATURE_ID_MASK = 0x07ffffffu;   // (1 << 27) - 1
public static const uint PXG_DREF_DEFORMABLE_SHIFT = 34u;

// INVALID_FLAG = 1 << 32, ACTIVE_FLAG = 1 << 33, as the two high-word bits they are.
public static const uint PXG_DREF_INVALID_FLAG_HI = 1u;
public static const uint PXG_DREF_ACTIVE_FLAG_HI = 2u;

// PX_MAX_NB_DEFORMABLE_SURFACE (0xfff) and PX_MAX_NB_DEFORMABLE_SURFACE_TRI (0xfffff).
public static const uint PXG_DREF_MAX_NB_DEFORMABLES = 0xfffu;
public static const uint PXG_DREF_MAX_NB_FEATURES = 0xfffffu;
// The element filter wildcard: a reserved feature id, not a flag.
public static const uint PXG_DREF_ANY_FEATURE = 0xfffffu;

public uint64_t PxgDeformableInvalidFlag() { return uint64_t(1) << 32; }
public uint64_t PxgDeformableActiveFlag() { return uint64_t(1) << 33; }

// PxgEncodeDeformableRef: both flags clear, local edge zero.
public uint64_t PxgEncodeDeformableRef(uint deformableId, uint type, uint featureId)
{
    return (uint64_t(deformableId) << PXG_DREF_DEFORMABLE_SHIFT)
         | uint64_t(type << (PXG_DREF_FEATURE_ID_BITS + PXG_DREF_LOCAL_EDGE_BITS))
         | uint64_t(featureId);
}

public uint64_t PxgEncodeInvalidDeformableRef(uint deformableId)
{
    return (uint64_t(deformableId) << PXG_DREF_DEFORMABLE_SHIFT) | PxgDeformableInvalidFlag();
}

public uint64_t PxgEncodeDeformableVertex(uint deformableId, uint vertexId)
{
    return PxgEncodeDeformableRef(deformableId, PXG_DREF_eVERTEX, vertexId);
}

public uint64_t PxgEncodeDeformableTriangle(uint deformableId, uint triangleId)
{
    return PxgEncodeDeformableRef(deformableId, PXG_DREF_eTRIANGLE, triangleId);
}

public uint64_t PxgEncodeDeformableTetrahedron(uint deformableId, uint tetrahedronId)
{
    return PxgEncodeDeformableRef(deformableId, PXG_DREF_eTETRAHEDRON, tetrahedronId);
}

public uint64_t PxgEncodeDeformableTriangleEdge(uint deformableId, uint triangleId, uint localEdge)
{
    return PxgEncodeDeformableRef(deformableId, PXG_DREF_eTRIANGLE_EDGE, triangleId)
         | uint64_t(localEdge << PXG_DREF_FEATURE_ID_BITS);
}

public uint PxgGetDeformableId(uint64_t ref) { return uint(ref >> PXG_DREF_DEFORMABLE_SHIFT); }

public uint PxgGetDeformableFeatureType(uint64_t ref)
{
    return uint(ref & uint64_t(0xffffffffu)) >> (PXG_DREF_FEATURE_ID_BITS + PXG_DREF_LOCAL_EDGE_BITS);
}

public uint PxgGetDeformableFeatureId(uint64_t ref)
{
    return uint(ref & uint64_t(0xffffffffu)) & PXG_DREF_FEATURE_ID_MASK;
}

public uint PxgGetDeformableEdgeTriangle(uint64_t ref) { return PxgGetDeformableFeatureId(ref); }

public uint PxgGetDeformableEdgeLocalIndex(uint64_t ref)
{
    return (uint(ref & uint64_t(0xffffffffu)) >> PXG_DREF_FEATURE_ID_BITS) & PXG_DREF_LOCAL_EDGE_MASK;
}

// ownsEdge (gpunarrowphase deformableCollision.cuh, 5.12): the three low bits of
// mTriangleVertexIndices_ownedEdges[tri].w say which of the triangle's edges it tests.
// Replaces 5.6.1's EdgeEncoding Type0/Type1 authorship predicates.
public bool ownsEdge(uint triangleEdgeOwnership, uint localEdgeIndex)
{
    return (triangleEdgeOwnership & (1u << localEdgeIndex)) != 0u;
}

// ---------------------------------------------------------------------------
// PxgParticleRef
// ---------------------------------------------------------------------------
public static const uint PXG_PREF_SYSTEM_SHIFT = 34u;

public uint64_t PxgEncodeParticleRef(uint particleSystemId, uint particleId)
{
    return (uint64_t(particleSystemId) << PXG_PREF_SYSTEM_SHIFT) | uint64_t(particleId);
}

public uint PxgGetParticleIndex(uint64_t ref) { return uint(ref & uint64_t(0xffffffffu)); }
public uint PxgGetParticleSystemId(uint64_t ref) { return uint(ref >> PXG_PREF_SYSTEM_SHIFT); }

// The first-stage radix key of a contact grouped by a rigid or a particle.
public uint PxgGetContactSortKey(uint64_t participantRef) { return uint(participantRef & uint64_t(0xffffffffu)); }

// ---------------------------------------------------------------------------
// Rigid participants (PxNodeIndex as its packed PxU64)
// ---------------------------------------------------------------------------
public uint64_t PxgEncodeRigidNode(uint64_t nodeIndexInd) { return nodeIndexInd; }

// PxNodeIndex (PxNodeIndex.h) accessors on the packed PxU64: mInd in the low word,
// mLinkID in the high word (bit 0 = isArticulation, linkId = mLinkID >> 1).
public bool PxgNodeIsStaticBody(uint64_t ind) { return uint(ind & uint64_t(0xffffffffu)) == 0xffffffffu; }
public uint PxgNodeIndexOf(uint64_t ind) { return uint(ind & uint64_t(0xffffffffu)); }
public bool PxgNodeIsArticulation(uint64_t ind) { return (uint(ind >> 32) & 1u) != 0u; }
public uint PxgNodeArticulationLinkId(uint64_t ind) { return uint(ind >> 32) >> 1u; }

// PX_INVALID_NODE in the low word, transform cache id + 1 in the high word.
public uint64_t PxgEncodeStaticShapeRef(uint transformCacheId)
{
    return uint64_t(0xffffffffu) | ((uint64_t(transformCacheId) + uint64_t(1)) << 32);
}

// PxNodeIndex::isStaticBody(): the low word (mInd) is PX_INVALID_NODE.
public uint64_t PxgEncodeRigidFilterRef(uint64_t nodeIndexInd, uint transformCacheId)
{
    const bool isStatic = uint(nodeIndexInd & uint64_t(0xffffffffu)) == 0xffffffffu;
    return isStatic ? PxgEncodeStaticShapeRef(transformCacheId) : PxgEncodeRigidNode(nodeIndexInd);
}

// ---------------------------------------------------------------------------
// PxgDeformablePairInfo {PxU64 ref0, ref1}, 16 bytes, align 16
// ---------------------------------------------------------------------------
public static const uint64_t PXG_PAIRINFO_STRIDE = 16;
public static const uint64_t PXG_PAIRINFO_ref0 = 0;
public static const uint64_t PXG_PAIRINFO_ref1 = 8;

public struct PxgDeformablePairInfoGpu
{
    public uint64_t ref0;
    public uint64_t ref1;
}

public PxgDeformablePairInfoGpu PxgMakePairInfo(uint64_t ref0, uint64_t ref1)
{
    PxgDeformablePairInfoGpu p;
    p.ref0 = ref0;
    p.ref1 = ref1;
    return p;
}

public PxgDeformablePairInfoGpu PxgLoadPairInfo(uint64_t base, uint index)
{
    return Ptr<PxgDeformablePairInfoGpu>(base)[index];
}

public void PxgStorePairInfo(uint64_t base, uint index, uint64_t ref0, uint64_t ref1)
{
    Ptr<PxgDeformablePairInfoGpu> p = Ptr<PxgDeformablePairInfoGpu>(base);
    p[index].ref0 = ref0;
    p[index].ref1 = ref1;
}

public bool PxgIsValidPairInfo(PxgDeformablePairInfoGpu pair)
{
    return (pair.ref1 & PxgDeformableInvalidFlag()) == uint64_t(0);
}

public PxgDeformablePairInfoGpu PxgInvalidatePairInfo(PxgDeformablePairInfoGpu pair)
{
    PxgDeformablePairInfoGpu p = pair;
    p.ref1 = p.ref1 | PxgDeformableInvalidFlag();
    return p;
}

public bool PxgIsActivePairInfo(PxgDeformablePairInfoGpu pair)
{
    return (pair.ref1 & PxgDeformableActiveFlag()) != uint64_t(0);
}

// Rewrites ref1 in full (PxgSetActivePairInfo); only the pair's owner may call it.
public uint64_t PxgSetActiveRef1(uint64_t ref1, bool active)
{
    return active ? (ref1 | PxgDeformableActiveFlag()) : (ref1 & ~PxgDeformableActiveFlag());
}

// Total order, ref0 then ref1, both unsigned 64-bit.
public int PxgComparePairInfo(PxgDeformablePairInfoGpu a, PxgDeformablePairInfoGpu b)
{
    if (a.ref0 < b.ref0)
        return -1;
    if (a.ref0 > b.ref0)
        return 1;
    if (a.ref1 < b.ref1)
        return -1;
    if (a.ref1 > b.ref1)
        return 1;
    return 0;
}

// ---------------------------------------------------------------------------
// PxgDeformableAttachment (PxgSimulationCoreDesc.h), 48 bytes, align 16
// ---------------------------------------------------------------------------
public static const uint64_t PXG_DATT_STRIDE = 48;
public static const uint64_t PXG_DATT_pair = 0;                    // PxgDeformablePairInfo
public static const uint64_t PXG_DATT_barycentricOrPosition0 = 16; // float4
public static const uint64_t PXG_DATT_barycentric1 = 32;           // float4

// ---------------------------------------------------------------------------
// Block-of-32 constraint records.  Offsets in bytes; the [offset] of a lane is
// added by the reader (float4 stride 16, PxReal / PxU32 stride 4, PxU64 stride 8).
// ---------------------------------------------------------------------------
// PxgRigidDbContactBlock (PxgDeformableConstraints.h), sizeof 2688
public static const uint64_t PXG_RDBC_raXn_resp = 0;
public static const uint64_t PXG_RDBC_raXnF0_resp = 512;
public static const uint64_t PXG_RDBC_raXnF1_resp = 1024;
public static const uint64_t PXG_RDBC_normal_errorW = 1536;
public static const uint64_t PXG_RDBC_fricTan0_invMass0 = 2048;
public static const uint64_t PXG_RDBC_maxPenBiasClamp = 2560;
public static const uint64_t PXG_RDBC_STRIDE = 2688;

// PxgRigidDbAttachmentBlock, sizeof 3200
public static const uint64_t PXG_RDBA_barycentric = 0;
public static const uint64_t PXG_RDBA_raXn0_biasW = 512;
public static const uint64_t PXG_RDBA_raXn1_biasW = 1024;
public static const uint64_t PXG_RDBA_raXn2_biasW = 1536;
public static const uint64_t PXG_RDBA_velMultiplierXYZ_invMassW = 2048;
public static const uint64_t PXG_RDBA_deformableRef = 2560;        // PxU64[32]
public static const uint64_t PXG_RDBA_rigidId = 2816;              // PxU64[32]
public static const uint64_t PXG_RDBA_rigidBodyRefCount = 3072;    // PxU32[32]
public static const uint64_t PXG_RDBA_STRIDE = 3200;

// PxgDbDbContactBlock, sizeof 640
public static const uint64_t PXG_DBDBC_normal_pen = 0;
public static const uint64_t PXG_DBDBC_maxPenBiasClamp = 512;
public static const uint64_t PXG_DBDBC_STRIDE = 640;

// PxgDbDbAttachmentBlock, sizeof 1536
public static const uint64_t PXG_DBDBA_barycentric0 = 0;
public static const uint64_t PXG_DBDBA_barycentric1 = 512;
public static const uint64_t PXG_DBDBA_deformableRef0 = 1024;      // PxU64[32]
public static const uint64_t PXG_DBDBA_deformableRef1 = 1280;      // PxU64[32]
public static const uint64_t PXG_DBDBA_STRIDE = 1536;

// PxgParticleDbContactBlock, sizeof 1152
public static const uint64_t PXG_PDBC_normal_pen = 0;
public static const uint64_t PXG_PDBC_barycentric = 512;
public static const uint64_t PXG_PDBC_maxPenBiasClamp = 1024;
public static const uint64_t PXG_PDBC_STRIDE = 1152;

// PxgRigidParticleContactBlock (PxgParticleSystem.h), sizeof 2688
public static const uint64_t PXG_RPC_raXn_resp = 0;
public static const uint64_t PXG_RPC_raXnF0_resp = 512;
public static const uint64_t PXG_RPC_raXnF1_resp = 1024;
public static const uint64_t PXG_RPC_normal_errorW = 1536;
public static const uint64_t PXG_RPC_fricTan0_invMass0 = 2048;
public static const uint64_t PXG_RPC_penBiasClampRigid = 2560;
public static const uint64_t PXG_RPC_STRIDE = 2688;

// ---------------------------------------------------------------------------
// The PxU16 streams (rigid material indices): this device has no 16-bit storage,
// so a PxU16 is read from and written through its containing 32-bit word.  The
// write is the atomic pair (gpuslang.subwordStore): two writers own the two
// halves of one word, and a plain read-modify-write would lose one of them.
// ---------------------------------------------------------------------------
public uint PxgLoadU16(uint64_t base, uint index)
{
    const uint64_t addr = base + uint64_t(index) * uint64_t(2);
    const uint word = Ptr<uint>(addr & ~uint64_t(3))[0];
    return ((addr & uint64_t(2)) != uint64_t(0)) ? (word >> 16u) : (word & 0xffffu);
}

public void PxgStoreU16(uint64_t base, uint index, uint value)
{
    const uint64_t addr = base + uint64_t(index) * uint64_t(2);
    Ptr<uint> w = Ptr<uint>(addr & ~uint64_t(3));
    const uint shift = ((addr & uint64_t(2)) != uint64_t(0)) ? 16u : 0u;
    uint prev;
    InterlockedAnd(w[0], ~(0xffffu << shift), prev);
    if ((value & 0xffffu) != 0u)
        InterlockedOr(w[0], (value & 0xffffu) << shift, prev);
}
