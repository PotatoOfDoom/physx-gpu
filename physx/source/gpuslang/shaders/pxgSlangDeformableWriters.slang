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

// pxgSlangDeformableWriters -- the four PhysX 5.12 deformable / particle contact
// writers, transcribed from gpusimulationcontroller/include/PxgDeformableContactWriters.h
// (PxgDbDbContactWriter, PxgRigidDbContactWriter, PxgParticleDbContactWriter) and
// PxgParticleSystem.h (PxgRigidParticleContactWriter).
//
// The CUDA kernels take each writer BY VALUE; the port inlines its fields into the
// kernel's push block (w_<field>, at the offset the bridge's paramAlign places the
// aggregate) and builds one of the structs below from them, so the write sequence is
// written once here.  The structs are NOT layout mirrors -- the push block is -- and
// carry no padding.
//
// Byte layout of the CUDA writers (offsetof, 5.12; pinned in testDeformableRef512):
//   PxgDbDbContactWriter          56 B: outPoint@0 outNormalPen@8 outBarycentric0@16
//                                 outBarycentric1@24 outContactInfo@32
//                                 totalContactCount@40 maxContacts@48
//   PxgRigidDbContactWriter       80 B: outPoint@0 outNormalPen@8 outBarycentric@16
//                                 outContactInfo@24 outRigidMaterialIndex@32
//                                 totalContactCount@40 contactByRigid@48
//                                 tempContactByRigid@56 contactIndexSortedByRigid@64
//                                 maxNumContacts@72
//   PxgParticleDbContactWriter    72 B: outPoint@0 outNormalPen@8 outBarycentric@16
//                                 outContactInfo@24 totalContactCount@32
//                                 contactByParticle@40 tempContactByParticle@48
//                                 contactIndexSortedByParticle@56 maxNumContacts@64
//   PxgRigidParticleContactWriter 80 B: particleContactInfos@0
//                                 particleContactNormalPens@8 numTotalContacts@16
//                                 contactSortedByParticle@24 tempContactByParticle@32
//                                 contactIndexSortedByParticle@40 contactByRigid@48
//                                 tempContactByRigid@56 contactIndexSortedByRigid@64
//                                 maxContacts@72
module pxgSlangDeformableWriters;

import pxgSlangDeformableRef;

// PxgDbDbContactWriter: the volume-volume and volume-surface streams.  ref1 is always
// the soft body tetrahedron; nothing is sorted.
public struct PxgDbDbContactWriterGpu
{
    public uint64_t outPoint;           // float4*
    public uint64_t outNormalPen;       // float4*
    public uint64_t outBarycentric0;    // float4*
    public uint64_t outBarycentric1;    // float4*
    public uint64_t outContactInfo;     // PxgDeformablePairInfo*
    public uint64_t totalContactCount;  // PxU32*
    public uint maxContacts;

    public bool writeBaseContact(uint index, float4 contact, float4 normalPen, uint64_t ref0, uint64_t ref1)
    {
        if (index >= maxContacts)
            return false;

        Ptr<float4>(outPoint)[index] = contact;
        Ptr<float4>(outNormalPen)[index] = normalPen;
        PxgStorePairInfo(outContactInfo, index, ref0, ref1);
        return true;
    }

    // Both barycentrics are written later by sb_fem_contact_remap_to_simLaunch.
    public bool writeVolumeVolumeContact(uint index, float4 contact, float4 normalPen,
                                         uint64_t vertexRef0, uint64_t tetRef1)
    {
        return writeBaseContact(index, contact, normalPen, vertexRef0, tetRef1);
    }

    // The surface's barycentric is final; the volume's is written by the remap.
    public bool writeSurfaceVolumeContact(uint index, float4 contact, float4 normalPen,
                                          uint64_t surfaceRef, float4 barycentric, uint64_t tetRef)
    {
        if (!writeBaseContact(index, contact, normalPen, surfaceRef, tetRef))
            return false;
        Ptr<float4>(outBarycentric0)[index] = barycentric;
        return true;
    }

    // Vertex overload: the barycentric slot is left untouched.
    public bool writeSurfaceVolumeContactVertex(uint index, float4 contact, float4 normalPen,
                                                uint64_t vertexRef, uint64_t tetRef)
    {
        return writeBaseContact(index, contact, normalPen, vertexRef, tetRef);
    }
}

// PxgRigidDbContactWriter: rigid-vs-volume and rigid-vs-surface.  ref0 is the rigid
// (PxNodeIndex) and the sort key; the rigid material index goes to its own PxU16
// stream.
public struct PxgRigidDbContactWriterGpu
{
    public uint64_t outPoint;                   // float4*
    public uint64_t outNormalPen;               // float4*
    public uint64_t outBarycentric;             // float4*
    public uint64_t outContactInfo;             // PxgDeformablePairInfo*
    public uint64_t outRigidMaterialIndex;      // PxU16*
    public uint64_t totalContactCount;          // PxU32*
    public uint64_t contactByRigid;             // PxU64*
    public uint64_t tempContactByRigid;         // PxU32*
    public uint64_t contactIndexSortedByRigid;  // PxU32*
    public uint maxNumContacts;

    public void writeSortKey(uint index, uint64_t ref0)
    {
        Ptr<uint64_t>(contactByRigid)[index] = ref0;
        Ptr<uint>(tempContactByRigid)[index] = PxgGetContactSortKey(ref0);
        Ptr<uint>(contactIndexSortedByRigid)[index] = index;
    }

    public bool writeBaseContact(uint index, float4 contact, float4 normalPen, uint64_t rigidRef,
                                 uint rigidMaterialIndex, uint64_t deformableRef)
    {
        if (index >= maxNumContacts)
            return false;

        Ptr<float4>(outPoint)[index] = contact;
        Ptr<float4>(outNormalPen)[index] = normalPen;
        PxgStorePairInfo(outContactInfo, index, rigidRef, deformableRef);
        // PxU16(rigidMaterialIndex): the narrowing is the store's.
        PxgStoreU16(outRigidMaterialIndex, index, rigidMaterialIndex & 0xffffu);
        return true;
    }

    // Soft body contacts are on the collision mesh: no barycentric here,
    // sb_other_contact_remap_to_simLaunch computes it.
    public bool writeVolumeContact(uint index, float4 contact, float4 normalPen, uint64_t rigidRef,
                                   uint rigidMaterialIndex, uint64_t deformableRef)
    {
        if (!writeBaseContact(index, contact, normalPen, rigidRef, rigidMaterialIndex, deformableRef))
            return false;
        writeSortKey(index, rigidRef);
        return true;
    }

    public bool writeSurfaceContact(uint index, float4 contact, float4 normalPen, uint64_t rigidRef,
                                    uint rigidMaterialIndex, uint64_t deformableRef, float4 barycentric)
    {
        if (!writeBaseContact(index, contact, normalPen, rigidRef, rigidMaterialIndex, deformableRef))
            return false;
        writeSortKey(index, rigidRef);
        Ptr<float4>(outBarycentric)[index] = barycentric;
        return true;
    }

    // Vertex overload: nothing is written to the barycentric slot.
    public bool writeSurfaceContactVertex(uint index, float4 contact, float4 normalPen, uint64_t rigidRef,
                                          uint rigidMaterialIndex, uint64_t vertexRef)
    {
        if (!writeBaseContact(index, contact, normalPen, rigidRef, rigidMaterialIndex, vertexRef))
            return false;
        writeSortKey(index, rigidRef);
        return true;
    }
}

// PxgParticleDbContactWriter: particle-vs-volume and particle-vs-surface.  ref0 is the
// particle and the sort key.
public struct PxgParticleDbContactWriterGpu
{
    public uint64_t outPoint;                       // float4*
    public uint64_t outNormalPen;                   // float4*
    public uint64_t outBarycentric;                 // float4*
    public uint64_t outContactInfo;                 // PxgDeformablePairInfo*
    public uint64_t totalContactCount;              // PxU32*
    public uint64_t contactByParticle;              // PxU64*
    public uint64_t tempContactByParticle;          // PxU32*
    public uint64_t contactIndexSortedByParticle;   // PxU32*
    public uint maxNumContacts;

    public void writeSortKey(uint index, uint64_t ref0)
    {
        Ptr<uint64_t>(contactByParticle)[index] = ref0;
        Ptr<uint>(tempContactByParticle)[index] = PxgGetContactSortKey(ref0);
        Ptr<uint>(contactIndexSortedByParticle)[index] = index;
    }

    public bool writeBaseContact(uint index, float4 contact, float4 normalPen, uint64_t ref0, uint64_t ref1)
    {
        if (index >= maxNumContacts)
            return false;

        Ptr<float4>(outPoint)[index] = contact;
        Ptr<float4>(outNormalPen)[index] = normalPen;
        PxgStorePairInfo(outContactInfo, index, ref0, ref1);
        return true;
    }

    public bool writeVolumeContact(uint index, float4 contact, float4 normalPen, uint64_t ref0, uint64_t ref1)
    {
        if (!writeBaseContact(index, contact, normalPen, ref0, ref1))
            return false;
        writeSortKey(index, ref0);
        return true;
    }

    public bool writeSurfaceContact(uint index, float4 contact, float4 normalPen, uint64_t ref0, uint64_t ref1,
                                    float4 barycentric)
    {
        if (!writeBaseContact(index, contact, normalPen, ref0, ref1))
            return false;
        writeSortKey(index, ref0);
        Ptr<float4>(outBarycentric)[index] = barycentric;
        return true;
    }
}

// PxgRigidParticleContactWriter (PxgParticleSystem.h): the SoA particle-rigid stream,
// sorted twice (by particle and by rigid).
public struct PxgRigidParticleContactWriterGpu
{
    public uint64_t particleContactInfos;           // PxgDeformablePairInfo*
    public uint64_t particleContactNormalPens;      // float4*
    public uint64_t numTotalContacts;               // PxU32*
    public uint64_t contactSortedByParticle;        // PxU64*
    public uint64_t tempContactByParticle;          // PxU32*
    public uint64_t contactIndexSortedByParticle;   // PxU32*
    public uint64_t contactByRigid;                 // PxNodeIndex*
    public uint64_t tempContactByRigid;             // PxU32*
    public uint64_t contactIndexSortedByRigid;      // PxU32*
    public uint maxContacts;

    // writeContact(index, normalPen, rigidId, particleRef), in the CUDA's store order.
    public void writeContact(uint index, float4 normalPen, uint64_t rigidId, uint64_t particleRef)
    {
        if (index < maxContacts)
        {
            Ptr<uint64_t>(contactByRigid)[index] = rigidId;
            Ptr<uint>(tempContactByRigid)[index] = PxgGetContactSortKey(PxgEncodeRigidNode(rigidId));
            Ptr<uint>(contactIndexSortedByRigid)[index] = index;

            Ptr<uint64_t>(contactSortedByParticle)[index] = particleRef;
            Ptr<uint>(tempContactByParticle)[index] = PxgGetContactSortKey(particleRef);
            Ptr<uint>(contactIndexSortedByParticle)[index] = index;

            PxgStorePairInfo(particleContactInfos, index, PxgEncodeRigidNode(rigidId), particleRef);
            Ptr<float4>(particleContactNormalPens)[index] = normalPen;
        }
    }
}
