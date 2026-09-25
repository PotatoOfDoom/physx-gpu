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

// npInsertPatch.slang
//
// Slang port of insertIntoPatchStream (gpunarrowphase/src/CUDA/contactPatchUtils.cuh):
// combines the two shapes' materials (PxsCombineMaterials, tranche 43) and writes
// the resulting PxContactPatch header into the patch stream at the byte offset
// returned by registerContactPatch (skipping 0xffffffff). Completes the contact
// output machinery: after this, a sphere/box narrow-phase kernel can allocate,
// register and emit its contact patch end to end.
//
// PxContactPatch (PxContact.h, PX_ALIGN(16), 64 bytes) as uint words:
//   0..3   mMassModification.{linear0,angular0,linear1,angular1} (all 1.0)
//   4..6   normal.xyz              7  restitution
//   8      dynamicFriction         9  staticFriction   10 damping
//   11     startContactIndex(0:16) | nbContacts(16:24) | materialFlags(24:32)
//   12     internalFlags(0)(0:16)  | materialIndex0(16:32)
//   13     materialIndex1(0:16)    | pad[0](16:32) -- NOT written, the stream's bytes stay
//   14,15  pad[1..4]                -- NOT written

struct MaterialData
{
    float dynamicFriction;
    float staticFriction;
    float restitution;
    float damping;
    uint flags;
    uint fricCombineMode;
    uint restCombineMode;
    uint dampingCombineMode;
}

struct PatchWork
{
    uint patchIndex;      // byte offset into the patch stream, or 0xffffffff
    uint nbContacts;
    uint materialIndex0;
    uint materialIndex1;
    float4 normal;        // .xyz
}

struct InsertParams { uint numTests; }

[[vk::push_constant]]
ConstantBuffer<InsertParams> gParams;

RWStructuredBuffer<MaterialData> gMaterials;   // binding 0
RWStructuredBuffer<PatchWork> gWork;           // binding 1
RWStructuredBuffer<uint> gPatchStream;         // binding 2  (word view)

import pxgSlangNpPatchTail;

static const uint MODE_AVERAGE = 0u, MODE_MIN = 1u, MODE_MULTIPLY = 2u, MODE_MAX = 3u;
static const uint FLAG_DISABLE_FRICTION = 1u, FLAG_DISABLE_STRONG_FRICTION = 2u, FLAG_COMPLIANT_ACC_SPRING = 16u;
static const uint INVALID = 0xffffffffu;

// P9.83b-7: THE COMBINE WAS A SECOND TRANSCRIPTION AND IS NOW A CALL.
//
// This file used to carry its own full copy of combineScalars + combineMaterials
// and import nothing.  So this fixture -- 5000 cases with every patch field
// compared against a C++ reference -- certified THAT copy, while the body the
// engine actually runs (pxgSlangNpPatchTail, reached from triPlaneNpBda and
// npCvxPlaneNphase) was covered only indirectly by scenes.  The two were
// equivalent when compared line by line; the point is that nothing would have
// noticed if they had stopped being so.
//
// The layouts still differ -- PxsMaterialDataGpu packs the flags and the friction
// and restitution modes into flagsModes0 and the damping mode into modes1, while
// this fixture carries them unpacked -- so the shared body takes UNPACKED
// scalars and the module's BDA entry unpacks and delegates to it.  This fixture
// now feeds that same body, which is what makes gpuslang.npInsertPatch evidence
// about the engine rather than about a copy.
void combineMaterials(MaterialData m0, MaterialData m1, out float sta, out float dyn,
                      out float rest, out float damping, out uint flags)
{
    NptMaterialIn a;
    a.dynamicFriction = m0.dynamicFriction;
    a.staticFriction = m0.staticFriction;
    a.restitution = m0.restitution;
    a.damping = m0.damping;
    a.flags = m0.flags;
    a.fricCombineMode = m0.fricCombineMode;
    a.restCombineMode = m0.restCombineMode;
    a.dampingCombineMode = m0.dampingCombineMode;

    NptMaterialIn b;
    b.dynamicFriction = m1.dynamicFriction;
    b.staticFriction = m1.staticFriction;
    b.restitution = m1.restitution;
    b.damping = m1.damping;
    b.flags = m1.flags;
    b.fricCombineMode = m1.fricCombineMode;
    b.restCombineMode = m1.restCombineMode;
    b.dampingCombineMode = m1.dampingCombineMode;

    const NptCombinedMaterial c = nptCombineMaterialsRaw(a, b);
    sta = c.staticFriction;
    dyn = c.dynamicFriction;
    rest = c.restitution;
    damping = c.damping;
    flags = c.materialFlags;
}

[shader("compute")]
[numthreads(256, 1, 1)]
void insertIntoPatchStream(uint3 dtid : SV_DispatchThreadID)
{
    const uint i = dtid.x;
    if (i >= gParams.numTests) return;

    const PatchWork w = gWork[i];
    if (w.patchIndex == INVALID) return;

    float sta, dyn, rest, damping;
    uint materialFlags;
    combineMaterials(gMaterials[w.materialIndex0], gMaterials[w.materialIndex1],
                     sta, dyn, rest, damping, materialFlags);

    const uint base = w.patchIndex / 4u; // byte offset -> word index

    const uint one = asuint(1.0f);
    gPatchStream[base + 0u] = one; // mMassModification.linear0
    gPatchStream[base + 1u] = one; // angular0
    gPatchStream[base + 2u] = one; // linear1
    gPatchStream[base + 3u] = one; // angular1

    gPatchStream[base + 4u] = asuint(w.normal.x);
    gPatchStream[base + 5u] = asuint(w.normal.y);
    gPatchStream[base + 6u] = asuint(w.normal.z);

    gPatchStream[base + 7u] = asuint(rest);
    gPatchStream[base + 8u] = asuint(dyn);
    gPatchStream[base + 9u] = asuint(sta);
    gPatchStream[base + 10u] = asuint(damping);

    // startContactIndex(0) | nbContacts(u8)<<16 | materialFlags(u8)<<24
    // (nbContacts and materialFlags are u8 fields in PxContactPatch — the CUDA
    //  truncates them via the struct assignment, so mask to a byte here)
    gPatchStream[base + 11u] = ((w.nbContacts & 0xffu) << 16) | ((materialFlags & 0xffu) << 24);
    // internalFlags(0) | materialIndex0<<16
    gPatchStream[base + 12u] = (w.materialIndex0 & 0xffffu) << 16;
    // materialIndex1 (low 16) is a PxU16 store (contactPatchUtils.cuh:171), so pad[0]
    // in the high 16 keeps the stream's bytes: read-modify-write, exact because this
    // invocation is the slot's only writer.  Words 14-15 (pad[1..4]) are not written.
    // (P9.87: this line zeroed pad[0] while its comment said the CUDA leaves it.)
    gPatchStream[base + 13u] = (gPatchStream[base + 13u] & 0xFFFF0000u) | (w.materialIndex1 & 0xffffu);
}
