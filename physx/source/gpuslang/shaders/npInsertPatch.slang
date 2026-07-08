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
//   13     materialIndex1(0:16)    | pad (untouched by the CUDA)
//   14,15  pad

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

static const uint MODE_AVERAGE = 0u, MODE_MIN = 1u, MODE_MULTIPLY = 2u, MODE_MAX = 3u;
static const uint FLAG_DISABLE_FRICTION = 1u, FLAG_DISABLE_STRONG_FRICTION = 2u, FLAG_COMPLIANT_ACC_SPRING = 16u;
static const uint INVALID = 0xffffffffu;

float combineScalars(float a, float b, uint mode)
{
    if (mode == MODE_AVERAGE) return 0.5f * (a + b);
    if (mode == MODE_MIN) return min(a, b);
    if (mode == MODE_MULTIPLY) return a * b;
    if (mode == MODE_MAX) return max(a, b);
    return 0.0f;
}

// mirrors PxsCombineMaterials -> (staticFriction, dynamicFriction, restitution, damping, flags)
void combineMaterials(MaterialData m0, MaterialData m1, out float sta, out float dyn,
                      out float rest, out float damping, out uint flags)
{
    const float r0 = m0.restitution, r1 = m1.restitution;
    const bool compliant0 = r0 < 0.0f, compliant1 = r1 < 0.0f;
    const bool exactlyOneCompliant = compliant0 != compliant1;
    const bool bothCompliant = compliant0 && compliant1;
    const bool cAcc0 = (m0.flags & FLAG_COMPLIANT_ACC_SPRING) != 0u;
    const bool cAcc1 = (m1.flags & FLAG_COMPLIANT_ACC_SPRING) != 0u;
    const bool exactlyOneAcc = cAcc0 != cAcc1;

    if (bothCompliant && exactlyOneAcc)
        rest = cAcc0 ? r0 : r1;
    else
    {
        const uint mode = exactlyOneCompliant ? MODE_MIN : max(m0.restCombineMode, m1.restCombineMode);
        const float flip = (bothCompliant && (mode == MODE_MULTIPLY)) ? -1.0f : 1.0f;
        rest = flip * combineScalars(r0, r1, mode);
    }

    if (bothCompliant && exactlyOneAcc)
        damping = cAcc0 ? m0.damping : m1.damping;
    else
    {
        const uint mode = exactlyOneCompliant ? MODE_MAX : max(m0.dampingCombineMode, m1.dampingCombineMode);
        damping = combineScalars(m0.damping, m1.damping, mode);
    }

    const uint combineFlags = m0.flags | m1.flags;
    if ((combineFlags & FLAG_DISABLE_FRICTION) == 0u)
    {
        const uint fm = max(m0.fricCombineMode, m1.fricCombineMode);
        const float dynF = combineScalars(m0.dynamicFriction, m1.dynamicFriction, fm);
        const float staF = combineScalars(m0.staticFriction, m1.staticFriction, fm);
        const float fDyn = max(dynF, 0.0f);
        dyn = fDyn;
        sta = (staF - fDyn) >= 0.0f ? staF : fDyn;
        flags = combineFlags;
    }
    else
    {
        flags = combineFlags | FLAG_DISABLE_STRONG_FRICTION;
        dyn = 0.0f;
        sta = 0.0f;
    }
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
    // materialIndex1 (low 16); pad in the high 16 is untouched by the CUDA
    gPatchStream[base + 13u] = (w.materialIndex1 & 0xffffu);
}
