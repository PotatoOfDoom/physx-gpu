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

// npPsMaterials.slang
//
// P9.1a -- ps_updateMaterials (particlesystem.cu:5780-5820), the per-phase-group
// derived-material precompute.  Ported to the Vulkan engine ABI (BDA).
//
// WHY THIS KERNEL FIRST.  It is the ONLY writer of
// PxgParticleSystem::mDerivedPBDMaterialData, and both PBD density kernels
// (ps_calculateDensityAndPotentialLaunch, ps_solveDensityLaunch) read that array
// every frame.  The engine allocates it with an explicit "no need to copy old
// data" comment (PxgPBDParticleSystemCore.cpp:2065) and PxgCudaBuffer::allocate
// does NOT zero, so with this kernel unmapped the fluid solve would consume
// uninitialised cohesion coefficients rather than fault.  It is already launched
// unconditionally every frame (PxgPBDParticleSystemCore.cpp:1332-1351, called
// from :141 before preIntegrateSystems), so mapping it is independently
// observable against the existing particle scene.
//
// LAUNCH.  args (particleSystems*, activeParticleSystems*, pbdMaterials*,
// invTotalDt) -> {8,8,8,4}; grid (ceil(mMaxNumPhaseToMaterials/256),
// nbActiveParticleSystems, 1), block (PS_SOLVE = 256, 1, 1), bpStream.  Guarded
// host-side by (mMaxNumPhaseToMaterials && mMaxParticles).
//
// Differences from the CUDA, each deliberate:
//
//   1. NO staging to drop.  Unlike every other ps_* kernel in this family this
//      one has no blockCopy of PxgParticleSystem into __shared__ -- it reads
//      particleSystems[particleId] straight from global.  The port's direct BDA
//      reads are therefore the same shape the CUDA already has, not a
//      simplification.
//
//   2. `PxReal surfaceTension = mat.surfaceTension;` (:5803) is DEAD -- the very
//      next use assigns over it at :5808 without reading it.  The port omits the
//      dead initialisation.  (Its presence is what makes the live line look like
//      a refinement of the raw value when it is in fact a replacement.)
//
//   3. `invTotalDt` is a DEAD PARAMETER: it appears in the signature and in the
//      host's kernelParams and is never referenced in the body.  It stays in the
//      push-constant layout because the ABI blob must line up, and is never read.
//
//   4. The engine indexes `pbdMaterials[mi]` as a TYPED C++ array -- stride
//      sizeof(PxsPBDMaterialData) = 60 fixed at compile time -- NOT through
//      getParticleMaterial with the runtime mParticleMaterialStride the fluid
//      kernels use.  The two are equal at runtime (mParticleMaterialStride is
//      assigned sizeof(PxsPBDMaterialData) at PxgPBDParticleSystemCore.cpp:2060),
//      so substituting one for the other is an identity here; the port spells the
//      static 60 because that is what this kernel does.
//
//   5. mPhaseGroupToMaterialHandle is a PxU16 array, read as a 32-bit word plus a
//      shift/mask so the software Vulkan target needs no 16-bit storage
//      capability -- the same loadPhaseToMaterial idiom particleSelfCollisionBda
//      already ships.
//
// Traps this kernel contains, each of which gets a mutant:
//
//   T1. The write index is `i` (the PHASE GROUP) while the material lookup index
//       is `mi = phaseToMat[i]`.  Allocation, fill and read all agree on GROUP
//       (PxgPBDParticleSystemCore.cpp:2049+2065 sizes the buffer by
//       mPhaseGroupToMaterialHandle.size(); particlesystem.cu:3271 and :3456 read
//       it by group), so two groups sharing one material legitimately get two
//       identical derived slots.  Swapping i and mi is wrong in both directions.
//   T2. The guard bound is the PER-SYSTEM mData.mNumPhaseToMaterials, while the
//       grid is sized by the ACROSS-SYSTEM maximum mMaxNumPhaseToMaterials.  The
//       in-kernel guard is the only thing keeping a small system in a large grid
//       from scribbling past its own array.
//   T3. surfaceTension is (invRestDensity * matSurfaceTension) /
//       fluidSurfaceConstraintScale -- a multiply THEN a divide, left to right.
//   T4. cohesion is matCohesion * particleContactDistance -- the CONTACT
//       distance, not the fluid rest distance.
//   T5. rest is fluidRestDistance / particleContactDistance where
//       fluidRestDistance = 2 * mFluidRestOffset -- the factor two is on the
//       fluid rest OFFSET, and mFluidRestOffset is a distinct field from
//       mSolidRestOffset and from mRestOffset.
//   T6. cohesion1 carries a UNARY MINUS on the whole numerator:
//       -(1 + rest) / sqr(rest).  Dropping it flips the cubic.
//   T7. cohesion2's numerator is sqr(rest) + rest + 1, in that order.
//   T8. The derived record is four floats in the order
//       (surfaceTensionDerived, cohesion, cohesion1, cohesion2) -- the ctor
//       argument order at :5817, which is also the struct's declaration order
//       (PxgParticleSystem.h:71-86, sizeof 16).
//
// The module is in PXG_NOCONTRACT_SHADERS: the derived coefficients are dense
// float arithmetic whose bit pattern the density kernels consume directly, and
// FMA contraction anywhere in them moves the result off the engine's.

import pxgSlangParticleDesc;

// PxgParticleSystemKernelBlockDim::PS_SOLVE (PxgParticleSystemCoreKernelIndices.h:46)
static const uint PS_UPDATE_MATERIALS_BLOCK = 256u;

// sizeof(PxsPBDMaterialData) -- the 20-byte PxsParticleMaterialData base plus ten
// PxReal (PxsPBDMaterialCore.h:37-52); generated and pinned by the fixture.
static const uint PXG_PBD_MATERIAL_STRIDE = 60u;
// PxsPBDMaterialData float indices used here: surfaceTension @32, cohesion @36.
static const uint PXG_PBD_MAT_F_SURFACE_TENSION = 8u;
static const uint PXG_PBD_MAT_F_COHESION = 9u;

// sizeof(PxgPBDParticleMaterialDerived) (PxgParticleSystem.h:71-86) -- four PxReal.
static const uint PXG_PBD_DERIVED_STRIDE = 16u;

float sqr(float x) { return x * x; }

// phaseToMat[group] where phaseToMat is a PxU16 array; a 32-bit load + shift/mask
// avoids requiring 16-bit storage on the software Vulkan target (as particleSim
// and particleSelfCollisionBda).
uint loadPhaseToMaterial(uint64_t handleBase, uint group)
{
    let word = (Ptr<uint>(handleBase))[group >> 1];
    return (word >> ((group & 1u) * 16u)) & 0xffffu;
}

// argSizes (launch site PxgPBDParticleSystemCore.cpp:1342-1348):
// { particleSystems*, activeParticleSystems*, pbdMaterials*, PxReal invTotalDt }
// -> {8,8,8,4} = 28, rounded to the 8-alignment -> 32-byte blob.
struct PsUpdateMaterialsParams
{
    uint64_t particleSystems;        // @0
    uint64_t activeParticleSystems;  // @8
    uint64_t pbdMaterials;           // @16
    float invTotalDt;                // @24  DEAD in the CUDA body (difference 3)
    uint _pad28;                     // @28 -> 32
}
[[vk::push_constant]]
ConstantBuffer<PsUpdateMaterialsParams> gUm;

[shader("compute")]
[numthreads(256, 1, 1)]
void ps_updateMaterials(uint3 groupThreadId : SV_GroupThreadID, uint3 groupId : SV_GroupID)
{
    Ptr<PxgParticleSystemGpu> systems = Ptr<PxgParticleSystemGpu>(gUm.particleSystems);
    const uint particleId = (Ptr<uint>(gUm.activeParticleSystems))[groupId.y];

    // T2: the bound is this system's own count, not the grid's max.
    const uint numPhaseToMaterials = systems[particleId].mData_NumPhaseToMaterials;

    const uint i = groupThreadId.x + PS_UPDATE_MATERIALS_BLOCK * groupId.x;

    if (i < numPhaseToMaterials)
    {
        // T1: material looked up BY GROUP, record written AT GROUP.
        const uint mi = loadPhaseToMaterial(systems[particleId].mPhaseGroupToMaterialHandle, i);

        // difference 4: static 60-byte stride, matching the CUDA's typed array.
        const uint64_t matBase = gUm.pbdMaterials + uint64_t(mi) * uint64_t(PXG_PBD_MATERIAL_STRIDE);
        const float matSurfaceTension = (Ptr<float>(matBase))[PXG_PBD_MAT_F_SURFACE_TENSION];
        const float matCohesion = (Ptr<float>(matBase))[PXG_PBD_MAT_F_COHESION];

        // T5: two times the FLUID rest offset.
        const float fluidRestDistance = 2.0f * systems[particleId].mData_FluidRestOffset;
        const float particleContactDistance = systems[particleId].mCommonData_ParticleContactDistance;

        // T3: multiply then divide, left to right (:5808).  The dead
        // `PxReal surfaceTension = mat.surfaceTension;` at :5803 is omitted.
        const float surfaceTension = systems[particleId].mData_InvRestDensity * matSurfaceTension
                                   / systems[particleId].mData_FluidSurfaceConstraintScale;

        // T4: scaled by the CONTACT distance (:5810).
        const float cohesion = matCohesion * particleContactDistance;

        // coefficients for a cubic that has C(0)=-1, C(restDistance)=0, C(1)=0,
        // dCdX(0)=0 (the CUDA's own comment at :5812).
        const float rest = fluidRestDistance / particleContactDistance;
        const float cohesion1 = -(1.0f + rest) / sqr(rest);              // T6
        const float cohesion2 = (sqr(rest) + rest + 1.0f) / sqr(rest);   // T7

        // T8: declaration order of PxgPBDParticleMaterialDerived.
        Ptr<float> derived = Ptr<float>(systems[particleId].mDerivedPBDMaterialData
                                        + uint64_t(i) * uint64_t(PXG_PBD_DERIVED_STRIDE));
        derived[0] = surfaceTension;
        derived[1] = cohesion;
        derived[2] = cohesion1;
        derived[3] = cohesion2;
    }
}
