// npArtiInverseDyn.slang
//
// P8.112: the direct-GPU-API inverse-dynamics family (inverseDynamic.cu),
// starting with `computeArtiCOM` (:1019-1075).  These kernels are RECORD-side
// -- one warp per articulation over the PxgArticulation pool, no block SoA --
// launched only through PxDirectGPUAPI::computeArticulationData, never by a
// scene.
//
// computeArtiCOM: grid ceil(nbElements/8), block (32, 8, 1) -- one warp per
// job, jobIndex = threadIdx.y + blockIdx.x * 8.  Sums link masses and
// mass-weighted link positions into data[jobIndex] via GLOBAL ATOMIC float
// adds, then lane 0 normalises and optionally moves the result into the root
// frame.
//
// TRAPS, each a sweep mutant unless marked:
//  1. gpuIndices REMAPS: the job index is not the articulation index.
//  2. The link loops are warp-strided (link = lane; link += 32) -- an
//     articulation with more than 32 links gives some lanes two links.
//  3. The total mass is accumulated INTO data[jobIndex].x and read back
//     (totalMass), then subtracted out in the tail before the divide -- the
//     caller contract is a ZEROED output buffer (a non-zero x would pollute
//     the DIVISOR, not just the sum).
//  4. mass = 1 / linkProps[link].invInertiaXYZ_invMass.w -- the .w lane of a
//     16-byte record, and linkBody2Worlds is a 28-byte PxTransform array
//     with p at offset 16 (q first).
//  5. The divide is PxVec3::operator/=: ONE reciprocal, three multiplies
//     (the P8.102b recip-multiply spelling).
//  6. The root-frame move is PxTransform::getInverse().transform(v):
//     p' = quatRotateInv(q, -p), q' = conjugate(q), result =
//     quatRotate(q', v) + p' -- two rotates and an add, NOT one rotate of
//     (v - p).
//
// EIGHTH UPSTREAM-BUG CANDIDATE (transcribed verbatim): there is NO
// jobIndex guard -- a launch whose nbElements is not a multiple of 8 has
// overhang warps that read gpuIndices past the end and dereference whatever
// articulation that garbage names.  The fixture keeps nbElements an exact
// multiple.
//
// DOCUMENTED DEVIATION: the CUDA has no sync between the position-sum loop
// and the lane-0 tail (it relies on warp-synchronous visibility of the
// red.global adds).  On our subgroup-8 emulation a warp spans four
// subgroups, so the port inserts AllMemoryBarrierWithGroupSync() there --
// value-preserving (the sums are complete either way in the engine's
// intended semantics), but a REAL ordering the CUDA text does not spell.
// The first __syncwarp (before the totalMass read) becomes the same
// block-wide barrier for the same reason; both barrier points are uniform
// across the block, so the stronger scope is safe.

import pxgSlangCommon;
import pxgSlangArticulationDesc;
import pxgSlangAtomicFloat;

static const uint ID_WARP = 32u;

struct ComParams
{
    uint64_t data;           // @0  PxVec3* (12-byte stride)
    uint64_t gpuIndices;     // @8  PxArticulationGPUIndex* (u32)
    uint rootFrameWord;      // @16 bool byte 0
    uint pad0;               // @20
    uint64_t articulations;  // @24 PxgArticulation* (544-byte records)
}

[[vk::push_constant]]
ConstantBuffer<ComParams> gCom;

[shader("compute")]
[numthreads(32, 8, 1)]
void computeArtiCOM(uint3 groupThreadId : SV_GroupThreadID, uint3 groupId : SV_GroupID)
{
    const uint jobIndex = groupThreadId.y + groupId.x * 8u;
    const uint threadIndex = groupThreadId.x;

    // trap 1 / eighth candidate: unguarded, remapped
    const uint artiIndex = (Ptr<uint>(gCom.gpuIndices))[jobIndex];
    const uint64_t record = gCom.articulations + uint64_t(artiIndex) * PXG_ARTI_STRIDE;

    const uint64_t comAddr = gCom.data + uint64_t(jobIndex) * 12u;
    const uint linkCount = artiU16Field(record, PXG_ARTI_DATA_NUM_LINKS);
    const uint64_t linkProps = artiPtrField(record, PXG_ARTI_LINK_PROPS);
    const uint64_t body2Worlds = artiPtrField(record, PXG_ARTI_LINK_BODY2WORLDS);

    // total mass -> data[jobIndex].x (trap 3)
    for (uint link = threadIndex; link < linkCount; link += ID_WARP)
    {
        const float mass = 1.0f / (Ptr<float>(linkProps + uint64_t(link) * 16u))[3];
        pxgAtomicAddF32(comAddr, 0u, mass);
    }

    AllMemoryBarrierWithGroupSync();   // __syncwarp, widened (see the header)

    const float totalMass = (Ptr<float>(comAddr))[0];

    // mass-weighted position sums (trap 2 / trap 4)
    for (uint link = threadIndex; link < linkCount; link += ID_WARP)
    {
        Ptr<float> t = Ptr<float>(body2Worlds + uint64_t(link) * PXG_ARTI_TRANSFORM_SIZE);
        const float mass = 1.0f / (Ptr<float>(linkProps + uint64_t(link) * 16u))[3];
        pxgAtomicAddF32(comAddr, 0u, t[4] * mass);
        pxgAtomicAddF32(comAddr, 1u, t[5] * mass);
        pxgAtomicAddF32(comAddr, 2u, t[6] * mass);
    }

    // the DEVIATION: the CUDA text has no barrier here
    AllMemoryBarrierWithGroupSync();

    if (threadIndex == 0u)
    {
        Ptr<float> com = Ptr<float>(comAddr);
        float x = com[0] - totalMass;   // remove the mass sum again (trap 3)
        float y = com[1];
        float z = com[2];

        // trap 5: PxVec3::operator/= -- one reciprocal, three multiplies
        const float inv = 1.0f / totalMass;
        x *= inv;
        y *= inv;
        z *= inv;

        if ((gCom.rootFrameWord & 0xFFu) != 0u)
        {
            // trap 6: getInverse() then transform(), the engine's own two-step
            Ptr<float> rt = Ptr<float>(body2Worlds);
            const float4 q = float4(rt[0], rt[1], rt[2], rt[3]);
            const float3 p = float3(rt[4], rt[5], rt[6]);
            const float3 invP = quatRotateInv(q, -p);
            const float4 invQ = float4(-q.x, -q.y, -q.z, q.w);
            const float3 v = quatRotate(invQ, float3(x, y, z)) + invP;
            x = v.x;
            y = v.y;
            z = v.z;
        }

        com[0] = x;
        com[1] = y;
        com[2] = z;
    }

    AllMemoryBarrierWithGroupSync();   // the trailing __syncwarp
}
