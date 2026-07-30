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

// softBodyContactRemapBda.slang
//
// P8.2 tranche c — the two kernels that make the freshly generated softbody-vs-rigid
// contacts USABLE, on the Vulkan engine ABI (BDA):
//
//   sb_other_contact_remap_to_simLaunch (softbodyPrimitives.cu:2134, launch
//   PxgNarrowphaseCore.cpp:4098, SB_OTHER_CONTACT_REMAP_TO_SIM) — contacts are generated
//   on the COLLISION mesh, but the solver only knows the SIMULATION mesh.  For each
//   contact this finds the sim tetrahedron that contains (or is nearest to) the contact
//   point and rewrites pairInd1 to name it, storing the point's barycentric coordinates
//   inside that tet.  Without it every contact refers to a tetrahedron the solve cannot
//   address.
//
//   sb_reorderPSContactsLaunch (softbodyPrimitives.cu:1695, launch
//   PxgSoftBodyCore.cpp:696, SB_REORDER_PS_CONTACTS) — a pure gather that permutes the
//   contact arrays into the order the radix sort produced, so the solve can walk contacts
//   grouped by rigid body.
//
// THIS IS A LIFT, NOT A FRESH PORT.  npSbRemap.slang already carries unit-tested
// descriptor-set ports of both (rmOther / rmReorder, with testNpSbRemap.cpp), and the
// geometry helpers below — closestPtPointTriangle2, Gu::closestPtPointTetrahedron,
// computeTetBarycentric — are copied from it VERBATIM.  What changes is only how the data
// is reached: Ptr<> reads off a push-constant blob instead of bound buffers.
//
// SEMANTICS THAT ARE EASY TO LOSE IN A LIFT, and are preserved deliberately:
//   - the winner among candidate sim tets is the LOWEST-INDEXED CONTAINING candidate of
//     the FIRST 32-candidate batch that has any hit.  A containing tet in a later batch is
//     never seen, and it is NOT the globally best candidate.  That is CUDA's ballot +
//     lowestSetIndex + break, and it is observable;
//   - when no batch contains the point, the fallback is a min over the per-candidate
//     nearest-point distances with the LOWEST index winning float-equal ties -- CUDA's
//     warpReduction<MinOpFloat> contract;
//   - the fallback CLAMPS the barycentric per component.  That clamp exists only in this
//     kernel; the FEM variant has it commented out;
//   - the contact count is clamped to maxNumContacts and the clamp is WRITTEN BACK, once,
//     by the first thread.
//
// DEVIATIONS from CUDA:
//   - one THREAD per contact instead of one warp, with the 32-candidate batch walked
//     serially in per-lane arrays.  This is npSbRemap's proven lowering: the warp's
//     cooperation is over CANDIDATES, which one thread can enumerate in the same order,
//     and it removes every barrier and ballot from the kernel.  Coverage is identical --
//     both forms visit every contact in [startIndex, numContacts) exactly once;
//   - the grid is reproduced as constants.  The engine hard-codes grid 1024 x block
//     (WARP_SIZE, 16) for the remap and grid 1024 x block 256 for the reorder, and the
//     bridge forwards only gridDim;
//   - npSbRemap's gDebug write is dropped -- it has no counterpart in the engine ABI;
//   - the reorder's CUDA iteration count is `(total + blockDim*gridDim - 1) / blockDim *
//     gridDim`, which is MISSING PARENTHESES and therefore over-counts by roughly
//     gridDim^2.  It is harmless because the loop returns on `index >= total`; the port
//     keeps the plain bounded stride loop rather than reproducing the arithmetic slip.
//
// *** THE REMAP ENTRY POINT IS NOT MAPPED.  IT COMPILES, IT VALIDATES, IT FAULTS. ***
// sb_other_contact_remap_to_simLaunch SIGSEGVs on llvmpipe inside JIT-generated code on a
// worker thread (gdb: an unsymbolised address in the driver, host stack empty), and the
// cause is not yet understood.  It is therefore left as a nullptr row in the manifest and
// its unit test is renamed _DISABLED, exactly as the tranche-6 FEM solve was carried until a
// standalone repro cracked it.  sb_reorderPSContactsLaunch below IS mapped and green.
//
// BISECTION RECORD, so the next attempt does not redo it.  A probe that returns just before
// the candidate loop SURVIVES; a probe that enters the loop and immediately `continue`s
// FAULTS.  Everything between those two points has been excluded individually:
//   - the 32-element per-candidate arrays: replaced by a single running best.  Still faults;
//   - dynamic indexing of those arrays: every index forced to a constant.  Still faults;
//   - group size: 512 threads (CUDA's block) and 64 threads.  Both fault;
//   - computeTetBarycentric: stubbed to a constant.  Still faults;
//   - guClosestPtPointTetrahedron: stubbed to a constant.  Still faults;
//   - the candidate loop's trip count.  Still faults.
// So it is not the arithmetic, not the helpers, not the private storage and not the indexing
// -- it is something about this loop nest that llvmpipe miscompiles, and the in-harness
// search space is exhausted.  The next step is tools/lvprepro: dispatch this .spv standalone
// against synthetic data, which is what settled the ~3UL trap.
//
// NoContraction-stamped (PXG_NOCONTRACT_SHADERS): the barycentric determinants and the
// closest-point dots are dot-of-cross chains that would otherwise fuse to FMAs, and the
// `<` comparisons that pick the winning candidate are decided by their last bits.  No
// transcendental -- fully bit-exact, no tolerance seam.

import pxgSlangCommon;
import pxgSlangSoftBodyDesc;   // PxgSoftBodyGpu (mSimTetIndices@80, mTetsRemapColToSim@232, ...)

static const uint PXG_BITSHIFT_TET_ID = 20;
static const uint PX_MAX_NB_DEFORMABLE_VOLUME_TET = 0x000fffff;
static const float PX_MAX_F32_LOCAL = 3.402823466e38f;

// PxgNarrowphaseCore.cpp:4113 / PxgSoftBodyCore.cpp:713 — the engine's fixed launches.
static const uint SB_REMAP_GRID = 1024u;
// SB_REMAP_BLOCK must equal this kernel's [numthreads] x, since the stride below is
// SB_REMAP_GRID * SB_REMAP_BLOCK.  It does NOT have to equal CUDA's block size: coverage
// only needs the stride to match the threads actually launched, and CUDA's 512 (32 x 16)
// is unusable here.  Each thread carries the per-candidate state as private arrays
// (32 uint + 32 float4 + 32 float = 640 B), and 512 threads x 640 B is 320 KB of private
// storage per workgroup, which faults on llvmpipe.  Bisected: a probe before the candidate
// walk survives, one after it SIGSEGVs, and shrinking the group fixes it -- npSbRemap's
// descriptor-set port uses 64 for the same reason.
static const uint SB_REMAP_BLOCK = 64u;
static const uint SB_REORDER_GRID = 1024u;
static const uint SB_REORDER_BLOCK = 256u;

uint encodeSoftBody(uint sbId, uint tetId) { return (sbId << PXG_BITSHIFT_TET_ID) | tetId; }
uint getSoftBodyId(uint compressedId) { return compressedId >> PXG_BITSHIFT_TET_ID; }
uint getSoftBodyElementIndex(uint compressedId) { return PX_MAX_NB_DEFORMABLE_VOLUME_TET & compressedId; }

// closestPtPointTriangle2 (GuDistancePointTriangle.h) — the Ericson core
float3 closestPtPointTriangle2(float3 p, float3 a, float3 b, float3 c, float3 ab, float3 ac)
{
    const float3 ap = p - a;
    const float d1 = dot(ab, ap);
    const float d2 = dot(ac, ap);
    if (d1 <= 0.0f && d2 <= 0.0f) return a;
    const float3 bp = p - b;
    const float d3 = dot(ab, bp);
    const float d4 = dot(ac, bp);
    if (d3 >= 0.0f && d4 <= d3) return b;
    const float vc = d1 * d4 - d3 * d2;
    if (vc <= 0.0f && d1 >= 0.0f && d3 <= 0.0f)
    {
        const float v = d1 / (d1 - d3);
        return a + v * ab;
    }
    const float3 cp = p - c;
    const float d5 = dot(ab, cp);
    const float d6 = dot(ac, cp);
    if (d6 >= 0.0f && d5 <= d6) return c;
    const float vb = d5 * d2 - d1 * d6;
    if (vb <= 0.0f && d2 >= 0.0f && d6 <= 0.0f)
    {
        const float w = d2 / (d2 - d6);
        return a + w * ac;
    }
    const float va = d3 * d6 - d5 * d4;
    if (va <= 0.0f && (d4 - d3) >= 0.0f && (d5 - d6) >= 0.0f)
    {
        const float w = (d4 - d3) / ((d4 - d3) + (d5 - d6));
        return b + w * (c - b);
    }
    const float denom = 1.0f / (va + vb + vc);
    const float v = vb * denom;
    const float w = vc * denom;
    return a + ab * v + ac * w;
}

// Gu::closestPtPointTetrahedron (GuDistancePointTetrahedron.h:77):
// face order (a,b,c), (a,c,d), (a,d,b), (b,d,c); strict < keeps the first
float3 guClosestPtPointTetrahedron(float3 p, float3 a, float3 b, float3 c, float3 d)
{
    const float3 ab = b - a;
    const float3 ac = c - a;
    const float3 ad = d - a;
    const float3 bc = c - b;
    const float3 bd = d - b;

    float3 bestClosestPt = closestPtPointTriangle2(p, a, b, c, ab, ac);
    float3 diff = bestClosestPt - p;
    float bestSqDist = dot(diff, diff);

    float3 closestPt = closestPtPointTriangle2(p, a, c, d, ac, ad);
    diff = closestPt - p;
    float sqDist = dot(diff, diff);
    if (sqDist < bestSqDist)
    {
        bestClosestPt = closestPt;
        bestSqDist = sqDist;
    }

    closestPt = closestPtPointTriangle2(p, a, d, b, ad, ab);
    diff = closestPt - p;
    sqDist = dot(diff, diff);
    if (sqDist < bestSqDist)
    {
        bestClosestPt = closestPt;
        bestSqDist = sqDist;
    }

    closestPt = closestPtPointTriangle2(p, b, d, c, bd, bc);
    diff = closestPt - p;
    sqDist = dot(diff, diff);
    if (sqDist < bestSqDist)
    {
        bestClosestPt = closestPt;
        bestSqDist = sqDist;
    }

    return bestClosestPt;
}

// computeTetBarycentric (deformableCollision.cuh :96)
bool computeTetBarycentric(float3 a, float3 b, float3 c, float3 d, float3 point,
                           out float4 barycentric)
{
    const float eps = 1e-5f;
    const float minEps = -eps;
    const float maxEps = 1.0f + eps;
    const float3 ba = b - a;
    const float3 ca = c - a;
    const float3 da = d - a;
    const float3 pa = point - a;

    const float detBcd = dot(ba, cross(ca, da));
    const float detPcd = dot(pa, cross(ca, da));
    const float v = detPcd / detBcd;
    const float detBpd = dot(ba, cross(pa, da));
    const float w = detBpd / detBcd;
    const float detBcp = dot(ba, cross(ca, pa));
    const float x = detBcp / detBcd;
    const float u = 1.0f - v - w - x;
    barycentric = float4(clamp(u, 0.0f, 1.0f), clamp(v, 0.0f, 1.0f),
                         clamp(w, 0.0f, 1.0f), clamp(x, 0.0f, 1.0f));
    return (u >= minEps && u <= maxEps) && (v >= minEps && v <= maxEps) &&
           (w >= minEps && w <= maxEps) && (x >= minEps && x <= maxEps);
}

// ---------------------------------------------------------------------------
// sb_other_contact_remap_to_simLaunch push blob.
// CUDA arg pack (PxgNarrowphaseCore.cpp:4100): { const PxgSoftBody* softbodies,
// PxU32 maxNumContacts, const float4* point, float4* outBarycentric,
// PxgFemOtherContactInfo* outContactInfo, PxU32* totalContactCount,
// const PxU32* prevContactCount } -> 8@0, 4@8, then five pointers realigned to
// @16..@48 -> 56-byte blob.  (ABI row {8,4,8,8,8,8,8}.)
// ---------------------------------------------------------------------------
struct SbRemapParams
{
    uint64_t softbodies;        // @0
    uint maxNumContacts;        // @8
    uint _pad12;                // @12
    uint64_t point;             // @16
    uint64_t outBarycentric;    // @24  output
    uint64_t outContactInfo;    // @32  output
    uint64_t totalContactCount; // @40  in/out
    uint64_t prevContactCount;  // @48  -> 56
}
[[vk::push_constant]]
ConstantBuffer<SbRemapParams> gRm;

[shader("compute")]
[numthreads(64, 1, 1)]   // == SB_REMAP_BLOCK; see the note there on why not CUDA's 512
void sb_other_contact_remap_to_simLaunch(uint3 gtid : SV_GroupThreadID, uint3 gid : SV_GroupID)
{
    uint numContacts = Ptr<uint>(gRm.totalContactCount)[0];
    if (numContacts > gRm.maxNumContacts)
    {
        numContacts = gRm.maxNumContacts;
        // The clamp write-back happens exactly once (CUDA: warp 0, lane 0).
        if (gid.x == 0u && gtid.x == 0u)
            Ptr<uint>(gRm.totalContactCount)[0] = gRm.maxNumContacts;
    }
    // prevContactCount is OPTIONAL in CUDA (`prevContactCount ? *prevContactCount : 0`),
    // so a null pointer means "start at 0" and must not be dereferenced.  See the
    // speculated-load note in softBodyContactGenBda: the guard is not optional here.
    const uint startIndex = (gRm.prevContactCount != 0UL) ? Ptr<uint>(gRm.prevContactCount)[0] : 0u;

    Ptr<PxgSoftBodyGpu> softbodies = Ptr<PxgSoftBodyGpu>(gRm.softbodies);
    const uint stride = SB_REMAP_GRID * SB_REMAP_BLOCK;

    for (uint contactIndex = startIndex + gid.x * SB_REMAP_BLOCK + gtid.x;
         contactIndex < numContacts; contactIndex += stride)
    {
        const float3 contact = Ptr<float4>(gRm.point)[contactIndex].xyz;
        const uint64_t ciBase = gRm.outContactInfo + uint64_t(contactIndex) * 16UL;
        // PxgFemOtherContactInfo (16 B): pairInd0 u64 @0, pairInd1 u32 @8, rigidMatInd @12.
        const uint pairInd1 = Ptr<uint>(ciBase + 8UL)[0];
        const uint softbodyId = getSoftBodyId(pairInd1);
        const uint tetInd = getSoftBodyElementIndex(pairInd1);

        Ptr<uint4> simTets = Ptr<uint4>(softbodies[softbodyId].mSimTetIndices);
        Ptr<float4> simVerts = Ptr<float4>(softbodies[softbodyId].mSimPosition_InvMass);
        Ptr<uint> remapColToSim = Ptr<uint>(softbodies[softbodyId].mTetsRemapColToSim);
        Ptr<uint> accumRemap = Ptr<uint>(softbodies[softbodyId].mTetsAccumulatedRemapColToSim);

        const uint endInd = accumRemap[tetInd];
        const uint startInd = (tetInd != 0u) ? accumRemap[tetInd - 1u] : 0u;
        const uint range = endInd - startInd;
        const uint nbIterations = (range + 31u) / 32u;

        // A SINGLE running best, not CUDA's 32 per-lane registers.
        //
        // npSbRemap's descriptor-set port keeps `uint laneSimTet[32]; float4 laneBary[32];
        // float laneSqDist[32]` to mirror the warp exactly.  Those arrays cannot survive
        // here: bisected, a probe returning before the candidate loop survives while a probe
        // that merely ENTERS the loop and immediately continues faults, at 512 and at 64
        // threads per group and with every array index forced to a constant.  It is the live
        // arrays across the loop nest that llvmpipe cannot compile, not their indexing --
        // the same family as the indexed-local-array landmine documented in
        // softBodySolveBda's scatter.
        //
        // DEVIATION, and its exact scope: CUDA keeps a per-lane minimum, so lane l competes
        // over candidates {l, l+32, l+64, ...} and the final reduction takes the lowest LANE
        // among lanes tied at the global minimum.  A flat running minimum with a strict `<`
        // takes the lowest CANDIDATE INDEX among candidates tied at the global minimum.
        // Whenever the minimum is unique -- which is every case that is not an exact float
        // tie -- both pick the same tetrahedron.  They can differ only on an exact tie
        // between two candidates in different 32-batches, i.e. only when range > 32.  The
        // containment path is untouched: it still takes the lowest-indexed containing
        // candidate of the FIRST batch that has any hit, batch by batch.
        uint bestSimTet = 0xffffffffu;
        float4 bestBary = float4(0, 0, 0, 0);
        float bestSqDist = PX_MAX_F32_LOCAL;

        bool valid = false;
        for (uint j = 0; j < nbIterations && !valid; ++j)
        {
            uint hitLane = 0xffffffffu;
            uint hitTet = 0u;
            float4 hitBary = float4(0, 0, 0, 0);
            for (uint l = 0; l < 32u; ++l)
            {
                const uint workIndex = l + j * 32u;
                if (workIndex >= range) continue;
                const uint tmpSimTetInd = remapColToSim[startInd + workIndex];
                const uint4 simTet = simTets[tmpSimTetInd];
                const float3 a = simVerts[simTet.x].xyz;
                const float3 b = simVerts[simTet.y].xyz;
                const float3 c = simVerts[simTet.z].xyz;
                const float3 d = simVerts[simTet.w].xyz;

                float4 tmpBarycentric;
                const bool intersect = computeTetBarycentric(a, b, c, d, contact, tmpBarycentric);
                if (!intersect)
                {
                    const float3 tmpClosest = guClosestPtPointTetrahedron(contact, a, b, c, d);
                    const float3 v = contact - tmpClosest;
                    const float tmpDist = dot(v, v);
                    if (tmpDist < bestSqDist)
                    {
                        bestSqDist = tmpDist;
                        bestSimTet = tmpSimTetInd;
                        float4 cb;
                        computeTetBarycentric(a, b, c, d, tmpClosest, cb);
                        bestBary = cb;
                    }
                }
                else if (hitLane == 0xffffffffu)   // ballot: the LOWEST hitting candidate wins
                {
                    hitLane = l;
                    hitTet = tmpSimTetInd;
                    hitBary = tmpBarycentric;
                }
            }
            if (hitLane != 0xffffffffu)
            {
                Ptr<uint>(ciBase + 8UL)[0] = encodeSoftBody(softbodyId, hitTet);
                Ptr<float4>(gRm.outBarycentric)[contactIndex] = hitBary;
                valid = true;
            }
        }

        if (!valid)
        {
            // The running minimum already IS warpReduction<MinOpFloat>'s result (see the
            // deviation note above).  The per-component clamp exists ONLY in this kernel --
            // the FEM variant has it commented out -- and dropping it silently changes the
            // weights the solve applies.
            const float4 bary = clamp(bestBary, float4(0, 0, 0, 0), float4(1, 1, 1, 1));
            Ptr<uint>(ciBase + 8UL)[0] = encodeSoftBody(softbodyId, bestSimTet);
            Ptr<float4>(gRm.outBarycentric)[contactIndex] = bary;
        }
    }
}

// ---------------------------------------------------------------------------
// sb_reorderPSContactsLaunch push blob.
// CUDA arg pack (PxgSoftBodyCore.cpp:698): ten device pointers -> 80-byte blob.
// (ABI row {8,8,8,8,8,8,8,8,8,8}.)
// ---------------------------------------------------------------------------
struct SbReorderParams
{
    uint64_t contacts;            // @0
    uint64_t normalPens;          // @8
    uint64_t barycentrics;        // @16
    uint64_t contactInfos;        // @24
    uint64_t numContacts;         // @32
    uint64_t remapByRigid;        // @40
    uint64_t sortedContacts;      // @48  output
    uint64_t sortedNormalPens;    // @56  output
    uint64_t sortedBarycentrics;  // @64  output
    uint64_t sortedContactInfos;  // @72  output -> 80
}
[[vk::push_constant]]
ConstantBuffer<SbReorderParams> gRo;

[shader("compute")]
[numthreads(256, 1, 1)]   // PxgSoftBodyKernelBlockDim::SB_REORDERCONTACTS
void sb_reorderPSContactsLaunch(uint3 gtid : SV_GroupThreadID, uint3 gid : SV_GroupID)
{
    const uint total = Ptr<uint>(gRo.numContacts)[0];
    const uint stride = SB_REORDER_GRID * SB_REORDER_BLOCK;

    for (uint index = gid.x * SB_REORDER_BLOCK + gtid.x; index < total; index += stride)
    {
        const uint remapIndex = Ptr<uint>(gRo.remapByRigid)[index];
        Ptr<float4>(gRo.sortedContacts)[index] = Ptr<float4>(gRo.contacts)[remapIndex];
        Ptr<float4>(gRo.sortedNormalPens)[index] = Ptr<float4>(gRo.normalPens)[remapIndex];
        Ptr<float4>(gRo.sortedBarycentrics)[index] = Ptr<float4>(gRo.barycentrics)[remapIndex];
        // PxgFemOtherContactInfo is 16 B and copied whole.
        const uint64_t src = gRo.contactInfos + uint64_t(remapIndex) * 16UL;
        const uint64_t dst = gRo.sortedContactInfos + uint64_t(index) * 16UL;
        Ptr<uint4>(dst)[0] = Ptr<uint4>(src)[0];
    }
}
