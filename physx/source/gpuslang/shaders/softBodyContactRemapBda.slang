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
//     kernel; the FEM variant has it commented out.  It is kept, and it is also provably
//     dead -- see the write-up at the site for why, and why that is upstream's shape too;
//   - the contact count is clamped to maxNumContacts and the clamp is WRITTEN BACK, once,
//     by the first thread.
//
// DEVIATIONS from CUDA:
//   - one THREAD per contact instead of one warp, with the 32-candidate batch walked
//     serially in per-lane arrays.  This is npSbRemap's proven lowering: the warp's
//     cooperation is over CANDIDATES, which one thread can enumerate in the same order,
//     and it removes every barrier and ballot from the kernel.  Coverage is identical --
//     both forms visit every contact in [startIndex, numContacts) exactly once.  The
//     per-lane state is kept as real 32-element arrays, so both tie-breaks the warp
//     reduction encodes stay bit-exact; softBodySimContactRemapTieBreaks pins them;
//   - the grid is reproduced as constants.  The engine hard-codes grid 1024 x block
//     (WARP_SIZE, 16) for the remap and grid 1024 x block 256 for the reorder, and the
//     bridge forwards only gridDim;
//   - npSbRemap's gDebug write is dropped -- it has no counterpart in the engine ABI;
//   - the reorder's CUDA iteration count is `(total + blockDim*gridDim - 1) / blockDim *
//     gridDim`, which is MISSING PARENTHESES and therefore over-counts by roughly
//     gridDim^2.  It is harmless because the loop returns on `index >= total`; the port
//     keeps the plain bounded stride loop rather than reproducing the arithmetic slip.
//
// TWO llvmpipe DEFECTS WERE FOUND HERE, and both are general enough that the next kernel
// in this port will meet them.  Each is written up at its own site below; this is the index.
//
//   1. A SPECULATED LOAD THROUGH A NULL POINTER, and the reason a guard does not stop it.
//      This kernel SIGSEGVd on every dispatch.  Slang emits a proper branch around the
//      optional `prevContactCount` read -- `OpSelectionMerge` with the `OpLoad` inside the
//      taken side, confirmed with spirv-dis -- and llvmpipe's backend if-converts it into an
//      unconditional load plus a masked select, so the load runs on a null address before
//      the test.  The fix cannot be in the control flow; it has to remove the faulting
//      address, by selecting the POINTER and loading once from something valid on both
//      paths.  See the write-up at the `prevAddr` select.
//
//      This also explains a bisection that pointed the wrong way for a long time.  A
//      read-only probe survived while adding a single store faulted, which read as "stores
//      are broken here" and led to blaming the per-candidate arrays, the helpers, the group
//      size and the loop nest -- all of them wrongly.  A read whose result is never stored
//      is dead-code eliminated: only the version that keeps the descriptor reads LIVE ever
//      reaches the null load.  Under DCE, "adding a store makes it fault" means "the reads
//      before it were being deleted", not "the store is at fault".
//
//   2. A DROPPED PER-LANE LOOP MASK when one block both redefines a loop bound and stores.
//      With the count clamp written as CUDA writes it, the whole first subgroup ran the
//      contact loop once regardless of the bound, silently writing barycentrics for
//      out-of-range contacts.  See the write-up at the clamp.
//
// Both were cornered with tools/lvprepro/repro3, which dispatches this .spv standalone
// against synthetic data -- the same tool that settled the ~3UL trap.  Neither was findable
// from inside the test harness, because both are shaped like an ordinary port bug.
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
    // CUDA writes this as one block -- `if (n > max) { n = max; if (warp0 && lane0) *total =
    // max; }` -- and transcribing it literally MISCOMPILES on llvmpipe.  With a declared
    // count of 12 clamped to 3, the first SUBGROUP (8 lanes) all ran the contact loop once
    // and stored, while every later lane correctly did nothing: the loop's per-lane exit
    // mask is dropped for the subgroup that has any active lane.  The write-back itself was
    // correct, so the symptom is silent -- out-of-range contacts get barycentrics written,
    // and nothing reports an error.  Reproduced standalone in tools/lvprepro/repro3.
    //
    // The trigger is the COMBINATION, verified by splitting it two ways: keeping the
    // mutable `numContacts` but lifting the store out of the block is correct, and keeping
    // the nested store but computing the count with `min` is correct.  Only a block that
    // both redefines the enclosing loop's bound AND stores is miscompiled.  The form below
    // removes both halves, so a later refactor cannot resurrect it by undoing one:
    // `numContacts` is a pure `min` with no phi, and the write-back is a flat conditional.
    //
    // Semantics are unchanged from CUDA: min() is the clamp, and the write-back still fires
    // only when the clamp actually bit, exactly once, from the first thread.
    const uint rawCount = Ptr<uint>(gRm.totalContactCount)[0];
    const uint numContacts = min(rawCount, gRm.maxNumContacts);
    if (rawCount > gRm.maxNumContacts && gid.x == 0u && gtid.x == 0u)
        Ptr<uint>(gRm.totalContactCount)[0] = gRm.maxNumContacts;
    // prevContactCount is OPTIONAL in CUDA (`prevContactCount ? *prevContactCount : 0`),
    // so a null pointer means "start at 0" and must not be dereferenced.
    //
    // SELECT THE POINTER, NOT THE LOADED VALUE.  Guarding the load with a branch is not
    // enough here, and the reason is worth spelling out because it costs a day to
    // rediscover.  Slang does emit a real branch -- `OpSelectionMerge` with the
    // `OpLoad` inside the taken side, verified with spirv-dis -- but llvmpipe's LLVM
    // backend if-converts it into an unconditional load plus a masked select:
    //
    //     mov  0x30(%rdx),%r10        ; r10 = prevContactCount == 0
    //     vmovd (%r10),%xmm1          ; <-- faults, BEFORE the null test
    //     neg  %r10 / sbb / kmovd     ; the "guard", evaluated afterwards
    //
    // Speculating a load across a null test is illegal, but the driver does it, and no
    // arrangement of `if`/`?:` in the source can forbid it -- the branch is already
    // there.  So the fix cannot live in the control flow: it has to remove the faulting
    // ADDRESS.  Selecting the pointer first leaves exactly one load, from an address
    // that is valid on both paths (totalContactCount was just dereferenced
    // unconditionally, so it is never null), and the result is discarded by the second
    // select.  Whether the driver speculates is then irrelevant.
    //
    // This is the stronger form of the speculated-load note in softBodyContactGenBda:
    // there a zero-trip loop hid the load, here an explicit ternary did not stop it.
    // Any optionally-null device pointer in this port needs THIS shape, not a guard.
    const uint64_t prevAddr =
        (gRm.prevContactCount != 0UL) ? gRm.prevContactCount : gRm.totalContactCount;
    const uint prevRaw = Ptr<uint>(prevAddr)[0];
    const uint startIndex = (gRm.prevContactCount != 0UL) ? prevRaw : 0u;

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

        // CUDA's 32 per-lane registers, mirrored exactly -- same as npSbRemap's
        // descriptor-set port.  Declared INSIDE the strided loop, unlike npSbRemap: there
        // one thread handles one contact, here a thread revisits the loop for
        // contactIndex + stride and the per-lane minima must start over each time.
        //
        // These arrays were briefly replaced by a single flat running minimum, on the
        // conclusion that llvmpipe could not compile them -- a probe entering the
        // candidate loop faulted while a probe returning before it survived.  That was a
        // misreading of the null prevContactCount load above: the arrays make the
        // descriptor reads live, a read whose result is never stored is dead-code
        // eliminated, and only the live version reached the speculated load.  With the
        // pointer selected instead of the value, the arrays compile and run -- verified
        // standalone at range = 1, 32, 33, 40 and 100 -- so there is no deviation to
        // document here and the warp reduction stays bit-exact.
        uint laneSimTet[32];
        float4 laneBary[32];
        float laneSqDist[32];
        for (uint li = 0u; li < 32u; ++li)
        {
            laneSimTet[li] = 0xffffffffu;
            laneBary[li] = float4(0, 0, 0, 0);
            laneSqDist[li] = PX_MAX_F32_LOCAL;
        }

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
                    if (tmpDist < laneSqDist[l])
                    {
                        laneSqDist[l] = tmpDist;
                        laneSimTet[l] = tmpSimTetInd;
                        float4 cb;
                        computeTetBarycentric(a, b, c, d, tmpClosest, cb);
                        laneBary[l] = cb;
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
            // warpReduction<MinOpFloat>: the global minimum, and among lanes holding a
            // float-equal minimum the LOWEST lane wins -- the `break` is what encodes that
            // tie-break, so it is load-bearing, not an early-out optimisation.
            float best = PX_MAX_F32_LOCAL;
            for (uint l = 0; l < 32u; ++l) best = min(best, laneSqDist[l]);
            uint winner = 0u;
            for (uint l = 0; l < 32u; ++l)
                if (laneSqDist[l] == best) { winner = l; break; }

            // This clamp is kept for line-for-line fidelity with CUDA
            // (softbodyPrimitives.cu:2261) but it is PROVABLY A NO-OP here, and the mutation
            // run says so: deleting it is the one mutant of ten that no fixture kills.  That
            // is not a gap in the fixtures -- it cannot be killed, for two independent
            // reasons:
            //   - computeTetBarycentric already clamps every component into [0,1] before
            //     returning (deformableCollision.cuh:133 does the same, so this is upstream's
            //     own double clamp, not something the port introduced);
            //   - the only value that never passed through that helper is the initial one,
            //     and it is float4(0,0,0,0) -- already in range.  CUDA declares `float4
            //     barycentric;` UNINITIALISED (softbodyPrimitives.cu:2187), so upstream's
            //     clamp does have a job on the range == 0 path: it bounds whatever was in the
            //     register.  Initialising instead of clamping is strictly better defined, and
            //     it is what makes the clamp redundant here rather than merely unexercised.
            // Deliberately not removed: it costs nothing, and the day computeTetBarycentric is
            // re-synced to a version without the internal clamp, this is the backstop.
            const float4 bary = clamp(laneBary[winner], float4(0, 0, 0, 0), float4(1, 1, 1, 1));
            Ptr<uint>(ciBase + 8UL)[0] = encodeSoftBody(softbodyId, laneSimTet[winner]);
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

// ===========================================================================
// P8.4 tranche b — sb_fem_contact_remap_to_simLaunch (softbodyPrimitives.cu:2277, launch
// PxgGpuNarrowphaseCore::softbodyFemContactApplyCollisionToSimMeshMapping,
// PxgNarrowphaseCore.cpp:4166, SB_FEM_CONTACT_REMAP_TO_SIM).
//
// The DEFORMABLE-vs-DEFORMABLE twin of sb_other_contact_remap_to_simLaunch above.  Same job
// -- turn a contact expressed against the COLLISION mesh into a (sim-tet, barycentric) pair
// against the SIMULATION mesh -- but for a contact whose BOTH sides are deformable, so it
// runs twice over the same contact array:
//
//     pairIndex = blockIdx.y;                       // gridDim.y == 2
//     pairIndex == 0 -> remap contactInfo.pairInd1, rewriting outBarycentric1
//     pairIndex == 1 -> remap contactInfo.pairInd0, rewriting outBarycentric0
//
// The barycentric slot is the INPUT as well as the output: before the remap it holds the
// contact POINT in world space (that is exactly what the self-collision midphase stores --
// barycentric0 = the query point, barycentric1 = the closest point), and after it holds real
// barycentrics against the winning sim tet.
//
// Everything the sibling above documents applies here unchanged and is not repeated: the
// `min()` count clamp instead of CUDA's redefine-and-store block, the SELECT-THE-POINTER
// shape for the optional prevContactCount, the 32 per-lane arrays declared inside the
// strided loop, and the LOWEST-lane tie-break that `break` encodes (warpReduction's
// winnerLane is `lowestSetIndex(ballot(best == input))`, reduction.cuh:214).
//
// TWO THINGS THIS ONE HAS THAT THE SIBLING DOES NOT:
//   - an `isValidTet` gate on each candidate, and a warp-wide `hasTet` ballot.  When NO
//     candidate in the whole warp is a valid tet the contact is remapped to tet 0 with a
//     ZERO barycentric, which is upstream's way of neutralising it rather than leaving a
//     stale collision-mesh index behind;
//   - a device pointer selected by pairIndex.  That is the safe direction of the
//     speculated-load trap: both `outBarycentric0` and `outBarycentric1` are valid
//     addresses, so whether llvmpipe speculates is irrelevant.  It is written as a pointer
//     select anyway, to keep every selected pointer in this file the same shape.
//
// *** THE GRID.  READ THIS BEFORE CHANGING SB_REMAP_GRID OR THE EARLY-OUT. ***
// This kernel is launched from THREE call sites with THREE different block shapes
// (PxgNarrowphaseCore.cpp:4149-4163): (8192, 4) for softbody-vs-softbody, (1024, 16) for
// self-collision, (1024, 16) for cloth-vs-softbody.  CUDA copes because its stride is
// `gridDim.x * blockDim.y` and blockDim travels with the launch; the Vulkan bridge DROPS
// blockDim, so one SPIR-V entry point cannot carry three strides.  Getting it wrong is not
// a slowdown: the remap is NOT idempotent -- a second visit reads an ALREADY-REMAPPED sim
// index as though it were a collision index and looks it up in the wrong table -- so an
// over-large dispatch with an under-large stride silently corrupts contacts.
//
// The fix is to pin ONE grid and refuse the rest: the loop uses stride
// SB_REMAP_GRID * SB_REMAP_BLOCK and groups at or above SB_REMAP_GRID return immediately.
// Coverage stays exact and complete for every call site (the surviving 1024 * 64 threads
// partition [startIndex, numContacts) with no gaps and no repeats); only the
// softbody-vs-softbody case loses parallelism it cannot use here anyway.
// ===========================================================================

// isValidTet (cudaNpCommon.h) — strictly positive signed volume.
bool remapIsValidTet(float3 v0, float3 v1, float3 v2, float3 v3)
{
    return dot(v1 - v0, cross(v2 - v0, v3 - v0)) > 1e-9f;
}

// The three writes CUDA makes at each of its three exits, in one place.  pairIndex 1 targets
// pairInd0, which is a PxU64 field taking a PxU32 value -- the high word must be ZEROED, not
// left holding whatever the contact carried before.
void femRemapWrite(uint64_t ciBase, uint64_t baryAddr, uint contactIndex, uint pairIndex,
                   uint newPairInd, float4 bary)
{
    if (pairIndex == 1u)
    {
        Ptr<uint>(ciBase)[0] = newPairInd;
        Ptr<uint>(ciBase + 4UL)[0] = 0u;
    }
    else
    {
        Ptr<uint>(ciBase + 8UL)[0] = newPairInd;
    }
    Ptr<float4>(baryAddr)[contactIndex] = bary;
}

// CUDA arg pack (PxgNarrowphaseCore.cpp:4139): { const PxgSoftBody* softbodies,
// float4* outBarycentric0, float4* outBarycentric1, PxgFemFemContactInfo* outContactInfo,
// PxU32* totalContactCount, PxU32* prevContactCount, const PxU32 maxNbContacts }
// -> six pointers @0..40 and a PxU32 @48 -> 56-byte blob.  (ABI row {8,8,8,8,8,8,4}.)
struct SbFemRemapParams
{
    uint64_t softbodies;        // @0
    uint64_t outBarycentric0;   // @8   in/out
    uint64_t outBarycentric1;   // @16  in/out
    uint64_t outContactInfo;    // @24  in/out
    uint64_t totalContactCount; // @32  in/out
    uint64_t prevContactCount;  // @40  (optional — may be null)
    uint maxNbContacts;         // @48
    uint _pad52;                // @52 -> 56
}
[[vk::push_constant]]
ConstantBuffer<SbFemRemapParams> gFr;

[shader("compute")]
[numthreads(64, 1, 1)]   // == SB_REMAP_BLOCK
void sb_fem_contact_remap_to_simLaunch(uint3 gtid : SV_GroupThreadID, uint3 gid : SV_GroupID)
{
    // See the grid note above: refuse the groups this shader's stride does not account for.
    if (gid.x >= SB_REMAP_GRID)
        return;

    const uint pairIndex = gid.y;   // blockIdx.y

    const uint rawCount = Ptr<uint>(gFr.totalContactCount)[0];
    const uint numContacts = min(rawCount, gFr.maxNbContacts);
    // CUDA guards this with `idxInWarp == 0 && globalWarpIdx == 0`, which is true in BOTH
    // y-grids; the write is the same value from both, so pinning it to pairIndex 0 removes
    // a benign race without changing the result.
    if (rawCount > gFr.maxNbContacts && gid.x == 0u && gtid.x == 0u && pairIndex == 0u)
        Ptr<uint>(gFr.totalContactCount)[0] = gFr.maxNbContacts;

    const uint64_t prevAddr =
        (gFr.prevContactCount != 0UL) ? gFr.prevContactCount : gFr.totalContactCount;
    const uint prevRaw = Ptr<uint>(prevAddr)[0];
    const uint startIndex = (gFr.prevContactCount != 0UL) ? prevRaw : 0u;

    // pairIndex picks BOTH which half of the pair is remapped and which barycentric array
    // carries it.  Both addresses are valid, so this select is the safe kind.
    const uint64_t baryAddr = (pairIndex == 1u) ? gFr.outBarycentric0 : gFr.outBarycentric1;

    Ptr<PxgSoftBodyGpu> softbodies = Ptr<PxgSoftBodyGpu>(gFr.softbodies);
    const uint stride = SB_REMAP_GRID * SB_REMAP_BLOCK;

    for (uint contactIndex = startIndex + gid.x * SB_REMAP_BLOCK + gtid.x;
         contactIndex < numContacts; contactIndex += stride)
    {
        // The barycentric slot holds the contact POINT until this kernel overwrites it.
        const float3 contact = Ptr<float4>(baryAddr)[contactIndex].xyz;

        // PxgFemFemContactInfo (PxgFEMCore.h:51), 16 B: pairInd0 PxU64 @0, pairInd1 PxU32
        // @8, auxInd @12.  Both halves carry a PxU32 encoding, so only the low word is read.
        const uint64_t ciBase = gFr.outContactInfo + uint64_t(contactIndex) * 16UL;
        const uint64_t pairAddr = (pairIndex == 1u) ? ciBase : (ciBase + 8UL);
        const uint pairInd = Ptr<uint>(pairAddr)[0];

        const uint softbodyId = getSoftBodyId(pairInd);
        const uint tetInd = getSoftBodyElementIndex(pairInd);

        Ptr<uint4> simTets = Ptr<uint4>(softbodies[softbodyId].mSimTetIndices);
        Ptr<float4> simVerts = Ptr<float4>(softbodies[softbodyId].mSimPosition_InvMass);
        Ptr<uint> remapColToSim = Ptr<uint>(softbodies[softbodyId].mTetsRemapColToSim);
        Ptr<uint> accumRemap = Ptr<uint>(softbodies[softbodyId].mTetsAccumulatedRemapColToSim);

        const uint endInd = accumRemap[tetInd];
        const uint startInd = (tetInd != 0u) ? accumRemap[tetInd - 1u] : 0u;
        const uint range = endInd - startInd;
        const uint nbIterations = (range + 31u) / 32u;

        uint laneSimTet[32];
        float4 laneBary[32];
        float laneSqDist[32];
        for (uint li = 0u; li < 32u; ++li)
        {
            laneSimTet[li] = 0xffffffffu;
            laneBary[li] = float4(0, 0, 0, 0);
            laneSqDist[li] = PX_MAX_F32_LOCAL;
        }

        bool valid = false;
        bool hasTet = false;   // __ballot_sync(FULL_MASK, hasTet), flattened over the lanes
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

                // The gate the OTHER remap does not have: a degenerate sim tet is no
                // candidate at all, and a contact with no candidate at all is neutralised
                // below rather than left pointing at a collision-mesh index.
                if (!remapIsValidTet(a, b, c, d)) continue;
                hasTet = true;

                float4 tmpBarycentric;
                const bool intersect = computeTetBarycentric(a, b, c, d, contact, tmpBarycentric);
                if (!intersect)
                {
                    const float3 tmpClosest = guClosestPtPointTetrahedron(contact, a, b, c, d);
                    const float3 v = contact - tmpClosest;
                    const float tmpDist = dot(v, v);
                    if (tmpDist < laneSqDist[l])
                    {
                        laneSqDist[l] = tmpDist;
                        laneSimTet[l] = tmpSimTetInd;
                        float4 cb;
                        computeTetBarycentric(a, b, c, d, tmpClosest, cb);
                        laneBary[l] = cb;
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
                femRemapWrite(ciBase, baryAddr, contactIndex, pairIndex,
                              encodeSoftBody(softbodyId, hitTet), hitBary);
                valid = true;
            }
        }

        if (!hasTet)
        {
            // No valid sim tet anywhere in the range: neutralise the contact.
            femRemapWrite(ciBase, baryAddr, contactIndex, pairIndex,
                          encodeSoftBody(softbodyId, 0u), float4(0, 0, 0, 0));
        }
        else if (!valid)
        {
            float best = PX_MAX_F32_LOCAL;
            for (uint l = 0; l < 32u; ++l) best = min(best, laneSqDist[l]);
            uint winner = 0u;
            for (uint l = 0; l < 32u; ++l)
                if (laneSqDist[l] == best) { winner = l; break; }

            femRemapWrite(ciBase, baryAddr, contactIndex, pairIndex,
                          encodeSoftBody(softbodyId, laneSimTet[winner]), laneBary[winner]);
        }
    }
}
