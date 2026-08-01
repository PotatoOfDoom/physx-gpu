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

// ===========================================================================
// softBodyClothSolveBda — sb_solveOutputSCDeltaVLaunch (softBody.cu:1566), on the engine BDA
// ABI.  Step 6 and the LAST of the softbody-vs-cloth chain: it solves the constraints step 5
// prepared and scatters the positional corrections back onto both bodies.
//
// The two branches mirror step 5's, and again clothBC.w chooses: a cloth TRIANGLE spreads its
// correction over three vertices, a cloth VERTEX takes all of it.  The solver body itself is
// the same in both -- tetTriCollisionConstraint and tetVertexCollisionConstraint
// (softBody.cu:650, :717) differ only in how many cloth vertices they weight:
//
//   relDelta   = (cloth side, barycentrically weighted) - (softbody side, ditto)
//   CN         = pen + dot(relDelta, normal)
//   dLambdaN   = max(-CN * denomInv, -lambdaN)      <- CLAMPED so the accumulated lambda can
//                                                      never go negative; that is what stops
//                                                      a contact from PULLING
//   commonVec  = dLambdaN * normal
//
// then, only if there is friction AND the tangential motion is above 1e-14:
//
//   tanDir     = normalise(relDelta - dot(relDelta, normal) * normal)
//   dLambdaT   = -min(max(0, dot(relDelta, tanDir) * denomInv), friction * |dLambdaN|)
//   commonVec += 0.5 * dLambdaT * tanDir            <- the 0.5 is upstream's "arbitrary
//                                                      scaling factor", transcribed as such
//
// The corrections are then `+w * bc * commonVec` on the cloth and `-w * bc * commonVec` on the
// softbody -- OPPOSITE SIGNS, which is the whole point: one body is pushed out of the other.
//
// THE VERTEX PATH IS NOT THE TRIANGLE PATH WITH bc = 1.  Its cloth correction is
// `w0 * commonVec` with NO barycentric factor at all, and its atomic is guarded only by
// `delta.w != 0` where the triangle path also requires `bc > 1e-6`.  Both asymmetries are
// upstream's and both are transcribed.
//
// The `denomInv == 0` case CONTINUES rather than returning -- a degenerate constraint skips
// its own contact and leaves the rest of the loop's contacts alone.
//
// ATOMICS: every scatter is a real float atomic add (VK_EXT_shader_atomic_float), because
// several contacts share vertices.  `AtomicAdd(dst, v, 1.f)` adds v to xyz AND 1 to .w -- the
// .w is a COUNT of contributions that the averaging pass downstream divides by, so dropping it
// silently changes every delta that follows.
//
// GRID: SB_UPDATEROTATION, 1024 blocks of 256 threads, with CUDA's two-level work split
// reproduced exactly.
// ===========================================================================
module softBodyClothSolveBda;

import pxgSlangSoftBodyDesc;
import pxgSlangFEMClothDesc;
import pxgSlangAtomicFloat;    // pxgAtomicAddF4XYZ / pxgAtomicAddF4W

static const uint SC_GRID = 1024u;
static const uint SC_BLOCK = 256u;

static const uint PXG_BITSHIFT_ID = 20u;
static const uint PX_MAX_ELEMENT = 0x000fffffu;
uint getId(uint compressed) { return compressed >> PXG_BITSHIFT_ID; }
uint getElementIndex(uint compressed) { return compressed & PX_MAX_ELEMENT; }

static const uint64_t FFCI_STRIDE = 16UL;

// PxgSoftBodySoftBodyConstraintBlock — block-of-32 SoA, 1792 B.
static const uint64_t SSC_barycentric0  = 0UL;
static const uint64_t SSC_barycentric1  = 512UL;
static const uint64_t SSC_normal_pen    = 1024UL;
static const uint64_t SSC_velMultiplier = 1536UL;
static const uint64_t SSC_friction      = 1664UL;
static const uint64_t SSC_STRIDE        = 1792UL;

// tetTriCollisionConstraint / tetVertexCollisionConstraint constants (softBody.cu:659, :703).
static const float TAN_THRESHOLD = 1.0e-14f;
static const float TAN_SCALE = 0.5f;
static const float BC_EPS = 1e-6f;

// ---------------------------------------------------------------------------
// Push-constant blob — BYTE-IDENTICAL to the CUDA pack (PxgSoftBodyCore.cpp:1693):
// { softbodies*, clothes*, contactInfos*, constraints*, numContacts*, lambdaNs* }
// -> six pointers, 48-byte blob.  ABI row {8,8,8,8,8,8}.
// ---------------------------------------------------------------------------
struct SbClothSolveParams
{
    uint64_t softbodies;    // @0
    uint64_t clothes;       // @8
    uint64_t contactInfos;  // @16
    uint64_t constraints;   // @24
    uint64_t numContacts;   // @32
    uint64_t lambdaNs;      // @40 -> 48
}

[[vk::push_constant]] ConstantBuffer<SbClothSolveParams> gSc;

// The shared solver body.  `relDelta` is already the barycentrically weighted difference, so
// the two callers differ only in how they build it and how they scatter the result.
void solveContact(float3 relDelta, float4 normalPen, float denomInv, float frictionCoefficient,
                  inout float lambdaN, out float3 commonVec)
{
    const float3 normal = normalPen.xyz;
    const float pen = normalPen.w;

    const float CN = pen + dot(relDelta, normal);

    // The clamp against -lambdaN is what keeps the accumulated impulse non-negative, i.e.
    // what stops a contact from pulling the bodies together.
    const float deltaLambdaN = max(-CN * denomInv, -lambdaN);
    lambdaN += deltaLambdaN;

    commonVec = deltaLambdaN * normal;

    float3 tanDir = relDelta - dot(relDelta, normal) * normal;
    const float tanMagSq = dot(tanDir, tanDir);

    if (frictionCoefficient != 0.0f && tanMagSq > TAN_THRESHOLD)
    {
        tanDir = tanDir / sqrt(tanMagSq);
        const float CT = dot(relDelta, tanDir);

        float deltaLambdaT = max(0.0f, CT * denomInv);
        deltaLambdaT = -min(deltaLambdaT, frictionCoefficient * abs(deltaLambdaN));

        commonVec += TAN_SCALE * deltaLambdaT * tanDir;
    }
}

// AtomicAdd(dst, v, 1.f): xyz gets v, .w gets ONE -- the contribution count.
void scatter(uint64_t base, uint index, float3 v)
{
    pxgAtomicAddF4XYZ(base, index, v);
    pxgAtomicAddF4W(base, index, 1.0f);
}

// ===========================================================================
[shader("compute")]
[numthreads(256, 1, 1)]   // == SC_BLOCK == PxgSoftBodyKernelBlockDim::SB_UPDATEROTATION
void sb_solveOutputSCDeltaVLaunch(uint3 gtid : SV_GroupThreadID, uint3 gid : SV_GroupID)
{
    const uint tNumContacts = Ptr<uint>(gSc.numContacts)[0];
    const uint nbBlocksRequired = (tNumContacts + SC_BLOCK - 1u) / SC_BLOCK;
    const uint nbIterationsPerBlock = (nbBlocksRequired + SC_GRID - 1u) / SC_GRID;

    Ptr<PxgSoftBodyGpu> softbodies = Ptr<PxgSoftBodyGpu>(gSc.softbodies);
    Ptr<PxgFEMClothGpu> clothes = Ptr<PxgFEMClothGpu>(gSc.clothes);

    for (uint i = 0u; i < nbIterationsPerBlock; ++i)
    {
        const uint workIndex = i * SC_BLOCK + gtid.x + nbIterationsPerBlock * gid.x * SC_BLOCK;
        if (workIndex >= tNumContacts)
            return;

        const uint64_t blockBase = gSc.constraints + uint64_t(workIndex / 32u) * SSC_STRIDE;
        const uint slot = workIndex & 31u;

        const float denomInv = Ptr<float>(blockBase + SSC_velMultiplier)[slot];
        // CONTINUE, not return: a degenerate constraint skips only its own contact.
        if (denomInv == 0.0f)
            continue;

        const uint64_t ci = gSc.contactInfos + uint64_t(workIndex) * FFCI_STRIDE;
        const uint pairInd0 = Ptr<uint>(ci)[0];          // cloth
        const uint pairInd1 = Ptr<uint>(ci + 8UL)[0];    // softbody

        const float4 clothBC = Ptr<float4>(blockBase + SSC_barycentric0)[slot];
        const float4 softBodyBC = Ptr<float4>(blockBase + SSC_barycentric1)[slot];
        const float4 normalPen = Ptr<float4>(blockBase + SSC_normal_pen)[slot];
        const float frictionCoefficient = Ptr<float>(blockBase + SSC_friction)[slot];

        const uint clothId = getId(pairInd0);
        const uint triId = getElementIndex(pairInd0);    // triangle OR vertex
        const uint softbodyId = getId(pairInd1);
        const uint tetId = getElementIndex(pairInd1);

        const uint4 tetVertId = Ptr<uint4>(softbodies[softbodyId].mSimTetIndices)[tetId];

        const uint64_t clothAccum = clothes[clothId].mAccumulatedDeltaPos;
        const uint64_t sbAccum = softbodies[softbodyId].mSimDeltaPos;
        const uint64_t clothOut = clothes[clothId].mDeltaPos;
        const uint64_t sbOut = softbodies[softbodyId].mSimDelta;

        const float4 sd0 = Ptr<float4>(sbAccum)[tetVertId.x];
        const float4 sd1 = Ptr<float4>(sbAccum)[tetVertId.y];
        const float4 sd2 = Ptr<float4>(sbAccum)[tetVertId.z];
        const float4 sd3 = Ptr<float4>(sbAccum)[tetVertId.w];

        const float3 sbWeighted = softBodyBC.x * sd0.xyz + softBodyBC.y * sd1.xyz +
                                  softBodyBC.z * sd2.xyz + softBodyBC.w * sd3.xyz;

        float lambdaN = Ptr<float>(gSc.lambdaNs)[workIndex];
        float3 commonVec;

        if (clothBC.w == 0.0f)
        {
            // ---- the cloth TRIANGLE path ----------------------------------
            const uint4 triVertId = Ptr<uint4>(clothes[clothId].mTriangleVertexIndices)[triId];
            const float4 cd0 = Ptr<float4>(clothAccum)[triVertId.x];
            const float4 cd1 = Ptr<float4>(clothAccum)[triVertId.y];
            const float4 cd2 = Ptr<float4>(clothAccum)[triVertId.z];

            const float3 relDelta = (clothBC.x * cd0.xyz + clothBC.y * cd1.xyz +
                                     clothBC.z * cd2.xyz) - sbWeighted;

            solveContact(relDelta, normalPen, denomInv, frictionCoefficient, lambdaN, commonVec);

            if (cd0.w != 0.0f && clothBC.x > BC_EPS)
                scatter(clothOut, triVertId.x, cd0.w * clothBC.x * commonVec);
            if (cd1.w != 0.0f && clothBC.y > BC_EPS)
                scatter(clothOut, triVertId.y, cd1.w * clothBC.y * commonVec);
            if (cd2.w != 0.0f && clothBC.z > BC_EPS)
                scatter(clothOut, triVertId.z, cd2.w * clothBC.z * commonVec);
        }
        else
        {
            // ---- the cloth VERTEX path ------------------------------------
            // NOTE: no barycentric factor on the cloth correction, and the atomic is guarded
            // by the inverse mass ALONE.  Not the triangle path with bc = 1.
            const float4 cd0 = Ptr<float4>(clothAccum)[triId];

            const float3 relDelta = cd0.xyz - sbWeighted;

            solveContact(relDelta, normalPen, denomInv, frictionCoefficient, lambdaN, commonVec);

            if (cd0.w != 0.0f)
                scatter(clothOut, triId, cd0.w * commonVec);
        }

        // The softbody side is identical in both branches, and its sign is OPPOSITE.
        if (sd0.w != 0.0f && softBodyBC.x > BC_EPS)
            scatter(sbOut, tetVertId.x, -sd0.w * softBodyBC.x * commonVec);
        if (sd1.w != 0.0f && softBodyBC.y > BC_EPS)
            scatter(sbOut, tetVertId.y, -sd1.w * softBodyBC.y * commonVec);
        if (sd2.w != 0.0f && softBodyBC.z > BC_EPS)
            scatter(sbOut, tetVertId.z, -sd2.w * softBodyBC.z * commonVec);
        if (sd3.w != 0.0f && softBodyBC.w > BC_EPS)
            scatter(sbOut, tetVertId.w, -sd3.w * softBodyBC.w * commonVec);

        Ptr<float>(gSc.lambdaNs)[workIndex] = lambdaN;
    }
}
