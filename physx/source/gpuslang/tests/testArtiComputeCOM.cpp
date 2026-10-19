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


// P8.112: the `computeArtiCOM` engine entry (inverseDynamic.cu:1019) -- the
// first of the direct-GPU-API inverse-dynamics family: one warp per job over
// the RECORD pool, global atomic float adds into the caller's PxVec3 output.
//
// ATOMIC-ORDER DISCIPLINE: float atomic adds are order-nondeterministic, so
// every summand here is EXACT -- masses are powers of two (invMass = 2^-n),
// link positions are small integers, all partial sums stay below 2^24 --
// which makes the totals bit-identical in EVERY summation order.  The mirror
// then computes them in plain loop order with the engine's own PxTransform /
// PxVec3 arithmetic for the lane-0 tail.
//
// 5.12 passes nbIndices first and guards the job index (the eighth upstream-bug
// candidate of 5.6.1, fixed upstream); the fixture keeps nbElements a multiple
// of 8 regardless.  gpuIndices PERMUTES jobs onto a larger
// record pool with unreferenced decoy records, link counts straddle 32 (the
// warp-stride loop runs twice on some lanes), and two rounds cover
// rootFrame off and on (fresh zeroed outputs each round -- the engine
// contract; a non-zero x would pollute the DIVISOR, which is why the
// fixture seeds the output at exactly zero and the mirror says so).

#include "pxgTestFramework.h"
#include "PxgCudaShim.h"

#include "foundation/PxVec3.h"
#include "foundation/PxQuat.h"
#include "foundation/PxTransform.h"

#include <cstdint>
#include <cstdio>
#include <cstring>
#include <vector>

namespace
{
using namespace pxgtest;
using physx::PxQuat;
using physx::PxTransform;
using physx::PxVec3;
using physx::gpuslang::PxgCudaShim;
using physx::gpuslang::PxgKernelDef;
using physx::gpuslang::PxgDevicePtr;
using physx::gpuslang::PxgResult;

static std::vector<uint8_t> artiComLoader(const char* file) { return loadShaderBlob(file); }

const uint32_t COM_NB_JOBS = 16;     // exact multiple of 8: no overhang warps
const uint32_t COM_NB_RECORDS = 24;  // pool larger than the job set
const uint32_t COM_MAX_LINKS = 40;   // some articulations exceed one warp pass
const uint32_t COM_RECORD_SIZE = 496;
const uint32_t COM_R_DATA_NUM_LINKS = 16;   // PxU16, pinned by the desc map
const uint32_t COM_R_LINK_BODY2WORLDS = 176;
const uint32_t COM_R_LINK_PROPS = 240;

// jobs permute onto the pool; records 1, 4, 7, ... stay unreferenced decoys
static uint32_t comArtiIndex(uint32_t job) { return (job * 5u + 3u) % COM_NB_RECORDS; }
static uint32_t comNumLinks(uint32_t rec) { return 1u + ((rec * 11u + 5u) % COM_MAX_LINKS); }
// mass = 2^(0..6): exact reciprocals, exact sums
static float comInvMass(uint32_t rec, uint32_t link)
{
    return 1.0f / float(1u << ((rec + link) % 7u));
}
static PxVec3 comPos(uint32_t rec, uint32_t link)
{
    return PxVec3(float(int((rec * 17u + link * 13u) % 511u) - 255),
                  float(int((rec * 29u + link * 7u) % 511u) - 255),
                  float(int((rec * 41u + link * 3u) % 511u) - 255));
}
static PxQuat comRootQ(uint32_t rec)
{
    PxQuat q(0.1f + 0.02f * float(rec % 9u), -0.3f + 0.05f * float(rec % 5u),
             0.2f + 0.03f * float(rec % 7u), 0.8f + 0.01f * float(rec % 3u));
    return q.getNormalized();
}

// ---------------------------------------------------------------------------

static size_t comRun(PxgCudaShim& shim, bool rootFrame)
{
    // record pool + the two per-record arrays
    std::vector<uint8_t> recImg(size_t(COM_NB_RECORDS) * COM_RECORD_SIZE, 0xAB);
    std::vector<float> props(size_t(COM_NB_RECORDS) * COM_MAX_LINKS * 4, -99.0f);
    std::vector<float> xforms(size_t(COM_NB_RECORDS) * COM_MAX_LINKS * 7, -88.0f);

    for (uint32_t r = 0; r < COM_NB_RECORDS; ++r)
    {
        const uint32_t nl = comNumLinks(r);
        const uint16_t nl16 = uint16_t(nl);
        std::memcpy(recImg.data() + size_t(r) * COM_RECORD_SIZE + COM_R_DATA_NUM_LINKS, &nl16, 2);
        for (uint32_t l = 0; l < nl; ++l)
        {
            float* pp = &props[(size_t(r) * COM_MAX_LINKS + l) * 4];
            pp[0] = 3.0f; pp[1] = 5.0f; pp[2] = 7.0f;      // decoys in xyz
            pp[3] = comInvMass(r, l);
            float* tt = &xforms[(size_t(r) * COM_MAX_LINKS + l) * 7];
            const PxQuat q = (l == 0) ? comRootQ(r) : PxQuat(physx::PxIdentity);
            tt[0] = q.x; tt[1] = q.y; tt[2] = q.z; tt[3] = q.w;
            const PxVec3 p = comPos(r, l);
            tt[4] = p.x; tt[5] = p.y; tt[6] = p.z;
        }
    }

    std::vector<uint32_t> indices(COM_NB_JOBS);
    for (uint32_t j = 0; j < COM_NB_JOBS; ++j)
        indices[j] = comArtiIndex(j);

    std::vector<float> out(size_t(COM_NB_JOBS) * 3, 0.0f);   // the ZEROED contract

    // the mirror: exact sums in loop order, then the engine tail
    std::vector<float> want(out);
    uint32_t nTwoPass = 0;
    for (uint32_t j = 0; j < COM_NB_JOBS; ++j)
    {
        const uint32_t r = indices[j];
        const uint32_t nl = comNumLinks(r);
        if (nl > 32) ++nTwoPass;
        float totalMass = 0.f;
        for (uint32_t l = 0; l < nl; ++l)
            totalMass += 1.0f / comInvMass(r, l);
        PxVec3 com(0.f);
        for (uint32_t l = 0; l < nl; ++l)
            com += comPos(r, l) * (1.0f / comInvMass(r, l));
        // x carried the mass sum: (0 + M + Cx) - M, exact by construction
        float x = (totalMass + com.x) - totalMass;
        PxVec3 v(x, com.y, com.z);
        v /= totalMass;   // the engine's recip-multiply
        if (rootFrame)
        {
            const PxTransform root(comPos(r, 0), comRootQ(r));
            v = root.getInverse().transform(v);
        }
        want[j * 3 + 0] = v.x;
        want[j * 3 + 1] = v.y;
        want[j * 3 + 2] = v.z;
    }
    PXG_CHECK_MSG(nTwoPass > 2, "no job exceeds one warp pass");

    PxgDevicePtr dRec = 0, dProps = 0, dXf = 0, dIdx = 0, dOut = 0;
    shim.memAlloc(&dRec, recImg.size());
    shim.memAlloc(&dProps, props.size() * 4);
    shim.memAlloc(&dXf, xforms.size() * 4);
    shim.memAlloc(&dIdx, indices.size() * 4);
    shim.memAlloc(&dOut, out.size() * 4);
    if (!dRec || !dProps || !dXf || !dIdx || !dOut)
    {
        std::printf("  NOTE: no BDA -> skip\n");
        return 0;
    }

    // write the two per-record pointers into each record image
    for (uint32_t r = 0; r < COM_NB_RECORDS; ++r)
    {
        const uint64_t pp = uint64_t(dProps) + uint64_t(r) * COM_MAX_LINKS * 16u;
        const uint64_t tp = uint64_t(dXf) + uint64_t(r) * COM_MAX_LINKS * 28u;
        std::memcpy(recImg.data() + size_t(r) * COM_RECORD_SIZE + COM_R_LINK_PROPS, &pp, 8);
        std::memcpy(recImg.data() + size_t(r) * COM_RECORD_SIZE + COM_R_LINK_BODY2WORLDS, &tp, 8);
    }
    shim.memcpyHtoD(dRec, recImg.data(), recImg.size());
    shim.memcpyHtoD(dProps, props.data(), props.size() * 4);
    shim.memcpyHtoD(dXf, xforms.data(), xforms.size() * 4);
    shim.memcpyHtoD(dIdx, indices.data(), indices.size() * 4);
    shim.memcpyHtoD(dOut, out.data(), out.size() * 4);

    struct Params
    {
        uint32_t nbIndices, pad0;   // 5.12: the job guard (ABI row kArtiCom {4,8,8,1,8})
        uint64_t data, gpuIndices;
        uint32_t rootFrameWord, pad;
        uint64_t articulations;
    } prm = { COM_NB_JOBS, 0u, uint64_t(dOut), uint64_t(dIdx), rootFrame ? 1u : 0u, 0u,
              uint64_t(dRec) };
    static_assert(sizeof(Params) == 40, "params blob");
    const uint32_t blocks = (COM_NB_JOBS + 7) / 8;
    PXG_CHECK_MSG(shim.launchKernel(0, blocks, 1, 1, &prm, sizeof(prm)) == PxgResult::Success,
                  "launch computeArtiCOM");
    shim.streamSynchronize();

    std::vector<float> got(out.size());
    shim.memcpyDtoH(got.data(), dOut, got.size() * 4);

    size_t bad = 0;
    const uint32_t* w = reinterpret_cast<const uint32_t*>(want.data());
    const uint32_t* g = reinterpret_cast<const uint32_t*>(got.data());
    for (size_t k = 0; k < want.size(); ++k)
        if (w[k] != g[k])
        {
            if (bad < 10)
                std::printf("MM com rf=%d word %zu  ref %08x (%g)  gpu %08x (%g)\n",
                            rootFrame ? 1 : 0, k, w[k], want[k], g[k], got[k]);
            ++bad;
        }

    shim.memFree(dRec);
    shim.memFree(dProps);
    shim.memFree(dXf);
    shim.memFree(dIdx);
    shim.memFree(dOut);
    return bad;
}

}  // namespace

PXG_TEST(artiComputeCOM)
{
    PxgSlangComputeDevice* dev = testDevice();
    const PxgKernelDef manifest[] = {
        { "npArtiInverseDyn.spv", "computeArtiCOM", 40u, 0 },
    };
    PxgCudaShim shim(dev, manifest, 1, &artiComLoader);

    size_t bad = 0;
    bad += comRun(shim, false);
    bad += comRun(shim, true);
    PXG_CHECK_MSG(bad == 0, "%zu mismatches", bad);
}
