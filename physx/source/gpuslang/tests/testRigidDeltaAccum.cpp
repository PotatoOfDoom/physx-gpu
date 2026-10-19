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

// P9.44 -- rigidDeltaAccumBda.slang: the five kernels of rigidDeltaAccum.cu that fold
// the per-contact rigid deltas of the deformable and particle solves into the solver
// bodies.  Host mirrors transcribed from the CUDA, block/warp structure included
// (the float sums are Hillis-Steele scans, so their ASSOCIATION is part of the
// contract and a serial sum would not be the reference):
//
//   * rigidDeltaAccum_firstSecond: accumulateDeltaVRigidFirstLaunch then
//     accumulateDeltaVRigidSecondLaunch on four id lists -- empty, five contacts, 700
//     (a run crossing the block-0/1 boundary), 20000 (two tiles per block, a 3000-long
//     run spanning three blocks, a partial trailing tile that leaves the last block's
//     carry at the sentinel) -- rigid, static and articulation-link ids mixed, under
//     PGS and TGS.  First's scanned deltaV / block carries and Second's solver-body
//     pool, articulation dirty bytes and link scratch impulses are compared WORD FOR
//     WORD (the pool's .w lanes carry negative zeros so the P9.43b fold would show).
//   * rigidDeltaAccum_multiChain: clearDeltaVRigidSecondLaunchMulti ->
//     accumulateDeltaVRigidSecondLaunchMultiStage1 -> Stage2 on a 3000-entry list in
//     which bodies head up to TWO runs (the attachment lists are sorted per buffer
//     only), useLocalRelax off/on, PGS/TGS.  Two float atomics onto a zero base are
//     order-independent (a + b == b + a), so the comparison stays word-exact; bodies
//     with one run keep a non-zero base.

#include "pxgTestFramework.h"

#include <cstdint>
#include <cstdio>
#include <cstring>
#include <vector>

namespace
{
using namespace pxgtest;

struct F4 { float x, y, z, w; };
static F4 add4(F4 a, F4 b) { return F4{ a.x + b.x, a.y + b.y, a.z + b.z, a.w + b.w }; }
// PxMax on float forwards to intrinsics::selectMax (PxMath.h:82 ->
// PxUnixMathIntrinsics.h:122), which is `a > b ? a : b` -- NOT the generic
// template's `a < b ? b : a`.  The two disagree for PxMax(+0, -0), which the
// signed-zero case below pins.
static float pxMax(float a, float b) { return (a > b) ? a : b; }

static const uint32_t AD_BLOCK = 512u;
static const uint32_t AD_GRID = 32u;
static const uint32_t WARP = 32u;
static const uint32_t NWARPS = AD_BLOCK / WARP;
static const uint64_t SENTINEL = 0x8fffffffffffffffull;
static const uint32_t STATIC_NODE = 0xffffffffu;
static const uint64_t STATIC_ID = uint64_t(STATIC_NODE);

// PxgArticulationBlockData / PxgArticulationBlockLinkData (pxgSlangArticulationBlocks.slang).
static const uint32_t AB_BLOCK_SIZE = 7168u;
static const uint32_t AB_mStateDirty = 7040u;            // PxU8[32]
static const uint32_t AB_LINK_SIZE = 29184u;
static const uint32_t AB_LINK_mScratchImpulse = 11776u;  // float4[32] top.xyz,bottom.x @+0, float2[32] bottom.yz @+512
static const uint32_t AB_LINK_mSolverSpatialImpulse = 14080u;  // same block layout (5.12: also accumulated here)
static const uint32_t AB_SVB_BYZ = 512u;
static const uint32_t AB_LINK_mDeltaScale = 20992u;      // float[32]
static const uint8_t ARTI_STATE_HAS_IMPULSES = 2u;       // PxgArtiStateDirtyFlag::eHAS_IMPULSES

// Descriptor mirrors (pxgSlangPrepDesc / pxgSlangSolverDesc / pxgSlangArticulationDesc).
static const uint32_t SZ_PREPREP_DESC = 760u, OFF_SOLVER_BODY_INDICES = 272u;
static const uint32_t SZ_CORE_DESC = 320u, OFF_NUM_SOLVER_BODIES = 268u, OFF_ACC_OFFSET = 276u;
static const uint32_t SZ_SHARED_PGS = 224u, OFF_VEL_POOL = 160u;
static const uint32_t SZ_ARTI_CORE_DESC = 352u, OFF_NB_ARTI = 44u, OFF_ARTI_BLOCKS = 128u,
                      OFF_LINK_BLOCKS = 136u, OFF_MAX_LINKS = 248u;

static uint64_t rigidNode(uint32_t i) { return uint64_t(i); }
static uint64_t artiNode(uint32_t i, uint32_t link) { return uint64_t(i) | (uint64_t((link << 1) | 1u) << 32); }
static bool nodeIsArti(uint64_t id) { return (uint32_t(id >> 32) & 1u) != 0u; }
static uint32_t nodeIndex(uint64_t id) { return uint32_t(id & 0xffffffffull); }
static uint32_t nodeLink(uint64_t id) { return uint32_t(id >> 32) >> 1; }

struct Rng
{
    uint32_t s;
    uint32_t next() { s ^= s << 13; s ^= s >> 17; s ^= s << 5; return s; }
    uint32_t below(uint32_t n) { return next() % n; }
    float uni(float lo, float hi) { return lo + (hi - lo) * float(next() & 0xffffffu) / 16777216.0f; }
    // mixed magnitudes so that the association of the sums matters
    float val() { const int e = int(below(6)) - 3; float m = uni(-1.0f, 1.0f); for (int k = 0; k < e; ++k) m *= 10.0f; for (int k = 0; k > e; --k) m *= 0.1f; return m; }
};

static uint64_t addrOf(PxgSlangComputeDevice* dev, PxgBufferHandle h) { return dev->getBufferDeviceAddress(h); }
static void putU64(std::vector<uint8_t>& b, size_t off, uint64_t v) { std::memcpy(b.data() + off, &v, 8); }
static void putU32(std::vector<uint8_t>& b, size_t off, uint32_t v) { std::memcpy(b.data() + off, &v, 4); }
static void putF32(std::vector<uint8_t>& b, size_t off, float v) { std::memcpy(b.data() + off, &v, 4); }
static float getF32(const std::vector<uint8_t>& b, size_t off) { float v; std::memcpy(&v, b.data() + off, 4); return v; }

// ---------------------------------------------------------------------------
// The world both kernels see.
// ---------------------------------------------------------------------------
struct World
{
    uint32_t n = 0;
    std::vector<uint64_t> ids;        // sorted-by-run PxNodeIndex list
    std::vector<F4> deltaV;           // 2n: linear [0,n), angular [n,2n)
    // rigid bodies
    uint32_t N = 0, acc = 0;
    std::vector<F4> pool;             // acc + 3N float4
    std::vector<uint32_t> solverBodyIndices;
    std::vector<float> tempDenom;     // N
    // articulations
    uint32_t nbArtiBlocks = 0, maxLinks = 0;
    std::vector<uint8_t> artiBlocks;  // nbArtiBlocks * AB_BLOCK_SIZE
    std::vector<uint8_t> linkBlocks;  // nbArtiBlocks * maxLinks * AB_LINK_SIZE
};

// ---------------------------------------------------------------------------
// accumulateDeltaVRigidFirstLaunch (rigidDeltaAccum.cu:55-220), block by block.
// ---------------------------------------------------------------------------
struct FirstOut
{
    std::vector<F4> deltaV;
    std::vector<F4> blockDeltaV;        // 2 * AD_GRID
    std::vector<uint64_t> blockRigidId; // AD_GRID
};

static FirstOut refFirst(const World& W)
{
    FirstOut o;
    o.deltaV = W.deltaV;
    o.blockDeltaV.assign(2 * AD_GRID, F4{ 0.0f, 0.0f, 0.0f, 0.0f });
    o.blockRigidId.assign(AD_GRID, SENTINEL);
    const uint32_t n = W.n;
    const uint32_t nbBlocksRequired = (n + AD_BLOCK - 1) / AD_BLOCK;
    const uint32_t nbIter = (nbBlocksRequired + AD_GRID - 1) / AD_GRID;

    for (uint32_t b = 0; b < AD_GRID; ++b)
    {
        F4 linBlock{ 0.0f, 0.0f, 0.0f, 0.0f }, angBlock{ 0.0f, 0.0f, 0.0f, 0.0f };
        uint64_t blockRid = SENTINEL;
        for (uint32_t i = 0; i < nbIter; ++i)
        {
            std::vector<F4> lin(AD_BLOCK, F4{ 0.0f, 0.0f, 0.0f, 0.0f }), ang(AD_BLOCK, F4{ 0.0f, 0.0f, 0.0f, 0.0f });
            std::vector<uint64_t> rid(AD_BLOCK, SENTINEL);
            std::vector<bool> valid(AD_BLOCK, false);
            for (uint32_t t = 0; t < AD_BLOCK; ++t)
            {
                const uint32_t w = AD_BLOCK * (b * nbIter + i) + t;
                valid[t] = w < n;
                if (valid[t]) { rid[t] = W.ids[w]; lin[t] = W.deltaV[w]; ang[t] = W.deltaV[w + n]; }
            }
            // .cu:110-125 warp-level segmented inclusive scan (Hillis-Steele)
            for (uint32_t r = 1; r < WARP; r <<= 1)
            {
                const std::vector<F4> lin0 = lin, ang0 = ang;
                for (uint32_t t = 0; t < AD_BLOCK; ++t)
                {
                    const uint32_t lane = t & (WARP - 1);
                    if (lane >= r && valid[t] && rid[t] == rid[t - r])
                    {
                        lin[t] = add4(lin0[t], lin0[t - r]);
                        ang[t] = add4(ang0[t], ang0[t - r]);
                    }
                }
            }
            F4 wLin[NWARPS], wAng[NWARPS];
            uint64_t wRid[NWARPS];
            for (uint32_t w = 0; w < NWARPS; ++w)
            {
                wLin[w] = lin[w * WARP + WARP - 1];
                wAng[w] = ang[w * WARP + WARP - 1];
                wRid[w] = rid[w * WARP + WARP - 1];
            }
            const F4 prevLin = linBlock, prevAng = angBlock;
            const uint64_t prevRid = blockRid;
            // .cu:157-168 the 16-wide scan of the warp totals
            F4 tLin[NWARPS], tAng[NWARPS];
            for (uint32_t w = 0; w < NWARPS; ++w) { tLin[w] = wLin[w]; tAng[w] = wAng[w]; }
            for (uint32_t r = 1; r < NWARPS; r <<= 1)
            {
                F4 l0[NWARPS], a0[NWARPS];
                for (uint32_t w = 0; w < NWARPS; ++w) { l0[w] = tLin[w]; a0[w] = tAng[w]; }
                for (uint32_t w = 0; w < NWARPS; ++w)
                    if (w >= r && wRid[w] == wRid[w - r]) { tLin[w] = add4(l0[w], l0[w - r]); tAng[w] = add4(a0[w], a0[w - r]); }
            }
            // .cu:170-182 lane 15 folds the tile's trailing run into the carry
            if (blockRid != wRid[NWARPS - 1]) { linBlock = F4{ 0.0f, 0.0f, 0.0f, 0.0f }; angBlock = F4{ 0.0f, 0.0f, 0.0f, 0.0f }; }
            linBlock = add4(linBlock, tLin[NWARPS - 1]);
            angBlock = add4(angBlock, tAng[NWARPS - 1]);
            blockRid = wRid[NWARPS - 1];
            // .cu:189-211
            for (uint32_t t = 0; t < AD_BLOCK; ++t)
            {
                if (!valid[t]) continue;
                const uint32_t w = AD_BLOCK * (b * nbIter + i) + t;
                const uint32_t warp = t / WARP;
                F4 accLin{ 0.0f, 0.0f, 0.0f, 0.0f }, accAng{ 0.0f, 0.0f, 0.0f, 0.0f };
                if (warp > 0 && rid[t] == wRid[warp - 1]) { accLin = tLin[warp - 1]; accAng = tAng[warp - 1]; }
                if (i != 0 && rid[t] == prevRid) { accLin = add4(accLin, prevLin); accAng = add4(accAng, prevAng); }
                o.deltaV[w] = add4(lin[t], accLin);
                o.deltaV[w + n] = add4(ang[t], accAng);
            }
        }
        o.blockDeltaV[b] = linBlock;
        o.blockDeltaV[b + AD_GRID] = angBlock;
        o.blockRigidId[b] = blockRid;
    }
    return o;
}

// .cu:247-289: the 32-wide segmented scan of the block carries.
static void refScanCarries(const FirstOut& f, std::vector<F4>& tLin, std::vector<F4>& tAng)
{
    tLin.assign(AD_GRID, F4{}); tAng.assign(AD_GRID, F4{});
    for (uint32_t b = 0; b < AD_GRID; ++b) { tLin[b] = f.blockDeltaV[b]; tAng[b] = f.blockDeltaV[b + AD_GRID]; }
    for (uint32_t r = 1; r < AD_GRID; r <<= 1)
    {
        const std::vector<F4> l0 = tLin, a0 = tAng;
        for (uint32_t b = 0; b < AD_GRID; ++b)
            if (b >= r && f.blockRigidId[b] == f.blockRigidId[b - r]) { tLin[b] = add4(l0[b], l0[b - r]); tAng[b] = add4(a0[b], a0[b - r]); }
    }
}

// Which block's partition holds contact w (.cu:304 inverted).
static uint32_t blockOf(uint32_t w, uint32_t n)
{
    const uint32_t nbBlocksRequired = (n + AD_BLOCK - 1) / AD_BLOCK;
    const uint32_t nbIter = (nbBlocksRequired + AD_GRID - 1) / AD_GRID;
    return (w / AD_BLOCK) / nbIter;
}

static void writeSpatialBlock(std::vector<uint8_t>& linkBlocks, size_t linkBase, uint32_t field, uint32_t aib,
                              const float top[3], const float bottom[3], bool accumulate)
{
    const size_t f4 = linkBase + field + size_t(aib) * 16;
    const size_t f2 = linkBase + field + AB_SVB_BYZ + size_t(aib) * 8;
    const float vals4[4] = { top[0], top[1], top[2], bottom[0] };
    const float vals2[2] = { bottom[1], bottom[2] };
    for (int k = 0; k < 4; ++k) putF32(linkBlocks, f4 + 4 * k, accumulate ? getF32(linkBlocks, f4 + 4 * k) + vals4[k] : vals4[k]);
    for (int k = 0; k < 2; ++k) putF32(linkBlocks, f2 + 4 * k, accumulate ? getF32(linkBlocks, f2 + 4 * k) + vals2[k] : vals2[k]);
}

// 5.12: the link impulse goes to mScratchImpulse AND mSolverSpatialImpulse
static void writeScratchImpulse(std::vector<uint8_t>& linkBlocks, size_t linkBase, uint32_t aib, const float top[3], const float bottom[3], bool accumulate)
{
    writeSpatialBlock(linkBlocks, linkBase, AB_LINK_mScratchImpulse, aib, top, bottom, accumulate);
    writeSpatialBlock(linkBlocks, linkBase, AB_LINK_mSolverSpatialImpulse, aib, top, bottom, accumulate);
}

// ---------------------------------------------------------------------------
// accumulateDeltaVRigidSecondLaunch (rigidDeltaAccum.cu, 5.12) on the World.
// ---------------------------------------------------------------------------
static void refSecond(World& W, const FirstOut& f, bool isTGS)
{
    std::vector<F4> tLin, tAng;
    refScanCarries(f, tLin, tAng);
    const uint32_t n = W.n;
    for (uint32_t w = 0; w < n; ++w)
    {
        const uint64_t id = W.ids[w];
        const bool isLast = (w == n - 1) || W.ids[w + 1] != id;
        if (!isLast) continue;
        F4 linVel = f.deltaV[w], angVel = f.deltaV[w + n];
        const uint32_t b = blockOf(w, n);
        const uint64_t pre = b > 0 ? f.blockRigidId[b - 1] : SENTINEL;
        if (id == pre) { linVel = add4(linVel, tLin[b - 1]); angVel = add4(angVel, tAng[b - 1]); }
        const uint32_t node = nodeIndex(id);
        if (node == STATIC_NODE) continue;
        const uint32_t sbi = W.solverBodyIndices[node];
        if (nodeIsArti(id))
        {
            const uint32_t blockIndex = sbi / WARP, aib = sbi % WARP;
            W.artiBlocks[size_t(blockIndex) * AB_BLOCK_SIZE + AB_mStateDirty + aib] = ARTI_STATE_HAS_IMPULSES;
            const uint32_t linkID = nodeLink(id);
            const size_t linkBase = size_t(blockIndex * W.maxLinks + linkID) * AB_LINK_SIZE;
            const float denom = pxMax(1.0f, linVel.w);
            const float ratio = 1.0f / denom;
            const float top[3] = { -(linVel.x * ratio), -(linVel.y * ratio), -(linVel.z * ratio) };
            const float bottom[3] = { -(angVel.x * ratio), -(angVel.y * ratio), -(angVel.z * ratio) };
            // 5.12: addSpatialVector into both blocks (5.6.1: storeSpatialVector into mScratchImpulse)
            writeScratchImpulse(W.linkBlocks, linkBase, aib, top, bottom, true);
        }
        else
        {
            F4 lv = W.pool[W.acc + sbi], av = W.pool[W.acc + sbi + W.N];
            const float denom = pxMax(1.0f, linVel.w);
            const float ratio = 1.0f / denom;
            linVel.w = 0.0f;
            if (isTGS)
            {
                lv.x += linVel.x * ratio; lv.y += linVel.y * ratio; lv.z += linVel.z * ratio;
                lv.w += angVel.x * ratio; av.x += angVel.y * ratio; av.y += angVel.z * ratio;
            }
            else
            {
                lv.x += linVel.x * ratio; lv.y += linVel.y * ratio; lv.z += linVel.z * ratio; lv.w += linVel.w * ratio;
                av.x += angVel.x * ratio; av.y += angVel.y * ratio; av.z += angVel.z * ratio; av.w += angVel.w * ratio;
            }
            W.pool[W.acc + sbi] = lv;
            W.pool[W.acc + sbi + W.N] = av;
        }
    }
}

// ---------------------------------------------------------------------------
// The Multi trio (rigidDeltaAccum.cu:433-874).  Runs of one body are visited in
// list order; with at most two runs per body onto a zero base the float sums are
// order-independent, which is what keeps the comparison word-exact.
// ---------------------------------------------------------------------------
static void refClearMulti(World& W)
{
    for (uint32_t w = 0; w < W.n; ++w)
    {
        const uint64_t id = W.ids[w];
        if (!((w == W.n - 1) || W.ids[w + 1] != id)) continue;
        const uint32_t node = nodeIndex(id);
        if (node == STATIC_NODE) continue;
        const uint32_t sbi = W.solverBodyIndices[node];
        if (nodeIsArti(id))
        {
            const uint32_t blockIndex = sbi / WARP, aib = sbi % WARP;
            const size_t linkBase = size_t(blockIndex * W.maxLinks + nodeLink(id)) * AB_LINK_SIZE;
            putF32(W.linkBlocks, linkBase + AB_LINK_mDeltaScale + size_t(aib) * 4, 0.0f);
        }
        else
            W.tempDenom[sbi] = 0.0f;
    }
}

static void refStage1(World& W, const FirstOut& f, bool useLocalRelax, float globalRelax)
{
    std::vector<F4> tLin, tAng;
    refScanCarries(f, tLin, tAng);
    for (uint32_t w = 0; w < W.n; ++w)
    {
        const uint64_t id = W.ids[w];
        if (!((w == W.n - 1) || W.ids[w + 1] != id)) continue;
        float linVelW = f.deltaV[w].w;
        const uint32_t b = blockOf(w, W.n);
        const uint64_t pre = b > 0 ? f.blockRigidId[b - 1] : SENTINEL;
        if (id == pre) linVelW += tLin[b - 1].w;
        const uint32_t node = nodeIndex(id);
        if (node == STATIC_NODE) continue;
        const uint32_t sbi = W.solverBodyIndices[node];
        float denom = globalRelax;
        if (useLocalRelax) denom = pxMax(denom, linVelW);
        if (nodeIsArti(id))
        {
            const uint32_t blockIndex = sbi / WARP, aib = sbi % WARP;
            const size_t off = size_t(blockIndex * W.maxLinks + nodeLink(id)) * AB_LINK_SIZE + AB_LINK_mDeltaScale + size_t(aib) * 4;
            putF32(W.linkBlocks, off, getF32(W.linkBlocks, off) + denom);
        }
        else
            W.tempDenom[sbi] = W.tempDenom[sbi] + denom;
    }
}

static void refStage2(World& W, const FirstOut& f, bool isTGS)
{
    std::vector<F4> tLin, tAng;
    refScanCarries(f, tLin, tAng);
    for (uint32_t w = 0; w < W.n; ++w)
    {
        const uint64_t id = W.ids[w];
        if (!((w == W.n - 1) || W.ids[w + 1] != id)) continue;
        F4 linVel = f.deltaV[w], angVel = f.deltaV[w + W.n];
        const uint32_t b = blockOf(w, W.n);
        const uint64_t pre = b > 0 ? f.blockRigidId[b - 1] : SENTINEL;
        if (id == pre) { linVel = add4(linVel, tLin[b - 1]); angVel = add4(angVel, tAng[b - 1]); }
        const uint32_t node = nodeIndex(id);
        if (node == STATIC_NODE) continue;
        const uint32_t sbi = W.solverBodyIndices[node];
        if (nodeIsArti(id))
        {
            const uint32_t blockIndex = sbi / WARP, aib = sbi % WARP;
            W.artiBlocks[size_t(blockIndex) * AB_BLOCK_SIZE + AB_mStateDirty + aib] = ARTI_STATE_HAS_IMPULSES;
            const size_t linkBase = size_t(blockIndex * W.maxLinks + nodeLink(id)) * AB_LINK_SIZE;
            const float denom = getF32(W.linkBlocks, linkBase + AB_LINK_mDeltaScale + size_t(aib) * 4);
            const float ratio = 1.0f / denom;
            const float top[3] = { -(linVel.x * ratio), -(linVel.y * ratio), -(linVel.z * ratio) };
            const float bottom[3] = { -(angVel.x * ratio), -(angVel.y * ratio), -(angVel.z * ratio) };
            writeScratchImpulse(W.linkBlocks, linkBase, aib, top, bottom, true);
        }
        else
        {
            const float ratio = 1.0f / W.tempDenom[sbi];
            F4& lv = W.pool[W.acc + sbi];
            F4& av = W.pool[W.acc + sbi + W.N];
            if (isTGS)
            {
                lv.x += linVel.x * ratio; lv.y += linVel.y * ratio; lv.z += linVel.z * ratio;
                lv.w += angVel.x * ratio; av.x += angVel.y * ratio; av.y += angVel.z * ratio;
            }
            else
            {
                lv.x += linVel.x * ratio; lv.y += linVel.y * ratio; lv.z += linVel.z * ratio;
                av.x += angVel.x * ratio; av.y += angVel.y * ratio; av.z += angVel.z * ratio;
            }
        }
    }
}

// ---------------------------------------------------------------------------
// World construction.
// ---------------------------------------------------------------------------
static const uint32_t NUM_RIGID_NODES = 700u;   // node r -> solver body 1 + r
static const uint32_t NB_ARTI_BLOCKS = 2u, MAX_LINKS = 4u;
static const uint32_t NUM_ARTI = NB_ARTI_BLOCKS * 32u;   // node NUM_RIGID_NODES + g -> articulation g

static void buildBodies(World& W, Rng& rng)
{
    W.N = NUM_RIGID_NODES + 2u;   // solver body 0 is the static slot, 1..700 the rigids, one spare
    W.acc = 7u;
    W.pool.resize(W.acc + 3u * W.N);
    for (size_t k = 0; k < W.pool.size(); ++k)
        W.pool[k] = F4{ rng.val(), rng.val(), rng.val(), rng.val() };
    // negative zeros in the .w lanes of every fifth body's linear and angular entries
    for (uint32_t sbi = 0; sbi < W.N; sbi += 5)
    {
        W.pool[W.acc + sbi].w = -0.0f;
        W.pool[W.acc + sbi + W.N].w = -0.0f;
    }
    W.solverBodyIndices.assign(NUM_RIGID_NODES + NUM_ARTI, 0u);
    for (uint32_t r = 0; r < NUM_RIGID_NODES; ++r) W.solverBodyIndices[r] = 1u + r;
    for (uint32_t g = 0; g < NUM_ARTI; ++g) W.solverBodyIndices[NUM_RIGID_NODES + g] = g;
    W.tempDenom.assign(W.N, 123.5f);
    W.nbArtiBlocks = NB_ARTI_BLOCKS;
    W.maxLinks = MAX_LINKS;
    W.artiBlocks.assign(size_t(NB_ARTI_BLOCKS) * AB_BLOCK_SIZE, 0x55u);
    W.linkBlocks.resize(size_t(NB_ARTI_BLOCKS) * MAX_LINKS * AB_LINK_SIZE);
    for (size_t k = 0; k < W.linkBlocks.size(); k += 4) putF32(W.linkBlocks, k, rng.val());
}

// The distinct id pool: every rigid node, every (articulation, link), one static id.
static std::vector<uint64_t> idPool(Rng& rng)
{
    std::vector<uint64_t> pool;
    for (uint32_t r = 0; r < NUM_RIGID_NODES; ++r) pool.push_back(rigidNode(r));
    for (uint32_t g = 0; g < NUM_ARTI; ++g)
        for (uint32_t l = 0; l < MAX_LINKS; ++l) pool.push_back(artiNode(NUM_RIGID_NODES + g, l));
    pool.push_back(STATIC_ID);
    for (size_t k = pool.size(); k > 1; --k) std::swap(pool[k - 1], pool[rng.below(uint32_t(k))]);
    return pool;
}

// Runs of random length, every id used once (Second's "sorted" premise).
static void fillRuns(World& W, Rng& rng, uint32_t n, uint32_t maxRun, uint32_t giantAt, uint32_t giantLen)
{
    W.n = n;
    W.ids.clear();
    const std::vector<uint64_t> pool = idPool(rng);
    size_t k = 0;
    uint32_t run = 0;
    while (W.ids.size() < n)
    {
        PXG_CHECK_MSG(k < pool.size(), "id pool exhausted (n=%u)", n);
        if (k >= pool.size()) break;
        const uint64_t id = pool[k++];
        uint32_t len = 1u + rng.below(maxRun);
        if (run == giantAt) len = giantLen;
        ++run;
        while (len-- && W.ids.size() < n) W.ids.push_back(id);
    }
    W.deltaV.resize(2u * n);
    for (uint32_t w = 0; w < 2u * n; ++w) W.deltaV[w] = F4{ rng.val(), rng.val(), rng.val(), rng.uni(0.0f, 3.0f) };
}

// Runs where a body may appear TWICE (the attachment lists), never adjacently.
static void fillMultiRuns(World& W, Rng& rng, uint32_t n, uint32_t maxRun, std::vector<uint32_t>& runsPerId)
{
    W.n = n;
    W.ids.clear();
    std::vector<uint64_t> pool = idPool(rng);
    pool.insert(pool.end(), pool.begin(), pool.end());
    for (size_t k = pool.size(); k > 1; --k) std::swap(pool[k - 1], pool[rng.below(uint32_t(k))]);
    size_t k = 0;
    uint64_t prev = SENTINEL;
    while (W.ids.size() < n)
    {
        PXG_CHECK_MSG(k < pool.size(), "id pool exhausted (n=%u)", n);
        if (k >= pool.size()) break;
        const uint64_t id = pool[k++];
        if (id == prev) continue;
        prev = id;
        uint32_t len = 1u + rng.below(maxRun);
        while (len-- && W.ids.size() < n) W.ids.push_back(id);
    }
    W.deltaV.resize(2u * n);
    for (uint32_t w = 0; w < 2u * n; ++w) W.deltaV[w] = F4{ rng.val(), rng.val(), rng.val(), rng.uni(0.0f, 3.0f) };
    // how many runs each id heads (indexed by the position in the sorted W.ids, via a map on the id)
    runsPerId.assign(NUM_RIGID_NODES + NUM_ARTI * MAX_LINKS + 1u, 0u);
    for (uint32_t w = 0; w < n; ++w)
    {
        if (w > 0 && W.ids[w] == W.ids[w - 1]) continue;
        const uint64_t id = W.ids[w];
        uint32_t slot;
        if (nodeIndex(id) == STATIC_NODE) slot = NUM_RIGID_NODES + NUM_ARTI * MAX_LINKS;
        else if (nodeIsArti(id)) slot = NUM_RIGID_NODES + (nodeIndex(id) - NUM_RIGID_NODES) * MAX_LINKS + nodeLink(id);
        else slot = nodeIndex(id);
        runsPerId[slot]++;
    }
}

// ---------------------------------------------------------------------------
// GPU side
// ---------------------------------------------------------------------------
struct FirstPush { uint64_t sortedRigidIds, numContacts, deltaV, blockDeltaV, blockRigidId; };
static_assert(sizeof(FirstPush) == 40, "First push 40 bytes");
struct SecondPush
{
    uint64_t sortedRigidIds, numContacts, deltaV, blockDeltaV, blockRigidId, prePrepDesc, solverCoreDesc, artiCoreDesc,
        solverBodyVelPool;   // 5.12: float4* (5.6.1: PxgSolverSharedDesc<IterativeSolveData>*)
    uint32_t isTGSWord; uint32_t pad;
};
static_assert(sizeof(SecondPush) == 80, "Second push 80 bytes");
struct ClearPush { uint64_t sortedRigidIds, numContacts, prePrepDesc, solverCoreDesc, artiCoreDesc, tempDenom; };
static_assert(sizeof(ClearPush) == 48, "clear push 48 bytes");
struct MultiPush
{
    uint64_t sortedRigidIds, numContacts, deltaV, blockDeltaV, blockRigidId, prePrepDesc, solverCoreDesc, artiCoreDesc, sharedDesc, tempDenom;
    uint32_t useLocalRelaxWord; float globalRelaxationCoefficient; uint32_t isTGSWord; uint32_t pad;
};
static_assert(sizeof(MultiPush) == 96, "Multi push 96 bytes");

struct Gpu
{
    PxgSlangComputeDevice* dev;
    PxgBufferHandle bIds, bNum, bDeltaV, bBlockDeltaV, bBlockRid, bPool, bSbi, bTempDenom, bArtiBlocks, bLinkBlocks;
    PxgBufferHandle bPrePrep, bCore, bShared, bArtiCore;
    std::vector<PxgBufferHandle> all;

    bool upload(const World& W)
    {
        std::vector<uint64_t> ids = W.ids;
        if (ids.empty()) ids.push_back(SENTINEL);   // a non-null pointer for the empty case
        std::vector<F4> dv = W.deltaV;
        if (dv.empty()) dv.push_back(F4{});
        bIds = createBufferFrom(dev, ids);
        bNum = createBufferFrom(dev, std::vector<uint32_t>(1, W.n));
        bDeltaV = createBufferFrom(dev, dv);
        bBlockDeltaV = createBufferFrom(dev, std::vector<F4>(2 * AD_GRID, F4{ 9.0f, 9.0f, 9.0f, 9.0f }));
        bBlockRid = createBufferFrom(dev, std::vector<uint64_t>(AD_GRID, 0x1234u));
        bPool = createBufferFrom(dev, W.pool);
        bSbi = createBufferFrom(dev, W.solverBodyIndices);
        bTempDenom = createBufferFrom(dev, W.tempDenom);
        bArtiBlocks = createBufferFrom(dev, W.artiBlocks);
        bLinkBlocks = createBufferFrom(dev, W.linkBlocks);
        all = { bIds, bNum, bDeltaV, bBlockDeltaV, bBlockRid, bPool, bSbi, bTempDenom, bArtiBlocks, bLinkBlocks };
        if (!addrOf(dev, bIds))
        {
            std::printf("  NOTE: no BDA -> skip\n");
            destroy();
            return false;
        }
        std::vector<uint8_t> prePrep(SZ_PREPREP_DESC, 0);
        putU64(prePrep, OFF_SOLVER_BODY_INDICES, addrOf(dev, bSbi));
        std::vector<uint8_t> core(SZ_CORE_DESC, 0);
        putU32(core, OFF_NUM_SOLVER_BODIES, W.N);
        putU32(core, OFF_ACC_OFFSET, W.acc);
        std::vector<uint8_t> shared(SZ_SHARED_PGS, 0);
        putU64(shared, OFF_VEL_POOL, addrOf(dev, bPool));
        std::vector<uint8_t> arti(SZ_ARTI_CORE_DESC, 0);
        putU32(arti, OFF_NB_ARTI, NUM_ARTI);
        putU64(arti, OFF_ARTI_BLOCKS, addrOf(dev, bArtiBlocks));
        putU64(arti, OFF_LINK_BLOCKS, addrOf(dev, bLinkBlocks));
        putU32(arti, OFF_MAX_LINKS, W.maxLinks);
        bPrePrep = createBufferFrom(dev, prePrep);
        bCore = createBufferFrom(dev, core);
        bShared = createBufferFrom(dev, shared);
        bArtiCore = createBufferFrom(dev, arti);
        all.insert(all.end(), { bPrePrep, bCore, bShared, bArtiCore });
        return true;
    }
    void destroy() { for (PxgBufferHandle h : all) dev->destroyBuffer(h); all.clear(); }

    void runFirst()
    {
        FirstPush p{};
        p.sortedRigidIds = addrOf(dev, bIds); p.numContacts = addrOf(dev, bNum); p.deltaV = addrOf(dev, bDeltaV);
        p.blockDeltaV = addrOf(dev, bBlockDeltaV); p.blockRigidId = addrOf(dev, bBlockRid);
        Kernel k("rigidDeltaAccumBda.spv", "accumulateDeltaVRigidFirstLaunch", {}, sizeof(FirstPush));
        k.dispatch(AD_GRID, {}, p);
        dev->flush();
    }
    void runSecond(bool isTGS)
    {
        SecondPush p{};
        p.sortedRigidIds = addrOf(dev, bIds); p.numContacts = addrOf(dev, bNum); p.deltaV = addrOf(dev, bDeltaV);
        p.blockDeltaV = addrOf(dev, bBlockDeltaV); p.blockRigidId = addrOf(dev, bBlockRid);
        p.prePrepDesc = addrOf(dev, bPrePrep); p.solverCoreDesc = addrOf(dev, bCore); p.artiCoreDesc = addrOf(dev, bArtiCore);
        p.solverBodyVelPool = addrOf(dev, bPool); p.isTGSWord = isTGS ? 1u : 0u;
        Kernel k("rigidDeltaAccumBda.spv", "accumulateDeltaVRigidSecondLaunch", {}, sizeof(SecondPush));
        k.dispatch(AD_GRID, {}, p);
        dev->flush();
    }
    void runClear()
    {
        ClearPush p{};
        p.sortedRigidIds = addrOf(dev, bIds); p.numContacts = addrOf(dev, bNum); p.prePrepDesc = addrOf(dev, bPrePrep);
        p.solverCoreDesc = addrOf(dev, bCore); p.artiCoreDesc = addrOf(dev, bArtiCore); p.tempDenom = addrOf(dev, bTempDenom);
        Kernel k("rigidDeltaAccumBda.spv", "clearDeltaVRigidSecondLaunchMulti", {}, sizeof(ClearPush));
        k.dispatch(AD_GRID, {}, p);
        dev->flush();
    }
    void runMulti(const char* entry, bool useLocalRelax, float globalRelax, bool isTGS)
    {
        MultiPush p{};
        p.sortedRigidIds = addrOf(dev, bIds); p.numContacts = addrOf(dev, bNum); p.deltaV = addrOf(dev, bDeltaV);
        p.blockDeltaV = addrOf(dev, bBlockDeltaV); p.blockRigidId = addrOf(dev, bBlockRid);
        p.prePrepDesc = addrOf(dev, bPrePrep); p.solverCoreDesc = addrOf(dev, bCore); p.artiCoreDesc = addrOf(dev, bArtiCore);
        p.sharedDesc = addrOf(dev, bShared); p.tempDenom = addrOf(dev, bTempDenom);
        p.useLocalRelaxWord = useLocalRelax ? 1u : 0u; p.globalRelaxationCoefficient = globalRelax; p.isTGSWord = isTGS ? 1u : 0u;
        Kernel k("rigidDeltaAccumBda.spv", entry, {}, sizeof(MultiPush));
        k.dispatch(AD_GRID, {}, p);
        dev->flush();
    }
};

// ---------------------------------------------------------------------------
// Comparators (word-exact).
// ---------------------------------------------------------------------------
static uint32_t cmpF4(const char* what, const std::vector<F4>& gpu, const std::vector<F4>& ref, uint32_t& printed)
{
    uint32_t bad = 0;
    for (size_t k = 0; k < ref.size(); ++k)
    {
        if (std::memcmp(&gpu[k], &ref[k], sizeof(F4)) == 0) continue;
        ++bad;
        if (printed < 8)
        {
            ++printed;
            std::printf("    %s[%zu] gpu(%.9g,%.9g,%.9g,%.9g) ref(%.9g,%.9g,%.9g,%.9g)\n", what, k,
                        gpu[k].x, gpu[k].y, gpu[k].z, gpu[k].w, ref[k].x, ref[k].y, ref[k].z, ref[k].w);
        }
    }
    return bad;
}

static uint32_t cmpBytes(const char* what, const std::vector<uint8_t>& gpu, const std::vector<uint8_t>& ref, uint32_t& printed)
{
    uint32_t bad = 0;
    for (size_t k = 0; k + 4 <= ref.size(); k += 4)
    {
        if (std::memcmp(&gpu[k], &ref[k], 4) == 0) continue;
        ++bad;
        if (printed < 8)
        {
            ++printed;
            uint32_t g, r; std::memcpy(&g, &gpu[k], 4); std::memcpy(&r, &ref[k], 4);
            std::printf("    %s word @%zu gpu 0x%08x (%.9g) ref 0x%08x (%.9g)\n", what, k, g, getF32(gpu, k), r, getF32(ref, k));
        }
    }
    return bad;
}

// ===========================================================================
// First + Second
// ===========================================================================
struct FsCase { const char* name; uint32_t n; uint32_t maxRun; uint32_t giantAt; uint32_t giantLen; uint32_t seed; };

PXG_TEST(rigidDeltaAccum_firstSecond)
{
    PxgSlangComputeDevice* dev = testDevice();
    const FsCase cases[] = {
        { "empty", 0u, 1u, 999u, 0u, 11u },
        { "five", 5u, 3u, 999u, 0u, 12u },
        { "700-two-blocks", 700u, 60u, 999u, 0u, 13u },
        { "20000-giant-run", 20000u, 90u, 50u, 3000u, 14u },
    };
    for (const FsCase& c : cases)
    {
        for (int tgs = 0; tgs < 2; ++tgs)
        {
            Rng rng{ 0x9e3779b9u ^ (c.seed * 7919u) ^ uint32_t(tgs) };
            World W;
            buildBodies(W, rng);
            fillRuns(W, rng, c.n, c.maxRun, c.giantAt, c.giantLen);

            Gpu G{ dev };
            if (!G.upload(W)) return;
            G.runFirst();
            const std::vector<F4> gDeltaV = downloadBuffer<F4>(dev, G.bDeltaV, W.deltaV.empty() ? 1 : W.deltaV.size());
            const std::vector<F4> gBlockDeltaV = downloadBuffer<F4>(dev, G.bBlockDeltaV, 2 * AD_GRID);
            const std::vector<uint64_t> gBlockRid = downloadBuffer<uint64_t>(dev, G.bBlockRid, AD_GRID);

            const FirstOut ref = refFirst(W);
            uint32_t printed = 0;
            uint32_t bad = 0;
            if (W.n) bad += cmpF4("deltaV", gDeltaV, ref.deltaV, printed);
            bad += cmpF4("blockDeltaV", gBlockDeltaV, ref.blockDeltaV, printed);
            uint32_t badRid = 0;
            for (uint32_t b = 0; b < AD_GRID; ++b)
                if (gBlockRid[b] != ref.blockRigidId[b]) { ++badRid; if (printed < 8) { ++printed; std::printf("    blockRigidId[%u] gpu %llx ref %llx\n", b, (unsigned long long)gBlockRid[b], (unsigned long long)ref.blockRigidId[b]); } }
            PXG_CHECK_MSG(bad == 0 && badRid == 0, "First[%s,tgs=%d]: %u float4 words and %u ids differ", c.name, tgs, bad, badRid);
            {
                uint32_t sentinels = 0;
                for (uint32_t b = 0; b < AD_GRID; ++b) sentinels += (ref.blockRigidId[b] == SENTINEL) ? 1u : 0u;
                std::printf("  First[%s,tgs=%d]: n=%u, %u/32 block carries at the sentinel\n", c.name, tgs, W.n, sentinels);
            }

            G.runSecond(tgs != 0);
            const std::vector<F4> gPool = downloadBuffer<F4>(dev, G.bPool, W.pool.size());
            const std::vector<uint8_t> gArti = downloadBuffer<uint8_t>(dev, G.bArtiBlocks, W.artiBlocks.size());
            const std::vector<uint8_t> gLinks = downloadBuffer<uint8_t>(dev, G.bLinkBlocks, W.linkBlocks.size());
            G.destroy();

            World R = W;
            refSecond(R, ref, tgs != 0);
            uint32_t printed2 = 0;
            uint32_t bad2 = cmpF4("pool", gPool, R.pool, printed2);
            bad2 += cmpBytes("artiBlocks", gArti, R.artiBlocks, printed2);
            bad2 += cmpBytes("linkBlocks", gLinks, R.linkBlocks, printed2);
            // count what Second touched, so the case is known to exercise its branches
            uint32_t rigidRuns = 0, artiRuns = 0, staticRuns = 0, carried = 0;
            for (uint32_t w = 0; w < W.n; ++w)
            {
                if (!((w == W.n - 1) || W.ids[w + 1] != W.ids[w])) continue;
                const uint64_t id = W.ids[w];
                if (nodeIndex(id) == STATIC_NODE) ++staticRuns; else if (nodeIsArti(id)) ++artiRuns; else ++rigidRuns;
                const uint32_t b = blockOf(w, W.n);
                if (b > 0 && ref.blockRigidId[b - 1] == id) ++carried;
            }
            std::printf("  Second[%s,tgs=%d]: runs rigid %u arti %u static %u, %u carried across a block boundary\n",
                        c.name, tgs, rigidRuns, artiRuns, staticRuns, carried);
            PXG_CHECK_MSG(bad2 == 0, "Second[%s,tgs=%d]: %u words differ", c.name, tgs, bad2);
            if (c.n >= 20000u)
                PXG_CHECK_MSG(carried >= 3u, "the giant run did not cross blocks (%u carried)", carried);
        }
    }
}

// ===========================================================================
// clear -> Stage1 -> Stage2 (the Multi trio) on First's output
// ===========================================================================
PXG_TEST(rigidDeltaAccum_multiChain)
{
    PxgSlangComputeDevice* dev = testDevice();
    const float globalRelax = 0.7f;
    for (int variant = 0; variant < 4; ++variant)
    {
        const bool useLocalRelax = (variant & 1) != 0;
        const bool isTGS = (variant & 2) != 0;
        Rng rng{ 0x51ed270bu ^ uint32_t(variant * 104729) };
        World W;
        buildBodies(W, rng);
        std::vector<uint32_t> runsPerId;
        fillMultiRuns(W, rng, 3000u, 12u, runsPerId);

        // bodies / links heading two runs start from a zero base (see header)
        uint32_t twoRuns = 0;
        for (uint32_t w = 0; w < W.n; ++w)
        {
            if (w > 0 && W.ids[w] == W.ids[w - 1]) continue;
            const uint64_t id = W.ids[w];
            if (nodeIndex(id) == STATIC_NODE) continue;
            uint32_t slot;
            if (nodeIsArti(id)) slot = NUM_RIGID_NODES + (nodeIndex(id) - NUM_RIGID_NODES) * MAX_LINKS + nodeLink(id);
            else slot = nodeIndex(id);
            if (runsPerId[slot] < 2u) continue;
            const uint32_t sbi = W.solverBodyIndices[nodeIndex(id)];
            if (nodeIsArti(id))
            {
                const uint32_t blockIndex = sbi / WARP, aib = sbi % WARP;
                const size_t linkBase = size_t(blockIndex * W.maxLinks + nodeLink(id)) * AB_LINK_SIZE;
                const float z[3] = { 0.0f, 0.0f, 0.0f };
                writeScratchImpulse(W.linkBlocks, linkBase, aib, z, z, false);
            }
            else
            {
                W.pool[W.acc + sbi] = F4{ 0.0f, 0.0f, 0.0f, 0.0f };
                W.pool[W.acc + sbi + W.N] = F4{ 0.0f, 0.0f, 0.0f, 0.0f };
            }
            ++twoRuns;
        }
        PXG_CHECK_MSG(twoRuns > 20u, "multi[%d]: only %u ids head two runs", variant, twoRuns);

        Gpu G{ dev };
        if (!G.upload(W)) return;
        G.runFirst();
        const FirstOut ref = refFirst(W);
        {
            const std::vector<F4> gDeltaV = downloadBuffer<F4>(dev, G.bDeltaV, W.deltaV.size());
            uint32_t printed = 0;
            const uint32_t bad = cmpF4("deltaV", gDeltaV, ref.deltaV, printed);
            PXG_CHECK_MSG(bad == 0, "multi[%d] First: %u words differ", variant, bad);
        }

        World R = W;
        // clear
        G.runClear();
        refClearMulti(R);
        {
            const std::vector<float> gDenom = downloadBuffer<float>(dev, G.bTempDenom, W.N);
            const std::vector<uint8_t> gLinks = downloadBuffer<uint8_t>(dev, G.bLinkBlocks, W.linkBlocks.size());
            uint32_t printed = 0, bad = 0, zeroed = 0;
            for (uint32_t k = 0; k < W.N; ++k) { if (std::memcmp(&gDenom[k], &R.tempDenom[k], 4) != 0) ++bad; if (R.tempDenom[k] == 0.0f) ++zeroed; }
            bad += cmpBytes("linkBlocks", gLinks, R.linkBlocks, printed);
            std::printf("  multi[%d] clear: %u tempDenom slots zeroed of %u\n", variant, zeroed, W.N);
            PXG_CHECK_MSG(bad == 0, "multi[%d] clear: %u words differ", variant, bad);
        }
        // Stage1
        G.runMulti("accumulateDeltaVRigidSecondLaunchMultiStage1", useLocalRelax, globalRelax, isTGS);
        refStage1(R, ref, useLocalRelax, globalRelax);
        {
            const std::vector<float> gDenom = downloadBuffer<float>(dev, G.bTempDenom, W.N);
            const std::vector<uint8_t> gLinks = downloadBuffer<uint8_t>(dev, G.bLinkBlocks, W.linkBlocks.size());
            uint32_t printed = 0, bad = 0;
            for (uint32_t k = 0; k < W.N; ++k)
                if (std::memcmp(&gDenom[k], &R.tempDenom[k], 4) != 0) { ++bad; if (printed < 8) { ++printed; std::printf("    tempDenom[%u] gpu %.9g ref %.9g\n", k, gDenom[k], R.tempDenom[k]); } }
            bad += cmpBytes("linkBlocks", gLinks, R.linkBlocks, printed);
            PXG_CHECK_MSG(bad == 0, "multi[%d] Stage1(local=%d): %u words differ", variant, int(useLocalRelax), bad);
        }
        // Stage2
        G.runMulti("accumulateDeltaVRigidSecondLaunchMultiStage2", useLocalRelax, globalRelax, isTGS);
        refStage2(R, ref, isTGS);
        {
            const std::vector<F4> gPool = downloadBuffer<F4>(dev, G.bPool, W.pool.size());
            const std::vector<uint8_t> gArti = downloadBuffer<uint8_t>(dev, G.bArtiBlocks, W.artiBlocks.size());
            const std::vector<uint8_t> gLinks = downloadBuffer<uint8_t>(dev, G.bLinkBlocks, W.linkBlocks.size());
            uint32_t printed = 0;
            uint32_t bad = cmpF4("pool", gPool, R.pool, printed);
            bad += cmpBytes("artiBlocks", gArti, R.artiBlocks, printed);
            bad += cmpBytes("linkBlocks", gLinks, R.linkBlocks, printed);
            PXG_CHECK_MSG(bad == 0, "multi[%d] Stage2(tgs=%d): %u words differ", variant, int(isTGS), bad);
        }
        G.destroy();
        std::printf("  multi[%d] local=%d tgs=%d: %u ids with two runs, all stages word-exact\n", variant, int(useLocalRelax), int(isTGS), twoRuns);
    }
}

// ===========================================================================
// The two traps the first mutation sweep of this module left standing.
//
//   * PxMax's operand order.  The float overload forwards to selectMax
//     (`a > b ? a : b`), and PxMax(+0, -0) is the only finite input that tells it
//     apart from the generic template's `a < b ? b : a`.  Stage1's denominator is
//     exactly such a PxMax, so a run whose weights are all -0.0 and a global
//     relaxation coefficient of +0.0 make the difference observable -- but ONLY if
//     the accumulator it lands in is itself -0.0, since (+0) + (-0) is (+0) and
//     would erase the sign.  The clear kernel is therefore not run for this case
//     and tempDenom starts at -0.0.
//   * The atomic add's retry.  Its compare-exchange loop only matters when two
//     lanes hit the same word at the same time, which no id list can force if the
//     platform serialises the lanes.  An A,B,A,B list of 512 entries makes every
//     lane of block 0 the last contact of its own run, so 256 lanes add to one
//     rigid body and 256 to one articulation link.  All contributions are equal
//     powers of two, so the sums are order-independent and the comparison stays
//     word-exact whether or not the lanes actually collide.
// ===========================================================================
PXG_TEST(rigidDeltaAccum_pxMaxSignedZero)
{
    PxgSlangComputeDevice* dev = testDevice();
    Rng rng{ 0x7f4a7c15u };
    World W;
    buildBodies(W, rng);

    // two runs: the first carries -0.0 weights, the second +0.0 weights
    const uint64_t idNeg = rigidNode(11u), idPos = rigidNode(12u);
    W.ids = { idNeg, idNeg, idNeg, idPos, idPos };
    W.n = uint32_t(W.ids.size());
    W.deltaV.assign(2u * W.n, F4{ 0.25f, -0.5f, 0.75f, 0.0f });
    for (uint32_t w = 0; w < 3u; ++w) W.deltaV[w].w = -0.0f;
    for (uint32_t w = 3u; w < W.n; ++w) W.deltaV[w].w = 0.0f;
    // no clear: the accumulator must start at -0.0 or the add erases the sign
    W.tempDenom.assign(W.N, -0.0f);

    // Stage1 is run WITHOUT First in front of it, and that is the point: First
    // finishes every run with `deltaV[w] + accum`, and accum's .w is a positive
    // zero for a run that crosses no warp or block boundary, so -0.0 + (+0.0)
    // would come back out as +0.0 and the PxMax below would never see a negative
    // zero.  Stage1 reads deltaV directly, so the weights land in it untouched.
    Gpu G{ dev };
    if (!G.upload(W)) return;
    FirstOut ref;
    ref.deltaV = W.deltaV;
    ref.blockDeltaV.assign(2 * AD_GRID, F4{ 9.0f, 9.0f, 9.0f, 9.0f });   // as Gpu::upload seeds them
    ref.blockRigidId.assign(AD_GRID, 0x1234ull);                          // never equal to a run's id
    G.runMulti("accumulateDeltaVRigidSecondLaunchMultiStage1", /*useLocalRelax*/ true, /*globalRelax*/ 0.0f, false);
    const std::vector<float> gDenom = downloadBuffer<float>(dev, G.bTempDenom, W.N);
    G.destroy();

    World R = W;
    refStage1(R, ref, true, 0.0f);

    const uint32_t sbiNeg = W.solverBodyIndices[nodeIndex(idNeg)];
    const uint32_t sbiPos = W.solverBodyIndices[nodeIndex(idPos)];
    uint32_t gNeg, rNeg, gPos, rPos;
    std::memcpy(&gNeg, &gDenom[sbiNeg], 4); std::memcpy(&rNeg, &R.tempDenom[sbiNeg], 4);
    std::memcpy(&gPos, &gDenom[sbiPos], 4); std::memcpy(&rPos, &R.tempDenom[sbiPos], 4);
    std::printf("  PxMax(+0,-0): negative-weight run gpu 0x%08x ref 0x%08x, positive-weight run gpu 0x%08x ref 0x%08x\n",
                gNeg, rNeg, gPos, rPos);
    PXG_CHECK_MSG(rNeg == 0x80000000u, "the reference itself lost the negative zero (0x%08x) -- selectMax order", rNeg);
    PXG_CHECK_MSG(gNeg == rNeg, "PxMax order: denominator of the -0.0 run is 0x%08x, expected 0x%08x", gNeg, rNeg);
    PXG_CHECK_MSG(gPos == rPos, "denominator of the +0.0 run is 0x%08x, expected 0x%08x", gPos, rPos);

    uint32_t bad = 0;
    for (uint32_t k = 0; k < W.N; ++k)
        if (std::memcmp(&gDenom[k], &R.tempDenom[k], 4) != 0) ++bad;
    PXG_CHECK_MSG(bad == 0, "%u tempDenom words differ", bad);
}

PXG_TEST(rigidDeltaAccum_atomicCollision)
{
    PxgSlangComputeDevice* dev = testDevice();
    const float globalRelax = 0.5f;
    Rng rng{ 0x1b873593u };
    World W;
    buildBodies(W, rng);

    // 512 alternating entries -> every lane of block 0 ends its own run.
    const uint64_t idRigid = rigidNode(3u);
    const uint64_t idArti = artiNode(NUM_RIGID_NODES + 1u, 2u);
    W.ids.clear();
    for (uint32_t k = 0; k < 256u; ++k) { W.ids.push_back(idRigid); W.ids.push_back(idArti); }
    W.n = uint32_t(W.ids.size());
    // equal powers of two: any summation order gives the same bits
    W.deltaV.assign(2u * W.n, F4{ 1.0f, -2.0f, 0.25f, 4.0f });

    const uint32_t sbiRigid = W.solverBodyIndices[nodeIndex(idRigid)];
    const uint32_t sbiArti = W.solverBodyIndices[nodeIndex(idArti)];
    W.pool[W.acc + sbiRigid] = F4{ 0.0f, 0.0f, 0.0f, 0.0f };
    W.pool[W.acc + sbiRigid + W.N] = F4{ 0.0f, 0.0f, 0.0f, 0.0f };
    {
        const uint32_t blockIndex = sbiArti / WARP, aib = sbiArti % WARP;
        const size_t linkBase = size_t(blockIndex * W.maxLinks + nodeLink(idArti)) * AB_LINK_SIZE;
        const float z[3] = { 0.0f, 0.0f, 0.0f };
        writeScratchImpulse(W.linkBlocks, linkBase, aib, z, z, false);
    }

    Gpu G{ dev };
    if (!G.upload(W)) return;
    G.runFirst();
    const FirstOut ref = refFirst(W);
    G.runClear();
    G.runMulti("accumulateDeltaVRigidSecondLaunchMultiStage1", false, globalRelax, false);
    const std::vector<float> gDenom = downloadBuffer<float>(dev, G.bTempDenom, W.N);
    const std::vector<uint8_t> gLinks1 = downloadBuffer<uint8_t>(dev, G.bLinkBlocks, W.linkBlocks.size());
    G.runMulti("accumulateDeltaVRigidSecondLaunchMultiStage2", false, globalRelax, false);
    const std::vector<F4> gPool = downloadBuffer<F4>(dev, G.bPool, W.pool.size());
    const std::vector<uint8_t> gLinks2 = downloadBuffer<uint8_t>(dev, G.bLinkBlocks, W.linkBlocks.size());
    G.destroy();

    World R = W;
    refClearMulti(R);
    refStage1(R, ref, false, globalRelax);
    const std::vector<float> refDenom = R.tempDenom;
    const std::vector<uint8_t> refLinks1 = R.linkBlocks;
    refStage2(R, ref, false);

    const uint32_t blockIndex = sbiArti / WARP, aib = sbiArti % WARP;
    const size_t linkBase = size_t(blockIndex * W.maxLinks + nodeLink(idArti)) * AB_LINK_SIZE;
    std::printf("  256 concurrent adds per target: tempDenom gpu %.9g ref %.9g (256 x %.3g); link scale gpu %.9g ref %.9g\n",
                gDenom[sbiRigid], refDenom[sbiRigid], globalRelax,
                getF32(gLinks1, linkBase + AB_LINK_mDeltaScale + size_t(aib) * 4),
                getF32(refLinks1, linkBase + AB_LINK_mDeltaScale + size_t(aib) * 4));
    uint32_t printed = 0;
    uint32_t bad = 0;
    for (uint32_t k = 0; k < W.N; ++k)
        if (std::memcmp(&gDenom[k], &refDenom[k], 4) != 0)
        { ++bad; if (printed < 4) { ++printed; std::printf("    tempDenom[%u] gpu %.9g ref %.9g\n", k, gDenom[k], refDenom[k]); } }
    bad += cmpBytes("linkBlocks(stage1)", gLinks1, refLinks1, printed);
    PXG_CHECK_MSG(bad == 0, "Stage1 under collision: %u words differ (a lost compare-exchange)", bad);

    uint32_t bad2 = cmpF4("pool", gPool, R.pool, printed);
    bad2 += cmpBytes("linkBlocks(stage2)", gLinks2, R.linkBlocks, printed);
    PXG_CHECK_MSG(bad2 == 0, "Stage2 under collision: %u words differ (a lost compare-exchange)", bad2);
}

} // namespace
