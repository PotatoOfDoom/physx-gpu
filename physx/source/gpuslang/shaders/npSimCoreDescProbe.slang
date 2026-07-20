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

// npSimCoreDescProbe — P5 Step 4: the empirical layout proof for the
// simulation-controller descriptor mirrors (pxgSlangSimCoreDesc.slang) and
// the element structs their kernels index (PxgShapeSim, PxgNewShapeSim,
// PxgBodySim, PxsCachedTransform, PxBounds3, PxgSolverBodySleepData,
// PxgShape) plus the raw PxgArticulation stride/offset constants.
//
// The host test (testSimCoreDescLayout.cpp) fills the REAL engine structs
// with distinct sentinels, uploads the raw bytes, and this kernel reads:
//   A) every PxgSimulationCoreDesc field through a sentinel-filled desc,
//   B) every PxgNewBodiesDesc field,
//   C) element [1] of every indexed element array through a second desc
//      whose pointers are LIVE device addresses (pinning both the field
//      offsets and the array STRIDES),
//   D) the PxgBodySim uint4 constants and the PxgArticulation stride/offset
//      constants (host compares against sizeof/offsetof), plus a functional
//      walk mArticulationPool -> articulation[1].linkBody2Worlds[2] /
//      .linkBody2Actors[2] through the raw-offset path the kernel uses.
// A single wrong offset, stride, or truncation mismatches bitwise.

import pxgSlangCommon;
import pxgSlangSimCoreDesc;

struct SimCoreProbeParams
{
    uint64_t scDescSentinel; // engine PxgSimulationCoreDesc, sentinel-filled
    uint64_t scDescLive;     // engine PxgSimulationCoreDesc, live element arrays
    uint64_t newBodiesDesc;  // engine PxgNewBodiesDesc, sentinel-filled
    uint64_t newShapeSims;   // engine PxgNewShapeSim[2]
    uint64_t out;            // uint64_t[PROBE_COUNT]
}

[[vk::push_constant]]
ConstantBuffer<SimCoreProbeParams> gProbe;

[shader("compute")]
[numthreads(1, 1, 1)]
void simCoreDescProbe(uint3 dtid : SV_DispatchThreadID)
{
    if (dtid.x != 0)
        return;
    PxgSimulationCoreDescGpu d = (Ptr<PxgSimulationCoreDescGpu>(gProbe.scDescSentinel))[0];
    PxgSimulationCoreDescGpu live = (Ptr<PxgSimulationCoreDescGpu>(gProbe.scDescLive))[0];
    PxgNewBodiesDescGpu nb = (Ptr<PxgNewBodiesDescGpu>(gProbe.newBodiesDesc))[0];
    Ptr<uint64_t> o = Ptr<uint64_t>(gProbe.out);
    uint i = 0;

    // ---- A) PxgSimulationCoreDesc: all 17 pointers + 4 counters in order ----
    o[i++] = d.mChangedAABBMgrHandles;
    o[i++] = d.mFrozen;
    o[i++] = d.mUnfrozen;
    o[i++] = d.mFrozenBlockAndRes;
    o[i++] = d.mUnfrozenBlockAndRes;
    o[i++] = d.mUpdated;
    o[i++] = d.mActivate;
    o[i++] = d.mDeactivate;
    o[i++] = d.mBodySimBufferDeviceData;
    o[i++] = d.mShapeSimsBufferDeviceData;
    o[i++] = d.mArticulationPool;
    o[i++] = d.mArticulationSleepDataPool;
    o[i++] = d.mTransformCache;
    o[i++] = d.mBounds;
    o[i++] = d.mBodyDataIndices;
    o[i++] = d.mSleepData;
    o[i++] = d.mShapes;
    o[i++] = uint64_t(d.mNbTotalShapes);
    o[i++] = uint64_t(d.mBitMapWordCounts);
    o[i++] = uint64_t(d.mTotalFrozenShapes);
    o[i++] = uint64_t(d.mTotalUnfrozenShapes);

    // ---- B) PxgNewBodiesDesc ----
    o[i++] = nb.mNewBodySim;
    o[i++] = nb.mBodySimBufferDeviceData;
    o[i++] = uint64_t(nb.mNbNewBodies);

    // ---- C) element [1] of every indexed array (offset + STRIDE proof) ----

    // PxgShapeSim[1] (72-byte stride)
    {
        PxgShapeSim s = (Ptr<PxgShapeSim>(live.mShapeSimsBufferDeviceData))[1];
        o[i++] = uint64_t(asuint(s.mTransform.qx));
        o[i++] = uint64_t(asuint(s.mTransform.pz));
        o[i++] = uint64_t(asuint(s.lbMinX));
        o[i++] = uint64_t(asuint(s.lbMaxZ));
        o[i++] = uint64_t(s.mBodySimIndex.x);
        o[i++] = uint64_t(s.mBodySimIndex.y);
        o[i++] = uint64_t(s.mHullDataIndex);
        o[i++] = uint64_t(s.mShapeFlags_mShapeType);
    }

    // PxgNewShapeSim[1] (80-byte stride + mElementIndex at 72)
    {
        PxgNewShapeSimGpu s = (Ptr<PxgNewShapeSimGpu>(gProbe.newShapeSims))[1];
        o[i++] = uint64_t(asuint(s.base.mTransform.qy));
        o[i++] = uint64_t(asuint(s.base.lbMaxY));
        o[i++] = uint64_t(s.base.mBodySimIndex.x);
        o[i++] = uint64_t(s.base.mHullDataIndex);
        o[i++] = uint64_t(s.base.mShapeFlags_mShapeType);
        o[i++] = uint64_t(s.mElementIndex);
    }

    // PxgBodySim[1] (224-byte stride)
    {
        PxgBodySim b = (Ptr<PxgBodySim>(live.mBodySimBufferDeviceData))[1];
        o[i++] = uint64_t(asuint(b.linearVelocityXYZ_inverseMassW.x));
        o[i++] = uint64_t(asuint(b.freezeThresholdX_wakeCounterY_sleepThresholdZ_bodySimIndex.w));
        o[i++] = uint64_t(asuint(b.body2World.q.x));
        o[i++] = uint64_t(asuint(b.body2World.p.z));
        o[i++] = uint64_t(asuint(b.body2Actor_maxImpulseW.q.y));
        o[i++] = uint64_t(asuint(b.body2Actor_maxImpulseW.p.x));
        o[i++] = uint64_t(b.articulationRemapId);
        o[i++] = uint64_t(b.internalFlags);
    }

    // PxgSolverBodySleepData[1] (8-byte stride), twice: mSleepData and
    // mArticulationSleepDataPool point at two different live arrays
    {
        PxgSolverBodySleepData s = (Ptr<PxgSolverBodySleepData>(live.mSleepData))[1];
        o[i++] = uint64_t(asuint(s.wakeCounter));
        o[i++] = uint64_t(s.internalFlags);
        PxgSolverBodySleepData a = (Ptr<PxgSolverBodySleepData>(live.mArticulationSleepDataPool))[1];
        o[i++] = uint64_t(a.internalFlags);
    }

    // PxsCachedTransform[1] (32-byte stride)
    {
        PxsCachedTransform c = (Ptr<PxsCachedTransform>(live.mTransformCache))[1];
        o[i++] = uint64_t(asuint(c.transform.qx));
        o[i++] = uint64_t(asuint(c.transform.pz));
        o[i++] = uint64_t(c.flags);
    }

    // PxBounds3[1] (24-byte stride)
    {
        PxBounds3Gpu b = (Ptr<PxBounds3Gpu>(live.mBounds))[1];
        o[i++] = uint64_t(asuint(b.minX));
        o[i++] = uint64_t(asuint(b.maxZ));
    }

    // PxgShape[1] (48-byte stride, u64 hullOrMeshPtr at 32)
    {
        PxgShapeGpu s = (Ptr<PxgShapeGpu>(live.mShapes))[1];
        o[i++] = uint64_t(asuint(s.scaleX));
        o[i++] = uint64_t(asuint(s.scaleRotW));
        o[i++] = uint64_t(s.materialIndex);
        o[i++] = s.hullOrMeshPtr;
        o[i++] = uint64_t(s.type);
        o[i++] = uint64_t(s.particleOrSoftbodyId);
    }

    // PxU32 mBodyDataIndices[1]
    o[i++] = uint64_t((Ptr<uint>(live.mBodyDataIndices))[1]);

    // ---- D) raw stride/offset constants + the articulation pointer walk ----
    o[i++] = uint64_t(PXG_BODY_SIM_UINT4_SIZE);
    o[i++] = uint64_t(PXG_BODY_SIM_BODYSIM_INDEX_IND);
    o[i++] = uint64_t(PXG_ARTICULATION_STRIDE);
    o[i++] = uint64_t(PXG_ARTI_LINK_BODY2WORLDS_OFFSET);
    o[i++] = uint64_t(PXG_ARTI_LINK_BODY2ACTORS_OFFSET);

    // mArticulationPool -> PxgArticulation[1] -> linkBody2Worlds[2] /
    // linkBody2Actors[2] through the exact path the converted kernel uses
    {
        Ptr<PxgTransform28> b2w = Ptr<PxgTransform28>(
            artiLinkTransformArray(live.mArticulationPool, 1, PXG_ARTI_LINK_BODY2WORLDS_OFFSET));
        const PxgTransform28 w = b2w[2];
        o[i++] = uint64_t(asuint(w.qx));
        o[i++] = uint64_t(asuint(w.qw));
        o[i++] = uint64_t(asuint(w.px));
        o[i++] = uint64_t(asuint(w.pz));

        Ptr<PxgTransform28> b2a = Ptr<PxgTransform28>(
            artiLinkTransformArray(live.mArticulationPool, 1, PXG_ARTI_LINK_BODY2ACTORS_OFFSET));
        const PxgTransform28 a = b2a[2];
        o[i++] = uint64_t(asuint(a.qy));
        o[i++] = uint64_t(asuint(a.py));
    }
}
