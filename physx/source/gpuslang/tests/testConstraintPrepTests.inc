// -*- mode: c++ -*-
// Included by testConstraintPrep3.inc — the four PXG_TEST drivers.

namespace
{

QuatV randQuat(std::mt19937& rng)
{
    std::uniform_real_distribution<float> d(-1.f, 1.f);
    QuatV q(d(rng), d(rng), d(rng), d(rng));
    const float m = std::sqrt(q.x * q.x + q.y * q.y + q.z * q.z + q.w * q.w);
    return m > 1e-4f ? QuatV(q.x / m, q.y / m, q.z / m, q.w / m) : QuatV(0, 0, 0, 1);
}

Mat33 randSymInvInertia(std::mt19937& rng)
{
    std::uniform_real_distribution<float> u(0.2f, 1.5f);
    const Mat33 rot(randQuat(rng));
    const Vec3 d(u(rng), u(rng), u(rng));
    return rot * Mat33(Vec3(d.x, 0, 0), Vec3(0, d.y, 0), Vec3(0, 0, d.z)) * rot.getTranspose();
}

AlignedTransform makeFrame(std::mt19937& rng)
{
    std::uniform_real_distribution<float> p(-2.f, 2.f);
    const QuatV q = randQuat(rng);
    AlignedTransform t;
    t.q = Float4{ q.x, q.y, q.z, q.w };
    t.p = Float4{ p(rng), p(rng), p(rng), 0.f };
    return t;
}

AFrame toAFrame(const AlignedTransform& t)
{
    AFrame f;
    f.q = QuatV(t.q.x, t.q.y, t.q.z, t.q.w);
    f.p = Vec3(t.p.x, t.p.y, t.p.z);
    return f;
}

bool eq(float a, float b, float rt = 2e-4f, float at = 2e-4f)
{
    return almostEqual(a, b, rt, at);
}
bool eq4(const Float4& a, const Float4& b, float rt = 2e-4f, float at = 2e-4f)
{
    return eq(a.x, b.x, rt, at) && eq(a.y, b.y, rt, at) && eq(a.z, b.z, rt, at) && eq(a.w, b.w, rt, at);
}
bool eq3(const Float4& a, const Float4& b, float rt = 2e-4f, float at = 2e-4f)
{
    return eq(a.x, b.x, rt, at) && eq(a.y, b.y, rt, at) && eq(a.z, b.z, rt, at);
}

// ---------------------------------------------------------------------------
// contact prep test (shared PGS/TGS setup)
// ---------------------------------------------------------------------------
struct ContactSetup
{
    uint32_t numBatches, rowsPerBatch, numBodies;
    std::vector<Batch> batches;
    std::vector<uint32_t> batchIndices;
    std::vector<SolverBodyData> bodyData;
    std::vector<SolverTxIData> txi;
    std::vector<AlignedTransform> bodyFrames;
    std::vector<BlockWorkUnit> workUnits;
    std::vector<BlockContactData> cdata;
    std::vector<BlockContactPoint> cpoints;
    std::vector<Mat33> si;
    std::vector<AFrame> frames;
};

ContactSetup buildContactSetup(std::mt19937& rng)
{
    std::uniform_real_distribution<float> u(0.f, 1.f), s(-1.f, 1.f);
    ContactSetup cs;
    cs.numBatches = 6;
    cs.rowsPerBatch = 4; // max contacts per batch
    cs.numBodies = 48;

    cs.bodyData.resize(cs.numBodies);
    cs.txi.resize(cs.numBodies);
    cs.bodyFrames.resize(cs.numBodies);
    cs.si.resize(cs.numBodies);
    cs.frames.resize(cs.numBodies);
    std::memset(cs.bodyData.data(), 0, cs.bodyData.size() * sizeof(SolverBodyData));
    std::memset(cs.txi.data(), 0, cs.txi.size() * sizeof(SolverTxIData));
    for (uint32_t b = 0; b < cs.numBodies; ++b)
    {
        cs.bodyData[b].initialLinVelXYZ_invMassW = Float4{ s(rng), s(rng), s(rng), u(rng) * 2.f };
        cs.bodyData[b].initialAngVelXYZ_penBiasClamp = Float4{ s(rng), s(rng), s(rng), -u(rng) * 4.f };
        cs.bodyData[b].maxImpulse = 100.f + u(rng) * 900.f;
        cs.bodyData[b].offsetSlop = u(rng) < 0.5f ? 0.f : u(rng) * 0.02f;
        cs.bodyData[b].flags = (u(rng) < 0.15f ? RB_KINEMATIC : 0u) | (u(rng) < 0.3f ? RB_SPEC_CCD : 0u);
        const AlignedTransform f = makeFrame(rng);
        cs.bodyFrames[b] = f;
        cs.frames[b] = toAFrame(f);
        const Mat33 m = randSymInvInertia(rng);
        cs.si[b] = m;
        m.store(cs.txi[b].sqrtInvInertia);
    }

    cs.batches.resize(cs.numBatches);
    cs.workUnits.resize(cs.numBatches);
    cs.cdata.resize(cs.numBatches);
    cs.cpoints.resize(cs.numBatches * cs.rowsPerBatch);
    cs.batchIndices.resize(cs.numBatches);
    std::memset(cs.batches.data(), 0, cs.batches.size() * sizeof(Batch));
    std::memset(cs.workUnits.data(), 0, cs.workUnits.size() * sizeof(BlockWorkUnit));
    std::memset(cs.cdata.data(), 0, cs.cdata.size() * sizeof(BlockContactData));
    std::memset(cs.cpoints.data(), 0, cs.cpoints.size() * sizeof(BlockContactPoint));

    uint32_t forceCursor = 0;
    for (uint32_t k = 0; k < cs.numBatches; ++k)
    {
        cs.batchIndices[k] = k;
        Batch& b = cs.batches[k];
        b.descStride = uint16_t(4 + rng() % 13);
        b.constraintType = 1;
        b.mConstraintBatchIndex = k;
        b.blockContactIndex = k * cs.rowsPerBatch;
        b.startConstraintIndex = k * cs.rowsPerBatch;
        b.startFrictionIndex = k * (cs.rowsPerBatch * 2 + 1);

        for (uint32_t lane = 0; lane < 32; ++lane)
        {
            b.bodyAIndex[lane] = rng() % cs.numBodies;
            b.bodyBIndex[lane] = rng() % cs.numBodies;

            const uint32_t nc = 1 + rng() % cs.rowsPerBatch;
            uint32_t flags = 0;
            if (u(rng) < 0.15f) flags |= SC_PER_POINT_FRICTION;
            if (u(rng) < 0.1f) flags |= SC_DISABLE_FRICTION;
            if (u(rng) < 0.5f) flags |= SC_HAS_FORCE_THRESHOLDS;
            if (u(rng) < 0.3f) flags |= SC_COMPLIANT_ACCEL_SPRING;

            const float restDistance = u(rng) * 0.02f;
            const float staticF = u(rng), dynamicF = u(rng) * 0.9f;
            // pack material word
            const uint32_t materialWord = (nc & 0xFF) | ((flags & 0xFF) << 8) | (0u << 16);
            float wbits;
            std::memcpy(&wbits, &materialWord, 4);
            cs.cdata[k].contactData[lane] = Float4{ restDistance, staticF, dynamicF, wbits };

            const Vec3 n = Vec3(s(rng), s(rng), s(rng)).getNormalized();
            const float restitution = u(rng) < 0.3f ? -u(rng) * 20.f : (u(rng) < 0.5f ? u(rng) * 0.8f : 0.f);
            cs.cdata[k].normal_restitutionW[lane] = Float4{ n.x, n.y, n.z, restitution };
            cs.cdata[k].damping[lane] = u(rng) * 5.f;
            cs.cdata[k].mInvMassScale[lane] = Float4{ 1.f, 1.f, 1.f, 1.f };

            for (uint32_t i = 0; i < cs.rowsPerBatch; ++i)
            {
                BlockContactPoint& cp = cs.cpoints[b.blockContactIndex + i];
                // put contact points near a random world spot, small separations
                const Vec3 pt(s(rng) * 2.f, s(rng) * 2.f, s(rng) * 2.f);
                const float sep = s(rng) * 0.03f;
                cp.point_separationW[lane] = Float4{ pt.x, pt.y, pt.z, sep };
                cp.targetVel_maxImpulseW[lane] = Float4{ s(rng) * 0.5f, s(rng) * 0.5f, s(rng) * 0.5f, 50.f + u(rng) * 50.f };
            }

            cs.workUnits[k].mWriteback[lane] = u(rng) < 0.85f ? (forceCursor += cs.rowsPerBatch) - cs.rowsPerBatch : 0xffffffffu;
            cs.workUnits[k].mEdgeIndex[lane] = k * 32 + lane;
            cs.workUnits[k].mPatchIndex[lane] = 0;
            cs.workUnits[k].mFrictionPatchIndex[lane] = k * 32 + lane;
            cs.workUnits[k].mTorsionalFrictionData[lane] =
                Float2{ u(rng) < 0.3f ? u(rng) * 0.1f : 0.f, u(rng) < 0.3f ? u(rng) * 0.05f : 0.f };
        }
    }
    return cs;
}

PXG_TEST(constraintPrepPGS_contact)
{
    using namespace physx;
    using physx::gpuslang::PxgCudaShim;
    using physx::gpuslang::PxgKernelDef;
    using physx::gpuslang::PxgDevicePtr;
    using physx::gpuslang::PxgResult;

    std::mt19937 rng(4100);
    ContactSetup cs = buildContactSetup(rng);
    // P9.86c: signed-zero directed lanes in the even batches (no rng consumed; compared
    // bitwise below).
    //  * maxContactImpulse zero ties: contactConstraintBlockPrep.cuh:402 and :513 are PxMin
    //    = selectMin (a < b ? a : b; a +-0 tie keeps b), not FMin (min(-0,+0) = -0 on llvmpipe).
    //    setMaxContactImpulse(-0.0f) passes the API's `maxImpulse >= 0.f` check
    //    (NpRigidBodyTemplate.h:820) and preIntegration copies it raw; a contact ignored by
    //    contact modification has maxImpulse +0 (PxContactSet::ignore).
    //  * a resting pair (all velocities +0) with a zero target velocity and an all-negative
    //    normal: cTargetVel = n . (0,0,0) = -0, vrel = +0, no bounce, so the CUDA's
    //    `cTargetVel + 0.f` (cuh:480) stores targetVelocity (+0) - (+0) = +0.
    cs.bodyData[2].maxImpulse = -0.0f;
    cs.bodyData[4].maxImpulse = 0.0f; // body 5 keeps its random 100..1000
    for (uint32_t rb : { 6u, 7u })
    {
        Float4& lv = cs.bodyData[rb].initialLinVelXYZ_invMassW;
        Float4& av = cs.bodyData[rb].initialAngVelXYZ_penBiasClamp;
        lv = Float4{ 0.f, 0.f, 0.f, lv.w };
        av = Float4{ 0.f, 0.f, 0.f, av.w };
    }
    const float negN = -1.f / std::sqrt(3.f);
    for (uint32_t k = 0; k < cs.numBatches; k += 2) // descStride >= 4: lanes 0..3 always run
    {
        Batch& b = cs.batches[k];
        // lane 0: (A +0, B -0) -> PxMin(+0,-0) = -0 in every row (pins the operand order)
        b.bodyAIndex[0] = 4; b.bodyBIndex[0] = 2;
        // lane 1: (A -0, B +0) -> PxMin(-0,+0) = +0; every point limit is 50..100 > 0
        b.bodyAIndex[1] = 2; b.bodyBIndex[1] = 4;
        // lane 3: (A -0, B positive) -> maxImpulse -0; contact 0 ignored (w = +0) -> PxMin(-0,+0) = +0
        b.bodyAIndex[3] = 2; b.bodyBIndex[3] = 5;
        cs.cpoints[b.blockContactIndex].targetVel_maxImpulseW[3].w = 0.0f;
        // lane 2: the resting pair, restitution 0, restDistance 0, separations +-0
        b.bodyAIndex[2] = 6; b.bodyBIndex[2] = 7;
        cs.cdata[k].normal_restitutionW[2] = Float4{ negN, negN, negN, 0.f };
        cs.cdata[k].contactData[2].x = 0.f;
        for (uint32_t i = 0; i < cs.rowsPerBatch; ++i)
        {
            BlockContactPoint& cp = cs.cpoints[b.blockContactIndex + i];
            cp.targetVel_maxImpulseW[2] = Float4{ 0.f, 0.f, 0.f, cp.targetVel_maxImpulseW[2].w };
            cp.point_separationW[2].w = (i & 1u) ? 0.0f : -0.0f;
        }
    }

    const float dt = 1.f / 60.f, invDt = 60.f;
    const float bounceThreshold = -2.f, frictionOffsetThreshold = 0.04f, correlationDistance = 0.025f;
    const float ccdMaxSeparation = 0.1f;

    PxgSlangComputeDevice* dev = testDevice();

    std::vector<BlockFrictionPatch> currPatches(cs.numBatches);
    std::vector<BlockFrictionAnchorPatch> currAnchors(cs.numBatches);
    std::memset(currPatches.data(), 0, currPatches.size() * sizeof(BlockFrictionPatch));
    std::memset(currAnchors.data(), 0, currAnchors.size() * sizeof(BlockFrictionAnchorPatch));

    std::vector<BlockSolverContactHeader> chs(cs.numBatches);
    std::vector<BlockSolverFrictionHeader> fhs(cs.numBatches);
    std::vector<BlockSolverContactPoint> outCP(cs.numBatches * cs.rowsPerBatch);
    std::vector<BlockSolverContactFriction> outFR(cs.numBatches * (cs.rowsPerBatch * 2 + 1));
    std::memset(chs.data(), 0, chs.size() * sizeof(BlockSolverContactHeader));
    std::memset(fhs.data(), 0, fhs.size() * sizeof(BlockSolverFrictionHeader));
    std::memset(outCP.data(), 0, outCP.size() * sizeof(BlockSolverContactPoint));
    std::memset(outFR.data(), 0, outFR.size() * sizeof(BlockSolverContactFriction));

    std::vector<uint64_t> currFricIdx(cs.numBatches * 32 + 1, 0);

    // ---- ENGINE ABI (P5 Step 5): live device buffers + REAL engine
    // descriptors, launched through PxgCudaShim exactly as
    // PxgCudaSolverCore::contactConstraintPrepareParallel does ----
    static std::vector<uint8_t> (*pgsLoader)(const char*) = [](const char* f) { return loadShaderBlob(f); };
    const PxgKernelDef manifest[] = {
        { "constraintPrepPGS.spv", "contactConstraintBlockPrepareParallelLaunch", 16u, 0 },
    };
    PxgCudaShim shim(dev, manifest, 1, pgsLoader);

    auto alloc = [&](const void* data, size_t bytes) -> PxgDevicePtr {
        PxgDevicePtr dp = 0;
        shim.memAlloc(&dp, bytes);
        if (dp && data)
            shim.memcpyHtoD(dp, data, bytes);
        return dp;
    };

    const PxgDevicePtr dBatches = alloc(cs.batches.data(), cs.batches.size() * sizeof(Batch));
    const PxgDevicePtr dBi = alloc(cs.batchIndices.data(), cs.batchIndices.size() * 4);
    const PxgDevicePtr dBd = alloc(cs.bodyData.data(), cs.bodyData.size() * sizeof(SolverBodyData));
    const PxgDevicePtr dTx = alloc(cs.txi.data(), cs.txi.size() * sizeof(SolverTxIData));
    const PxgDevicePtr dBf = alloc(cs.bodyFrames.data(), cs.bodyFrames.size() * sizeof(AlignedTransform));
    const PxgDevicePtr dWu = alloc(cs.workUnits.data(), cs.workUnits.size() * sizeof(BlockWorkUnit));
    const PxgDevicePtr dCd = alloc(cs.cdata.data(), cs.cdata.size() * sizeof(BlockContactData));
    const PxgDevicePtr dCp = alloc(cs.cpoints.data(), cs.cpoints.size() * sizeof(BlockContactPoint));
    const PxgDevicePtr dPatch = alloc(currPatches.data(), currPatches.size() * sizeof(BlockFrictionPatch));
    const PxgDevicePtr dAnch = alloc(currAnchors.data(), currAnchors.size() * sizeof(BlockFrictionAnchorPatch));
    const PxgDevicePtr dCfi = alloc(currFricIdx.data(), currFricIdx.size() * 8);
    const PxgDevicePtr dCh = alloc(chs.data(), chs.size() * sizeof(BlockSolverContactHeader));
    const PxgDevicePtr dFh = alloc(fhs.data(), fhs.size() * sizeof(BlockSolverFrictionHeader));
    const PxgDevicePtr dOutCP = alloc(outCP.data(), outCP.size() * sizeof(BlockSolverContactPoint));
    const PxgDevicePtr dOutFR = alloc(outFR.data(), outFR.size() * sizeof(BlockSolverContactFriction));
    if (!dBatches) { std::printf("  NOTE: no BDA -> skip\n"); return; }

    PxgConstraintPrepareDesc prepDesc;
    std::memset(&prepDesc, 0xCD, sizeof(prepDesc)); // untouched fields must not matter
    prepDesc.contactConstraintBatchIndices = reinterpret_cast<PxU32*>(dBi);
    prepDesc.solverBodyDataPool = reinterpret_cast<PxgSolverBodyData*>(dBd);
    prepDesc.solverBodyTxIDataPool = reinterpret_cast<PxgSolverTxIData*>(dTx);
    prepDesc.blockWorkUnit = reinterpret_cast<PxgBlockWorkUnit*>(dWu);
    prepDesc.blockCurrentFrictionIndices = reinterpret_cast<PxgBlockFrictionIndex*>(dCfi);
    prepDesc.blockPreviousFrictionIndices = NULL; // no previous frame (== hasPrevFrictionIndices false)
    prepDesc.blockContactCurrentPrepPool = reinterpret_cast<physx::PxgBlockContactData*>(dCd);
    prepDesc.blockContactPoints = reinterpret_cast<physx::PxgBlockContactPoint*>(dCp);
    prepDesc.blockCurrentAnchorPatches = reinterpret_cast<physx::PxgBlockFrictionAnchorPatch*>(dAnch);
    prepDesc.blockPreviousAnchorPatches = NULL;
    prepDesc.body2WorldPool = reinterpret_cast<PxAlignedTransform*>(dBf);
    // split the on-device batch-count sum (numContactBatches +
    // numStaticContactBatches) to pin the full expression
    prepDesc.numContactBatches = cs.numBatches - 2;
    prepDesc.numStaticContactBatches = 2;
    prepDesc.bounceThresholdF32 = bounceThreshold;
    prepDesc.frictionOffsetThreshold = frictionOffsetThreshold;
    prepDesc.correlationDistance = correlationDistance;
    prepDesc.ccdMaxSeparation = ccdMaxSeparation;
    prepDesc.totalPreviousEdges = 0;
    prepDesc.totalCurrentEdges = cs.numBatches * 32;

    PxgSolverSharedDesc<IterativeSolveData> sharedDesc;
    std::memset(&sharedDesc, 0xCD, sizeof(sharedDesc));
    sharedDesc.blockCurrentFrictionPatches = reinterpret_cast<physx::PxgBlockFrictionPatch*>(dPatch);
    sharedDesc.blockPreviousFrictionPatches = NULL;
    sharedDesc.dt = dt;
    sharedDesc.invDtF32 = invDt;
    sharedDesc.iterativeData.blockConstraintBatch = reinterpret_cast<PxgBlockConstraintBatch*>(dBatches);
    sharedDesc.iterativeData.blockContactHeaders = reinterpret_cast<PxgBlockSolverContactHeader*>(dCh);
    sharedDesc.iterativeData.blockFrictionHeaders = reinterpret_cast<PxgBlockSolverFrictionHeader*>(dFh);
    sharedDesc.iterativeData.blockContactPoints = reinterpret_cast<PxgBlockSolverContactPoint*>(dOutCP);
    sharedDesc.iterativeData.blockFrictions = reinterpret_cast<PxgBlockSolverContactFriction*>(dOutFR);

    const PxgDevicePtr dPrepDesc = alloc(&prepDesc, sizeof(prepDesc));
    const PxgDevicePtr dSharedDesc = alloc(&sharedDesc, sizeof(sharedDesc));

    // engine launch: 2 warps/block (CONSTRAINT_PREPARE_BLOCK_PARALLEL = 64)
    const uint32_t nbWarpsPerBlock = 64 / 32;
    const uint32_t nbBlocks = (cs.numBatches + nbWarpsPerBlock - 1) / nbWarpsPerBlock;
    struct { uint64_t prepDesc, sharedDesc; } prm = { dPrepDesc, dSharedDesc };
    PXG_CHECK_MSG(shim.launchKernel(0, nbBlocks, 1, 1, &prm, sizeof(prm)) == PxgResult::Success,
                  "launch contactConstraintBlockPrepareParallelLaunch");
    shim.streamSynchronize();

    // ---- reference ----
    std::vector<BlockFrictionPatch> refPatches = currPatches;
    std::vector<BlockFrictionAnchorPatch> refAnchors = currAnchors;
    std::vector<BlockSolverContactHeader> refCH = chs;
    std::vector<BlockSolverFrictionHeader> refFH = fhs;
    std::vector<BlockSolverContactPoint> refCP = outCP;
    std::vector<BlockSolverContactFriction> refFR = outFR;
    std::vector<BlockFrictionPatch> prevPatchesDummy;
    std::vector<BlockFrictionAnchorPatch> prevAnchorsDummy;
    std::vector<uint64_t> prevIdxDummy;

    for (uint32_t k = 0; k < cs.numBatches; ++k)
    {
        const Batch& b = cs.batches[k];
        for (uint32_t lane = 0; lane < b.descStride; ++lane)
        {
            const uint32_t bA = b.bodyAIndex[lane], bB = b.bodyBIndex[lane];
            const AFrame& bf0 = cs.frames[bA];
            const AFrame& bf1 = cs.frames[bB];

            const Float4 data_ = cs.cdata[k].contactData[lane];
            uint32_t materialWord;
            std::memcpy(&materialWord, &data_.w, 4);
            const uint32_t nbContacts = materialWord & 0xFF;
            const uint32_t flags = (materialWord >> 8) & 0xFF;
            const float restDistance = data_.x, staticF = data_.y, dynamicF = data_.z;
            const Float4 n4 = cs.cdata[k].normal_restitutionW[lane];
            const Vec3 normal(n4.x, n4.y, n4.z);
            const uint32_t contactBase = b.blockContactIndex;

            refCorrelatePatches(refPatches[k], lane, nbContacts, normal, bf0, bf1);
            const bool perPointFriction = (flags & SC_PER_POINT_FRICTION) != 0;
            const bool disableFriction = (flags & SC_DISABLE_FRICTION) != 0;
            refPatches[k].broken[lane] = 0;
            float patchExtents = 0.f;
            if (!(perPointFriction || disableFriction))
                refGetFrictionPatches(refPatches[k], refAnchors[k], lane, prevIdxDummy, false, cs.workUnits[k].mEdgeIndex[lane],
                                      prevPatchesDummy, prevAnchorsDummy, 0, bf0, bf1, correlationDistance,
                                      /*totalPreviousEdges*/ 0u, patchExtents, nbContacts);
            if (!disableFriction)
                refGrowPatches(refPatches[k], refAnchors[k], lane, cs.cpoints, contactBase, nbContacts, bf0, bf1,
                               frictionOffsetThreshold + restDistance, patchExtents);

            refSetupPGS(cs.cdata[k], cs.cpoints, contactBase, nbContacts, refPatches[k], refAnchors[k], bf0, bf1,
                        cs.bodyData[bA], cs.bodyData[bB], cs.si[bA], cs.si[bB], invDt, dt, bounceThreshold, lane,
                        cs.workUnits[k].mWriteback[lane], perPointFriction, refCH[k], refFH[k], refCP, contactBase,
                        refFR, b.startFrictionIndex, restDistance, staticF, dynamicF, flags, ccdMaxSeparation,
                        std::fmax(cs.bodyData[bA].offsetSlop, cs.bodyData[bB].offsetSlop));

            refPatches[k].patchIndex[lane] = cs.workUnits[k].mFrictionPatchIndex[lane];
            if (refPatches[k].anchorCount[lane] >= 1)
            {
                const Float4 a = refAnchors[k].body0Anchors0[lane];
                const Vec3 w = aTransform(bf0, Vec3(a.x, a.y, a.z));
                refPatches[k].anchorPoints0[lane] = Float4{ w.x, w.y, w.z, 0.f };
            }
            if (refPatches[k].anchorCount[lane] == 2)
            {
                const Float4 a = refAnchors[k].body0Anchors1[lane];
                const Vec3 w = aTransform(bf0, Vec3(a.x, a.y, a.z));
                refPatches[k].anchorPoints1[lane] = Float4{ w.x, w.y, w.z, 0.f };
            }
        }
    }

    // ---- compare ----
    std::vector<BlockSolverContactHeader> gCH(chs.size());
    std::vector<BlockSolverFrictionHeader> gFH(fhs.size());
    std::vector<BlockSolverContactPoint> gCP(outCP.size());
    std::vector<BlockSolverContactFriction> gFR(outFR.size());
    std::vector<BlockFrictionPatch> gPatch(currPatches.size());
    std::vector<BlockFrictionAnchorPatch> gAnch(currAnchors.size());
    shim.memcpyDtoH(gCH.data(), dCh, gCH.size() * sizeof(BlockSolverContactHeader));
    shim.memcpyDtoH(gFH.data(), dFh, gFH.size() * sizeof(BlockSolverFrictionHeader));
    shim.memcpyDtoH(gCP.data(), dOutCP, gCP.size() * sizeof(BlockSolverContactPoint));
    shim.memcpyDtoH(gFR.data(), dOutFR, gFR.size() * sizeof(BlockSolverContactFriction));
    shim.memcpyDtoH(gPatch.data(), dPatch, gPatch.size() * sizeof(BlockFrictionPatch));
    shim.memcpyDtoH(gAnch.data(), dAnch, gAnch.size() * sizeof(BlockFrictionAnchorPatch));

    size_t mism = 0;
    for (uint32_t k = 0; k < cs.numBatches && mism < 12; ++k)
        for (uint32_t l = 0; l < cs.batches[k].descStride && mism < 12; ++l)
        {
            if (gCH[k].numNormalConstr[l] != refCH[k].numNormalConstr[l]) { PXG_CHECK_MSG(false, "b%u l%u numNormal g%u r%u", k, l, gCH[k].numNormalConstr[l], refCH[k].numNormalConstr[l]); ++mism; }
            if (gCH[k].forceWritebackOffset[l] != refCH[k].forceWritebackOffset[l]) { ++mism; }
            if (gCH[k].numNormalConstr[l] == 0) continue;
            if (gCH[k].flags[l] != refCH[k].flags[l]) ++mism;
            if (!eq4(gCH[k].invMass0_1_angDom0_1[l], refCH[k].invMass0_1_angDom0_1[l])) ++mism;
            if (!eq4(gCH[k].normal_staticFriction[l], refCH[k].normal_staticFriction[l])) ++mism;
            if (!eq(gCH[k].restitution[l], refCH[k].restitution[l])) ++mism;
            if (gFH[k].numFrictionConstr[l] != refFH[k].numFrictionConstr[l]) { PXG_CHECK_MSG(false, "b%u l%u numFric g%u r%u", k, l, gFH[k].numFrictionConstr[l], refFH[k].numFrictionConstr[l]); ++mism; }
            if (!eq(gFH[k].dynamicFriction[l], refFH[k].dynamicFriction[l])) ++mism;
            if (gFH[k].numFrictionConstr[l] != 0)
            {
                if (!eq3(gFH[k].frictionNormals0[l], refFH[k].frictionNormals0[l])) ++mism;
                if (!eq3(gFH[k].frictionNormals1[l], refFH[k].frictionNormals1[l])) ++mism;
                if (gFH[k].broken[l] != refFH[k].broken[l]) ++mism;
            }
            if (gPatch[k].anchorCount[l] != refPatches[k].anchorCount[l]) { PXG_CHECK_MSG(false, "b%u l%u anchors g%u r%u", k, l, gPatch[k].anchorCount[l], refPatches[k].anchorCount[l]); ++mism; }
            if (gPatch[k].patchIndex[l] != refPatches[k].patchIndex[l]) ++mism;
        }
    PXG_CHECK_MSG(mism == 0, "%zu header/patch mismatches", mism);

    // P9.86c: rbXn_maxImpulse.w is a pure selection of input floats (two PxMin selects), so
    // it is compared BITWISE on every row; so are targetVelocity/coeff0/coeff1 of the resting
    // lane, whose values are exact signed zeros.  The twins model the old port spellings.
    auto bitsOf = [](float f) { uint32_t u; std::memcpy(&u, &f, 4); return u; };
    auto selMin = [](float a, float b) { return (a < b) ? a : b; }; // PxMin<float>
    auto lpFMin = [](float a, float b) {                               // llvmpipe FMin (F4)
        return (a == 0.f && b == 0.f) ? (std::signbit(a) ? a : b) : ((a < b) ? a : b);
    };
    size_t maxImpBits = 0, bodyTieRows = 0, pointTieRows = 0, orderTieRows = 0;
    size_t restBits = 0, restNegZeroRows = 0;
    mism = 0;
    for (uint32_t k = 0; k < cs.numBatches; ++k)
        for (uint32_t l = 0; l < cs.batches[k].descStride; ++l)
        {
            const uint32_t nc = refCH[k].numNormalConstr[l];
            for (uint32_t i = 0; i < nc; ++i)
            {
                const uint32_t r = cs.batches[k].startConstraintIndex + i;
                if (!eq4(gCP[r].raXn_targetVelocity[l], refCP[r].raXn_targetVelocity[l])) ++mism;
                if (!eq4(gCP[r].rbXn_maxImpulse[l], refCP[r].rbXn_maxImpulse[l])) ++mism;
                if (!eq(gCP[r].resp0[l], refCP[r].resp0[l])) ++mism;
                if (!eq(gCP[r].resp1[l], refCP[r].resp1[l])) ++mism;
                if (!eq(gCP[r].coeff0[l], refCP[r].coeff0[l])) ++mism;
                if (!eq(gCP[r].coeff1[l], refCP[r].coeff1[l])) ++mism;

                const uint32_t rw = bitsOf(refCP[r].rbXn_maxImpulse[l].w);
                if (bitsOf(gCP[r].rbXn_maxImpulse[l].w) != rw) ++maxImpBits;
                const float d0 = cs.bodyData[cs.batches[k].bodyAIndex[l]].maxImpulse;
                const float d1 = cs.bodyData[cs.batches[k].bodyBIndex[l]].maxImpulse;
                const Float4 tv4 = cs.cpoints[cs.batches[k].blockContactIndex + i].targetVel_maxImpulseW[l];
                if (bitsOf(selMin(lpFMin(d0, d1), tv4.w)) != rw) ++bodyTieRows;  // old min() at :258 (cuh:402)
                if (bitsOf(lpFMin(selMin(d0, d1), tv4.w)) != rw) ++pointTieRows; // old min() at :358 (cuh:513)
                if (bitsOf(selMin(selMin(d1, d0), tv4.w)) != rw) ++orderTieRows; // operands swapped
                if (k % 2u == 0u && l == 2u) // the resting lane
                {
                    if (bitsOf(gCP[r].raXn_targetVelocity[l].w) != bitsOf(refCP[r].raXn_targetVelocity[l].w)) ++restBits;
                    if (bitsOf(gCP[r].coeff0[l]) != bitsOf(refCP[r].coeff0[l])) ++restBits;
                    if (bitsOf(gCP[r].coeff1[l]) != bitsOf(refCP[r].coeff1[l])) ++restBits;
                    const Float4 n4 = cs.cdata[k].normal_restitutionW[l];
                    const float cTargetVel = n4.x * tv4.x + n4.y * tv4.y + n4.z * tv4.z;
                    if (bitsOf(cTargetVel) == 0x80000000u && bitsOf(refCP[r].raXn_targetVelocity[l].w) == 0u)
                        ++restNegZeroRows;
                }
            }
            const uint32_t nf = refFH[k].numFrictionConstr[l];
            for (uint32_t i = 0; i < nf; ++i)
            {
                const uint32_t r = cs.batches[k].startFrictionIndex + i;
                if (!eq4(gFR[r].raXn_bias[l], refFR[r].raXn_bias[l])) ++mism;
                if (!eq4(gFR[r].rbXn_targetVelW[l], refFR[r].rbXn_targetVelW[l])) ++mism;
                if (!eq(gFR[r].resp0[l], refFR[r].resp0[l])) ++mism;
                if (!eq(gFR[r].resp1[l], refFR[r].resp1[l])) ++mism;
            }
        }
    PXG_CHECK_MSG(mism == 0, "%zu contact/friction row mismatches", mism);
    std::printf("[ info ] PGS rbXn_maxImpulse.w: body-tie rows %zu, point-tie rows %zu, order rows %zu, "
                "bit mismatches %zu; resting rows (cTargetVel -0 -> +0) %zu, bit mismatches %zu\n",
                bodyTieRows, pointTieRows, orderTieRows, maxImpBits, restNegZeroRows, restBits);
    PXG_CHECK_MSG(maxImpBits == 0, "%zu rbXn_maxImpulse.w words differ bitwise", maxImpBits);
    PXG_CHECK_MSG(bodyTieRows > 0 && pointTieRows > 0 && orderTieRows > 0,
                  "the fixture must reach every PxMin zero-tie shape (%zu, %zu, %zu)", bodyTieRows,
                  pointTieRows, orderTieRows);
    PXG_CHECK_MSG(restBits == 0, "%zu resting-lane targetVelocity/coeff words differ bitwise", restBits);
    PXG_CHECK_MSG(restNegZeroRows > 0, "the fixture must reach cTargetVel = -0 on a resting pair");

    mism = 0;
    for (uint32_t k = 0; k < cs.numBatches; ++k)
        for (uint32_t l = 0; l < cs.batches[k].descStride; ++l)
            if (gPatch[k].anchorCount[l] >= 1)
            {
                if (!eq3(gPatch[k].anchorPoints0[l], refPatches[k].anchorPoints0[l])) ++mism;
                if (gPatch[k].anchorCount[l] == 2 && !eq3(gPatch[k].anchorPoints1[l], refPatches[k].anchorPoints1[l])) ++mism;
                if (!eq3(gAnch[k].body0Anchors0[l], refAnchors[k].body0Anchors0[l])) ++mism;
            }
    PXG_CHECK_MSG(mism == 0, "%zu anchor mismatches", mism);

    for (PxgDevicePtr pfree : { dBatches, dBi, dBd, dTx, dBf, dWu, dCd, dCp, dPatch, dAnch, dCfi, dCh,
                                dFh, dOutCP, dOutFR, dPrepDesc, dSharedDesc })
        shim.memFree(pfree);
}

PXG_TEST(constraintPrepTGS_contact)
{
    using namespace physx;
    using physx::gpuslang::PxgCudaShim;
    using physx::gpuslang::PxgKernelDef;
    using physx::gpuslang::PxgDevicePtr;
    using physx::gpuslang::PxgResult;

    std::mt19937 rng(4200);
    ContactSetup cs = buildContactSetup(rng);
    // P9.86: kinematic bodies at rest, with +0 or -0 angular velocity, so the
    // torsional row's zero-seeded targetVel (contactConstraintBlockPrep.cuh:1138-1142)
    // meets `0 + (-0)` and `0 - (+0)`.  Compared bitwise below.
    cs.bodyData[1].flags &= ~RB_KINEMATIC;
    for (uint32_t b = 0; b < cs.numBodies; b += 3)
    {
        cs.bodyData[b].flags |= RB_KINEMATIC;
        const float z = (b & 1u) ? -0.0f : 0.0f;
        Float4& av = cs.bodyData[b].initialAngVelXYZ_penBiasClamp;
        av = Float4{ z, z, z, av.w };
    }
    for (uint32_t k = 0; k < cs.numBatches; ++k)
        for (uint32_t lane = 0; lane < 32; ++lane)
        {
            if (lane % 2u == 0u)
                cs.workUnits[k].mTorsionalFrictionData[lane].x = 0.05f + 0.001f * float(lane);
            // directed pairs: body 3 is kinematic with -0 angVel, body 0 kinematic with
            // +0, body 1 dynamic -- lane%4==0: (kin -0, dyn), lane%4==2: (dyn, kin +0)
            if (lane % 4u == 0u) { cs.batches[k].bodyAIndex[lane] = 3; cs.batches[k].bodyBIndex[lane] = 1; }
            if (lane % 4u == 2u) { cs.batches[k].bodyAIndex[lane] = 1; cs.batches[k].bodyBIndex[lane] = 0; }
            // an all-positive normal makes n . (-0,-0,-0) = -0 (every product -0)
            if (lane % 4u == 0u)
            {
                Float4& n4 = cs.cdata[k].normal_restitutionW[lane];
                n4.x = std::fabs(n4.x); n4.y = std::fabs(n4.y); n4.z = std::fabs(n4.z);
            }
        }
    // P9.86c: signed-zero directed lanes 1 and 3 (live in every batch: descStride >= 4) and a
    // bounce lane 5 (where descStride > 5).  No rng consumed; compared bitwise below.
    //  * max contact impulses -0 (body 4) and +0 (body 5): setMaxContactImpulse(-0.0f) passes
    //    the API's `maxImpulse >= 0.f` check (NpRigidBodyTemplate.h:820) and preIntegration
    //    copies it raw.  Lane 1 (A -0, B +0): PxMin(-0,+0) = +0 (cuh:846) and every point
    //    limit is 50..100.  Lane 3 (A +0, B -0): PxMin(+0,-0) = -0, and every contact of the
    //    lane is ignored by contact modification (w = +0): PxMin(-0,+0) = +0 (cuh:969).
    //    FMin gives -0 for both ties.
    //  * exactly touching contacts, restDistance +0, torsional patch radius 0.05 with the
    //    default minimum +0 and one anchor: lane 1 has one contact at separation +0, so
    //    maxPenetration = +0 and PxMax(0, -0) = -0 -> sqrt(-0 * R) = -0 -> PxMax(+0, -0) = -0
    //    is stored (cuh:1128; FMax gives +0).  Lane 3 has a second contact at the same point
    //    with separation -0: PxMin(-0, +0) keeps +0 (cuh:947; FMin gives -0 -> +0 stored).
    //  * both lanes are resting-contact shapes: restitution 0, zero target velocity, an
    //    all-negative normal (cTargetVel = -0) and two dynamic bodies, so the CUDA's
    //    `cTargetVel + 0.f` (cuh:951) stores +0.
    //  * lane 5, a bouncing pair with exact dot products (normal +y, zero angular velocities):
    //    cTargetVel and vrel are exact, so targetVelocity = cTargetVel + (-vrel) * restitution
    //    is one rounded product and one rounded add -- the mirror's -ffp-contract=off order.
    //    The target velocities are picked so that an FMA would round differently; this pins
    //    the bounce path of cuh:951 bitwise (on llvmpipe even an explicit fma() rounds twice).
    const uint32_t kImpNeg = 4, kImpPos = 5, kBncA = 7, kBncB = 8;
    const float kBncNegVrel = 3.25f, kBncRest = 0.3f; // vrel = -1.25 - 2.0
    cs.bodyData[kImpNeg].maxImpulse = -0.0f;
    cs.bodyData[kImpPos].maxImpulse = 0.0f;
    for (uint32_t rb : { kImpNeg, kImpPos, kBncA, kBncB })
        cs.bodyData[rb].flags &= ~RB_KINEMATIC;
    {
        Float4& lA = cs.bodyData[kBncA].initialLinVelXYZ_invMassW;
        Float4& lB = cs.bodyData[kBncB].initialLinVelXYZ_invMassW;
        Float4& aA = cs.bodyData[kBncA].initialAngVelXYZ_penBiasClamp;
        Float4& aB = cs.bodyData[kBncB].initialAngVelXYZ_penBiasClamp;
        lA = Float4{ 0.25f, -1.25f, 0.5f, lA.w };
        lB = Float4{ -0.5f, 2.0f, 0.125f, lB.w };
        aA = Float4{ 0.f, 0.f, 0.f, aA.w };
        aB = Float4{ 0.f, 0.f, 0.f, aB.w };
    }
    auto fmaSensitive = [&](float cT) {
        const float unf = cT + kBncNegVrel * kBncRest, fus = std::fma(kBncNegVrel, kBncRest, cT);
        uint32_t a, b; std::memcpy(&a, &unf, 4); std::memcpy(&b, &fus, 4);
        return a != b;
    };
    for (uint32_t k = 0; k < cs.numBatches; ++k)
    {
        Batch& b = cs.batches[k];
        for (uint32_t lane = 1; lane <= 5; lane += 2)
        {
            if (lane >= b.descStride)
                continue;
            const bool l1 = (lane == 1u), l5 = (lane == 5u);
            b.bodyAIndex[lane] = l5 ? kBncA : (l1 ? kImpNeg : kImpPos);
            b.bodyBIndex[lane] = l5 ? kBncB : (l1 ? kImpPos : kImpNeg);
            Float4& cd = cs.cdata[k].contactData[lane];
            uint32_t mw; std::memcpy(&mw, &cd.w, 4);
            const uint32_t fl = ((mw >> 8) & 0xFFu) & ~(SC_DISABLE_FRICTION | SC_PER_POINT_FRICTION);
            mw = (l5 ? cs.rowsPerBatch : (l1 ? 1u : 2u)) | (fl << 8); // prevFrictionPatchCount 0
            std::memcpy(&cd.w, &mw, 4);
            cd.x = 0.0f; // restDistance +0
            Float4& n4 = cs.cdata[k].normal_restitutionW[lane];
            BlockContactPoint* cp = &cs.cpoints[b.blockContactIndex];
            if (l5)
            {
                n4 = Float4{ 0.f, 1.f, 0.f, kBncRest };
                float tvy = 0.1f;
                for (uint32_t i = 0; i < cs.rowsPerBatch; ++i)
                {
                    while (!fmaSensitive(tvy))
                        tvy += 0.0078125f;
                    cp[i].point_separationW[lane].w = 0.01f; // -vrel = 3.25 > 0.01 * 60
                    Float4& tv = cp[i].targetVel_maxImpulseW[lane];
                    tv = Float4{ 0.f, tvy, 0.f, tv.w };
                    tvy += 0.0078125f;
                }
                continue;
            }
            n4 = Float4{ -std::fabs(n4.x), -std::fabs(n4.y), -std::fabs(n4.z), 0.0f }; // restitution 0
            cs.workUnits[k].mTorsionalFrictionData[lane] = Float2{ 0.05f, 0.0f };
            const Float4 p0 = cp[0].point_separationW[lane];
            cp[0].point_separationW[lane] = Float4{ p0.x, p0.y, p0.z, 0.0f };
            if (!l1)
                cp[1].point_separationW[lane] = Float4{ p0.x, p0.y, p0.z, -0.0f }; // same point: one anchor
            for (uint32_t i = 0; i < cs.rowsPerBatch; ++i)
            {
                Float4& tv = cp[i].targetVel_maxImpulseW[lane];
                tv = Float4{ 0.0f, 0.0f, 0.0f, l1 ? tv.w : 0.0f };
            }
        }
    }

    const float stepDt = 1.f / 240.f, invStepDt = 240.f, totalDt = 1.f / 60.f, invTotalDt = 60.f;
    const float bounceThreshold = -2.f, frictionOffsetThreshold = 0.04f, correlationDistance = 0.025f;
    const float ccdMaxSeparation = 0.1f, biasCoefficient = 2.f;

    PxgSlangComputeDevice* dev = testDevice();

    std::vector<BlockFrictionPatch> currPatches(cs.numBatches);
    std::vector<BlockFrictionAnchorPatch> currAnchors(cs.numBatches);
    std::memset(currPatches.data(), 0, currPatches.size() * sizeof(BlockFrictionPatch));
    std::memset(currAnchors.data(), 0, currAnchors.size() * sizeof(BlockFrictionAnchorPatch));

    std::vector<TGSContactHeader> chs(cs.numBatches);
    std::vector<TGSFrictionHeader> fhs(cs.numBatches);
    std::vector<TGSContactPoint> outCP(cs.numBatches * cs.rowsPerBatch);
    std::vector<TGSFriction> outFR(cs.numBatches * (cs.rowsPerBatch * 2 + 1));
    std::memset(chs.data(), 0, chs.size() * sizeof(TGSContactHeader));
    std::memset(fhs.data(), 0, fhs.size() * sizeof(TGSFrictionHeader));
    std::memset(outCP.data(), 0, outCP.size() * sizeof(TGSContactPoint));
    std::memset(outFR.data(), 0, outFR.size() * sizeof(TGSFriction));

    std::vector<uint64_t> currFricIdx(cs.numBatches * 32 + 1, 0);

    // ---- ENGINE ABI: live device buffers + REAL engine descriptors, launched
    // through PxgCudaShim exactly as PxgTGSCudaSolverCore::
    // contactConstraintPrepareParallel does ----
    static std::vector<uint8_t> (*tgsLoader)(const char*) = [](const char* f) { return loadShaderBlob(f); };
    const PxgKernelDef manifest[] = {
        { "constraintPrepTGS.spv", "contactConstraintBlockPrepareParallelLaunchTGS", 16u, 0 },
    };
    PxgCudaShim shim(dev, manifest, 1, tgsLoader);

    auto alloc = [&](const void* data, size_t bytes) -> PxgDevicePtr {
        PxgDevicePtr dp = 0;
        shim.memAlloc(&dp, bytes);
        if (dp && data)
            shim.memcpyHtoD(dp, data, bytes);
        return dp;
    };

    const PxgDevicePtr dBatches = alloc(cs.batches.data(), cs.batches.size() * sizeof(Batch));
    const PxgDevicePtr dBi = alloc(cs.batchIndices.data(), cs.batchIndices.size() * 4);
    const PxgDevicePtr dBd = alloc(cs.bodyData.data(), cs.bodyData.size() * sizeof(SolverBodyData));
    const PxgDevicePtr dTx = alloc(cs.txi.data(), cs.txi.size() * sizeof(SolverTxIData));
    const PxgDevicePtr dBf = alloc(cs.bodyFrames.data(), cs.bodyFrames.size() * sizeof(AlignedTransform));
    const PxgDevicePtr dWu = alloc(cs.workUnits.data(), cs.workUnits.size() * sizeof(BlockWorkUnit));
    const PxgDevicePtr dCd = alloc(cs.cdata.data(), cs.cdata.size() * sizeof(BlockContactData));
    const PxgDevicePtr dCp = alloc(cs.cpoints.data(), cs.cpoints.size() * sizeof(BlockContactPoint));
    const PxgDevicePtr dPatch = alloc(currPatches.data(), currPatches.size() * sizeof(BlockFrictionPatch));
    const PxgDevicePtr dAnch = alloc(currAnchors.data(), currAnchors.size() * sizeof(BlockFrictionAnchorPatch));
    const PxgDevicePtr dCfi = alloc(currFricIdx.data(), currFricIdx.size() * 8);
    const PxgDevicePtr dCh = alloc(chs.data(), chs.size() * sizeof(TGSContactHeader));
    const PxgDevicePtr dFh = alloc(fhs.data(), fhs.size() * sizeof(TGSFrictionHeader));
    const PxgDevicePtr dOutCP = alloc(outCP.data(), outCP.size() * sizeof(TGSContactPoint));
    const PxgDevicePtr dOutFR = alloc(outFR.data(), outFR.size() * sizeof(TGSFriction));
    if (!dBatches) { std::printf("  NOTE: no BDA -> skip\n"); return; }

    PxgConstraintPrepareDesc prepDesc;
    std::memset(&prepDesc, 0xCD, sizeof(prepDesc)); // untouched fields must not matter
    prepDesc.contactConstraintBatchIndices = reinterpret_cast<PxU32*>(dBi);
    prepDesc.solverBodyDataPool = reinterpret_cast<PxgSolverBodyData*>(dBd);
    prepDesc.solverBodyTxIDataPool = reinterpret_cast<PxgSolverTxIData*>(dTx);
    prepDesc.blockWorkUnit = reinterpret_cast<PxgBlockWorkUnit*>(dWu);
    prepDesc.blockCurrentFrictionIndices = reinterpret_cast<PxgBlockFrictionIndex*>(dCfi);
    prepDesc.blockPreviousFrictionIndices = NULL; // no previous frame (== hasPrevFrictionIndices false)
    prepDesc.blockContactCurrentPrepPool = reinterpret_cast<physx::PxgBlockContactData*>(dCd);
    prepDesc.blockContactPoints = reinterpret_cast<physx::PxgBlockContactPoint*>(dCp);
    prepDesc.blockCurrentAnchorPatches = reinterpret_cast<physx::PxgBlockFrictionAnchorPatch*>(dAnch);
    prepDesc.blockPreviousAnchorPatches = NULL;
    prepDesc.body2WorldPool = reinterpret_cast<PxAlignedTransform*>(dBf);
    prepDesc.numContactBatches = cs.numBatches;
    prepDesc.numStaticContactBatches = 0;
    prepDesc.bounceThresholdF32 = bounceThreshold;
    prepDesc.frictionOffsetThreshold = frictionOffsetThreshold;
    prepDesc.correlationDistance = correlationDistance;
    prepDesc.ccdMaxSeparation = ccdMaxSeparation;
    prepDesc.totalPreviousEdges = 0;
    prepDesc.totalCurrentEdges = cs.numBatches * 32;
    prepDesc.biasCoefficient = biasCoefficient;

    PxgSolverSharedDesc<IterativeSolveDataTGS> sharedDesc;
    std::memset(&sharedDesc, 0xCD, sizeof(sharedDesc));
    sharedDesc.blockCurrentFrictionPatches = reinterpret_cast<physx::PxgBlockFrictionPatch*>(dPatch);
    sharedDesc.blockPreviousFrictionPatches = NULL;
    sharedDesc.dt = totalDt;
    sharedDesc.stepDt = stepDt;
    sharedDesc.invDtF32 = invTotalDt;
    sharedDesc.stepInvDtF32 = invStepDt;
    sharedDesc.iterativeData.blockConstraintBatch = reinterpret_cast<PxgBlockConstraintBatch*>(dBatches);
    sharedDesc.iterativeData.blockContactHeaders = reinterpret_cast<PxgTGSBlockSolverContactHeader*>(dCh);
    sharedDesc.iterativeData.blockFrictionHeaders = reinterpret_cast<PxgTGSBlockSolverFrictionHeader*>(dFh);
    sharedDesc.iterativeData.blockContactPoints = reinterpret_cast<PxgTGSBlockSolverContactPoint*>(dOutCP);
    sharedDesc.iterativeData.blockFrictions = reinterpret_cast<PxgTGSBlockSolverContactFriction*>(dOutFR);

    const PxgDevicePtr dPrepDesc = alloc(&prepDesc, sizeof(prepDesc));
    const PxgDevicePtr dSharedDesc = alloc(&sharedDesc, sizeof(sharedDesc));

    // engine launch: 2 warps/block (CONSTRAINT_PREPARE_BLOCK_PARALLEL = 64)
    const uint32_t nbWarpsPerBlock = 64 / 32;
    const uint32_t nbBlocks = (cs.numBatches + nbWarpsPerBlock - 1) / nbWarpsPerBlock;
    struct { uint64_t prepDesc, sharedDesc; } prm = { dPrepDesc, dSharedDesc };
    PXG_CHECK_MSG(shim.launchKernel(0, nbBlocks, 1, 1, &prm, sizeof(prm)) == PxgResult::Success,
                  "launch contactConstraintBlockPrepareParallelLaunchTGS");
    shim.streamSynchronize();

    // ---- reference ----
    std::vector<BlockFrictionPatch> refPatches = currPatches;
    std::vector<BlockFrictionAnchorPatch> refAnchors = currAnchors;
    std::vector<TGSContactHeader> refCH = chs;
    std::vector<TGSFrictionHeader> refFH = fhs;
    std::vector<TGSContactPoint> refCP = outCP;
    std::vector<TGSFriction> refFR = outFR;
    std::vector<BlockFrictionPatch> prevPatchesDummy;
    std::vector<BlockFrictionAnchorPatch> prevAnchorsDummy;
    std::vector<uint64_t> prevIdxDummy;

    for (uint32_t k = 0; k < cs.numBatches; ++k)
    {
        const Batch& b = cs.batches[k];
        for (uint32_t lane = 0; lane < b.descStride; ++lane)
        {
            const uint32_t bA = b.bodyAIndex[lane], bB = b.bodyBIndex[lane];
            const AFrame& bf0 = cs.frames[bA];
            const AFrame& bf1 = cs.frames[bB];
            const Float4 data_ = cs.cdata[k].contactData[lane];
            uint32_t materialWord;
            std::memcpy(&materialWord, &data_.w, 4);
            const uint32_t nbContacts = materialWord & 0xFF;
            const uint32_t flags = (materialWord >> 8) & 0xFF;
            const float restDistance = data_.x, staticF = data_.y, dynamicF = data_.z;
            const Float4 n4 = cs.cdata[k].normal_restitutionW[lane];
            const Vec3 normal(n4.x, n4.y, n4.z);
            const uint32_t contactBase = b.blockContactIndex;

            refCorrelatePatches(refPatches[k], lane, nbContacts, normal, bf0, bf1);
            const bool perPointFriction = (flags & SC_PER_POINT_FRICTION) != 0;
            const bool disableFriction = (flags & SC_DISABLE_FRICTION) != 0;
            refPatches[k].broken[lane] = 0;
            float patchExtents = 0.f;
            if (!(perPointFriction || disableFriction))
                refGetFrictionPatches(refPatches[k], refAnchors[k], lane, prevIdxDummy, false, cs.workUnits[k].mEdgeIndex[lane],
                                      prevPatchesDummy, prevAnchorsDummy, 0, bf0, bf1, correlationDistance,
                                      /*totalPreviousEdges*/ 0u, patchExtents, nbContacts);
            if (!disableFriction)
                refGrowPatches(refPatches[k], refAnchors[k], lane, cs.cpoints, contactBase, nbContacts, bf0, bf1,
                               frictionOffsetThreshold + restDistance, patchExtents);

            refSetupTGS(cs.cdata[k], cs.cpoints, contactBase, nbContacts, refPatches[k], refAnchors[k], bf0, bf1,
                        cs.bodyData[bA], cs.bodyData[bB], cs.si[bA], cs.si[bB], invStepDt, stepDt, totalDt, invTotalDt,
                        bounceThreshold, biasCoefficient, lane, cs.workUnits[k].mWriteback[lane], perPointFriction,
                        refCH[k], refFH[k], refCP, contactBase, refFR, b.startFrictionIndex, restDistance, staticF,
                        dynamicF, flags, std::fmax(cs.bodyData[bA].offsetSlop, cs.bodyData[bB].offsetSlop),
                        cs.workUnits[k].mTorsionalFrictionData[lane]);
        }
    }

    // ---- compare ----
    std::vector<TGSContactHeader> gCH(chs.size());
    std::vector<TGSFrictionHeader> gFH(fhs.size());
    std::vector<TGSContactPoint> gCP(outCP.size());
    std::vector<TGSFriction> gFR(outFR.size());
    shim.memcpyDtoH(gCH.data(), dCh, gCH.size() * sizeof(TGSContactHeader));
    shim.memcpyDtoH(gFH.data(), dFh, gFH.size() * sizeof(TGSFrictionHeader));
    shim.memcpyDtoH(gCP.data(), dOutCP, gCP.size() * sizeof(TGSContactPoint));
    shim.memcpyDtoH(gFR.data(), dOutFR, gFR.size() * sizeof(TGSFriction));

    size_t mism = 0, torsBits = 0, torsPlusNegZero = 0, torsMinusPosZero = 0;
    // P9.86c: maxImpulse is two PxMin selects of input floats and torsionalFrictionScale is a
    // select-min over `separation - restDistance`, a negation, two PxMax selects, one multiply
    // and a correctly rounded sqrt -- no fusable multiply-add -- so both are compared BITWISE
    // on every row.  The targetVelocity of the directed lanes (1, 3: exact signed zeros; 5:
    // exact dot products) is compared bitwise too.  The twins model the pre-P9.86 port
    // spellings: GLSL FMin/FMax as llvmpipe evaluates a mixed +-0 pair (F4: min -> -0,
    // max -> +0 in either operand order).
    auto bitsOf = [](float f) { uint32_t u; std::memcpy(&u, &f, 4); return u; };
    auto lpFMin = [](float a, float b) { return (a == 0.f && b == 0.f) ? (std::signbit(a) ? a : b) : cpSelMin(a, b); };
    auto lpFMax = [](float a, float b) { return (a == 0.f && b == 0.f) ? (std::signbit(a) ? b : a) : cpSelMax(a, b); };
    size_t maxImpBits = 0, maxImpTwin846 = 0, maxImpTwin969 = 0;
    size_t torsScaleBits = 0, torsTwin947 = 0, torsTwin1128 = 0;
    size_t tvBits = 0, tvPlusZero = 0, bounceFmaRows = 0;
    for (uint32_t k = 0; k < cs.numBatches && mism < 12; ++k)
        for (uint32_t l = 0; l < cs.batches[k].descStride && mism < 12; ++l)
        {
            if (gCH[k].numNormalConstr[l] != refCH[k].numNormalConstr[l]) { PXG_CHECK_MSG(false, "b%u l%u numNormal g%u r%u", k, l, gCH[k].numNormalConstr[l], refCH[k].numNormalConstr[l]); ++mism; }
            if (gCH[k].numNormalConstr[l] == 0) continue;
            if (gCH[k].flags[l] != refCH[k].flags[l]) ++mism;
            if (!eq(gCH[k].maxPenBias[l], refCH[k].maxPenBias[l])) ++mism;
            if (!eq(gCH[k].restitutionXdt[l], refCH[k].restitutionXdt[l])) ++mism;
            if (!eq(gCH[k].p8[l], refCH[k].p8[l])) ++mism;
            if (!eq4(gCH[k].invMass0_1_angDom0_1[l], refCH[k].invMass0_1_angDom0_1[l])) ++mism;
            if (!eq4(gCH[k].normal_staticFriction[l], refCH[k].normal_staticFriction[l])) ++mism;
            if (gFH[k].numFrictionConstr[l] != refFH[k].numFrictionConstr[l]) { PXG_CHECK_MSG(false, "b%u l%u numFric g%u r%u", k, l, gFH[k].numFrictionConstr[l], refFH[k].numFrictionConstr[l]); ++mism; }
            if (!eq(gFH[k].dynamicFriction[l], refFH[k].dynamicFriction[l])) ++mism;
            if (gFH[k].numFrictionConstr[l] != 0)
            {
                if (!eq(gFH[k].biasCoefficient[l], refFH[k].biasCoefficient[l])) ++mism;
                if (!eq3(gFH[k].frictionNormals0[l], refFH[k].frictionNormals0[l])) ++mism;
            }
        }
    PXG_CHECK_MSG(mism == 0, "%zu TGS header mismatches", mism);

    mism = 0;
    for (uint32_t k = 0; k < cs.numBatches; ++k)
        for (uint32_t l = 0; l < cs.batches[k].descStride; ++l)
        {
            const uint32_t nc = refCH[k].numNormalConstr[l];
            for (uint32_t i = 0; i < nc; ++i)
            {
                const uint32_t r = cs.batches[k].startConstraintIndex + i;
                if (!eq4(gCP[r].raXn_extraCoeff[l], refCP[r].raXn_extraCoeff[l])) ++mism;
                if (!eq4(gCP[r].rbXn_targetVelW[l], refCP[r].rbXn_targetVelW[l])) ++mism;
                if (!eq(gCP[r].separation[l], refCP[r].separation[l])) ++mism;
                const uint32_t rmi = bitsOf(refCP[r].maxImpulse[l]);
                if (bitsOf(gCP[r].maxImpulse[l]) != rmi) ++maxImpBits;
                const float d0 = cs.bodyData[cs.batches[k].bodyAIndex[l]].maxImpulse;
                const float d1 = cs.bodyData[cs.batches[k].bodyBIndex[l]].maxImpulse;
                const Float4 tv4 = cs.cpoints[cs.batches[k].blockContactIndex + i].targetVel_maxImpulseW[l];
                if (bitsOf(cpSelMin(lpFMin(d0, d1), tv4.w)) != rmi) ++maxImpTwin846; // old min() (cuh:846)
                if (bitsOf(lpFMin(cpSelMin(d0, d1), tv4.w)) != rmi) ++maxImpTwin969; // old min() (cuh:969)
                if (l == 1u || l == 3u || l == 5u) // the directed lanes (rewritten in every batch they exist)
                {
                    const uint32_t rtv = bitsOf(refCP[r].rbXn_targetVelW[l].w);
                    if (bitsOf(gCP[r].rbXn_targetVelW[l].w) != rtv) ++tvBits;
                    const Float4 n4 = cs.cdata[k].normal_restitutionW[l];
                    const float cT = n4.x * tv4.x + n4.y * tv4.y + n4.z * tv4.z;
                    if (l != 5u && bitsOf(cT) == 0x80000000u && rtv == 0u) ++tvPlusZero; // -0 + 0 = +0 (cuh:951)
                    if (l == 5u && fmaSensitive(cT) && rtv == bitsOf(cT + kBncNegVrel * kBncRest)) ++bounceFmaRows;
                }
                if (!eq(gCP[r].biasCoefficient[l], refCP[r].biasCoefficient[l])) ++mism;
                if (!eq(gCP[r].resp0[l], refCP[r].resp0[l])) ++mism;
                if (!eq(gCP[r].resp1[l], refCP[r].resp1[l])) ++mism;
            }
            const uint32_t nf = refFH[k].numFrictionConstr[l];
            for (uint32_t i = 0; i < nf; ++i)
            {
                const uint32_t r = cs.batches[k].startFrictionIndex + i;
                if (!eq4(gFR[r].raXn_error[l], refFR[r].raXn_error[l])) ++mism;
                if (!eq4(gFR[r].rbXn_targetVelW[l], refFR[r].rbXn_targetVelW[l])) ++mism;
                if (!eq(gFR[r].resp0[l], refFR[r].resp0[l])) ++mism;
                if (!eq(gFR[r].resp1[l], refFR[r].resp1[l])) ++mism;
            }
            if (refFH[k].numFrictionConstr[l] & 1) // torsional anchor present
            {
                const uint32_t rts = bitsOf(refFH[k].torsionalFrictionScale[l]);
                if (bitsOf(gFH[k].torsionalFrictionScale[l]) != rts) ++torsScaleBits;
                {
                    const Float2 td = cs.workUnits[k].mTorsionalFrictionData[l];
                    const float restD = cs.cdata[k].contactData[l].x;
                    float mpSel = PX_MAX_F32, mpLp = PX_MAX_F32;
                    for (uint32_t j = 0; j < nc; ++j)
                    {
                        const float pen = cs.cpoints[cs.batches[k].blockContactIndex + j].point_separationW[l].w - restD;
                        mpSel = cpSelMin(pen, mpSel);
                        mpLp = lpFMin(pen, mpLp);
                    }
                    // old min() at cuh:947, old max() pair at cuh:1128
                    if (bitsOf(cpSelMax(td.y, std::sqrt(cpSelMax(0.f, -mpLp) * td.x))) != rts) ++torsTwin947;
                    if (bitsOf(lpFMax(td.y, std::sqrt(lpFMax(0.f, -mpSel) * td.x))) != rts) ++torsTwin1128;
                }
                // P9.86: the torsional targetVel is the CUDA's own float sequence in the
                // reference, so it is compared BITWISE (the sign of a zero included)
                const uint32_t r = cs.batches[k].startFrictionIndex + 2;
                uint32_t gw, rw;
                std::memcpy(&gw, &gFR[r].rbXn_targetVelW[l].w, 4);
                std::memcpy(&rw, &refFR[r].rbXn_targetVelW[l].w, 4);
                if (gw != rw) ++torsBits;
                const bool k0 = (cs.bodyData[cs.batches[k].bodyAIndex[l]].flags & RB_KINEMATIC) != 0;
                const bool k1 = (cs.bodyData[cs.batches[k].bodyBIndex[l]].flags & RB_KINEMATIC) != 0;
                // the two shapes the port once folded: 0 + (-0) (kin0 only) and 0 - (+0) (kin1 only)
                if (k0 && !k1 && rw == 0u) {
                    const Float4 av = cs.bodyData[cs.batches[k].bodyAIndex[l]].initialAngVelXYZ_penBiasClamp;
                    const Float4 n4 = cs.cdata[k].normal_restitutionW[l];
                    const float d = n4.x * av.x + n4.y * av.y + n4.z * av.z;
                    uint32_t dw; std::memcpy(&dw, &d, 4);
                    if (dw == 0x80000000u) ++torsPlusNegZero;
                }
                if (k1 && !k0 && rw == 0u) ++torsMinusPosZero;
            }
        }
    std::printf("[ info ] TGS torsional targetVel: 0+(-0) rows %zu, 0-(+0) rows %zu, bit mismatches %zu\n",
                torsPlusNegZero, torsMinusPosZero, torsBits);
    PXG_CHECK_MSG(torsBits == 0, "%zu torsional targetVel words differ bitwise", torsBits);
    PXG_CHECK_MSG(torsPlusNegZero > 0 && torsMinusPosZero > 0,
                  "the fixture must reach both zero-seeded targetVel shapes (%zu, %zu)",
                  torsPlusNegZero, torsMinusPosZero);
    PXG_CHECK_MSG(mism == 0, "%zu TGS row mismatches", mism);
    std::printf("[ info ] TGS maxImpulse: cuh:846 tie rows %zu, cuh:969 tie rows %zu, bit mismatches %zu; "
                "torsionalFrictionScale: cuh:947 tie rows %zu, cuh:1128 tie rows %zu, bit mismatches %zu; "
                "targetVelocity: -0+0 rows %zu, FMA-sensitive bounce rows %zu, bit mismatches %zu\n",
                maxImpTwin846, maxImpTwin969, maxImpBits, torsTwin947, torsTwin1128, torsScaleBits, tvPlusZero,
                bounceFmaRows, tvBits);
    PXG_CHECK_MSG(maxImpBits == 0, "%zu maxImpulse words differ bitwise", maxImpBits);
    PXG_CHECK_MSG(torsScaleBits == 0, "%zu torsionalFrictionScale words differ bitwise", torsScaleBits);
    PXG_CHECK_MSG(tvBits == 0, "%zu directed-lane targetVelocity words differ bitwise", tvBits);
    PXG_CHECK_MSG(maxImpTwin846 > 0 && maxImpTwin969 > 0,
                  "the fixture must reach both maxImpulse PxMin zero ties (%zu, %zu)", maxImpTwin846, maxImpTwin969);
    PXG_CHECK_MSG(torsTwin947 > 0 && torsTwin1128 > 0,
                  "the fixture must reach both torsional zero ties (%zu, %zu)", torsTwin947, torsTwin1128);
    PXG_CHECK_MSG(tvPlusZero > 0, "the fixture must reach cTargetVel = -0 on a resting pair");
    PXG_CHECK_MSG(bounceFmaRows > 0, "the fixture must reach an FMA-sensitive bounce targetVelocity");

    // the current-frame friction patches/anchors are engine-ABI outputs too:
    // spot-check the write-through-desc path (anchor counts + patch indices)
    std::vector<BlockFrictionPatch> gPatch(currPatches.size());
    shim.memcpyDtoH(gPatch.data(), dPatch, gPatch.size() * sizeof(BlockFrictionPatch));
    mism = 0;
    for (uint32_t k = 0; k < cs.numBatches; ++k)
        for (uint32_t l = 0; l < cs.batches[k].descStride; ++l)
        {
            if (gPatch[k].anchorCount[l] != refPatches[k].anchorCount[l]) ++mism;
            if (gPatch[k].patchIndex[l] != cs.workUnits[k].mFrictionPatchIndex[l]) ++mism;
        }
    PXG_CHECK_MSG(mism == 0, "%zu TGS patch mismatches", mism);

    for (PxgDevicePtr h : { dBatches, dBi, dBd, dTx, dBf, dWu, dCd, dCp, dPatch, dAnch, dCfi, dCh, dFh,
                            dOutCP, dOutFR, dPrepDesc, dSharedDesc })
        shim.memFree(h);
}

// ---------------------------------------------------------------------------
// joint prep tests
// ---------------------------------------------------------------------------
struct JointSetup
{
    uint32_t numBatches, numBodies;
    std::vector<Batch> batches;
    std::vector<uint32_t> batchIndices;
    std::vector<SolverBodyData> bodyData;
    std::vector<SolverTxIData> txi;
    std::vector<Mat33> si;
    std::vector<BlockConstraint1DData> jdata;
    std::vector<BlockConstraint1DVelocities> jvel;
    std::vector<BlockConstraint1DParameters> jpar;
    std::vector<uint32_t> uniqueIndices;
    std::vector<uint64_t> constantData; // PxgSolverConstraintManagerConstants (2x uint32 -> uint64)
};

JointSetup buildJointSetup(std::mt19937& rng, bool tgs)
{
    std::uniform_real_distribution<float> u(0.f, 1.f), s(-1.f, 1.f);
    JointSetup js;
    js.numBatches = 6;
    js.numBodies = 48;
    const uint32_t startPartition = 4096;

    js.bodyData.resize(js.numBodies);
    js.txi.resize(js.numBodies);
    js.si.resize(js.numBodies);
    std::memset(js.bodyData.data(), 0, js.bodyData.size() * sizeof(SolverBodyData));
    std::memset(js.txi.data(), 0, js.txi.size() * sizeof(SolverTxIData));
    for (uint32_t b = 0; b < js.numBodies; ++b)
    {
        js.bodyData[b].initialLinVelXYZ_invMassW = Float4{ s(rng), s(rng), s(rng), u(rng) * 2.f };
        js.bodyData[b].initialAngVelXYZ_penBiasClamp = Float4{ s(rng), s(rng), s(rng), -u(rng) };
        js.bodyData[b].flags = (tgs && u(rng) < 0.15f) ? RB_KINEMATIC : 0u;
        const Mat33 m = randSymInvInertia(rng);
        js.si[b] = m;
        m.store(js.txi[b].sqrtInvInertia);
    }

    js.batches.resize(js.numBatches);
    js.batchIndices.resize(js.numBatches);
    js.jdata.resize(js.numBatches);
    js.jvel.resize(js.numBatches * MAX_CONSTRAINT_ROWS);
    js.jpar.resize(js.numBatches * MAX_CONSTRAINT_ROWS);
    js.uniqueIndices.resize(startPartition + js.numBatches * 32);
    js.constantData.resize(js.numBatches * 32 + 16);
    std::memset(js.batches.data(), 0, js.batches.size() * sizeof(Batch));
    std::memset(js.jdata.data(), 0, js.jdata.size() * sizeof(BlockConstraint1DData));
    std::memset(js.jvel.data(), 0, js.jvel.size() * sizeof(BlockConstraint1DVelocities));
    std::memset(js.jpar.data(), 0, js.jpar.size() * sizeof(BlockConstraint1DParameters));

    for (uint32_t i = 0; i < js.constantData.size(); ++i)
        js.constantData[i] = (uint64_t(i * 2 + 1) << 32) | uint64_t(i); // (mEdgeIndex, mConstraintWriteBackIndex)

    for (uint32_t k = 0; k < js.numBatches; ++k)
    {
        js.batchIndices[k] = k;
        Batch& b = js.batches[k];
        b.descStride = uint16_t(4 + rng() % 13);
        b.constraintType = 0;
        b.mConstraintBatchIndex = k;
        b.startConstraintIndex = k * MAX_CONSTRAINT_ROWS;
        b.mStartPartitionIndex = startPartition + k * 32;

        for (uint32_t lane = 0; lane < 32; ++lane)
        {
            b.bodyAIndex[lane] = rng() % js.numBodies;
            b.bodyBIndex[lane] = rng() % js.numBodies;
            js.uniqueIndices[b.mStartPartitionIndex + lane] = k * 32 + lane;

            const uint32_t numRows = 1 + rng() % 6;
            js.jdata[k].mNumRows[lane] = numRows;
            js.jdata[k].mInvMassScale[lane] = Float4{ 0.5f + u(rng), 0.5f + u(rng), 0.5f + u(rng), 0.5f + u(rng) };
            const bool breakable = u(rng) < 0.5f;
            const float linBreak = breakable ? 10.f + u(rng) * 100.f : PX_MAX_F32;
            const float angBreak = breakable ? 10.f + u(rng) * 100.f : PX_MAX_F32;
            js.jdata[k].mRAWorld_linBreakForce[lane] = Float4{ s(rng) * 0.3f, s(rng) * 0.3f, s(rng) * 0.3f, linBreak };
            js.jdata[k].mRBWorld_AngBreakForce[lane] = Float4{ s(rng) * 0.3f, s(rng) * 0.3f, s(rng) * 0.3f, angBreak };
            // bit 6 is UNDEFINED in PxConstraintFlag -- salted as noise to pin
            // that eDISABLE_PREPROCESSING is bit 8, not bit 6 (review-182 find)
            js.jdata[k].mFlags[lane] = (u(rng) < 0.2f ? CFLAG_DRIVE_LIMITS_ARE_FORCES : 0u) |
                                       (u(rng) < 0.2f ? CFLAG_DISABLE_PREPROCESSING : 0u) |
                                       (u(rng) < 0.1f ? (1u << 6) : 0u);

            for (uint32_t r = 0; r < numRows; ++r)
            {
                const uint32_t rowIdx = k * MAX_CONSTRAINT_ROWS + r;
                const Vec3 l0 = Vec3(s(rng), s(rng), s(rng)).getNormalized();
                const Vec3 l1 = Vec3(s(rng), s(rng), s(rng)).getNormalized();
                const Vec3 a0(s(rng) * 0.5f, s(rng) * 0.5f, s(rng) * 0.5f);
                const Vec3 a1(s(rng) * 0.5f, s(rng) * 0.5f, s(rng) * 0.5f);
                js.jvel[rowIdx].linear0XYZ_geometricErrorW[lane] = Float4{ l0.x, l0.y, l0.z, s(rng) * 0.1f };
                js.jvel[rowIdx].angular0XYZ_velocityTargetW[lane] = Float4{ a0.x, a0.y, a0.z, s(rng) * 0.5f };
                js.jvel[rowIdx].linear1XYZ_minImpulseW[lane] = Float4{ l1.x, l1.y, l1.z, -50.f - u(rng) * 50.f };
                js.jvel[rowIdx].angular1XYZ_maxImpulseW[lane] = Float4{ a1.x, a1.y, a1.z, 50.f + u(rng) * 50.f };

                uint32_t flags = 0;
                const float fr = u(rng);
                if (fr < 0.25f) flags |= PX1D_SPRING | (u(rng) < 0.5f ? PX1D_ACCELERATION_SPRING : 0u);
                else if (fr < 0.45f) flags |= PX1D_RESTITUTION;
                if (u(rng) < 0.3f) flags |= PX1D_OUTPUT_FORCE;
                if (u(rng) < 0.2f) flags |= PX1D_KEEPBIAS;
                if (u(rng) < 0.2f) flags |= PX1D_HAS_DRIVE_LIMIT;
                if (u(rng) < 0.4f) flags |= PX1D_ANGULAR_CONSTRAINT;
                js.jpar[rowIdx].flags[lane] = flags;
                js.jpar[rowIdx].springStiffness_restitution[lane] = (flags & PX1D_SPRING) ? (50.f + u(rng) * 500.f) : (u(rng) * 0.8f);
                js.jpar[rowIdx].springDamping_velocityThreshold[lane] = (flags & PX1D_SPRING) ? (1.f + u(rng) * 20.f) : (u(rng) * 2.f);
                // solve hints: mix of eq/ineq/rotational/zero to exercise orthogonalize
                const float hr = u(rng);
                uint32_t hint = 0;
                if (hr < 0.3f) hint = HINT_EQ;
                else if (hr < 0.45f) hint = HINT_INEQ;
                else if (hr < 0.55f) hint = HINT_ROT_EQ;
                js.jpar[rowIdx].solveHint[lane] = hint;
            }
        }
    }
    return js;
}

PXG_TEST(constraintPrepPGS_joint)
{
    using namespace physx;
    using physx::gpuslang::PxgCudaShim;
    using physx::gpuslang::PxgKernelDef;
    using physx::gpuslang::PxgDevicePtr;
    using physx::gpuslang::PxgResult;

    std::mt19937 rng(4300);
    JointSetup js = buildJointSetup(rng, false);

    const float dt = 1.f / 60.f, recipdt = 60.f;
    PxgSlangComputeDevice* dev = testDevice();

    std::vector<BlockSolverConstraint1DHeader> hdrs(js.numBatches);
    std::vector<BlockSolverConstraint1DCon> cons(js.numBatches * MAX_CONSTRAINT_ROWS);
    std::vector<BlockSolverConstraint1DMod> mods(js.numBatches * MAX_CONSTRAINT_ROWS);
    std::memset(hdrs.data(), 0, hdrs.size() * sizeof(BlockSolverConstraint1DHeader));
    std::memset(cons.data(), 0, cons.size() * sizeof(BlockSolverConstraint1DCon));
    std::memset(mods.data(), 0, mods.size() * sizeof(BlockSolverConstraint1DMod));

    // ---- ENGINE ABI (P5 Step 5): live device buffers + REAL engine
    // descriptors, launched through PxgCudaShim exactly as
    // PxgCudaSolverCore::jointConstraintPrepareParallel does ----
    static std::vector<uint8_t> (*pgsJointLoader)(const char*) = [](const char* f) { return loadShaderBlob(f); };
    const PxgKernelDef manifest[] = {
        { "constraintPrepPGS.spv", "jointConstraintBlockPrepareParallelLaunch", 16u, 0 },
    };
    PxgCudaShim shim(dev, manifest, 1, pgsJointLoader);

    auto alloc = [&](const void* data, size_t bytes) -> PxgDevicePtr {
        PxgDevicePtr dp = 0;
        shim.memAlloc(&dp, bytes);
        if (dp && data)
            shim.memcpyHtoD(dp, data, bytes);
        return dp;
    };

    const PxgDevicePtr dBatches = alloc(js.batches.data(), js.batches.size() * sizeof(Batch));
    const PxgDevicePtr dBi = alloc(js.batchIndices.data(), js.batchIndices.size() * 4);
    const PxgDevicePtr dBd = alloc(js.bodyData.data(), js.bodyData.size() * sizeof(SolverBodyData));
    const PxgDevicePtr dTx = alloc(js.txi.data(), js.txi.size() * sizeof(SolverTxIData));
    const PxgDevicePtr dJd = alloc(js.jdata.data(), js.jdata.size() * sizeof(BlockConstraint1DData));
    const PxgDevicePtr dJv = alloc(js.jvel.data(), js.jvel.size() * sizeof(BlockConstraint1DVelocities));
    const PxgDevicePtr dJp = alloc(js.jpar.data(), js.jpar.size() * sizeof(BlockConstraint1DParameters));
    const PxgDevicePtr dUi = alloc(js.uniqueIndices.data(), js.uniqueIndices.size() * 4);
    const PxgDevicePtr dSc = alloc(js.constantData.data(), js.constantData.size() * 8);
    const PxgDevicePtr dHdr = alloc(hdrs.data(), hdrs.size() * sizeof(BlockSolverConstraint1DHeader));
    const PxgDevicePtr dCon = alloc(cons.data(), cons.size() * sizeof(BlockSolverConstraint1DCon));
    const PxgDevicePtr dMod = alloc(mods.data(), mods.size() * sizeof(BlockSolverConstraint1DMod));
    if (!dBatches) { std::printf("  NOTE: no BDA -> skip\n"); return; }

    PxgConstraintPrepareDesc prepDesc;
    std::memset(&prepDesc, 0xCD, sizeof(prepDesc)); // untouched fields must not matter
    prepDesc.jointConstraintBatchIndices = reinterpret_cast<PxU32*>(dBi);
    prepDesc.solverBodyDataPool = reinterpret_cast<PxgSolverBodyData*>(dBd);
    prepDesc.solverBodyTxIDataPool = reinterpret_cast<PxgSolverTxIData*>(dTx);
    prepDesc.blockJointPrepPool = reinterpret_cast<PxgBlockConstraint1DData*>(dJd);
    prepDesc.blockJointPrepPool0 = reinterpret_cast<PxgBlockConstraint1DVelocities*>(dJv);
    prepDesc.blockJointPrepPool1 = reinterpret_cast<PxgBlockConstraint1DParameters*>(dJp);
    prepDesc.constraintUniqueIndices = reinterpret_cast<PxU32*>(dUi);
    prepDesc.solverConstantData = reinterpret_cast<PxgSolverConstraintManagerConstants*>(dSc);
    // split the on-device batch-count sum (num1dConstraintBatches +
    // numStatic1dConstraintBatches) to pin the full expression
    prepDesc.num1dConstraintBatches = js.numBatches - 1;
    prepDesc.numStatic1dConstraintBatches = 1;

    PxgSolverSharedDesc<IterativeSolveData> sharedDesc;
    std::memset(&sharedDesc, 0xCD, sizeof(sharedDesc));
    sharedDesc.dt = dt;
    sharedDesc.invDtF32 = recipdt;
    sharedDesc.iterativeData.blockConstraintBatch = reinterpret_cast<PxgBlockConstraintBatch*>(dBatches);
    sharedDesc.iterativeData.blockJointConstraintHeaders =
        reinterpret_cast<PxgBlockSolverConstraint1DHeader*>(dHdr);
    sharedDesc.iterativeData.blockJointConstraintRowsCon =
        reinterpret_cast<PxgBlockSolverConstraint1DCon*>(dCon);
    sharedDesc.iterativeData.blockJointConstraintRowsMod =
        reinterpret_cast<PxgBlockSolverConstraint1DMod*>(dMod);

    const PxgDevicePtr dPrepDesc = alloc(&prepDesc, sizeof(prepDesc));
    const PxgDevicePtr dSharedDesc = alloc(&sharedDesc, sizeof(sharedDesc));

    // engine launch: nbBlocks = ceil(nbJointBatches * 32 /
    // CONSTRAINT_PREPARE_BLOCK_PARALLEL = 64) = 2 warps/block
    const uint32_t nbBlocks = (js.numBatches * 32 + 63) / 64;
    struct { uint64_t prepDesc, sharedDesc; } prm = { dPrepDesc, dSharedDesc };
    PXG_CHECK_MSG(shim.launchKernel(0, nbBlocks, 1, 1, &prm, sizeof(prm)) == PxgResult::Success,
                  "launch jointConstraintBlockPrepareParallelLaunch");
    shim.streamSynchronize();

    // ---- reference ----
    std::vector<BlockConstraint1DVelocities> refVel = js.jvel;
    std::vector<BlockSolverConstraint1DHeader> refHdr = hdrs;
    std::vector<BlockSolverConstraint1DCon> refCon = cons;
    std::vector<BlockSolverConstraint1DMod> refMod = mods;

    for (uint32_t k = 0; k < js.numBatches; ++k)
    {
        const Batch& b = js.batches[k];
        for (uint32_t lane = 0; lane < b.descStride; ++lane)
        {
            const uint32_t bA = b.bodyAIndex[lane], bB = b.bodyBIndex[lane];
            const uint32_t uniq = js.uniqueIndices[b.mStartPartitionIndex + lane];
            const uint32_t wbIndex = uint32_t(js.constantData[uniq] >> 32);
            refJointPGS(js.jdata, k, refVel, js.jpar, k * MAX_CONSTRAINT_ROWS, js.bodyData[bA], js.bodyData[bB],
                        js.si[bA], js.si[bB], dt, recipdt, lane, wbIndex, refHdr[k], refCon, refMod,
                        b.startConstraintIndex);
        }
    }

    // ---- compare ----
    std::vector<BlockSolverConstraint1DHeader> gHdr(hdrs.size());
    std::vector<BlockSolverConstraint1DCon> gCon(cons.size());
    std::vector<BlockSolverConstraint1DMod> gMod(mods.size());
    std::vector<BlockConstraint1DVelocities> gVel(js.jvel.size());
    shim.memcpyDtoH(gHdr.data(), dHdr, gHdr.size() * sizeof(BlockSolverConstraint1DHeader));
    shim.memcpyDtoH(gCon.data(), dCon, gCon.size() * sizeof(BlockSolverConstraint1DCon));
    shim.memcpyDtoH(gMod.data(), dMod, gMod.size() * sizeof(BlockSolverConstraint1DMod));
    shim.memcpyDtoH(gVel.data(), dJv, gVel.size() * sizeof(BlockConstraint1DVelocities));

    auto u16 = [](const uint32_t* arr, uint32_t lane) { return (arr[lane >> 1] >> ((lane & 1) * 16)) & 0xFFFFu; };

    size_t mism = 0;
    for (uint32_t k = 0; k < js.numBatches && mism < 12; ++k)
        for (uint32_t l = 0; l < js.batches[k].descStride && mism < 12; ++l)
        {
            if (u16(gHdr[k].rowCounts, l) != u16(refHdr[k].rowCounts, l)) { PXG_CHECK_MSG(false, "b%u l%u rowCount g%u r%u", k, l, u16(gHdr[k].rowCounts, l), u16(refHdr[k].rowCounts, l)); ++mism; }
            if (u16(gHdr[k].breakable, l) != u16(refHdr[k].breakable, l)) ++mism;
            if (gHdr[k].writeBackOffset[l] != refHdr[k].writeBackOffset[l]) ++mism;
            if (!eq4(gHdr[k].body0WorldOffset_linBreakImpulse[l], refHdr[k].body0WorldOffset_linBreakImpulse[l])) ++mism;
            if (!eq(gHdr[k].angBreakImpulse[l], refHdr[k].angBreakImpulse[l])) ++mism;
            if (!eq(gHdr[k].invMass0D0[l], refHdr[k].invMass0D0[l])) ++mism;
            if (!eq(gHdr[k].invMass1D1[l], refHdr[k].invMass1D1[l])) ++mism;
            if (!eq(gHdr[k].invInertiaScale0[l], refHdr[k].invInertiaScale0[l])) ++mism;
            if (!eq(gHdr[k].invInertiaScale1[l], refHdr[k].invInertiaScale1[l])) ++mism;
        }
    PXG_CHECK_MSG(mism == 0, "%zu joint header mismatches", mism);

    mism = 0;
    for (uint32_t k = 0; k < js.numBatches; ++k)
        for (uint32_t l = 0; l < js.batches[k].descStride; ++l)
        {
            const uint32_t nr = js.jdata[k].mNumRows[l];
            for (uint32_t r = 0; r < nr; ++r)
            {
                const uint32_t o = js.batches[k].startConstraintIndex + r;
                if (!eq4(gCon[o].lin0XYZ_minImpulse[l], refCon[o].lin0XYZ_minImpulse[l])) ++mism;
                if (!eq4(gCon[o].lin1XYZ_maxImpulse[l], refCon[o].lin1XYZ_maxImpulse[l])) ++mism;
                if (!eq4(gCon[o].ang0XYZ_resp0[l], refCon[o].ang0XYZ_resp0[l])) ++mism;
                if (!eq4(gCon[o].ang1XYZ_resp1[l], refCon[o].ang1XYZ_resp1[l])) ++mism;
                if (!eq(gCon[o].initJointSpeed[l], refCon[o].initJointSpeed[l])) ++mism;
                if (gMod[o].flags[l] != refMod[o].flags[l]) ++mism;
                if (!eq(gMod[o].coeff0[l], refMod[o].coeff0[l])) ++mism;
                if (!eq(gMod[o].coeff1[l], refMod[o].coeff1[l])) ++mism;
                for (int c = 0; c < 3; ++c)
                    if (!eq(gMod[o].ang0Writeback[l * 3 + c], refMod[o].ang0Writeback[l * 3 + c])) ++mism;
            }
        }
    PXG_CHECK_MSG(mism == 0, "%zu joint row mismatches", mism);

    mism = 0;
    for (uint32_t k = 0; k < js.numBatches; ++k)
        for (uint32_t l = 0; l < js.batches[k].descStride; ++l)
        {
            const uint32_t nr = js.jdata[k].mNumRows[l];
            for (uint32_t r = 0; r < nr; ++r)
            {
                const uint32_t idx = k * MAX_CONSTRAINT_ROWS + r;
                if (!eq4(gVel[idx].linear0XYZ_geometricErrorW[l], refVel[idx].linear0XYZ_geometricErrorW[l])) ++mism;
                if (!eq4(gVel[idx].angular0XYZ_velocityTargetW[l], refVel[idx].angular0XYZ_velocityTargetW[l])) ++mism;
            }
        }
    PXG_CHECK_MSG(mism == 0, "%zu orthogonalize (jointVel) mismatches", mism);

    for (PxgDevicePtr pfree : { dBatches, dBi, dBd, dTx, dJd, dJv, dJp, dUi, dSc, dHdr, dCon, dMod,
                                dPrepDesc, dSharedDesc })
        shim.memFree(pfree);
}

PXG_TEST(constraintPrepTGS_joint)
{
    std::mt19937 rng(4400);
    JointSetup js = buildJointSetup(rng, true);

    const float stepDt = 1.f / 240.f, recipStepDt = 240.f, simDt = 1.f / 60.f, recipSimDt = 60.f;
    const float lengthScale = 1.f, biasCoefficient = 2.f;
    PxgSlangComputeDevice* dev = testDevice();

    std::vector<TGSJointHeader> hdrs(js.numBatches);
    std::vector<TGSJointRowCon> cons(js.numBatches * MAX_CONSTRAINT_ROWS);
    std::memset(hdrs.data(), 0, hdrs.size() * sizeof(TGSJointHeader));
    std::memset(cons.data(), 0, cons.size() * sizeof(TGSJointRowCon));

    // ---- ENGINE ABI (P5 Step 5 tranche 13): live device buffers + REAL engine
    // descriptors, launched through PxgCudaShim exactly as
    // PxgTGSCudaSolverCore::jointConstraintPrepareParallel does ----
    using namespace physx;
    using physx::gpuslang::PxgCudaShim;
    using physx::gpuslang::PxgKernelDef;
    using physx::gpuslang::PxgDevicePtr;
    using physx::gpuslang::PxgResult;

    static std::vector<uint8_t> (*tgsJointLoader)(const char*) = [](const char* f) { return loadShaderBlob(f); };
    const PxgKernelDef manifest[] = {
        { "constraintPrepTGS.spv", "jointConstraintBlockPrepareParallelLaunchTGS", 16u, 0 },
    };
    PxgCudaShim shim(dev, manifest, 1, tgsJointLoader);

    auto alloc = [&](const void* data, size_t bytes) -> PxgDevicePtr {
        PxgDevicePtr dp = 0;
        shim.memAlloc(&dp, bytes);
        if (dp && data)
            shim.memcpyHtoD(dp, data, bytes);
        return dp;
    };

    const PxgDevicePtr dBatches = alloc(js.batches.data(), js.batches.size() * sizeof(Batch));
    const PxgDevicePtr dBi = alloc(js.batchIndices.data(), js.batchIndices.size() * 4);
    const PxgDevicePtr dBd = alloc(js.bodyData.data(), js.bodyData.size() * sizeof(SolverBodyData));
    const PxgDevicePtr dTx = alloc(js.txi.data(), js.txi.size() * sizeof(SolverTxIData));
    const PxgDevicePtr dJd = alloc(js.jdata.data(), js.jdata.size() * sizeof(BlockConstraint1DData));
    const PxgDevicePtr dJv = alloc(js.jvel.data(), js.jvel.size() * sizeof(BlockConstraint1DVelocities));
    const PxgDevicePtr dJp = alloc(js.jpar.data(), js.jpar.size() * sizeof(BlockConstraint1DParameters));
    const PxgDevicePtr dUi = alloc(js.uniqueIndices.data(), js.uniqueIndices.size() * 4);
    const PxgDevicePtr dSc = alloc(js.constantData.data(), js.constantData.size() * 8);
    const PxgDevicePtr dHdr = alloc(hdrs.data(), hdrs.size() * sizeof(TGSJointHeader));
    const PxgDevicePtr dCon = alloc(cons.data(), cons.size() * sizeof(TGSJointRowCon));
    if (!dBatches) { std::printf("  NOTE: no BDA -> skip\n"); return; }

    PxgConstraintPrepareDesc prepDesc;
    std::memset(&prepDesc, 0xCD, sizeof(prepDesc)); // untouched fields must not matter
    prepDesc.jointConstraintBatchIndices = reinterpret_cast<PxU32*>(dBi);
    prepDesc.solverBodyDataPool = reinterpret_cast<PxgSolverBodyData*>(dBd);
    prepDesc.solverBodyTxIDataPool = reinterpret_cast<PxgSolverTxIData*>(dTx);
    prepDesc.blockJointPrepPool = reinterpret_cast<PxgBlockConstraint1DData*>(dJd);
    prepDesc.blockJointPrepPool0 = reinterpret_cast<PxgBlockConstraint1DVelocities*>(dJv);
    prepDesc.blockJointPrepPool1 = reinterpret_cast<PxgBlockConstraint1DParameters*>(dJp);
    prepDesc.constraintUniqueIndices = reinterpret_cast<PxU32*>(dUi);
    prepDesc.solverConstantData = reinterpret_cast<PxgSolverConstraintManagerConstants*>(dSc);
    // split the on-device batch-count sum (num1dConstraintBatches +
    // numStatic1dConstraintBatches) to pin the full expression
    prepDesc.num1dConstraintBatches = js.numBatches - 1;
    prepDesc.numStatic1dConstraintBatches = 1;
    // TGS reads erp = 0.5 * biasCoefficient from the prepare desc
    prepDesc.biasCoefficient = biasCoefficient;

    PxgSolverSharedDesc<IterativeSolveDataTGS> sharedDesc;
    std::memset(&sharedDesc, 0xCD, sizeof(sharedDesc));
    sharedDesc.stepDt = stepDt;
    sharedDesc.stepInvDtF32 = recipStepDt;
    sharedDesc.dt = simDt;
    sharedDesc.invDtF32 = recipSimDt;
    sharedDesc.lengthScale = lengthScale;
    sharedDesc.iterativeData.blockConstraintBatch = reinterpret_cast<PxgBlockConstraintBatch*>(dBatches);
    sharedDesc.iterativeData.blockJointConstraintHeaders =
        reinterpret_cast<PxgTGSBlockSolverConstraint1DHeader*>(dHdr);
    sharedDesc.iterativeData.blockJointConstraintRowsCon =
        reinterpret_cast<PxgTGSBlockSolverConstraint1DCon*>(dCon);

    const PxgDevicePtr dPrepDesc = alloc(&prepDesc, sizeof(prepDesc));
    const PxgDevicePtr dSharedDesc = alloc(&sharedDesc, sizeof(sharedDesc));

    // engine launch: nbBlocks = ceil(nbJointBatches * 32 /
    // CONSTRAINT_PREPARE_BLOCK_PARALLEL = 64) = 2 warps/block
    const uint32_t nbBlocks = (js.numBatches * 32 + 63) / 64;
    struct { uint64_t prepDesc, sharedDesc; } prm = { dPrepDesc, dSharedDesc };
    PXG_CHECK_MSG(shim.launchKernel(0, nbBlocks, 1, 1, &prm, sizeof(prm)) == PxgResult::Success,
                  "launch jointConstraintBlockPrepareParallelLaunchTGS");
    shim.streamSynchronize();

    // ---- reference ----
    std::vector<BlockConstraint1DVelocities> refVel = js.jvel;
    std::vector<BlockConstraint1DParameters> refPar = js.jpar;
    std::vector<TGSJointHeader> refHdr = hdrs;
    std::vector<TGSJointRowCon> refCon = cons;

    for (uint32_t k = 0; k < js.numBatches; ++k)
    {
        const Batch& b = js.batches[k];
        for (uint32_t lane = 0; lane < b.descStride; ++lane)
        {
            const uint32_t bA = b.bodyAIndex[lane], bB = b.bodyBIndex[lane];
            const uint32_t uniq = js.uniqueIndices[b.mStartPartitionIndex + lane];
            const uint32_t wbIndex = uint32_t(js.constantData[uniq] >> 32);
            refJointTGS(js.jdata, k, refVel, refPar, k * MAX_CONSTRAINT_ROWS, js.bodyData[bA], js.bodyData[bB],
                        js.si[bA], js.si[bB], stepDt, recipStepDt, simDt, recipSimDt, lengthScale, biasCoefficient,
                        lane, wbIndex, refHdr[k], refCon, b.startConstraintIndex);
        }
    }

    // ---- compare ----
    std::vector<TGSJointHeader> gHdr(hdrs.size());
    std::vector<TGSJointRowCon> gCon(cons.size());
    std::vector<BlockConstraint1DVelocities> gVel(js.jvel.size());
    std::vector<BlockConstraint1DParameters> gPar(js.jpar.size());
    shim.memcpyDtoH(gHdr.data(), dHdr, gHdr.size() * sizeof(TGSJointHeader));
    shim.memcpyDtoH(gCon.data(), dCon, gCon.size() * sizeof(TGSJointRowCon));
    shim.memcpyDtoH(gVel.data(), dJv, gVel.size() * sizeof(BlockConstraint1DVelocities));
    shim.memcpyDtoH(gPar.data(), dJp, gPar.size() * sizeof(BlockConstraint1DParameters));

    size_t mism = 0;
    for (uint32_t k = 0; k < js.numBatches && mism < 12; ++k)
        for (uint32_t l = 0; l < js.batches[k].descStride && mism < 12; ++l)
        {
            if (gHdr[k].rowCounts_breakable_orthoAxisCount[l] != refHdr[k].rowCounts_breakable_orthoAxisCount[l])
            {
                PXG_CHECK_MSG(false, "b%u l%u rc_b_ortho g%08x r%08x", k, l, gHdr[k].rowCounts_breakable_orthoAxisCount[l], refHdr[k].rowCounts_breakable_orthoAxisCount[l]);
                ++mism;
            }
            if (gHdr[k].writeBackOffset[l] != refHdr[k].writeBackOffset[l]) ++mism;
            if (!eq4(gHdr[k].rAWorld_invMass0D0[l], refHdr[k].rAWorld_invMass0D0[l])) ++mism;
            if (!eq4(gHdr[k].rBWorld_invMass1D1[l], refHdr[k].rBWorld_invMass1D1[l])) ++mism;
            if (!eq(gHdr[k].invInertiaScale0[l], refHdr[k].invInertiaScale0[l])) ++mism;
            if (!eq(gHdr[k].invInertiaScale1[l], refHdr[k].invInertiaScale1[l])) ++mism;
            if (!eq(gHdr[k].linBreakImpulse[l], refHdr[k].linBreakImpulse[l])) ++mism;
            if (!eq(gHdr[k].angBreakImpulse[l], refHdr[k].angBreakImpulse[l])) ++mism;
            for (int a = 0; a < 3; ++a)
            {
                if (!eq4(gHdr[k].angOrthoAxis0_recipResponseW[a][l], refHdr[k].angOrthoAxis0_recipResponseW[a][l])) ++mism;
                if (!eq4(gHdr[k].angOrthoAxis1_ErrorW[a][l], refHdr[k].angOrthoAxis1_ErrorW[a][l])) ++mism;
            }
        }
    PXG_CHECK_MSG(mism == 0, "%zu TGS joint header mismatches", mism);

    mism = 0;
    for (uint32_t k = 0; k < js.numBatches; ++k)
        for (uint32_t l = 0; l < js.batches[k].descStride; ++l)
        {
            const uint32_t nr = js.jdata[k].mNumRows[l];
            for (uint32_t r = 0; r < nr; ++r)
            {
                const uint32_t o = js.batches[k].startConstraintIndex + r;
                if (!eq4(gCon[o].lin0XYZ_initBiasOrCoeff0[l], refCon[o].lin0XYZ_initBiasOrCoeff0[l])) ++mism;
                if (!eq4(gCon[o].lin1XYZ_biasScaleOrCoeff1[l], refCon[o].lin1XYZ_biasScaleOrCoeff1[l])) ++mism;
                if (!eq4(gCon[o].ang0XYZ_velMultiplierOrCoeff2[l], refCon[o].ang0XYZ_velMultiplierOrCoeff2[l])) ++mism;
                if (!eq4(gCon[o].ang1XYZ_velTargetOrCoeff3[l], refCon[o].ang1XYZ_velTargetOrCoeff3[l])) ++mism;
                if (!eq(gCon[o].maxBias[l], refCon[o].maxBias[l])) ++mism;
                if (!eq(gCon[o].angularErrorScale[l], refCon[o].angularErrorScale[l])) ++mism;
                if (!eq(gCon[o].minImpulse[l], refCon[o].minImpulse[l])) ++mism;
                if (!eq(gCon[o].maxImpulse[l], refCon[o].maxImpulse[l])) ++mism;
                if (gCon[o].flags[l] != refCon[o].flags[l]) ++mism;
            }
        }
    PXG_CHECK_MSG(mism == 0, "%zu TGS joint row mismatches", mism);

    mism = 0;
    for (uint32_t k = 0; k < js.numBatches; ++k)
        for (uint32_t l = 0; l < js.batches[k].descStride; ++l)
        {
            const uint32_t nr = js.jdata[k].mNumRows[l];
            for (uint32_t r = 0; r < nr; ++r)
            {
                const uint32_t idx = k * MAX_CONSTRAINT_ROWS + r;
                if (!eq4(gVel[idx].linear0XYZ_geometricErrorW[l], refVel[idx].linear0XYZ_geometricErrorW[l])) ++mism;
                if (gPar[idx].solveHint[l] != refPar[idx].solveHint[l]) ++mism;
            }
        }
    PXG_CHECK_MSG(mism == 0, "%zu TGS orthogonalize/hint mismatches", mism);

    for (PxgDevicePtr pfree : { dBatches, dBi, dBd, dTx, dJd, dJv, dJp, dUi, dSc, dHdr, dCon,
                                dPrepDesc, dSharedDesc })
        shim.memFree(pfree);
}

// Exercises the getFrictionPatches() previous-frame anchor carry-over path,
// which the other contact tests skip (hasPrevFrictionIndices = 0).  Uses
// identity body frames so the normal/anchor correlation succeeds
// deterministically, and previous patches carrying 1 or 2 anchors.
PXG_TEST(constraintPrepPGS_frictionCarryover)
{
    std::mt19937 rng(4500);
    std::uniform_real_distribution<float> u(0.f, 1.f), s(-1.f, 1.f);

    const uint32_t numBatches = 3, rowsPerBatch = 4, numBodies = 32;
    const float dt = 1.f / 60.f, invDt = 60.f;
    const float bounceThreshold = -2.f, frictionOffsetThreshold = 0.05f, correlationDistance = 0.05f;
    const float ccdMaxSeparation = 0.1f;

    ContactSetup cs;
    cs.numBatches = numBatches;
    cs.rowsPerBatch = rowsPerBatch;
    cs.numBodies = numBodies;
    cs.bodyData.resize(numBodies);
    cs.txi.resize(numBodies);
    cs.bodyFrames.resize(numBodies);
    cs.si.resize(numBodies);
    cs.frames.resize(numBodies);
    std::memset(cs.bodyData.data(), 0, cs.bodyData.size() * sizeof(SolverBodyData));
    std::memset(cs.txi.data(), 0, cs.txi.size() * sizeof(SolverTxIData));
    for (uint32_t b = 0; b < numBodies; ++b)
    {
        cs.bodyData[b].initialLinVelXYZ_invMassW = Float4{ s(rng) * 0.2f, s(rng) * 0.2f, s(rng) * 0.2f, u(rng) * 2.f };
        cs.bodyData[b].initialAngVelXYZ_penBiasClamp = Float4{ s(rng) * 0.2f, s(rng) * 0.2f, s(rng) * 0.2f, -u(rng) };
        cs.bodyData[b].maxImpulse = 500.f;
        cs.bodyData[b].offsetSlop = 0.f;
        cs.bodyData[b].flags = 0;
        AlignedTransform f;
        f.q = Float4{ 0, 0, 0, 1 }; // identity
        f.p = Float4{ 0, 0, 0, 0 };
        cs.bodyFrames[b] = f;
        cs.frames[b] = toAFrame(f);
        const Mat33 m = randSymInvInertia(rng);
        cs.si[b] = m;
        m.store(cs.txi[b].sqrtInvInertia);
    }

    cs.batches.resize(numBatches);
    cs.workUnits.resize(numBatches);
    cs.cdata.resize(numBatches);
    cs.cpoints.resize(numBatches * rowsPerBatch);
    cs.batchIndices.resize(numBatches);
    std::memset(cs.batches.data(), 0, cs.batches.size() * sizeof(Batch));
    std::memset(cs.workUnits.data(), 0, cs.workUnits.size() * sizeof(BlockWorkUnit));
    std::memset(cs.cdata.data(), 0, cs.cdata.size() * sizeof(BlockContactData));
    std::memset(cs.cpoints.data(), 0, cs.cpoints.size() * sizeof(BlockContactPoint));

    const uint32_t totalPreviousEdges = numBatches * 32;
    std::vector<BlockFrictionPatch> prevPatches(numBatches);
    std::vector<BlockFrictionAnchorPatch> prevAnchors(numBatches);
    std::vector<uint64_t> prevFricIdx(totalPreviousEdges, 0);
    std::memset(prevPatches.data(), 0, prevPatches.size() * sizeof(BlockFrictionPatch));
    std::memset(prevAnchors.data(), 0, prevAnchors.size() * sizeof(BlockFrictionAnchorPatch));

    for (uint32_t k = 0; k < numBatches; ++k)
    {
        cs.batchIndices[k] = k;
        Batch& b = cs.batches[k];
        b.descStride = uint16_t(4 + rng() % 13);
        b.constraintType = 1;
        b.mConstraintBatchIndex = k;
        b.blockContactIndex = k * rowsPerBatch;
        b.startConstraintIndex = k * rowsPerBatch;
        b.startFrictionIndex = k * (rowsPerBatch * 2 + 1);

        for (uint32_t lane = 0; lane < 32; ++lane)
        {
            b.bodyAIndex[lane] = rng() % numBodies;
            b.bodyBIndex[lane] = rng() % numBodies;

            const uint32_t nc = 2 + rng() % (rowsPerBatch - 1); // 2..4 contacts
            const uint32_t prevAnchorCnt = 1 + (rng() & 1);      // carry 1 or 2 anchors
            const uint32_t materialWord = (nc & 0xFF) | (0u << 8) | (1u << 16); // prevFrictionPatchCount = 1
            float wbits;
            std::memcpy(&wbits, &materialWord, 4);
            cs.cdata[k].contactData[lane] = Float4{ 0.01f, u(rng), u(rng) * 0.9f, wbits };
            const Vec3 n = Vec3(s(rng), s(rng), s(rng)).getNormalized();
            cs.cdata[k].normal_restitutionW[lane] = Float4{ n.x, n.y, n.z, 0.f };
            cs.cdata[k].damping[lane] = 0.f;
            cs.cdata[k].mInvMassScale[lane] = Float4{ 1, 1, 1, 1 };

            for (uint32_t i = 0; i < rowsPerBatch; ++i)
            {
                BlockContactPoint& cp = cs.cpoints[b.blockContactIndex + i];
                const Vec3 pt(s(rng), s(rng), s(rng));
                cp.point_separationW[lane] = Float4{ pt.x, pt.y, pt.z, s(rng) * 0.02f };
                cp.targetVel_maxImpulseW[lane] = Float4{ s(rng) * 0.3f, s(rng) * 0.3f, s(rng) * 0.3f, 50.f };
            }

            cs.workUnits[k].mWriteback[lane] = k * rowsPerBatch * 32; // any valid offset
            const uint32_t edgeIndex = k * 32 + lane;
            cs.workUnits[k].mEdgeIndex[lane] = edgeIndex;
            cs.workUnits[k].mPatchIndex[lane] = 0;
            cs.workUnits[k].mFrictionPatchIndex[lane] = edgeIndex;
            cs.workUnits[k].mTorsionalFrictionData[lane] = Float2{ 0.f, 0.f };

            // build a previous patch (index = k, lane = lane) with a matching
            // normal and prevAnchorCnt anchors, then point the prev index at it
            prevPatches[k].body0Normal[lane] = Float4{ n.x, n.y, n.z, 0.f };
            prevPatches[k].body1Normal[lane] = Float4{ n.x, n.y, n.z, 0.f };
            prevPatches[k].anchorCount[lane] = prevAnchorCnt;
            prevPatches[k].broken[lane] = 0;
            const Vec3 a0(s(rng) * 0.5f, s(rng) * 0.5f, s(rng) * 0.5f);
            const Vec3 a1(s(rng) * 0.5f, s(rng) * 0.5f, s(rng) * 0.5f);
            prevAnchors[k].body0Anchors0[lane] = Float4{ a0.x, a0.y, a0.z, 0.f };
            prevAnchors[k].body1Anchors0[lane] = Float4{ a0.x, a0.y, a0.z, 0.f }; // == body0 so pointsAreClose holds (identity frames)
            prevAnchors[k].body0Anchors1[lane] = Float4{ a1.x, a1.y, a1.z, 0.f };
            prevAnchors[k].body1Anchors1[lane] = Float4{ a1.x, a1.y, a1.z, 0.f };
            // prev friction index: patchIndex = k, oldThreadId = lane
            prevFricIdx[edgeIndex] = (uint64_t(k) << 5) | lane;
        }
    }

    PxgSlangComputeDevice* dev = testDevice();

    std::vector<BlockFrictionPatch> currPatches(numBatches);
    std::vector<BlockFrictionAnchorPatch> currAnchors(numBatches);
    std::memset(currPatches.data(), 0, currPatches.size() * sizeof(BlockFrictionPatch));
    std::memset(currAnchors.data(), 0, currAnchors.size() * sizeof(BlockFrictionAnchorPatch));
    std::vector<BlockSolverContactHeader> chs(numBatches);
    std::vector<BlockSolverFrictionHeader> fhs(numBatches);
    std::vector<BlockSolverContactPoint> outCP(numBatches * rowsPerBatch);
    std::vector<BlockSolverContactFriction> outFR(numBatches * (rowsPerBatch * 2 + 1));
    std::memset(chs.data(), 0, chs.size() * sizeof(BlockSolverContactHeader));
    std::memset(fhs.data(), 0, fhs.size() * sizeof(BlockSolverFrictionHeader));
    std::memset(outCP.data(), 0, outCP.size() * sizeof(BlockSolverContactPoint));
    std::memset(outFR.data(), 0, outFR.size() * sizeof(BlockSolverContactFriction));
    std::vector<uint64_t> currFricIdx(numBatches * 32 + 1, 0);

    // ---- ENGINE ABI (P5 Step 5): the previous-frame patches/anchors/indices
    // are LIVE device buffers and blockPreviousFrictionIndices is non-null,
    // driving the getFrictionPatches carry-over path ----
    using namespace physx;
    using physx::gpuslang::PxgCudaShim;
    using physx::gpuslang::PxgKernelDef;
    using physx::gpuslang::PxgDevicePtr;
    using physx::gpuslang::PxgResult;

    static std::vector<uint8_t> (*carryLoader)(const char*) = [](const char* f) { return loadShaderBlob(f); };
    const PxgKernelDef manifest[] = {
        { "constraintPrepPGS.spv", "contactConstraintBlockPrepareParallelLaunch", 16u, 0 },
    };
    PxgCudaShim shim(dev, manifest, 1, carryLoader);

    auto alloc = [&](const void* data, size_t bytes) -> PxgDevicePtr {
        PxgDevicePtr dp = 0;
        shim.memAlloc(&dp, bytes);
        if (dp && data)
            shim.memcpyHtoD(dp, data, bytes);
        return dp;
    };

    const PxgDevicePtr dBatches = alloc(cs.batches.data(), cs.batches.size() * sizeof(Batch));
    const PxgDevicePtr dBi = alloc(cs.batchIndices.data(), cs.batchIndices.size() * 4);
    const PxgDevicePtr dBd = alloc(cs.bodyData.data(), cs.bodyData.size() * sizeof(SolverBodyData));
    const PxgDevicePtr dTx = alloc(cs.txi.data(), cs.txi.size() * sizeof(SolverTxIData));
    const PxgDevicePtr dBf = alloc(cs.bodyFrames.data(), cs.bodyFrames.size() * sizeof(AlignedTransform));
    const PxgDevicePtr dWu = alloc(cs.workUnits.data(), cs.workUnits.size() * sizeof(BlockWorkUnit));
    const PxgDevicePtr dCd = alloc(cs.cdata.data(), cs.cdata.size() * sizeof(BlockContactData));
    const PxgDevicePtr dCp = alloc(cs.cpoints.data(), cs.cpoints.size() * sizeof(BlockContactPoint));
    const PxgDevicePtr dPatch = alloc(currPatches.data(), currPatches.size() * sizeof(BlockFrictionPatch));
    const PxgDevicePtr dPPatch = alloc(prevPatches.data(), prevPatches.size() * sizeof(BlockFrictionPatch));
    const PxgDevicePtr dAnch = alloc(currAnchors.data(), currAnchors.size() * sizeof(BlockFrictionAnchorPatch));
    const PxgDevicePtr dPAnch = alloc(prevAnchors.data(), prevAnchors.size() * sizeof(BlockFrictionAnchorPatch));
    const PxgDevicePtr dCfi = alloc(currFricIdx.data(), currFricIdx.size() * 8);
    const PxgDevicePtr dPfi = alloc(prevFricIdx.data(), prevFricIdx.size() * 8);
    const PxgDevicePtr dCh = alloc(chs.data(), chs.size() * sizeof(BlockSolverContactHeader));
    const PxgDevicePtr dFh = alloc(fhs.data(), fhs.size() * sizeof(BlockSolverFrictionHeader));
    const PxgDevicePtr dOutCP = alloc(outCP.data(), outCP.size() * sizeof(BlockSolverContactPoint));
    const PxgDevicePtr dOutFR = alloc(outFR.data(), outFR.size() * sizeof(BlockSolverContactFriction));
    if (!dBatches) { std::printf("  NOTE: no BDA -> skip\n"); return; }

    PxgConstraintPrepareDesc prepDesc;
    std::memset(&prepDesc, 0xCD, sizeof(prepDesc));
    prepDesc.contactConstraintBatchIndices = reinterpret_cast<PxU32*>(dBi);
    prepDesc.solverBodyDataPool = reinterpret_cast<PxgSolverBodyData*>(dBd);
    prepDesc.solverBodyTxIDataPool = reinterpret_cast<PxgSolverTxIData*>(dTx);
    prepDesc.blockWorkUnit = reinterpret_cast<PxgBlockWorkUnit*>(dWu);
    prepDesc.blockCurrentFrictionIndices = reinterpret_cast<PxgBlockFrictionIndex*>(dCfi);
    prepDesc.blockPreviousFrictionIndices = reinterpret_cast<PxgBlockFrictionIndex*>(dPfi); // LIVE
    prepDesc.blockContactCurrentPrepPool = reinterpret_cast<physx::PxgBlockContactData*>(dCd);
    prepDesc.blockContactPoints = reinterpret_cast<physx::PxgBlockContactPoint*>(dCp);
    prepDesc.blockCurrentAnchorPatches = reinterpret_cast<physx::PxgBlockFrictionAnchorPatch*>(dAnch);
    prepDesc.blockPreviousAnchorPatches = reinterpret_cast<physx::PxgBlockFrictionAnchorPatch*>(dPAnch);
    prepDesc.body2WorldPool = reinterpret_cast<PxAlignedTransform*>(dBf);
    prepDesc.numContactBatches = numBatches;
    prepDesc.numStaticContactBatches = 0;
    prepDesc.bounceThresholdF32 = bounceThreshold;
    prepDesc.frictionOffsetThreshold = frictionOffsetThreshold;
    prepDesc.correlationDistance = correlationDistance;
    prepDesc.ccdMaxSeparation = ccdMaxSeparation;
    prepDesc.totalPreviousEdges = totalPreviousEdges;
    prepDesc.totalCurrentEdges = numBatches * 32;

    PxgSolverSharedDesc<IterativeSolveData> sharedDesc;
    std::memset(&sharedDesc, 0xCD, sizeof(sharedDesc));
    sharedDesc.blockCurrentFrictionPatches = reinterpret_cast<physx::PxgBlockFrictionPatch*>(dPatch);
    sharedDesc.blockPreviousFrictionPatches = reinterpret_cast<physx::PxgBlockFrictionPatch*>(dPPatch);
    sharedDesc.dt = dt;
    sharedDesc.invDtF32 = invDt;
    sharedDesc.iterativeData.blockConstraintBatch = reinterpret_cast<PxgBlockConstraintBatch*>(dBatches);
    sharedDesc.iterativeData.blockContactHeaders = reinterpret_cast<PxgBlockSolverContactHeader*>(dCh);
    sharedDesc.iterativeData.blockFrictionHeaders = reinterpret_cast<PxgBlockSolverFrictionHeader*>(dFh);
    sharedDesc.iterativeData.blockContactPoints = reinterpret_cast<PxgBlockSolverContactPoint*>(dOutCP);
    sharedDesc.iterativeData.blockFrictions = reinterpret_cast<PxgBlockSolverContactFriction*>(dOutFR);

    const PxgDevicePtr dPrepDesc = alloc(&prepDesc, sizeof(prepDesc));
    const PxgDevicePtr dSharedDesc = alloc(&sharedDesc, sizeof(sharedDesc));

    const uint32_t nbBlocks = (numBatches + 1) / 2; // 2 warps/block
    struct { uint64_t prepDesc, sharedDesc; } prm = { dPrepDesc, dSharedDesc };
    PXG_CHECK_MSG(shim.launchKernel(0, nbBlocks, 1, 1, &prm, sizeof(prm)) == PxgResult::Success,
                  "launch contactConstraintBlockPrepareParallelLaunch (carry-over)");
    shim.streamSynchronize();

    // ---- reference ----
    std::vector<BlockFrictionPatch> refPatches = currPatches;
    std::vector<BlockFrictionAnchorPatch> refAnchors = currAnchors;
    std::vector<BlockSolverContactHeader> refCH = chs;
    std::vector<BlockSolverFrictionHeader> refFH = fhs;
    std::vector<BlockSolverContactPoint> refCP = outCP;
    std::vector<BlockSolverContactFriction> refFR = outFR;

    uint32_t carriedOver = 0;
    for (uint32_t k = 0; k < numBatches; ++k)
    {
        const Batch& b = cs.batches[k];
        for (uint32_t lane = 0; lane < b.descStride; ++lane)
        {
            const uint32_t bA = b.bodyAIndex[lane], bB = b.bodyBIndex[lane];
            const AFrame& bf0 = cs.frames[bA];
            const AFrame& bf1 = cs.frames[bB];
            const Float4 data_ = cs.cdata[k].contactData[lane];
            uint32_t materialWord;
            std::memcpy(&materialWord, &data_.w, 4);
            const uint32_t nbContacts = materialWord & 0xFF;
            const uint32_t prevFrictionPatchCount = (materialWord >> 16) & 0xFF;
            const float restDistance = data_.x, staticF = data_.y, dynamicF = data_.z;
            const Float4 n4 = cs.cdata[k].normal_restitutionW[lane];
            const Vec3 normal(n4.x, n4.y, n4.z);
            const uint32_t contactBase = b.blockContactIndex;

            refCorrelatePatches(refPatches[k], lane, nbContacts, normal, bf0, bf1);
            refPatches[k].broken[lane] = 0;
            float patchExtents = 0.f;
            refGetFrictionPatches(refPatches[k], refAnchors[k], lane, prevFricIdx, true,
                                  cs.workUnits[k].mEdgeIndex[lane], prevPatches, prevAnchors,
                                  prevFrictionPatchCount, bf0, bf1, correlationDistance, totalPreviousEdges,
                                  patchExtents, nbContacts);
            if (refPatches[k].anchorCount[lane] > 0)
                ++carriedOver;
            refGrowPatches(refPatches[k], refAnchors[k], lane, cs.cpoints, contactBase, nbContacts, bf0, bf1,
                           frictionOffsetThreshold + restDistance, patchExtents);
            refSetupPGS(cs.cdata[k], cs.cpoints, contactBase, nbContacts, refPatches[k], refAnchors[k], bf0, bf1,
                        cs.bodyData[bA], cs.bodyData[bB], cs.si[bA], cs.si[bB], invDt, dt, bounceThreshold, lane,
                        cs.workUnits[k].mWriteback[lane], false, refCH[k], refFH[k], refCP, contactBase, refFR,
                        b.startFrictionIndex, restDistance, staticF, dynamicF, 0, ccdMaxSeparation, 0.f);
        }
    }

    std::vector<BlockSolverContactFriction> gFR(outFR.size());
    std::vector<BlockSolverFrictionHeader> gFH(fhs.size());
    std::vector<BlockFrictionAnchorPatch> gAnch(currAnchors.size());
    std::vector<BlockFrictionPatch> gPatch(currPatches.size());
    shim.memcpyDtoH(gFR.data(), dOutFR, gFR.size() * sizeof(BlockSolverContactFriction));
    shim.memcpyDtoH(gFH.data(), dFh, gFH.size() * sizeof(BlockSolverFrictionHeader));
    shim.memcpyDtoH(gAnch.data(), dAnch, gAnch.size() * sizeof(BlockFrictionAnchorPatch));
    shim.memcpyDtoH(gPatch.data(), dPatch, gPatch.size() * sizeof(BlockFrictionPatch));

    PXG_CHECK_MSG(carriedOver > 0, "no previous friction patches were carried over (test is vacuous)");

    size_t mism = 0;
    for (uint32_t k = 0; k < numBatches; ++k)
        for (uint32_t l = 0; l < cs.batches[k].descStride; ++l)
        {
            if (gPatch[k].anchorCount[l] != refPatches[k].anchorCount[l]) { PXG_CHECK_MSG(false, "b%u l%u anchors g%u r%u", k, l, gPatch[k].anchorCount[l], refPatches[k].anchorCount[l]); ++mism; }
            if (gFH[k].numFrictionConstr[l] != refFH[k].numFrictionConstr[l]) ++mism;
            const uint32_t nf = refFH[k].numFrictionConstr[l];
            for (uint32_t i = 0; i < nf; ++i)
            {
                const uint32_t r = cs.batches[k].startFrictionIndex + i;
                if (!eq4(gFR[r].raXn_bias[l], refFR[r].raXn_bias[l])) ++mism;
                if (!eq4(gFR[r].rbXn_targetVelW[l], refFR[r].rbXn_targetVelW[l])) ++mism;
            }
            for (uint32_t a = 0; a < 2; ++a)
            {
                if (gPatch[k].anchorCount[l] <= a)
                    continue;
                const Float4& g = a == 0 ? gAnch[k].body0Anchors0[l] : gAnch[k].body0Anchors1[l];
                const Float4& rf = a == 0 ? refAnchors[k].body0Anchors0[l] : refAnchors[k].body0Anchors1[l];
                if (!eq3(g, rf)) ++mism;
            }
        }
    PXG_CHECK_MSG(mism == 0, "%zu carry-over mismatches", mism);

    for (PxgDevicePtr pfree : { dBatches, dBi, dBd, dTx, dBf, dWu, dCd, dCp, dPatch, dPPatch, dAnch,
                                dPAnch, dCfi, dPfi, dCh, dFh, dOutCP, dOutFR, dPrepDesc, dSharedDesc })
        shim.memFree(pfree);
}

} // namespace
