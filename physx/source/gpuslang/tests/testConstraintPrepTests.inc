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
    std::mt19937 rng(4100);
    ContactSetup cs = buildContactSetup(rng);

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
    std::vector<uint32_t> dummy(64, 0);

    // buffers
    PxgBufferHandle batchBuf = createBufferFrom(dev, cs.batches);
    PxgBufferHandle biBuf = createBufferFrom(dev, cs.batchIndices);
    PxgBufferHandle bdBuf = createBufferFrom(dev, cs.bodyData);
    PxgBufferHandle txBuf = createBufferFrom(dev, cs.txi);
    PxgBufferHandle dummyBuf = createBufferFrom(dev, dummy);
    PxgBufferHandle bfBuf = createBufferFrom(dev, cs.bodyFrames);
    PxgBufferHandle wuBuf = createBufferFrom(dev, cs.workUnits);
    PxgBufferHandle cdBuf = createBufferFrom(dev, cs.cdata);
    PxgBufferHandle cpBuf = createBufferFrom(dev, cs.cpoints);
    PxgBufferHandle cpatchBuf = createBufferFrom(dev, currPatches);
    PxgBufferHandle canchBuf = createBufferFrom(dev, currAnchors);
    PxgBufferHandle cfiBuf = createBufferFrom(dev, currFricIdx);
    PxgBufferHandle chBuf = createBufferFrom(dev, chs);
    PxgBufferHandle fhBuf = createBufferFrom(dev, fhs);
    PxgBufferHandle outCPBuf = createBufferFrom(dev, outCP);
    PxgBufferHandle outFRBuf = createBufferFrom(dev, outFR);

    Kernel kern("constraintPrepPGS.spv", "contactConstraintBlockPrepareParallelLaunch",
                std::vector<uint32_t>(26, 4), sizeof(PrepParams));

    PrepParams p = {};
    p.numBatches = cs.numBatches;
    p.totalPreviousEdges = 0;
    p.totalCurrentEdges = cs.numBatches * 32;
    p.hasPrevFrictionIndices = 0;
    p.dtOrStepDt = dt;
    p.invDtOrStepInvDt = invDt;
    p.totalDt = dt;
    p.invTotalDt = invDt;
    p.bounceThreshold = bounceThreshold;
    p.frictionOffsetThreshold = frictionOffsetThreshold;
    p.correlationDistance = correlationDistance;
    p.ccdMaxSeparation = ccdMaxSeparation;

    std::vector<PxgBufferHandle> bufs = { batchBuf, biBuf, bdBuf, txBuf, dummyBuf, dummyBuf, dummyBuf,
                                          dummyBuf, dummyBuf, dummyBuf, dummyBuf, dummyBuf, bfBuf, wuBuf,
                                          cdBuf, cpBuf, cpatchBuf, dummyBuf /*prevPatches*/, canchBuf,
                                          dummyBuf /*prevAnchors*/, cfiBuf, dummyBuf /*prevIdx*/, chBuf,
                                          fhBuf, outCPBuf, outFRBuf };
    kern.dispatch(cs.numBatches, bufs, p);
    dev->flush();

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
                                      p.totalPreviousEdges, patchExtents, nbContacts);
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
    const auto gCH = downloadBuffer<BlockSolverContactHeader>(dev, chBuf, chs.size());
    const auto gFH = downloadBuffer<BlockSolverFrictionHeader>(dev, fhBuf, fhs.size());
    const auto gCP = downloadBuffer<BlockSolverContactPoint>(dev, outCPBuf, outCP.size());
    const auto gFR = downloadBuffer<BlockSolverContactFriction>(dev, outFRBuf, outFR.size());
    const auto gPatch = downloadBuffer<BlockFrictionPatch>(dev, cpatchBuf, currPatches.size());
    const auto gAnch = downloadBuffer<BlockFrictionAnchorPatch>(dev, canchBuf, currAnchors.size());

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

    for (PxgBufferHandle h : { batchBuf, biBuf, bdBuf, txBuf, dummyBuf, bfBuf, wuBuf, cdBuf, cpBuf, cpatchBuf,
                               canchBuf, cfiBuf, chBuf, fhBuf, outCPBuf, outFRBuf })
        dev->destroyBuffer(h);
}

PXG_TEST(constraintPrepTGS_contact)
{
    std::mt19937 rng(4200);
    ContactSetup cs = buildContactSetup(rng);

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
    std::vector<uint32_t> dummy(64, 0);

    PxgBufferHandle batchBuf = createBufferFrom(dev, cs.batches);
    PxgBufferHandle biBuf = createBufferFrom(dev, cs.batchIndices);
    PxgBufferHandle bdBuf = createBufferFrom(dev, cs.bodyData);
    PxgBufferHandle txBuf = createBufferFrom(dev, cs.txi);
    PxgBufferHandle dummyBuf = createBufferFrom(dev, dummy);
    PxgBufferHandle bfBuf = createBufferFrom(dev, cs.bodyFrames);
    PxgBufferHandle wuBuf = createBufferFrom(dev, cs.workUnits);
    PxgBufferHandle cdBuf = createBufferFrom(dev, cs.cdata);
    PxgBufferHandle cpBuf = createBufferFrom(dev, cs.cpoints);
    PxgBufferHandle cpatchBuf = createBufferFrom(dev, currPatches);
    PxgBufferHandle canchBuf = createBufferFrom(dev, currAnchors);
    PxgBufferHandle cfiBuf = createBufferFrom(dev, currFricIdx);
    PxgBufferHandle chBuf = createBufferFrom(dev, chs);
    PxgBufferHandle fhBuf = createBufferFrom(dev, fhs);
    PxgBufferHandle outCPBuf = createBufferFrom(dev, outCP);
    PxgBufferHandle outFRBuf = createBufferFrom(dev, outFR);

    Kernel kern("constraintPrepTGS.spv", "contactConstraintBlockPrepareParallelLaunchTGS",
                std::vector<uint32_t>(25, 4), sizeof(PrepParams));

    PrepParams p = {};
    p.numBatches = cs.numBatches;
    p.totalCurrentEdges = cs.numBatches * 32;
    p.hasPrevFrictionIndices = 0;
    p.dtOrStepDt = stepDt;
    p.invDtOrStepInvDt = invStepDt;
    p.totalDt = totalDt;
    p.invTotalDt = invTotalDt;
    p.bounceThreshold = bounceThreshold;
    p.frictionOffsetThreshold = frictionOffsetThreshold;
    p.correlationDistance = correlationDistance;
    p.ccdMaxSeparation = ccdMaxSeparation;
    p.biasCoefficient = biasCoefficient;

    // TGS contact bindings: 0-3 then 11-24
    std::vector<PxgBufferHandle> bufs = { batchBuf, biBuf, bdBuf, txBuf, dummyBuf, dummyBuf, dummyBuf,
                                          dummyBuf, dummyBuf, dummyBuf, dummyBuf, bfBuf, wuBuf, cdBuf,
                                          cpBuf, cpatchBuf, dummyBuf, canchBuf, dummyBuf, cfiBuf, dummyBuf,
                                          chBuf, fhBuf, outCPBuf, outFRBuf };
    kern.dispatch(cs.numBatches, bufs, p);
    dev->flush();

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
                                      p.totalPreviousEdges, patchExtents, nbContacts);
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
    const auto gCH = downloadBuffer<TGSContactHeader>(dev, chBuf, chs.size());
    const auto gFH = downloadBuffer<TGSFrictionHeader>(dev, fhBuf, fhs.size());
    const auto gCP = downloadBuffer<TGSContactPoint>(dev, outCPBuf, outCP.size());
    const auto gFR = downloadBuffer<TGSFriction>(dev, outFRBuf, outFR.size());

    size_t mism = 0;
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
                if (!eq(gCP[r].maxImpulse[l], refCP[r].maxImpulse[l])) ++mism;
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
                if (!eq(gFH[k].torsionalFrictionScale[l], refFH[k].torsionalFrictionScale[l])) ++mism;
        }
    PXG_CHECK_MSG(mism == 0, "%zu TGS row mismatches", mism);

    for (PxgBufferHandle h : { batchBuf, biBuf, bdBuf, txBuf, dummyBuf, bfBuf, wuBuf, cdBuf, cpBuf, cpatchBuf,
                               canchBuf, cfiBuf, chBuf, fhBuf, outCPBuf, outFRBuf })
        dev->destroyBuffer(h);
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
            js.jdata[k].mFlags[lane] = (u(rng) < 0.2f ? CFLAG_DRIVE_LIMITS_ARE_FORCES : 0u) |
                                       (u(rng) < 0.2f ? CFLAG_DISABLE_PREPROCESSING : 0u);

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

    PxgBufferHandle batchBuf = createBufferFrom(dev, js.batches);
    PxgBufferHandle biBuf = createBufferFrom(dev, js.batchIndices);
    PxgBufferHandle bdBuf = createBufferFrom(dev, js.bodyData);
    PxgBufferHandle txBuf = createBufferFrom(dev, js.txi);
    PxgBufferHandle jdBuf = createBufferFrom(dev, js.jdata);
    PxgBufferHandle jvBuf = createBufferFrom(dev, js.jvel);
    PxgBufferHandle jpBuf = createBufferFrom(dev, js.jpar);
    PxgBufferHandle uiBuf = createBufferFrom(dev, js.uniqueIndices);
    PxgBufferHandle scBuf = createBufferFrom(dev, js.constantData);
    PxgBufferHandle hdrBuf = createBufferFrom(dev, hdrs);
    PxgBufferHandle conBuf = createBufferFrom(dev, cons);
    PxgBufferHandle modBuf = createBufferFrom(dev, mods);

    Kernel kern("constraintPrepPGS.spv", "jointConstraintBlockPrepareParallelLaunch",
                std::vector<uint32_t>(12, 4), sizeof(PrepParams));

    PrepParams p = {};
    p.numBatches = js.numBatches;
    p.dtOrStepDt = dt;
    p.invDtOrStepInvDt = recipdt;
    kern.dispatch(js.numBatches, { batchBuf, biBuf, bdBuf, txBuf, jdBuf, jvBuf, jpBuf, uiBuf, scBuf, hdrBuf, conBuf, modBuf }, p);
    dev->flush();

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
    const auto gHdr = downloadBuffer<BlockSolverConstraint1DHeader>(dev, hdrBuf, hdrs.size());
    const auto gCon = downloadBuffer<BlockSolverConstraint1DCon>(dev, conBuf, cons.size());
    const auto gMod = downloadBuffer<BlockSolverConstraint1DMod>(dev, modBuf, mods.size());
    const auto gVel = downloadBuffer<BlockConstraint1DVelocities>(dev, jvBuf, js.jvel.size());

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

    for (PxgBufferHandle h : { batchBuf, biBuf, bdBuf, txBuf, jdBuf, jvBuf, jpBuf, uiBuf, scBuf, hdrBuf, conBuf, modBuf })
        dev->destroyBuffer(h);
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

    PxgBufferHandle batchBuf = createBufferFrom(dev, js.batches);
    PxgBufferHandle biBuf = createBufferFrom(dev, js.batchIndices);
    PxgBufferHandle bdBuf = createBufferFrom(dev, js.bodyData);
    PxgBufferHandle txBuf = createBufferFrom(dev, js.txi);
    PxgBufferHandle jdBuf = createBufferFrom(dev, js.jdata);
    PxgBufferHandle jvBuf = createBufferFrom(dev, js.jvel);
    PxgBufferHandle jpBuf = createBufferFrom(dev, js.jpar);
    PxgBufferHandle uiBuf = createBufferFrom(dev, js.uniqueIndices);
    PxgBufferHandle scBuf = createBufferFrom(dev, js.constantData);
    PxgBufferHandle hdrBuf = createBufferFrom(dev, hdrs);
    PxgBufferHandle conBuf = createBufferFrom(dev, cons);

    Kernel kern("constraintPrepTGS.spv", "jointConstraintBlockPrepareParallelLaunchTGS",
                std::vector<uint32_t>(11, 4), sizeof(PrepParams));

    PrepParams p = {};
    p.numBatches = js.numBatches;
    p.dtOrStepDt = stepDt;
    p.invDtOrStepInvDt = recipStepDt;
    p.totalDt = simDt;
    p.invTotalDt = recipSimDt;
    p.lengthScale = lengthScale;
    p.biasCoefficient = biasCoefficient;
    kern.dispatch(js.numBatches, { batchBuf, biBuf, bdBuf, txBuf, jdBuf, jvBuf, jpBuf, uiBuf, scBuf, hdrBuf, conBuf }, p);
    dev->flush();

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
    const auto gHdr = downloadBuffer<TGSJointHeader>(dev, hdrBuf, hdrs.size());
    const auto gCon = downloadBuffer<TGSJointRowCon>(dev, conBuf, cons.size());
    const auto gVel = downloadBuffer<BlockConstraint1DVelocities>(dev, jvBuf, js.jvel.size());
    const auto gPar = downloadBuffer<BlockConstraint1DParameters>(dev, jpBuf, js.jpar.size());

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

    for (PxgBufferHandle h : { batchBuf, biBuf, bdBuf, txBuf, jdBuf, jvBuf, jpBuf, uiBuf, scBuf, hdrBuf, conBuf })
        dev->destroyBuffer(h);
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
    std::vector<uint32_t> dummy(64, 0);

    PxgBufferHandle batchBuf = createBufferFrom(dev, cs.batches);
    PxgBufferHandle biBuf = createBufferFrom(dev, cs.batchIndices);
    PxgBufferHandle bdBuf = createBufferFrom(dev, cs.bodyData);
    PxgBufferHandle txBuf = createBufferFrom(dev, cs.txi);
    PxgBufferHandle dummyBuf = createBufferFrom(dev, dummy);
    PxgBufferHandle bfBuf = createBufferFrom(dev, cs.bodyFrames);
    PxgBufferHandle wuBuf = createBufferFrom(dev, cs.workUnits);
    PxgBufferHandle cdBuf = createBufferFrom(dev, cs.cdata);
    PxgBufferHandle cpBuf = createBufferFrom(dev, cs.cpoints);
    PxgBufferHandle cpatchBuf = createBufferFrom(dev, currPatches);
    PxgBufferHandle ppatchBuf = createBufferFrom(dev, prevPatches);
    PxgBufferHandle canchBuf = createBufferFrom(dev, currAnchors);
    PxgBufferHandle panchBuf = createBufferFrom(dev, prevAnchors);
    PxgBufferHandle cfiBuf = createBufferFrom(dev, currFricIdx);
    PxgBufferHandle pfiBuf = createBufferFrom(dev, prevFricIdx);
    PxgBufferHandle chBuf = createBufferFrom(dev, chs);
    PxgBufferHandle fhBuf = createBufferFrom(dev, fhs);
    PxgBufferHandle outCPBuf = createBufferFrom(dev, outCP);
    PxgBufferHandle outFRBuf = createBufferFrom(dev, outFR);

    Kernel kern("constraintPrepPGS.spv", "contactConstraintBlockPrepareParallelLaunch",
                std::vector<uint32_t>(26, 4), sizeof(PrepParams));

    PrepParams p = {};
    p.numBatches = numBatches;
    p.totalPreviousEdges = totalPreviousEdges;
    p.totalCurrentEdges = numBatches * 32;
    p.hasPrevFrictionIndices = 1;
    p.dtOrStepDt = dt;
    p.invDtOrStepInvDt = invDt;
    p.totalDt = dt;
    p.invTotalDt = invDt;
    p.bounceThreshold = bounceThreshold;
    p.frictionOffsetThreshold = frictionOffsetThreshold;
    p.correlationDistance = correlationDistance;
    p.ccdMaxSeparation = ccdMaxSeparation;

    std::vector<PxgBufferHandle> bufs = { batchBuf, biBuf, bdBuf, txBuf, dummyBuf, dummyBuf, dummyBuf,
                                          dummyBuf, dummyBuf, dummyBuf, dummyBuf, dummyBuf, bfBuf, wuBuf,
                                          cdBuf, cpBuf, cpatchBuf, ppatchBuf, canchBuf, panchBuf, cfiBuf,
                                          pfiBuf, chBuf, fhBuf, outCPBuf, outFRBuf };
    kern.dispatch(numBatches, bufs, p);
    dev->flush();

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

    const auto gFR = downloadBuffer<BlockSolverContactFriction>(dev, outFRBuf, outFR.size());
    const auto gFH = downloadBuffer<BlockSolverFrictionHeader>(dev, fhBuf, fhs.size());
    const auto gAnch = downloadBuffer<BlockFrictionAnchorPatch>(dev, canchBuf, currAnchors.size());
    const auto gPatch = downloadBuffer<BlockFrictionPatch>(dev, cpatchBuf, currPatches.size());

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

    for (PxgBufferHandle h : { batchBuf, biBuf, bdBuf, txBuf, dummyBuf, bfBuf, wuBuf, cdBuf, cpBuf, cpatchBuf,
                               ppatchBuf, canchBuf, panchBuf, cfiBuf, pfiBuf, chBuf, fhBuf, outCPBuf, outFRBuf })
        dev->destroyBuffer(h);
}

} // namespace
