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

// pxgArtiLinkRwRef.h
//
// 5.12 replaced the inverse-dynamics kernels' `body2World.p -
// parentBody2World.p` with computeLinkRwInvDyn (inverseDynamic.cu:40): Rw from
// the two world orientations and the joint's link-local anchors
// (joints[link].parentPose.p / childPose.p), plus the prismatic slide
// motionMatrix[link][0].bottom * jointPositions[jointData[link].jointOffset].
// The record-pool fixtures of those kernels share this seeding: per (record,
// link) anchors, a prismatic joint on some links, and per-dof joint
// positions; the expectation calls the ENGINE's Dy::computeLinkRw.

#ifndef PXG_ARTI_LINK_RW_REF_H
#define PXG_ARTI_LINK_RW_REF_H

#include "foundation/PxQuat.h"
#include "foundation/PxTransform.h"
#include "foundation/PxVec3.h"
#include "DyArticulationJointCore.h"
#include "DyCpuGpuArticulation.h"

#include <cstdint>
#include <cstring>
#include <vector>

namespace pxgtest
{

// some links carry exactly-zero anchors (a joint at the body origin, the
// common authoring case): computeLinkRw then evaluates `PxVec3(0.f) -
// childOffset` on +0 components, the IEEE 0 - (+0) = +0 a literal-zero fold
// would turn into -0, and rotates zero parent offsets
inline bool lrZeroAnchor(uint32_t rec, uint32_t link)
{
    return link != 0u && ((rec * 2u + link) % 3u) == 1u;
}

inline physx::PxVec3 lrParentOffset(uint32_t rec, uint32_t link)
{
    if (lrZeroAnchor(rec, link))
        return physx::PxVec3(0.0f);
    return physx::PxVec3(0.25f + 0.125f * float((rec + link) % 5u),
                         -0.375f + 0.0625f * float((rec * 3u + link) % 7u),
                         0.5f - 0.25f * float(link % 3u));
}

inline physx::PxVec3 lrChildOffset(uint32_t rec, uint32_t link)
{
    if (lrZeroAnchor(rec, link))
        return physx::PxVec3(0.0f, 0.0f, (link % 2u) ? 0.0f : 0.3125f);
    return physx::PxVec3(-0.125f - 0.0625f * float(link % 4u),
                         0.625f - 0.125f * float((rec + 2u * link) % 5u),
                         0.1875f + 0.125f * float((rec + link) % 3u));
}

// a prismatic joint on some links (never the root)
inline bool lrPrismatic(uint32_t rec, uint32_t link)
{
    return link != 0u && ((rec + link) % 4u) == 2u;
}

inline float lrJointPosition(uint32_t rec, uint32_t dof)
{
    return 0.375f - 0.125f * float((rec * 5u + dof * 3u) % 7u);
}

// joint-core images (sizeof(Dy::ArticulationJointCore) per link) carrying the
// anchors and the joint type; every other byte is the engine constructor's
inline void lrBuildJoints(std::vector<uint8_t>& joints, uint32_t nbRecords, uint32_t maxLinks)
{
    const size_t jc = sizeof(physx::Dy::ArticulationJointCore);
    joints.assign(size_t(nbRecords) * maxLinks * jc, 0xAB);
    for (uint32_t r = 0; r < nbRecords; ++r)
        for (uint32_t l = 0; l < maxLinks; ++l)
        {
            const physx::PxTransform parentPose(lrParentOffset(r, l), physx::PxQuat(physx::PxIdentity));
            const physx::PxTransform childPose(lrChildOffset(r, l), physx::PxQuat(physx::PxIdentity));
            physx::Dy::ArticulationJointCore j(parentPose, childPose);
            j.jointType = physx::PxU8(lrPrismatic(r, l) ? physx::PxArticulationJointType::ePRISMATIC
                                                        : physx::PxArticulationJointType::eREVOLUTE);
            std::memcpy(joints.data() + (size_t(r) * maxLinks + l) * jc, &j, jc);
        }
}

inline void lrBuildJointPositions(std::vector<float>& jp, uint32_t nbRecords, uint32_t maxDofs)
{
    jp.assign(size_t(nbRecords) * maxDofs, 0.f);
    for (uint32_t r = 0; r < nbRecords; ++r)
        for (uint32_t d = 0; d < maxDofs; ++d)
            jp[size_t(r) * maxDofs + d] = lrJointPosition(r, d);
}

// computeLinkRwInvDyn over the fixture's seeding; motionCol0Bottom is the
// link's motionMatrix column 0 bottom (read only for a prismatic joint)
inline physx::PxVec3 lrLinkRw(uint32_t rec, uint32_t link, const physx::PxQuat& parentQW,
                              const physx::PxQuat& childQW, uint32_t jointOffset,
                              const physx::PxVec3& motionCol0Bottom)
{
    physx::PxVec3 slide(0.f);
    if (lrPrismatic(rec, link))
        slide = motionCol0Bottom * lrJointPosition(rec, jointOffset);
    return physx::Dy::computeLinkRw(parentQW, childQW, lrParentOffset(rec, link),
                                    lrChildOffset(rec, link), slide);
}

} // namespace pxgtest

#endif
