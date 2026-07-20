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

// npNpAbiProbe — P5 Step 4: empirical layout proof for the narrowphase
// element-struct mirrors in pxgSlangNpAbi.slang.  The host test fills
// TWO-element arrays of the REAL engine structs with distinct sentinels
// and this kernel reads every field of BOTH elements through the mirrors
// (element [1] proves the stride).  One wrong offset, one wrong stride,
// one drifted engine struct -> bitwise mismatch on the host.

import pxgSlangNpAbi;

struct NpAbiProbeParams
{
    uint64_t cmInput;    // PxgContactManagerInput[2]
    uint64_t shape;      // PxgShape[2]
    uint64_t xf;         // PxsCachedTransform[2]
    uint64_t manifold;   // PxgPersistentContactManifold[2]
    uint64_t cmOutput;   // PxsContactManagerOutput[2]
    uint64_t material;   // PxsMaterialData[2]
    uint64_t counters;   // PxgPatchAndContactCounters[2]
    uint64_t gpuPair;    // PxGpuContactPair[2]
    uint64_t outWords;   // uint64_t[PROBE_COUNT]
}

[[vk::push_constant]]
ConstantBuffer<NpAbiProbeParams> gProbe;

[shader("compute")]
[numthreads(1, 1, 1)]
void npAbiProbe(uint3 dtid : SV_DispatchThreadID)
{
    if (dtid.x != 0)
        return;
    Ptr<uint64_t> o = Ptr<uint64_t>(gProbe.outWords);
    uint i = 0;

    Ptr<PxgContactManagerInputGpu> cmIn = Ptr<PxgContactManagerInputGpu>(gProbe.cmInput);
    for (uint e = 0; e < 2; ++e)
    {
        o[i++] = uint64_t(cmIn[e].shapeRef0);
        o[i++] = uint64_t(cmIn[e].shapeRef1);
        o[i++] = uint64_t(cmIn[e].transformCacheRef0);
        o[i++] = uint64_t(cmIn[e].transformCacheRef1);
    }

    Ptr<PxgShapeGpu> sh = Ptr<PxgShapeGpu>(gProbe.shape);
    for (uint e = 0; e < 2; ++e)
    {
        o[i++] = uint64_t(asuint(sh[e].scaleX));
        o[i++] = uint64_t(asuint(sh[e].scaleY));
        o[i++] = uint64_t(asuint(sh[e].scaleZ));
        o[i++] = uint64_t(asuint(sh[e].scaleRotX));
        o[i++] = uint64_t(asuint(sh[e].scaleRotY));
        o[i++] = uint64_t(asuint(sh[e].scaleRotZ));
        o[i++] = uint64_t(asuint(sh[e].scaleRotW));
        o[i++] = uint64_t(sh[e].materialIndex);
        o[i++] = sh[e].hullOrMeshPtr;
        o[i++] = uint64_t(sh[e].type);
        o[i++] = uint64_t(sh[e].particleOrSoftbodyId);
    }

    Ptr<PxsCachedTransformGpu> xf = Ptr<PxsCachedTransformGpu>(gProbe.xf);
    for (uint e = 0; e < 2; ++e)
    {
        o[i++] = uint64_t(asuint(xf[e].qx));
        o[i++] = uint64_t(asuint(xf[e].qy));
        o[i++] = uint64_t(asuint(xf[e].qz));
        o[i++] = uint64_t(asuint(xf[e].qw));
        o[i++] = uint64_t(asuint(xf[e].px));
        o[i++] = uint64_t(asuint(xf[e].py));
        o[i++] = uint64_t(asuint(xf[e].pz));
        o[i++] = uint64_t(xf[e].flags);
    }

    Ptr<PxgPersistentContactManifoldGpu> mf = Ptr<PxgPersistentContactManifoldGpu>(gProbe.manifold);
    for (uint e = 0; e < 2; ++e)
    {
        // one probe per float4 array: element 0 and 3 (start + stride)
        o[i++] = uint64_t(asuint(mf[e].localNormalPen[0].x));
        o[i++] = uint64_t(asuint(mf[e].localNormalPen[3].w));
        o[i++] = uint64_t(asuint(mf[e].localContactA[0].x));
        o[i++] = uint64_t(asuint(mf[e].localContactA[3].w));
        o[i++] = uint64_t(asuint(mf[e].localContactB[0].x));
        o[i++] = uint64_t(asuint(mf[e].localContactB[3].w));
        o[i++] = uint64_t(asuint(mf[e].relativePos.x));
        o[i++] = uint64_t(asuint(mf[e].relativePos.w));
        o[i++] = uint64_t(asuint(mf[e].quatA.x));
        o[i++] = uint64_t(asuint(mf[e].quatB.w));
        o[i++] = uint64_t(mf[e].nbContacts);
        o[i++] = uint64_t(mf[e].nbWarmStartPoints);
        o[i++] = uint64_t(mf[e].warmStartA);
        o[i++] = uint64_t(mf[e].warmStartB);
    }

    Ptr<PxsContactManagerOutputGpu> cmOut = Ptr<PxsContactManagerOutputGpu>(gProbe.cmOutput);
    for (uint e = 0; e < 2; ++e)
    {
        o[i++] = cmOut[e].contactPatches;
        o[i++] = cmOut[e].contactPoints;
        o[i++] = cmOut[e].contactForces;
        o[i++] = cmOut[e].frictionPatches;
        o[i++] = uint64_t(cmOut[e].allflagsWord);
        o[i++] = uint64_t(cmOut[e].nbContactsFlags);
    }

    Ptr<PxsMaterialDataGpu> mat = Ptr<PxsMaterialDataGpu>(gProbe.material);
    for (uint e = 0; e < 2; ++e)
    {
        o[i++] = uint64_t(asuint(mat[e].dynamicFriction));
        o[i++] = uint64_t(asuint(mat[e].staticFriction));
        o[i++] = uint64_t(asuint(mat[e].restitution));
        o[i++] = uint64_t(asuint(mat[e].damping));
        o[i++] = uint64_t(mat[e].flagsModes0);
        o[i++] = uint64_t(mat[e].modes1 & 0xffu); // pad bytes excluded
    }

    Ptr<PxgPatchAndContactCountersGpu> pc = Ptr<PxgPatchAndContactCountersGpu>(gProbe.counters);
    for (uint e = 0; e < 2; ++e)
    {
        o[i++] = uint64_t(pc[e].patchesBytes);
        o[i++] = uint64_t(pc[e].contactsBytes);
        o[i++] = uint64_t(pc[e].forceAndIndiceBytes);
        o[i++] = uint64_t(pc[e].overflowError);
    }

    Ptr<PxGpuContactPairGpu> gp = Ptr<PxGpuContactPairGpu>(gProbe.gpuPair);
    for (uint e = 0; e < 2; ++e)
    {
        o[i++] = gp[e].contactPatches;
        o[i++] = gp[e].contactPoints;
        o[i++] = gp[e].contactForces;
        o[i++] = gp[e].frictionPatches;
        o[i++] = uint64_t(gp[e].transformCacheRef0);
        o[i++] = uint64_t(gp[e].transformCacheRef1);
        o[i++] = gp[e].nodeIndex0;
        o[i++] = gp[e].nodeIndex1;
        o[i++] = gp[e].actor0;
        o[i++] = gp[e].actor1;
        o[i++] = uint64_t(gp[e].nbContactsPatches);
    }
}
