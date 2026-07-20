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

// solverZeroBodies.slang
//
// Slang port of the solver body-velocity reset kernels that run once before the
// iterative solve begins:
//   - ZeroBodies    (gpusolver/src/CUDA/solverMultiBlock.cu)    — PGS
//   - ZeroBodiesTGS (gpusolver/src/CUDA/solverMultiBlockTGS.cu) — TGS
//
// Both zero the motion-velocity array and the solver body-velocity pool over
// ranges derived from the solver-body and constraint-batch counts.  The CUDA
// reads those counts and the delta-V offset from device descriptors
// (PxgSolverCoreDesc / PxgSolverSharedDesc); here the host supplies the already
// resolved counts as push constants, so the kernels reduce to plain grid-stride
// float4 stores.
//
// Range definitions (matching the CUDA):
//   PGS  totalNumBodies            = numSolverBodies * 2
//        totalNumBodiesConstraints = (numBatches + numArticBatches) * 32 * 2 * 2
//        motionVelocities[a] = 0           for a in [0, totalNumBodies)
//        bodyVelocities[offset + a] = 0    for a in [0, totalNumBodies)
//        bodyVelocities[a] = 0             for a in [0, totalNumBodiesConstraints)
//   TGS  totalNumBodies            = numSolverBodies * 2
//        totalNumBodiesConstraints = (numArticBatches + numBatches) * 32 * 3 * 2
//                                     + numSolverBodies * 3
//        motionVelocities[a] = 0           for a in [0, totalNumBodies)
//        bodyVelocities[a] = 0             for a in [0, totalNumBodiesConstraints)
//
// The CUDA launches these with a 1D grid; the grid stride blockDim.x*gridDim.x
// is passed as gridStride so the host's launch geometry is reproduced exactly.
//
// ZeroBodiesTGS is on the P5 Step 4 BDA engine-launch ABI: it takes the
// engine's (PxgSolverCoreDesc*, PxgSolverSharedDesc<IterativeSolveDataTGS>*)
// pack and reads the counts and pool pointers on-device through the
// byte-exact pxgSlangSolverDesc mirrors (proven by testSolverDescLayout).
// The PGS ZeroBodies entry still uses the harness push constants + bindings.

import pxgSlangSolverDesc;

struct ZeroBodiesParams
{
    uint totalNumBodies;             // numSolverBodies * 2
    uint totalNumBodiesConstraints;  // see ranges above
    uint offset;                     // accumulatedBodyDeltaVOffset (PGS only)
    uint gridStride;                 // blockDim.x * gridDim.x
}

[[vk::push_constant]]
ConstantBuffer<ZeroBodiesParams> gParams;

RWStructuredBuffer<float4> gMotionVelocities;  // binding 0  motionVelocityArray
RWStructuredBuffer<float4> gBodyVelocities;    // binding 1  solverBodyVelPool

static const float4 ZERO = float4(0.0f, 0.0f, 0.0f, 0.0f);

// ---------------------------------------------------------------------------
// ZeroBodies (PGS)
// ---------------------------------------------------------------------------
[shader("compute")]
[numthreads(256, 1, 1)]
void ZeroBodies(uint3 dtid : SV_DispatchThreadID)
{
    const uint stride = gParams.gridStride;
    const uint offset = gParams.offset;

    // (1) motion velocities, and (2) the delta velocities at the accumulation offset
    for (uint a = dtid.x; a < gParams.totalNumBodies; a += stride)
    {
        gMotionVelocities[a] = ZERO;
        gBodyVelocities[offset + a] = ZERO;
    }

    // (3) all solver body velocities
    for (uint a = dtid.x; a < gParams.totalNumBodiesConstraints; a += stride)
        gBodyVelocities[a] = ZERO;
}

// ---------------------------------------------------------------------------
// ZeroBodiesTGS (TGS) — engine ABI: (PxgSolverCoreDesc*, PxgSolverSharedDesc*)
// ---------------------------------------------------------------------------
struct ZeroBodiesTGSAbiParams
{
    uint64_t solverCoreDesc; // @0 const PxgSolverCoreDesc*
    uint64_t sharedDesc;     // @8 const PxgSolverSharedDesc<IterativeSolveDataTGS>*
}
[[vk::push_constant]]
ConstantBuffer<ZeroBodiesTGSAbiParams> gZeroTGS;

[shader("compute")]
[numthreads(256, 1, 1)]
void ZeroBodiesTGS(uint3 dtid : SV_DispatchThreadID)
{
    PxgSolverCoreDescGpu desc = (Ptr<PxgSolverCoreDescGpu>(gZeroTGS.solverCoreDesc))[0];
    PxgSolverSharedDescTGSGpu shared = (Ptr<PxgSolverSharedDescTGSGpu>(gZeroTGS.sharedDesc))[0];

    Ptr<float4> bodyVelocities = Ptr<float4>(shared.iter_solverBodyVelPool);
    Ptr<float4> motionVelocities = Ptr<float4>(desc.motionVelocityArray);
    // *2 because of linear and angular components
    const uint totalNumBodies = desc.numSolverBodies * 2u;
    const uint totalNumBodiesConstraints =
        (desc.numArticBatches + desc.numBatches) * 32u * 3u * 2u + desc.numSolverBodies * 3u;

    const uint stride = WorkgroupCount().x * 256u;

    // (1) motion velocities
    for (uint a = dtid.x; a < totalNumBodies; a += stride)
        motionVelocities[a] = ZERO;

    // (2) all solver body velocities
    for (uint a = dtid.x; a < totalNumBodiesConstraints; a += stride)
        bodyVelocities[a] = ZERO;
}
