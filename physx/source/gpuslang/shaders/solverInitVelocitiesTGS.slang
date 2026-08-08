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

// solverInitVelocitiesTGS.slang
//
// Slang port of initializeSolverVelocitiesTGS
// (gpusolver/src/CUDA/solverMultiBlockTGS.cu).  Before the TGS iterative solve
// begins, this kernel seeds the solver body-velocity pool with each body's
// pre-solve velocity, repacked into the 6-component / 32-strided slab layout:
//   vel0 = (linVel.xyz, angVel.x)
//   vel1 = (angVel.y, angVel.z, 0, 0)
//   vel2 = 0
// For every slab reference of a dynamic body it scatters (vel0, vel1, vel2) into
// the remapped slot triple [r, r+32, r+64]; and once per body it writes the
// accumulation-offset copy at [outputOffset + bodyId + {0, N, 2N}] (N = body
// count).
//
// Work layout: the CUDA uses a 2D block (32 lanes x blockDim.y = 8 warps) with
// numThreadsPerBody = min(nextPow2(numSlabs), 32) cooperating threads per body
// and numBodiesPerWarp = 32 / numThreadsPerBody bodies per warp.  Here each warp
// maps to one 32-thread group (group id == warpIndex == blockIdx.x*8 +
// threadIdx.y), so the dispatch multiplies the engine's block count by 8.
//
// P5 Step 4 BDA engine-launch ABI: the kernel takes the engine's
// (PxgSolverCoreDesc*, PxgSolverSharedDesc<IterativeSolveDataTGS>*) pack and
// reads bodyOffset / numSlabs / numSolverBodies / accumulatedBodyDeltaVOffset
// and the pool pointers on-device through the byte-exact pxgSlangSolverDesc
// mirrors (proven by testSolverDescLayout); numThreadsPerBody is computed
// on-device from numSlabs exactly like the .cu.

import pxgSlangSolverDesc;

struct InitVelAbiParams
{
    uint64_t solverCoreDesc; // @0 const PxgSolverCoreDesc*
    uint64_t sharedDesc;     // @8 const PxgSolverSharedDesc<IterativeSolveDataTGS>*
}

[[vk::push_constant]]
ConstantBuffer<InitVelAbiParams> gParams;

static const uint INVALID = 0xffffffffu;

uint nextPowerOfTwo(uint x)
{
    x -= 1;
    x |= x >> 1;
    x |= x >> 2;
    x |= x >> 4;
    x |= x >> 8;
    x |= x >> 16;
    return x + 1;
}

[shader("compute")]
[numthreads(32, 1, 1)]
void initializeSolverVelocitiesTGS(uint3 gtid : SV_GroupThreadID, uint3 gid : SV_GroupID)
{
    PxgSolverCoreDescGpu desc = (Ptr<PxgSolverCoreDescGpu>(gParams.solverCoreDesc))[0];
    PxgSolverSharedDescTGSGpu shared = (Ptr<PxgSolverSharedDescTGSGpu>(gParams.sharedDesc))[0];

    Ptr<uint> solverReferences = Ptr<uint>(desc.solverBodyReferences); // PxgSolverReferences
    Ptr<float4> bodyVelocities = Ptr<float4>(shared.iter_solverBodyVelPool);
    Ptr<PxgIslandContextGpu> islandContextPool = Ptr<PxgIslandContextGpu>(desc.islandContextPool);
    const uint bodyOffset = islandContextPool[0].mBodyStartIndex;
    const uint numSlabs = desc.numSlabs;
    const uint allBodiesCount = desc.numSolverBodies;
    Ptr<float4> velocities = Ptr<float4>(desc.outSolverVelocity);
    const uint outputOffset = desc.accumulatedBodyDeltaVOffset;

    // THE SIZE OF solverBodyVelPool, DERIVED EXACTLY -- not guessed.
    // PxgTGSCudaSolverCore.cpp:412-423 allocates
    //     totalBodiesSize = mirrorBodies + accumulationBodies + numOutputBodies
    // and sets mSolverBodyOutputVelocityOffset = mirrorBodies + accumulationBodies,
    // with numOutputBodies = roundUp32(numSolverBodies) * 3.  PxgSolverCore.cpp:469
    // assigns that same offset to accumulatedBodyDeltaVOffset, so the pool's total
    // float4 count is exactly the offset plus the output region -- both of which
    // this shader already holds.  No descriptor change, no inferred layout.
    const uint velPoolSize = outputOffset + (((allBodiesCount + 31u) & ~31u) * 3u);

    const uint pow2Slabs = (numSlabs & (numSlabs - 1u)) == 0 ? numSlabs : nextPowerOfTwo(numSlabs);
    const uint numThreadsPerBody = min(pow2Slabs, 32u);
    const uint numBodiesPerWarp = 32u / numThreadsPerBody;

    const uint warpIndex = gid.x;                       // one 32-thread group per warp
    const uint lane = gtid.x;
    const uint threadIdxInWorkUnit = lane & (numThreadsPerBody - 1u);
    const uint bodyInWarp = lane / numThreadsPerBody;
    const uint bodyId = warpIndex * numBodiesPerWarp + bodyInWarp;

    if (bodyId < allBodiesCount)
    {
        const float4 linVel = velocities[bodyId];
        const float4 angVel = velocities[bodyId + allBodiesCount];

        const float4 vel0 = float4(linVel.x, linVel.y, linVel.z, angVel.x);
        const float4 vel1 = float4(angVel.y, angVel.z, 0.0f, 0.0f);
        const float4 vel2 = float4(0.0f, 0.0f, 0.0f, 0.0f);

        if (bodyId >= bodyOffset)
        {
            for (uint b = threadIdxInWorkUnit; b < numSlabs; b += numThreadsPerBody)
            {
                const uint bodyIndex = numSlabs * (bodyId - bodyOffset) + b;
                const uint remappedBodyIndex = solverReferences[bodyIndex];

                // THE BOUND IS A DELIBERATE DIVERGENCE FROM THE CUDA, and it is the
                // only one in this shader.  `remappedBodyIndex` is read straight out
                // of solverBodyReferences and used to address the pool; the CUDA
                // guards it against INVALID and nothing else, which is safe there
                // only because the constraint-prep path always fills that array.
                // When an upstream kernel is UNMAPPED -- as every `arti*`
                // constraint-prep kernel still is -- the array holds allocator
                // garbage, garbage is overwhelmingly unlikely to equal INVALID, and
                // the write lands at an arbitrary index.  On llvmpipe that is a
                // SIGSEGV inside JIT code with no usable stack, which is precisely
                // the fault P8.55-P8.57 spent three tranches chasing.
                //
                // Rejecting an out-of-pool index cannot change a correct execution:
                // a legitimate index is in the pool by construction.  It only
                // suppresses a write that is already undefined behaviour, and turns
                // a process kill into a survivable frame.
                if (remappedBodyIndex != INVALID && (remappedBodyIndex + 64u) < velPoolSize)
                {
                    bodyVelocities[remappedBodyIndex] = vel0;
                    bodyVelocities[remappedBodyIndex + 32u] = vel1;
                    bodyVelocities[remappedBodyIndex + 64u] = vel2;
                }
            }
        }

        if (threadIdxInWorkUnit == 0u)
        {
            bodyVelocities[outputOffset + bodyId] = vel0;
            bodyVelocities[outputOffset + bodyId + allBodiesCount] = vel1;
            bodyVelocities[outputOffset + bodyId + 2u * allBodiesCount] = vel2;
        }
    }
}
