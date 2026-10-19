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


// pxgSlangSolverBlocks.slang
//
// P8.99a: the TGS solver block SoA structures as an IMPORTABLE module —
// hoisted from solverTGS.slang (which now imports them) so the articulation
// ext-body solve module (npArtiSolveExtTGS) can share them rather than carry
// a second copy.  Byte-identical mirrors of the engine structs in
// PxgSolverConstraintBlock1D.h / PxgSolverContact.h / PxgConstraintBlock.h;
// the layouts are pinned by the solver fixtures that write them from the
// HOST engine structs and compare byte for byte.
//
// solverPGS.slang keeps its own (PGS-typed) file-local mirrors — only the
// TGS family and the shared batch/response blocks live here.

public struct PxgBlockConstraintBatch     // 1536 bytes
{
    public uint descStride_constraintType;
    public uint blockContactIndex;
    public uint mConstraintBatchIndex;
    public uint startConstraintIndex;
    public uint startFrictionIndex;
    public uint mStartPartitionIndex;
    public uint mArticulationResponseIndex;
    public uint mask;
    public uint pad0[24];
    public uint2 bodyANodeIndex[32];
    public uint2 bodyBNodeIndex[32];
    public uint bodyAIndex[32];
    public uint bodyBIndex[32];
    public uint remappedBodyAIndex[32];
    public uint remappedBodyBIndex[32];
    public uint slabId[32];
    public uint2 shapeInteraction[32];
}

public struct PxgTGSBlockSolverContactHeader   // 2048 bytes
{
    public float4 invMass0_1_angDom0_1[32];  // 0
    public float4 normal_staticFriction[32]; // 512
    public float minNormalForce[32];         // 1024
    public float maxPenBias[32];             // 1152
    public uint flags[32];                   // 1280
    public uint numNormalConstr[32];         // 1408
    public uint forceWritebackOffset[32];    // 1536
    public float restitutionXdt[32];         // 1664
    public float cfm[32];                    // 1792
    public float biasCoefficient[32];        // 1920 (5.6.1: p8)
}

public struct PxgTGSBlockSolverFrictionHeader  // 1664 bytes
{
    public float4 frictionNormals0[32];   // 0
    public float4 frictionNormals1[32];   // 512
    public uint numFrictionConstr[32];    // 1024
    public float dynamicFriction[32];     // 1152
    public uint broken[32];               // 1280
    public float biasCoefficient[32];     // 1408
    public float torsionalFrictionScale[32]; // 1536
}

public struct PxgTGSBlockSolverContactPoint    // 1792 bytes
{
    public float4 raXn_extraCoeff[32]; // 0 (w = compliant coefficient "a")
    public float4 rbXn_targetVelW[32]; // 512
    public float separation[32];       // 1024
    public float maxImpulse[32];       // 1152
    public float appliedForce[32];     // 1280
    public float biasCoefficient[32];  // 1408
    public float resp0[32];            // 1536
    public float resp1[32];            // 1664
}

public struct PxgTGSBlockSolverContactFriction // 1408 bytes
{
    public float4 raXn_error[32];      // 0 (w = initial error)
    public float4 rbXn_targetVelW[32]; // 512
    public float appliedForce[32];     // 1024
    public float resp0[32];            // 1152
    public float resp1[32];            // 1280
}

public struct PxgTGSBlockSolverConstraint1DHeader  // 4992 bytes
{
    public uint rowCounts_breakable_orthoAxisCount[32]; // 0 (uchar4 per lane)
    public float4 rAWorld_invMass0D0[32];               // 128
    public float4 rBWorld_invMass1D1[32];               // 640
    public float invInertiaScale0[32];                  // 1152
    public float invInertiaScale1[32];                  // 1280
    public uint writeBackOffset[32];                    // 1408
    public float4 angOrthoAxis0_recipResponseW0[32];    // 1536
    public float4 angOrthoAxis0_recipResponseW1[32];    // 2048
    public float4 angOrthoAxis0_recipResponseW2[32];    // 2560
    public float4 angOrthoAxis1_ErrorW0[32];            // 3072
    public float4 angOrthoAxis1_ErrorW1[32];            // 3584
    public float4 angOrthoAxis1_ErrorW2[32];            // 4096
    public float linBreakImpulse[32];                   // 4608
    public float angBreakImpulse[32];                   // 4736
    public float cfm[32];                               // 4864
}

public struct PxgTGSBlockSolverConstraint1DCon     // 3200 bytes
{
    public float4 lin0XYZ_initBiasOrCoeff0[32];      // 0
    public float4 lin1XYZ_biasScaleOrCoeff1[32];     // 512
    public float4 ang0XYZ_velMultiplierOrCoeff2[32]; // 1024
    public float4 ang1XYZ_velTargetOrCoeff3[32];     // 1536
    public float resp0[32];                          // 2048
    public float resp1[32];                          // 2176
    public float geometricError[32];                 // 2304
    public float minImpulse[32];                     // 2432
    public float maxImpulse[32];                     // 2560
    public float maxBias[32];                        // 2688
    public float angularErrorScale[32];              // 2816
    public uint flags[32];                           // 2944
    public float appliedForce[32];                   // 3072
    // 5.12 removed the trailing residual[32] (row stride 3328 -> 3200)
}

// mirrors PxgArticulationBlockResponse (PxgConstraintBlock.h:99, 1536 bytes):
// the per-constraint-row articulation impulse response, twelve float[32] rows
public struct PxgArticulationBlockResponse
{
    public float deltaRALin_x[32];
    public float deltaRALin_y[32];
    public float deltaRALin_z[32];
    public float deltaRAAng_x[32];
    public float deltaRAAng_y[32];
    public float deltaRAAng_z[32];
    public float deltaRBLin_x[32];
    public float deltaRBLin_y[32];
    public float deltaRBLin_z[32];
    public float deltaRBAng_x[32];
    public float deltaRBAng_y[32];
    public float deltaRBAng_z[32];
}
