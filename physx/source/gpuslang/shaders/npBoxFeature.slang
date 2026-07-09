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

// npBoxFeature.slang
//
// Slang port of the feature -> newTransformV switch of doBoxBoxGenerateContacts()
// from gpunarrowphase/src/CUDA/cudaBox.cu (lines 640-794).  Given the SAT-selected
// separating feature (0..5 = box A face x/y/z, box B face x/y/z) and its sign, it
// builds the reference-face frame:
//   - mtd            : the minimum-translation (separating) axis
//   - newTransformV  : the reference face's local frame (rotation columns + origin)
//   - e0, e1         : the reference face's two in-plane half-extents
// then derives the two quantities that feed getIncidentPolygon4:
//   - transform1ToNew = newTransformV.transformTranspose(other), other being the
//     OTHER box's transform (box B for a box-A feature, box A for a box-B feature)
//   - localNormal     = newTransformV.m.transformTranspose(mtd)
//
// This step is per-pair uniform (in the CUDA all 4 lanes compute it identically
// from the broadcast feature/sign), so the port runs one thread per pair with no
// groupshared / barriers.  The only branch, `0 >= sign`, tests the uploaded sign
// bit-identically on GPU and CPU.

struct FeatIn
{
    float4 b0e, b1e;            // box0Extent / box1Extent (.xyz)
    float4 c00, c01, c02, p0;   // transform0: rotation columns + translation
    float4 c10, c11, c12, p1;   // transform1
    float4 misc;                // (feature as float, sign, unused, unused)
}

struct FeatOut
{
    float4 mtd;
    float4 ntCol0, ntCol1, ntCol2, ntP;   // newTransformV
    float4 e;                             // (e0, e1, 0, 0)
    float4 t1nCol0, t1nCol1, t1nCol2, t1nP; // transform1ToNew
    float4 localNormal;
}

struct BoxFeatParams { uint numPairs; }

[[vk::push_constant]]
ConstantBuffer<BoxFeatParams> gParams;

RWStructuredBuffer<FeatIn>  gIn;    // binding 0
RWStructuredBuffer<FeatOut> gOut;   // binding 1

// PxMat33::transformTranspose(v) = (col0.v, col1.v, col2.v)
float3 mat33TransposeMul(float3 col0, float3 col1, float3 col2, float3 v)
{
    return float3(dot(col0, v), dot(col1, v), dot(col2, v));
}

// ---------------------------------------------------------------------------
// boxFeatureSetup: mirrors the feature switch of doBoxBoxGenerateContacts()
// ---------------------------------------------------------------------------
[shader("compute")]
[numthreads(256, 1, 1)]
void boxFeatureSetup(uint3 dtid : SV_DispatchThreadID)
{
    const uint t = dtid.x;
    if (t >= gParams.numPairs) return;

    const FeatIn s = gIn[t];
    const float3 box0e = s.b0e.xyz, box1e = s.b1e.xyz;
    const float3 a00 = s.c00.xyz, a01 = s.c01.xyz, a02 = s.c02.xyz; // transform0 axes
    const float3 a10 = s.c10.xyz, a11 = s.c11.xyz, a12 = s.c12.xyz; // transform1 axes
    const float3 tp0 = s.p0.xyz, tp1 = s.p1.xyz;
    const uint feature = (uint)(s.misc.x + 0.5f);
    const float sign = s.misc.y;
    const bool neg = 0.0f >= sign;

    float3 mtd, col0, col1, col2, p;
    float e0, e1;

    if (feature == 0u) // ua0
    {
        if (neg) { mtd = a00;  col0 = -a02; col1 = a01;  col2 = a00; }
        else     { mtd = -a00; col0 = a02;  col1 = a01;  col2 = -a00; }
        p = tp0 - mtd * box0e.x; e0 = box0e.z; e1 = box0e.y;
    }
    else if (feature == 1u) // ua1
    {
        if (neg) { mtd = a01;  col0 = a00; col1 = -a02; col2 = a01; }
        else     { mtd = -a01; col0 = a00; col1 = a02;  col2 = -a01; }
        p = tp0 - mtd * box0e.y; e0 = box0e.x; e1 = box0e.z;
    }
    else if (feature == 2u) // ua2
    {
        if (neg) { mtd = a02;  col0 = a00; col1 = a01;  col2 = a02; }
        else     { mtd = -a02; col0 = a00; col1 = -a01; col2 = -a02; }
        p = tp0 - mtd * box0e.z; e0 = box0e.x; e1 = box0e.y;
    }
    else if (feature == 3u) // ub0
    {
        if (neg) { mtd = a10;  col0 = a12;  col1 = a11; col2 = -a10; }
        else     { mtd = -a10; col0 = -a12; col1 = a11; col2 = a10; }
        p = tp1 + mtd * box1e.x; e0 = box1e.z; e1 = box1e.y;
    }
    else if (feature == 4u) // ub1
    {
        if (neg) { mtd = a11;  col0 = a10; col1 = a12;  col2 = -a11; }
        else     { mtd = -a11; col0 = a10; col1 = -a12; col2 = a11; }
        p = tp1 + mtd * box1e.y; e0 = box1e.x; e1 = box1e.z;
    }
    else // feature == 5, ub2
    {
        if (neg) { mtd = a12;  col0 = a10; col1 = -a11; col2 = -a12; }
        else     { mtd = -a12; col0 = a10; col1 = a11;  col2 = a12; }
        p = tp1 + mtd * box1e.z; e0 = box1e.x; e1 = box1e.y;
    }

    // transform1ToNew = newTransformV.transformTranspose(other):
    //   other = (feature < 3) ? transform1 : transform0
    const bool useT1 = feature < 3u;
    const float3 oc0 = useT1 ? a10 : a00;
    const float3 oc1 = useT1 ? a11 : a01;
    const float3 oc2 = useT1 ? a12 : a02;
    const float3 oP = useT1 ? tp1 : tp0;

    const float3 t1n0 = mat33TransposeMul(col0, col1, col2, oc0);
    const float3 t1n1 = mat33TransposeMul(col0, col1, col2, oc1);
    const float3 t1n2 = mat33TransposeMul(col0, col1, col2, oc2);
    const float3 t1nP = mat33TransposeMul(col0, col1, col2, oP - p);
    const float3 localNormal = mat33TransposeMul(col0, col1, col2, mtd);

    FeatOut o;
    o.mtd = float4(mtd, 0.0f);
    o.ntCol0 = float4(col0, 0.0f);
    o.ntCol1 = float4(col1, 0.0f);
    o.ntCol2 = float4(col2, 0.0f);
    o.ntP = float4(p, 0.0f);
    o.e = float4(e0, e1, 0.0f, 0.0f);
    o.t1nCol0 = float4(t1n0, 0.0f);
    o.t1nCol1 = float4(t1n1, 0.0f);
    o.t1nCol2 = float4(t1n2, 0.0f);
    o.t1nP = float4(t1nP, 0.0f);
    o.localNormal = float4(localNormal, 0.0f);
    gOut[t] = o;
}
