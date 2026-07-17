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

// npJacobiRotate.slang
//
// Slang port of jacobiRotate (matrixDecomposition.cuh L101-136): one Jacobi
// rotation in the (p,q)-plane that zeroes the symmetric matrix element A(p,q),
// accumulating the rotation into R (the IEEE non-fast path, NOT the
// USE_FAST_JACOBI jacobiRotateT which uses __fdividef/rsqrtf).
//   if A(p,q)==0 return;
//   d = (A(p,p)-A(q,q)) / (2*A(p,q));  t = 1/(|d|+sqrt(d*d+1)); if d<0 t=-t;
//   c = 1/sqrt(t*t+1);  s = t*c;
//   A(p,p) += t*A(p,q);  A(q,q) -= t*A(p,q);  A(p,q)=A(q,p)=0;
//   for k!=p,q: Akp = c*A(k,p)+s*A(k,q); Akq = -s*A(k,p)+c*A(k,q);
//               A(k,p)=A(p,k)=Akp; A(k,q)=A(q,k)=Akq;
//   for k: Rkp = c*R(k,p)+s*R(k,q); Rkq = -s*R(k,p)+c*R(k,q); R(k,p)=Rkp; R(k,q)=Rkq;
//
// PxMat33 operator()(row,col) is the math element (column-major storage).  We
// store A/R row-major internally (A[row][col] = math element) so A(p,q)=A[p][q].
// fabs->abs, sqrtf->sqrt (IEEE).  All float mul/adds NoContraction-stamped; +-0 eqFZ.

struct JrIn
{
    float4 aC0, aC1, aC2;   // A columns (row-major math element A[r][c] = aC{c}[r])
    float4 rC0, rC1, rC2;   // R columns
    int4 pq;                // p, q, _, _
}

struct JrOut
{
    float4 aC0, aC1, aC2;   // A' columns
    float4 rC0, rC1, rC2;   // R' columns
}

struct JrParams { uint numTests; }

[[vk::push_constant]]
ConstantBuffer<JrParams> gParams;

RWStructuredBuffer<JrIn> gIn;    // binding 0
RWStructuredBuffer<JrOut> gOut;  // binding 1

[shader("compute")]
[numthreads(64, 1, 1)]
void jacobiRotate(uint3 dtid : SV_DispatchThreadID)
{
    const uint tid = dtid.x;
    if (tid >= gParams.numTests) return;
    const JrIn s = gIn[tid];

    // A[row][col] = math element; column c of PxMat33 holds rows 0,1,2
    float A[3][3];
    A[0][0] = s.aC0.x; A[1][0] = s.aC0.y; A[2][0] = s.aC0.z;
    A[0][1] = s.aC1.x; A[1][1] = s.aC1.y; A[2][1] = s.aC1.z;
    A[0][2] = s.aC2.x; A[1][2] = s.aC2.y; A[2][2] = s.aC2.z;
    float R[3][3];
    R[0][0] = s.rC0.x; R[1][0] = s.rC0.y; R[2][0] = s.rC0.z;
    R[0][1] = s.rC1.x; R[1][1] = s.rC1.y; R[2][1] = s.rC1.z;
    R[0][2] = s.rC2.x; R[1][2] = s.rC2.y; R[2][2] = s.rC2.z;

    const int p = s.pq.x, q = s.pq.y;

    if (A[p][q] != 0.0f)
    {
        const float apq = A[p][q];
        const float d = (A[p][p] - A[q][q]) / (2.0f * apq);
        float t = 1.0f / (abs(d) + sqrt(d * d + 1.0f));
        if (d < 0.0f) t = -t;
        const float c = 1.0f / sqrt(t * t + 1.0f);
        const float sn = t * c;

        A[p][p] += t * apq;
        A[q][q] -= t * apq;
        A[p][q] = 0.0f; A[q][p] = 0.0f;

        for (int k = 0; k < 3; ++k)
        {
            if (k != p && k != q)
            {
                const float Akp = c * A[k][p] + sn * A[k][q];
                const float Akq = -sn * A[k][p] + c * A[k][q];
                A[k][p] = Akp; A[p][k] = Akp;
                A[k][q] = Akq; A[q][k] = Akq;
            }
        }

        for (int k = 0; k < 3; ++k)
        {
            const float Rkp = c * R[k][p] + sn * R[k][q];
            const float Rkq = -sn * R[k][p] + c * R[k][q];
            R[k][p] = Rkp;
            R[k][q] = Rkq;
        }
    }

    JrOut o;
    o.aC0 = float4(A[0][0], A[1][0], A[2][0], 0.0f);
    o.aC1 = float4(A[0][1], A[1][1], A[2][1], 0.0f);
    o.aC2 = float4(A[0][2], A[1][2], A[2][2], 0.0f);
    o.rC0 = float4(R[0][0], R[1][0], R[2][0], 0.0f);
    o.rC1 = float4(R[0][1], R[1][1], R[2][1], 0.0f);
    o.rC2 = float4(R[0][2], R[1][2], R[2][2], 0.0f);
    gOut[tid] = o;
}
