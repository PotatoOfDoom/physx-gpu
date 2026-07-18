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

// npSdfHoles.slang
//
// Slang port of the remaining SDF-construction portable units
// (SDFConstruction.cu), selected by mode:
//
// mode 0 -- bvhCalculateKeyDeltas (L85-101):
//   deltas = PxReal(a ^ b)  -- signed int XOR then int->float CONVERSION
//   (not a reinterpret; negative XORs of priority keys convert to negative
//   floats exactly as in CUDA)
// mode 1 -- bvhCalculateKeyDeltasSquaredDistance (L104-126):
//   centers via getCenter (0.5f*(lower+upper)); (cA - cB).magnitudeSquared()
// mode 2 -- storeQuantized (L536-556):
//   u8:  PxU8(255.0f * PxClamp(v, 0, 1));  u16: PxU16(65535.0f * PxClamp(...))
//   float->uint truncation; PxClamp = PxMin(1, PxMax(0, v)) selects
// mode 3 -- the findHoles compute core (L649-696; the globalScanExclusive
//   stream-compaction tail is atomic glue):
//   initial = sdf[id]; newValue = |initial|;  idToXYZ (mod/div, GuSDF.h L411)
//   3x3x3 loop z/y/x with bounds PxMax(1,p)-1 .. PxMin(dim-1, p+1), center
//   skipped, the (defensively dead) idx3D >= w*h*d guard transcribed;
//   PxSign(a) = (a >= 0) ? 1 : -1;  on sign change:
//     distance = sqrt(sum of cellSize_axis^2 over differing axes)
//     delta = |value - initial|
//     if (0.99f*delta > distance):
//       scaling = distance/delta;  v = 0.99f*scaling*initial  ((0.99*s)*i)
//       newValue = PxMin(newValue, |v|)
//   if (initial < 0) newValue = -newValue;  valueChanged = newValue != initial
//   The 27-value neighborhood block is passed directly (global sdf gathers
//   collapse); block index (x-px+1) + 3*(y-py+1) + 9*(z-pz+1), center 13.
//   An id >= w*h*d row leaves newValue = 0, valueChanged = false.
//
// All float arithmetic NoContraction-stamped; sqrt IEEE; +-0 eqFZ; ints exact.

float pabs(float a) { return a < 0.0f ? -a : a; }
float selMin(float a, float b) { return a < b ? a : b; }
float selMax(float a, float b) { return a > b ? a : b; }
uint uselMin(uint a, uint b) { return a < b ? a : b; }
uint uselMax(uint a, uint b) { return a < b ? b : a; }
float psign(float a) { return (a >= 0.0f) ? 1.0f : -1.0f; }

struct HoIn
{
    float4 misc;    // asfloat(mode), vNormalized, asfloat(bytesPerPixel), asfloat(keyA i32)
    float4 misc2;   // asfloat(keyB i32), asfloat(id u32), asfloat(width u32), asfloat(height u32)
    float4 misc3;   // asfloat(depth u32), cellSize.xyz
    float4 la, ua, lb, ub;   // mode 1: item lower/upper pairs
    float4 sdf0, sdf1, sdf2, sdf3, sdf4, sdf5, sdf6;  // mode 3: 27-value block (+1 pad)
}

struct HoOut
{
    float4 aux;     // mode 0/1: result float; mode 2: asfloat(quantized u32);
                    // mode 3: newValue, asfloat(valueChanged)
}

struct HoParams { uint numTests; }

[[vk::push_constant]]
ConstantBuffer<HoParams> gParams;

RWStructuredBuffer<HoIn> gIn;    // binding 0
RWStructuredBuffer<HoOut> gOut;  // binding 1

float sdfAt(HoIn s, uint idx)
{
    // 27-value block packed across sdf0..sdf6 (x fastest)
    const uint v4 = idx / 4u;
    const uint c = idx & 3u;
    float4 f;
    if (v4 == 0u) f = s.sdf0;
    else if (v4 == 1u) f = s.sdf1;
    else if (v4 == 2u) f = s.sdf2;
    else if (v4 == 3u) f = s.sdf3;
    else if (v4 == 4u) f = s.sdf4;
    else if (v4 == 5u) f = s.sdf5;
    else f = s.sdf6;
    return c == 0u ? f.x : c == 1u ? f.y : c == 2u ? f.z : f.w;
}

[shader("compute")]
[numthreads(64, 1, 1)]
void sdfHoles(uint3 dtid : SV_DispatchThreadID)
{
    const uint tid = dtid.x;
    if (tid >= gParams.numTests) return;
    const HoIn s = gIn[tid];

    const uint mode = asuint(s.misc.x);

    HoOut o;
    o.aux = float4(0.0f, 0.0f, 0.0f, 0.0f);

    if (mode == 0u)
    {
        // bvhCalculateKeyDeltas: PxReal(a ^ b) -- int convert
        const int a = asint(s.misc.w);
        const int b = asint(s.misc2.x);
        const int x = a ^ b;
        o.aux = float4(float(x), 0.0f, 0.0f, 0.0f);
    }
    else if (mode == 1u)
    {
        // bvhCalculateKeyDeltasSquaredDistance
        const float3 centerA = 0.5f * (s.la.xyz + s.ua.xyz);
        const float3 centerB = 0.5f * (s.lb.xyz + s.ub.xyz);
        const float3 d = centerA - centerB;
        const float distanceSquared = d.x * d.x + d.y * d.y + d.z * d.z;
        o.aux = float4(distanceSquared, 0.0f, 0.0f, 0.0f);
    }
    else if (mode == 2u)
    {
        // storeQuantized
        const float v = s.misc.y;
        const uint bytesPerPixel = asuint(s.misc.z);
        uint q = 0u;
        if (bytesPerPixel == 1u)
            q = uint(255.0f * selMin(1.0f, selMax(0.0f, v)));
        else if (bytesPerPixel == 2u)
            q = uint(65535.0f * selMin(1.0f, selMax(0.0f, v)));
        o.aux = float4(asfloat(q), 0.0f, 0.0f, 0.0f);
    }
    else
    {
        // findHoles compute core
        const uint id = asuint(s.misc2.y);
        const uint width = asuint(s.misc2.z);
        const uint height = asuint(s.misc2.w);
        const uint depth = asuint(s.misc3.x);
        const float3 cellSize = float3(s.misc3.y, s.misc3.z, s.misc3.w);

        bool valueChanged = false;
        float newValue = 0.0f;

        if (id < width * height * depth)
        {
            const float initialValue = sdfAt(s, 13u);   // sdf[id] = block center
            newValue = pabs(initialValue);

            // Gu::idToXYZ: x = id % w; id /= w; y = id % h; z = id / h
            uint rem = id;
            const uint px = rem % width; rem /= width;
            const uint py = rem % height;
            const uint pz = rem / height;

            for (uint z = uselMax(1u, pz) - 1u; z <= uselMin(depth - 1u, pz + 1u); ++z)
                for (uint y = uselMax(1u, py) - 1u; y <= uselMin(height - 1u, py + 1u); ++y)
                    for (uint x = uselMax(1u, px) - 1u; x <= uselMin(width - 1u, px + 1u); ++x)
                    {
                        if (x == px && y == py && z == pz)
                            continue;

                        // Gu::idx3D (defensively dead in-domain, transcribed)
                        const uint index = (z * height + y) * width + x;
                        if (index >= width * height * depth)
                            continue;

                        const float value = sdfAt(s, (x - px + 1u) + 3u * (y - py + 1u) + 9u * (z - pz + 1u));

                        if (psign(initialValue) != psign(value))
                        {
                            float distance = 0.0f;
                            if (x != px)
                                distance += cellSize.x * cellSize.x;
                            if (y != py)
                                distance += cellSize.y * cellSize.y;
                            if (z != pz)
                                distance += cellSize.z * cellSize.z;

                            distance = sqrt(distance);

                            const float delta = pabs(value - initialValue);

                            if (0.99f * delta > distance)
                            {
                                const float scaling = distance / delta;
                                const float v = 0.99f * scaling * initialValue;
                                newValue = selMin(newValue, pabs(v));
                            }
                        }
                    }

            if (initialValue < 0.0f)
                newValue = -newValue;

            valueChanged = newValue != initialValue;
        }

        o.aux = float4(newValue, valueChanged ? asfloat(1u) : asfloat(0u), 0.0f, 0.0f);
    }
    gOut[tid] = o;
}
