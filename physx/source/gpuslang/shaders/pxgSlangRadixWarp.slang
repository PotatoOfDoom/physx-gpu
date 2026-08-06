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

// pxgSlangRadixWarp.slang
//
// `radixSortSingleWarpKeysOnly` (RadixSort.cuh:700-720) and the pass it drives
// (:579-699), as a 32-lane virtual warp.  Eight four-bit passes, least
// significant nibble first, so a full 32-bit key sorts ascending.
//
// This is what `sortTriangleIndices` (convexMeshPostProcess.cu:623) calls, and it
// is the one piece of the convex-vs-trimesh chain with NO Phase-1 Slang of any
// kind to lean on.
//
// ===========================================================================
// THE SHIFT IS THE WHOLE ALGORITHM
// ===========================================================================
//
// Both the histogram and the scatter select a sub-field by shifting a 1 by an
// amount that is DELIBERATELY out of range for every key that does not belong to
// the slot being accumulated:
//
//     accum = (1u << ((radix.x - bit) << 4));      // histogram, 2 radixes/word
//     accum0 = (1u << (radixRankX << 3));          // scatter, 4 radixes/word
//
// `radix - bit` is UNSIGNED.  For the radixes the word owns it is 0 or 1 (or 0..3
// in the scatter), giving a shift of 0, 16 (or 0, 8, 16, 24).  For every other
// radix it is 2 or more — or, when `radix < bit`, an enormous wrapped value — so
// the shift is 32 or more, and PTX `shl.b32` CLAMPS that to zero.  The foreign
// radix contributes NOTHING to a word it does not belong to.
//
// SPIR-V leaves a shift at or past the operand width UNDEFINED, and the plausible
// wrong answer — masking the amount to five bits, which is the x86 rule — would
// add 1 to the word's LOW half and quietly corrupt the histogram of a sort.  So
// every such shift here goes through `rsShl`, which returns zero explicitly.
//
// The same construct appears in `pxgSlangCvxTriEmit` with the OPPOSITE intent:
// there `(1 << 32) - 1` must come out as 0xFFFFFFFF, which is what clamping gives.
// One rule, two uses, and neither survives an assumption.
//
// ===========================================================================
// TWO MORE THINGS THE TRANSCRIPTION KEEPS
// ===========================================================================
//
// `rsSanitizeKeys` mirrors a FALLTHROUGH switch (RadixSort.cuh:117-125): three bad
// values set .y, .z and .w; two set .z and .w; one sets .w alone.  Written as a
// normal switch it would leave stale words in the tail of the final uint4, and
// they would sort in as ordinary keys.  It never touches .x, which is safe only
// because the caller's index arithmetic keeps at least one value good.
//
// THE PASS COUNT'S PARITY IS LOAD-BEARING.  The driver ping-pongs between the
// input and the temp buffer once per pass; with EIGHT passes the sorted result
// ends up back in the INPUT buffer, which is where `sortTriangleIndices` reads it.
// An odd count would leave the answer in the scratch half.

static const uint RS_WARP = 32u;
static const uint RS_RADIX_SIZE = 16u;
public static const uint RS_PASSES = 8u;

groupshared uint gsRsStage[RS_WARP];
groupshared uint gsRsRunsum[RS_RADIX_SIZE];

// --------------------------------------------------------------------------
// THE GUARDED SHIFT.  PTX clamps a shift at or past the register width to zero;
// SPIR-V leaves it undefined.  Written out so neither is assumed.
// --------------------------------------------------------------------------
uint rsShl(uint v, uint s) { return s >= 32u ? 0u : (v << s); }

uint rsExchange(uint lane, uint value, uint src)
{
    gsRsStage[lane] = value;
    GroupMemoryBarrierWithGroupSync();
    const uint r = gsRsStage[src & (RS_WARP - 1u)];
    GroupMemoryBarrierWithGroupSync();
    return r;
}

// warpReduction<AddOpPxU32>: XOR butterfly, result broadcast to every lane.
uint rsReduceAdd(uint lane, uint v)
{
    v += rsExchange(lane, v, lane ^ 16u);
    v += rsExchange(lane, v, lane ^ 8u);
    v += rsExchange(lane, v, lane ^ 4u);
    v += rsExchange(lane, v, lane ^ 2u);
    v += rsExchange(lane, v, lane ^ 1u);
    return v;
}

// warpScan<AddOpPxU32>: INCLUSIVE up-shuffle scan.
uint rsScanAddInclusive(uint lane, uint v)
{
    for (uint radius = 1u; radius < RS_WARP; radius <<= 1)
    {
        const uint n = rsExchange(lane, v, lane - radius);
        if (lane >= radius)
            v += n;
    }
    return v;
}

// RadixSort.cuh:48 — the four nibbles of a uint4 at `startBit`.
uint4 rsGetRadix(uint4 input, uint startBit)
{
    return uint4((input.x >> startBit) & 0xFu, (input.y >> startBit) & 0xFu,
                 (input.z >> startBit) & 0xFu, (input.w >> startBit) & 0xFu);
}

// RadixSort.cuh:110 — the FALLTHROUGH switch, written as a cascade so the
// fallthrough is explicit rather than implied.
void rsSanitizeKeys(inout uint4 keyValue, uint id, uint count)
{
    const uint goodVals = count - id;
    if (goodVals < 4u)
    {
        const uint badVals = 4u - goodVals;
        if (badVals >= 3u) keyValue.y = 0xffffffffu;
        if (badVals >= 2u) keyValue.z = 0xffffffffu;
        if (badVals >= 1u) keyValue.w = 0xffffffffu;
    }
}

// --------------------------------------------------------------------------
// One four-bit pass over `numUint4` uint4s.  `gInputKeys` and `gOutputKeys` are
// buffer device addresses; the output is addressed as plain uints, as the CUDA's
// reinterpret_cast does.
//
// `numUint4` and `gNumOfKeys` must be warp-uniform: both loops below are bounded
// by them and both contain barriers.
// --------------------------------------------------------------------------
void rsPass(uint lane, uint64_t gInputKeys, uint gNumOfKeys, uint numUint4,
            uint64_t gOutputKeys, uint startBit)
{
    // ---- histogram (radixSortWarp, RadixSort.cuh:135) ----------------------
    // Eight words, each holding TWO radix counts in 16 bits apiece.
    uint radixAccum[8];
    for (uint i = 0u; i < 8u; ++i)
        radixAccum[i] = 0u;

    for (uint i = lane; i < numUint4; i += RS_WARP)
    {
        uint4 keyValue = Ptr<uint4>(gInputKeys)[i];
        rsSanitizeKeys(keyValue, i * 4u, gNumOfKeys);
        const uint4 radix = rsGetRadix(keyValue, startBit);

        for (uint bit = 0u; bit < 16u; bit += 2u)
        {
            uint accum = rsShl(1u, (radix.x - bit) << 4u);
            accum += rsShl(1u, (radix.y - bit) << 4u);
            accum += rsShl(1u, (radix.z - bit) << 4u);
            accum += rsShl(1u, (radix.w - bit) << 4u);
            radixAccum[bit / 2u] += accum;
        }
    }

    // ---- the exclusive runsum over the sixteen radixes ---------------------
    uint accumValue = 0u;
    for (uint i = 0u; i < RS_RADIX_SIZE; i += 2u)
    {
        const uint val = rsReduceAdd(lane, radixAccum[i / 2u]);
        const uint v0 = val & 0xffffu;
        const uint v1 = val >> 16;

        if (lane == 0u)
        {
            gsRsRunsum[i] = accumValue;
            gsRsRunsum[i + 1u] = accumValue + v0;
        }
        accumValue += v0 + v1;
    }
    GroupMemoryBarrierWithGroupSync();

    // ---- scatter -----------------------------------------------------------
    for (uint i = 0u; i < numUint4; i += RS_WARP)
    {
        const uint inputKeyIndex = i + lane;

        uint4 keyValue = uint4(0u, 0u, 0u, 0u);
        uint4 radix = uint4(0xffu, 0xffu, 0xffu, 0xffu);

        // A lane past the end pads its radix with 0xff.  Eight passes of four
        // bits never produce a radix above 0xf, so 0xff can never match a real
        // one and the lane contributes nothing.
        if (inputKeyIndex < numUint4)
        {
            keyValue = Ptr<uint4>(gInputKeys)[inputKeyIndex];
            rsSanitizeKeys(keyValue, inputKeyIndex * 4u, gNumOfKeys);
            radix = rsGetRadix(keyValue, startBit);
        }

        for (uint r = 0u; r < RS_RADIX_SIZE; r += 4u)
        {
            const uint rankX = radix.x - r;
            const uint rankY = radix.y - r;
            const uint rankZ = radix.z - r;
            const uint rankW = radix.w - r;

            const uint accum = rsShl(1u, rankX << 3u) + rsShl(1u, rankY << 3u)
                             + rsShl(1u, rankZ << 3u) + rsShl(1u, rankW << 3u);

            uint val = rsScanAddInclusive(lane, accum);
            const uint val2 = rsExchange(lane, val, RS_WARP - 1u);

            // Inclusive minus own contribution == exclusive.
            val -= accum;

            if (accum != 0u)
            {
                if (rankX < 4u)
                {
                    const uint outputIndex = gsRsRunsum[radix.x] + ((val >> (rankX << 3u)) & 0xffu);
                    val += 1u << (rankX << 3u);
                    Ptr<uint>(gOutputKeys)[outputIndex] = keyValue.x;
                }
                if (rankY < 4u)
                {
                    const uint outputIndex = gsRsRunsum[radix.y] + ((val >> (rankY << 3u)) & 0xffu);
                    val += 1u << (rankY << 3u);
                    Ptr<uint>(gOutputKeys)[outputIndex] = keyValue.y;
                }
                if (rankZ < 4u)
                {
                    const uint outputIndex = gsRsRunsum[radix.z] + ((val >> (rankZ << 3u)) & 0xffu);
                    val += 1u << (rankZ << 3u);
                    Ptr<uint>(gOutputKeys)[outputIndex] = keyValue.z;
                }
                if (rankW < 4u)
                {
                    const uint outputIndex = gsRsRunsum[radix.w] + ((val >> (rankW << 3u)) & 0xffu);
                    val += 1u << (rankW << 3u);
                    Ptr<uint>(gOutputKeys)[outputIndex] = keyValue.w;
                }
            }

            GroupMemoryBarrierWithGroupSync();

            if (lane == 0u)
            {
                gsRsRunsum[r] += val2 & 0xffu;
                gsRsRunsum[r + 1u] += (val2 >> 8) & 0xffu;
                gsRsRunsum[r + 2u] += (val2 >> 16) & 0xffu;
                gsRsRunsum[r + 3u] += (val2 >> 24) & 0xffu;
            }

            GroupMemoryBarrierWithGroupSync();
        }
    }
}

// --------------------------------------------------------------------------
// Eight passes, ping-ponging.  The EVEN count is what puts the answer back in
// `gInputKeys`, which is where the caller reads it.
// --------------------------------------------------------------------------
public void rsSortSingleWarpKeysOnly(uint lane, uint64_t gInputKeys, uint gNumOfKeys,
                                     uint numUint4, uint64_t gTempKeys)
{
    uint64_t k0 = gInputKeys;
    uint64_t k1 = gTempKeys;

    for (uint i = 0u; i < RS_PASSES; ++i)
    {
        rsPass(lane, k0, gNumOfKeys, numUint4, k1, i * 4u);
        const uint64_t t = k0; k0 = k1; k1 = t;
    }
}
