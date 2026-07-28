// Does `~3ull` behave like `~3UL` (32-bit complement, top half of the address cleared)?
struct P { uint64_t src; uint64_t dst; }
[[vk::push_constant]] ConstantBuffer<P> gP;

[shader("compute")]
[numthreads(1,1,1)]
void ulMain()
{
    // Write back what each masking form produces, so the host can compare the bit patterns.
    const uint64_t a = gP.src + 6UL;            // deliberately unaligned
    Ptr<uint64_t>(gP.dst)[0] = a;
    Ptr<uint64_t>(gP.dst)[1] = a & ~3UL;
    Ptr<uint64_t>(gP.dst)[2] = a & ~3ull;
    Ptr<uint64_t>(gP.dst)[3] = a & ~uint64_t(3);
    Ptr<uint64_t>(gP.dst)[4] = a - (a % 4UL);
}
