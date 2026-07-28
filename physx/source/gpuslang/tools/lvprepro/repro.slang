// Minimal reproducer for an llvmpipe fault seen while porting PhysX's softbody FEM solve
// to Slang/Vulkan.  No PhysX dependency: three buffer-device-address arrays and a push
// constant, one workgroup, four threads' worth of work done by thread 0.
//
// Shape taken from the real kernel: gather four float4 vertices through an indirection
// (a per-item index selects between two arrays), compute a per-vertex delta, scatter the
// vertices back through a second indirection.
//
// Observed: with DELTA == 0 the shader runs; with any non-zero DELTA it SIGSEGVs inside
// the llvmpipe JIT worker thread.

struct Params
{
    uint64_t posA;       // @0   float4[]  "unordered" source
    uint64_t posB;       // @8   float4[]  "ordered" source AND destination
    uint64_t pullIdx;    // @16  uint4[]   selects A vs B per vertex
    uint64_t writeIdx;   // @24  uint4[]   destination slots
    uint count;          // @32
    float delta;         // @36  set to 0 to make it run, non-zero to fault
}
[[vk::push_constant]] ConstantBuffer<Params> gP;

static const uint PULL_MASK = 0x7fffffffu;

void gather(Ptr<float4> a, Ptr<float4> b, uint pull, uint bIndex, out float4 p)
{
    if (pull < PULL_MASK) p = a[pull];
    else                  p = b[bIndex];
}

void scatter(Ptr<float4> b, uint index, float4 p, float3 d)
{
    b[index] = float4(p.x + d.x, p.y + d.y, p.z + d.z, p.w);
}

[shader("compute")]
[numthreads(64, 1, 1)]
void reproMain(uint3 gtid : SV_GroupThreadID, uint3 gid : SV_GroupID)
{
    const uint work = gtid.x + gid.x * 64u;
    if (work >= gP.count) return;

    Ptr<float4> a = Ptr<float4>(gP.posA);
    Ptr<float4> b = Ptr<float4>(gP.posB);

    uint4 pull = Ptr<uint4>(gP.pullIdx)[work];
    pull.x &= PULL_MASK;

    float4 p0, p1, p2, p3;
    gather(a, b, pull.x, work + 0u * gP.count, p0);
    gather(a, b, pull.y, work + 1u * gP.count, p1);
    gather(a, b, pull.z, work + 2u * gP.count, p2);
    gather(a, b, pull.w, work + 3u * gP.count, p3);

    const float d = gP.delta;
    const float3 c0 = float3(d, d * 2.0f, d * 3.0f);
    const float3 c1 = float3(d * 4.0f, d * 5.0f, d * 6.0f);
    const float3 c2 = float3(d * 7.0f, d * 8.0f, d * 9.0f);

    const uint4 wi = Ptr<uint4>(gP.writeIdx)[work];
    scatter(b, wi.x, p0, -(c0 + c1 + c2) * p0.w);
    scatter(b, wi.y, p1, c0 * p1.w);
    scatter(b, wi.z, p2, c1 * p2.w);
    scatter(b, wi.w, p3, c2 * p3.w);
}
