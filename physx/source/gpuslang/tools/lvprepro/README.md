# `lvprepro` — a standalone Vulkan harness for isolating driver-level faults

Built to settle one question during the P8.0 softbody solve port: *is this SIGSEGV a bug
in the ported shader, or in Mesa llvmpipe?*  It answered **the shader** — see below — but
the harness is kept because that question recurs, and answering it inside the PhysX test
suite is slow and inconclusive.

Nothing here depends on PhysX. Two programs:

* **`repro.cpp` + `repro.slang`** — a from-scratch minimal case (gather through an
  indirection, per-vertex delta, scatter through a second indirection) with a `delta`
  knob on the command line. Useful as a template when you suspect a *shape* of code.
* **`repro2.cpp`** — dispatches a **real** compiled kernel from the build tree, building
  the `PxgSoftBody` descriptor by raw byte offset. This is the one that pays off: it runs
  the actual shader against synthetic data with no engine, no descriptor sets and no test
  framework in the way.

## Build and run

```sh
export PATH=/usr/local/lib/python3.11/dist-packages/slangtorch/bin:$PATH
slangc -O0 -fvk-use-entrypoint-name -target spirv -entry reproMain repro.slang -o repro.spv
g++ -O0 -g repro.cpp  -lvulkan -o repro   && ./repro 1
g++ -O0 -g repro2.cpp -lvulkan -o repro2  && ./repro2 ../../../../build-gpuslang/shaders/softBodySolveBda.spv
```

`repro2` takes the `.spv` path as its argument; edit the entry-point name near the bottom
to point it at a different kernel.

## What it found

`sb_gm_cp_solveTetrahedronsPartitionLaunch` faulted inside a driver worker thread. Weeks
of in-harness bisection had produced only a misleading correlation ("it crashes whenever
the computed delta is non-zero") because the material index feeding the crash is *only*
read on the path the delta enables — with the solve stubbed out, the read is dead code and
gets optimised away.

Dropped into this harness, the same shader faulted the same way, which took PhysX out of
the picture. Two substitutions then isolated it exactly:

```slang
Ptr<uint>(addr & ~3UL)[0]            // faults
Ptr<uint>(addr & ~uint64_t(3))[0]    // clean
Ptr<uint>(addr & 0xFFFFFFFFFFFFFFFCUL)[0]  // clean
Ptr<uint>(addr - (addr % 4UL))[0]    // clean
```

**Slang evaluates `~3UL` in 32 bits.** The resulting `0xFFFFFFFC` is zero-extended for the
64-bit AND, which silently clears the *top half* of a buffer-device address. The read then
lands at a garbage address and the driver faults — correctly.

This is worth knowing for any BDA port: the `UL` suffix does not make the *complement*
64-bit. Write `~uint64_t(3)`, or avoid the mask.
