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
* **`repro3.cpp`** — the same idea aimed at `sb_other_contact_remap_to_simLaunch`, with the
  fixture parameterised on the command line so a hypothesis is one re-run rather than one
  rebuild: `repro3 <spv> [candidates] [contactCount] [maxNumContacts] [sliverThickness]
  [probeX]`. Every buffer is 64 KB and the barycentric buffer is sentinel-filled, so an
  out-of-range write shows up as a changed value instead of silence. It found the two
  defects described under *What it found, part two* below.

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

## What it found, part two

`sb_other_contact_remap_to_simLaunch` (P8.2 tranche c) was carried unmapped for a while
behind a SIGSEGV that in-harness bisection had pinned on "the candidate loop nest", after
excluding the per-lane arrays, the dynamic indexing, the group size, both geometry helpers
and the trip count. All of those exclusions were wrong, and `repro3` showed why in two
runs. **Two independent llvmpipe defects were sitting on top of each other.**

**1. A conditional load is if-converted across its own null test.** The kernel's
`prevContactCount` argument is optional — CUDA writes `prevContactCount ? *prevContactCount
: 0` — and Slang emits a real branch for it (`OpSelectionMerge` with the `OpLoad` inside
the taken side; check with `spirv-dis`). The backend flattens it anyway:

```
mov   0x30(%rdx),%r10      ; r10 = prevContactCount == 0
vmovd (%r10),%xmm1         ; <-- faults, BEFORE the null test
neg   %r10 / sbb / kmovd   ; the "guard", evaluated afterwards
```

No arrangement of `if` / `?:` fixes this, because the branch is already there. The fix is
to remove the faulting *address*: select the pointer, not the loaded value, so the single
load reads something valid on both paths and the result is discarded by a second select.
**Any optionally-null device pointer in a BDA port needs that shape.**

This also explains why the earlier bisection pointed the wrong way. A read-only probe
survived while adding one store faulted, which reads as "stores are broken" — but a read
whose result is never stored is dead-code eliminated, so only the version keeping the
descriptor reads *live* ever reached the null load. Under DCE, "adding a store makes it
fault" means "the reads before it were being deleted".

**2. A loop loses its per-lane exit mask when the bound is redefined next to a store.**
With the contact-count clamp transcribed the way CUDA writes it —

```slang
uint n = count[0];
if (n > max) { n = max; if (firstThread) count[0] = max; }   // MISCOMPILED
for (uint i = base; i < n; i += stride) { ... }
```

— the whole first *subgroup* ran the loop once regardless of `n`, storing results for
out-of-range indices. The write-back itself was correct, so nothing reported an error: with
12 contacts clamped to 3, lanes 0..7 all wrote. Splitting the block two ways showed the
trigger is the **combination**: keeping the mutable `n` but lifting the store out is
correct, and keeping the nested store but computing `n` with `min` is correct. Only a block
that both redefines the loop's bound and stores is miscompiled. Writing it as a pure `min`
plus a flat conditional store removes both halves at once.

An **early return** in place of the loop is *not* affected — verified by patching the same
kernel into that shape, which is why `npSbRemap.slang`'s two identical-looking clamp sites
need no change. `particleSelfCollisionBda.slang` does have the loop shape (an accumulator
assigned in a storing conditional, then used as a bound) and is verified clean;
`particleSelfCollisionDivergentCounts` now pins it with counts that actually differ across
lanes, since the pre-existing fixture gave every lane the same trip count and could not
have seen it.

## `ultest` — the suffix sweep

`ultest.slang` / `ultest.cpp` is the follow-up: it masks one known device address with every
spelling of the constant and writes each result back, so the failure mode is a *printed
address* rather than a crash. On llvmpipe:

```
addr           0x00007f0d6fc7e004
& ~3UL         0x000000006fc7e004    <-- top half cleared
& ~3ull        0x00007f0d6fc7e004
& ~uint64_t(3) 0x00007f0d6fc7e004
```

So `ull` widens the complement correctly and `UL` does not. This was run as an audit over the
whole shader tree: the only other 64-bit masking site is `npCvxPlaneNphase.slang:159`, which
uses `~3ull` and is therefore correct. No second instance of the bug exists in the tree.

Build it the same way as `repro2` (see above), substituting `ultest` for the file stem.
