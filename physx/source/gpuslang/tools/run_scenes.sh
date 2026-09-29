#!/bin/sh
# run_scenes.sh -- the canonical gpu-sdk-step scene sweep.
#
# Every tranche that changes the manifest must rebuild build-sdk and re-run the
# scenes.  Until P9.8 there was no runner for that in the tree, so each tranche
# improvised one -- and improvised runners have now made the same class of
# mistake twice:
#
#   * P9.3a passed `PXG_SCENE=particlebag`, which is not a scene name.  The
#     binary's unknown-scene branch prints NEITHER verdict marker, so a runner
#     that greps only for `[ FAIL ]` reads the silence as a pass.
#   * P9.8 passed `PXG_PARTICLE_STARTY=5.0` for the settling bag (canonical is
#     4.0, mean 4.75) and the cloud was still compacting when the rest window
#     opened; and it ran the three DIAGNOSTIC control modes as if they were
#     pass/fail scenes.  They are not: `PXG_PARTICLE_FLUID_CONTROL=1` exists to
#     FAIL the fluid checks -- its measured numbers are printed by the fluid
#     scene as the reference the real run must beat.
#
# So this runner does three things an ad-hoc loop does not:
#   1. it carries the scene inventory and the bag's canonical drop height;
#   2. it treats a MISSING verdict as its own outcome, never as a pass;
#   3. it keeps the diagnostics in a separate list that is reported and never
#      scored.
#
# Usage:   sh run_scenes.sh [--diagnostics]
# Expects: PXG_SHADER_DIR set, and the working directory to be the build-sdk
#          tree that holds gpu-sdk-step.

set -u

if [ ! -x ./gpu-sdk-step ]; then
    echo "run_scenes.sh: ./gpu-sdk-step not found -- run this from build-sdk" >&2
    exit 2
fi
# PXG_SHADER_DIR, derived rather than demanded (P9.50).
#
# This used to be a hard "not set -> exit 2", and refusing loudly is better than
# running on the wrong shaders.  But it is still a runner that only works when
# its caller remembers something, and in one session that cost three runs: twice
# in run_suite.sh, where a missing export turned into a RED SUITE that looked
# like the tranche's fault, and once here.
#
# The sibling build directory is the answer in every case that has ever been run:
# build-sdk and build-gpuslang sit next to each other, and the .spv live in the
# latter.  An explicit value from the environment still wins, so a deliberately
# different shader set stays testable; only the guessing is removed.
if [ -z "${PXG_SHADER_DIR:-}" ]; then
    for cand in ../build-gpuslang/shaders ./shaders; do
        if [ -d "$cand" ]; then
            PXG_SHADER_DIR="$(cd "$cand" && pwd)"
            break
        fi
    done
fi
if [ -z "${PXG_SHADER_DIR:-}" ] || [ ! -d "$PXG_SHADER_DIR" ]; then
    echo "run_scenes.sh: PXG_SHADER_DIR is not set and no shader directory was found" >&2
    echo "  looked for ../build-gpuslang/shaders and ./shaders next to $(pwd)" >&2
    exit 2
fi
export PXG_SHADER_DIR
echo "run_scenes.sh: PXG_SHADER_DIR=$PXG_SHADER_DIR" 

# The 32 named scenes.  The settling bag below makes 33 -- the figure
# RUNTIME_STATUS.md's header carries.  clothbox joined in P9.15a: `cloth` is a
# free fall and stays green whether or not the cloth lands, so the landing needed
# a scene of its own.  particlediffuse joined in P9.13d, when
# ps_diffuseParticleCreate made it pass; it had been a diagnostic since P9.13a.
# particleconvex joined in P9.14c and had been a pending-work entry since P9.14a:
# it named the two convex particle kernels, P9.14b landed the fluid one and
# P9.14c the diffuse one, and it is scored on BOTH.  clothconvex joined in P9.21
# the same way: it was a pending-work entry from P9.20, which mapped the TRIANGLE
# half of the cloth-vs-convex chain and measured the cloth still falling through
# to -73.81, and P9.21's VERTEX half is what catches it.  particlepush and
# articulationjoint joined in P9.44 and clothnet in P9.45; all three had been
# pending-work entries since P9.41, which added them to MEASURE physics that was
# silently missing rather than to score it.  clothnet is the one that took two
# tranches: P9.44 mapped the five kernels that ACCUMULATE the rigid delta, and the
# box still fell through to 0.2500 because their producer wrote zeros; P9.45
# ported that producer (FEMCollision::computeRigidChange) and the box now rests on
# the net at 2.1015.
#
# P9.58 added `hfbigbox`: hffloor with a half-3 box instead of a half-0.5 one.  Its
# only job is to exceed the convex heightfield core's single-pass ceiling -- the host
# launches that kernel with 32 warps per contact manager, so ~32 midphase pairs, and a
# half-0.5 box needs about eight.  Before the grid-stride loop was restored this scene
# dropped the box straight THROUGH the terrain (final y = -70.8, |v| = 39.2); the
# smaller box never reached the ceiling and every scene passed.
#
# P9.51c added `hffloor`: meshfloor with the cooked triangle mesh swapped for a
# cooked heightfield and nothing else changed, so the two are one comparison.  It
# is the observing scene for the rigid-vs-heightfield narrowphase, which before
# P9.51 was unmapped end to end and therefore a silent no-op -- the box fell
# through with no diagnostic.  The list is 36 named scenes plus the settling
# bag = 37.
SCENES="box stack plane sphere capsule capsulebox convex joint meshfloor spheremesh capsulemesh convexcoreplane convexcorebox convexcoremesh convexcoretrimesh convexcoretrough hffloor spherehf hfbigbox hfhugebox articulation
        particle particlefluid particlefluidboundary particlemeshfloor particlehffloor
        particlecloth particleinflatable particlediffuse particleconvex
        cloth clothbox clothconvex clothhang clothdamp clothdrape clothselfcollide clothcloth
        softbody softbodyselfcollide softbodycloth softbodystack softbodyattach softbodyhang
        particlepush articulationjoint clothnet particlesoftbody particleattach directapi particlerigid
        sdfmesh forcethreshold"

pass=0; fail=0; missing=0

# P9.40: --strict exports PXG_REQUIRE_MAPPED=1, under which a scene that launched an
# unmapped kernel FAILS.  Without it the scene is scored as before, but the kernels
# it silently skipped are listed under its line, so the sweep's own output says
# which "ok" is an ok about the physics and which is an ok about the checks.
if [ "${1:-}" = "--strict" ]; then export PXG_REQUIRE_MAPPED=1; shift; fi

# P9.90: --resume <log> re-scores every scene that <log> (the output of an earlier
# run) already lists as "ok" without running it again, and runs only the rest.  The
# container this runs in restarts now and then, and a full run is over an hour, so a
# run that dies at scene 40 must not cost the first 39 again.  A FAIL or MISSING in
# <log> is re-run, never carried over, and the summary counts carried-over scenes as
# ok the same way a fresh run would.  Only use a log produced from the same shaders.
RESUME_LOG=""
if [ "${1:-}" = "--resume" ]; then RESUME_LOG="${2:?--resume needs a log file}"; shift 2; fi
carried=0

run_one() {
    label="$1"; shift
    if [ -n "$RESUME_LOG" ] && grep -qxF "ok       $label" "$RESUME_LOG"; then
        echo "ok       $label   (carried over from $RESUME_LOG)"
        pass=$((pass+1)); carried=$((carried+1))
        return
    fi
    out=$(env "$@" PXG_GPU_STEP=1 ./gpu-sdk-step 2>&1)
    verdict=$(printf '%s\n' "$out" | grep -E '^\[ *(OK|FAIL) *\] gpu-sdk-step' | tail -1)
    unmapped=$(printf '%s\n' "$out" | grep -oE '\[pxg-unmapped\] [A-Za-z0-9_]+ was launched [0-9]+' \
               | sed -E 's/\[pxg-unmapped\] ([A-Za-z0-9_]+) was launched ([0-9]+)/\1(x\2)/' | tr '\n' ' ')
    case "$verdict" in
        "[  OK  ]"*)  echo "ok       $label"; pass=$((pass+1)) ;;
        "[ FAIL ]"*)  echo "FAIL     $label"
                      printf '%s\n' "$out" | grep -E '^FAIL:|^\[ FAIL \]' | head -4 | sed 's/^/           /'
                      fail=$((fail+1)) ;;
        *)            echo "MISSING  $label   (no verdict line -- mistyped scene or a crash)"
                      printf '%s\n' "$out" | tail -3 | sed 's/^/           /'
                      missing=$((missing+1)) ;;
    esac
    [ -n "$unmapped" ] && echo "           unmapped-but-launched: $unmapped"
}

for s in $SCENES; do
    run_one "$s" PXG_SCENE=$s
done

# Scene 26.  STARTY=4.0 is CANONICAL: it puts the cloud's mean at 4.75, which is
# the height every bag measurement since P6.5 has been taken from.  Dropping it
# from higher leaves the pile still compacting when the rest window opens, and
# the rest check (|dy| over the last 60 steps < 0.01) then fails on a scene that
# is working correctly.
run_one "particle (settling bag, STARTY=4.0)" PXG_SCENE=particle PXG_PARTICLE_STARTY=4.0

# THE SOFTBODY FLOOR MATRIX, added in P9.81.  `softbody` above runs the DEFAULT floor
# (the convex-hull ground box) and nothing else, so five of the scene's six floors --
# each of which routes the same cube to a DIFFERENT narrowphase kernel -- were never
# swept.  Three of them name a kernel this port had to supply, and those three are
# scored here.  It is the same cube at the same height every time, so their settle
# numbers are directly comparable, which is what makes the matrix a measurement
# rather than three separate scenes:
#
#   box (the default, above)  0.4893    mesh  0.4883    heightfield  0.4891
#
# plane and sphere are left unscored: both route to sb_primitiveContactGenLaunch,
# which the box arm already covers, so scoring them would buy runtime and no reach.
run_one "softbody (mesh floor)"        PXG_SCENE=softbody PXG_SB_FLOOR=mesh
run_one "softbody (convexcore floor)"  PXG_SCENE=softbody PXG_SB_FLOOR=convexcore
run_one "softbody (heightfield floor)" PXG_SCENE=softbody PXG_SB_FLOOR=heightfield

# P9.91: the force-threshold pipeline (accumulateThresholdStream.cu) is launched by BOTH
# GPU solver cores -- PxgCudaSolverCore and PxgTGSCudaSolverCore each carry their own
# copy of accumulatedForceThresholdStream() -- so the scene is scored under both.  Its
# reference is the CPU PGS solver (PXG_CPU_CONTROL=1 PXG_SOLVER=pgs); the CPU TGS solver
# builds no force-change stream at all (see the scene's header in testGpuSdkStep.cpp).
run_one "forcethreshold (PGS)" PXG_SCENE=forcethreshold PXG_SOLVER=pgs

echo "=== scenes: $pass ok, $fail fail, $missing missing (of $((pass+fail+missing))) ==="
[ "$carried" -gt 0 ] && echo "    ($carried ok carried over from $RESUME_LOG, $((pass+fail+missing-carried)) run now)"

if [ "${1:-}" = "--diagnostics" ]; then
    echo
    echo "=== diagnostics (NOT scored: these controls exist to fail the checks they"
    echo "    are the reference for -- read their numbers, not their verdicts) ==="
    for spec in "particlefluid PXG_PARTICLE_FLUID_CONTROL" \
                "particlecloth PXG_PARTICLE_CLOTH_CONTROL" \
                "particleinflatable PXG_PARTICLE_INFL_CONTROL" \
                "particlefluidboundary PXG_PARTICLE_BOUNDARY_CONTROL" \
                "particlemeshfloor PXG_PARTICLE_BOUNDARY_CONTROL" \
                "particlediffuse PXG_DIFFUSE_CONTROL"; do
        sc=${spec%% *}; var=${spec##* }
        echo "--- $sc + $var=1"
        env PXG_SCENE=$sc $var=1 PXG_GPU_STEP=1 ./gpu-sdk-step 2>&1 \
            | grep -E '^  particle |^FAIL:|^\[ *(OK|FAIL) *\] gpu-sdk-step' | sed 's/^/    /'
    done

    # A SECOND diagnostic list, and a different kind.  The controls above are
    # references that exist to fail; these are scenes that exist to NAME KERNELS
    # THAT ARE NOT MAPPED YET.  They fail because the work is not done, not
    # because failing is their job, and each moves into SCENES above in the
    # tranche that makes it pass -- as particlediffuse did in P9.13d.
    echo
    echo "=== pending work (NOT scored: these name kernels that are not mapped yet;"
    echo "    each moves into the scored list when its chain lands) ==="
    # P9.82a -- sdfmesh.  The FIRST DYNAMIC TRIANGLE MESH this harness cooks, and
    # therefore the first scene that can reach the rigid trimesh-vs-plane and
    # trimesh-vs-trimesh narrowphase at all: PhysX will not simulate a non-kinematic
    # PxTriangleMeshGeometry without an SDF, and no mesh in this tree had ever been
    # cooked with one.  Its RED measurement, on record:
    #   sdfBoxOnPlane  5.0000 -> -73.8070 (rest 0.500)   trimeshPlaneNarrowphase  183x SKIPPED
    #   sdfBoxOnMesh   5.0000 -> -73.8070 (rest 2.500)   triangleTriangleOverlaps  10x SKIPPED
    #                                                    triangleTriangleCollision 10x SKIPPED
    # P9.82 mapped the PLANE half, and the scene now reads the third row of its own
    # table -- "tri-tri missing", which is what it was built to be able to say:
    #   sdfBoxOnPlane  5.0000 -> 0.5000 (rest 0.500)  CAUGHT
    #   sdfBoxOnMesh   5.0000 -> 0.5000 (rest 2.500)  falls THROUGH the raised mesh
    #                                                 and lands on the plane instead
    # P9.83b-pre ADDED THE STACK ARM, because the scene above it could reach only ONE
    # of the kernel's two SDF arms and ONE of its two passes.  The plane has no SDF and
    # the shelf is cooked without one, so every contact took the "only one object has an
    # SDF" branch (convexMeshMidphase.cu:1100/:1120), which throws the SDF gradient away
    # and substitutes the triangle normal -- and the outer loop's second pass is entered
    # only `if (mesh0.mTexObject)` (:1045-1055), so it broke after pass 0 every time.
    # Two dynamic SDF boxes stacked put an SDF on BOTH sides of one contact manager.
    # Measured with the kernel still unmapped:
    #   sdfBoxStackBottom  2.0000 -> 0.5000 (rest 0.500)  PASSES -- the control, and it
    #                      rests through the PLANE kernel P9.82 mapped, so a failure of
    #                      the row below cannot be blamed on the plane path
    #   sdfBoxStackTop     6.0000 -> 0.5000 (rest 1.500)  falls THROUGH the bottom box
    #   triangleTriangleCollision launched 10x -> 44x     (the new pair really reaches it)
    # P9.83a mapped triangleTriangleOverlaps, the FIRST of the two mesh-vs-mesh
    # kernels.  The heights are unchanged -- the overlap byte is only the input to
    # the second kernel, so nothing it decides can reach a contact yet -- but the
    # measurement that DID change is the guard line:
    #   before  triangleTriangleOverlaps  10x SKIPPED
    #           triangleTriangleCollision 10x SKIPPED
    #   after   triangleTriangleCollision 10x SKIPPED     (one kernel, not two)
    # That is the whole observable P9.83a is entitled to claim, and it is worth
    # writing down precisely because the resting heights look identical.
    # TWO bodies, because a plane is infinite: one over the plane alone and one over a
    # static mesh raised to y = 2, so each kernel family has its own resting height and a
    # single number cannot be credited to the wrong one.
    # sdfmesh MOVED INTO THE SCORED LIST in P9.83b-8 T7, once triangleTriangleCollision
    # was mapped -- and once its box was wound OUTWARD: the inward winding had made
    # the cooked SDF positive inside (+0.475 at the centre), every mesh-mesh contact
    # read separation +0.025, and both mesh-supported boxes sank through at ~0.9 m/s
    # after landing.  Green measurement: 0.5000 / 2.5000 / 0.5000 / 1.5054.
    # particlerigid MOVED INTO THE SCORED LIST in P9.80, and its RED measurement is
    # the record of why it was ever here (kept in RUNTIME_STATUS.md, P9.80):
    #   cluster y-extent 1.5000 -> 0.0000  (all 64 particles flat on the floor)
    #   volume bound still (-999, -999, -999) -- the poison, untouched
    #   ps_solveShapes launched 2160x, ps_update_volume_bound 240x, all SKIPPED
    # Self-collision is deliberately OFF in that scene: with it ON the cluster kept
    # its shape (extent 1.4990 of 1.5000) with ps_solveShapes skipped entirely, so
    # the observable measured nothing.  See the scene comment for that correction.
    # softbodyhang MOVED INTO THE SCORED LIST in P9.76c, and its three measurements are
    # the record of why it was ever here (kept in RUNTIME_STATUS.md, P9.76b/c):
    #   ABI row 72 B, kernel REFUSED:            4.5000 -> -6.6589   (it falls off its pins)
    #   ABI row 80 B, kernel runs the PGS body:  4.5000 -> NaN       (a second defect, EXPOSED)
    #   ABI row 80 B, kernel runs the TGS body:  4.5000 ->  4.4969   (it hangs, and stays)
    echo "    clothmesh -- and it is NOT waiting on this port.  P9.26 mapped the last"
    echo "     two kernels of the cloth-vs-cooked-mesh chain, and the cloth still falls"
    echo "     to -39.3333 with |dy| = 0.452011: EXACTLY the figure measured before they"
    echo "     were mapped, to the digit.  That is the tranche's experiment and its"
    echo "     result.  cloth_meshVertexContactGenLaunch emits no contacts because its"
    echo "     only \`hasContacts\` assignment is commented out upstream, and by the"
    echo "     ordering law P9.15a/P9.21 measured -- the VERTEX path catches the cloth,"
    echo "     the TRIANGLE path refines -- a chain with no vertex producer cannot catch"
    echo "     one.  The triangle half does write (free fall is ~0.65 per step, this is"
    echo "     0.45), it just cannot hold it alone.  Scoring the scene would mean scoring"
    echo "     someone else's commented-out line, so it stays here and is measured."
    echo "     tools/check_disabled_kernels.py fails the build the day that line changes."
    echo "--- clothmesh"
    # NOT run_one: that increments the pass/fail counters this script exits on, and a
    # pending scene must never decide the sweep's verdict.  Printed, like the controls.
    env PXG_SCENE=clothmesh PXG_GPU_STEP=1 ./gpu-sdk-step 2>&1 \
        | grep -E '^  clothmesh:|^FAIL:|^\[ *(OK|FAIL) *\] gpu-sdk-step' | sed 's/^/    /'

    # P9.41's three observing scenes have all moved into SCENES above: particlepush and
    # articulationjoint in P9.44, clothnet in P9.45, and particlesoftbody in P9.48d --
    # it spent exactly one tranche in this block, which is how long it took to find why
    # it failed.  The pending-work list is back to clothmesh alone, and that one is not
    # waiting on this port.
fi

[ "$fail" -eq 0 ] && [ "$missing" -eq 0 ]
