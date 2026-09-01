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
if [ -z "${PXG_SHADER_DIR:-}" ]; then
    echo "run_scenes.sh: PXG_SHADER_DIR is not set" >&2
    exit 2
fi

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
# to -73.81, and P9.21's VERTEX half is what catches it.  The pending-work list
# at the bottom is empty again as a result, and stays in the script for the next
# scene that needs it.
SCENES="box stack plane sphere capsule capsulebox convex joint meshfloor articulation
        particle particlefluid particlefluidboundary particlemeshfloor particlehffloor
        particlecloth particleinflatable particlediffuse particleconvex
        cloth clothbox clothconvex clothhang clothdamp clothdrape clothselfcollide clothcloth
        softbody softbodyselfcollide softbodycloth softbodystack softbodyattach"

pass=0; fail=0; missing=0

# P9.40: --strict exports PXG_REQUIRE_MAPPED=1, under which a scene that launched an
# unmapped kernel FAILS.  Without it the scene is scored as before, but the kernels
# it silently skipped are listed under its line, so the sweep's own output says
# which "ok" is an ok about the physics and which is an ok about the checks.
if [ "${1:-}" = "--strict" ]; then export PXG_REQUIRE_MAPPED=1; shift; fi

run_one() {
    label="$1"; shift
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

echo "=== scenes: $pass ok, $fail fail, $missing missing (of $((pass+fail+missing))) ==="

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

    # P9.41: three scenes that MEASURE the physics P9.40's guard showed to be
    # missing -- the ones no scored scene can see because none of them lets a
    # dynamic rigid depend on a deformable, or reads an articulation's joint state
    # back.  Each names the kernels it waits on and prints its number now, so the
    # tranche that maps them has a before-figure to beat.  Each moves into SCENES
    # the day it passes.  Not scored.
    echo
    echo "--- clothnet          (waits on accumulateDeltaVRigidFirst/SecondLaunch: the FEM-to-rigid impulse)"
    env PXG_SCENE=clothnet PXG_GPU_STEP=1 ./gpu-sdk-step 2>&1 \
        | grep -E '^  clothnet:|^FAIL:|^\[ *(OK|FAIL) *\] gpu-sdk-step|\[pxg-unmapped\]' | sed 's/^/    /'
    echo "--- particlepush      (waits on ps_solvePCOutputRigidDeltaVTGSLaunch and the same two accumulate kernels)"
    env PXG_SCENE=particlepush PXG_GPU_STEP=1 ./gpu-sdk-step 2>&1 \
        | grep -E '^  particlepush:|^FAIL:|^\[ *(OK|FAIL) *\] gpu-sdk-step|\[pxg-unmapped\]' | sed 's/^/    /'
    echo "--- articulationjoint (waits on updateBodiesLaunch1T for the joint state, updateBodiesLaunch_Part3 for sleep)"
    env PXG_SCENE=articulationjoint PXG_GPU_STEP=1 ./gpu-sdk-step 2>&1 \
        | grep -E '^  articulationjoint:|^FAIL:|^\[ *(OK|FAIL) *\] gpu-sdk-step|\[pxg-unmapped\]' | sed 's/^/    /'
fi

[ "$fail" -eq 0 ] && [ "$missing" -eq 0 ]
