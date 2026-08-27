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

# The 31 named scenes.  The settling bag below makes 32 -- the figure
# RUNTIME_STATUS.md's header carries.  clothbox joined in P9.15a: `cloth` is a
# free fall and stays green whether or not the cloth lands, so the landing needed
# a scene of its own.  particlediffuse joined in P9.13d, when
# ps_diffuseParticleCreate made it pass; it had been a diagnostic since P9.13a.
# particleconvex joined in P9.14c and had been a pending-work entry since P9.14a:
# it named the two convex particle kernels, P9.14b landed the fluid one and
# P9.14c the diffuse one, and it is scored on BOTH.  The pending-work list at the
# bottom is empty as a result, and stays in the script for the next scene that
# needs it.
SCENES="box stack plane sphere capsule capsulebox convex joint meshfloor articulation
        particle particlefluid particlefluidboundary particlemeshfloor particlehffloor
        particlecloth particleinflatable particlediffuse particleconvex
        cloth clothbox clothhang clothdamp clothdrape clothselfcollide clothcloth
        softbody softbodyselfcollide softbodycloth softbodystack softbodyattach"

pass=0; fail=0; missing=0

run_one() {
    label="$1"; shift
    out=$(env "$@" PXG_GPU_STEP=1 ./gpu-sdk-step 2>&1)
    verdict=$(printf '%s\n' "$out" | grep -E '^\[ *(OK|FAIL) *\] gpu-sdk-step' | tail -1)
    case "$verdict" in
        "[  OK  ]"*)  echo "ok       $label"; pass=$((pass+1)) ;;
        "[ FAIL ]"*)  echo "FAIL     $label"
                      printf '%s\n' "$out" | grep -E '^FAIL:|^\[ FAIL \]' | head -4 | sed 's/^/           /'
                      fail=$((fail+1)) ;;
        *)            echo "MISSING  $label   (no verdict line -- mistyped scene or a crash)"
                      printf '%s\n' "$out" | tail -3 | sed 's/^/           /'
                      missing=$((missing+1)) ;;
    esac
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
    echo "    (empty since P9.14c -- particleconvex, the only entry this list has"
    echo "     ever held, is scored above)"
    for sc in ; do
        echo "--- $sc"
        env PXG_SCENE=$sc PXG_GPU_STEP=1 ./gpu-sdk-step 2>&1 \
            | grep -E '^  particle |^  cooked |^  \[ diag \]|^FAIL:|^\[ *(OK|FAIL) *\] gpu-sdk-step' \
            | sed 's/^/    /'
        env PXG_SCENE=$sc PXG_GPU_STEP=1 ./gpu-sdk-step 2>&1 \
            | grep -E '\] GPU .*fail to launch' | sed 's/.*GPU //' | sort | uniq -c \
            | sed 's/^/    /'
    done
fi

[ "$fail" -eq 0 ] && [ "$missing" -eq 0 ]
