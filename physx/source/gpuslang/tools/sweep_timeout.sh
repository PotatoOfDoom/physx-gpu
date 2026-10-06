# sweep_timeout.sh -- the per-run time bound shared by the mutation-sweep runners (P9.97).
#
# Sourced (POSIX sh; also fine from bash) by run_mutation_sweep.sh (prefix MUTATION_SWEEP),
# run_scene_mutation_sweep.sh (prefix SCENE_SWEEP) and sweep_common.sh (prefix
# MUTATION_SWEEP).  run_host_mutation_sweep.py carries the same policy in Python
# (derive_timeout); tools/test_sweep_timeout.sh checks both against one literal table.
#
# WHY.  Up to P9.96 every runner bounded each suite run by a FIXED number (300 s for
# fixtures, 900 s for scenes).  That bound has two jobs -- end a mutant that never
# terminates (P9.10: t3-child-shift-12 spun 37 minutes) and still let every honest run
# finish -- and a fixed number does the second job only for suites far below it:
#   * P9.96 made the triTriCollisionBda suite ~435 s unmutated.  The default sweep then
#     refused at the baseline (correctly -- see baseline_ok) and, before refusing, ran
#     the suite a SECOND time just to print its tail, which timed out again: ten
#     minutes for a refusal, and RUNTIME_STATUS had to say "use 1200".
#   * a suite NEAR the bound is worse than one over it: the baseline fits, and a mutant
#     that is only slower (machine load -- this box runs several sweeps at once) is
#     reported KILLED-HANG.  That is a kill the fixture did not earn.
#
# POLICY.
#   <P>_TIMEOUT set    -> FIXED, exactly as before: every run, the baseline included, is
#                         bounded by it; a baseline that does not fit is refused.  New: a
#                         baseline that uses more than HALF of it gets a warning.
#   <P>_TIMEOUT unset  -> DERIVED: the baseline is bounded by <P>_BASELINE_TIMEOUT
#                         (default 3600 s) and timed; every later run is bounded by
#                             max(<P>_TIMEOUT_FLOOR, <P>_TIMEOUT_FACTOR x baseline)
#                         with the floor defaulting to the old fixed value, so a suite
#                         whose baseline is at most floor/factor gets exactly the old bound.
#   All values are whole seconds > 0 (timeout(1) reads 0 as "no bound" and accepts
#   suffixes like 20m that the arithmetic here cannot).  FACTOR must be >= 2.
#
# HANG = THE BOUND EXPIRED.  timeout exits 124 when SIGTERM ends the run, but 137 when
# the program ignored SIGTERM and the -k SIGKILL ended it -- and 137 is >= 128, which the
# runners read as a platform crash and RETRY, each retry costing a whole bound.  So 137
# is mapped back to 124 when the bound had really elapsed; a SIGKILL from anywhere else
# (the OOM killer) arrives early and stays a fault.

SWEEP_KILL_GRACE=${SWEEP_KILL_GRACE:-10}

# Digits only, no leading 0: that refuses 0 itself (timeout(1) reads it as "no bound")
# and 0300, which sh arithmetic reads as OCTAL (192; 08 is an error under dash).
sweep_uint() { case "$1" in ''|0*|*[!0-9]*) return 1 ;; esac; }

# sweep_init <PREFIX> <default-floor>
#   Validates the knobs; sets SWEEP_PREFIX, SWEEP_FIXED (explicit bound or empty),
#   SWEEP_FACTOR, SWEEP_FLOOR, and SWEEP_TMO / SWEEP_TMO_KNOB for the BASELINE run.
#   Returns 2 (with a message) on a bad value.
sweep_init() {
    SWEEP_PREFIX=$1
    eval "SWEEP_FIXED=\${${1}_TIMEOUT:-}"
    eval "SWEEP_FACTOR=\${${1}_TIMEOUT_FACTOR:-3}"
    eval "SWEEP_FLOOR=\${${1}_TIMEOUT_FLOOR:-$2}"
    eval "SWEEP_BASELINE_BOUND=\${${1}_BASELINE_TIMEOUT:-3600}"
    if [ -n "$SWEEP_FIXED" ] && ! sweep_uint "$SWEEP_FIXED"; then
        echo "!! ${1}_TIMEOUT='$SWEEP_FIXED' is not a whole number of seconds > 0" >&2; return 2
    fi
    for _st_kv in "${1}_TIMEOUT_FLOOR=$SWEEP_FLOOR" "${1}_BASELINE_TIMEOUT=$SWEEP_BASELINE_BOUND" \
                  "${1}_TIMEOUT_FACTOR=$SWEEP_FACTOR"; do
        sweep_uint "${_st_kv#*=}" || {
            echo "!! ${_st_kv%%=*}='${_st_kv#*=}' is not a whole number > 0" >&2; return 2; }
    done
    [ "$SWEEP_FACTOR" -ge 2 ] || {
        echo "!! ${1}_TIMEOUT_FACTOR=$SWEEP_FACTOR leaves no headroom over the baseline (need >= 2)" >&2
        return 2; }
    if [ -n "$SWEEP_FIXED" ]; then SWEEP_TMO=$SWEEP_FIXED; SWEEP_TMO_KNOB="${1}_TIMEOUT"
    else SWEEP_TMO=$SWEEP_BASELINE_BOUND; SWEEP_TMO_KNOB="${1}_BASELINE_TIMEOUT"; fi
    SWEEP_BASE_SECS=""
}

# sweep_derive_timeout <baseline-secs> <factor> <floor>  ->  echoes max(floor, factor*baseline)
sweep_derive_timeout() {
    _st_t=$(( $1 * $2 ))
    [ "$_st_t" -lt "$3" ] && _st_t=$3
    echo "$_st_t"
}

# sweep_set_bound <baseline-secs>
#   Called once, after a GREEN baseline.  Fixes SWEEP_TMO for every later run (mutants
#   and the closing baseline) and sets SWEEP_TMO_HOW for the log line.
sweep_set_bound() {
    SWEEP_BASE_SECS=$1
    if [ -n "$SWEEP_FIXED" ]; then SWEEP_TMO_KNOB="${SWEEP_PREFIX}_TIMEOUT"
    else SWEEP_TMO_KNOB="the derived bound"; fi
    if [ -n "$SWEEP_FIXED" ]; then
        SWEEP_TMO=$SWEEP_FIXED
        SWEEP_TMO_HOW="fixed by ${SWEEP_PREFIX}_TIMEOUT"
        if [ $(( 2 * $1 )) -gt "$SWEEP_FIXED" ]; then
            echo "!! warning: the clean run took ${1}s, more than half of ${SWEEP_PREFIX}_TIMEOUT=${SWEEP_FIXED}s;" >&2
            echo "   a mutant that is merely slower will read as KILLED-HANG.  Unset it to derive the bound." >&2
        fi
    else
        SWEEP_TMO=$(sweep_derive_timeout "$1" "$SWEEP_FACTOR" "$SWEEP_FLOOR")
        SWEEP_TMO_HOW="max(floor ${SWEEP_FLOOR}s, ${SWEEP_FACTOR} x baseline ${1}s)"
    fi
}

# sweep_run_bounded <cmd...>
#   Runs cmd under SWEEP_TMO.  Returns cmd's rc, or 124 whenever the bound expired.
#   Pass environment with `export` (in a subshell), not as a prefix assignment: a prefix
#   on a FUNCTION call is not portable in sh.
sweep_run_bounded() {
    _st_s=$(date +%s)
    timeout -k "$SWEEP_KILL_GRACE" "$SWEEP_TMO" "$@"
    _st_r=$?
    if [ "$_st_r" -eq 137 ] && [ $(( $(date +%s) - _st_s )) -ge "$SWEEP_TMO" ]; then _st_r=124; fi
    return "$_st_r"
}
