#!/bin/sh
# test_sweep_timeout.sh -- self-test of the sweep time-bound policy (P9.97).
# Registered as ctest gpuslang.sweepTimeoutPolicy; no GPU, ~20 s.
#
# Part 0 is a timeout(1) CONTROL: a 2 s run under a 1 s bound must come back as 124
# through sweep_run_bounded, or nothing below can see a hang.  (It is not a negative
# control of the old policy -- the RUNNERS' behaviour, old and new, is pinned end to end
# by tools/test_sweep_runners.sh, ctest gpuslang.sweepRunners.)
set -u
HERE="$(cd "$(dirname "$0")" && pwd)"
. "$HERE/sweep_timeout.sh"
fails=0
ok()  { echo "ok    $1"; }
bad() { echo "FAIL  $1"; fails=$((fails+1)); }
eq()  { if [ "$2" = "$3" ]; then ok "$1 = $3"; else bad "$1: got '$2', want '$3'"; fi; }
GREEN='echo "1 test(s) ran, 0 failed, 0 check failure(s)"'
unset MUTATION_SWEEP_TIMEOUT MUTATION_SWEEP_TIMEOUT_FACTOR MUTATION_SWEEP_TIMEOUT_FLOOR \
      MUTATION_SWEEP_BASELINE_TIMEOUT SCENE_SWEEP_TIMEOUT SCENE_SWEEP_TIMEOUT_FACTOR \
      SCENE_SWEEP_TIMEOUT_FLOOR SCENE_SWEEP_BASELINE_TIMEOUT SWEEP_KILL_GRACE 2>/dev/null

# --- 0. timeout(1) control: a run longer than the bound comes back as 124
eq "default SWEEP_KILL_GRACE (fresh shell, unset before sourcing)" \
   "$(env -u SWEEP_KILL_GRACE sh -c '. "$1"; echo "$SWEEP_KILL_GRACE"' sh "$HERE/sweep_timeout.sh")" 10
SWEEP_KILL_GRACE=1
SWEEP_TMO=1; sweep_run_bounded sh -c "sleep 2; $GREEN" >/dev/null; eq "a 2 s suite under a 1 s bound" $? 124

# --- 1. derive: literal oracle (baseline, factor, floor) -> bound
eq "derive(0,3,300)"   "$(sweep_derive_timeout 0 3 300)"   300
eq "derive(100,3,300)" "$(sweep_derive_timeout 100 3 300)" 300
eq "derive(101,3,300)" "$(sweep_derive_timeout 101 3 300)" 303
eq "derive(435,3,300)" "$(sweep_derive_timeout 435 3 300)" 1305
eq "derive(614,3,300)" "$(sweep_derive_timeout 614 3 300)" 1842
eq "derive(300,3,900)" "$(sweep_derive_timeout 300 3 900)" 900
eq "derive(301,3,900)" "$(sweep_derive_timeout 301 3 900)" 903
eq "derive(100,4,300)" "$(sweep_derive_timeout 100 4 300)" 400
eq "derive(10,2,1)"    "$(sweep_derive_timeout 10 2 1)"    20

# --- 2. knob validation: each of these must be refused with rc 2
for kv in MUTATION_SWEEP_TIMEOUT=0 MUTATION_SWEEP_TIMEOUT=20m MUTATION_SWEEP_TIMEOUT=-5 \
          MUTATION_SWEEP_TIMEOUT=0300 MUTATION_SWEEP_TIMEOUT=08 \
          MUTATION_SWEEP_TIMEOUT_FACTOR=1 MUTATION_SWEEP_TIMEOUT_FLOOR=0 MUTATION_SWEEP_BASELINE_TIMEOUT=x; do
    ( export "$kv"; sweep_init MUTATION_SWEEP 300 2>/dev/null ); eq "reject $kv" $? 2
done
( sweep_init MUTATION_SWEEP 300 ); eq "accept defaults" $? 0

# --- 3. resolution: fixed keeps its value (and warns only under 2x headroom); unset derives
( MUTATION_SWEEP_TIMEOUT=1200; export MUTATION_SWEEP_TIMEOUT
  sweep_init MUTATION_SWEEP 300; echo "$SWEEP_TMO" > "${TMPDIR:-/tmp}/sto.$$"
  sweep_set_bound 435 2>"${TMPDIR:-/tmp}/ste.$$"; echo "$SWEEP_TMO" >> "${TMPDIR:-/tmp}/sto.$$" )
eq "fixed 1200: baseline bound, mutant bound" "$(tr '\n' ' ' < "${TMPDIR:-/tmp}/sto.$$")" "1200 1200 "
eq "fixed 1200 over a 435 s baseline: no warning" "$(wc -c < "${TMPDIR:-/tmp}/ste.$$" | tr -d ' ')" 0
( MUTATION_SWEEP_TIMEOUT=800; export MUTATION_SWEEP_TIMEOUT
  sweep_init MUTATION_SWEEP 300; sweep_set_bound 435 2>"${TMPDIR:-/tmp}/ste.$$" )
grep -q "more than half" "${TMPDIR:-/tmp}/ste.$$"; eq "fixed 800 over a 435 s baseline: warns" $? 0
( MUTATION_SWEEP_TIMEOUT=870; export MUTATION_SWEEP_TIMEOUT
  sweep_init MUTATION_SWEEP 300; sweep_set_bound 435 2>"${TMPDIR:-/tmp}/ste.$$" )
eq "fixed 870 over a 435 s baseline (exactly 2x): no warning" "$(wc -c < "${TMPDIR:-/tmp}/ste.$$" | tr -d ' ')" 0
( MUTATION_SWEEP_TIMEOUT=869; export MUTATION_SWEEP_TIMEOUT
  sweep_init MUTATION_SWEEP 300; sweep_set_bound 435 2>"${TMPDIR:-/tmp}/ste.$$" )
grep -q "more than half" "${TMPDIR:-/tmp}/ste.$$"; eq "fixed 869 over a 435 s baseline: warns" $? 0
( sweep_init MUTATION_SWEEP 300; echo "$SWEEP_TMO $SWEEP_TMO_KNOB" > "${TMPDIR:-/tmp}/sto.$$"
  sweep_set_bound 435; echo "$SWEEP_TMO" >> "${TMPDIR:-/tmp}/sto.$$" )
eq "derived: baseline bound, mutant bound" "$(tr '\n' ' ' < "${TMPDIR:-/tmp}/sto.$$")" \
   "3600 MUTATION_SWEEP_BASELINE_TIMEOUT 1305 "
eq "derived, fast fixture: the old 300 s" "$(sweep_init MUTATION_SWEEP 300; sweep_set_bound 10; echo $SWEEP_TMO)" 300
eq "derived, fast scene: the old 900 s"   "$(sweep_init SCENE_SWEEP 900; sweep_set_bound 10; echo $SWEEP_TMO)" 900
eq "scene prefix honours SCENE_SWEEP_TIMEOUT" \
   "$(SCENE_SWEEP_TIMEOUT=77; export SCENE_SWEEP_TIMEOUT; sweep_init SCENE_SWEEP 900; sweep_set_bound 10; echo $SWEEP_TMO)" 77
rm -f "${TMPDIR:-/tmp}/sto.$$" "${TMPDIR:-/tmp}/ste.$$"

# --- 4. classification by sweep_run_bounded
SWEEP_TMO=5
sweep_run_bounded true;                 eq "rc of a clean exit" $? 0
sweep_run_bounded sh -c 'exit 3';       eq "rc of a failing suite" $? 3
( sweep_run_bounded sh -c 'kill -KILL $$' ) 2>/dev/null; eq "early SIGKILL stays a fault" $? 137
SWEEP_TMO=1
( timeout -k 1 1 sh -c 'trap "" TERM; sleep 5'; exit $? ) 2>/dev/null; eq "control: raw timeout on a TERM-ignoring run" $? 137
s=$(date +%s)
( sweep_run_bounded sh -c 'trap "" TERM; sleep 5' ) 2>/dev/null; eq "TERM-ignoring hang is a hang" $? 124
[ $(( $(date +%s) - s )) -le 4 ]; eq "... and is ended by SIGKILL at bound + grace" $? 0

# --- 5. the P9.96 situation in miniature (2 s suite), under the DERIVED policy
( MUTATION_SWEEP_TIMEOUT_FLOOR=1; export MUTATION_SWEEP_TIMEOUT_FLOOR
  sweep_init MUTATION_SWEEP 300
  s=$(date +%s); sweep_run_bounded sh -c "sleep 2; $GREEN" >/dev/null; r=$?; b=$(( $(date +%s) - s ))
  [ $r -eq 0 ] || exit 11                                  # baseline fits the baseline bound
  sweep_set_bound "$b"
  [ "$SWEEP_TMO" -eq $((3 * b)) ] || exit 12               # 6..9 s
  sweep_run_bounded sh -c "sleep 3; $GREEN" >/dev/null || exit 13   # 1.5x slower: NOT a hang
  ( sweep_run_bounded sleep 60 ) 2>/dev/null; [ $? -eq 124 ] || exit 14    # a real hang still is one
  exit 0 ); eq "scaled P9.96 sweep under the derived bound" $? 0

# --- 6. the Python host runner carries the same table
PYOUT=$(python3 -B - "$HERE" <<'PY'
import sys; sys.path.insert(0, sys.argv[1])
import run_host_mutation_sweep as h
print(' '.join(str(h.derive_timeout(b, k, f)) for b, k, f in
      [(0,3,300),(100,3,300),(101,3,300),(435,3,300),(614,3,300),(300,3,900),(301,3,900),
       (100,4,300),(10,2,1)]))
PY
)
eq "python derive_timeout table" "$PYOUT" "300 300 303 1305 1842 900 903 400 20"
PYOUT=$(python3 -B - "$HERE" <<'PY'
import os, sys, time; sys.path.insert(0, sys.argv[1])
import run_host_mutation_sweep as h
r = []
for v in ('0', '0300', '20m', '-5', '7'):
    os.environ['STO_KNOB'] = v
    try:
        r.append(str(h.knob('STO_KNOB', None)))
    except SystemExit:
        r.append('refused')
print(' '.join(r))
PY
)
eq "python knob(): 0 0300 20m -5 7" "$PYOUT" "refused refused refused refused 7"
# The outer bound kills the whole PROCESS GROUP: a backgrounded grandchild must not
# survive it (killing only the shell left the test binary running).  The sleep length
# carries this shell's PID, so pgrep (machine-wide) cannot see a concurrent copy's sleeps.
PYOUT=$(python3 -B - "$HERE" "41.$$" <<'PY'
import os, sys, time; sys.path.insert(0, sys.argv[1])
import run_host_mutation_sweep as h
t0 = time.monotonic()
s = 'sleep ' + sys.argv[2]
rc, _ = h.run_bounded('%s & %s; wait' % (s, s), '.', dict(os.environ), 1)
print(rc, 'fast' if time.monotonic() - t0 < 3 else 'slow')
PY
)
eq "python run_bounded on a hang" "$PYOUT" "TIMEOUT fast"
sleep 0.2
pgrep -fx "sleep 41\\.$$" >/dev/null; eq "... and no process of its group survives" $? 1

# --- 7. every runner uses the helper; no fixed bound is left behind
for f in run_mutation_sweep.sh run_scene_mutation_sweep.sh sweep_common.sh; do
    grep -q 'sweep_timeout.sh' "$HERE/$f";  eq "$f sources sweep_timeout.sh" $? 0
    grep -Eq '(^|[ ;(&|])timeout( -k [^ ]+)? +"?\$' "$HERE/$f"
    eq "$f calls timeout(1) only through sweep_run_bounded" $? 1
done

echo
[ $fails -eq 0 ] && { echo "sweepTimeoutPolicy: all checks passed"; exit 0; }
echo "sweepTimeoutPolicy: $fails check(s) FAILED"; exit 1
