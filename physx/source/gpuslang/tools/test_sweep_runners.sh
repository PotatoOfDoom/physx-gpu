#!/bin/sh
# test_sweep_runners.sh -- hermetic end-to-end test of the four mutation-sweep runners
# (P9.97).  Registered as ctest gpuslang.sweepRunners; no GPU, no slangc, ~90 s.
#
# WHY.  tools/test_sweep_timeout.sh pins the POLICY (sweep_timeout.sh, derive_timeout),
# but none of the runner behaviour P9.97 changed is visible to it: a runner that re-ran
# the suite on a refusal, bounded every mutant by 1 s, let a crash overwrite a hang,
# dropped the process-group kill or re-ran a refused host baseline still left it green
# (P9.97 review: seven such runner mutants, all survivors).  So this builds a MIRROR of
# the tree in a temp dir and drives COPIES of the real runners against it:
#
#   $M/physx/source/gpuslang/tools/   the four runners + sweep_timeout.sh (copied from
#                                     this directory, so a mutant of a runner is what runs)
#   $M/physx/source/gpuslang/shaders/fx.slang   a TEXT file ("CLEAN"); a mutant rewrites it
#   $M/build-gpuslang                 a Unix-Makefiles CMake project: shaders/fx.spv is a
#                                     copy of fx.slang (target gpuslang-shaders, so the
#                                     SPV_ONLY build.make rule exists), plus five ctest
#                                     tests mir.h1..h5 for the host runner
#   $M/build-gpuslang/gpuslang-tests  a FAKE suite binary and
#   $M/build-sdk/gpu-sdk-step         a FAKE scene binary: both read the built fx.spv --
#                                     CLEAN -> green after MIRROR_SLOW s; FAIL -> red;
#                                     HANG -> sleep; HANGTERM -> sleep ignoring SIGTERM;
#                                     CRASH -> rc 139 (also the first MIRROR_CRASH_FIRST
#                                     invocations: a platform fault); "<filter>=<mode>" lines pick a
#                                     mode per suite filter; any other text -> green after
#                                     MIRROR_MUTANT_SLOW s (an equivalent but slower mutant)
#
# and checks literal oracles: verdict lines, the printed bound, the exit code, the
# NUMBER OF INVOCATIONS of each fake (that is how a second suite run, a fault retry or a
# re-run host baseline becomes visible), and that the mirror source and .spv are
# restored after every row.  Timing is scaled to seconds; bounds are 1-9 s.
#
# SR_ROWS="R1 R2 ..." runs a subset (default: every row).  Exit 0 iff every check passed.
set -u
HERE="$(cd "$(dirname "$0")" && pwd)"
W=$(mktemp -d "${TMPDIR:-/tmp}/sweepRunners.XXXXXX") || exit 2
trap 'rm -rf "$W"' EXIT
M=$W/mirror
G=$M/physx/source/gpuslang
mkdir -p $G/tools $G/shaders $G/src $M/proj $M/build-sdk
for f in run_mutation_sweep.sh run_scene_mutation_sweep.sh run_host_mutation_sweep.py sweep_common.sh \
         sweep_timeout.sh; do
    [ -f "$HERE/$f" ] && cp "$HERE/$f" $G/tools/
done
printf 'CLEAN\n' > $G/shaders/fx.slang
printf 'CLEAN\n' > $G/src/host.txt
cp $G/shaders/fx.slang $W/clean.txt

# --- the fake binaries (one script; its name picks the output format)
cat > $W/fake.sh <<EOF
#!/bin/sh
me=\$(basename "\$0")
echo "\$*" >> $M/inv_\$me.log
n=\$(wc -l < $M/inv_\$me.log)
spv=\${PXG_SHADER_DIR:-$M/build-gpuslang/shaders}/fx.spv
mode=\$(sed -n "s/^\${1:-x}=//p" "\$spv" | head -1)
[ -n "\$mode" ] || mode=\$(head -1 "\$spv")
[ -n "\${MIRROR_FORCE:-}" ] && mode=\$MIRROR_FORCE
if [ -n "\${MIRROR_LATE_HANG:-}" ] && [ "\$n" -ge "\$MIRROR_LATE_HANG" ]; then mode=HANG; fi
if [ -n "\${MIRROR_CRASH_FIRST:-}" ] && [ "\$n" -le "\$MIRROR_CRASH_FIRST" ]; then mode=CRASH; fi
ok() { if [ "\$me" = gpu-sdk-step ]; then echo "[  OK  ] gpu-sdk-step: scene '\${PXG_SCENE:-}' (fake)"
       else echo; echo "1 test(s) ran, 0 failed, 0 check failure(s)"; fi; exit 0; }
case "\$mode" in
    HANG)     exec sleep 600 ;;
    HANGTERM) trap '' TERM; exec sleep 600 ;;
    CRASH)    exit 139 ;;
    FAIL)     if [ "\$me" = gpu-sdk-step ]; then echo "FAIL: fake"; echo "[ FAIL ] gpu-sdk-step: scene (fake)"
              else echo; echo "1 test(s) ran, 1 failed, 1 check failure(s)"; fi; exit 1 ;;
    TENFAIL)  echo; echo "10 test(s) ran, 10 failed, 0 check failure(s)"; exit 1 ;;
    CLEAN)    pad=\${MIRROR_SLOW:-0} ;;
    *)        pad=\${MIRROR_MUTANT_SLOW:-\${MIRROR_SLOW:-0}} ;;
esac
[ "\$pad" = 0 ] || sleep "\$pad"
ok
EOF
cat > $W/hostcase.sh <<EOF
#!/bin/sh
echo "\$1" >> $M/inv_host.log
[ "\${HOST_FORCE:-}" = FAIL ] && exit 1
case "\$(head -1 $G/src/host.txt)" in
    FAIL) exit 1 ;;
    HANG) exec sleep 600 ;;
esac
sleep "\${HOST_SLOW:-0}"
EOF

printf '#!/bin/sh\necho built >> %s/hostbuild.log\n' "$M" > $W/hostbuild.sh

# --- the mirror build
cat > $M/proj/CMakeLists.txt <<EOF
cmake_minimum_required(VERSION 3.16)
project(sweepRunnersMirror NONE)
add_custom_command(OUTPUT \${CMAKE_BINARY_DIR}/shaders/fx.spv
    COMMAND \${CMAKE_COMMAND} -E make_directory \${CMAKE_BINARY_DIR}/shaders
    COMMAND \${CMAKE_COMMAND} -E copy $G/shaders/fx.slang \${CMAKE_BINARY_DIR}/shaders/fx.spv
    DEPENDS $G/shaders/fx.slang)
add_custom_target(gpuslang-shaders ALL DEPENDS \${CMAKE_BINARY_DIR}/shaders/fx.spv)
add_custom_target(hostnoop COMMAND sh $W/hostbuild.sh)
enable_testing()
foreach(i 1 2 3 4 5)
    add_test(NAME mir.h\${i} COMMAND sh $W/hostcase.sh \${i})
endforeach()
EOF
if ! { cmake -S $M/proj -B $M/build-gpuslang -G "Unix Makefiles" && cmake --build $M/build-gpuslang; } >$W/cmake.log 2>&1; then
    echo "mirror build failed:"; tail -20 $W/cmake.log; exit 2
fi
cp $W/fake.sh $M/build-gpuslang/gpuslang-tests
cp $W/fake.sh $M/build-sdk/gpu-sdk-step
chmod +x $M/build-gpuslang/gpuslang-tests $M/build-sdk/gpu-sdk-step
( cd $M && git init -q . && git add physx && git -c user.name=t -c user.email=t@t commit -qm mirror ) \
    || { echo "git init of the mirror failed"; exit 2; }

fails=0
ROWS=${SR_ROWS:-"R1 R2 R3 R4 R6 R7 S1 S2 S3 C1 C2 C3 C4 C5 H1 H2 H3"}
want() {   # want <description> <shell condition> <row>
    if eval "$2"; then echo "ok    $1"
    else echo "FAIL  $1   (rc=$RC inv=$INV)"; sed 's/^/        | /' $W/$3.log | tail -12; fails=$((fails+1)); fi
}
clean_env() {
    for v in MUTATION_SWEEP_TIMEOUT MUTATION_SWEEP_TIMEOUT_FACTOR MUTATION_SWEEP_TIMEOUT_FLOOR \
             MUTATION_SWEEP_BASELINE_TIMEOUT MUTATION_SWEEP_OUTER_SLACK MUTATION_SWEEP_SPV_ONLY \
             SCENE_SWEEP_TIMEOUT SCENE_SWEEP_TIMEOUT_FACTOR SCENE_SWEEP_TIMEOUT_FLOOR \
             SCENE_SWEEP_BASELINE_TIMEOUT SWEEP_KILL_GRACE SWEEP_REPO TMO MIRROR_SLOW \
             MIRROR_MUTANT_SLOW MIRROR_FORCE MIRROR_LATE_HANG MIRROR_CRASH_FIRST HOST_FORCE HOST_SLOW \
             MUTATION_SWEEP_FAULT_RETRIES PXG_BUILD_DIR PXG_BUILD_TARGET PXG_SHADER_DIR; do unset $v 2>/dev/null; done
}
restored() {   # the mirror source and its built .spv are CLEAN again
    cmp -s $G/shaders/fx.slang $W/clean.txt && cmp -s $M/build-gpuslang/shaders/fx.spv $W/clean.txt
}
has() { echo " $ROWS " | grep -q " $1 "; }
tsv() { printf '%s\n' "$@" > $W/m.tsv; }
T='	'

# ---- fixture runner (run_mutation_sweep.sh) ------------------------------------------
fx() {   # fx <row> <suite> <env...> -- runs the fixture runner on the mirror
    row=$1; suite=$2; shift 2; rm -f $M/inv_gpuslang-tests.log
    ( clean_env; env MUTATION_SWEEP_SPV_ONLY=1 SWEEP_KILL_GRACE=1 "$@" \
        sh $G/tools/run_mutation_sweep.sh fx "$suite" $W/m.tsv ) > $W/$row.log 2>&1
    RC=$?; INV=$(cat $M/inv_gpuslang-tests.log 2>/dev/null | wc -l)
    want "$row: source and .spv restored" 'restored' $row
}
if has R1; then   # a FIXED bound below the clean run: refuse, and run the suite ONCE (E3)
    tsv "m-fail${T}CLEAN${T}FAIL"
    fx R1 fxsuite MUTATION_SWEEP_TIMEOUT=1 MIRROR_SLOW=2
    want "R1: refuses (rc 1)" '[ $RC -eq 1 ] && grep -q "BASELINE IS NOT GREEN" $W/R1.log' R1
    want "R1: names the knob in force" 'grep -q "exceeded MUTATION_SWEEP_TIMEOUT=1s" $W/R1.log' R1
    want "R1: the clean suite ran ONCE, not again for the tail" '[ $INV -eq 1 ]' R1
fi
if has R2; then   # DERIVED: kill, hang, SIGTERM-ignoring hang (137 -> hang, not a retried fault)
    tsv "m-fail${T}CLEAN${T}FAIL" "m-hang${T}CLEAN${T}HANG" "m-hangterm${T}CLEAN${T}HANGTERM"
    fx R2 fxsuite MUTATION_SWEEP_TIMEOUT_FLOOR=1 MIRROR_SLOW=2
    want "R2: rc 0" '[ $RC -eq 0 ]' R2
    want "R2: the derived bound line" \
         'grep -Eq "^baseline green in [23]s; every run below is bounded by [69]s \(max\(floor 1s, 3 x baseline [23]s\)\)$" $W/R2.log' R2
    want "R2: m-fail KILLED" 'grep -Eq "^KILLED +m-fail$" $W/R2.log' R2
    want "R2: m-hang KILLED-HANG with bound and baseline" \
         'grep -Eq "^KILLED-HANG +m-hang +\(exceeded [69]s; baseline [23]s\)$" $W/R2.log' R2
    want "R2: m-hangterm (ignores SIGTERM) KILLED-HANG, not KILLED-FAULT" \
         'grep -Eq "^KILLED-HANG +m-hangterm " $W/R2.log' R2
    want "R2: 5 suite runs (baseline, 3 mutants once each, closing baseline)" '[ $INV -eq 5 ]' R2
    want "R2: closing baseline green" 'grep -q "^baseline green again in [0-9]*s" $W/R2.log' R2
fi
if has R3; then   # DERIVED: an equivalent mutant 2x slower than the baseline survives
    tsv "m-equiv${T}CLEAN${T}EQUIV"
    fx R3 fxsuite MUTATION_SWEEP_TIMEOUT_FLOOR=1 MIRROR_SLOW=2 MIRROR_MUTANT_SLOW=4
    want "R3: slower equivalent mutant SURVIVED (rc 1 forces the diagnosis)" \
         '[ $RC -eq 1 ] && grep -Eq "^SURVIVED +m-equiv$" $W/R3.log' R3
fi
if has R4; then   # FIXED with < 2x headroom: the same mutant reads as a hang, and that is warned about
    tsv "m-equiv${T}CLEAN${T}EQUIV"
    fx R4 fxsuite MUTATION_SWEEP_TIMEOUT=3 MIRROR_SLOW=2 MIRROR_MUTANT_SLOW=4
    want "R4: warns that the baseline used more than half of the fixed bound" \
         'grep -q "more than half of MUTATION_SWEEP_TIMEOUT=3s" $W/R4.log' R4
    want "R4: (expected under that bound) KILLED-HANG" 'grep -Eq "^KILLED-HANG +m-equiv " $W/R4.log' R4
fi
if has R6; then   # suite a,b: a hangs, b crashes -- the hang is sticky, no fault retries
    tsv "m-mixed${T}CLEAN${T}a=HANG\\nb=CRASH"
    fx R6 a,b MUTATION_SWEEP_TIMEOUT_FLOOR=3
    want "R6: KILLED-HANG, not KILLED-FAULT" 'grep -Eq "^KILLED-HANG +m-mixed " $W/R6.log' R6
    want "R6: 6 runs (2 baseline, 2 mutant -- no retry, 2 closing)" '[ $INV -eq 6 ]' R6
fi
if has R7; then   # the CLOSING baseline times out: its own message, no knob advice
    tsv "m-fail${T}CLEAN${T}FAIL"
    fx R7 fxsuite MUTATION_SWEEP_TIMEOUT_FLOOR=3 MIRROR_SLOW=1 MIRROR_LATE_HANG=3
    want "R7: rc 1 and the restored-suite message" \
         '[ $RC -eq 1 ] && grep -q "the RESTORED suite did not finish inside" $W/R7.log' R7
    want "R7: no 'raise it' advice for a bound that is no knob" '! grep -q "raise it" $W/R7.log' R7
fi

# ---- scene runner (run_scene_mutation_sweep.sh) ---------------------------------------
sc() {
    row=$1; shift; rm -f $M/inv_gpu-sdk-step.log
    ( clean_env; env SWEEP_KILL_GRACE=1 "$@" sh $G/tools/run_scene_mutation_sweep.sh fx fakescene $W/m.tsv ) \
        > $W/$row.log 2>&1
    RC=$?; INV=$(cat $M/inv_gpu-sdk-step.log 2>/dev/null | wc -l)
    want "$row: source and .spv restored" 'restored' $row
}
if has S1; then
    tsv "s-fail${T}CLEAN${T}FAIL"
    sc S1 SCENE_SWEEP_TIMEOUT=1 MIRROR_SLOW=2
    want "S1: fixed bound below the clean scene: refuses" \
         '[ $RC -eq 1 ] && grep -q "CLEAN scene exceeded SCENE_SWEEP_TIMEOUT=1s" $W/S1.log && [ $INV -eq 1 ]' S1
fi
if has S2; then
    tsv "s-fail${T}CLEAN${T}FAIL" "s-hang${T}CLEAN${T}HANG"
    sc S2 SCENE_SWEEP_TIMEOUT_FLOOR=1 MIRROR_SLOW=2
    want "S2: rc 0" '[ $RC -eq 0 ]' S2
    want "S2: the derived bound line" \
         'grep -Eq "^baseline green in [23]s; every run below is bounded by [69]s \(max\(floor 1s, 3 x baseline [23]s\)\)$" $W/S2.log' S2
    want "S2: s-fail KILLED" 'grep -Eq "^  s-fail +KILLED$" $W/S2.log' S2
    want "S2: s-hang KILLED-HANG with bound and baseline" \
         'grep -Eq "^  s-hang +KILLED-HANG \(exceeded [69]s; baseline [23]s\)$" $W/S2.log' S2
fi
if has S3; then
    tsv "s-fail${T}CLEAN${T}FAIL"
    sc S3 SCENE_SWEEP_TIMEOUT_FLOOR=3 MIRROR_SLOW=1 MIRROR_LATE_HANG=3
    want "S3: a timed-out closing scene is reported as such, not as 'left something behind'" \
         '[ $RC -eq 1 ] && grep -q "RESTORED scene did not finish inside" $W/S3.log && ! grep -q "left something behind" $W/S3.log' S3
fi

# ---- sweep_common.sh (a bash library sourced by hand-written drivers) -----------------
cat > $W/common_driver.sh <<EOF
SH=$G/shaders/fx.slang
SPV=$M/build-gpuslang/shaders/fx.spv
TEST=fxsuite
. $G/tools/sweep_common.sh
run_mutant c-kill "CLEAN" "FAIL"
run_mutant c-hang "CLEAN" "HANG"
sweep_finish
EOF
cm() {
    row=$1; shift; rm -f $M/inv_gpuslang-tests.log
    ( clean_env; env SWEEP_KILL_GRACE=1 "$@" bash $W/common_driver.sh ) > $W/$row.log 2>&1
    RC=$?; INV=$(cat $M/inv_gpuslang-tests.log 2>/dev/null | wc -l)
    want "$row: source and .spv restored" 'restored' $row
}
if has C1; then   # TMO below the clean run: refuse BEFORE any mutant (was: every mutant "killed (HANG)")
    cm C1 TMO=1 MIRROR_SLOW=2
    want "C1: REFUSING (rc 3), because the clean run exceeded the bound" \
         '[ $RC -eq 3 ] && grep -q "REFUSING TO RUN: the CLEAN fxsuite exceeded MUTATION_SWEEP_TIMEOUT=1s" $W/C1.log' C1
    want "C1: no mutant ran" '[ $INV -eq 1 ] && ! grep -q "^killed" $W/C1.log' C1
fi
if has C2; then   # TMO unset: derived (was: set -u in the $( ), every mutant "killed" unrun)
    cm C2 MUTATION_SWEEP_TIMEOUT_FLOOR=1 MIRROR_SLOW=2
    want "C2: derived bound line" \
         'grep -Eq "^baseline green in [23]s; every run below is bounded by [69]s" $W/C2.log' C2
    want "C2: c-kill killed" 'grep -Eq "^killed +c-kill$" $W/C2.log' C2
    want "C2: c-hang killed as a HANG with bound and baseline" \
         'grep -Eq "^killed +c-hang \(HANG > [69]s; baseline [23]s\)$" $W/C2.log' C2
    want "C2: summary killed 2, 3 runs" 'grep -q "=== fxsuite: killed 2, survived 0, not-run 0" $W/C2.log && [ $INV -eq 3 ]' C2
fi
if has C3; then   # "10 failed" is not green (the bare "0 failed, 0 check failure" grep said it was)
    cm C3 MUTATION_SWEEP_TIMEOUT_FLOOR=1 MIRROR_FORCE=TENFAIL
    want "C3: a '10 failed' baseline is REFUSED (rc 3)" \
         '[ $RC -eq 3 ] && grep -q "REFUSING TO RUN: the CLEAN fxsuite is not green" $W/C3.log' C3
fi
if has C4; then   # ONE baseline crash (llvmpipe fault) is retried, not a refusal
    cm C4 MUTATION_SWEEP_TIMEOUT_FLOOR=1 MIRROR_CRASH_FIRST=1
    want "C4: baseline green after a retry, both mutants run (2 baseline + 2 mutant runs)" \
         '! grep -q REFUSING $W/C4.log && grep -q "^baseline green in" $W/C4.log && grep -q "=== fxsuite: killed 2," $W/C4.log && [ $INV -eq 4 ]' C4
fi
if has C5; then   # a baseline that crashes EVERY time is refused after 1 + MUTATION_SWEEP_FAULT_RETRIES runs
    cm C5 MUTATION_SWEEP_TIMEOUT_FLOOR=1 MUTATION_SWEEP_FAULT_RETRIES=2 MIRROR_FORCE=CRASH
    want "C5: REFUSING (rc 3) after 3 runs, no mutant ran" \
         '[ $RC -eq 3 ] && grep -q "is not green (rc=139, 3 run(s))" $W/C5.log && [ $INV -eq 3 ] && ! grep -q "^killed" $W/C5.log' C5
fi

# ---- host runner (run_host_mutation_sweep.py) -----------------------------------------
hs() {   # hs <row> <regex> <env...>
    row=$1; rx=$2; shift 2; rm -f $M/inv_host.log $M/hostbuild.log
    ( clean_env; env PXG_BUILD_TARGET=hostnoop "$@" python3 -B $G/tools/run_host_mutation_sweep.py \
        src/host.txt "$rx" $W/m.tsv ) > $W/$row.log 2>&1
    RC=$?; INV=$(cat $M/inv_host.log 2>/dev/null | wc -l)
    BUILDS=$(cat $M/hostbuild.log 2>/dev/null | wc -l)
    want "$row: host source restored" 'cmp -s $G/src/host.txt $W/clean.txt' $row
}
if has H1; then   # a regex that matches nothing: refused before any build
    tsv "h-fail${T}CLEAN${T}FAIL"
    hs H1 'mir\.nomatch$'
    want "H1: refuses (rc 1) without building" \
         '[ $RC -eq 1 ] && grep -q "matched NO test" $W/H1.log && [ $BUILDS -eq 0 ] && [ $INV -eq 0 ]' H1
fi
if has H2; then   # FIXED 2 s per test, 5 tests of 1.2 s: the outer bound is 5 x 2 + slack, not 2 x 2 + slack
    tsv "h-fail${T}CLEAN${T}FAIL"
    hs H2 'mir\.h[1-5]$' MUTATION_SWEEP_TIMEOUT=2 MUTATION_SWEEP_OUTER_SLACK=1 HOST_SLOW=1.2
    want "H2: baseline green under the N-tests outer bound (11 s)" \
         'grep -q "^baseline green in [0-9]*s; every ctest test below is bounded by 2s (fixed by MUTATION_SWEEP_TIMEOUT); 5 test(s), outer bound 11s" $W/H2.log' H2
    want "H2: h-fail KILLED, closing re-run green, rc 0" \
         '[ $RC -eq 0 ] && grep -Eq "^KILLED +h-fail$" $W/H2.log && grep -q "baseline rerun green" $W/H2.log' H2
fi
if has H3; then   # a RED baseline is reported and NOT re-run (nothing was ever mutated)
    tsv "h-fail${T}CLEAN${T}FAIL"
    hs H3 'mir\.h1$' HOST_FORCE=FAIL
    want "H3: BASELINE RED, rc 1" '[ $RC -eq 1 ] && grep -q "^BASELINE RED" $W/H3.log' H3
    want "H3: the test set ran exactly once, built once" '[ $INV -eq 1 ] && [ $BUILDS -eq 1 ]' H3
fi

echo
[ $fails -eq 0 ] && { echo "sweepRunners: all checks passed"; exit 0; }
echo "sweepRunners: $fails check(s) FAILED"; exit 1
