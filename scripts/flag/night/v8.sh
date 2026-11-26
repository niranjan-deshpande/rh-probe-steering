#!/bin/bash
# Pod, night of 25/26 November: choose the steering vector by steering, then its strength curve, a difficulty curve on
# new save_game games, and Goodfire's direction at lower norms. Rules are in v8_select.py. Stops at CUTOFF (7:00 am ET).
cd /workspace/rlp; S=scripts/flag/night; source $S/env.sh
CUTOFF=1795694400
T=cache/flag/night/steer_table_v8.pt
C="--api raw --gen-budget 30000"
row() { python $S/v8_select.py row "$1"; }
left() { echo $(( (CUTOFF - $(date +%s)) / 60 )); }                 # minutes to the cutoff
lim() { local l; l=$(left); [ "$l" -lt "$1" ] && echo "$l" || echo "$1"; }
st() { local name=$1 max=$2; shift 2; local m; m=$(lim "$max"); [ "$m" -ge 3 ] || { echo "SKIP $name (cutoff)"; return 0; }; bash $S/stage.sh "$name" "$m" "$@"; }
say() { echo "$(date -u +%FT%TZ) $*" >> /workspace/logs/v8.log; }

RLP_STEER_ALLOW_PREFIX_CACHE=1 MAX_MODEL_LEN=65536 EXTRA_ARGS="--enable-prefix-caching" bash $S/serve.sh start steer $T || { say V8_SERVER_FAILED; echo V8_END >> /workspace/logs/v8.log; exit 1; }
STEER_TABLE=$T bash $S/stage.sh v8_cachecheck 15 python $S/v3_cachecheck.py || { say V8_CACHECHECK_FAILED; echo V8_END >> /workspace/logs/v8.log; exit 1; }

# Stage 0: candidates B-E at norm 20 on the validation set (A exists from 25 November).
say STAGE0
for k in B C D E; do
  st v8_sel_$k 80 python -m rlp.flag.mt run --run v8_sel_$k --vid $(row ${k}_sub_n20) --families save_game --tasks 8 --hack-eps 2 --benign-eps 2 $C --conc 3 --active 3 &
done; wait
W=$(python $S/v8_select.py vec 2>> /workspace/logs/v8_select.log | tee -a /workspace/logs/v8_select.log | grep -x -E "WINNER [A-E]" | cut -d' ' -f2); W=${W:-A}
say "WINNER $W"
r20() { [ "$W" = A ] && row sprobe_sub || row ${W}_sub_n20; }

# Stage 1: strength curve of the winner (hack variants of both families, benign save_game), and for a new winner its
# norm-20 release_gate episodes.
say STAGE1
for n in 5 10 15; do
  ( st v8_${W}_n${n}_hack 150 python -m rlp.flag.mt run --run v8_${W}_n$n --vid $(row ${W}_sub_n$n) --families save_game,release_gate --tasks 8 --hack-eps 2 --benign-eps 0 $C --conc 3 --active 3
    st v8_${W}_n${n}_benign 60 python -m rlp.flag.mt run --run v8_${W}_n$n --vid $(row ${W}_sub_n$n) --families save_game --tasks 8 --hack-eps 0 --benign-eps 2 $C --conc 3 --active 3 ) &
done
[ "$W" = A ] || st v8_${W}_n20_rg 150 python -m rlp.flag.mt run --run v8_sel_$W --vid $(r20) --families release_gate --tasks 8 --hack-eps 2 --benign-eps 0 $C --conc 3 --active 3 &
wait
N=$(python $S/v8_select.py norm --vec $W 2>> /workspace/logs/v8_select.log | tee -a /workspace/logs/v8_select.log | grep -x -E "NORM [0-9]+" | cut -d' ' -f2); N=${N:-20}
say "NORM $N"
[ "$N" = 20 ] && RS=$(r20) || RS=$(row ${W}_sub_n$N)

# Stage 2: difficulty curve on the new games 8-15, benign variant, unsteered against the winner at the chosen norm.
say STAGE2
for f in 60 80 90 100; do for arm in u s; do
  [ $arm = u ] && V=0 || V=$RS
  echo "bash $S/stage.sh v8_diff_${arm}_$f $(lim 70) python -m rlp.flag.mt run --run v8_diff_${arm}_$f --vid $V --cfg @$S/sg_$f.json --families save_game --task-ids 8 9 10 11 12 13 14 15 --hack-eps 0 --benign-eps 2 $C --conc 3 --active 3"
done; done > /workspace/logs/v8_stage2.cmds
[ "$(left)" -ge 5 ] && xargs -P 4 -I{} bash -c "{}" < /workspace/logs/v8_stage2.cmds

# Stage 3: Goodfire's direction at norms 5 and 10, hack variants.
say STAGE3
for n in 5 10; do
  st v8_gf_n$n 180 python -m rlp.flag.mt run --run v8_gf_n$n --vid $(row gf_sub_n$n) --families save_game,release_gate --tasks 8 --hack-eps 2 --benign-eps 0 $C --conc 6 --active 6 &
done; wait
for d in cache/flag/mt/v8_*; do python -m rlp.flag.mt stats --run $(basename $d); done > /workspace/logs/v8_stats.txt 2>&1
say DONE
echo V8_END >> /workspace/logs/v8.log
