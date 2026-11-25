#!/bin/bash
# Pod A, stage V1 part a: steering table, reference fingerprint, patched server, canary self-test, pilot of the two
# new single-turn environments. The lead reads the pilot counts before v1b.sh.
cd /workspace/rlp; S=scripts/flag/night; source $S/env.sh
T=cache/flag/night/steer_table.pt; mkdir -p cache/flag/night
LIN=""; grep -qx LINFIT_CHECK_PASS /workspace/logs/lingrad.log && LIN="--lin cache/flag/onp/linfit_vec.pt"
bash $S/stage.sh gfdirs 10 python -m rlp.flag.gf27 dirs || exit 1
bash $S/stage.sh table 10 python -m rlp.flag.steer_table --mode v1 --out $T --require gf noise $LIN || exit 1
bash $S/serve.sh start plain || exit 1
bash $S/stage.sh ref 10 python -m rlp.flag.steer_selftest ref --out /workspace/ref_fp.json
bash $S/serve.sh stop
bash $S/serve.sh start steer $T || { echo V1A_SERVER_FAILED; exit 1; }
bash $S/stage.sh selftest 15 python -m rlp.flag.steer_selftest live --table $T --ref /workspace/ref_fp.json \
  || bash $S/stage.sh selftest_soft 15 python -m rlp.flag.steer_selftest live --table $T --ref /workspace/ref_fp.json --proj-soft \
  || { echo V1A_SELFTEST_FAILED; exit 1; }
bash $S/stage.sh pilot 40 python -m rlp.flag.gen_multi run --table $T --prefix np_ --conds base --label \
  --groups '[{"beh":"puzzle_leak","cond":"tempt","n":8,"samples":4},{"beh":"recon_total","cond":"tempt","n":8,"samples":4}]'
echo V1A_END
