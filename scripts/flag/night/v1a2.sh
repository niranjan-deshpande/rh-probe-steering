#!/bin/bash
# Pod A: rerun the self-test with the corrected leak criterion, then the pilot (server is already up with the table).
cd /workspace/rlp; S=scripts/flag/night; source $S/env.sh
T=cache/flag/night/steer_table.pt
bash $S/stage.sh selftest3 20 python -m rlp.flag.steer_selftest live --table $T --ref /workspace/ref_fp.json \
  || bash $S/stage.sh selftest3_soft 20 python -m rlp.flag.steer_selftest live --table $T --ref /workspace/ref_fp.json --proj-soft \
  || { echo V1A_SELFTEST_FAILED; exit 1; }
bash $S/stage.sh pilot 40 python -m rlp.flag.gen_multi run --table $T --prefix np_ --conds base --label \
  --groups '[{"beh":"puzzle_leak","cond":"tempt","n":8,"samples":4},{"beh":"recon_total","cond":"tempt","n":8,"samples":4}]'
echo V1A_END
