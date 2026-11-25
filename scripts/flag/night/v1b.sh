#!/bin/bash
# Pod A, stage V1 part b: the mixed steering run. Usage: v1b.sh [EXTRA_BEH:COND ...]  (new single-turn environments
# that passed the pilot). Queue order as in the spec: main set, fitting group, pairing with 24 November, story.
cd /workspace/rlp; S=scripts/flag/night; source $S/env.sh
T=cache/flag/night/steer_table.pt
EX=""; [ $# -gt 0 ] && EX="--extra-behs $*"
bash $S/stage.sh v1_main 200 python -m rlp.flag.gen_multi run --table $T --prefix n1_ --conds @main --preset main $EX --label
bash $S/stage.sh v1_fit 60 python -m rlp.flag.gen_multi run --table $T --prefix n1_ --conds @fitting --preset fitting --label
bash $S/stage.sh v1_pair 60 python -m rlp.flag.gen_multi run --table $T --prefix n3_ --conds @pairing --preset pairing --label
bash $S/stage.sh v1_story 20 python -m rlp.flag.gen_multi story --table $T --prefix n1_ --conds @story
echo V1B_END
