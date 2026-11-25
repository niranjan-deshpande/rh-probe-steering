#!/bin/bash
# Pod A, stage H1 (transformers): fast-kernel check, Goodfire harvest, gradient pass, text scoring.
cd /workspace/rlp; S=scripts/flag/night
bash $S/stage.sh fla 12 python $S/fla_check.py
bash $S/stage.sh gf27 40 python -m rlp.flag.gf27 harvest
bash $S/stage.sh lingrad 75 python -m rlp.flag.lingrad --stage all
bash $S/stage.sh textscore 30 python -m rlp.flag.textscore
echo H1_END
