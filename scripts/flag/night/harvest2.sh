#!/bin/bash
# Usage: harvest2.sh START_EPOCH [st]   As harvest.sh, but kills the server hard and waits for the GPU to be free
# (serve.sh stop left the vLLM process alive on pod A, so the first harvest ran out of memory).
cd /workspace/rlp; S=scripts/flag/night; source $S/env.sh
while [ $(date +%s) -lt $1 ]; do sleep 20; done
bash $S/killall_mt.sh; bash $S/serve.sh stop
pkill -9 -f "vllm serve"; pkill -9 -f "multiprocessing.resource_tracker"
# the process that holds the GPU is titled VLLM::EngineCore, which the patterns above do not match
for q in $(nvidia-smi --query-compute-apps=pid --format=csv,noheader); do kill -9 $q; done
for i in $(seq 60); do m=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | head -1); [ "$m" -lt 2000 ] && break; sleep 5; pkill -9 -f "vllm"; done
echo "gpu memory before harvest: $m MiB"
rm -f /workspace/failed/h2_mt /workspace/failed/h2_st
for r in $(ls cache/flag/mt | grep -v smoke); do python -m rlp.flag.mt stats --run $r > /workspace/logs/stats_$r.txt 2>&1; done
RUNS=$(ls cache/flag/mt | grep -v smoke | tr "\n" " ")
bash $S/stage.sh h2_mt 100 python -m rlp.flag.mt_harvest mt --run $RUNS
[ "$2" = st ] && bash $S/stage.sh h2_st 20 python -m rlp.flag.mt_harvest st --names np_base n1_base --tag night
echo HARVEST2_END
