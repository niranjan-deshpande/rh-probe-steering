#!/bin/bash
# Stop the ImpossibleBench runners only (scripted families keep running).
for p in $(ps -eo pid,args | grep -E "m2_r[0-9]\.sh|stage\.sh ib_|rlp\.flag\.mt run --run ib_" | grep -v grep | awk '{print $1}'); do kill $p 2>/dev/null; done
sleep 3
for p in $(ps -eo pid,args | grep -E "rlp\.flag\.mt run --run ib_" | grep -v grep | awk '{print $1}'); do kill -9 $p 2>/dev/null; done
rm -f /workspace/failed/ib_* /workspace/running/ib_* 2>/dev/null; echo KILLED_IB
