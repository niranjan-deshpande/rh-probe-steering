#!/bin/bash
# Stop the multi-turn drivers and runners on the pod (kill by PID; the runner resumes from checkpoints).
for p in $(ps -eo pid,args | grep -E "m1\.sh|m2_r1\.sh|stage\.sh (m1|ib_)|rlp\.flag\.mt run" | grep -v grep | awk '{print $1}'); do kill $p 2>/dev/null; done
sleep 3
for p in $(ps -eo pid,args | grep -E "rlp\.flag\.mt run" | grep -v grep | awk '{print $1}'); do kill -9 $p 2>/dev/null; done
rm -f /workspace/failed/m1 /workspace/failed/ib_r1_* /workspace/running/* 2>/dev/null
echo KILLED
