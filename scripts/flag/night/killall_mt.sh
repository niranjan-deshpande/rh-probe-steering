#!/bin/bash
# Stop every multi-turn driver and runner on this pod (by PID). Finished episodes are already on disk.
for p in $(ps -eo pid,args | grep -E "scripts/flag/night/(m1|m1b|m2_r[0-9]b?|a_ib|b_ib2)\.sh|stage\.sh (m1|ib_)|rlp\.flag\.mt run" | grep -v grep | awk '{print $1}'); do kill $p 2>/dev/null; done
sleep 4
for p in $(ps -eo pid,args | grep -E "rlp\.flag\.mt run" | grep -v grep | awk '{print $1}'); do kill -9 $p 2>/dev/null; done
rm -f /workspace/running/* 2>/dev/null; echo KILLED_ALL
