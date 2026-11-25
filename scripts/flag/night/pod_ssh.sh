#!/bin/bash
# Run a command on a pod (or open a shell) with the night's ssh options.
# Usage: pod_ssh.sh POD_SSH_HOST PORT [COMMAND...]
#   pod_ssh.sh 1.2.3.4 12345 'tail -n 1 /workspace/logs/setup.log | grep -x SETUP_OK'
#   pod_ssh.sh 1.2.3.4 12345 'setsid -f bash /workspace/rlp/scripts/flag/night/setup_pod.sh < /dev/null > /dev/null 2>&1'
# Long jobs must be started with setsid -f and all three streams redirected, or ssh hangs until the job ends.
[ $# -ge 2 ] || { echo "usage: pod_ssh.sh POD_SSH_HOST PORT [COMMAND...]" >&2; exit 2; }
HOST="$1"; PORT="$2"; shift 2
source "$(dirname "${BASH_SOURCE[0]}")/laptop_common.sh"
if [ $# -eq 0 ]; then exec $SSH_CMD "root@$HOST"; fi
pod "$@"
