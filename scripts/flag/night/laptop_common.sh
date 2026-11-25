# Shared by the LAPTOP scripts (pod_ssh.sh, push.sh, laptop_sync.sh). Source it after setting HOST and PORT.
# The ssh command is kept as one string without spaces inside any argument, because the macOS rsync (2.6.9) splits
# its -e value on spaces: so the key and the known-hosts file must live at paths without spaces.
#   SSH_KEY          private key (default ~/.runpod/ssh/runpodctl-ssh-key)
#   NIGHT_KNOWN_HOSTS known-hosts file used only for the night's pods (default ~/.ssh/known_hosts_rlp_night). RunPod
#                    reuses address and port pairs, so if ssh refuses a pod with "host key has changed", remove
#                    that line from this file (ssh-keygen -R "[host]:port" -f <file>) and try again.
#   LAPTOP_TEST_REMOTE  a local directory that stands in for the pod's / (tests only; no ssh is used)
REPO="${NIGHT_LAPTOP_REPO:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)}"   # override: tests only
SSH_KEY="${SSH_KEY:-$HOME/.runpod/ssh/runpodctl-ssh-key}"
NIGHT_KNOWN_HOSTS="${NIGHT_KNOWN_HOSTS:-$HOME/.ssh/known_hosts_rlp_night}"
case "$SSH_KEY$NIGHT_KNOWN_HOSTS" in *" "*) echo "SSH_KEY and NIGHT_KNOWN_HOSTS must not contain spaces" >&2; exit 2 ;; esac
case "${PORT:-}" in ""|*[!0-9]*) echo "PORT must be a number, got '${PORT:-}'" >&2; exit 2 ;; esac
case "${HOST:-}" in ""|*[!A-Za-z0-9._:-]*) echo "bad POD_SSH_HOST '${HOST:-}'" >&2; exit 2 ;; esac
SSH_CMD="ssh -p $PORT -i $SSH_KEY -o StrictHostKeyChecking=accept-new -o UserKnownHostsFile=$NIGHT_KNOWN_HOSTS"
SSH_CMD="$SSH_CMD -o IdentitiesOnly=yes -o BatchMode=yes -o LogLevel=ERROR -o ServerAliveInterval=30 -o ConnectTimeout=20"
if [ -n "${LAPTOP_TEST_REMOTE:-}" ]; then
  REMOTE="$LAPTOP_TEST_REMOTE"
  RSYNC=(rsync)
  pod() { echo "(test mode, not run on a pod: $*)"; }
else
  REMOTE="root@$HOST:"
  RSYNC=(rsync -e "$SSH_CMD")
  pod() { $SSH_CMD "root@$HOST" "$@"; }
fi
# Never copied in either direction, whatever the include list says.
NEVER=(--exclude ".env" --exclude ".env.*" --exclude ".venv" --exclude "__pycache__" --exclude "*.pyc" --exclude ".DS_Store")
et_now() { TZ=America/New_York date "+%-I:%M %p ET" | tr 'APM' 'apm'; }
