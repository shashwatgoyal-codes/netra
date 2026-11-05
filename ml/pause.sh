#!/usr/bin/env bash
# Freeze (or with "resume", unfreeze) every training process in place.
#   ./pause.sh          stop all pipeline processes, record them in paused.list
#   ./pause.sh resume   continue them
# Frozen processes keep their memory and use no CPU/GPU; sleep is fine, a reboot is not.
set -uo pipefail
cd "$(dirname "$0")"

if [[ "${1:-}" == "resume" ]]; then
  xargs kill -CONT < paused.list && echo "resumed $(wc -l < paused.list | tr -d ' ') processes"
  exit
fi

pids=$(pgrep -f 'run_all.sh|train.py --model|export.py --model|evaluate.py --model|external.py --model' || true)
children=""
for p in $pids; do children="$children $(pgrep -P "$p" || true)"; done
printf '%s\n' $pids $children | grep -v '^$' | sort -u > paused.list
if [[ ! -s paused.list ]]; then echo "nothing running"; exit 0; fi
xargs kill -STOP < paused.list
sleep 2
not_stopped=$(ps -o stat= -p "$(paste -sd, paused.list)" | grep -vc '^T' || true)
echo "$(date '+%H:%M') paused $(wc -l < paused.list | tr -d ' ') processes, not stopped: $not_stopped"
