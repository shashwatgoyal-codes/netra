#!/usr/bin/env bash
# Freeze (or with "resume", unfreeze) every training process in place.
#   ./pause.sh          stop all pipeline processes, record them in paused.list
#   ./pause.sh resume   continue them
# Frozen processes keep their memory and use no CPU/GPU; sleep is fine, a reboot is not.
set -uo pipefail
cd "$(dirname "$0")"

if [[ "${1:-}" == "resume" ]]; then
  resumed=0
  while read -r p; do kill -CONT "$p" 2>/dev/null && resumed=$((resumed + 1)); done < paused.list
  echo "resumed $resumed of $(wc -l < paused.list | tr -d ' ') processes (the rest had already exited)"
  exit
fi

# Anchored to the start of the command line, so a shell that merely mentions
# run_all.sh (a scheduler, a grep) is not frozen along with the pipeline.
pids=$(pgrep -f '^bash \./run_all\.sh|^(uv run )?[^ ]*python3? (train|export|evaluate|external)\.py --model' || true)
children=""
for p in $pids; do children="$children $(pgrep -P "$p" || true)"; done
printf '%s\n' $pids $children | grep -v '^$' | sort -u > paused.list
if [[ ! -s paused.list ]]; then echo "nothing running"; exit 0; fi
xargs kill -STOP < paused.list
sleep 2
not_stopped=$(ps -o stat= -p "$(paste -sd, paused.list)" | grep -vc '^T' || true)
echo "$(date '+%H:%M') paused $(wc -l < paused.list | tr -d ' ') processes, not stopped: $not_stopped"
