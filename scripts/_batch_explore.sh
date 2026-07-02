#!/bin/bash
cd /teamspace/studios/this_studio
echo "[batch] waiting for the running tune_run (Round 3) to finish..."
while pgrep -f "tune_run.py" >/dev/null 2>&1; do sleep 30; done
run() {
  local tag="$1"; shift
  echo "[batch] ===== $tag ====="
  python3 scripts/tune_run.py --dataset twitter2015 --device cuda --tag "$tag" "$@" || echo "[batch] $tag FAILED"
}
# isolated single-lever tests around the proven evidence_dropout lever + fusion capacity
run E_evid01   --evidence-dropout 0.1  --kan-hidden 768      --patience 8
run E_evid005  --evidence-dropout 0.05 --kan-hidden 768      --patience 8
run E_kan1024  --evidence-dropout 0.2  --kan-hidden 1024,512 --patience 8
echo "[batch] BATCH_EXPLORE_DONE"
