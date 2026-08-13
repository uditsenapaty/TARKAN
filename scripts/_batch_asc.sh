#!/bin/bash
cd /teamspace/studios/this_studio
echo "[asc] waiting for config batch-explore to finish..."
while pgrep -f "_batch_explore.sh" >/dev/null 2>&1 || pgrep -f "tune_run.py" >/dev/null 2>&1; do sleep 30; done
run() { local tag=$1; shift; echo "[asc] ===== $tag ====="; python3 scripts/tune_run.py --dataset twitter2015 --device cuda --tag "$tag" "$@" || echo "[asc] $tag FAILED"; }
# A7 = dedicated ASC polarity head (targets the MATE-vs-joint polarity gap), isolated + combined
run A_asc        --evidence-dropout 0.2 --kan-hidden 768 --patience 8 --aux-asc-head
run A_asc_A3     --evidence-dropout 0.2 --kan-hidden 768 --patience 8 --aux-asc-head --layerwise-lr 1e-4
run A_asc_evid01 --evidence-dropout 0.1 --kan-hidden 768 --patience 8 --aux-asc-head
echo "[asc] BATCH_ASC_DONE"
