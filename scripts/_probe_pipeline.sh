#!/bin/bash
# Fusion-patch probes (A9 conf-append, A10 feat-gate) on the CALIBRATED champion.
# Waits for recal pipeline's Table 1+3, intercepts before its 25h ablation tail.
cd /teamspace/studios/this_studio
python3 -c "import transformers" 2>/dev/null || pip install "huggingface-hub>=0.34.0,<1.0" >/dev/null 2>&1
echo "[probe] waiting for calibrated Table 1+3 (recal 'Analyses' marker)..."
until grep -q "Analyses (Tables 8, 9, 5, 7)" results/logs/recal_pipeline.log 2>/dev/null; do sleep 120; done
echo "[probe] intercepting recal pipeline before ablations"
pkill -f _recal_pipeline.sh 2>/dev/null; sleep 2
pkill -9 -f "analysis/" 2>/dev/null; pkill -9 -f run_ablations 2>/dev/null; pkill -9 -f run_fusion 2>/dev/null; sleep 3
run() { local tag=$1; shift; echo "[probe] ===== $tag ====="; python3 scripts/tune_run.py --dataset twitter2015 --device cuda --tag "$tag" "$@" || echo "[probe] $tag FAILED"; }
BASE="--evidence-dropout 0.2 --kan-hidden 768 --patience 8 --crf --aux-asc-head"
run P_A9_conf      $BASE --conf-append
run P_A10_gate     $BASE --feat-gate
run P_A9A10_both   $BASE --conf-append --feat-gate
echo "[probe] PROBES_DONE"
