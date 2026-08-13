#!/bin/bash
cd /teamspace/studios/this_studio
python3 -c "import transformers" 2>/dev/null || pip install "huggingface-hub>=0.34.0,<1.0" >/dev/null 2>&1
echo "[crf] waiting for current tune_run to finish..."
while pgrep -f "tune_run.py" >/dev/null 2>&1; do sleep 30; done
run() { local tag=$1; shift; echo "[crf] ===== $tag ====="; python3 scripts/tune_run.py --dataset twitter2015 --device cuda --tag "$tag" "$@" || echo "[crf] $tag FAILED"; }
run D1_crf          --evidence-dropout 0.2 --kan-hidden 768 --patience 8 --crf
run D2_crf_richasc  --evidence-dropout 0.2 --kan-hidden 768 --patience 8 --crf --aux-asc-head
echo "[crf] BATCH_CRF_DONE"
