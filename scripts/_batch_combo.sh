#!/bin/bash
cd /teamspace/studios/this_studio
# guard against env reverting huggingface-hub on server restore (breaks transformers import)
python3 -c "import transformers" 2>/dev/null || pip install "huggingface-hub>=0.34.0,<1.0" >/dev/null 2>&1
run() { local tag=$1; shift; echo "[combo] ===== $tag ====="; python3 scripts/tune_run.py --dataset twitter2015 --device cuda --tag "$tag" "$@" || echo "[combo] $tag FAILED"; }
run C1_richasc     --evidence-dropout 0.2 --kan-hidden 768 --patience 8 --aux-asc-head
run C2_richasc_A3  --evidence-dropout 0.2 --kan-hidden 768 --patience 8 --aux-asc-head --layerwise-lr 1e-4
run C3_richasc_A3_long --evidence-dropout 0.2 --kan-hidden 768 --patience 10 --epochs 45 --aux-asc-head --layerwise-lr 1e-4
echo "[combo] BATCH_COMBO_DONE_V2"
