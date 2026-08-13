#!/bin/bash
# A11: best-available text backbone (deberta-v3-large) on the calibrated champion config.
cd /teamspace/studios/this_studio
python3 -c "import transformers" 2>/dev/null || pip install "huggingface-hub>=0.34.0,<1.0" >/dev/null 2>&1
echo "[bb] waiting for fusion probes to finish..."
until grep -q "PROBES_DONE" results/logs/probe_pipeline.log 2>/dev/null; do sleep 180; done
while pgrep -f "tune_run.py" >/dev/null 2>&1; do sleep 60; done
run() { local tag=$1 ds=$2; shift 2; echo "[bb] ===== $tag ($ds) ====="; python3 scripts/tune_run.py --dataset $ds --device cuda --tag "$tag" "$@" || echo "[bb] $tag FAILED"; }
BASE="--evidence-dropout 0.2 --kan-hidden 768 --patience 8 --crf --aux-asc-head --text-model microsoft/deberta-v3-large --batch-size 4 --grad-accum 4"
run A11_deberta_t15 twitter2015 $BASE
run A11_deberta_t17 twitter2017 $BASE
echo "[bb] BACKBONE_PROBES_DONE"
