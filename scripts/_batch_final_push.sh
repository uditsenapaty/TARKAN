#!/bin/bash
cd /teamspace/studios/this_studio
python3 -c "import transformers" 2>/dev/null || pip install "huggingface-hub>=0.34.0,<1.0" >/dev/null 2>&1
echo "[push] waiting for E8 large run to finish..."
while pgrep -f "_launch_large.sh" >/dev/null 2>&1 || pgrep -f "tune_run.py" >/dev/null 2>&1; do sleep 60; done
run() { local tag=$1 ds=$2; shift 2; echo "[push] ===== $tag ($ds) ====="; python3 scripts/tune_run.py --dataset $ds --device cuda --tag "$tag" "$@" || echo "[push] $tag FAILED"; }
# full composition: bertweet-large + CRF + rich ASC head, best config base
run E9_large_crf_richasc twitter2015 --text-model vinai/bertweet-large --batch-size 8 --grad-accum 2 \
    --evidence-dropout 0.2 --kan-hidden 768 --patience 8 --crf --aux-asc-head
run T17_large_crf_richasc twitter2017 --text-model vinai/bertweet-large --batch-size 8 --grad-accum 2 \
    --evidence-dropout 0.2 --kan-hidden 768 --patience 8 --crf --aux-asc-head
echo "[push] BATCH_PUSH_DONE"
