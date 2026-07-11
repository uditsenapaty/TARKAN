#!/bin/bash
# Fine-tune Qwen2.5-VL on one dataset then eval the dev-selected adapter on test.
# Usage: bash run_ft.sh <dataset twitter2015|twitter2017> <model_dir> <tag> [extra train args...]
set -e
cd /teamspace/studios/this_studio
# guard: server restore silently reverts huggingface-hub to >=1.0 and breaks transformers 4.57
pip install -q "huggingface-hub>=0.34.0,<1.0" 2>/dev/null || true
DS=$1; MODEL=$2; TAG=$3; shift 3
OUT=mllm/runs/$TAG
mkdir -p mllm/logs mllm/runs mllm/preds
{
  echo "############ TRAIN $TAG ($DS) $(python -c 'import torch;print(torch.cuda.get_device_name(0))') ############"
  python -u mllm/train_qwen.py --dataset "$DS" --model "$MODEL" --out "$OUT" "$@"
  echo "############ TEST EVAL $TAG ############"
  python -u mllm/eval_qwen.py --dataset "$DS" --split test --model "$MODEL" \
    --adapter "$OUT/best" --dump mllm/preds/${TAG}_test.jsonl
  echo "############ DEV EVAL $TAG ############"
  python -u mllm/eval_qwen.py --dataset "$DS" --split dev --model "$MODEL" \
    --adapter "$OUT/best" --dump mllm/preds/${TAG}_dev.jsonl
  echo "############ DONE $TAG ############"
} > mllm/logs/${TAG}.log 2>&1 &
echo "launched $TAG -> PID $! ; log: mllm/logs/${TAG}.log"
