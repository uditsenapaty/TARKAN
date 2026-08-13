#!/bin/bash
# Queue 14 — MATE out-of-fold folds + annotation-policy span reranker (C4).
set -e
cd /teamspace/studios/this_studio
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
echo "=== [1/3] MATE OOF fold 0 ==="
python3 experts/mate_expert.py --seed 42 --lr 1e-5 --head-lr 1e-4 --epochs 12 --patience 4 \
  --dropout 0.3 --fold 0 --nfolds 2 --out runs/mate_oof_f0
echo "=== [2/3] MATE OOF fold 1 ==="
python3 experts/mate_expert.py --seed 42 --lr 1e-5 --head-lr 1e-4 --epochs 12 --patience 4 \
  --dropout 0.3 --fold 1 --nfolds 2 --out runs/mate_oof_f1
echo "=== [3/3] train span reranker ==="
MATE="runs/mate_deb_s42 runs/mate_deb_s43 runs/mate_deb_s44 runs/mate_probeA runs/mate_probeB"
python3 experts/span_rerank.py --oof runs/mate_oof_f0 runs/mate_oof_f1 \
  --cand runs/mate_ens5 --mate $MATE --cand-thr 0.12 --out runs/rerank
echo "=== QUEUE14 DONE ==="
