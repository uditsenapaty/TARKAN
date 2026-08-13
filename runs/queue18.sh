#!/bin/bash
# Queue 18 — strengthen the span judge (now the highest-leverage component).
# The current reranker is ONE bertweet-large at dev AUC 0.8102, trained on candidates from
# 2-fold half-data MATE models. Judges are cheap (~10 min) and averaging them attacks the
# variance directly, which matters because the judge now feeds a geometric evidence product
# where a single noisy score propagates.
set -e
cd /teamspace/studios/this_studio
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
MATE="runs/mate_deb_s42 runs/mate_deb_s43 runs/mate_deb_s44 runs/mate_probeA runs/mate_probeB"

echo "=== [1/2] judge 2: deberta-v3-large (arch-diverse judge) ==="
python3 experts/span_rerank.py --oof runs/mate_oof_f0 runs/mate_oof_f1 --cand runs/mate_ens5 \
  --mate $MATE --cand-thr 0.12 --model microsoft/deberta-v3-large --seed 101 --epochs 5 \
  --out runs/rerank_deb

echo "=== [2/2] judge 3: bertweet-large, different seed ==="
python3 experts/span_rerank.py --oof runs/mate_oof_f0 runs/mate_oof_f1 --cand runs/mate_ens5 \
  --mate $MATE --cand-thr 0.12 --model vinai/bertweet-large --seed 102 --epochs 6 \
  --out runs/rerank_btwL2
echo "=== QUEUE18 DONE ==="
