#!/bin/bash
# Queue 19 — C2c: complete the DQPSA objective (ITC + ITM + EPE, all weight 1.0).
# We had been running their `no_its_and_itm` branch. The three heads resume from BLIP-2's
# 129M-pair stage-1 pretraining rather than starting cold, so this is the one PDQ variant
# that actually exercises the vision->text bridge the way DQPSA does.
set -e
cd /teamspace/studios/this_studio
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
S=runs/mate_ens5_hr
echo "=== [1/2] PDQ+ITC+ITM, bertweet-large ==="
python3 experts/pdq.py --text-model vinai/bertweet-large --itc 1.0 --itm 1.0 --seed 60 \
  --lr 1e-5 --epochs 8 --spans $S --out runs/pdq_btwL_itcitm
echo "=== [2/2] PDQ+ITC+ITM, twitter-roberta ==="
python3 experts/pdq.py --text-model cardiffnlp/twitter-roberta-base-sentiment-latest \
  --itc 1.0 --itm 1.0 --seed 61 --lr 2e-5 --epochs 8 --spans $S --out runs/pdq_twrob_itcitm
echo "=== QUEUE19 DONE ==="
