#!/usr/bin/env bash
# D3 — Qwen2.5-VL counterfactual evidence teacher over the full TRAIN split, then retrain
# the PDS members on the image-grounded direction labels instead of the caption-based ones.
#
# This is the one remaining route that does NOT depend on selecting a combiner on dev:
# PDS is a TRAINING-time mechanism that changes the members themselves, so it sidesteps
# the failure mode behind all seven dev/test inversions in Chapters C and D.
set -u
echo "=== [0] sanity: 40-aspect label distribution under the fixed mapping ==="
python3 -u experts/masc_qwenvl.py --counterfactual --split train --limit 40 \
    --out runs/qwen_cf_smoke2 2>&1 | grep -v "it/s\]" | tail -5

echo "=== [1] Qwen2.5-VL counterfactual teacher, full TRAIN split ==="
python3 -u experts/masc_qwenvl.py --counterfactual --split train \
    --out data/pds_qwen/twitter2015 2>&1 | grep -v "it/s\]" | tail -6

for m in "qpds_btwL vinai/bertweet-large" \
         "qpds_deb microsoft/deberta-v3-large" \
         "qpds_robL roberta-large"; do
  set -- $m
  echo "=== [2] PDS member on Qwen-VL directions: $1 ==="
  python3 -u experts/masc_pds.py --residual --w-none 0 --pds data/pds_qwen/twitter2015 \
      --model "$2" --spans runs/mate_ens5_hr --out runs/$1 2>&1 | grep -v "it/s\]" | tail -12
done
echo "=== QUEUE29 DONE ==="
