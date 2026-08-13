#!/bin/bash
# Queue 5 — polarity is the dominant gap (a 77.9 -> needs 82.4), so add STRONGER and
# architecture-diverse MASC members. Chapter A measured a dedicated ASC head on
# DeBERTa-v3-large at 78.5 MASC acc, above both current text members (77.3 / 78.1), and
# Chapter A/B both used bertweet-large. These are ~15 min each on the T4.
set -e
cd /teamspace/studios/this_studio

echo "=== [1/3] MASC: deberta-v3-large ==="
python3 experts/masc_text.py --model microsoft/deberta-v3-large --seed 44 \
  --batch 8 --lr 1e-5 --epochs 8 --spans runs/mate_ens3 --out runs/masc_deb_s44

echo "=== [2/3] MASC: bertweet-large ==="
python3 experts/masc_text.py --model vinai/bertweet-large --seed 45 \
  --batch 8 --lr 1e-5 --epochs 8 --spans runs/mate_ens3 --out runs/masc_btwL_s45

echo "=== [3/3] diagnose + assemble everything available ==="
MASC=""
for d in runs/masc_twrob_s42 runs/masc_btw_s43 runs/pdq_s42 runs/pdq_btw_s42 \
         runs/pdq_twrob_s43 runs/masc_deb_s44 runs/masc_btwL_s45; do
  [ -f "$d/probs_test.npz" ] && MASC="$MASC $d"
done
echo "members:$MASC"
python3 experts/diagnose.py --masc $MASC
python3 experts/assemble.py \
  --mate runs/mate_deb_s42 runs/mate_deb_s43 runs/mate_deb_s44 \
  --masc $MASC --mode span  --out results/assembleN_span.json
python3 experts/assemble.py \
  --mate runs/mate_deb_s42 runs/mate_deb_s43 runs/mate_deb_s44 \
  --masc $MASC --mode joint --out results/assembleN_joint.json
echo "=== QUEUE5 DONE ==="
