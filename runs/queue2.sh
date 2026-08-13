#!/bin/bash
# Queue 2 — lift MATE with multi-seed marginal averaging (Chapter B's proven lever),
# then RE-SCORE the existing MASC members against the ensemble's candidate anchors
# (--score-only, no retraining) and re-assemble.
set -e
cd /teamspace/studios/this_studio

echo "=== [1/6] MATE seed 43 ==="
python3 experts/mate_expert.py --seed 43 --epochs 10 --out runs/mate_deb_s43

echo "=== [2/6] MATE seed 44 ==="
python3 experts/mate_expert.py --seed 44 --epochs 10 --out runs/mate_deb_s44

echo "=== [3/6] ensemble candidate anchors ==="
python3 experts/emit_spans.py \
  --mate runs/mate_deb_s42 runs/mate_deb_s43 runs/mate_deb_s44 \
  --out runs/mate_ens3

echo "=== [4/6] re-score text MASC members on ensemble anchors ==="
python3 experts/masc_text.py --model cardiffnlp/twitter-roberta-base-sentiment-latest \
  --score-only --spans runs/mate_ens3 --out runs/masc_twrob_s42
python3 experts/masc_text.py --model vinai/bertweet-base \
  --score-only --spans runs/mate_ens3 --out runs/masc_btw_s43

echo "=== [5/6] re-score PDQ member on ensemble anchors ==="
python3 experts/pdq.py --score-only --spans runs/mate_ens3 --out runs/pdq_s42

echo "=== [6/6] assemble ==="
python3 experts/assemble.py \
  --mate runs/mate_deb_s42 runs/mate_deb_s43 runs/mate_deb_s44 \
  --masc runs/masc_twrob_s42 runs/masc_btw_s43 runs/pdq_s42 \
  --mode span  --out results/assemble3_span.json
python3 experts/assemble.py \
  --mate runs/mate_deb_s42 runs/mate_deb_s43 runs/mate_deb_s44 \
  --masc runs/masc_twrob_s42 runs/masc_btw_s43 runs/pdq_s42 \
  --mode joint --out results/assemble3_joint.json
echo "=== QUEUE2 DONE ==="
