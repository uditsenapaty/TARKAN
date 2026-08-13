#!/bin/bash
# Queue 3 — PDQ is the most decorrelated member we have (ratio 2.54 vs 3.03 text-text,
# unique-right 3.95%) but the weakest (76.37) because its text encoder is vanilla
# bert-base-uncased on tweets. DQPSA used a task-pretrained BERT there. Swap in
# Twitter-domain encoders: the decorrelation lives in the Q-Former/EPE mechanism, so it
# should survive while the member gets stronger.
set -e
cd /teamspace/studios/this_studio

echo "=== [1/4] PDQ + bertweet text encoder ==="
python3 experts/pdq.py --text-model vinai/bertweet-base --seed 42 \
  --spans runs/mate_ens3 --out runs/pdq_btw_s42

echo "=== [2/4] PDQ + twitter-roberta text encoder ==="
python3 experts/pdq.py --text-model cardiffnlp/twitter-roberta-base-sentiment-latest \
  --seed 43 --spans runs/mate_ens3 --out runs/pdq_twrob_s43

echo "=== [3/4] correlation diagnostic over all MASC members ==="
python3 experts/diagnose.py --masc runs/masc_twrob_s42 runs/masc_btw_s43 \
  runs/pdq_s42 runs/pdq_btw_s42 runs/pdq_twrob_s43

echo "=== [4/4] assemble with the full member set ==="
python3 experts/assemble.py \
  --mate runs/mate_deb_s42 runs/mate_deb_s43 runs/mate_deb_s44 \
  --masc runs/masc_twrob_s42 runs/masc_btw_s43 runs/pdq_s42 runs/pdq_btw_s42 runs/pdq_twrob_s43 \
  --mode span  --out results/assemble5_span.json
python3 experts/assemble.py \
  --mate runs/mate_deb_s42 runs/mate_deb_s43 runs/mate_deb_s44 \
  --masc runs/masc_twrob_s42 runs/masc_btw_s43 runs/pdq_s42 runs/pdq_btw_s42 runs/pdq_twrob_s43 \
  --mode joint --out results/assemble5_joint.json
echo "=== QUEUE3 DONE ==="
