#!/bin/bash
# Serial GPU queue — the T4 is a single card, so everything runs one at a time.
set -e
cd /teamspace/studios/this_studio

echo "=== [1/4] text MASC member: twitter-roberta ==="
python3 experts/masc_text.py \
  --model cardiffnlp/twitter-roberta-base-sentiment-latest \
  --seed 42 --spans runs/mate_deb_s42 --out runs/masc_twrob_s42

echo "=== [2/4] text MASC member: bertweet (architecture diversity) ==="
python3 experts/masc_text.py \
  --model vinai/bertweet-base \
  --seed 43 --spans runs/mate_deb_s42 --out runs/masc_btw_s43

echo "=== [3/4] PDQ member (BLIP-2 Q-Former, the decorrelation test) ==="
python3 experts/pdq.py --seed 42 --spans runs/mate_deb_s42 --out runs/pdq_s42

echo "=== [4/4] assemble: Chapter-B span-tau vs C3 joint-product-tau ==="
python3 experts/assemble.py --mate runs/mate_deb_s42 \
  --masc runs/masc_twrob_s42 runs/masc_btw_s43 runs/pdq_s42 \
  --mode span  --out results/assemble_span.json
python3 experts/assemble.py --mate runs/mate_deb_s42 \
  --masc runs/masc_twrob_s42 runs/masc_btw_s43 runs/pdq_s42 \
  --mode joint --out results/assemble_joint.json
echo "=== QUEUE DONE ==="
