#!/bin/bash
# Queue 7 — MATE ens5 hits 87.01 (clears MADSC 86.60 / CORSA 86.30 / AoM 86.20).
# Re-score every MASC member against the NEW ens5 candidate anchors (score-only, no
# retraining), then add two more members. Polarity is now the whole gap: joint = 87.01*a,
# so beating MADSC's 72.9 needs a = 83.8 (we are at 80.91).
set -e
cd /teamspace/studios/this_studio
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
S=runs/mate_ens5

echo "=== [1/4] re-score existing MASC members on ens5 anchors ==="
python3 experts/masc_text.py --model cardiffnlp/twitter-roberta-base-sentiment-latest --score-only --spans $S --out runs/masc_twrob_s42
python3 experts/masc_text.py --model vinai/bertweet-base    --score-only --spans $S --out runs/masc_btw_s43
python3 experts/masc_text.py --model microsoft/deberta-v3-large --score-only --spans $S --out runs/masc_deb_s44
python3 experts/masc_text.py --model vinai/bertweet-large   --score-only --spans $S --out runs/masc_btwL_s45
python3 experts/pdq.py --text-model bert-base-uncased       --score-only --spans $S --out runs/pdq_s42
python3 experts/pdq.py --text-model cardiffnlp/twitter-roberta-base-sentiment-latest --score-only --spans $S --out runs/pdq_twrob_s43
python3 experts/pdq.py --text-model vinai/bertweet-base     --score-only --spans $S --out runs/pdq_btw_s42

echo "=== [2/4] NEW member: roberta-large MASC ==="
python3 experts/masc_text.py --model roberta-large --seed 46 --batch 8 --lr 1e-5 \
  --epochs 8 --spans $S --out runs/masc_robL_s46

echo "=== [3/4] NEW member: PDQ + bertweet-large (decorrelated mechanism x strongest tower) ==="
python3 experts/pdq.py --text-model vinai/bertweet-large --seed 47 --batch 8 --lr 1e-5 \
  --epochs 8 --spans $S --out runs/pdq_btwL_s47

echo "=== [4/4] diagnose + assemble ==="
MASC=""
for d in runs/masc_twrob_s42 runs/masc_btw_s43 runs/masc_deb_s44 runs/masc_btwL_s45 \
         runs/masc_robL_s46 runs/pdq_s42 runs/pdq_twrob_s43 runs/pdq_btw_s42 runs/pdq_btwL_s47; do
  [ -f "$d/probs_test.npz" ] && MASC="$MASC $d"
done
echo "members:$MASC"
python3 experts/diagnose.py --masc $MASC
MATE="runs/mate_deb_s42 runs/mate_deb_s43 runs/mate_deb_s44 runs/mate_probeA runs/mate_probeB"
python3 experts/assemble.py --mate $MATE --masc $MASC --mode span  --out results/q7_span.json
python3 experts/assemble.py --mate $MATE --masc $MASC --mode joint --out results/q7_joint.json
echo "=== QUEUE7 DONE ==="
