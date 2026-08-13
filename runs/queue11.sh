#!/bin/bash
# Queue 11 — the measured hole is POS/NEG recall, not overall accuracy.
#   ours: POS R 72.24 / NEU R 89.46 / NEG R 61.95, acc 81.20, macro 76.84
#   MADSC: POS F1 84.4 / NEU 78.8 / NEG 74.3,      acc 82.34, macro 78.38
# We BEAT MADSC on NEU and lose ~8 F1 on POS and ~6 on NEG; 86 POS and 35 NEG aspects are
# predicted NEU. Chapter B's "class weighting hurts" was measured on the joint 7-tag BIO
# head, where re-weighting collapsed EXTRACTION -- structurally impossible on a dedicated
# 3-way head. These are added as extra ensemble members, never as replacements.
set -e
cd /teamspace/studios/this_studio
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
S=runs/mate_ens5

echo "=== [1/4] balanced-CE member: bertweet-large ==="
python3 experts/masc_text.py --model vinai/bertweet-large --class-weight balanced \
  --seed 70 --batch 8 --lr 1e-5 --epochs 8 --spans $S --out runs/masc_btwL_bal

echo "=== [2/4] sqrt-balanced member: deberta-v3-large (gentler reweighting) ==="
python3 experts/masc_text.py --model microsoft/deberta-v3-large --class-weight sqrt \
  --seed 71 --batch 8 --lr 1e-5 --epochs 8 --spans $S --out runs/masc_deb_sqrt

echo "=== [3/4] balanced-CE member: twitter-roberta (cheap, diverse) ==="
python3 experts/masc_text.py --model cardiffnlp/twitter-roberta-base-sentiment-latest \
  --class-weight balanced --seed 72 --batch 16 --lr 2e-5 --epochs 6 --spans $S \
  --out runs/masc_twrob_bal

echo "=== [4/4] diagnose + assemble ==="
MASC=""
for d in runs/masc_twrob_s42 runs/masc_btw_s43 runs/masc_deb_s44 runs/masc_btwL_s45 \
         runs/masc_robL_s46 runs/pdq_s42 runs/pdq_twrob_s43 runs/pdq_btw_s42 \
         runs/pdq_btwL_s47 runs/masc_btwL_bal runs/masc_deb_sqrt runs/masc_twrob_bal \
         runs/masc_btwL_aadg_s48 runs/masc_deb_aadg_s49 \
         runs/masc_gate_btwL_cvx runs/masc_gate_btwL_cat runs/masc_gate_deb_cat; do
  [ -f "$d/probs_test.npz" ] && MASC="$MASC $d"
done
echo "members:$MASC"
python3 experts/diagnose.py --masc $MASC
MATE="runs/mate_deb_s42 runs/mate_deb_s43 runs/mate_deb_s44 runs/mate_probeA runs/mate_probeB"
python3 experts/assemble.py --mate $MATE --masc $MASC --mode span  --out results/q11_span.json
python3 experts/assemble.py --mate $MATE --masc $MASC --mode joint --out results/q11_joint.json
echo "=== QUEUE11 DONE ==="
