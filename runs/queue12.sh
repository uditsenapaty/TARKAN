#!/bin/bash
# Queue 12 — sibling-aspect marking (C10) + combination with class balancing.
# The measured failure is 86 POS -> NEU: the model falls back on the tweet's overall tone
# instead of the target aspect. Every member so far marks ONLY the target aspect, so it has
# no way to know the tweet contains other aspects competing for the sentiment. t2015 has
# 1.51 aspects/sentence and 444 within-tweet pairs with DIFFERENT gold polarity. Chapter B
# attacked this at the loss level (SupCon -> weak members); this is the input-level version,
# which costs nothing.
set -e
cd /teamspace/studios/this_studio
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
S=runs/mate_ens5

echo "=== [1/4] siblings: bertweet-large ==="
python3 experts/masc_text.py --model vinai/bertweet-large --mark-siblings \
  --seed 80 --batch 8 --lr 1e-5 --epochs 8 --spans $S --out runs/masc_btwL_sib

echo "=== [2/4] siblings: deberta-v3-large ==="
python3 experts/masc_text.py --model microsoft/deberta-v3-large --mark-siblings \
  --seed 81 --batch 8 --lr 1e-5 --epochs 8 --spans $S --out runs/masc_deb_sib

echo "=== [3/4] siblings + balanced: bertweet-large ==="
python3 experts/masc_text.py --model vinai/bertweet-large --mark-siblings \
  --class-weight sqrt --seed 82 --batch 8 --lr 1e-5 --epochs 8 --spans $S \
  --out runs/masc_btwL_sibbal

echo "=== [4/4] diagnose + assemble ==="
MASC=""
for d in runs/masc_twrob_s42 runs/masc_btw_s43 runs/masc_deb_s44 runs/masc_btwL_s45 \
         runs/masc_robL_s46 runs/pdq_s42 runs/pdq_twrob_s43 runs/pdq_btw_s42 \
         runs/pdq_btwL_s47 runs/masc_btwL_bal runs/masc_deb_sqrt runs/masc_twrob_bal \
         runs/masc_btwL_sib runs/masc_deb_sib runs/masc_btwL_sibbal \
         runs/masc_btwL_aadg_s48 runs/masc_deb_aadg_s49 \
         runs/masc_gate_btwL_cvx runs/masc_gate_btwL_cat runs/masc_gate_deb_cat; do
  [ -f "$d/probs_test.npz" ] && MASC="$MASC $d"
done
echo "members:$MASC"
python3 experts/diagnose.py --masc $MASC
MATE="runs/mate_deb_s42 runs/mate_deb_s43 runs/mate_deb_s44 runs/mate_probeA runs/mate_probeB"
python3 experts/assemble.py --mate $MATE --masc $MASC --mode span  --out results/q12_span.json
python3 experts/assemble.py --mate $MATE --masc $MASC --mode joint --out results/q12_joint.json
echo "=== QUEUE12 DONE ==="
