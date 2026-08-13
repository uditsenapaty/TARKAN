#!/bin/bash
# Queue 13 — three tiny, surgical attacks on the ONE measured failure
# (86 POS + 35 NEG predicted NEU; the model answers "what is this tweet's mood" instead of
# "what does it say about THIS aspect"). None of these adds a module or a parameter beyond
# a scalar; all three are training-signal changes.
#
#  C12 minority-margin : require POS/NEG to beat NEU by m, do NOTHING on NEU examples.
#                        Unlike Chapter B's A1 global class weighting this is
#                        one-directional, matching the one-directional error.
#  C13 sibling-logit   : on the 444 within-tweet different-polarity pairs, require each
#                        aspect to prefer its own label over its sibling's, jointly.
#                        Chapter B's B9 did this on REPRESENTATIONS (SupCon, weak members);
#                        this constrains the DECISION.
#  C14 opinion-dropout : randomly mask SenticNet opinion words that lie outside +-4 tokens
#                        of the target, removing the tweet-level shortcut at the input.
set -e
cd /teamspace/studios/this_studio
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
S=runs/mate_ens5
M=vinai/bertweet-large

echo "=== [1/4] C12 minority-margin 0.5 ==="
python3 experts/masc_text.py --model $M --minority-margin 0.5 --seed 90 \
  --batch 8 --lr 1e-5 --epochs 8 --spans $S --out runs/masc_btwL_mm

echo "=== [2/4] C13 sibling-logit loss 0.5 ==="
python3 experts/masc_text.py --model $M --sibling-loss 0.5 --seed 91 \
  --batch 8 --lr 1e-5 --epochs 8 --spans $S --out runs/masc_btwL_sibloss

echo "=== [3/4] C12+C13+C14 combined ==="
python3 experts/masc_text.py --model $M --minority-margin 0.5 --sibling-loss 0.5 \
  --opinion-dropout 0.3 --mark-siblings --seed 92 \
  --batch 8 --lr 1e-5 --epochs 8 --spans $S --out runs/masc_btwL_combo

echo "=== [4/4] diagnose + assemble EVERYTHING ==="
MASC=""
for d in runs/masc_twrob_s42 runs/masc_btw_s43 runs/masc_deb_s44 runs/masc_btwL_s45 \
         runs/masc_robL_s46 runs/pdq_s42 runs/pdq_twrob_s43 runs/pdq_btw_s42 \
         runs/pdq_btwL_s47 runs/masc_btwL_bal runs/masc_deb_sqrt runs/masc_twrob_bal \
         runs/masc_btwL_sib runs/masc_deb_sib runs/masc_btwL_sibbal \
         runs/masc_btwL_mm runs/masc_btwL_sibloss runs/masc_btwL_combo \
         runs/masc_btwL_aadg_s48 runs/masc_deb_aadg_s49 \
         runs/masc_gate_btwL_cvx runs/masc_gate_btwL_cat runs/masc_gate_deb_cat; do
  [ -f "$d/probs_test.npz" ] && MASC="$MASC $d"
done
echo "members:$MASC"
python3 experts/diagnose.py --masc $MASC
MATE="runs/mate_deb_s42 runs/mate_deb_s43 runs/mate_deb_s44 runs/mate_probeA runs/mate_probeB"
python3 experts/assemble.py --mate $MATE --masc $MASC --mode span  --out results/q13_span.json
python3 experts/assemble.py --mate $MATE --masc $MASC --mode joint --out results/q13_joint.json
echo "=== QUEUE13 DONE ==="
