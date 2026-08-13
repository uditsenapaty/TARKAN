#!/bin/bash
# Queue 9 — MADSC's calibrated modality gate (their largest single ablation: -5.74 MABSA
# Mac-F1 without it). Two fusion variants: `convex` is faithful to Eq. 14/16 (classify from
# z alone), `concat` also keeps the raw aspect and sentence vectors, which is the obvious
# minimal deviation if the convex form starves the classifier of text.
set -e
cd /teamspace/studios/this_studio
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
S=runs/mate_ens5
D=data/aadg/twitter2015

echo "=== [1/4] gated MASC, bertweet-large, convex fusion (faithful Eq.14/16) ==="
python3 experts/masc_gated.py --model vinai/bertweet-large --fuse convex --seed 50 \
  --desc $D --spans $S --out runs/masc_gate_btwL_cvx

echo "=== [2/4] gated MASC, bertweet-large, concat fusion (minimal deviation) ==="
python3 experts/masc_gated.py --model vinai/bertweet-large --fuse concat --seed 51 \
  --desc $D --spans $S --out runs/masc_gate_btwL_cat

echo "=== [3/4] gated MASC, deberta-v3-large, concat fusion ==="
python3 experts/masc_gated.py --model microsoft/deberta-v3-large --fuse concat --seed 52 \
  --desc $D --spans $S --out runs/masc_gate_deb_cat

echo "=== [4/4] FINAL diagnose + assemble over every member ==="
MASC=""
for d in runs/masc_twrob_s42 runs/masc_btw_s43 runs/masc_deb_s44 runs/masc_btwL_s45 \
         runs/masc_robL_s46 runs/pdq_s42 runs/pdq_twrob_s43 runs/pdq_btw_s42 \
         runs/pdq_btwL_s47 runs/masc_btwL_aadg_s48 runs/masc_deb_aadg_s49 \
         runs/masc_gate_btwL_cvx runs/masc_gate_btwL_cat runs/masc_gate_deb_cat; do
  [ -f "$d/probs_test.npz" ] && MASC="$MASC $d"
done
echo "members:$MASC"
python3 experts/diagnose.py --masc $MASC
MATE="runs/mate_deb_s42 runs/mate_deb_s43 runs/mate_deb_s44 runs/mate_probeA runs/mate_probeB"
python3 experts/assemble.py --mate $MATE --masc $MASC --mode span  --out results/q9_span.json
python3 experts/assemble.py --mate $MATE --masc $MASC --mode joint --out results/q9_joint.json
echo "=== QUEUE9 DONE ==="
