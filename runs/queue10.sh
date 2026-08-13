#!/bin/bash
# Queue 10 — out-of-fold stacking (C8).
# Every Chapter-B combiner was fit on dev (n=1122, binomial sigma ~1.2 > the entire gap
# being chased) and every one of them lost on test. This fits on 3179 train-OOF aspects
# instead. Fold runs write to their own dirs; only probs_oof*.npz is copied back next to
# the FULL member's dev/test/span probabilities, so the stacker is applied with full-data
# members at test time.
set -e
cd /teamspace/studios/this_studio
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
D=data/aadg/twitter2015

oof () {   # $1=model  $2=member_dir  $3=tag  $4=lr  $5=batch
  for f in 0 1; do
    python3 experts/masc_text.py --model "$1" --fold $f --nfolds 2 --seed 60 \
      --batch "$5" --lr "$4" --epochs 6 --out "runs/oof_$3_f$f"
    cp "runs/oof_$3_f$f/probs_oof$f.npz" "$2/" 2>/dev/null || true
  done
}

echo "=== [1/5] OOF: bertweet-large ==="
oof vinai/bertweet-large runs/masc_btwL_s45 btwL 1e-5 8
echo "=== [2/5] OOF: bertweet-base ==="
oof vinai/bertweet-base runs/masc_btw_s43 btw 2e-5 16
echo "=== [3/5] OOF: twitter-roberta ==="
oof cardiffnlp/twitter-roberta-base-sentiment-latest runs/masc_twrob_s42 twrob 2e-5 16
echo "=== [4/5] OOF: deberta-v3-large ==="
oof microsoft/deberta-v3-large runs/masc_deb_s44 deb 1e-5 8

echo "=== [5/5] fit stacker + assemble ==="
python3 experts/stack.py --masc runs/masc_btwL_s45 runs/masc_btw_s43 \
  runs/masc_twrob_s42 runs/masc_deb_s44

MATE="runs/mate_deb_s42 runs/mate_deb_s43 runs/mate_deb_s44 runs/mate_probeA runs/mate_probeB"
# stacked block treated as ONE member alongside the decorrelated PDQ/gated members
MASC="runs/stack_masc"
for d in runs/pdq_s42 runs/pdq_twrob_s43 runs/pdq_btw_s42 runs/pdq_btwL_s47 \
         runs/masc_robL_s46 runs/masc_btwL_aadg_s48 runs/masc_deb_aadg_s49 \
         runs/masc_gate_btwL_cvx runs/masc_gate_btwL_cat runs/masc_gate_deb_cat; do
  [ -f "$d/probs_test.npz" ] && MASC="$MASC $d"
done
echo "members:$MASC"
python3 experts/assemble.py --mate $MATE --masc $MASC --mode span  --out results/q10_span.json
python3 experts/assemble.py --mate $MATE --masc $MASC --mode joint --out results/q10_joint.json
echo "=== QUEUE10 DONE ==="
