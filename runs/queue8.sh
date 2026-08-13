#!/bin/bash
# Queue 8 — MADSC's AADG. Polarity is now the entire gap (joint = 87.01 * a; need a=83.8,
# have 80.91). MADSC reaches a~84.2 with a WEAKER extractor than ours, and attributes it to
# aspect-aware descriptions + calibrated gating rather than to the backbone. Chapter B's
# generic-caption experiment diluted the ensemble, which is precisely the failure mode MADSC
# describes -- so the aspect-conditioned rewrite is a genuinely untried lever, not a repeat.
set -e
cd /teamspace/studios/this_studio
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

echo "=== [1/5] captions (BLIP) ==="
python3 experts/aadg.py --stage captions
echo "=== [2/5] region embeddings (CLIP, 3x3 grid + full) ==="
python3 experts/aadg.py --stage regions
echo "=== [3/5] build D_aspect ==="
python3 experts/aadg.py --stage describe --spans runs/mate_ens5

echo "=== [4/5] AADG members (paired with the two strongest towers) ==="
python3 experts/masc_text.py --model vinai/bertweet-large --seed 48 --batch 8 --lr 1e-5 \
  --epochs 8 --spans runs/mate_ens5 --desc data/aadg/twitter2015 --out runs/masc_btwL_aadg_s48
python3 experts/masc_text.py --model microsoft/deberta-v3-large --seed 49 --batch 8 --lr 1e-5 \
  --epochs 8 --spans runs/mate_ens5 --desc data/aadg/twitter2015 --out runs/masc_deb_aadg_s49

echo "=== [5/5] diagnose + assemble ==="
MASC=""
for d in runs/masc_twrob_s42 runs/masc_btw_s43 runs/masc_deb_s44 runs/masc_btwL_s45 \
         runs/masc_robL_s46 runs/pdq_s42 runs/pdq_twrob_s43 runs/pdq_btw_s42 \
         runs/pdq_btwL_s47 runs/masc_btwL_aadg_s48 runs/masc_deb_aadg_s49; do
  [ -f "$d/probs_test.npz" ] && MASC="$MASC $d"
done
echo "members:$MASC"
python3 experts/diagnose.py --masc $MASC
MATE="runs/mate_deb_s42 runs/mate_deb_s43 runs/mate_deb_s44 runs/mate_probeA runs/mate_probeB"
python3 experts/assemble.py --mate $MATE --masc $MASC --mode span  --out results/q8_span.json
python3 experts/assemble.py --mate $MATE --masc $MASC --mode joint --out results/q8_joint.json
echo "=== QUEUE8 DONE ==="
