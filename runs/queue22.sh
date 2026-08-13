#!/bin/bash
# Queue 22 — PDS done properly, TARKAN-native (nothing imported from another paper).
# Three fixes, each aimed at a measured defect of the first attempt:
#  (1) RESIDUAL, zero-init  -> starts exactly at the strong plain classifier (79.75), so the
#      6.5%-unique-right PDS signal can only CORRECT it, never replace it.
#  (2) SIGNED MARGIN on the decision, not L2 on the representation -> evidence may change the
#      representation freely; only the SIGN of its effect on the polarity margin is constrained.
#  (3) POS/NEG inverse-frequency weighting INSIDE the auxiliary loss only (teacher gives
#      687 POS vs 101 NEG) -> stops PDS teaching "images make things positive".
#      The MASC classifier itself is never class-weighted.
set -e
cd /teamspace/studios/this_studio
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
S=runs/mate_ens5_hr
echo "=== [1/3] PDS residual, bertweet-large, w_none 0.05 ==="
python3 experts/masc_pds.py --residual --model vinai/bertweet-large --seed 80 \
  --lambda-pds 0.5 --w-none 0.05 --spans $S --out runs/pds_res_btwL
echo "=== [2/3] PDS residual, deberta-v3-large, w_none 0.05 ==="
python3 experts/masc_pds.py --residual --model microsoft/deberta-v3-large --seed 81 \
  --lambda-pds 0.5 --w-none 0.05 --spans $S --out runs/pds_res_deb
echo "=== [3/3] CONTROL: w_none 0 (is ANY no-shift supervision useful?) ==="
python3 experts/masc_pds.py --residual --model vinai/bertweet-large --seed 82 \
  --lambda-pds 0.5 --w-none 0.0 --spans $S --out runs/pds_res_btwL_wn0
echo "=== QUEUE22 DONE ==="
