#!/bin/bash
set -e
cd /teamspace/studios/this_studio
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
S=runs/mate_ens5_hr
echo "=== [1/4] PDS w_none=0.1 (direction-only supervision), bertweet-large ==="
python3 experts/masc_pds.py --model vinai/bertweet-large --seed 72 --lambda-pds 0.5 \
  --w-none 0.1 --spans $S --out runs/masc_pds_btwL_wn01
echo "=== [2/4] PDS w_none=0.1, deberta-v3-large ==="
python3 experts/masc_pds.py --model microsoft/deberta-v3-large --seed 73 --lambda-pds 0.5 \
  --w-none 0.1 --spans $S --out runs/masc_pds_deb_wn01
echo "=== [3/4] more ITC/ITM (it ADDS: +0.33): deberta-v3-large tower ==="
python3 experts/pdq.py --text-model microsoft/deberta-v3-large --itc 1.0 --itm 1.0 --seed 62 \
  --lr 1e-5 --epochs 8 --spans $S --out runs/pdq_deb_itcitm
echo "=== [4/4] more ITC/ITM: bertweet-large seed 63 ==="
python3 experts/pdq.py --text-model vinai/bertweet-large --itc 1.0 --itm 1.0 --seed 63 \
  --lr 1e-5 --epochs 8 --spans $S --out runs/pdq_btwL_itcitm2
echo "=== QUEUE21 DONE ==="
