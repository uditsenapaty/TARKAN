#!/bin/bash
set -e
cd /teamspace/studios/this_studio
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
S=runs/mate_ens5_hr
echo "=== [1/5] PDS residual, bertweet-large, w_none 0.05 ==="
python3 experts/masc_pds.py --residual --model vinai/bertweet-large --seed 80 \
  --lambda-pds 0.5 --w-none 0.05 --spans $S --out runs/pds_res_btwL
echo "=== [2/5] PDS residual, deberta-v3-large, w_none 0.05 ==="
python3 experts/masc_pds.py --residual --model microsoft/deberta-v3-large --seed 81 \
  --lambda-pds 0.5 --w-none 0.05 --spans $S --out runs/pds_res_deb
echo "=== [3/5] CONTROL: w_none 0 ==="
python3 experts/masc_pds.py --residual --model vinai/bertweet-large --seed 82 \
  --lambda-pds 0.5 --w-none 0.0 --spans $S --out runs/pds_res_btwL_wn0
echo "=== [4/5] ITC/ITM tower: deberta-v3-large ==="
python3 experts/pdq.py --text-model microsoft/deberta-v3-large --itc 1.0 --itm 1.0 --seed 62 \
  --lr 1e-5 --epochs 8 --spans $S --out runs/pdq_deb_itcitm
echo "=== [5/5] ITC/ITM tower: bertweet-large seed 63 ==="
python3 experts/pdq.py --text-model vinai/bertweet-large --itc 1.0 --itm 1.0 --seed 63 \
  --lr 1e-5 --epochs 8 --spans $S --out runs/pdq_btwL_itcitm2
echo "=== QUEUE23 DONE ==="
