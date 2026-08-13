#!/bin/bash
# Queue 24 — scale the TWO families that are now demonstrably productive.
#   ITC/ITM  : weaker standalone, adds to the ensemble (unique-right up to 7.33%)
#   PDS-res  : the residual + signed-margin + POS/NEG-balanced fix turned PDS from
#              -0.09 into +0.26, and w_none=0 beat w_none=0.05, so no-shift supervision
#              is harmful rather than merely over-weighted -> use 0 everywhere now.
set -e
cd /teamspace/studios/this_studio
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
S=runs/mate_ens5_hr
echo "=== [1/6] PDS-res w_none 0, deberta-v3-large ==="
python3 experts/masc_pds.py --residual --model microsoft/deberta-v3-large --seed 83 \
  --lambda-pds 0.5 --w-none 0.0 --spans $S --out runs/pds_res_deb_wn0
echo "=== [2/6] PDS-res w_none 0, twitter-roberta ==="
python3 experts/masc_pds.py --residual --model cardiffnlp/twitter-roberta-base-sentiment-latest \
  --seed 84 --lambda-pds 0.5 --w-none 0.0 --lr 2e-5 --spans $S --out runs/pds_res_twrob_wn0
echo "=== [3/6] PDS-res w_none 0, roberta-large ==="
python3 experts/masc_pds.py --residual --model roberta-large --seed 85 \
  --lambda-pds 0.5 --w-none 0.0 --spans $S --out runs/pds_res_robL_wn0
echo "=== [4/6] PDS-res stronger lambda (1.0), bertweet-large ==="
python3 experts/masc_pds.py --residual --model vinai/bertweet-large --seed 86 \
  --lambda-pds 1.0 --w-none 0.0 --spans $S --out runs/pds_res_btwL_l1
echo "=== [5/6] ITC/ITM tower: roberta-large ==="
python3 experts/pdq.py --text-model roberta-large --itc 1.0 --itm 1.0 --seed 64 \
  --lr 1e-5 --epochs 8 --spans $S --out runs/pdq_robL_itcitm
echo "=== [6/6] ITC/ITM tower: bertweet-base (cheap, diverse) ==="
python3 experts/pdq.py --text-model vinai/bertweet-base --itc 1.0 --itm 1.0 --seed 65 \
  --lr 2e-5 --epochs 8 --spans $S --out runs/pdq_btw_itcitm
echo "=== QUEUE24 DONE ==="
