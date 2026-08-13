#!/usr/bin/env bash
# D12 — Qwen2.5-VL as a DIRECT offline evidence teacher on the original image.
# The missing cell of the 2x2: ask-directly x pixels. §C.25 asked Llama directly but it
# only ever saw a BLIP caption (+0.26, members 78.50); §D.9/D.10 gave Qwen the real pixels
# but estimated the shift by DIFFERENCING two noisy arms (-1.1 per member) and concluded
# that asking outright beat differencing. This does both right things at once.
set -u
echo "=== [1] direct image teacher, full TRAIN split ==="
python3 -u experts/masc_qwenvl.py --direct-teacher --split train \
    --out data/pds_qdirect/twitter2015 2>&1 | grep -v "it/s\]" | tail -5

for m in "qdpds_btwL vinai/bertweet-large" \
         "qdpds_deb microsoft/deberta-v3-large" \
         "qdpds_robL roberta-large"; do
  set -- $m
  echo "=== [2] PDS member on DIRECT image directions: $1 ==="
  python3 -u experts/masc_pds.py --residual --w-none 0 --pds data/pds_qdirect/twitter2015 \
      --model "$2" --spans runs/mate_ens5_hr --out runs/$1 2>&1 | grep -v "it/s\]" | tail -11
done
echo "=== QUEUE31 DONE ==="

# D.9/D.10 CONFOUND CHECK: those members were trained with w_pos=0.147, the inverse
# frequency of the CAPTION teacher's 687:101, while their own teacher was 1394:644
# (w_pos 0.462). Re-run one tower with the correct auto-derived weights so the -1.1 is
# attributable to the labels rather than to a mis-weighted loss.
echo "=== [3] counterfactual teacher, CORRECT weights (confound check) ==="
python3 -u experts/masc_pds.py --residual --w-none 0 --pds data/pds_qwen/twitter2015 \
    --model vinai/bertweet-large --spans runs/mate_ens5_hr --out runs/qpds_btwL_bal \
    2>&1 | grep -v "it/s\]" | tail -11
echo "=== QUEUE31B DONE ==="
