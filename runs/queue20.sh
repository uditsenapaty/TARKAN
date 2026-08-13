#!/bin/bash
set -e
cd /teamspace/studios/this_studio
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
echo "=== [1/3] PDS teacher labelling (Llama-3.1-8B 4-bit, scored not generated) ==="
python3 experts/pds_teacher.py --split train --batch 4
echo "=== [2/3] PDS student: bertweet-large ==="
python3 experts/masc_pds.py --model vinai/bertweet-large --seed 70 --lambda-pds 0.5 \
  --spans runs/mate_ens5_hr --out runs/masc_pds_btwL
echo "=== [3/3] PDS student: deberta-v3-large ==="
python3 experts/masc_pds.py --model microsoft/deberta-v3-large --seed 71 --lambda-pds 0.5 \
  --spans runs/mate_ens5_hr --out runs/masc_pds_deb
echo "=== QUEUE20 DONE ==="
