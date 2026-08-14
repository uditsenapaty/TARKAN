#!/usr/bin/env bash
# Matched baselines for the deberta / roberta CER towers. Without these the CER numbers on
# those towers are uninterpretable -- only bertweet-large had a same-recipe control.
set -u
S=runs/mate_ens5_hr
run(){ n=$1; m=$2; echo "=== $n ==="
  python3 -u experts/masc_text.py --model $m --seed 45 --epochs 6 --spans $S \
      --out runs/$n 2>&1 | grep -v "it/s\]" | grep -aE "gold-span"; }
run d15_base_deb  microsoft/deberta-v3-large
run d15_base_robL roberta-large
echo "=== QUEUE37 DONE ==="
