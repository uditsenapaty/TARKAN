#!/usr/bin/env bash
# D30b — rebuild AADG on the Qwen descriptions, then retrain the members on the new evidence.
# ONLY the description source changes. Dual similarity, greedy one-to-one matching, the
# TRAIN-ECDF u, the calibrated gate and the residual are all untouched -- so this isolates
# exactly the axis MADSC's ablation prices at -1.1 for BLIP2 (71.8) against GPT-4o (72.9).
set -u
echo "=== [1] AADG describe on Qwen captions ==="
python3 -u experts/aadg.py --stage describe --captions captions_qwen.json --suffix _qwen \
    --spans runs/mate_ens5_hr 2>&1 | grep -v "it/s\]" | grep -aE "description source|desc \||grounded"

echo "=== [2] PDS members on the Qwen evidence (baselines: 78.50 / 76.76 / 79.27) ==="
for m in "q30_pds_btwL vinai/bertweet-large" \
         "q30_pds_deb  microsoft/deberta-v3-large" \
         "q30_pds_robL roberta-large"; do
  set -- $m
  echo "--- $1"
  python3 -u experts/masc_pds.py --residual --w-none 0 --model "$2" \
      --desc data/aadg/twitter2015 --aspect-suffix _qwen \
      --spans runs/mate_ens5_hr --out runs/$1 2>&1 | grep -v "it/s\]" | grep -aE "gold-span"
done
echo "=== QUEUE53 DONE ==="
