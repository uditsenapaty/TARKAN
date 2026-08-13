#!/usr/bin/env bash
# D3b — is the Qwen-VL teacher WORSE than the caption teacher because it is noisier?
# 76% of aspects "moved" at --shift-floor 0.05, which is likely Qwen reacting to an image
# being present rather than to aspect-specific sentiment evidence. Re-run the teacher once
# (now caching the raw arms), then re-threshold for free and retrain the SAME tower.
set -u
echo "=== [1] teacher again, caching p_img/p_txt ==="
python3 -u experts/masc_qwenvl.py --counterfactual --split train \
    --out data/pds_qwen/twitter2015 2>&1 | grep -v "it/s\]" | tail -5

for f in 0.20 0.35; do
  echo "=== [2] remap at floor $f ==="
  python3 -u experts/remap_direction.py --src data/pds_qwen/twitter2015 \
      --floor $f --out data/pds_qwen_f${f/./}/twitter2015
done

for f in 020 035; do
  echo "=== [3] PDS bertweet-large on floor 0.${f:1} directions ==="
  python3 -u experts/masc_pds.py --residual --w-none 0 \
      --pds data/pds_qwen_f${f}/twitter2015 --model vinai/bertweet-large \
      --spans runs/mate_ens5_hr --out runs/qpds_btwL_f${f} 2>&1 | grep -v "it/s\]" | tail -12
done
echo "=== QUEUE30 DONE ==="
