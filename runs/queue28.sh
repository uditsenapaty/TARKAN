#!/usr/bin/env bash
# D3 feasibility on the T4 before committing hours to Qwen2.5-VL-7B.
# (1) does a 4-bit + frozen-vision + LoRA training step fit and how slow is it,
# (2) does the counterfactual teacher path run end to end and does the image actually
#     move the distribution (if delta ~ 0 everywhere the whole idea is dead and we stop).
set -u
echo "=== [1] Qwen2.5-VL training-step benchmark ==="
python3 -u experts/masc_qwenvl.py --out runs/qwen_bench --limit-steps 6 \
    --batch 1 --accum 8 2>&1 | grep -v "it/s\]" | tail -25

echo "=== [2] Qwen2.5-VL counterfactual smoke test (40 aspects) ==="
python3 -u experts/masc_qwenvl.py --counterfactual --split train --limit 40 \
    --out runs/qwen_cf_smoke 2>&1 | grep -v "it/s\]" | tail -15
echo "=== QUEUE28 DONE ==="
