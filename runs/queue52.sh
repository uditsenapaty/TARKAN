#!/usr/bin/env bash
# D30 — swap the description source: BLIP -> Qwen2.5-VL reading the ORIGINAL image.
# MADSC's own ablation on t2015 JMASA: GPT-4o 72.9, LLaVA 72.1, BLIP2 71.8. Every
# aspect-aware description in this repo is built from BLIP captions, i.e. their weakest
# configuration, priced at -1.1 in their table. Nothing downstream changes -- aadg.py
# consumes captions.json and masc_gated/masc_pds consume the vis/u it produces.
set -u
echo "=== [1] Qwen2.5-VL descriptions for every referenced image ==="
python3 -u experts/qwen_describe.py --dataset twitter2015 --batch 8 --max-new 44 2>&1 \
  | grep -v "it/s\]" | grep -aE "images to describe|s/img|wrote|\.jpg:"
echo "=== QUEUE52 DONE ==="
