#!/usr/bin/env bash
# D — re-score the trained 8B member with option-order TTA into a SEPARATE member dir,
# so the plain and TTA variants can both be measured (and both entered as ensemble
# members if they turn out to be usefully decorrelated). Training is not repeated.
set -u
python3 -u experts/masc_llm.py --score-only \
    --adapter runs/masc_llm_s42/adapter \
    --tta --spans runs/mate_union --out runs/masc_llm_s42_tta 2>&1 | grep -v "it/s\]"
echo "=== QUEUE27 DONE ==="
