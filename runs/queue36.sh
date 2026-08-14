#!/usr/bin/env bash
# D16 — RER: replay only RECOVERABLE failures, optionally weighted by 4c(1-c).
# Measured recoverability by consensus confidence: 92-93% below 0.90, but only 26% above
# 0.95 -- and those 425 ultra-confident cases are the MAJORITY of all 750 failures. So
# margin 0.5's -4.34 collapse was replaying label noise, and conf>0.95 filtering
# underperformed conf>0.7 because it selected precisely that noise.
set -u
S=runs/mate_ens5_hr
B=vinai/bertweet-large
run(){ n=$1; m=$2; shift 2; echo "=== $n ==="
  python3 -u experts/masc_text.py --model $m --seed 45 --epochs 6 --spans $S \
      "$@" --out runs/$n 2>&1 | grep -v "it/s\]" | grep -aE "CER buffer|gold-span"; }
run d16_rer      $B --cer 0.05 --cer-recoverable
run d16_rer_k    $B --cer 0.05 --cer-recoverable --cer-kernel
run d16_rer_k10  $B --cer 0.10 --cer-recoverable --cer-kernel
run d16_rer_rep  $B --cer 0.05 --cer-recoverable --cer-kernel --cer-upweight 1.3
echo "=== QUEUE36 DONE ==="
