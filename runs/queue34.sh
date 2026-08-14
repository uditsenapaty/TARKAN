#!/usr/bin/env bash
# Fair test of CER before declaring it dead: the first run used untuned weights
# (margin 0.5 + replay 2.0) and lost 4.34. Separate the two components, both weak.
set -u
M=vinai/bertweet-large
S=runs/mate_ens5_hr
run(){ n=$1; shift; echo "=== $n ==="
  python3 -u experts/masc_text.py --model $M --seed 45 --epochs 6 --spans $S \
      "$@" --out runs/$n 2>&1 | grep -v "it/s\]" | grep -aE "gold-span"; }
run d15_cer_m05  --cer 0.05                      # margin only, 10x weaker
run d15_cer_r13  --cer 0.0 --cer-upweight 1.3    # replay only, mild
run d15_cer_c95  --cer 0.05 --cer-conf 0.95      # margin only, hardest failures only
echo "=== QUEUE34 DONE ==="
