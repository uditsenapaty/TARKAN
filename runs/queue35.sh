#!/usr/bin/env bash
# The margin-only CER at 0.05 is POSITIVE and dev agrees (dev 78.79 / test 79.27 vs
# baseline 77.45 / 78.69). Map the weight, re-run the replay arm that the gating bug
# silently skipped, and build the other two towers so the ensemble can be tested.
set -u
S=runs/mate_ens5_hr
run(){ n=$1; m=$2; shift 2; echo "=== $n ==="
  python3 -u experts/masc_text.py --model $m --seed 45 --epochs 6 --spans $S \
      "$@" --out runs/$n 2>&1 | grep -v "it/s\]" | grep -aE "gold-span"; }
B=vinai/bertweet-large
run d15_cer_m02  $B --cer 0.02
run d15_cer_m10  $B --cer 0.10
run d15_cer_r13b $B --cer 0.0 --cer-upweight 1.3     # the arm the gating bug skipped
run d15_cer_deb  microsoft/deberta-v3-large --cer 0.05
run d15_cer_robL roberta-large --cer 0.05
echo "=== QUEUE35 DONE ==="
