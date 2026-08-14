#!/usr/bin/env bash
# D22b — PACS with the PAIRWISE hard-negative margin (the mean-based version was vacuous).
# Dose-response on the coupling weight rather than a single point: if a_selected moves
# monotonically with lam_joint, that is signal; if it scatters, it is noise. Same GPU cost as
# paired seeds and a stronger test.
# Control (already measured, unaffected by the fix): MATE@tau 84.83, a_selected 76.91,
# joint 65.24 at lam_joint 0.
set -u
run(){ n=$1; lj=$2; echo "=== $n (lam_joint $lj) ==="
  python3 -u experts/pacs.py --seed 42 --epochs 16 --lam-joint $lj --out runs/$n 2>&1 \
    | grep -v "it/s\]" | grep -aE "^ep(4|8|12|16) |MATE@tau|GATE"; }
run d22b_pacs_lj05 0.5
run d22b_pacs_lj20 2.0
echo "=== QUEUE43 DONE ==="
