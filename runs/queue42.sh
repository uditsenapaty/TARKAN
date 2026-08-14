#!/usr/bin/env bash
# D22 — PACS. The ablation IS the experiment: lam_joint 0 vs 0.5, everything else identical.
#   lam_joint 0.0 : shared encoder, two heads, NO joint margin  (control)
#   lam_joint 0.5 : + U(gold,y*) > U(hard_neg,y_hat) + m        (the coupling claim)
# Gate is a_selected (the §C.18 quantity), not joint F1. PACS is ONE model, so it must not be
# compared against the 5-MATE + 19-MASC ensemble's 70.62 -- this ablation is the fair control.
set -u
run(){ n=$1; lj=$2; echo "=== $n (lam_joint $lj) ==="
  python3 -u experts/pacs.py --seed 42 --epochs 16 --lam-joint $lj --out runs/$n 2>&1 \
    | grep -v "it/s\]" | grep -aE "^ep(4|8|12|16) |MATE@tau|GATE"; }
run d22_pacs_nojoint 0.0
run d22_pacs_joint   0.5
echo "=== QUEUE42 DONE ==="
