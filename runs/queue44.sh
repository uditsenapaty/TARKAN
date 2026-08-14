#!/usr/bin/env bash
# D22c — PACS with DETERMINABILITY RANKING, the direct §C.18 fix.
#
# §C.18: MASC scores 80.52 on MATE-extracted gold spans but 86.73 on MATE-MISSED ones -- the
# extractor prefers the spans its classifier cannot handle. The fix is arithmetically clean:
# extracting a gold span whose polarity will be wrong yields 1 FP + 1 FN, skipping it yields
# only the FN, so de-prioritising undeterminable spans is STRICTLY beneficial. That is D.4's
# "drop 72 false positives" as representation learning rather than the post-hoc selection
# §D.2 measured as impossible even when fitted on TEST.
#
# Determinability comes from 4 independent OOF towers; 356 of 3179 train aspects (11.2%)
# have none correct. Boundary contrast (lam_joint) attacks the smaller pot -- §D.1 found only
# 125 of 1032 kept predictions are non-gold spans -- so it is tested as an add-on, not alone.
#
# Control (measured, runs/d22_pacs_nojoint): MATE@tau 84.83, a_selected 76.91, joint 65.24.
# GATE: a_selected vs 76.91. NOT the 80.52 the script prints -- that is the 19-member MASC
# ensemble's figure and PACS is a single model.
set -u
run(){ n=$1; shift; echo "=== $n ($*) ==="
  python3 -u experts/pacs.py --seed 42 --epochs 16 "$@" --out runs/$n 2>&1 \
    | grep -v "it/s\]" | grep -aE "determinability buffer|^ep(4|8|12|16) |MATE@tau|GATE"; }
run d22c_det      --lam-det 1.0 --lam-joint 0.0
run d22c_det_both --lam-det 1.0 --lam-joint 0.5
echo "=== QUEUE44 DONE ==="
