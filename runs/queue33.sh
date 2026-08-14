#!/usr/bin/env bash
# D15 — the two mechanisms that attack §D.11 directly, on a strong tower, controlled.
#   factorized : P(non-neutral) x P(POS|non-neutral) -- changes the DECISION GEOMETRY,
#                unlike §C.9 class weighting and §C.12 margins which stayed inside the
#                flat 3-way softmax and both failed.
#   CER        : train against the family's own confident failures (653 OOF cases whose
#                signature matches the test failures: 0.65/4 members right vs 3.62/4,
#                347/653 collapsing to NEU, NEG worst).
set -u
M=vinai/bertweet-large
S=runs/mate_ens5_hr
run(){ n=$1; shift; echo "=== $n ==="
  python3 -u experts/masc_text.py --model $M --seed 45 --epochs 6 --spans $S \
      "$@" --out runs/$n 2>&1 | grep -v "it/s\]" | grep -aE "^ep[0-9]|CER buffer|gold-span"; }
run d15_base
run d15_fact  --factorized
run d15_cer   --cer 0.5 --cer-upweight 2.0
run d15_both  --factorized --cer 0.5 --cer-upweight 2.0
echo "=== QUEUE33 DONE ==="
