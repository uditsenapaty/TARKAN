#!/usr/bin/env bash
# D19 — Annotation-Policy Separation, t2015 ONLY. No t2017 anywhere: t2015 and t2017 are
# separate experiments and t2017 must stay clean for its own run.
#
# Because no independent set is available, the substitute control is SEEDS: three paired
# baseline/APS runs on the same tower. §D.18's lesson was that same-sign draws on one test
# set are weak evidence when the draws are correlated; independent seeds at least separate
# seed variance from effect, which a single pair cannot.
set -u
S=runs/mate_ens5_hr
M=vinai/bertweet-large
run(){ n=$1; shift; echo "=== $n ==="
  python3 -u experts/masc_text.py --model $M --epochs 6 --spans $S "$@" \
      --out runs/$n 2>&1 | grep -v "it/s\]" | grep -aE "APS agreement|gold-span"; }
for sd in 45 46 47; do
  run d19_base_s$sd --seed $sd
  run d19_aps_s$sd  --seed $sd --aps
done
echo "=== QUEUE39 DONE ==="
