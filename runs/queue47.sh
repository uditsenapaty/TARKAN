#!/usr/bin/env bash
# D26 — CET evidence swapping. t2015 ONLY.
#
# Feeding an aspect its SIBLING's evidence must not move its decision: evidence belongs to
# an ASPECT, not to the tweet. §C13 (sibling-logit) acted on the final logits and §C10
# (sibling marking) on the input text -- both weak/negative. This corrupts the
# evidence-aspect BINDING, which nothing has tested. The intervention is real: across 400
# multi-aspect tweets the AADG visual vector differs between aspects in 304 and the
# aspect-image similarity u in 393, and 799 of 2101 train tweets have >1 aspect.
#
# NOT run: CET-1 (remove evidence, measure delta). On this additive architecture
# lg_full - lg_novis ~ scale*g*delta, which is exactly the quantity §C.27's signed-margin
# loss already constrains -- it would re-measure PDS.
#
# Baselines are the §C.27 caption-teacher PDS members on the identical recipe:
#   bertweet-large 78.50 | deberta-v3-large 76.76 | roberta-large 79.27
# Three towers rather than three seeds: a mechanism about evidence binding should show up
# across architectures, and §D.20 puts the single-pair floor at +/-1.31.
set -u
S=runs/mate_ens5_hr
run(){ n=$1; m=$2; w=$3; echo "=== $n ($m, cet-swap $w) ==="
  python3 -u experts/masc_pds.py --residual --w-none 0 --cet-swap $w --model "$m" \
      --spans $S --out runs/$n 2>&1 | grep -v "it/s\]" | grep -aE "^ep[0-9]|gold-span"; }
run d26_cet_btwL vinai/bertweet-large       0.1
run d26_cet_deb  microsoft/deberta-v3-large 0.1
run d26_cet_robL roberta-large              0.1
echo "=== QUEUE47 DONE ==="
