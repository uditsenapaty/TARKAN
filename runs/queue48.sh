#!/usr/bin/env bash
# D27 — ASOE: Aspect-Signed Evidence Ownership. t2015 ONLY.
#
# Every visual experiment here asked "is the image useful?" and measured ~0. ASOE asks which
# visual evidence is OWNED by this aspect and whether the decision actually NEEDS it. That was
# previously unaskable: every prior member compressed the image to ONE pooled vector, so there
# was no set to route over. This routes over the cached EVA-ViT-g 256-patch grid.
#
#   sufficiency : the routed evidence must RAISE the gold margin (weighted by the teacher's
#                 1 - P(no-shift)); "select evidence the classifier needs, not evidence that
#                 merely looks relevant". Never supervised before in this project.
#   ownership   : evidence routed by a SIBLING aspect must not move this aspect's decision.
#                 Siblings share the image, so this is free. Applied to the RAW delta, not
#                 alpha*delta -- gating it is satisfiable by alpha -> 0, which a smoke run
#                 did (0.100 -> 0.003) before the fix.
#   separation  : ownership maps of different-polarity siblings must diverge (236 such pairs
#                 seen per epoch under instance batching).
#
# Baselines: the §C.27 PDS members on the identical recipe -- 78.50 / 76.76 / 79.27.
# Three towers, because §D.26 showed a single tower can read +0.87 while the mean is -0.20.
set -u
S=runs/mate_ens5_hr
run(){ n=$1; m=$2; echo "=== $n ($m) ==="
  python3 -u experts/asoe.py --model "$m" --spans $S --epochs 8 \
      --lam-suf 0.3 --lam-own 0.3 --lam-sep 0.1 --out runs/$n 2>&1 \
      | grep -v "it/s\]" | grep -aE "^ep[0-9]|gold-span"; }
run d27_asoe_btwL vinai/bertweet-large
run d27_asoe_deb  microsoft/deberta-v3-large
run d27_asoe_robL roberta-large
echo "=== QUEUE48 DONE ==="
