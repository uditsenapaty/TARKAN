#!/usr/bin/env bash
# D29 — DUAL-ANCHOR TARKAN. t2015 ONLY.
#
# TARKAN anchors the ASPECT but never the OPINION, so nothing in the architecture answers
# "which sentiment cue belongs to THIS target". §D.1 puts 158 of 186 errors at minority<->NEU,
# the signature of a model reading the tweet's overall tone. Crucially §D.28 showed image
# grounding is LEARNABLE but POLARITY-IRRELEVANT -- so ASOE routed ownership over the wrong
# modality. These errors are text-driven, so ownership belongs over TEXT tokens.
#
#   DUAL-1  opinion salience o_j + signed direction d_j + relation MLP over
#           [h_a, h_j, h_a*h_j, |h_a-h_j|] -> ownership alpha -> e_a, D_a
#   DUAL-2  + cross-aspect COMPETITION: each salient token owned by ONE aspect of the tweet.
#           §C13 acted on final logits, §C10 on the input text; this acts on the evidence
#           ATTRIBUTION itself, using the 444 different-polarity sibling pairs.
#
# Paired seeds against the same baselines TBRF/TORF used: 78.69 / 78.50 / 78.50.
set -u
S=runs/mate_ens5_hr
M=vinai/bertweet-large
for sd in 45 46 47; do
  echo "=== d29_dual_s$sd ==="
  python3 -u experts/masc_text.py --model $M --seed $sd --epochs 6 --spans $S --dual \
      --out runs/d29_dual_s$sd 2>&1 | grep -v "it/s\]" | grep -aE "gold-span"
done
for sd in 45 46 47; do
  echo "=== d29_dualown_s$sd ==="
  python3 -u experts/masc_text.py --model $M --seed $sd --epochs 6 --spans $S --dual \
      --dual-own 0.1 --out runs/d29_dualown_s$sd 2>&1 | grep -v "it/s\]" | grep -aE "gold-span"
done
echo "=== QUEUE51 DONE ==="
