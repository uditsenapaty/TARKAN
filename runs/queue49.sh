#!/usr/bin/env bash
# D28 — TTP: TARKAN Task-Pretraining. t2015 ONLY.
#
# §D.27 measured that evidence mechanisms have NEGATIVE marginal value above a baseline of
# ~77.93 (delta = -0.868*baseline + 67.67, r = -0.981) and our towers sit above it. So stop
# adding mechanisms on top of a strong base and try to move the BASE instead.
#
# Stage 1 uses NO polarity labels: aspect-conditioned contrastive alignment between t_a and
# its own routed image evidence E_a. Ordinary tweet-image alignment cannot express this --
# two aspects of one tweet share an image and would be positives for each other -- but E_a is
# aspect-routed and instance batching puts siblings in the SAME batch, making them hard
# negatives and forcing the routing to be aspect-discriminative.
# Stage 2 is ordinary supervised training with all mechanism weights at ZERO, so the only
# difference between the arms is the pretraining.
#
# Paired seeds; §D.20 puts the single-pair detection floor at +/-1.31.
set -u
S=runs/mate_ens5_hr
M=vinai/bertweet-large
run(){ n=$1; shift; echo "=== $n ($*) ==="
  python3 -u experts/asoe.py --model $M --spans $S --epochs 8 \
      --lam-suf 0 --lam-own 0 --lam-sep 0 "$@" --out runs/$n 2>&1 \
      | grep -v "it/s\]" | grep -aE "\[TTP\]|^ep(2|4|6|8) |gold-span"; }
for sd in 45 46; do
  run d28_ctrl_s$sd --seed $sd --ttp-epochs 0
  run d28_ttp_s$sd  --seed $sd --ttp-epochs 4
done
echo "=== QUEUE49 DONE ==="
