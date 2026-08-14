#!/usr/bin/env bash
# D20 — TBRF: Target-Background Residual Fusion. t2015 only. No annotation head, no t2017,
# no extra members, no teacher calls. One architectural change to the aspect representation.
#
# Before: features = [mean_pool(seq), max_pool(seq)] -- the aspect span is never used, it
#         enters only as bracket characters, so nothing stops the model answering "what is
#         this tweet's mood".
# After : the aspect term QUERIES the tweet by attention (no fixed window, so negation and
#         long-range modifiers stay reachable), and the head also receives
#         t_local - t_global: what is special about this aspect vs the tweet's overall tone.
#
# Paired seeds, because §D.18 showed a single test-set comparison cannot separate an effect
# from seed variance.
set -u
S=runs/mate_ens5_hr
M=vinai/bertweet-large
run(){ n=$1; shift; echo "=== $n ==="
  python3 -u experts/masc_text.py --model $M --epochs 6 --spans $S "$@" \
      --out runs/$n 2>&1 | grep -v "it/s\]" | grep -aE "gold-span"; }
run d20_tbrf_s45 --seed 45 --tbrf
run d20_base_s46 --seed 46
run d20_tbrf_s46 --seed 46 --tbrf
run d20_base_s47 --seed 47
run d20_tbrf_s47 --seed 47 --tbrf
echo "=== QUEUE40 DONE ==="
