#!/usr/bin/env bash
# D25 — TORF: Target-Opinion Relational Fusion. t2015 ONLY.
#
# §D.1: 158 of 186 polarity errors are minority<->NEU. §D.20's TBRF gave the head a
# target-conditioned context vector and was FLAT (-0.07 over 3 paired seeds), so "what is
# near this aspect" is not the missing piece. TORF changes what is different: a learned
# per-token opinion score s_j splits the target-conditioned attention BY SIGN into
#   E+ ~ exp(r_j)*sigmoid( s_j)   and   E- ~ exp(r_j)*sigmoid(-s_j)
# and the head receives [t_a, E+, E-, E+ - E-], so the POS/NEG decision has an explicit
# directional contrast rather than having to recover one from a single pooled vector.
# s_j is LEARNED -- §C14 measured that hard SenticNet opinion manipulation hurts.
#
# Paired seeds against the SAME baselines used for TBRF (§D.20): s45 78.69, s46 78.50,
# s47 78.50. §D.20 put the single-pair detection floor at +/-1.31, so the paired mean is
# the number that counts.
set -u
S=runs/mate_ens5_hr
M=vinai/bertweet-large
for sd in 45 46 47; do
  echo "=== d25_torf_s$sd ==="
  python3 -u experts/masc_text.py --model $M --seed $sd --epochs 6 --spans $S --torf \
      --out runs/d25_torf_s$sd 2>&1 | grep -v "it/s\]" | grep -aE "^ep[0-9]|gold-span"
done
echo "=== QUEUE46 DONE ==="
