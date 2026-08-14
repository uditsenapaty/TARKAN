#!/usr/bin/env bash
# D23 — train P(pair correct) from TEXT, then use it as the acceptance score.
# Oracle projection with true q: F1 74.68 at the cut q < F1/2 = 0.353. This measures how much
# of that is reachable when q must be predicted from the input instead of known.
set -u
echo "=== qpredict (bertweet-large on OOF MATE candidates) ==="
python3 -u experts/qpredict.py --model vinai/bertweet-large --epochs 4 \
    --out runs/qpred_btwL 2>&1 | grep -v "it/s\]" | grep -aE "train candidates|^ep|scored"
for mix in 1.0 0.5; do
  echo "=== decide with q_hat, mix $mix ==="
  python3 -u experts/decide.py --pool pools/final19 --w-grid 0.0 \
      --qhat runs/qpred_btwL --qhat-mix $mix --out results/qhat_$mix.json 2>&1 | sed -n '2p;7,10p'
done
echo "=== QUEUE45 DONE ==="
