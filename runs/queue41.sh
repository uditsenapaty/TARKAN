#!/usr/bin/env bash
# D21 — architectural diversity on the EXTRACTION side, which has never been tested.
# All 5 MATE ensemble members are deberta-v3-large, varying only by seed/recipe, while the
# polarity side carries 7 distinct backbones. §C.1 concluded "more seeds is the wrong lever,
# the recipe is" and stopped -- but a different ARCHITECTURE is neither a seed nor a recipe,
# and this project's own principle (masc_text: "diversity, not individual strength, is what
# they contribute") was applied to polarity and never to extraction.
# Recipe is the §C.6 winner: lr 1e-5, head-lr 1e-4, dropout 0.3, 16 epochs.
set -u
run(){ n=$1; m=$2; echo "=== $n ($m) ==="
  python3 -u experts/mate_expert.py --model "$m" --seed 42 --epochs 16 \
      --lr 1e-5 --head-lr 1e-4 --dropout 0.3 --out runs/$n 2>&1 \
      | grep -v "it/s\]" | grep -aE "^ep1?[0-9] |MATE P|test"; }
run mate_btwL vinai/bertweet-large
run mate_robL roberta-large
echo "=== QUEUE41 DONE ==="
