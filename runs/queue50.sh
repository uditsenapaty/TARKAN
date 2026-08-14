#!/usr/bin/env bash
# D28b — TTP with the encoder FROZEN during stage 1.
# The unfrozen version collapsed -19.8 (58.05 vs a 77.82 matched control) because the
# contrastive task is barely learnable: InfoNCE starts at chance (ln(8)=2.08) and plateaus
# at 1.63, only ~0.45 nats below it. Optimising the encoder against a near-unlearnable
# target wrecks the text representation. Freezing lets the ROUTING learn whatever
# aspect-image alignment exists without paying that cost -- the correct form of the test.
set -u
python3 -u experts/asoe.py --model vinai/bertweet-large --spans runs/mate_ens5_hr \
    --epochs 8 --lam-suf 0 --lam-own 0 --lam-sep 0 --seed 45 \
    --ttp-epochs 4 --ttp-freeze --out runs/d28_ttpfz_s45 2>&1 \
    | grep -v "it/s\]" | grep -aE "\[TTP\]|^ep(2|4|6|8) |gold-span"
echo "=== QUEUE50 DONE ==="
