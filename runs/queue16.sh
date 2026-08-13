#!/bin/bash
set -e
cd /teamspace/studios/this_studio
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
S=runs/mate_ens5_hr
echo "=== re-score remaining members on the high-recall candidate set ==="
python3 experts/masc_gated.py --model vinai/bertweet-large --fuse concat --score-only --desc data/aadg/twitter2015 --spans $S --out runs/masc_gate_btwL_cat
python3 experts/masc_gated.py --model vinai/bertweet-large --fuse convex --score-only --desc data/aadg/twitter2015 --spans $S --out runs/masc_gate_btwL_cvx
python3 experts/masc_gated.py --model microsoft/deberta-v3-large --fuse concat --score-only --desc data/aadg/twitter2015 --spans $S --out runs/masc_gate_deb_cat
python3 experts/pdq.py --text-model bert-base-uncased --score-only --spans $S --out runs/pdq_s42
python3 experts/pdq.py --text-model vinai/bertweet-base --score-only --spans $S --out runs/pdq_btw_s42
python3 experts/masc_text.py --model vinai/bertweet-large --class-weight balanced --score-only --spans $S --out runs/masc_btwL_bal
python3 experts/masc_text.py --model vinai/bertweet-large --sibling-loss 0.5 --score-only --spans $S --out runs/masc_btwL_sibloss
echo "=== QUEUE16 DONE ==="
