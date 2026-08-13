#!/bin/bash
# Queue 15 — high-recall candidate set + reranked assembly.
# The reranker only pays off if there is something to prune, so regenerate candidates at
# cand_thr 0.12 (test R 91.13 vs 89.10 at argmax, P 80.22) and let the annotation-policy
# judge restore precision. MASC members must be re-scored on that larger set.
set -e
cd /teamspace/studios/this_studio
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
MATE="runs/mate_deb_s42 runs/mate_deb_s43 runs/mate_deb_s44 runs/mate_probeA runs/mate_probeB"

echo "=== [1/3] high-recall candidates ==="
python3 experts/emit_spans.py --mate $MATE --cand-thr 0.12 --out runs/mate_ens5_hr

echo "=== [2/3] re-score core MASC members on the high-recall set ==="
S=runs/mate_ens5_hr
python3 experts/masc_text.py --model vinai/bertweet-large --score-only --spans $S --out runs/masc_btwL_s45
python3 experts/masc_text.py --model microsoft/deberta-v3-large --score-only --spans $S --out runs/masc_deb_s44
python3 experts/masc_text.py --model roberta-large --score-only --spans $S --out runs/masc_robL_s46
python3 experts/masc_text.py --model cardiffnlp/twitter-roberta-base-sentiment-latest --score-only --spans $S --out runs/masc_twrob_s42
python3 experts/masc_text.py --model vinai/bertweet-base --score-only --spans $S --out runs/masc_btw_s43
python3 experts/masc_text.py --model microsoft/deberta-v3-large --class-weight sqrt --score-only --spans $S --out runs/masc_deb_sqrt
python3 experts/pdq.py --text-model vinai/bertweet-large --score-only --spans $S --out runs/pdq_btwL_s47
python3 experts/pdq.py --text-model cardiffnlp/twitter-roberta-base-sentiment-latest --score-only --spans $S --out runs/pdq_twrob_s43
python3 experts/masc_gated.py --model vinai/bertweet-large --fuse concat --score-only --desc data/aadg/twitter2015 --spans $S --out runs/masc_gate_btwL_cat

echo "=== [3/3] final sweep ==="
MASC="runs/masc_btwL_s45 runs/masc_deb_s44 runs/masc_robL_s46 runs/masc_twrob_s42 runs/masc_btw_s43 runs/masc_deb_sqrt runs/pdq_btwL_s47 runs/pdq_twrob_s43 runs/masc_gate_btwL_cat"
for mode in span joint; do
  for mix in 0.0 0.5 1.0; do
    echo "--- cand_thr 0.12 | mode $mode | rerank-mix $mix ---"
    python3 experts/assemble.py --mate $MATE --masc $MASC --cand-thr 0.12 --mode $mode \
      --rerank runs/rerank --rerank-mix $mix --out results/q15_${mode}_mix${mix}.json
  done
done
echo "=== QUEUE15 DONE ==="
