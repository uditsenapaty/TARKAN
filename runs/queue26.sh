#!/usr/bin/env bash
# D.3 — re-score every existing MASC member on the UNION candidate set
# (BIO cand_thr 0.12 + PDQ-MATE decoded spans, test pool recall 91.13 -> 93.44).
# Score-only: no member is retrained, only the anchor set changes.
set -u
S=runs/mate_union
i=0
run() { i=$((i+1)); echo "=== [$i] $* ==="; "$@" 2>&1 | grep -v "it/s\]"; }

for m in "masc_btwL_s45 vinai/bertweet-large" \
         "masc_deb_s44 microsoft/deberta-v3-large" \
         "masc_robL_s46 roberta-large" \
         "masc_twrob_s42 cardiffnlp/twitter-roberta-base-sentiment-latest" \
         "masc_btw_s43 vinai/bertweet-base" \
         "masc_deb_sqrt microsoft/deberta-v3-large"; do
  set -- $m
  run python3 -u experts/masc_text.py --score-only --model "$2" --spans $S --out runs/$1
done

for m in "pdq_btwL_s47 vinai/bertweet-large" \
         "pdq_twrob_s43 cardiffnlp/twitter-roberta-base-sentiment-latest" \
         "pdq_btwL_itcitm vinai/bertweet-large" \
         "pdq_twrob_itcitm cardiffnlp/twitter-roberta-base-sentiment-latest" \
         "pdq_deb_itcitm microsoft/deberta-v3-large" \
         "pdq_robL_itcitm roberta-large"; do
  set -- $m
  run python3 -u experts/pdq.py --score-only --text-model "$2" --spans $S --out runs/$1
done

for m in "pds_res_btwL_wn0 vinai/bertweet-large" \
         "pds_res_deb_wn0 microsoft/deberta-v3-large" \
         "pds_res_robL_wn0 roberta-large"; do
  set -- $m
  run python3 -u experts/masc_pds.py --score-only --residual --w-none 0 \
      --model "$2" --spans $S --out runs/$1
done

echo "=== QUEUE26 DONE ==="
