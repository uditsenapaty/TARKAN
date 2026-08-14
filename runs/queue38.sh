#!/usr/bin/env bash
# D18 — validate the CER effect on an INDEPENDENT, LARGER labelled sample.
# §D.17's binding constraint is that n_dev = 1122 (sigma ~1.2) cannot resolve the 0.4-point
# difference that separates 70.62 from 71.03. t2015 test must stay untouched, but t2017
# train+dev is 4738 labelled aspects that are neither t2015 test nor t2017 test -- a
# legitimate, ~4x larger selection sample. If CER's +0.58 replicates there, the core upgrade
# becomes selectable without ever consulting t2015 test.
set -u
score(){ n=$1; m=$2; ck=$3; echo "=== $n ==="
  python3 -u experts/masc_text.py --score-only --dataset twitter2017 --model "$m" \
      --ckpt runs/$ck --out runs/t17_$n 2>&1 | grep -v "it/s\]" | grep -aE "gold-span"; }
score base_btwL vinai/bertweet-large              masc_btwL_s45
score cer_btwL  vinai/bertweet-large              d15_cer_m05
score base_deb  microsoft/deberta-v3-large        d15_base_deb
score cer_deb   microsoft/deberta-v3-large        d15_cer_deb
score base_robL roberta-large                     d15_base_robL
score cer_robL  roberta-large                     d15_cer_robL
echo "=== QUEUE38 DONE ==="
