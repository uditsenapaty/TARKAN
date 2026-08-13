#!/usr/bin/env bash
# Re-run the two PDQ members that failed the strict state_dict load in queue26
# (trained before the ITC/ITM heads existed; pdq.py now loads them non-strictly).
set -u
S=runs/mate_union
for m in "pdq_btwL_s47 vinai/bertweet-large" \
         "pdq_twrob_s43 cardiffnlp/twitter-roberta-base-sentiment-latest"; do
  set -- $m
  echo "=== $1 ==="
  python3 -u experts/pdq.py --score-only --text-model "$2" --spans $S --out runs/$1 \
      2>&1 | grep -v "it/s\]"
done

# ...and the three PDS members, which failed the same way on the NEU-escape `beta`.
for m in "pds_res_btwL_wn0 vinai/bertweet-large" \
         "pds_res_deb_wn0 microsoft/deberta-v3-large" \
         "pds_res_robL_wn0 roberta-large"; do
  set -- $m
  echo "=== $1 ==="
  python3 -u experts/masc_pds.py --score-only --residual --w-none 0 \
      --model "$2" --spans $S --out runs/$1 2>&1 | grep -v "it/s\]"
done
echo "=== QUEUE26B DONE ==="
