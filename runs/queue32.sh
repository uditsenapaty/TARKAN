#!/usr/bin/env bash
# Complete the §D.9/D.10 correction: those members were trained with w_pos=0.147 (the
# CAPTION teacher's inverse frequency) against a 2.16:1 teacher needing 0.462. The
# bertweet-large re-run with correct weights went 77.24 -> 78.50, exactly the caption
# baseline, so the recorded -1.1 was an artifact. Re-run the other two towers.
set -u
for m in "qpds_deb_bal microsoft/deberta-v3-large" \
         "qpds_robL_bal roberta-large"; do
  set -- $m
  echo "=== $1 ==="
  python3 -u experts/masc_pds.py --residual --w-none 0 --pds data/pds_qwen/twitter2015 \
      --model "$2" --spans runs/mate_ens5_hr --out runs/$1 2>&1 | grep -v "it/s\]" | tail -4
done
echo "=== QUEUE32 DONE ==="
