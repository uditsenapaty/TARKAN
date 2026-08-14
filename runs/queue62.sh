#!/bin/bash
# Queue 62 — D.33c: does MAMS add on top of the other three corpora?
#
# queue60's arm used twitter,rest,laptop = 12184 aspects. MAMS adds 11186 more (total
# 23370 = 7.35x t2015) and is the only corpus built so that every sentence carries
# multiple aspects with DIFFERENT sentiment -- the exact shortcut t2015 punishes
# (86 POS aspects predicted NEU by falling back on the tweet's overall tone).
#
# Same three seeds as queue60, so its CONTROL arm is reused rather than re-run and the
# comparison stays paired across all three arms.
set -e
cd /teamspace/studios/this_studio
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

M=vinai/bertweet-large
for s in 42 43 44; do
  echo "=== [pre4 s$s] bertweet-large, 2 epochs external x4 -> t2015 ==="
  python3 experts/masc_text.py --model "$M" --seed $s --batch 8 --lr 1e-5 --epochs 6 \
    --pre-epochs 2 --pre-data twitter,rest,laptop,mams \
    --out "runs/d33_pre4_btwL_s$s"
done

echo "=== THREE-ARM PAIRED SUMMARY (gold-span MASC accuracy) ==="
python3 - <<'PY'
import json, statistics as st
def g(d, k):
    return json.load(open(f"runs/{d}/metrics.json"))[k]
print(f"{'seed':>5} {'ctl':>7} {'pre3':>7} {'pre4':>7} | {'d3':>6} {'d4':>6}   (test gold-span)")
d3, d4 = [], []
for s in (42, 43, 44):
    c = g(f"d33_ctl_btwL_s{s}", "test_acc_goldspan")
    a = g(f"d33_pre_btwL_s{s}", "test_acc_goldspan")
    b = g(f"d33_pre4_btwL_s{s}", "test_acc_goldspan")
    d3.append(a - c); d4.append(b - c)
    print(f"{s:>5} {c:7.2f} {a:7.2f} {b:7.2f} | {a-c:+6.2f} {b-c:+6.2f}")
print(f"{'mean':>5} {'':7} {'':7} {'':7} | {st.mean(d3):+6.2f} {st.mean(d4):+6.2f}"
      f"   paired sd {st.stdev(d3):.2f} / {st.stdev(d4):.2f}")
PY
