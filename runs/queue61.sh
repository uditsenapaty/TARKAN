#!/bin/bash
# Queue 61 — D.33b: the same external-supervision lever on the EXTRACTION side.
#
# joint = MATE@tau x a. At 87.80 x 80.44 = 70.62 the bar needs +2.83 on one factor or
# +1.4 on each, so the polarity-only version cannot reach it alone. §D.21 showed the
# extraction side has no architectural diversity and that filling that gap changed
# nothing; what it has never had is more DATA. t2015 train is 2101 sentences. The
# external corpora add 9733 sentences / 12160 aspect terms, BIO round-trip verified
# 0/9733 failures, leak gate clean.
#
# Paired seeds again (§D.20). The recipe is §C.6's fixed one: head-lr 1e-4.
set -e
cd /teamspace/studios/this_studio
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

for s in 42 43 44; do
  echo "=== [ctl s$s] deberta MATE, t2015 only ==="
  python3 experts/mate_expert.py --seed $s --lr 1e-5 --head-lr 1e-4 --epochs 12 \
    --patience 4 --out "runs/d33_ctl_mate_s$s"
  echo "=== [pre s$s] deberta MATE, 2 epochs external -> t2015 ==="
  python3 experts/mate_expert.py --seed $s --lr 1e-5 --head-lr 1e-4 --epochs 12 \
    --patience 4 --pre-epochs 2 --pre-data twitter,rest,laptop \
    --out "runs/d33_pre_mate_s$s"
done

echo "=== PAIRED SUMMARY (MATE F1) ==="
python3 - <<'PY'
import json, statistics as st
rows = []
for s in (42, 43, 44):
    a = json.load(open(f"runs/d33_ctl_mate_s{s}/metrics.json"))
    b = json.load(open(f"runs/d33_pre_mate_s{s}/metrics.json"))
    g = lambda d, k: d.get(k, d.get("best_dev_F1", float("nan")))
    rows.append((s, a["best_dev_F1"], b["best_dev_F1"],
                 g(a, "test_F1"), g(b, "test_F1")))
print(f"{'seed':>5} {'dev ctl':>8} {'dev pre':>8} {'d dev':>7} "
      f"{'test ctl':>9} {'test pre':>9} {'d test':>7}")
for s, dc, dp, tc, tp in rows:
    print(f"{s:>5} {dc:8.2f} {dp:8.2f} {dp-dc:+7.2f} {tc:9.2f} {tp:9.2f} {tp-tc:+7.2f}")
dd = [r[2]-r[1] for r in rows]; dt = [r[4]-r[3] for r in rows]
print(f"mean d dev {st.mean(dd):+.2f}   mean d test {st.mean(dt):+.2f}   "
      f"paired sd(test) {st.stdev(dt):.2f}")
PY
