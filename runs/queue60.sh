#!/bin/bash
# Queue 60 — D.33: external aspect-sentiment supervision as intermediate training.
#
# The one input class never tried. Every MASC tower in this project has seen 3179 aspects
# and nothing else; §D.1 measured polarity losing twice what extraction loses; §B.8 named
# "new information" as the only escape from a saturated ensemble. 12184 external aspects
# (Dong-2014 Twitter + SemEval-14 Rest/Laptop) is 3.83x more supervision for the losing
# task and 8.8x more NEG, the campaign's worst class. Leak gate: 0 exact / 0 near-dup
# against all three t2015 splits, max Jaccard 0.000 (experts/absa_extra.py --check).
#
# PAIRED SEEDS, because §D.20 measured the single-run detection floor at +-1.31 F1 and
# most per-patch verdicts in Chapters C/D were never detectable. Three seeds per arm,
# same seed compared across arms, so the comparison is paired.
set -e
cd /teamspace/studios/this_studio
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

M=vinai/bertweet-large       # strongest text family (test gold-span 79.75)
for s in 42 43 44; do
  echo "=== [ctl s$s] bertweet-large, t2015 only ==="
  python3 experts/masc_text.py --model "$M" --seed $s --batch 8 --lr 1e-5 --epochs 6 \
    --out "runs/d33_ctl_btwL_s$s"
  echo "=== [pre s$s] bertweet-large, 2 epochs external -> t2015 ==="
  python3 experts/masc_text.py --model "$M" --seed $s --batch 8 --lr 1e-5 --epochs 6 \
    --pre-epochs 2 --pre-data twitter,rest,laptop \
    --out "runs/d33_pre_btwL_s$s"
done

echo "=== PAIRED SUMMARY (gold-span MASC accuracy) ==="
python3 - <<'PY'
import json, statistics as st
rows = []
for s in (42, 43, 44):
    a = json.load(open(f"runs/d33_ctl_btwL_s{s}/metrics.json"))
    b = json.load(open(f"runs/d33_pre_btwL_s{s}/metrics.json"))
    rows.append((s, a["dev_acc_goldspan"], b["dev_acc_goldspan"],
                 a["test_acc_goldspan"], b["test_acc_goldspan"]))
print(f"{'seed':>5} {'dev ctl':>8} {'dev pre':>8} {'d dev':>7} "
      f"{'test ctl':>9} {'test pre':>9} {'d test':>7}")
for s, dc, dp, tc, tp in rows:
    print(f"{s:>5} {dc:8.2f} {dp:8.2f} {dp-dc:+7.2f} {tc:9.2f} {tp:9.2f} {tp-tc:+7.2f}")
dd = [r[2]-r[1] for r in rows]; dt = [r[4]-r[3] for r in rows]
print(f"{'mean':>5} {st.mean(r[1] for r in rows):8.2f} {st.mean(r[2] for r in rows):8.2f} "
      f"{st.mean(dd):+7.2f} {st.mean(r[3] for r in rows):9.2f} "
      f"{st.mean(r[4] for r in rows):9.2f} {st.mean(dt):+7.2f}")
print(f"paired sd(test delta) {st.stdev(dt):.2f}")
PY
