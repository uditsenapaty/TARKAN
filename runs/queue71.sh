#!/bin/bash
# Queue 71 — D.36: paired seeds on text-only vs interaction-KAN.
#
# The single-seed run gave D-A = +0.20 test / -0.81 dev with a stratified pattern that is
# exactly on-hypothesis (image-irrelevant harm eliminated -0.87 -> +0.00, image-useful gain
# tripled +0.29 -> +0.87, multi-aspect flipped -0.81 -> +1.13). Coherent mechanism, aggregate
# inside the +-1.31 floor, dev disagreeing with test. Only paired seeds settle it.
set -e
cd /teamspace/studios/this_studio
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
S=runs/mate_ens5_hr; D=data/aadg/twitter2015; M=vinai/bertweet-large
for s in 51 52 53; do
  python3 experts/masc_gated.py --model $M --desc $D --spans $S --seed $s \
    --epochs 8 --batch 8 --lr 1e-5 --head-lr 1e-3 --head mlp --no-vis \
    --out runs/k71_A_s$s
  python3 experts/masc_gated.py --model $M --desc $D --spans $S --seed $s \
    --epochs 8 --batch 8 --lr 1e-5 --head-lr 1e-3 --head ikan \
    --out runs/k71_D_s$s
done
echo "=== PAIRED SUMMARY: text-only (A) vs interaction-KAN (D) ==="
python3 - <<'PY'
import json, statistics as st
rows=[]
for s in (51,52,53):
    a=json.load(open(f"runs/k71_A_s{s}/metrics.json")); b=json.load(open(f"runs/k71_D_s{s}/metrics.json"))
    rows.append((s,a["dev_acc_goldspan"],b["dev_acc_goldspan"],a["test_acc_goldspan"],b["test_acc_goldspan"]))
print(f"{'seed':>5}{'dev A':>9}{'dev D':>9}{'d dev':>8}{'test A':>9}{'test D':>9}{'d test':>8}")
for s,da,db,ta,tb in rows: print(f"{s:>5}{da:9.2f}{db:9.2f}{db-da:+8.2f}{ta:9.2f}{tb:9.2f}{tb-ta:+8.2f}")
dd=[r[2]-r[1] for r in rows]; dt=[r[4]-r[3] for r in rows]
print(f"{'mean':>5}{'':9}{'':9}{st.mean(dd):+8.2f}{'':9}{'':9}{st.mean(dt):+8.2f}")
sd=st.stdev(dt); print(f"\npaired sd(test) {sd:.2f}  SE {sd/3**.5:.2f}  t(2)={st.mean(dt)/(sd/3**.5):.2f}"
                      f"   all positive: {all(x>0 for x in dt)}")
PY
