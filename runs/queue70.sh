#!/bin/bash
# Queue 70 — KAN, done as the proposal specifies: diagnostic FIRST, then the head.
#
# The decisive question is not KAN width (§A measured capacity flat). It is whether the
# visual stream carries information the text does not, CONDITIONAL on the aspect. If
# text+visual == text, no KAN topology can manufacture a gain and the experiment is over.
#
#   A  mlp  --no-vis   text only, gate hard-closed        <- diagnostic baseline
#   B  mlp             text+visual, incumbent Linear      <- B-A = I(y; v | t,a) proxy
#   C  kan             text+visual, the paper's 2-layer width-256 KAN over the SAME
#                      fused vector (the "blunt concatenation" use)
#   D  ikan            text+visual, interaction-KAN: per-modality LayerNorm'd 192-d
#                      projections, explicit [t, v, t*v, |t-v|], residual text baseline,
#                      alpha init 0.1 (NOT zero -- that is the PACS dead-gradient trap)
#
# Everything else identical: bertweet-large, seed 50, 8 epochs, lr 1e-5, head-lr 1e-3,
# same AADG descriptions, same candidate spans, same optimiser.
#
# NOTE: the KG stream is absent (data/kg_index gone, §D.34), so this is the 2-modality
# subset of the paper's [t; v; g]. Recorded, not hidden.
set -e
cd /teamspace/studios/this_studio
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
S=runs/mate_ens5_hr
D=data/aadg/twitter2015
M=vinai/bertweet-large

arm () {  # $1=tag  $2...=extra flags
  tag=$1; shift
  echo "=== [$tag] ==="
  python3 experts/masc_gated.py --model $M --desc $D --spans $S \
    --seed 50 --epochs 8 --batch 8 --lr 1e-5 --head-lr 1e-3 \
    "$@" --out "runs/k70_$tag"
}

arm A_mlp_textonly --head mlp  --no-vis
arm B_mlp_tv       --head mlp
arm C_kan_tv       --head kan
arm D_ikan_tv      --head ikan

echo "=== SUMMARY (gold-span MASC accuracy, identical recipe) ==="
python3 - <<'PY'
import json
from pathlib import Path
rows = []
for t in ("A_mlp_textonly", "B_mlp_tv", "C_kan_tv", "D_ikan_tv"):
    f = Path(f"runs/k70_{t}/metrics.json")
    if not f.exists():
        print(f"  {t}: missing"); continue
    d = json.load(open(f))
    rows.append((t, d["dev_acc_goldspan"], d["test_acc_goldspan"]))
print(f"{'arm':<18}{'dev':>8}{'test':>8}")
for t, dv, te in rows:
    print(f"{t:<18}{dv:8.2f}{te:8.2f}")
if len(rows) == 4:
    a, b, c, d_ = (r[2] for r in rows)
    ad, bd, cd, dd = (r[1] for r in rows)
    print(f"\nDIAGNOSTIC  visual adds (B-A): dev {bd-ad:+.2f}  test {b-a:+.2f}")
    print(f"KAN         concat-KAN (C-B): dev {cd-bd:+.2f}  test {c-b:+.2f}")
    print(f"KAN         interact.  (D-B): dev {dd-bd:+.2f}  test {d_-b:+.2f}")
    print("\nIf B-A ~ 0, no KAN topology can create a gain and C/D are expected null.")
PY
