#!/bin/bash
# Queue 6 — reruns what OOM'd when queue4 and an orphaned queue5 collided on the single
# T4. Strictly serial. MATE first: it is 85.75 vs Chapter B's known-achievable 87-88 on
# the same architecture, so it is the largest and highest-confidence remaining deficit.
set -e
cd /teamspace/studios/this_studio
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

echo "=== [1/4] MATE probe A: lr 1e-5, head-lr 1e-4, dropout 0.3, 16ep ==="
python3 experts/mate_expert.py --seed 42 --lr 1e-5 --head-lr 1e-4 --epochs 16 \
  --patience 5 --dropout 0.3 --out runs/mate_probeA

echo "=== [2/4] MATE probe B: lr 2e-5, head-lr 1e-4, dropout 0.1, 16ep ==="
python3 experts/mate_expert.py --seed 42 --lr 2e-5 --head-lr 1e-4 --epochs 16 \
  --patience 5 --out runs/mate_probeB

echo "=== [3/4] MASC: bertweet-large (OOM'd in queue5) ==="
python3 experts/masc_text.py --model vinai/bertweet-large --seed 45 \
  --batch 8 --lr 1e-5 --epochs 8 --spans runs/mate_ens3 --out runs/masc_btwL_s45

echo "=== [4/4] report MATE probe outcomes ==="
for d in runs/mate_deb_s42 runs/mate_probeA runs/mate_probeB; do
  [ -f "$d/metrics.json" ] && python3 -c "
import json;d=json.load(open('$d/metrics.json'))
print(f\"{'$d':<26} dev {d['dev']['F1']:.2f}  test P {d['test']['P']:.2f} R {d['test']['R']:.2f} F1 {d['test']['F1']:.2f}\")"
done
echo "=== QUEUE6 DONE ==="
