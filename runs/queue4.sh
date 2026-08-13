#!/bin/bash
# Queue 4 — MATE recipe probe.
# C1 seed 42 reached test 85.77 vs Chapter B's single-seed 87.00, and train loss hit
# 0.013 by ep9 (heavy overfit) with a 1.1 dev->test gap Chapter B did not report. The
# prime suspect is the optimisation recipe, not the architecture: head-lr 1e-3 on the CRF
# transition matrix is aggressive, and DeBERTa-v3-large usually prefers a lower encoder LR
# with more epochs. Closing a 1.2 recipe gap is cheaper and more certain than any new
# module, so probe it before building more machinery.
set -e
cd /teamspace/studios/this_studio

echo "=== [1/2] MATE probe: lr 1e-5, head-lr 1e-4, 16 epochs, dropout 0.3 ==="
python3 experts/mate_expert.py --seed 42 --lr 1e-5 --head-lr 1e-4 --epochs 16 \
  --patience 5 --dropout 0.3 --out runs/mate_probe_lo

echo "=== [2/2] MATE probe: lr 2e-5, head-lr 1e-4, 16 epochs ==="
python3 experts/mate_expert.py --seed 42 --lr 2e-5 --head-lr 1e-4 --epochs 16 \
  --patience 5 --out runs/mate_probe_mid
echo "=== QUEUE4 DONE ==="
