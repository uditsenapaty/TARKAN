#!/bin/bash
# Queue 25 — PDS-v2 (continuous signed teacher target) + NEU-escape gate.
#  PDS-v2 : the 3-way {POS,NEG,NONE} target discards 75% of the teacher's output, because
#           "no shift" mostly means "the teacher could not determine a direction", not
#           "the effect is zero". Use the soft distribution directly:
#               q = 1 - P(NONE);  target = q * (P(POS) - P(NEG))  in [-1,1]
#           regressed onto the student's signed residual with SmoothL1, so ambiguous cases
#           pull toward 0 smoothly instead of being clamped (the original suppression bug).
#  NEU-esc: amplify the correction where the measured errors actually are -- 86 POS and
#           35 NEG aspects predicted NEU -- via alpha*(1 + beta*P_base(NEU)), beta init 0
#           so it starts identical to the plain residual and must earn its effect.
set -e
cd /teamspace/studios/this_studio
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
S=runs/mate_ens5_hr
echo "=== [1/4] PDS-v2 continuous, bertweet-large ==="
python3 experts/masc_pds.py --residual --pds-mode continuous --model vinai/bertweet-large \
  --seed 90 --lambda-pds 0.5 --spans $S --out runs/pds_v2_btwL
echo "=== [2/4] PDS-v2 continuous, deberta-v3-large ==="
python3 experts/masc_pds.py --residual --pds-mode continuous --model microsoft/deberta-v3-large \
  --seed 91 --lambda-pds 0.5 --spans $S --out runs/pds_v2_deb
echo "=== [3/4] PDS-v2 + NEU-escape, bertweet-large ==="
python3 experts/masc_pds.py --residual --pds-mode continuous --neu-escape \
  --model vinai/bertweet-large --seed 92 --lambda-pds 0.5 --spans $S --out runs/pds_v2_esc_btwL
echo "=== [4/4] PDS-margin + NEU-escape (isolates the gate), bertweet-large ==="
python3 experts/masc_pds.py --residual --neu-escape --w-none 0.0 \
  --model vinai/bertweet-large --seed 93 --lambda-pds 0.5 --spans $S --out runs/pds_esc_btwL
echo "=== QUEUE25 DONE ==="
