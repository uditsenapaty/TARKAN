#!/bin/bash
# Final non-backbone t2015 push: best-known base (D2) + rules + 3-seed ensemble.
cd /teamspace/studios/this_studio
python3 -c "import transformers" 2>/dev/null || pip install "huggingface-hub>=0.34.0,<1.0" >/dev/null 2>&1
echo "[endg] waiting for all-in pipeline..."
until grep -q "ALLIN_PIPELINE_DONE" results/logs/allin_pipeline.log 2>/dev/null; do sleep 180; done
while pgrep -f "tune_run.py|tune_neurosymbolic" >/dev/null 2>&1; do sleep 60; done
S() { echo; echo "[endg] ############ $1 ############"; }
CK=results/checkpoints
D2="--evidence-dropout 0.2 --kan-hidden 768 --patience 8 --crf --aux-asc-head"

S "D2 seed 42 (regenerate champion checkpoint)"
python3 scripts/tune_run.py --dataset twitter2015 --device cuda --tag D2_s42 $D2 --seed 42 || echo "[endg] s42 FAILED"
cp $CK/twitter2015_best.pt $CK/t2015_D2_s42.pt

S "NS rules grid on the D2 champion checkpoint"
python3 scripts/tune_neurosymbolic.py --dataset twitter2015 --device cuda || echo "[endg] NS-D2 FAILED"

S "D2 seed 43"
python3 scripts/tune_run.py --dataset twitter2015 --device cuda --tag D2_s43 $D2 --seed 43 || echo "[endg] s43 FAILED"
cp $CK/twitter2015_best.pt $CK/t2015_D2_s43.pt

S "D2 seed 44"
python3 scripts/tune_run.py --dataset twitter2015 --device cuda --tag D2_s44 $D2 --seed 44 || echo "[endg] s44 FAILED"
cp $CK/twitter2015_best.pt $CK/t2015_D2_s44.pt

S "3-seed ensemble (plain)"
python3 scripts/ensemble_eval.py --dataset twitter2015 --device cuda \
  --checkpoints $CK/t2015_D2_s42.pt $CK/t2015_D2_s43.pt $CK/t2015_D2_s44.pt || echo "[endg] ens FAILED"

S "3-seed ensemble (+ NS rules bio + alpha 0.6)"
python3 scripts/ensemble_eval.py --dataset twitter2015 --device cuda --ns-bio --ns-alpha 0.6 \
  --checkpoints $CK/t2015_D2_s42.pt $CK/t2015_D2_s43.pt $CK/t2015_D2_s44.pt || echo "[endg] ens-ns FAILED"

echo "[endg] T2015_ENDGAME_DONE"
