#!/bin/bash
# ALL-IN first (user directive): every improving patch together per dataset, then the
# neurosymbolic rules grid on the resulting checkpoint. Attribution probes deferred.
cd /teamspace/studios/this_studio
python3 -c "import transformers" 2>/dev/null || pip install "huggingface-hub>=0.34.0,<1.0" >/dev/null 2>&1
echo "[allin] waiting for t2017 KG recalibration..."
until grep -q "RECALIBRATED twitter2017" results/logs/recal_pipeline.log 2>/dev/null; do sleep 60; done
echo "[allin] intercepting recal pipeline (its run_main is superseded by the all-in runs)"
pkill -f _recal_pipeline.sh 2>/dev/null; sleep 2
pkill -9 -f "run_main.py" 2>/dev/null; sleep 2
S() { echo; echo "[allin] ############ $1 ############"; }

S "ALL-IN t2015: base + calibrated-KG + CRF + richASC + A9 conf-append + A10 gates"
python3 scripts/tune_run.py --dataset twitter2015 --device cuda --tag ALLIN_t2015 \
  --evidence-dropout 0.2 --kan-hidden 768 --patience 8 --crf --aux-asc-head \
  --conf-append --feat-gate || echo "[allin] ALLIN_t2015 FAILED"

S "NS rules grid on the all-in t2015 checkpoint"
python3 scripts/tune_neurosymbolic.py --dataset twitter2015 --device cuda \
  --conf-append --feat-gate || echo "[allin] NS t2015 FAILED"

S "ALL-IN t2017: large + calibrated-KG + CRF + richASC + A9 + A10"
python3 scripts/tune_run.py --dataset twitter2017 --device cuda --tag ALLIN_t2017 \
  --evidence-dropout 0.2 --kan-hidden 768 --patience 8 --crf --aux-asc-head \
  --conf-append --feat-gate --text-model vinai/bertweet-large --batch-size 8 --grad-accum 2 \
  || echo "[allin] ALLIN_t2017 FAILED"

S "NS rules grid on the all-in t2017 checkpoint"
python3 scripts/tune_neurosymbolic.py --dataset twitter2017 --device cuda \
  --conf-append --feat-gate || echo "[allin] NS t2017 FAILED"

echo "[allin] ALLIN_PIPELINE_DONE"
