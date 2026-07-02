#!/bin/bash
# t2015-ONLY (user directive): DeBERTa-v3-large all-in + rules. t2017 deferred entirely.
cd /teamspace/studios/this_studio
python3 -c "import transformers" 2>/dev/null || pip install "huggingface-hub>=0.34.0,<1.0" >/dev/null 2>&1
while pgrep -f "tune_run.py" >/dev/null 2>&1; do sleep 30; done
S() { echo; echo "[deb] ############ $1 ############"; }
STACK="--evidence-dropout 0.2 --kan-hidden 768 --patience 8 --crf --aux-asc-head --conf-append --feat-gate"
DEB="--text-model microsoft/deberta-v3-large --batch-size 4 --grad-accum 4"

S "ALL-IN DeBERTa t2015"
python3 scripts/tune_run.py --dataset twitter2015 --device cuda --tag ALLIN_deb_t2015 $STACK $DEB || echo "[deb] t2015 FAILED"
cp results/checkpoints/twitter2015_best.pt results/checkpoints/t2015_deb_allin.pt 2>/dev/null

S "NS rules grid on DeBERTa t2015"
python3 scripts/tune_neurosymbolic.py --dataset twitter2015 --device cuda --conf-append --feat-gate \
  --text-model microsoft/deberta-v3-large --checkpoint results/checkpoints/t2015_deb_allin.pt || echo "[deb] NS t2015 FAILED"

echo "[deb] DEBERTA_ALLIN_DONE"
