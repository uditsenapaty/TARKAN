#!/bin/bash
cd /teamspace/studios/this_studio
python3 -c "import transformers" 2>/dev/null || pip install "huggingface-hub>=0.34.0,<1.0" >/dev/null 2>&1
echo "[large] waiting for CRF batch + any tune_run to finish..."
while pgrep -f "_batch_crf.sh" >/dev/null 2>&1 || pgrep -f "tune_run.py" >/dev/null 2>&1; do sleep 60; done
# compose flags from the best-joint t2015 row (greedy: keep exactly what won)
FLAGS=$(python3 - <<'PY'
import csv
rows=[r for r in csv.DictReader(open('results/tables/iterations.csv')) if r['dataset']=='twitter2015']
best=max(rows,key=lambda r:float(r['joint_F1']))
f=[f"--evidence-dropout {best['evidence_dropout']}"]
kh=best['kan_hidden'].strip('()').replace(' ','').rstrip(',')
f.append(f"--kan-hidden {kh}")
f.append(f"--patience {best['patience']}")
if best.get('class_weight')=='True': f.append('--class-weight')
if float(best.get('label_smoothing') or 0)>0: f.append(f"--label-smoothing {best['label_smoothing']}")
if best.get('layerwise_lr') not in (None,'','None'): f.append(f"--layerwise-lr {best['layerwise_lr']}")
if best.get('aux_asc_head')=='True': f.append('--aux-asc-head')
if best.get('use_crf')=='True': f.append('--crf')
print(' '.join(f))
import sys; print('[large] base config from:', best['tag'], best['joint_F1'], file=sys.stderr)
PY
)
echo "[large] composed flags: $FLAGS"
echo "[large] ===== E8_large_best ====="
python3 scripts/tune_run.py --dataset twitter2015 --device cuda --tag E8_large_best \
  --text-model vinai/bertweet-large --batch-size 8 --grad-accum 2 $FLAGS || echo "[large] E8 FAILED"
echo "[large] BATCH_LARGE_DONE"
