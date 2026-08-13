#!/bin/bash
# Final t2015 no-backbone card: 3-seed ensemble of the BEST measured config (auto-picked).
cd /teamspace/studios/this_studio
python3 -c "import transformers" 2>/dev/null || pip install "huggingface-hub>=0.34.0,<1.0" >/dev/null 2>&1
echo "[ens] waiting for DeBERTa all-in chain..."
until grep -q "DEBERTA_ALLIN_DONE" results/logs/deberta_allin.log 2>/dev/null; do sleep 180; done
while pgrep -f "tune_run.py" >/dev/null 2>&1 || pgrep -f "tune_neurosymbolic" >/dev/null 2>&1; do sleep 60; done

FLAGS=$(python3 - <<'PY'
import csv
rows=[r for r in csv.DictReader(open('results/tables/iterations.csv'))
      if r['dataset']=='twitter2015' and r.get('joint_F1') and not r['tag'].startswith('NS_')]
best=max(rows,key=lambda r:float(r['joint_F1']))
f=[f"--evidence-dropout {best['evidence_dropout']}"]
kh=best['kan_hidden'].strip('()').replace(' ','').rstrip(',')
f.append(f"--kan-hidden {kh}"); f.append(f"--patience {best['patience']}")
if best.get('class_weight')=='True': f.append('--class-weight')
if float(best.get('label_smoothing') or 0)>0: f.append(f"--label-smoothing {best['label_smoothing']}")
if best.get('layerwise_lr') not in (None,'','None'): f.append(f"--layerwise-lr {best['layerwise_lr']}")
if best.get('aux_asc_head')=='True': f.append('--aux-asc-head')
if best.get('use_crf')=='True': f.append('--crf')
if best.get('conf_append')=='True': f.append('--conf-append')
if best.get('feat_gate')=='True': f.append('--feat-gate')
tm=best.get('text_model') or 'vinai/bertweet-base'
if tm!='vinai/bertweet-base':
    f.append(f"--text-model {tm}")
    f.append(f"--batch-size {best.get('batch_size') or 8}"); f.append(f"--grad-accum {best.get('grad_accum') or 2}")
print(' '.join(f))
import sys; print(f"[ens] base config: {best['tag']} joint={best['joint_F1']}", file=sys.stderr)
PY
)
echo "[ens] composed flags: $FLAGS"
CK=results/checkpoints
for SEED in 42 43 44; do
  echo "[ens] ############ seed $SEED ############"
  python3 scripts/tune_run.py --dataset twitter2015 --device cuda --tag ENS_s$SEED --seed $SEED $FLAGS || echo "[ens] s$SEED FAILED"
  cp $CK/twitter2015_best.pt $CK/t2015_ens_s$SEED.pt 2>/dev/null
done
# extra flags for the voter must match the trained architecture
VFLAGS=""
case "$FLAGS" in *"--conf-append"*) VFLAGS="$VFLAGS --conf-append";; esac
case "$FLAGS" in *"--feat-gate"*)  VFLAGS="$VFLAGS --feat-gate";;  esac
echo "[ens] ############ 3-seed ensemble (plain) ############"
python3 scripts/ensemble_eval.py --dataset twitter2015 --device cuda $VFLAGS \
  --checkpoints $CK/t2015_ens_s42.pt $CK/t2015_ens_s43.pt $CK/t2015_ens_s44.pt || echo "[ens] vote FAILED"
echo "[ens] ############ 3-seed ensemble (+rules) ############"
python3 scripts/ensemble_eval.py --dataset twitter2015 --device cuda $VFLAGS --ns-bio --ns-alpha 0.6 \
  --checkpoints $CK/t2015_ens_s42.pt $CK/t2015_ens_s43.pt $CK/t2015_ens_s44.pt || echo "[ens] vote-ns FAILED"
echo "[ens] T2015_ENDGAME_DONE"
