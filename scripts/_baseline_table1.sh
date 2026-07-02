#!/bin/bash
cd /teamspace/studios/this_studio
LAB_LOG="/tmp/claude-1000/-teamspace-studios-this-studio/cbf6ce77-4bd5-4054-a398-28ca22e320b4/tasks/bna1j7z27.output"
echo "[baseline] waiting for teacher labeling to complete..."
until ! pgrep -f _full_labeling.sh >/dev/null 2>&1; do sleep 30; done
if ! grep -q "ALL LABELING DONE" "$LAB_LOG" 2>/dev/null; then
  echo "[baseline] ABORT: labeling did not finish cleanly (no DONE marker)"; exit 1
fi
echo "[baseline] teacher-label balance:"
python3 -c "import pandas as pd,glob; [print(' ',f.split('/')[-1], pd.read_parquet(f).label.value_counts().to_dict()) for f in sorted(glob.glob('data/teacher_labels/*.parquet'))]"
echo "[baseline] === training Table 1 baseline (default faithful config, cuda, both datasets) ==="
python3 experiments/run_main.py --device cuda
echo "[baseline] TABLE1_BASELINE_DONE"
