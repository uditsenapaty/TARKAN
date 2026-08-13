#!/bin/bash
# Master evaluation suite — fully resumable. Order: checkpoints+Table1 -> Table3 ->
# analyses (5,7,8,9) -> Table6 -> Table10 -> plots. Run AFTER champion config is baked.
cd /teamspace/studios/this_studio
python3 -c "import transformers" 2>/dev/null || pip install "huggingface-hub>=0.34.0,<1.0" >/dev/null 2>&1
while pgrep -f "tune_run.py" >/dev/null 2>&1; do sleep 60; done
S() { echo; echo "[suite] ############ $1 ############"; }
S "Table 1 (champion retrain both datasets, regenerates clean checkpoints)"
python3 experiments/run_main.py --device cuda || echo "[suite] run_main FAILED"
S "Table 3 (MATE/MASC from champion checkpoints)"
python3 experiments/run_subtasks.py --device cuda || echo "[suite] run_subtasks FAILED"
S "Table 8 (KG diagnostics)"
python3 analysis/kg_diagnostics.py --device cuda || echo "[suite] kg_diagnostics FAILED"
S "Table 9 (visual relevance buckets)"
python3 analysis/visual_relevance_diag.py --device cuda || echo "[suite] visual_relevance FAILED"
S "Table 5 (error analysis)"
python3 analysis/error_analysis.py --device cuda || echo "[suite] error_analysis FAILED"
S "Table 7 (teacher quality; reports what exists, human subsets absent -> honest partial)"
python3 analysis/teacher_quality.py || echo "[suite] teacher_quality FAILED (expected without human parquets)"
S "Table 6 (component ablations, incremental)"
python3 ablations/run_ablations.py --device cuda || echo "[suite] ablations FAILED"
S "Table 10 (fusion ablations, incremental)"
python3 ablations/run_fusion_ablation.py --device cuda || echo "[suite] fusion FAILED"
S "Plots"
for p in plot_main_results plot_ablation plot_kg_stats kan_spline_viz; do
  python3 visualizations/$p.py || echo "[suite] $p FAILED"
done
python3 visualizations/plot_relevance.py --dataset twitter2015 --device cuda || echo "[suite] plot_relevance FAILED"
echo "[suite] MASTER_SUITE_DONE"
