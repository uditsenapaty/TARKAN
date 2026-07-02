#!/bin/bash
# Repair-and-continue suite (post encoder-wiring fix). t2015 Table-1 row + checkpoint are
# good (64.98, reproduced) — only t2017 headline is re-trained. Everything else follows.
cd /teamspace/studios/this_studio
python3 -c "import transformers" 2>/dev/null || pip install "huggingface-hub>=0.34.0,<1.0" >/dev/null 2>&1
S() { echo; echo "[suite2] ############ $1 ############"; }
S "Table 1 REPAIR (t2017 champion retrain, fixed encoder wiring)"
python3 experiments/run_main.py --device cuda --datasets twitter2017 || echo "[suite2] run_main FAILED"
S "Table 3 (both datasets, from champion checkpoints)"
python3 experiments/run_subtasks.py --device cuda || echo "[suite2] run_subtasks FAILED"
S "Table 8 (KG diagnostics)"
python3 analysis/kg_diagnostics.py --device cuda || echo "[suite2] kg_diagnostics FAILED"
S "Table 9 (visual relevance buckets)"
python3 analysis/visual_relevance_diag.py --device cuda || echo "[suite2] visual_relevance FAILED"
S "Table 5 (error analysis)"
python3 analysis/error_analysis.py --device cuda || echo "[suite2] error_analysis FAILED"
S "Table 7 (teacher quality; human subsets absent -> honest partial)"
python3 analysis/teacher_quality.py || echo "[suite2] teacher_quality PARTIAL (expected)"
S "Table 6 (component ablations, incremental)"
python3 ablations/run_ablations.py --device cuda || echo "[suite2] ablations FAILED"
S "Table 10 (fusion ablations, incremental)"
python3 ablations/run_fusion_ablation.py --device cuda || echo "[suite2] fusion FAILED"
S "Plots"
for p in plot_main_results plot_ablation plot_kg_stats kan_spline_viz; do
  python3 visualizations/$p.py || echo "[suite2] $p FAILED"
done
python3 visualizations/plot_relevance.py --dataset twitter2015 --device cuda || echo "[suite2] plot_relevance FAILED"
echo "[suite2] MASTER_SUITE2_DONE"
