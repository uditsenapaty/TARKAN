#!/bin/bash
# Recalibrate teacher KG labels to paper Table-8 operating point, then re-run everything.
cd /teamspace/studios/this_studio
python3 -c "import transformers" 2>/dev/null || pip install "huggingface-hub>=0.34.0,<1.0" >/dev/null 2>&1
S() { echo; echo "[recal] ############ $1 ############"; }
S "KG label recalibration (graded teacher scores, top-3/aspect)"
python3 scripts/recalibrate_kg_labels.py --dataset twitter2015 --device cuda --batch-size 32 || { echo "[recal] t2015 FAILED"; exit 1; }
python3 scripts/recalibrate_kg_labels.py --dataset twitter2017 --device cuda --batch-size 32 || { echo "[recal] t2017 FAILED"; exit 1; }
S "clear stale ablation CSVs (labels changed)"
rm -f results/tables/ablation_components.csv results/tables/ablation_fusion.csv
S "Table 1 (champion retrain BOTH datasets, calibrated KG)"
python3 experiments/run_main.py --device cuda || echo "[recal] run_main FAILED"
S "Table 3"
python3 experiments/run_subtasks.py --device cuda || echo "[recal] run_subtasks FAILED"
S "Analyses (Tables 8, 9, 5, 7)"
python3 analysis/kg_diagnostics.py --device cuda || echo "[recal] kg_diag FAILED"
python3 analysis/visual_relevance_diag.py --device cuda || echo "[recal] vis_rel FAILED"
python3 analysis/error_analysis.py --device cuda || echo "[recal] err FAILED"
python3 analysis/teacher_quality.py || echo "[recal] teacher_quality PARTIAL (expected)"
S "Table 6 (component ablations, incremental)"
python3 ablations/run_ablations.py --device cuda || echo "[recal] ablations FAILED"
S "Table 10 (fusion ablations, incremental)"
python3 ablations/run_fusion_ablation.py --device cuda || echo "[recal] fusion FAILED"
S "Plots"
for p in plot_main_results plot_ablation plot_kg_stats kan_spline_viz; do
  python3 visualizations/$p.py || echo "[recal] $p FAILED"
done
python3 visualizations/plot_relevance.py --dataset twitter2015 --device cuda || echo "[recal] plot_relevance FAILED"
echo "[recal] RECAL_PIPELINE_DONE"
