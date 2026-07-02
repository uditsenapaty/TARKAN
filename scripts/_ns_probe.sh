#!/bin/bash
cd /teamspace/studios/this_studio
echo "[ns] waiting for backbone probes..."
until grep -q "BACKBONE_PROBES_DONE" results/logs/backbone_probe.log 2>/dev/null; do sleep 180; done
while pgrep -f "tune_run.py" >/dev/null 2>&1; do sleep 60; done
echo "[ns] ===== neurosymbolic dev-grid, both datasets ====="
python3 scripts/tune_neurosymbolic.py --dataset twitter2015 --device cuda || echo "[ns] t2015 FAILED"
python3 scripts/tune_neurosymbolic.py --dataset twitter2017 --device cuda || echo "[ns] t2017 FAILED"
echo "[ns] NS_PROBES_DONE"
