"""Check TARKAN's real Table-1 numbers against the per-column best baseline.

Reads results/tables/main_results.csv (written by experiments/run_main.py) and reports,
per metric cell, whether TARKAN beats EVERY referred baseline in that column (the user's
win criterion: >=90% of cells must beat all baselines). Joint MABSA = micro P/R/F1.

Usage: python scripts/check_table1.py [results/tables/main_results.csv]
"""
from __future__ import annotations

import csv
import sys
from pathlib import Path

# Per-column best baseline to beat (exact, from the paper's Table 1 LaTeX).
BEST = {
    "twitter2015": {"joint_P": 72.3, "joint_R": 72.7, "joint_F1": 72.5},  # VLHA
    "twitter2017": {"joint_P": 71.1, "joint_R": 72.1, "joint_F1": 71.4},  # DQPS P / SGBIS,VLHA R / VLHA F1
}
TARGET = {  # paper's (placeholder) TARKAN row, for reference only
    "twitter2015": {"joint_P": 73.4, "joint_R": 74.8, "joint_F1": 74.1},
    "twitter2017": {"joint_P": 72.0, "joint_R": 73.6, "joint_F1": 72.9},
}


def main():
    path = Path(sys.argv[1] if len(sys.argv) > 1 else "results/tables/main_results.csv")
    if not path.exists():
        print(f"no results yet at {path}")
        return
    rows = list(csv.DictReader(open(path)))
    by_ds = {r["dataset"]: r for r in rows if r.get("model") == "TARKAN"}

    total, won = 0, 0
    print(f"{'dataset':12} {'metric':9} {'TARKAN':>7} {'best_base':>9} {'paper':>6}  verdict")
    print("-" * 60)
    for ds, cols in BEST.items():
        r = by_ds.get(ds)
        if not r:
            print(f"{ds:12} (no TARKAN row yet)")
            continue
        for m, base in cols.items():
            val = float(r[m])
            total += 1
            ok = val > base
            won += ok
            print(f"{ds:12} {m:9} {val:7.1f} {base:9.1f} {TARGET[ds][m]:6.1f}  {'WIN ' if ok else 'lose'} "
                  f"({val-base:+.1f})")
    if total:
        frac = 100.0 * won / total
        print("-" * 60)
        print(f"Table-1 cells beating EVERY baseline: {won}/{total} = {frac:.0f}%  "
              f"({'>=90% ACHIEVED' if frac >= 90 else 'below 90% — keep applying patches'})")


if __name__ == "__main__":
    main()
