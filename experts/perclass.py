"""Per-class breakdown for a paired arm comparison — overall accuracy hides the mechanism.

§D.33 targets NEG specifically: t2015 train carries **368 NEG aspects** against the external
corpora's **6001**, and NEG recall is this campaign's worst number. A flat overall accuracy
is compatible with two very different outcomes — the lever did nothing, or it moved NEG and
lost the same amount on NEU. Only the per-class table separates them, and the ensemble cares
about the second case (a member that trades classes is DECORRELATED even when its accuracy
is unchanged, which is the property §B.8 said the ensemble responds to).

    python experts/perclass.py --arms runs/d33_ctl_btwL_s42 runs/d33_pre_btwL_s42
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from experts.common import POL2ID, POLARITIES, load, masc_examples  # noqa: E402


def table(run: Path, split: str):
    z = np.load(run / f"probs_{split}.npz")
    probs, keys = z["probs"], z["keys"]
    gold = {e.key: POL2ID[e.polarity] for e in masc_examples(load("twitter2015", split))}
    y = np.array([gold[tuple(int(x) for x in k)] for k in keys])
    p = probs.argmax(1)
    out = {}
    for i, name in enumerate(POLARITIES):
        m = y == i
        out[name] = (100.0 * (p[m] == i).mean() if m.any() else float("nan"), int(m.sum()))
    out["ALL"] = (100.0 * (p == y).mean(), len(y))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arms", nargs="+", required=True)
    ap.add_argument("--splits", nargs="+", default=["dev", "test"])
    args = ap.parse_args()
    for split in args.splits:
        print(f"\n=== {split} — recall per gold class ===")
        rows = {}
        for a in args.arms:
            try:
                rows[a] = table(Path(a), split)
            except FileNotFoundError as e:
                print(f"  {a}: missing ({e.filename})")
        if not rows:
            continue
        cols = ["NEG", "NEU", "POS", "ALL"]
        n = next(iter(rows.values()))
        print(f"{'arm':<26}" + "".join(f"{c+f'(n={n[c][1]})':>16}" for c in cols))
        base = None
        for a, r in rows.items():
            line = f"{Path(a).name:<26}" + "".join(f"{r[c][0]:16.2f}" for c in cols)
            print(line)
            if base is None:
                base = r
            else:
                print(f"{'  delta':<26}" + "".join(f"{r[c][0]-base[c][0]:+16.2f}"
                                                   for c in cols))


if __name__ == "__main__":
    main()
