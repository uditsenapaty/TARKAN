"""Member error-correlation diagnostic — the measurement that decides C2.

Chapter B (§7v/§7x) established that on this task a member's *individual* accuracy does
not predict whether it helps: the rationale-distilled members were the strongest singles
of the session (79.2) and HURT the ensemble, and LLaVA-1.5-7B matched the Qwen block on
accuracy yet still hurt, because its errors were correlated with it --

    both_wrong = 0.149  vs  independence expectation 0.043  ->  ratio 3.43

So a new member is only worth keeping if it is **decorrelated**, i.e. ratio near 1.
This reports, for every pair, the observed both-wrong rate against the rate implied by
independence, plus the oracle ceiling and each member's unique-right share.

    python experts/diagnose.py --masc runs/masc_twrob_s42 runs/masc_btw_s43 runs/pdq_s42
"""
from __future__ import annotations

import argparse
import sys
from itertools import combinations
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from experts.common import POLARITIES, load, masc_examples  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="twitter2015")
    ap.add_argument("--split", default="test")
    ap.add_argument("--masc", nargs="+", required=True)
    args = ap.parse_args()

    ex = masc_examples(load(args.dataset, args.split))
    gold = {e.key: e.polarity for e in ex}

    names, correct = [], []
    for d in args.masc:
        z = np.load(Path(d) / f"probs_{args.split}.npz", allow_pickle=True)
        probs, keys = z["probs"], z["keys"]
        ok = {}
        for p, k in zip(probs, keys):
            key = (int(k[0]), int(k[1]), int(k[2]))
            if key in gold:
                ok[key] = int(POLARITIES[int(np.argmax(p))] == gold[key])
        names.append(Path(d).name)
        correct.append(ok)

    shared = set(correct[0])
    for c in correct[1:]:
        shared &= set(c)
    shared = sorted(shared)
    M = np.array([[c[k] for k in shared] for c in correct])   # [n_members, n_items]
    n = M.shape[1]
    print(f"{len(names)} members on {n} shared gold aspects ({args.split})\n")

    for i, nm in enumerate(names):
        print(f"  {nm:<24} acc {100*M[i].mean():.2f}")
    print(f"\n  MAJORITY-VOTE-equivalent oracle (any member right): "
          f"{100*M.max(0).mean():.2f}")
    print(f"  all-wrong (nothing can fix): {100*(1-M.max(0)).mean():.2f}\n")

    print("  pairwise error correlation (ratio > 1 = correlated, ~1 = independent):")
    print(f"  {'pair':<44} {'both_wrong':>10} {'indep':>8} {'ratio':>7}")
    for a, b in combinations(range(len(names)), 2):
        wa, wb = 1 - M[a], 1 - M[b]
        both = float((wa * wb).mean())
        indep = float(wa.mean() * wb.mean())
        ratio = both / indep if indep > 0 else float("nan")
        print(f"  {names[a][:20]+' | '+names[b][:20]:<44} {both:>10.3f} "
              f"{indep:>8.3f} {ratio:>7.2f}")

    print("\n  unique-right share (member correct where ALL others are wrong):")
    for i, nm in enumerate(names):
        others = np.delete(M, i, axis=0)
        uniq = float(((M[i] == 1) & (others.max(0) == 0)).mean())
        print(f"    {nm:<24} {100*uniq:.2f}%")


if __name__ == "__main__":
    main()
