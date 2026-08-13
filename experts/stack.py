"""C8 — OOF stacking: the combiner fix Chapter B never tried.

Chapter B's §7x verdict was that the ensemble is stuck at `a` ~81 not because the diversity
is missing (oracle 93.4, reproduced here at 93.44) but because **no realizable combiner
extracts it**: Dawid-Skene, spectral, dev-confusion, dev-weighting, per-class log-bias,
confidence-gating and max-conf routing were all fit on **dev**, all gained ~+0.9 dev, and
all LOST on test. Dev has n=1122 and a binomial sigma of ~1.2, which is larger than the
entire dev/test gap being chased -- so those experiments could not have worked.

This fits the combiner on **train out-of-fold predictions** instead: 3179 aspects, 2.8x dev,
and never seen by the member that predicts them. That is the standard stacking protocol and
the one thing the failed-combiner list has in common by omission.

Dev is then left to do the only job it can do reliably: pick between `stack` and the
selection-free log-average baseline.

    python experts/stack.py --masc runs/masc_btwL_s45 runs/masc_deb_s44 ...
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from experts.common import POL2ID, POLARITIES, load, masc_examples  # noqa: E402

Key = Tuple[int, int, int]


def read(path: Path) -> Dict[Key, np.ndarray]:
    if not path.exists():
        return {}
    z = np.load(path, allow_pickle=True)
    return {(int(k[0]), int(k[1]), int(k[2])): p.astype(np.float64)
            for p, k in zip(z["probs"], z["keys"])}


def matrix(dirs: List[str], files: List[str], keys: List[Key]) -> np.ndarray:
    """[n_keys, n_members*3] design matrix of log-probabilities."""
    cols = []
    for d in dirs:
        m: Dict[Key, np.ndarray] = {}
        for f in files:
            m.update(read(Path(d) / f))
        cols.append(np.stack([np.log(np.clip(m.get(k, np.full(3, 1 / 3)), 1e-6, 1))
                              for k in keys]))
    return np.concatenate(cols, axis=1)


def logavg(dirs: List[str], files: List[str], keys: List[Key]) -> np.ndarray:
    acc = np.zeros((len(keys), 3))
    for d in dirs:
        m: Dict[Key, np.ndarray] = {}
        for f in files:
            m.update(read(Path(d) / f))
        acc += np.stack([np.log(np.clip(m.get(k, np.full(3, 1 / 3)), 1e-6, 1))
                         for k in keys])
    e = np.exp(acc / len(dirs))
    return e / e.sum(1, keepdims=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="twitter2015")
    ap.add_argument("--masc", nargs="+", required=True)
    ap.add_argument("--C", type=float, default=1.0)
    args = ap.parse_args()

    from sklearn.linear_model import LogisticRegression

    # members that actually produced OOF predictions
    dirs = [d for d in args.masc
            if (Path(d) / "probs_oof0.npz").exists() or (Path(d) / "probs_oof1.npz").exists()]
    if not dirs:
        print("no OOF predictions found -- run masc_text.py --fold 0 / --fold 1 first")
        return
    print(f"{len(dirs)} members with OOF: {[Path(d).name for d in dirs]}")

    tr_ex = masc_examples(load(args.dataset, "train"))
    gold_tr = {e.key: POL2ID[e.polarity] for e in tr_ex}
    oof_keys = sorted(set(gold_tr) & set(
        k for d in dirs for f in ("probs_oof0.npz", "probs_oof1.npz")
        for k in read(Path(d) / f)))
    print(f"OOF fitting points: {len(oof_keys)}")

    X = matrix(dirs, ["probs_oof0.npz", "probs_oof1.npz"], oof_keys)
    y = np.array([gold_tr[k] for k in oof_keys])
    clf = LogisticRegression(max_iter=2000, C=args.C)
    clf.fit(X, y)
    print(f"stacker fit: train-OOF acc {100*clf.score(X, y):.2f}")

    for split in ("dev", "test"):
        ex = masc_examples(load(args.dataset, split))
        gold = {e.key: POL2ID[e.polarity] for e in ex}
        keys = sorted(gold)
        Xs = matrix(dirs, [f"probs_{split}.npz"], keys)
        ys = np.array([gold[k] for k in keys])
        acc_stack = 100.0 * float((clf.predict(Xs) == ys).mean())
        acc_log = 100.0 * float((logavg(dirs, [f"probs_{split}.npz"], keys).argmax(1) == ys).mean())
        print(f"[{split}] gold-span MASC acc  log-avg {acc_log:.2f}  |  STACK {acc_stack:.2f}"
              f"   (delta {acc_stack-acc_log:+.2f})")

        # emit stacked probabilities over the predicted-span set for assemble.py
        span_keys = sorted(set(k for d in dirs for k in read(Path(d) / f"probs_span_{split}.npz")))
        if span_keys:
            Xp = matrix(dirs, [f"probs_span_{split}.npz"], span_keys)
            P = clf.predict_proba(Xp)
            out = Path("runs/stack_masc"); out.mkdir(parents=True, exist_ok=True)
            np.savez_compressed(out / f"probs_span_{split}.npz", probs=P,
                                keys=np.array(span_keys, dtype=np.int64))
            np.savez_compressed(out / f"probs_{split}.npz",
                                probs=clf.predict_proba(Xs),
                                keys=np.array(keys, dtype=np.int64))
            print(f"      wrote stacked probs for {len(span_keys)} predicted spans")


if __name__ == "__main__":
    main()
