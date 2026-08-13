"""C28b — what is actually left on the table, measured on the frozen pool.

§C.24 concluded "the selection axis is saturated" from four negatives, but each of those
was fitted on DEV (n=1122, binomial sigma ~1.2) and each rebuilt the pool. This asks a
sharper question against one frozen pool: **how much of the 84.59 perfect-selector ceiling
is reachable from the signals we already have?**

The answer is the fork in the road:
  * if a TEST-FITTED selector over the existing features barely beats the geometric
    product, the features are exhausted and only a new model can help;
  * if it clears the bar comfortably, the signal is present and the problem is purely one
    of estimation -- fit it on more data, not on dev.

Everything here is diagnostic. Nothing it prints is a reportable system score.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from experts.common import POLARITIES  # noqa: E402


def load_pool(d: Path, split: str):
    return json.load(open(d / f"pool_{split}.json"))["rows"]


def f1(tp: int, npred: int, ngold: int):
    p = 100.0 * tp / npred if npred else 0.0
    r = 100.0 * tp / ngold
    return {"P": p, "R": r, "F1": 2 * p * r / (p + r) if p + r else 0.0}


def correct(r) -> bool:
    return r["gold"] >= 0 and int(np.argmax(r["masc"])) == r["gold"]


def geo(r, use=("tag", "judge", "pdqm", "masc")) -> float:
    """The §C.23 evidence product: equal-weight geometric mean of every signal."""
    v = []
    if "tag" in use:
        v.append(r["tag"])
    if "judge" in use:
        v += r["judge"]
    if "pdqm" in use:
        v.append(r["pdqm"])
    if "masc" in use:
        v.append(max(r["masc"]))
    return float(np.exp(np.mean(np.log(np.clip(v, 1e-9, 1.0)))))


def sweep(rows_dev, rows_test, score_dev, score_test, ngold_dev, ngold_test):
    """Tune tau on dev, report test. The only selection-free protocol available."""
    best = (-1.0, 0.0)
    for tau in np.arange(0.0, 0.999, 0.005):
        tp = sum(correct(r) for r, s in zip(rows_dev, score_dev) if s > tau)
        n = sum(1 for s in score_dev if s > tau)
        sc = f1(tp, n, ngold_dev)
        if sc["F1"] > best[0]:
            best = (sc["F1"], float(tau))
    dev_f1, tau = best
    tp = sum(correct(r) for r, s in zip(rows_test, score_test) if s > tau)
    n = sum(1 for s in score_test if s > tau)
    return tau, dev_f1, f1(tp, n, ngold_test)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pool", default="pools/best15")
    args = ap.parse_args()
    d = Path(args.pool)
    dev, test = load_pool(d, "dev"), load_pool(d, "test")
    NG = {"dev": 1122, "test": 1037}

    print("=" * 74)
    print("POOL")
    for name, rows in (("dev", dev), ("test", test)):
        gs = sum(1 for r in rows if r["gold"] >= 0)
        ok = sum(correct(r) for r in rows)
        ceil = f1(ok, ok, NG[name])
        print(f"  {name}: {len(rows):5d} cand | {gs:4d} gold spans | {ok:4d} with correct "
              f"polarity -> PERFECT-SELECTOR F1 {ceil['F1']:.2f} (R {ceil['R']:.2f})")

    print("=" * 74)
    print("BASELINE — §C.23 equal-weight geometric evidence product")
    sd = [geo(r) for r in dev]
    st = [geo(r) for r in test]
    tau, dev_f1, sc = sweep(dev, test, sd, st, NG["dev"], NG["test"])
    print(f"  tau={tau:.3f} dev {dev_f1:.2f} | TEST P {sc['P']:.2f} R {sc['R']:.2f} "
          f"F1 {sc['F1']:.2f}")
    base_tau = tau

    # what the kept set is made of, at the dev-selected operating point
    kept = [r for r, s in zip(test, st) if s > base_tau]
    tp = sum(correct(r) for r in kept)
    wrongpol = sum(1 for r in kept if r["gold"] >= 0 and not correct(r))
    nongold = sum(1 for r in kept if r["gold"] < 0)
    dropped_ok = sum(correct(r) for r, s in zip(test, st) if s <= base_tau)
    print(f"  kept {len(kept)}: {tp} correct | {wrongpol} gold-span/wrong-polarity | "
          f"{nongold} non-gold   (dropped {dropped_ok} correct)")
    need_fp = len(kept) + NG['test'] - int(round(2 * tp / 0.729))
    need_flip = int(np.ceil((0.729 * (len(kept) + NG["test"]) - 2 * tp) / 2))
    print(f"  TO REACH 72.9 from here: drop {need_fp} false positives at constant TP, "
          f"OR flip {need_flip} wrong polarities")

    print("=" * 74)
    print("ABLATION — which signals carry the product")
    for use in (("tag",), ("tag", "judge"), ("tag", "judge", "pdqm"),
                ("tag", "judge", "masc"), ("tag", "judge", "pdqm", "masc")):
        sd = [geo(r, use) for r in dev]
        st = [geo(r, use) for r in test]
        t, dv, sc = sweep(dev, test, sd, st, NG["dev"], NG["test"])
        print(f"  {'+'.join(use):26s} dev {dv:.2f}  TEST {sc['F1']:.2f}")

    print("=" * 74)
    print("FREE CHECKS — structural filters that cost no training")
    st = [geo(r) for r in test]
    sd = [geo(r) for r in dev]
    for name, keep in (
        ("drop overlapping (lower-scored)", lambda r: r["overlap"] == 0),
        ("drop len>=4 spans", lambda r: r["len"] < 4),
    ):
        sd2 = [s if keep(r) else -1.0 for r, s in zip(dev, sd)]
        st2 = [s if keep(r) else -1.0 for r, s in zip(test, st)]
        t, dv, sc = sweep(dev, test, sd2, st2, NG["dev"], NG["test"])
        n_removed = sum(1 for r in test if not keep(r))
        n_removed_ok = sum(1 for r in test if not keep(r) and correct(r))
        print(f"  {name:34s} removes {n_removed:3d} ({n_removed_ok} correct) "
              f"dev {dv:.2f} TEST {sc['F1']:.2f}")

    print("=" * 74)
    print("★ THE FORK — how much do the EXISTING features allow, if fitted perfectly?")
    from sklearn.linear_model import LogisticRegression

    def feats(rows):
        X = []
        for r in rows:
            m = np.array(r["masc"])
            srt = np.sort(m)[::-1]
            X.append([np.log(max(r["tag"], 1e-6))] +
                     [np.log(max(x, 1e-6)) for x in r["judge"]] +
                     [np.log(max(r["pdqm"], 1e-6)),
                      np.log(max(srt[0], 1e-6)), srt[0] - srt[1],
                      float(-(m * np.log(np.clip(m, 1e-9, 1))).sum()),
                      float(np.argmax(m) == POLARITIES.index("NEU")),
                      r["rel"], np.log(r["n_cand"]), r["len"], r["overlap"]])
        return np.array(X)

    ydev = np.array([correct(r) for r in dev], dtype=int)
    ytest = np.array([correct(r) for r in test], dtype=int)
    Xdev, Xtest = feats(dev), feats(test)
    mu, sd_ = Xdev.mean(0), Xdev.std(0) + 1e-6

    for name, (Xf, yf) in (("fitted on DEV (honest)", (Xdev, ydev)),
                           ("fitted on TEST (CHEATING — ceiling only)", (Xtest, ytest))):
        lr = LogisticRegression(max_iter=2000, C=1.0).fit((Xf - mu) / sd_, yf)
        pd_, pt = (lr.predict_proba((Xdev - mu) / sd_)[:, 1],
                   lr.predict_proba((Xtest - mu) / sd_)[:, 1])
        t, dv, sc = sweep(dev, test, pd_, pt, NG["dev"], NG["test"])
        print(f"  {name:44s} dev {dv:.2f}  TEST P {sc['P']:.2f} R {sc['R']:.2f} "
              f"F1 {sc['F1']:.2f}")


if __name__ == "__main__":
    main()
