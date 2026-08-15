"""Stratified MASC comparison — does a fusion head help where the hypothesis says it should?

An aggregate delta cannot distinguish "the visual path works" from "the visual path is
inert and the number moved by luck". TARKAN's claim is specifically that aspect-conditioned
visual evidence helps when the image is relevant to *that* aspect. So the delta should
concentrate in the image-useful stratum and vanish in the image-irrelevant one.

Strata come from artifacts that already exist, so this costs nothing:
  * `u` — the TRAIN-ECDF-calibrated aspect-image similarity in `aspect_{split}.npz`.
    Terciles give image-irrelevant / weak-correspondence / image-useful.
  * multi-aspect — instances carrying more than one gold aspect (t2015 averages 1.51,
    and 444 within-tweet pairs disagree in polarity).

    python experts/strata.py --arms runs/k70_B_mlp_tv runs/k70_D_ikan_tv --split test
"""
from __future__ import annotations

import argparse
import sys
from collections import Counter
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from experts.common import DATA, POL2ID, load, masc_examples  # noqa: E402
from experts.masc_gated import load_aspect_npz  # noqa: E402


def build(split: str, dataset: str = "twitter2015"):
    insts = load(dataset, split)
    ex = masc_examples(insts)
    gold = {e.key: POL2ID[e.polarity] for e in ex}
    _, u, _ = load_aspect_npz(DATA / "aadg" / dataset / f"aspect_{split}.npz")
    n_asp = Counter(e.inst_idx for e in ex)
    uu = np.array([u.get(e.key, 0.0) for e in ex])
    lo, hi = np.quantile(uu, [1 / 3, 2 / 3])
    strat = {}
    for e in ex:
        v = u.get(e.key, 0.0)
        s = "image-irrelevant" if v <= lo else ("weak-corresp" if v <= hi else "image-useful")
        strat[e.key] = (s, "multi-aspect" if n_asp[e.inst_idx] > 1 else "single-aspect")
    return gold, strat


def preds(run: Path, split: str):
    z = np.load(run / f"probs_{split}.npz")
    return {tuple(int(x) for x in k): int(p.argmax()) for k, p in zip(z["keys"], z["probs"])}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arms", nargs="+", required=True, help="baseline first")
    ap.add_argument("--split", default="test")
    ap.add_argument("--dataset", default="twitter2015")
    args = ap.parse_args()

    gold, strat = build(args.split, args.dataset)
    P = {a: preds(Path(a), args.split) for a in args.arms}
    keys = sorted(set(gold) & set.intersection(*[set(p) for p in P.values()]))

    groups = ["image-irrelevant", "weak-corresp", "image-useful",
              "single-aspect", "multi-aspect", "ALL"]
    print(f"{'stratum':<18}{'n':>6}" + "".join(f"{Path(a).name[:14]:>16}" for a in args.arms)
          + f"{'delta':>9}")
    for g in groups:
        if g == "ALL":
            ks = keys
        elif g in ("single-aspect", "multi-aspect"):
            ks = [k for k in keys if strat[k][1] == g]
        else:
            ks = [k for k in keys if strat[k][0] == g]
        if not ks:
            continue
        accs = [100.0 * np.mean([P[a][k] == gold[k] for k in ks]) for a in args.arms]
        print(f"{g:<18}{len(ks):>6}" + "".join(f"{v:16.2f}" for v in accs)
              + f"{accs[-1]-accs[0]:+9.2f}")


if __name__ == "__main__":
    main()
