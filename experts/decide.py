"""D — the decision rule, as ONE command over the frozen pool.

`assemble.py` grew a mode flag, a rerank flag and a mix flag, and the §C.23/§C.25 results
still needed transcript scripts on top of it. This reads the frozen table from
`experts/pool.py` and applies the whole rule in one place:

    span evidence   S = geomean(tagger conf, C4 judge(s), PDQ-MATE span prob)
    polarity        p = softmax( w * log p_llm  +  (1-w) * log p_encoders )
    accept iff      geomean(S repeated k times, max p) > tau

`w` and `tau` are the only fitted quantities and both are tuned on dev. Two scalars is a
deliberate ceiling: §C.24(b) showed a 25-cell dev-fit acceptance surface loses 1.7 F1 to
the smooth 1-D product, and §C.26 showed dev cannot resolve differences of this size, so
every additional dev-fitted degree of freedom is a liability. `w` exists for one structural
reason (see `pool.py:build`): a log-average is equal-weight, so a single much stronger
member among fifteen weaker ones is averaged away rather than listened to.

    python experts/decide.py --pool pools/union16
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from experts.common import POLARITIES  # noqa: E402

BAR = {"P": 72.8, "R": 73.1, "F1": 72.9}
NGOLD = {"dev": 1122, "test": 1037}


def load(pool: Path, split: str):
    rows = json.load(open(pool / f"pool_{split}.json"))["rows"]
    S = np.array([np.exp(np.mean(np.log(np.clip(
        [r["tag"]] + r["judge"] + [r["pdqm"]], 1e-9, 1.0)))) for r in rows])
    A = np.log(np.clip(np.array([r["masc"] for r in rows]), 1e-9, 1.0))
    B = np.log(np.clip(np.array([r["masc2"] for r in rows]), 1e-9, 1.0))
    g = np.array([r["gold"] for r in rows])
    return rows, S, A, B, g


NEU = POLARITIES.index("NEU")


def mix(A, B, w, delta=None):
    """Log-space interpolation of the two MASC groups.

    Note this IS the residual form `z = z_base + alpha*(z_llm - z_base)` with alpha = w;
    there is no separate experiment to run for it.

    `delta` additionally restricts the correction to rows where the base is undecided at
    the NEU/minority boundary -- |p(NEU) - max(p(POS), p(NEG))| < delta -- which is where
    §D.1 measured 158 of the 186 polarity errors. This is NOT the §C.27 NEU-escape gate
    that failed: that one AMPLIFIED a residual trained from the same features exactly where
    the base was most often right (NEU recall 88.2). This one hands the decision to a
    *different model class* only where the base is admitting it cannot tell.
    """
    if delta is not None:
        pb = np.exp(A - A.max(1, keepdims=True))
        pb /= pb.sum(1, keepdims=True)
        minority = np.delete(pb, NEU, axis=1).max(1)
        undecided = np.abs(pb[:, NEU] - minority) < delta
        w = np.where(undecided, w, 0.0)[:, None]
    lg = (1 - w) * A + w * B
    p = np.exp(lg - lg.max(1, keepdims=True))
    return p / p.sum(1, keepdims=True)


def score(S, A, B, w, k, delta=None, qh=None, qmix=1.0):
    p = mix(A, B, w, delta)
    base = np.exp((k * np.log(np.clip(S, 1e-9, 1)) + np.log(p.max(1))) / (k + 1))
    if qh is None:
        return base, p.argmax(1)
    q = np.clip(qh, 1e-6, 1.0)
    return np.exp(qmix * np.log(q) + (1 - qmix) * np.log(np.clip(base, 1e-9, 1))), p.argmax(1)


def prf(tp, npred, ngold):
    P = 100.0 * tp / npred if npred else 0.0
    R = 100.0 * tp / ngold
    return {"P": P, "R": R, "F1": 2 * P * R / (P + R) if P + R else 0.0}


def evaluate(sc, j, g, tau, ngold):
    k = sc > tau
    tp = int(((g >= 0) & (j == g) & k).sum())
    out = prf(tp, int(k.sum()), ngold)
    span_tp = int(((g >= 0) & k).sum())
    out["MATE"] = prf(span_tp, int(k.sum()), ngold)["F1"]
    out["a"] = 100.0 * tp / span_tp if span_tp else 0.0
    out["n_pred"] = int(k.sum())
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pool", required=True)
    ap.add_argument("--k", type=int, default=3,
                    help="how many times the span evidence is weighted against polarity; "
                         "3 = the §C.23 equal-weight convention (tagger, judge, PDQ-MATE)")
    ap.add_argument("--w-grid", type=float, nargs="+",
                    default=[0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0])
    ap.add_argument("--gate-grid", type=float, nargs="*", default=[],
                    help="also try the selective gate at these deltas (see mix()). Costs a "
                         "third dev-fitted scalar, so it only wins if dev says so clearly.")
    ap.add_argument("--qhat", default=None,
                    help="dir with qhat_{dev,test}.npz from experts/qpredict.py -- a direct "
                         "estimate of P(pair correct). Micro-F1 is 2*TP/(|pred|+|gold|), so "
                         "the optimal acceptance cut is q > F1/2 (~0.353 here), not 0.5.")
    ap.add_argument("--qhat-mix", type=float, default=1.0,
                    help="1.0 = q_hat alone, 0.0 = the geometric product, in between = "
                         "geometric interpolation of the two")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    d = Path(args.pool)
    D = {s: load(d, s) for s in ("dev", "test")}
    QH = {}
    if args.qhat:
        for sp in ("dev", "test"):
            z = np.load(Path(args.qhat) / f"qhat_{sp}.npz")
            QH[sp] = {(int(k[0]), int(k[1]), int(k[2])): float(v)
                      for v, k in zip(z["q"], z["keys"])}

    taus = np.arange(0.0, 0.999, 0.005)
    print(f"{'w':>5} {'tau':>6} {'dev F1':>7} | {'TEST P':>7} {'R':>7} {'F1':>7} "
          f"{'MATE@t':>7} {'a':>6}")
    best = None
    rows_out = []
    for w in args.w_grid:
        qd = qt = None
        if QH:
            rows_d = json.load(open(d / "pool_dev.json"))["rows"]
            rows_t = json.load(open(d / "pool_test.json"))["rows"]
            qd = np.array([QH["dev"].get((r["inst"], r["s"], r["e"]), 0.5) for r in rows_d])
            qt = np.array([QH["test"].get((r["inst"], r["s"], r["e"]), 0.5) for r in rows_t])
        sd, jd = score(D["dev"][1], D["dev"][2], D["dev"][3], w, args.k, None, qd,
                       args.qhat_mix)
        st, jt = score(D["test"][1], D["test"][2], D["test"][3], w, args.k, None, qt,
                       args.qhat_mix)
        bd = max((evaluate(sd, jd, D["dev"][4], t, NGOLD["dev"])["F1"], t) for t in taus)
        dev_f1, tau = bd
        sc = evaluate(st, jt, D["test"][4], tau, NGOLD["test"])
        rows_out.append({"w": w, "tau": tau, "dev_F1": dev_f1, "test": sc})
        star = ""
        if best is None or dev_f1 > best[0]:
            best, star = (dev_f1, w, tau, sc), "  <- dev-best"
        print(f"{w:5.2f} {tau:6.3f} {dev_f1:7.2f} | {sc['P']:7.2f} {sc['R']:7.2f} "
              f"{sc['F1']:7.2f} {sc['MATE']:7.2f} {sc['a']:6.2f}{star}")

    if args.gate_grid:
        print(f"\nSELECTIVE GATE — hand the decision to group 2 only where the base is "
              f"undecided at the NEU/minority boundary")
        print(f"{'w':>5} {'delta':>6} {'tau':>6} {'dev F1':>7} | {'TEST F1':>7} {'a':>6}")
        for w in args.w_grid:
            if w == 0.0:
                continue
            for delta in args.gate_grid:
                sd, jd = score(D["dev"][1], D["dev"][2], D["dev"][3], w, args.k, delta)
                st, jt = score(D["test"][1], D["test"][2], D["test"][3], w, args.k, delta)
                dev_f1, tau = max((evaluate(sd, jd, D["dev"][4], t, NGOLD["dev"])["F1"], t)
                                  for t in taus)
                sc = evaluate(st, jt, D["test"][4], tau, NGOLD["test"])
                rows_out.append({"w": w, "delta": delta, "tau": tau, "dev_F1": dev_f1,
                                 "test": sc})
                star = ""
                if dev_f1 > best[0]:
                    best, star = (dev_f1, w, tau, sc), "  <- dev-best"
                if star or dev_f1 > best[0] - 0.4:
                    print(f"{w:5.2f} {delta:6.2f} {tau:6.3f} {dev_f1:7.2f} | "
                          f"{sc['F1']:7.2f} {sc['a']:6.2f}{star}")

    dev_f1, w, tau, sc = best
    m = min(sc["P"] - BAR["P"], sc["R"] - BAR["R"], sc["F1"] - BAR["F1"])
    print("\n" + "=" * 68)
    print(f"DEV-SELECTED (w={w:.2f}, tau={tau:.3f}, dev {dev_f1:.2f})")
    print(f"  MATE@tau {sc['MATE']:.2f}   a {sc['a']:.2f}   "
          f"identity {sc['MATE']*sc['a']/100:.2f}")
    print(f"  JOINT  P {sc['P']:.2f}  R {sc['R']:.2f}  F1 {sc['F1']:.2f}")
    print(f"  BAR (MADSC) P>{BAR['P']} R>{BAR['R']} F1>{BAR['F1']}  ->  worst margin "
          f"{m:+.2f}")
    print("=" * 68)
    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        json.dump({"grid": rows_out, "selected": {"w": w, "tau": tau, "dev_F1": dev_f1,
                                                  "test": sc, "margin": m}},
                  open(args.out, "w"), indent=2, default=float)


if __name__ == "__main__":
    main()
