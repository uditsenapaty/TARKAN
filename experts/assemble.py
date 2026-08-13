"""C3 / B3 — two-stage assembly with JOINT-expected-F1 decoding.

Stage 1 (candidate aspect anchors) -> stage 2 (polarity on those anchors), exactly the
inference flow of the new TARKAN manuscript (§3.3: "at inference, candidate anchors are
obtained solely from preliminary predictions") and of DQPSA's own
`eval_MABSA(MATE_model, MASC_model, ...)`.

Two things here are new relative to Chapter B:

**C3 — threshold the PRODUCT, not the span.** Chapter B tuned a single span confidence
`tau` and left polarity untouched. But the Table-1 metric counts a pair only when BOTH the
span and the polarity are right, so a candidate's expected contribution to a true positive
is `P(span) * P(polarity)`. A span with crisp boundaries and coin-flip polarity is
*negative* expected value at the margin and should be dropped. DQPSA independently does a
two-threshold version of this (`MATE_limit=0.5, MASC_limit=0.3`); thresholding the product
is the principled unification and costs nothing to evaluate.

MATE members are ensembled by averaging **word-level tag marginals** (never by voting on
decoded spans); MASC members by **log-average** (geometric mean) -- both are the
Chapter-B-measured robust choices.

    python experts/assemble.py --mate runs/mate_deb_s42 --masc runs/pdq_s42 --objective margin
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from experts.common import (POLARITIES, gold_pairs, gold_spans, load,  # noqa: E402
                            polarity_on_extracted, score_joint, score_spans,
                            spans_from_obi)

# the new bar: MADSC (PR 2026) on t2015. Old bar was VLHA 72.3/72.7/72.5.
BAR = {"P": 72.8, "R": 73.1, "F1": 72.9}
O, B, I = 0, 1, 2


# --------------------------------------------------------------------------- #
# members
# --------------------------------------------------------------------------- #
def load_mate(dirs: Sequence[str], split: str) -> List[np.ndarray]:
    """Average word-level tag marginals across MATE members."""
    acc = None
    for d in dirs:
        z = np.load(Path(d) / f"marginals_{split}.npz")
        cur = [z[str(i)] for i in range(len(z.files))]
        if acc is None:
            acc = [c.astype(np.float64) for c in cur]
        else:
            for i, c in enumerate(cur):
                acc[i] += c
    return [a / len(dirs) for a in acc]


def load_masc(dirs: Sequence[str], split: str) -> Dict[Tuple[int, int, int], np.ndarray]:
    """Log-average (geometric mean) of member polarity distributions, keyed by span."""
    logsum: Dict[Tuple[int, int, int], np.ndarray] = {}
    count: Dict[Tuple[int, int, int], int] = {}
    for d in dirs:
        # probs_span_* = scored on stage-1 candidate anchors (what we need);
        # probs_* = gold-span probs, used only if a member was run without --spans.
        f = Path(d) / f"probs_span_{split}.npz"
        if not f.exists():
            f = Path(d) / f"probs_{split}.npz"
        z = np.load(f, allow_pickle=True)
        probs, keys = z["probs"], z["keys"]
        for p, k in zip(probs, keys):
            key = (int(k[0]), int(k[1]), int(k[2]))
            lp = np.log(np.clip(p.astype(np.float64), 1e-9, 1.0))
            logsum[key] = logsum.get(key, 0.0) + lp
            count[key] = count.get(key, 0) + 1
    out = {}
    for k, v in logsum.items():
        e = np.exp(v / count[k])
        out[k] = e / e.sum()
    return out


# --------------------------------------------------------------------------- #
# stage 1 decode
# --------------------------------------------------------------------------- #
def decode_spans(marg: np.ndarray, cand_thr: float = 0.0
                 ) -> List[Tuple[int, int, float]]:
    """Decode candidate anchors + a confidence per span from averaged marginals.

    cand_thr = 0 -> plain argmax over {O,B,I} (the standard operating point).
    cand_thr > 0 -> a token counts as aspect-bearing whenever P(B)+P(I) > cand_thr,
    which trades precision for recall and gives the product-threshold something to
    select from.
    """
    if cand_thr <= 0:
        tags = marg.argmax(-1).tolist()
    else:
        nonO = marg[:, B] + marg[:, I]
        tags = []
        for t in range(len(marg)):
            if nonO[t] > cand_thr:
                tags.append(B if marg[t, B] >= marg[t, I] else I)
            else:
                tags.append(O)
    spans = spans_from_obi(tags)
    out = []
    for (s, e) in spans:
        conf = float(np.mean(1.0 - marg[s:e, O]))
        out.append((s, e, conf))
    return out


# --------------------------------------------------------------------------- #
# assembly
# --------------------------------------------------------------------------- #
def assemble(cands: List[List[Tuple[int, int, float]]],
             pol: Dict[Tuple[int, int, int], np.ndarray],
             tau: float, mode: str) -> List[List[Tuple[int, int, str]]]:
    out = []
    for i, spans in enumerate(cands):
        pairs = []
        for (s, e, conf) in spans:
            p = pol.get((i, s, e))
            if p is None:
                continue
            j = int(np.argmax(p))
            score = conf if mode == "span" else conf * float(p[j])
            if score > tau:
                pairs.append((s, e, POLARITIES[j]))
        out.append(pairs)
    return out


def margin(sc: Dict[str, float]) -> float:
    """Worst per-cell margin against the bar. F1 alone is blind to a P/R imbalance
    that fails an individual cell, and the bar is per-cell."""
    return min(sc["P"] - BAR["P"], sc["R"] - BAR["R"], sc["F1"] - BAR["F1"])


def tune_tau(cands, pol, gold, mode: str, objective: str) -> Tuple[float, Dict[str, float]]:
    best_tau, best_val, best_sc = 0.0, -1e9, None
    for tau in np.arange(0.0, 0.96, 0.01):
        pairs = assemble(cands, pol, float(tau), mode)
        sc = score_joint(pairs, gold)
        val = margin(sc) if objective == "margin" else sc["F1"]
        if val > best_val:
            best_tau, best_val, best_sc = float(tau), val, sc
    return best_tau, best_sc


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="twitter2015")
    ap.add_argument("--mate", nargs="+", required=True)
    ap.add_argument("--masc", nargs="+", required=True)
    ap.add_argument("--mode", choices=["span", "joint"], default="joint",
                    help="'span' = Chapter-B tau on span confidence; "
                         "'joint' = C3, tau on P(span)*P(polarity)")
    ap.add_argument("--objective", choices=["margin", "f1"], default="margin")
    ap.add_argument("--cand-thr", type=float, default=0.0)
    ap.add_argument("--rerank", default=None,
                    help="dir with spanscore_{dev,test}.npz from experts/span_rerank.py; "
                         "replaces the tagger's mean(1-P(O)) span confidence")
    ap.add_argument("--rerank-mix", type=float, default=1.0,
                    help="1.0 = reranker only, 0.0 = tagger only, in between = geometric mix")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    insts = {s: load(args.dataset, s) for s in ("dev", "test")}
    gold = {s: gold_pairs(v) for s, v in insts.items()}

    res = {"mate_members": args.mate, "masc_members": args.masc,
           "mode": args.mode, "objective": args.objective, "cand_thr": args.cand_thr,
           "bar": BAR}
    cands, pol = {}, {}
    for s in ("dev", "test"):
        marg = load_mate(args.mate, s)
        cands[s] = [decode_spans(m, args.cand_thr) for m in marg]
        if args.rerank:
            z = np.load(Path(args.rerank) / f"spanscore_{s}.npz")
            rs = {(int(k[0]), int(k[1]), int(k[2])): float(v)
                  for v, k in zip(z["score"], z["keys"])}
            mix = args.rerank_mix
            cands[s] = [[(a, b, (rs.get((i, a, b), c) ** mix) * (c ** (1 - mix)))
                         for (a, b, c) in inst] for i, inst in enumerate(cands[s])]
        pol[s] = load_masc(args.masc, s)
        sp = [[(a, b) for (a, b, _) in c] for c in cands[s]]
        res[f"{s}_MATE"] = score_spans(sp, gold_spans(insts[s]))

    tau, dev_sc = tune_tau(cands["dev"], pol["dev"], gold["dev"], args.mode, args.objective)
    res["tau"] = tau
    res["dev_joint"] = dev_sc

    test_pairs = assemble(cands["test"], pol["test"], tau, args.mode)
    test_sc = score_joint(test_pairs, gold["test"])
    res["test_joint"] = test_sc
    res["test_a"] = polarity_on_extracted(test_pairs, gold["test"])
    res["test_MATE_at_tau"] = score_spans(
        [[(a, b) for (a, b, _) in p] for p in test_pairs], gold_spans(insts["test"]))
    res["margin_vs_bar"] = margin(test_sc)

    m = res["test_MATE_at_tau"]
    print(f"tau={tau:.2f} mode={args.mode}")
    print(f"MATE   P={m['P']:.2f} R={m['R']:.2f} F1={m['F1']:.2f}")
    print(f"a (polarity on correctly-extracted) = {res['test_a']:.2f}")
    print(f"JOINT  P={test_sc['P']:.2f} R={test_sc['R']:.2f} F1={test_sc['F1']:.2f}"
          f"   (dev F1 {dev_sc['F1']:.2f})")
    print(f"identity check: {m['F1']:.2f} x {res['test_a']/100:.4f} = "
          f"{m['F1']*res['test_a']/100:.2f}")
    print(f"BAR P>{BAR['P']} R>{BAR['R']} F1>{BAR['F1']}  ->  worst margin "
          f"{res['margin_vs_bar']:+.2f}")
    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        json.dump(res, open(args.out, "w"), indent=2, default=float)


if __name__ == "__main__":
    main()
