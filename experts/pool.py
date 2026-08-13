"""C28 — materialise the candidate pool with EVERY decision signal attached.

Chapter C ends with the §C.23 evidence product (tagger x judge x PDQ-MATE x MASC,
equal weights, geometric) as the decision rule, reached through a chain of ad-hoc
scripts. Two problems with that:

  * the 71.37 configuration is not reproducible from one command, and
  * every attempt to improve the *selection* (§C.24: tau sweep, 2-D dev-fit surface,
    judge ensemble, more members) was measured against a pool that was rebuilt each
    time, so a negative could never be attributed to the selector rather than the pool.

This module builds the pool ONCE and writes a flat table -- one row per candidate span,
with the tagger confidence, the C4 judge score, the PDQ-MATE span probability, the
log-averaged MASC distribution, sentence-level context features, and the gold label
(-1 = not a gold span, else the gold polarity index). Everything downstream -- the
existing geometric product, any calibrated selector, the perfect-selector ceiling, the
false-positive anatomy -- then reads the same frozen table.

    python experts/pool.py --out pools/best --mate ... --masc ... --rerank ... --pdqmate ...
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List, Sequence

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from experts.assemble import decode_spans, load_mate, load_masc  # noqa: E402
from experts.common import POLARITIES, load  # noqa: E402

PDQM_FLOOR = 0.05   # PDQ-MATE writes O=1.0 for spans it did not decode; without a floor
                    # a single abstention becomes a hard veto inside the geometric product
                    # (the §C.23 failure mode when it entered as an averaging member).


def pdqmate_marginals(dirs: Sequence[str], split: str) -> List[np.ndarray]:
    """PDQ-MATE's decoded pseudo-marginals, averaged over sources (loaded once)."""
    return load_mate(dirs, split)


def judge_scores(dirs: Sequence[str], split: str) -> List[Dict]:
    per = []
    for d in dirs:
        z = np.load(Path(d) / f"spanscore_{split}.npz")
        per.append({(int(k[0]), int(k[1]), int(k[2])): float(v)
                    for v, k in zip(z["score"], z["keys"])})
    return per


def build(dataset: str, split: str, mate: Sequence[str], masc: Sequence[str],
          rerank: Sequence[str], pdqmate: Sequence[str], cand_thr: float,
          masc2: Sequence[str] = (), spans: str = None) -> Dict:
    insts = load(dataset, split)
    marg = load_mate(mate, split)
    if spans:                      # candidate set fixed externally (e.g. a union pool)
        fixed = json.load(open(Path(spans) / f"spans_{split}.json"))
        cands = [[(int(a), int(b), float(np.mean(1.0 - m[int(a):int(b), 0])))
                  for (a, b) in sp] for sp, m in zip(fixed, marg)]
    else:
        cands = [decode_spans(m, cand_thr) for m in marg]
    pol = load_masc(masc, split)
    # A second MASC group is kept SEPARATE rather than merged into the log-average.
    # The ensemble is an equal-weight geometric mean, so one much stronger member among
    # fifteen weaker ones contributes 1/16 of the log-odds and is simply averaged away.
    # Storing the groups apart lets a single dev-tuned mixing weight decide -- one scalar,
    # not the 25-cell dev-fit surface that overfitted in §C.24(b).
    pol2 = load_masc(masc2, split) if masc2 else {}
    judges = judge_scores(rerank, split) if rerank else []
    pdqm = pdqmate_marginals(pdqmate, split) if pdqmate else None

    rows, feats = [], []
    for i, (inst, spans) in enumerate(zip(insts, cands)):
        gmap = {(s, e): POLARITIES.index(p) for (s, e, p) in inst.aspects}
        n = len(spans)
        confs = [c for (_, _, c) in spans]
        cmax = max(confs) if confs else 1.0
        for (s, e, conf) in spans:
            key = (i, s, e)
            p = pol.get(key)
            p2 = pol2.get(key)
            if p is None and p2 is None:
                continue
            if p is None:
                p = p2
            if p2 is None:
                p2 = p
            jd = [j.get(key, conf) for j in judges]
            pm = (float(np.clip(np.mean(1.0 - pdqm[i][s:e, 0]), PDQM_FLOOR, 1.0))
                  if pdqm is not None else 1.0)
            # overlap: does a higher-confidence candidate in this sentence share a token?
            ov = int(any(a < e and s < b and c > conf for (a, b, c) in spans))
            rows.append({
                "inst": i, "s": s, "e": e,
                "gold": gmap.get((s, e), -1),
                "tag": float(conf),
                "judge": [float(x) for x in jd],
                "pdqm": pm,
                "masc": [float(x) for x in p],
                "masc2": [float(x) for x in p2],
                "n_cand": n, "rel": float(conf / cmax) if cmax > 0 else 1.0,
                "len": e - s, "overlap": ov,
            })
    return {"rows": rows, "n_sent": len(insts),
            "n_gold": sum(len(x.aspects) for x in insts)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="twitter2015")
    ap.add_argument("--mate", nargs="+", required=True)
    ap.add_argument("--masc", nargs="+", required=True)
    ap.add_argument("--masc2", nargs="*", default=[],
                    help="second MASC group, kept separate from the log-average so a "
                         "single dev-tuned weight can mix the two (see build())")
    ap.add_argument("--rerank", nargs="*", default=[])
    ap.add_argument("--pdqmate", nargs="*", default=[])
    ap.add_argument("--cand-thr", type=float, default=0.12)
    ap.add_argument("--spans", default=None,
                    help="use this externally-fixed candidate set (spans_{split}.json) "
                         "instead of decoding at --cand-thr")
    ap.add_argument("--splits", nargs="+", default=["dev", "test"])
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    meta = {k: v for k, v in vars(args).items()}
    for split in args.splits:
        d = build(args.dataset, split, args.mate, args.masc, args.rerank,
                  args.pdqmate, args.cand_thr, args.masc2, args.spans)
        json.dump(d, open(out / f"pool_{split}.json", "w"))
        ng = sum(1 for r in d["rows"] if r["gold"] >= 0)
        print(f"[{split}] {len(d['rows'])} candidates, {ng} hit a gold span "
              f"({d['n_gold']} gold pairs in {d['n_sent']} sentences)", flush=True)
        meta[split] = {"n_cand": len(d["rows"]), "n_gold_span_hits": ng,
                       "n_gold": d["n_gold"]}
    json.dump(meta, open(out / "meta.json", "w"), indent=2, default=str)


if __name__ == "__main__":
    main()
