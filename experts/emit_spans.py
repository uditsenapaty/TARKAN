"""Emit candidate aspect anchors from an ENSEMBLE of MATE members.

MATE members are combined by averaging word-level tag marginals (never by voting on
decoded spans -- Chapter B, B3), so the ensemble's candidate set is not any single
member's `spans_*.json`. This writes the ensemble's set so MASC members can be re-scored
against it with `--score-only` instead of retrained.

    python experts/emit_spans.py --mate runs/mate_deb_s42 runs/mate_deb_s43 \
        --out runs/mate_ens
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from experts.assemble import decode_spans, load_mate  # noqa: E402
from experts.common import gold_spans, load, score_spans  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="twitter2015")
    ap.add_argument("--mate", nargs="+", required=True)
    ap.add_argument("--cand-thr", type=float, default=0.0)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    report = {"members": args.mate, "cand_thr": args.cand_thr}
    for split in ("dev", "test"):
        marg = load_mate(args.mate, split)
        cands = [decode_spans(m, args.cand_thr) for m in marg]
        spans = [[[s, e] for (s, e, _) in c] for c in cands]
        json.dump(spans, open(out / f"spans_{split}.json", "w"))
        sc = score_spans([[(s, e) for (s, e, _) in c] for c in cands],
                         gold_spans(load(args.dataset, split)))
        report[split] = sc
        print(f"[{split}] ensemble MATE P {sc['P']:.2f} R {sc['R']:.2f} F1 {sc['F1']:.2f} "
              f"({sum(len(s) for s in spans)} candidates)", flush=True)
    json.dump(report, open(out / "mate_ensemble.json", "w"), indent=2)


if __name__ == "__main__":
    main()
