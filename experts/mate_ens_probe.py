"""D21 — does ARCHITECTURAL diversity help the MATE ensemble?

All five MATE members in the standing ensemble are deberta-v3-large (seeds 42/43/44 +
probeA/B). §C.1 measured the seed lottery and concluded "more seeds is the wrong lever, the
recipe is", then fixed the recipe (§C.6, 87.01) and stopped. A different *architecture* is
neither a seed nor a recipe, and the polarity side carries seven backbones while the
extraction side carries one.

Members are combined the only way Chapter B found robust: averaging word-level tag
marginals, never voting on decoded spans.

    python experts/mate_ens_probe.py --extra runs/mate_btwL runs/mate_robL
"""
from __future__ import annotations

import argparse
import itertools
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from experts.assemble import decode_spans, load_mate  # noqa: E402
from experts.common import gold_spans, load, score_spans  # noqa: E402

DEB5 = ["runs/mate_deb_s42", "runs/mate_deb_s43", "runs/mate_deb_s44",
        "runs/mate_probeA", "runs/mate_probeB"]


def evaluate(members, gold, cand_thr=0.0):
    out = {}
    for split in ("dev", "test"):
        marg = load_mate(members, split)
        sp = [[(a, b) for (a, b, _) in decode_spans(m, cand_thr)] for m in marg]
        out[split] = score_spans(sp, gold[split])
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--extra", nargs="*", default=[])
    ap.add_argument("--cand-thr", type=float, default=0.0)
    args = ap.parse_args()
    gold = {s: gold_spans(load("twitter2015", s)) for s in ("dev", "test")}

    avail = [e for e in args.extra if (Path(e) / "marginals_test.npz").exists()]
    sets = {"5 deberta (standing)": DEB5}
    for e in avail:
        sets[f"5 deberta + {Path(e).name}"] = DEB5 + [e]
        sets[f"{Path(e).name} alone"] = [e]
    if len(avail) > 1:
        sets["5 deberta + both"] = DEB5 + avail
        sets["one-per-architecture (3)"] = ["runs/mate_probeA"] + avail

    print(f"{'MATE member set':32s} {'dev F1':>7} | {'test P':>7} {'test R':>7} {'test F1':>8}")
    for name, ms in sets.items():
        r = evaluate(ms, gold, args.cand_thr)
        star = "  <-- beats 87.01" if r["test"]["F1"] > 87.01 else ""
        print(f"{name:32s} {r['dev']['F1']:7.2f} | {r['test']['P']:7.2f} "
              f"{r['test']['R']:7.2f} {r['test']['F1']:8.2f}{star}")


if __name__ == "__main__":
    main()
