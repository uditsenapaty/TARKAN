"""Re-threshold cached counterfactual arms into PDS direction labels — no GPU.

`masc_qwenvl.py --counterfactual` caches the two arms (`p_img`, `p_txt`) alongside the
mapped labels, so the "how large must a shift be to count" question can be answered
without re-running the teacher.

    python experts/remap_direction.py --src data/pds_qwen/twitter2015 --floor 0.20 \
        --out data/pds_qwen_f20/twitter2015
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from experts.common import POLARITIES  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True)
    ap.add_argument("--split", default="train")
    ap.add_argument("--floor", type=float, default=0.20)
    ap.add_argument("--temp", type=float, default=0.1)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    z = np.load(Path(args.src) / f"direction_{args.split}.npz")
    if "p_img" not in z:
        raise SystemExit(f"{args.src} has no cached arms — re-run the teacher with the "
                         f"current masc_qwenvl.py to get p_img/p_txt")
    d = z["p_img"].astype(np.float64) - z["p_txt"].astype(np.float64)
    NEG, NEU, POS = (POLARITIES.index(x) for x in ("NEG", "NEU", "POS"))
    pos_s = np.clip(d[:, POS] - d[:, NEU], 0, None)
    neg_s = np.clip(d[:, NEG] - d[:, NEU], 0, None)
    zz = np.stack([pos_s, neg_s, np.full_like(pos_s, args.floor)], 1)
    probs = np.exp((zz - zz.max(1, keepdims=True)) / args.temp)
    probs /= probs.sum(1, keepdims=True)

    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out / f"direction_{args.split}.npz", probs=probs.astype(np.float32),
                        keys=z["keys"], p_img=z["p_img"], p_txt=z["p_txt"])
    hard = probs.argmax(1)
    print(f"floor={args.floor} temp={args.temp} -> {out}")
    print("  distribution:", {n: int((hard == i).sum())
                              for i, n in enumerate(["POS", "NEG", "NONE"])})


if __name__ == "__main__":
    main()
