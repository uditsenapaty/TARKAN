"""Tune the neurosymbolic inference layer (A12/A13/A14) on DEV against a frozen checkpoint.

Zero training: each grid point is an eval pass. Picks the best dev joint-F1 config,
then reports TEST with it (and appends a row to iterations.csv tagged NS_<dataset>).

Usage: python scripts/tune_neurosymbolic.py --dataset twitter2015 [--checkpoint path]
"""
from __future__ import annotations

import argparse
import csv
import itertools
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from config import CONFIG, cfg_for  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="twitter2015")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--checkpoint", default=None)
    ap.add_argument("--conf-append", action="store_true", help="A9 was on in the checkpoint (widened KAN input)")
    ap.add_argument("--feat-gate", action="store_true", help="A10 was on in the checkpoint (gate params)")
    ap.add_argument("--text-model", default=None, help="text encoder id used by the checkpoint")
    args = ap.parse_args()

    CONFIG.fusion_conf_append = bool(args.conf_append)
    CONFIG.fusion_feat_gate = bool(args.feat_gate)
    extra = {"text_model_id": args.text_model} if args.text_model else {}
    cfg = cfg_for(args.dataset, device=args.device, **extra)
    ckpt = Path(args.checkpoint) if args.checkpoint else cfg.paths.checkpoints / f"{args.dataset}_best.pt"
    if not ckpt.exists():
        print(f"no checkpoint at {ckpt}; train first")
        return

    from train import make_loader
    from evaluate import evaluate_all, _build_kg_and_entities
    from models import TarkanStudent
    from utils import load_checkpoint

    kg, ent = _build_kg_and_entities(cfg)
    model = TarkanStudent(cfg, kg=kg, entity_embedder=ent).to(cfg.device)
    load_checkpoint(model, ckpt, map_location=cfg.device)
    dev = make_loader(args.dataset, "dev", cfg, shuffle=False)
    test = make_loader(args.dataset, "test", cfg, shuffle=False)

    # grid: A12 on/off x A13 alpha x window x A14 (tau fixed 0.5; alpha=0 rows cover "prior off")
    grid = list(itertools.product([False, True], [0.0, 0.3, 0.6, 1.0], [4, 6], [False, True]))
    results = []
    for bio, alpha, win, consist in grid:
        if alpha == 0.0 and win != 4:
            continue  # window irrelevant when the prior is off
        model.cfg.ns_bio_rules = bio
        model.cfg.ns_lexicon_alpha = alpha
        model.cfg.ns_window = win
        model.cfg.ns_aspect_consistency = consist
        m = evaluate_all(model, dev, cfg.device)
        f1 = m["joint"]["F1"]
        results.append(((bio, alpha, win, consist), f1))
        print(f"dev bio={int(bio)} alpha={alpha} win={win} consist={int(consist)} -> joint {f1:.2f}")

    (bio, alpha, win, consist), best_dev = max(results, key=lambda x: x[1])
    print(f"\nBEST on dev: bio={bio} alpha={alpha} win={win} consist={consist} (dev joint {best_dev:.2f})")
    model.cfg.ns_bio_rules = bio
    model.cfg.ns_lexicon_alpha = alpha
    model.cfg.ns_window = win
    model.cfg.ns_aspect_consistency = consist
    m = evaluate_all(model, test, cfg.device)
    print(f"TEST with best NS config: {m}")

    out = ROOT / "results" / "tables" / "iterations.csv"
    if out.exists():
        rows = list(csv.DictReader(open(out)))
        fieldnames = list(rows[0].keys())
        row = {k: "" for k in fieldnames}
        row.update({"tag": f"NS_{args.dataset}", "dataset": args.dataset,
                    "joint_P": round(m["joint"]["P"], 2), "joint_R": round(m["joint"]["R"], 2),
                    "joint_F1": round(m["joint"]["F1"], 2), "mate_F1": round(m["mate"]["F1"], 2),
                    "masc_Acc": round(m["masc"]["Acc"], 2), "masc_F1": round(m["masc"]["F1"], 2),
                    "pool_mode": f"ns:bio={int(bio)},a={alpha},w={win},c={int(consist)}"})
        with open(out, "a", newline="") as f:
            w = csv.DictWriter(f, fieldnames=fieldnames)
            w.writerow(row)
        print(f"logged NS_{args.dataset} -> {out}")


if __name__ == "__main__":
    main()
