"""Seed-ensemble evaluation: majority-vote joint MABSA predictions across N checkpoints.

Each checkpoint is the SAME architecture/config trained with a different seed. Votes:
  - a span (s, e) is kept if >= ceil(N/2) models predict it;
  - its polarity is the majority among the models that predicted it;
  - MASC: per-gold-aspect majority polarity across all models.

Usage:
  python scripts/ensemble_eval.py --dataset twitter2015 --checkpoints a.pt b.pt c.pt [--ns]
`--ns` applies the tuned neurosymbolic knobs (set them on CONFIG before predict).
"""
from __future__ import annotations

import argparse
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from config import CONFIG, cfg_for  # noqa: E402


def vote_joint(all_preds):
    """all_preds: list over models of list over instances of [(s,e,pol)]."""
    n_models = len(all_preds)
    need = (n_models // 2) + 1
    n_inst = len(all_preds[0])
    voted = []
    for i in range(n_inst):
        span_votes: Counter = Counter()
        pol_votes = {}
        for m in range(n_models):
            for (s, e, p) in all_preds[m][i]:
                span_votes[(s, e)] += 1
                pol_votes.setdefault((s, e), []).append(p)
        inst = []
        for (s, e), v in span_votes.items():
            if v >= need:
                pol = Counter(pol_votes[(s, e)]).most_common(1)[0][0]
                inst.append((s, e, pol))
        voted.append(sorted(inst))
    return voted


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="twitter2015")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--checkpoints", nargs="+", required=True)
    ap.add_argument("--ns-bio", action="store_true")
    ap.add_argument("--ns-alpha", type=float, default=0.0)
    ap.add_argument("--conf-append", action="store_true")
    ap.add_argument("--feat-gate", action="store_true")
    args = ap.parse_args()

    CONFIG.fusion_conf_append = bool(args.conf_append)
    CONFIG.fusion_feat_gate = bool(args.feat_gate)
    cfg = cfg_for(args.dataset, device=args.device)
    cfg.ns_bio_rules = bool(args.ns_bio)
    cfg.ns_lexicon_alpha = float(args.ns_alpha)

    from train import make_loader
    from evaluate import predict_joint, predict_masc, _build_kg_and_entities
    from models import TarkanStudent
    from metrics import joint_prf, mate_prf, masc_acc_f1
    from utils import load_checkpoint

    kg, ent = _build_kg_and_entities(cfg)
    test = make_loader(args.dataset, "test", cfg, shuffle=False)

    all_preds, golds = [], None
    all_masc = []
    for ck in args.checkpoints:
        model = TarkanStudent(cfg, kg=kg, entity_embedder=ent).to(cfg.device)
        model.cfg = cfg
        load_checkpoint(model, ck, map_location=cfg.device)
        preds, g = predict_joint(model, test, cfg.device)
        yt, yp = predict_masc(model, test, cfg.device)
        all_preds.append(preds)
        all_masc.append(yp)
        golds = g
        y_true = yt
        del model
        import torch
        torch.cuda.empty_cache()
        print(f"  member {Path(ck).name}: joint {joint_prf(preds, golds)['F1']:.2f}")

    voted = vote_joint(all_preds)
    masc_voted = [Counter(col).most_common(1)[0][0] for col in zip(*all_masc)]
    print("\n=== ENSEMBLE ===")
    print("joint:", {k: round(v, 2) for k, v in joint_prf(voted, golds).items()})
    print("mate :", {k: round(v, 2) for k, v in mate_prf(voted, golds).items()})
    print("masc :", {k: round(v, 2) for k, v in masc_acc_f1(y_true, masc_voted).items()})


if __name__ == "__main__":
    main()
