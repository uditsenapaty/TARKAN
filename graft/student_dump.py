"""Dump the student champion's predictions aligned to AoM's sample order (word spans).

Matches AoM {split}.json records to our student instances by image_id (validated),
runs predict_joint, converts our (ws, we_excl, POL) to (ws, we_incl, POL), writes
graft/student_t15_{split}.json aligned to AoM order ([] where unmatched).

Run in the MAIN env: python3 graft/student_dump.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from config import CONFIG, cfg_for  # noqa: E402

# student champion = DeBERTa all-in (conf-append + feat-gate + CRF + richASC)
CONFIG.fusion_conf_append = True
CONFIG.fusion_feat_gate = True
CKPT = ROOT / "results" / "checkpoints" / "t2015_deb_allin.pt"
NS = dict(ns_bio_rules=True, ns_lexicon_alpha=0.6, ns_window=4)  # tuned rules (NS_twitter2015)


def main():
    cfg = cfg_for("twitter2015", device="cuda", text_model_id="microsoft/deberta-v3-large",
                  batch_size=4, grad_accum=4)
    for k, v in NS.items():
        setattr(cfg, k, v)

    from train import make_loader
    from evaluate import predict_joint, _build_kg_and_entities
    from models import TarkanStudent
    from utils import load_checkpoint

    kg, ent = _build_kg_and_entities(cfg)
    model = TarkanStudent(cfg, kg=kg, entity_embedder=ent).to(cfg.device)
    model.cfg = cfg
    load_checkpoint(model, CKPT, map_location=cfg.device)

    aom_dir = ROOT / "graft" / "AoM_full" / "src" / "data" / "twitter2015"
    for split in ("dev", "test"):
        aom = json.load(open(aom_dir / f"{split}.json"))
        loader = make_loader("twitter2015", split, cfg, shuffle=False)
        insts = loader.dataset.instances
        preds, _ = predict_joint(model, loader, cfg.device)
        by_img = {}
        for inst, p in zip(insts, preds):
            by_img.setdefault(inst.image_id, []).append((inst, p))

        out, matched, token_ok = [], 0, 0
        for rec in aom:
            cands = by_img.get(rec["image_id"], [])
            hit = None
            for inst, p in cands:
                if [t.lower() for t in inst.tokens] == [w.lower() for w in rec["words"]]:
                    hit = (inst, p, True)
                    break
            if hit is None and len(cands) == 1:
                hit = (*cands[0], False)
            if hit is None:
                out.append([])
                continue
            inst, p, tok_exact = hit
            matched += 1
            token_ok += int(tok_exact)
            out.append([[s, e - 1, pol] for (s, e, pol) in p])  # excl -> incl end
        json.dump(out, open(ROOT / "graft" / f"student_t15_{split}.json", "w"))
        print(f"{split}: {matched}/{len(aom)} matched ({token_ok} token-exact) -> student_t15_{split}.json")


if __name__ == "__main__":
    main()
