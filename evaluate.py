"""Student-only unified BIO evaluation (paper Section 3.10).

JMASA/MATE predict anchors without gold boundaries. MASC is a separate diagnostic
with gold anchors; its polarity is read from the unified BIO head.
"""
from __future__ import annotations
import torch
from config import CONFIG, ID2TAG, ID2POL, POLARITIES
from metrics import joint_prf, mate_prf, masc_acc_f1
from utils import bio_to_spans

def _to_device(batch, device):
    return {k: v.to(device) if torch.is_tensor(v) else v for k, v in batch.items()}

def decode_word_tags(logits, word_ids, n_words):
    sub = logits.argmax(-1).tolist()
    tags, seen = ["O"] * n_words, set()
    for i, wid in enumerate(word_ids):
        if 0 <= wid < n_words and wid not in seen:
            seen.add(wid)
            tags[wid] = ID2TAG[sub[i]]
    return tags

def _majority_pol(tags, s, e):
    pols = [tag.split("-", 1)[1] for tag in tags[s:e] if tag != "O"]
    # Fixed tie order; never depend on Python's randomized set iteration.
    return max(POLARITIES, key=pols.count) if pols else "NEU"

def decode_spans(tags):
    return bio_to_spans(tags)

@torch.no_grad()
def predict_joint(model, loader, device=None):
    device = device or model.cfg.device
    model.eval()
    predicted, gold = [], []
    for batch in loader:
        batch = _to_device(batch, device)
        amp = str(device).startswith("cuda") and model.cfg.use_amp
        with torch.autocast("cuda", dtype=torch.float16, enabled=amp):
            out = model(batch)
        for b in range(len(batch["word_ids"])):
            tags = decode_word_tags(out["tag_logits"][b], batch["word_ids"][b], batch["n_words"][b])
            predicted.append(decode_spans(tags))
            gold.append(batch["gold_aspects"][b])
    return predicted, gold

@torch.no_grad()
def predict_masc(model, loader, device=None):
    device = device or model.cfg.device
    model.eval()
    true, predicted = [], []
    for batch in loader:
        batch = _to_device(batch, device)
        out = model(batch, use_gold_anchors=True)
        for b, aspects in enumerate(batch["gold_aspects"]):
            tags = decode_word_tags(out["tag_logits"][b], batch["word_ids"][b], batch["n_words"][b])
            for s, e, pol in aspects:
                # Truncated gold aspects are not fabricated as the BOS token.
                if not any(s <= wid < e for wid in batch["word_ids"][b]):
                    continue
                true.append(pol)
                predicted.append(_majority_pol(tags, s, e))
    return true, predicted

def evaluate_all(model, loader, device=None):
    pred, gold = predict_joint(model, loader, device)
    true, polarity = predict_masc(model, loader, device)
    return dict(joint=joint_prf(pred, gold), mate=mate_prf(pred, gold),
                masc=masc_acc_f1(true, polarity))

if __name__ == "__main__":
    import argparse
    from dataclasses import replace
    from config import config_from_dict
    from models import TarkanStudent
    from train import build_kg, check_device, make_loader
    from utils import save_json
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--dataset", choices=("twitter2015", "twitter2017"))
    ap.add_argument("--split", choices=("dev", "test"), default="test")
    ap.add_argument("--device")
    ap.add_argument("--output", required=True)
    args = ap.parse_args()
    payload = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
    if "config" not in payload:
        raise ValueError("Checkpoint lacks its training configuration; retrain with the manuscript profile.")
    cfg = config_from_dict(payload["config"])
    if args.device:
        cfg = replace(cfg, device=args.device)
    dataset = args.dataset or payload["dataset"]
    check_device(cfg)
    model = TarkanStudent(cfg, kg=build_kg(cfg)).to(cfg.device)
    model.load_state_dict(payload["model_state"])
    metrics = evaluate_all(model, make_loader(dataset, args.split, cfg), cfg.device)
    save_json(dict(metrics=metrics, config=cfg.to_dict(), dataset=dataset, split=args.split,
                   checkpoint=args.checkpoint), args.output)
    print(metrics)

