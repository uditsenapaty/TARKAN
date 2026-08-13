"""C2 step 1 — cache FROZEN BLIP-2 EVA-ViT-g patch features for every referenced image.

DQPSA never fine-tunes its visual encoder: `DQPSA_dataset.__getitem__` reads
`self.data[index]["image_feature"]` straight out of a pickle. That is exactly what makes
the whole architecture trainable on a T4 -- the 1B-parameter ViT runs ONCE, offline, and
training afterwards only touches the Q-Former + text encoder (~230M params).

Writes a single fp16 memmap [N, 257, 1408] plus an image_id -> row index.

    python experts/cache_vit.py --dataset twitter2015
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from experts.common import DATA, image_path, load  # noqa: E402

MODEL_ID = "Salesforce/blip2-itm-vit-g"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="twitter2015")
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    out_dir = Path(args.out) if args.out else DATA / "vit_cache" / args.dataset
    out_dir.mkdir(parents=True, exist_ok=True)

    ids = []
    seen = set()
    for split in ("train", "dev", "test"):
        for inst in load(args.dataset, split):
            if inst.image_id not in seen:
                seen.add(inst.image_id)
                ids.append(inst.image_id)
    print(f"{len(ids)} unique images for {args.dataset}", flush=True)

    from transformers import AutoProcessor, Blip2ForImageTextRetrieval
    proc = AutoProcessor.from_pretrained(MODEL_ID)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    full = Blip2ForImageTextRetrieval.from_pretrained(MODEL_ID, dtype=torch.float16)
    vision = full.vision_model.to(device).eval()
    del full

    n_tok = (224 // 14) ** 2 + 1
    dim = 1408
    mm = np.memmap(out_dir / "feats.f16", dtype=np.float16, mode="w+",
                   shape=(len(ids), n_tok, dim))

    missing = []
    for i in range(0, len(ids), args.batch):
        chunk = ids[i:i + args.batch]
        imgs = []
        for iid in chunk:
            p = image_path(args.dataset, iid)
            try:
                imgs.append(Image.open(p).convert("RGB"))
            except Exception:
                missing.append(iid)
                imgs.append(Image.new("RGB", (224, 224), (128, 128, 128)))
        px = proc(images=imgs, return_tensors="pt")["pixel_values"].to(device, torch.float16)
        with torch.no_grad():
            h = vision(pixel_values=px).last_hidden_state
        mm[i:i + len(chunk)] = h.cpu().numpy().astype(np.float16)
        if (i // args.batch) % 50 == 0:
            print(f"  {i+len(chunk)}/{len(ids)}", flush=True)

    mm.flush()
    json.dump({"ids": ids, "index": {k: i for i, k in enumerate(ids)},
               "shape": [len(ids), n_tok, dim], "model": MODEL_ID,
               "missing": missing},
              open(out_dir / "index.json", "w"))
    print(f"done -> {out_dir}  (missing images: {len(missing)})")


if __name__ == "__main__":
    main()
