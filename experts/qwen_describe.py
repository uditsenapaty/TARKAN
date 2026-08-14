"""D30 — replace the BLIP caption source with Qwen2.5-VL reading the ORIGINAL image.

**Why this specific change, and why it is the best-evidenced lever available.**
MADSC's own ablation isolates the description generator on t2015 JMASA:

    GPT-4o descriptions (theirs)   72.9
    LLaVA                          72.1
    BLIP2                          71.8      <- the configuration this project has been in

Every aspect-aware description in this repo is built from **BLIP** captions
(`data/aadg/twitter2015/captions.json`, `blip-image-captioning-base`). So the entire AADG →
dual-similarity → calibrated `u` → gate chain has been running on the weakest description
source MADSC measured, and their table prices that at **−1.1**.

Nothing downstream needs to change. `aadg.py stage_describe` consumes `captions.json`, and
`masc_gated.py` / `masc_pds.py` consume the `vis` / `u` it produces. Swapping the source
therefore upgrades the evidence for every member built on it, without touching the
architecture that §D.27's law says is saturated.

**Direct image only.** Qwen2.5-VL is given the pixels and the tweet — never a BLIP caption.
It is asked to DESCRIBE, not to judge sentiment: §D.9/§D.12 measured Qwen-as-polarity-teacher
at −0.2 to −0.9, and §D.28 measured aspect-image grounding as learnable but
polarity-irrelevant. The role that remains untested is exactly this one -- generating a
better *description* for a pipeline that already knows how to ground and calibrate it.

Offline and cached; no Qwen at inference, so student-only inference is preserved.

    python experts/qwen_describe.py --dataset twitter2015
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import torch
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from experts.common import DATA, load  # noqa: E402

MODEL = "Qwen/Qwen2.5-VL-7B-Instruct"

# Mirrors what a captioner is asked for, so `aadg.py`'s spaCy object extraction and
# dual-similarity matching keep working unchanged: one flat sentence naming the visible
# entities. The tweet is included because MADSC's descriptions are text-aware, and it costs
# nothing here.
PROMPT = ("Tweet: {text}\n\n"
          "Describe what is visible in this image in one sentence. Name the concrete "
          "objects, people and places you can actually see, and any visible text on signs "
          "or clothing. Do not mention the tweet, do not speculate, do not give opinions.")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="twitter2015")
    ap.add_argument("--model", default=MODEL)
    ap.add_argument("--batch", type=int, default=4)
    ap.add_argument("--max-new", type=int, default=48)
    ap.add_argument("--min-pixels", type=int, default=64 * 28 * 28)
    ap.add_argument("--max-pixels", type=int, default=256 * 28 * 28)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    device = torch.device("cuda")
    from transformers import (AutoProcessor, BitsAndBytesConfig,
                              Qwen2_5_VLForConditionalGeneration)
    proc = AutoProcessor.from_pretrained(args.model, min_pixels=args.min_pixels,
                                         max_pixels=args.max_pixels, padding_side="left")
    bnb = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
                             bnb_4bit_compute_dtype=torch.float16,
                             bnb_4bit_use_double_quant=True)
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        args.model, quantization_config=bnb, device_map={"": 0})
    model.eval()

    # one representative tweet per image, so the description is text-aware without
    # duplicating work for images shared across tweets
    text_of = {}
    for split in ("train", "dev", "test"):
        for inst in load(args.dataset, split):
            text_of.setdefault(inst.image_id, " ".join(inst.tokens))
    root = DATA / "images" / args.dataset
    ids = [i for i in text_of if (root / i).exists()]
    if args.limit:
        ids = ids[:args.limit]
    print(f"{len(ids)} images to describe", flush=True)

    out_path = Path(args.out) if args.out else DATA / "aadg" / args.dataset / "captions_qwen.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    caps, t0 = {}, time.time()
    for i in range(0, len(ids), args.batch):
        chunk = ids[i:i + args.batch]
        msgs, imgs = [], []
        for k in chunk:
            imgs.append(Image.open(root / k).convert("RGB"))
            msgs.append([{"role": "user", "content": [
                {"type": "image"},
                {"type": "text", "text": PROMPT.format(text=text_of[k])}]}])
        texts = [proc.apply_chat_template(m, tokenize=False, add_generation_prompt=True)
                 for m in msgs]
        enc = proc(text=texts, images=imgs, return_tensors="pt", padding=True).to(device)
        with torch.no_grad():
            gen = model.generate(**enc, max_new_tokens=args.max_new, do_sample=False)
        for k, g, n in zip(chunk, gen, enc["input_ids"]):
            s = proc.tokenizer.decode(g[len(n):], skip_special_tokens=True).strip()
            caps[k] = " ".join(s.replace("\n", " ").split())
        if (i // args.batch) % 25 == 0:
            done = i + len(chunk)
            rate = (time.time() - t0) / max(done, 1)
            print(f"  {done}/{len(ids)}  {rate:.2f}s/img  eta "
                  f"{rate*(len(ids)-done)/60:.0f}m", flush=True)
            json.dump(caps, open(out_path, "w"))      # checkpoint as we go
    json.dump(caps, open(out_path, "w"))
    print(f"wrote {len(caps)} Qwen descriptions -> {out_path}")
    ex = list(caps.items())[:3]
    for k, v in ex:
        print(f"  {k}: {v[:150]}")


if __name__ == "__main__":
    main()
