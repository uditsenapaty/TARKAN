"""C15 — PDS: Polarity-DIRECTION supervision from the LLM teacher.

Every teacher signal TARKAN has ever used is a *relevance* label:
    "is this image relevant to the aspect?"     (r^T)
    "is this KG triple useful?"                 (s^T)
Relevance does not say what the evidence does. An image can be highly relevant to
"restaurant" and still be sentiment-neutral about it. Chapter B measured both relevance
streams at ~0 on the joint metric, and Chapter C measured the modality gate built on them
at +0.10 — consistent with the signal being real but *directionless*.

PDS asks a different question, once, offline:

    relative to the tweet text ALONE, does this image evidence shift the sentiment
    expressed toward THIS aspect more positive, more negative, or not at all?

producing a distribution over {POS-shift, NEG-shift, no-shift} per training aspect. The
student then learns to reproduce that shift (see `experts/masc_pds.py`), which is a
constraint on how evidence may move the decision — not merely on whether to look at it.
R1 (student-only inference) is preserved: labels are cached offline for TRAIN only.

Two deliberate refinements over a naive implementation:
  * **Score, don't generate.** One forward pass per aspect, reading the logits of the
    single-token options A/B/C, instead of sampling text. Deterministic, ~3x faster, and it
    yields a SOFT distribution -- so teacher uncertainty stays uncertainty rather than being
    frozen into a hard label the student must then fit exactly.
  * **The image is presented as its caption**, matching TARKAN's own teacher design (the
    teacher never sees pixels), so no new modality enters the teacher.

    python experts/pds_teacher.py --split train
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from experts.common import DATA, load, masc_examples  # noqa: E402

TEACHER = "meta-llama/Llama-3.1-8B-Instruct"
DIRS = ["POS", "NEG", "NONE"]          # A / B / C
LETTERS = ["A", "B", "C"]

TEMPLATE = """Tweet: {text}
Image description: {caption}
Target: {aspect}

Relative to the tweet text alone, does the image shift the sentiment expressed toward "{aspect}"?
A. more positive
B. more negative
C. no shift

Reply with one letter."""


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="twitter2015")
    ap.add_argument("--split", default="train")
    ap.add_argument("--batch", type=int, default=4)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
    device = torch.device("cuda")
    tok = AutoTokenizer.from_pretrained(TEACHER)
    tok.padding_side = "left"
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    bnb = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
                             bnb_4bit_compute_dtype=torch.float16,
                             bnb_4bit_use_double_quant=True)
    model = AutoModelForCausalLM.from_pretrained(TEACHER, quantization_config=bnb,
                                                 device_map={"": 0})
    model.eval()
    opt_ids = [tok.encode(l, add_special_tokens=False)[0] for l in LETTERS]
    print("option token ids", opt_ids, flush=True)

    caps = json.load(open(DATA / "aadg" / args.dataset / "captions.json"))
    insts = load(args.dataset, args.split)
    ex = masc_examples(insts)
    prompts = []
    for e in ex:
        msg = TEMPLATE.format(text=" ".join(e.tokens),
                              caption=caps.get(e.image_id, "an image"),
                              aspect=e.term)
        prompts.append(tok.apply_chat_template([{"role": "user", "content": msg}],
                                               tokenize=False, add_generation_prompt=True))
    print(f"{len(prompts)} prompts for {args.split}", flush=True)

    out = np.zeros((len(prompts), 3), dtype=np.float32)
    for i in range(0, len(prompts), args.batch):
        enc = tok(prompts[i:i + args.batch], return_tensors="pt", padding=True,
                  truncation=True, max_length=512).to(device)
        with torch.no_grad():
            lg = model(**enc).logits[:, -1, :].float()
        out[i:i + enc["input_ids"].size(0)] = torch.softmax(
            lg[:, opt_ids], dim=-1).cpu().numpy()
        if (i // args.batch) % 100 == 0:
            print(f"  {i+enc['input_ids'].size(0)}/{len(prompts)}", flush=True)

    d = Path(args.out) if args.out else DATA / "pds" / args.dataset
    d.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(d / f"direction_{args.split}.npz",
                        probs=out, keys=np.array([e.key for e in ex], dtype=np.int64))
    hard = out.argmax(1)
    print(f"wrote {len(out)} direction labels -> {d}")
    print("distribution:", {DIRS[i]: int((hard == i).sum()) for i in range(3)})
    print("mean confidence %.3f" % float(out.max(1).mean()))


if __name__ == "__main__":
    main()
