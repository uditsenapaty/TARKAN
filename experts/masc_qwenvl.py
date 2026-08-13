"""D2 — polarity from the ORIGINAL IMAGE: Qwen2.5-VL as a member and as an evidence teacher.

Everything multimodal in this pipeline has reached the polarity model through a **BLIP
caption**: the AADG descriptions, the PDS teacher, the `masc_llm` prompt. A caption is a
lossy, sentiment-blind summary — it drops embedded text, facial expression, object
condition, and any contrast between image and tweet, which is exactly the material a
sarcastic or mixed-sentiment tweet turns on. Chapter B §7c measured the visual stream
contributing ~0 to the joint metric and blamed missing MABSA pretraining; the caption
bottleneck is the other half of that explanation. §D.2 then showed every signal the system
currently produces is exhausted (a selector fitted ON TEST cannot beat the honest one), so
the only way forward is information that is not in the system yet. The pixels are it.

Two uses of one model, sharing all the scoring machinery:

**(a) `--counterfactual` — an evidence TEACHER, offline, train split only.**
The §C.25 PDS teacher asked Llama, subjectively, "does this image shift the sentiment?" and
answered "no shift" 75% of the time. Here the teacher performs the intervention itself:

    delta(a) = P_qwen(y | tweet, aspect, IMAGE) - P_qwen(y | tweet, aspect)

Only the **sign** is kept, because §C.27 measured that regressing on the teacher's
*magnitude* fails — the teacher's scale is a bias to be resisted, not a target. This drops
into `masc_pds.py` in the exact format `pds_teacher.py` writes, so the proven formulation
(zero-init residual + signed-margin hinge + POS/NEG inverse-frequency + w_none=0) consumes
it unchanged. Inference stays student-only: labels are cached for TRAIN and never needed
again.

**(b) default — a MASC member that sees the image.** Same restricted-vocab A/B/C scoring as
`masc_llm.py`, so it is a classifier rather than a generator and its output drops into the
existing log-average. A15 measured this backbone at **MASC 80.76**, above every encoder
member here (best 79.94), and it would be the only member in the ensemble looking at
pixels — which is what the ensemble converts (§C.25: the best partners are the decorrelated
ones, not the strongest ones).

    python experts/masc_qwenvl.py --out runs/masc_qwen --spans runs/mate_union
    python experts/masc_qwenvl.py --counterfactual --split train --out data/pds_qwen
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from torch.utils.data import DataLoader, Dataset

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from experts.common import (DATA, POLARITIES, load, masc_examples,  # noqa: E402
                            masc_examples_for_spans, set_seed)

MODEL = "Qwen/Qwen2.5-VL-7B-Instruct"
LETTERS = ["A", "B", "C"]
NAMES = {POLARITIES.index("NEG"): "negative", POLARITIES.index("NEU"): "neutral",
         POLARITIES.index("POS"): "positive"}
ORDERS = {"fwd": [POLARITIES.index(x) for x in ("NEG", "NEU", "POS")],
          "rev": [POLARITIES.index(x) for x in ("POS", "NEU", "NEG")]}

PROMPT = """Tweet: {text}
Target: "{term}"

What sentiment does the tweet express toward the target?
{options}
Answer with one letter."""


class VLDS(Dataset):
    """One item per aspect. `use_image=False` gives the text-only arm of the counterfactual."""

    def __init__(self, ex, dataset, order="fwd", use_image=True):
        self.ex, self.order, self.use_image = ex, ORDERS[order], use_image
        self.root = DATA / "images" / dataset

    def __len__(self):
        return len(self.ex)

    def __getitem__(self, i):
        e = self.ex[i]
        opts = "\n".join(f"{L}. {NAMES[p]}" for L, p in zip(LETTERS, self.order)) + "\n"
        txt = PROMPT.format(text=e.marked_text(), term=e.term, options=opts)
        content = []
        img = None
        if self.use_image:
            p = self.root / e.image_id
            if p.exists():
                img = Image.open(p).convert("RGB")
                content.append({"type": "image"})
        content.append({"type": "text", "text": txt})
        return {"msg": [{"role": "user", "content": content}], "img": img,
                "y": self.order.index(POLARITIES.index(e.polarity)), "key": e.key}


def make_collate(proc):
    def collate(batch):
        texts = [proc.apply_chat_template(b["msg"], tokenize=False,
                                          add_generation_prompt=True) for b in batch]
        imgs = [b["img"] for b in batch if b["img"] is not None]
        enc = proc(text=texts, images=imgs or None, return_tensors="pt", padding=True)
        enc["y"] = torch.tensor([b["y"] for b in batch], dtype=torch.long)
        enc["key"] = [b["key"] for b in batch]
        return enc
    return collate


def option_logits(model, enc, opt_ids, device):
    kw = {k: v.to(device) for k, v in enc.items()
          if isinstance(v, torch.Tensor) and k != "y"}
    return model(**kw).logits[:, -1, :][:, opt_ids].float()


@torch.no_grad()
def run(model, loader, opt_ids, device, order):
    model.eval()
    P, Y, K = [], [], []
    for enc in loader:
        with torch.autocast("cuda", dtype=torch.float16):
            lg = option_logits(model, enc, opt_ids, device)
        P.append(torch.softmax(lg, -1).cpu().numpy())
        Y.extend(enc["y"].tolist())
        K.extend(enc["key"])
    P = np.concatenate(P) if P else np.zeros((0, 3))
    acc = 100.0 * float((P.argmax(1) == np.array(Y)).mean()) if Y else 0.0
    out = np.zeros_like(P)
    for oi, pi in enumerate(ORDERS[order]):       # option space -> POLARITIES space
        out[:, pi] = P[:, oi]
    return acc, out, K


def load_model(args, train: bool):
    from peft import (LoraConfig, PeftModel, get_peft_model,
                      prepare_model_for_kbit_training)
    from transformers import (AutoProcessor, BitsAndBytesConfig,
                              Qwen2_5_VLForConditionalGeneration)
    proc = AutoProcessor.from_pretrained(
        args.model, min_pixels=args.min_pixels, max_pixels=args.max_pixels,
        padding_side="left")   # option logits are read at position -1
    bnb = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
                             bnb_4bit_compute_dtype=torch.float16,
                             bnb_4bit_use_double_quant=True)
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        args.model, quantization_config=bnb, device_map={"": 0})
    model.config.use_cache = False
    if args.adapter:
        model = PeftModel.from_pretrained(model, args.adapter, is_trainable=train)
    elif train:
        model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=True)
        model.gradient_checkpointing_enable()
        model.enable_input_require_grads()
        # Adapt the LANGUAGE tower only. The vision encoder stays frozen (A15 did the same)
        # -- it is the expensive half on a T4 and the task is reading an image, not learning
        # to see. The regex is what keeps LoRA off the visual blocks, whose projections
        # share names with the language ones.
        model = get_peft_model(model, LoraConfig(
            r=args.lora_r, lora_alpha=2 * args.lora_r, lora_dropout=0.05, bias="none",
            target_modules=r".*language_model.*\.(q_proj|k_proj|v_proj|o_proj|"
                           r"gate_proj|up_proj|down_proj)$"))
        model.print_trainable_parameters()
    return proc, model


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=MODEL)
    ap.add_argument("--dataset", default="twitter2015")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--epochs", type=int, default=2)
    ap.add_argument("--batch", type=int, default=1)
    ap.add_argument("--accum", type=int, default=16)
    ap.add_argument("--eval-batch", type=int, default=4)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--lora-r", type=int, default=16)
    ap.add_argument("--min-pixels", type=int, default=64 * 28 * 28)
    ap.add_argument("--max-pixels", type=int, default=256 * 28 * 28,
                    help="Qwen2.5-VL is dynamic-resolution; this caps the visual token "
                         "count, which is the whole memory/speed story on a 16GB T4")
    ap.add_argument("--tta", action="store_true", help="average both option orderings")
    ap.add_argument("--counterfactual", action="store_true",
                    help="emit PDS direction labels from the WITH-IMAGE minus TEXT-ONLY "
                         "difference instead of training a member")
    ap.add_argument("--split", default="train", help="counterfactual mode: which split")
    ap.add_argument("--shift-floor", type=float, default=0.05,
                    help="how large a probability shift has to be before it counts as a "
                         "direction rather than no-shift")
    ap.add_argument("--shift-temp", type=float, default=0.1)
    ap.add_argument("--spans", default=None)
    ap.add_argument("--score-only", action="store_true")
    ap.add_argument("--adapter", default=None)
    ap.add_argument("--limit-steps", type=int, default=0)
    ap.add_argument("--limit", type=int, default=0, help="debug: cap examples")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    set_seed(args.seed)
    device = torch.device("cuda")
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    train_mode = not (args.counterfactual or args.score_only)
    proc, model = load_model(args, train_mode)
    tok = proc.tokenizer
    opt_ids = [tok.encode(l, add_special_tokens=False)[0] for l in LETTERS]
    assert len(set(opt_ids)) == 3, opt_ids
    coll = make_collate(proc)

    insts = {s: load(args.dataset, s) for s in ("train", "dev", "test")}
    ex = {s: masc_examples(v) for s, v in insts.items()}
    if args.limit:
        ex = {s: v[:args.limit] for s, v in ex.items()}
    mk = lambda e, bs, sh, o="fwd", im=True: DataLoader(
        VLDS(e, args.dataset, o, im), batch_size=bs, shuffle=sh, collate_fn=coll)
    orders = ["fwd", "rev"] if args.tta else ["fwd"]

    # ---------------------------------------------------------------- counterfactual
    if args.counterfactual:
        e = ex[args.split]
        _, P_img, K = run(model, mk(e, args.eval_batch, False, "fwd", True),
                          opt_ids, device, "fwd")
        _, P_txt, _ = run(model, mk(e, args.eval_batch, False, "fwd", False),
                          opt_ids, device, "fwd")
        d = P_img - P_txt
        NEG, NEU, POS = (POLARITIES.index(x) for x in ("NEG", "NEU", "POS"))
        # pds_teacher.py's format: columns are [POS-shift, NEG-shift, no-shift]. The shift is
        # read as a signed contrast so that "the image made POS more likely at NEU's expense"
        # is POS-shift, and the no-shift mass is whatever the intervention did not move.
        pos_s = np.clip(d[:, POS] - d[:, NEU], 0, None)
        neg_s = np.clip(d[:, NEG] - d[:, NEU], 0, None)
        # The three columns must COMPETE. `1 - pos - neg` does not: it starts near 1 while
        # the shift terms are ~0.1-0.3, so no-shift would win by construction whatever the
        # image did. `--shift-floor` is instead an explicit "how large must a shift be to
        # count" logit, and the temperature controls how hard the resulting label is.
        # Only the RELATIVE POS/NEG weight is ever read downstream: §C.27 settled that the
        # no-shift column must be ignored entirely (w_none = 0).
        z = np.stack([pos_s, neg_s, np.full_like(pos_s, args.shift_floor)], 1)
        probs = np.exp((z - z.max(1, keepdims=True)) / args.shift_temp)
        probs /= probs.sum(1, keepdims=True)
        out.mkdir(parents=True, exist_ok=True)
        # Keep the RAW arms as well as the mapped labels. --shift-floor / --shift-temp are
        # first guesses, and without these the only way to re-threshold is another full
        # teacher pass over the split.
        np.savez_compressed(out / f"direction_{args.split}.npz", probs=probs,
                            keys=np.array(K, dtype=np.int64),
                            p_img=P_img.astype(np.float32),
                            p_txt=P_txt.astype(np.float32))
        hard = probs.argmax(1)
        print(f"wrote {len(probs)} Qwen-VL direction labels -> {out}")
        print("distribution:", {n: int((hard == i).sum())
                                for i, n in enumerate(["POS", "NEG", "NONE"])})
        print(f"mean |delta| {float(np.abs(d).sum(1).mean()):.3f}   "
              f"moved >0.05: {int((np.abs(d).sum(1) > 0.05).sum())}/{len(d)}")
        return

    # ---------------------------------------------------------------- member
    best = -1.0
    if train_mode:
        tr = mk(ex["train"], args.batch, True)
        opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],
                                lr=args.lr, weight_decay=0.0)
        total = (len(tr) // args.accum + 1) * args.epochs
        sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=args.lr, total_steps=total,
                                                    pct_start=0.06)
        scaler = torch.amp.GradScaler("cuda")
        step, t0 = 0, time.time()
        for ep in range(args.epochs):
            model.train()
            for i, enc in enumerate(tr):
                with torch.autocast("cuda", dtype=torch.float16):
                    lg = option_logits(model, enc, opt_ids, device)
                loss = nn.functional.cross_entropy(lg, enc["y"].to(device)) / args.accum
                scaler.scale(loss).backward()
                if (i + 1) % args.accum == 0:
                    scaler.unscale_(opt)
                    torch.nn.utils.clip_grad_norm_(
                        [p for p in model.parameters() if p.requires_grad], 1.0)
                    scaler.step(opt); scaler.update(); opt.zero_grad(set_to_none=True)
                    if sched.last_epoch < total - 1:
                        sched.step()
                    step += 1
                    if step % 25 == 0:
                        print(f"  ep{ep} step {step}/{total} loss "
                              f"{loss.item()*args.accum:.4f} "
                              f"{(time.time()-t0)/step:.2f}s/step", flush=True)
                    if args.limit_steps and step >= args.limit_steps:
                        print(f"BENCHMARK: {(time.time()-t0)/step:.2f}s/opt-step -> "
                              f"{(time.time()-t0)/step*total/3600:.2f}h", flush=True)
                        return
            acc, _, _ = run(model, mk(ex["dev"], args.eval_batch, False), opt_ids,
                            device, "fwd")
            print(f"[epoch {ep}] dev gold-span acc {acc:.2f}", flush=True)
            if acc > best:
                best = acc
                model.save_pretrained(str(out / "adapter"))
                print(f"  saved (best {best:.2f})", flush=True)

    res = {"model": args.model, "best_dev_acc": best, "max_pixels": args.max_pixels}
    for s in ("dev", "test"):
        acc, P, K = None, None, None
        for o in orders:
            a, p, k = run(model, mk(ex[s], args.eval_batch, False, o), opt_ids, device, o)
            P = p if P is None else P + p
            acc, K = a, k
        P /= len(orders)
        res[f"{s}_acc"] = acc
        np.savez_compressed(out / f"probs_{s}.npz", probs=P,
                            keys=np.array(K, dtype=np.int64))
        print(f"[{s}] gold-span acc {acc:.2f}", flush=True)
    if args.spans:
        for s in ("dev", "test"):
            sp = json.load(open(Path(args.spans) / f"spans_{s}.json"))
            e2 = masc_examples_for_spans(insts[s], [[tuple(x) for x in i] for i in sp])
            P = None
            for o in orders:
                _, p, K2 = run(model, mk(e2, args.eval_batch, False, o), opt_ids, device, o)
                P = p if P is None else P + p
            np.savez_compressed(out / f"probs_span_{s}.npz", probs=P / len(orders),
                                keys=np.array(K2, dtype=np.int64))
            print(f"[{s}] scored {len(K2)} candidate anchors", flush=True)
    json.dump(res, open(out / "metrics.json", "w"), indent=2)


if __name__ == "__main__":
    main()
