"""C29 — MASC as a QLoRA'd 8B decoder, scored on three option tokens.

Why this and not another encoder member. On the frozen pool (§C.28) the 1037 test gold
pairs are lost like this:

      92  span never entered the candidate pool      (extraction)
     186  span in the pool, POLARITY wrong           (polarity)
     759  recoverable

Polarity loses **twice** what extraction loses, which reverses §C.19's reading of the same
system. The confusion is not diffuse either -- it is one direction: POS->NEU 75, NEG->NEU
37, NEU->POS 46, with **NEG recall 57.4**. Chapter C then measured every cheap way to move
that: class-balanced CE (§C.20, buys minority recall, loses more NEU), minority margins
(§C.12, backfired), decode-time class bias (measured here, dev is flat across the whole
range), 15-member ensembling, PDS, ITC/ITM. `a` sits at 80.3-81.2 no matter what.

Reaching the bar needs `a` ~ 83.5, which is **above every published t2015 gold-span
number** (MADSC 82.34, DEQA 82.10, VLHA 81.50). No rearrangement of ~200M-parameter
encoders gets there, and the baselines all have MABSA-specific vision-language pretraining
that this pipeline lacks. So the honest move is a model class the baseline table does not
contain, disclosed as exactly that.

Two design choices make this a classifier rather than a generator:

  * **Restricted-vocab scoring, in training as well as inference.** One forward pass; read
    the logits of the single-token options A/B/C at the final position and softmax over
    just those three. The training objective is then *literally the inference
    computation*, so there is no generation, no decoding mismatch, no format drift -- and
    the output is a calibrated 3-way distribution that drops straight into the existing
    log-average ensemble (`probs_*.npz`, same keys/shape as every other member).
  * **The aspect is marked in place** with `[ ]` (Chapter B, B2): t2015 contains tweets
    where the same surface form appears twice with different gold polarity, so a bare
    "Target: X" prompt is genuinely ambiguous.

The image enters the same way it does for the teacher -- as its BLIP caption -- so no new
modality and no new pretrained vision weights enter the system.

    python experts/masc_llm.py --out runs/masc_llm --spans runs/mate_ens5_hr
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import List

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from experts.common import (DATA, POLARITIES, load, masc_examples,  # noqa: E402
                            masc_examples_for_spans, set_seed)

LETTERS = ["A", "B", "C"]
NAMES = {POLARITIES.index("NEG"): "negative", POLARITIES.index("NEU"): "neutral",
         POLARITIES.index("POS"): "positive"}
# Which polarity each letter denotes. Decoders carry a real position bias over lettered
# options, so scoring the same example under the reversed order and averaging the two
# distributions cancels the part of the answer that depends on where an option was listed
# rather than on the tweet (--tta). Costs one extra forward pass, no extra training.
ORDERS = {"fwd": [POLARITIES.index(x) for x in ("NEG", "NEU", "POS")],
          "rev": [POLARITIES.index(x) for x in ("POS", "NEU", "NEG")]}
LETTER2POL = ORDERS["fwd"]

TEMPLATE = """Tweet: {text}
Image: {caption}
Target: "{term}"
{siblings}
What sentiment does the tweet express toward the target?
{options}
Answer with one letter."""

# §C.10 tried aspect-conditioning by bracketing the OTHER aspects with < > and POS recall
# fell 12.3 points -- inserting punctuation noise disrupts a pretrained encoder's reading of
# the tweet more than it helps. The idea was never the problem, the encoding was: an
# instruction-tuned decoder can be told the same fact in words. t2015 has 444 within-tweet
# aspect pairs whose gold polarities DIFFER, and the measured failure (112 minority->NEU)
# is precisely the model falling back on the tweet's overall tone.
#
# With --siblings off, `{siblings}` renders empty and the prompt is byte-identical to the
# plain member's -- so an adapter trained under either flag can be re-scored by either.
SIBLING_LINE = ('Other targets in this tweet: {others}\n'
                'Judge the marked target only, not the overall tone of the tweet.\n')


class PromptDS(Dataset):
    def __init__(self, ex, tok, caps, max_len, siblings=False, order="fwd"):
        self.ex, self.tok, self.caps, self.max_len = ex, tok, caps, max_len
        self.siblings, self.order = siblings, ORDERS[order]

    def __len__(self):
        return len(self.ex)

    def __getitem__(self, i):
        e = self.ex[i]
        sib = ""
        if self.siblings and e.siblings:
            others = ", ".join(f'"{" ".join(e.tokens[a:b])}"' for (a, b) in e.siblings)
            sib = SIBLING_LINE.format(others=others)
        opts = "\n".join(f"{L}. {NAMES[p]}" for L, p in zip(LETTERS, self.order)) + "\n"
        msg = TEMPLATE.format(text=e.marked_text(), term=e.term, siblings=sib, options=opts,
                              caption=self.caps.get(e.image_id, "an image"))
        s = self.tok.apply_chat_template([{"role": "user", "content": msg}],
                                         tokenize=False, add_generation_prompt=True)
        ids = self.tok(s, truncation=True, max_length=self.max_len)["input_ids"]
        y = self.order.index(POLARITIES.index(e.polarity))   # gold as an OPTION index
        return {"ids": ids, "y": y, "key": e.key}


def collate(batch, pad_id):
    """LEFT padding: the option logits are read at position -1, which must be the true
    last prompt token for every row in the batch."""
    L = max(len(b["ids"]) for b in batch)
    ids = torch.full((len(batch), L), pad_id, dtype=torch.long)
    mask = torch.zeros((len(batch), L), dtype=torch.long)
    for i, b in enumerate(batch):
        n = len(b["ids"])
        ids[i, L - n:] = torch.tensor(b["ids"])
        mask[i, L - n:] = 1
    return {"ids": ids, "mask": mask,
            "y": torch.tensor([b["y"] for b in batch], dtype=torch.long),
            "key": [b["key"] for b in batch]}


def option_logits(model, ids, mask, opt_ids):
    lg = model(input_ids=ids, attention_mask=mask).logits[:, -1, :]
    return lg[:, opt_ids].float()


@torch.no_grad()
def evaluate(model, loader, opt_ids, device, order="fwd"):
    """Returns (accuracy, probs in POLARITIES column order, keys)."""
    model.eval()
    P, Y, K = [], [], []
    for b in loader:
        with torch.autocast("cuda", dtype=torch.float16):
            lg = option_logits(model, b["ids"].to(device), b["mask"].to(device), opt_ids)
        P.append(torch.softmax(lg, -1).cpu().numpy())
        Y.extend(b["y"].tolist())
        K.extend(b["key"])
    P = np.concatenate(P) if P else np.zeros((0, 3))
    acc = 100.0 * float((P.argmax(1) == np.array(Y)).mean()) if Y else 0.0
    out = np.zeros_like(P)
    for oi, pi in enumerate(ORDERS[order]):     # option space -> POLARITIES space
        out[:, pi] = P[:, oi]
    return acc, out, K


def evaluate_tta(model, mkl, ex, opt_ids, device, orders, gold=None):
    """Average the POLARITIES-space distributions over option orderings (see ORDERS)."""
    acc, P, K = None, None, None
    for o in orders:
        a, p, k = evaluate(model, mkl(ex, o), opt_ids, device, o)
        P = p if P is None else P + p
        K = k
        if len(orders) == 1:
            acc = a
    P /= len(orders)
    if acc is None:
        y = np.array([POLARITIES.index(e.polarity) for e in ex])
        acc = 100.0 * float((P.argmax(1) == y).mean()) if len(y) else 0.0
    return acc, P, K


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="meta-llama/Llama-3.1-8B-Instruct")
    ap.add_argument("--dataset", default="twitter2015")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--batch", type=int, default=2)
    ap.add_argument("--accum", type=int, default=8)
    ap.add_argument("--eval-batch", type=int, default=8)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--max-len", type=int, default=224)
    ap.add_argument("--lora-r", type=int, default=16)
    ap.add_argument("--no-caption", action="store_true")
    ap.add_argument("--siblings", action="store_true",
                    help="name the tweet's OTHER aspects in the prompt (see SIBLING_LINE)")
    ap.add_argument("--tta", action="store_true",
                    help="average over both option orderings at scoring time (see ORDERS); "
                         "training is unaffected")
    ap.add_argument("--spans", default=None,
                    help="dir with spans_{dev,test}.json; also score those candidate anchors")
    ap.add_argument("--score-only", action="store_true")
    ap.add_argument("--limit-steps", type=int, default=0, help="benchmark mode")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    set_seed(args.seed)
    device = torch.device("cuda")
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)

    from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training, PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

    tok = AutoTokenizer.from_pretrained(args.model)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    opt_ids = [tok.encode(l, add_special_tokens=False)[0] for l in LETTERS]
    assert len(set(opt_ids)) == 3, opt_ids

    caps = ({} if args.no_caption
            else json.load(open(DATA / "aadg" / args.dataset / "captions.json")))
    insts = {s: load(args.dataset, s) for s in ("train", "dev", "test")}
    ex = {s: masc_examples(v) for s, v in insts.items()}
    mk = lambda e, bs, sh, o="fwd": DataLoader(
        PromptDS(e, tok, caps, args.max_len, args.siblings, o), batch_size=bs, shuffle=sh,
        num_workers=2, collate_fn=lambda b: collate(b, tok.pad_token_id))
    orders = ["fwd", "rev"] if args.tta else ["fwd"]
    mkl = lambda e, o: mk(e, args.eval_batch, False, o)

    bnb = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
                             bnb_4bit_compute_dtype=torch.float16,
                             bnb_4bit_use_double_quant=True)
    model = AutoModelForCausalLM.from_pretrained(args.model, quantization_config=bnb,
                                                 device_map={"": 0})
    model.config.use_cache = False
    if args.score_only:
        model = PeftModel.from_pretrained(model, str(out / "adapter"))
    else:
        model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=True)
        model.gradient_checkpointing_enable()
        model.enable_input_require_grads()
        model = get_peft_model(model, LoraConfig(
            r=args.lora_r, lora_alpha=2 * args.lora_r, lora_dropout=0.05,
            bias="none", task_type="CAUSAL_LM",
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                            "gate_proj", "up_proj", "down_proj"]))
        model.print_trainable_parameters()

    dev_loader = mk(ex["dev"], args.eval_batch, False)
    best = -1.0
    if not args.score_only:
        tr = mk(ex["train"], args.batch, True)
        opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],
                                lr=args.lr, weight_decay=0.0)
        total = (len(tr) // args.accum + 1) * args.epochs
        sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=args.lr, total_steps=total,
                                                    pct_start=0.06, anneal_strategy="cos")
        scaler = torch.amp.GradScaler("cuda")
        step, t0 = 0, time.time()
        for ep in range(args.epochs):
            model.train()
            for i, b in enumerate(tr):
                with torch.autocast("cuda", dtype=torch.float16):
                    lg = option_logits(model, b["ids"].to(device), b["mask"].to(device),
                                       opt_ids)
                loss = nn.functional.cross_entropy(lg, b["y"].to(device)) / args.accum
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
                        print(f"  ep{ep} step {step}/{total} loss {loss.item()*args.accum:.4f} "
                              f"{(time.time()-t0)/step:.2f}s/step", flush=True)
                    if args.limit_steps and step >= args.limit_steps:
                        print(f"BENCHMARK: {(time.time()-t0)/step:.2f}s/opt-step, "
                              f"{total} steps -> {(time.time()-t0)/step*total/3600:.2f}h",
                              flush=True)
                        return
            acc, _, _ = evaluate(model, dev_loader, opt_ids, device)
            print(f"[epoch {ep}] dev gold-span acc {acc:.2f}", flush=True)
            if acc > best:
                best = acc
                model.save_pretrained(str(out / "adapter"))
                print(f"  saved (best {best:.2f})", flush=True)
        from peft import PeftModel as _PM
        model = _PM.from_pretrained(model.get_base_model(), str(out / "adapter"))

    res = {"model": args.model, "best_dev_acc": best, "seed": args.seed}
    for s in ("dev", "test"):
        acc, P, K = evaluate_tta(model, mkl, ex[s], opt_ids, device, orders)
        res[f"{s}_acc"] = acc
        np.savez_compressed(out / f"probs_{s}.npz", probs=P,
                            keys=np.array(K, dtype=np.int64))
        print(f"[{s}] gold-span acc {acc:.2f}", flush=True)
    if args.spans:
        for s in ("dev", "test"):
            sp = json.load(open(Path(args.spans) / f"spans_{s}.json"))
            e2 = masc_examples_for_spans(insts[s], [[tuple(x) for x in i] for i in sp])
            _, P2, K2 = evaluate_tta(model, mkl, e2, opt_ids, device, orders)
            np.savez_compressed(out / f"probs_span_{s}.npz", probs=P2,
                                keys=np.array(K2, dtype=np.int64))
            print(f"[{s}] scored {len(K2)} candidate anchors", flush=True)
    json.dump(res, open(out / "metrics.json", "w"), indent=2)


if __name__ == "__main__":
    main()
