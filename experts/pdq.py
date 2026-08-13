"""C2 — PDQ: a DQPSA-mechanism MASC expert built on PUBLIC BLIP-2 weights.

Chapter B's definitive negative (§7v/§7x) was that the 71.26 ceiling is *backbone error
correlation*: every strong member inherited Qwen2.5-VL's blind spots, and even a
"different family" member (LLaVA-1.5-7B) failed to decorrelate (both_wrong 0.149 vs
independence 0.043 -> ratio 3.43) because LLaVA is architecturally the same animal --
CLIP-ViT + autoregressive instruction-tuned LLM.

This member differs on every axis that plausibly drives error correlation:
  * EVA ViT-g visual features, not CLIP-ViT
  * a Q-Former cross-attention bridge pretrained with ITC/ITM on ~129M image-text pairs,
    not instruction tuning
  * DISCRIMINATIVE bi-affine span scoring (GlobalPointer/EPE), not an autoregressive
    verbalizer over LM-head tokens

Faithful to `referred_clones/DQPSA` (audited 2026-08-12), with public substitutes for the
two Baidu-expired checkpoints:
  DQPSA `checkpoints/pretrain_ckp/*.pt`  ->  `Salesforce/blip2-itm-vit-g` (the same BLIP-2
                                             stage-1 ITC/ITM rootstock PDQ builds on)
  DQPSA `Text_encoder/model_best`        ->  `bert-base-uncased` (or any BERT-family id)

Mechanism (DQPSA `model.py` + `PDQ/PDQ.py` + `Text_encoder/{epe,sparse_attn_model}.py`):
  prompt tokens ARE the Q-Former queries ("Prompt as Dual Query") and cross-attend the
  frozen image features; their 32 outputs are projected and PREPENDED to the text
  encoder; an EPE RoPE bi-affine head scores every (start,end) cell; a `prompt_mask`
  restricts scoring to the answer region -- for MASC that region is the literal
  `[ positive , neutral , negative ]` option list inside the prompt, so classification is
  performed as span-pointing.

    python experts/pdq.py --seed 42 --out runs/pdq_s42
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Dict, List

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from experts.common import (DATA, POLARITIES, load, masc_examples,  # noqa: E402
                            masc_examples_for_spans, set_seed)

BLIP_ID = "Salesforce/blip2-itm-vit-g"
OPTIONS = ["positive", "neutral", "negative"]
OPT2POL = {"positive": "POS", "neutral": "NEU", "negative": "NEG"}


# --------------------------------------------------------------------------- #
# EPE / GlobalPointer  (port of DQPSA Text_encoder/epe.py)
# --------------------------------------------------------------------------- #
class EPE(nn.Module):
    def __init__(self, hidden_size: int = 768, inner_dim: int = 64, rope: bool = True):
        super().__init__()
        self.inner_dim = inner_dim
        self.rope = rope
        self.dense = nn.Linear(hidden_size, inner_dim * 2)

    @staticmethod
    def _sinusoid(seq_len: int, dim: int, device) -> torch.Tensor:
        pos = torch.arange(seq_len, dtype=torch.float, device=device).unsqueeze(-1)
        idx = torch.arange(0, dim // 2, dtype=torch.float, device=device)
        idx = torch.pow(10000, -2 * idx / dim)
        emb = pos * idx
        emb = torch.stack([torch.sin(emb), torch.cos(emb)], dim=-1)
        return emb.reshape(seq_len, dim)

    def forward(self, h: torch.Tensor, attn_mask: torch.Tensor) -> torch.Tensor:
        B, L, _ = h.shape
        out = self.dense(h)
        qw, kw = out[..., :self.inner_dim], out[..., self.inner_dim:]
        if self.rope:
            pe = self._sinusoid(L, self.inner_dim, h.device)          # [L, d]
            cos = pe[None, :, 1::2].repeat_interleave(2, dim=-1)
            sin = pe[None, :, ::2].repeat_interleave(2, dim=-1)
            def rot(x):
                x2 = torch.stack([-x[..., 1::2], x[..., ::2]], dim=-1).reshape(x.shape)
                return x * cos + x2 * sin
            qw, kw = rot(qw), rot(kw)
        logits = torch.einsum("bmd,bnd->bmn", qw, kw) / self.inner_dim ** 0.5
        pad = attn_mask[:, None, :] * attn_mask[:, :, None]
        logits = logits * pad - (1 - pad) * 1e12
        logits = logits - torch.tril(torch.ones_like(logits), -1) * 1e12   # no end<start
        return logits


# --------------------------------------------------------------------------- #
# model
# --------------------------------------------------------------------------- #
class PDQ(nn.Module):
    def __init__(self, text_model: str = "bert-base-uncased", n_query: int = 32,
                 dropout: float = 0.1):
        super().__init__()
        from transformers import AutoModel, Blip2ForImageTextRetrieval
        blip = Blip2ForImageTextRetrieval.from_pretrained(BLIP_ID, dtype=torch.float32)
        self.q_embeddings = blip.embeddings      # Blip2TextEmbeddings
        self.qformer = blip.qformer              # Blip2QFormerModel (ITC/ITM pretrained)
        # C2c: DQPSA trains ITC + ITM + EPE (all weight 1.0); we previously ran the
        # epe-only path (their own `no_its_and_itm` branch). These three heads are the
        # pretrained BLIP-2 stage-1 heads, so the contrastive objective resumes from where
        # BLIP-2's 129M-pair pretraining left off rather than starting cold.
        self.vision_projection = blip.vision_projection
        self.text_projection = blip.text_projection
        self.itm_head = blip.itm_head
        self.temp = nn.Parameter(torch.tensor(0.07))
        del blip
        self.n_query = n_query
        self.text = AutoModel.from_pretrained(text_model, dtype=torch.float32)
        h_t = self.text.config.hidden_size
        h_q = self.qformer.config.hidden_size
        self.proj = nn.Linear(h_q, h_t)
        self.drop = nn.Dropout(dropout)
        self.epe = EPE(hidden_size=h_t)
        self.bce = nn.BCEWithLogitsLoss(reduction="sum")

    def itc_itm(self, image_embeds, query_ids, txt_ids, txt_mask, qout):
        """Image-text contrastive + image-text matching over in-batch negatives.

        Deviation from DQPSA: they mine hard negatives from the ITC similarity matrix; we
        use one shuffled in-batch negative per sample (batch 8 makes mining pointless).
        """
        B = image_embeds.size(0)
        if B < 2:
            z = image_embeds.sum() * 0.0
            return z, z
        img_f = torch.nn.functional.normalize(
            self.vision_projection(qout[:, :self.n_query]), dim=-1)        # [B,Q,d]
        t_out = self.qformer(query_embeds=self.q_embeddings(input_ids=txt_ids),
                             query_length=0, attention_mask=txt_mask,
                             encoder_hidden_states=None).last_hidden_state
        txt_f = torch.nn.functional.normalize(
            self.text_projection(t_out[:, 0, :]), dim=-1)                  # [B,d]
        temp = self.temp.clamp(0.01, 0.5)
        # sim[i,j] = max over the Q query slots of  img_f[i,q] . txt_f[j]
        sim_i2t = torch.einsum("iqd,jd->ijq", img_f, txt_f).max(-1).values / temp
        tgt = torch.arange(B, device=image_embeds.device)
        loss_itc = 0.5 * (torch.nn.functional.cross_entropy(sim_i2t, tgt)
                          + torch.nn.functional.cross_entropy(sim_i2t.t(), tgt))

        neg = (tgt + 1 + torch.randint(0, B - 1, (B,), device=tgt.device)) % B
        img2 = torch.cat([image_embeds, image_embeds[neg]], 0)
        qid2 = torch.cat([query_ids, query_ids], 0)
        tid2 = torch.cat([txt_ids, txt_ids], 0)
        tmk2 = torch.cat([txt_mask, txt_mask], 0)
        qe = self.q_embeddings(input_ids=qid2, query_embeds=None)
        pair = torch.cat([qe, self.q_embeddings(input_ids=tid2)], 1)
        att = torch.cat([torch.ones(qe.shape[:-1], dtype=tmk2.dtype, device=tmk2.device),
                         tmk2], 1)
        iatt = torch.ones(img2.shape[:-1], dtype=torch.long, device=img2.device)
        out = self.qformer(query_embeds=pair, query_length=qe.size(1), attention_mask=att,
                           encoder_hidden_states=img2,
                           encoder_attention_mask=iatt).last_hidden_state
        logits = self.itm_head(out[:, :qe.size(1)]).mean(1)                # [2B,2]
        lbl = torch.cat([torch.ones(B, dtype=torch.long, device=tgt.device),
                         torch.zeros(B, dtype=torch.long, device=tgt.device)])
        loss_itm = torch.nn.functional.cross_entropy(logits, lbl)
        return loss_itc, loss_itm

    def forward(self, image_embeds, query_ids, ie_ids, ie_mask, prompt_mask,
                span_labels=None, txt_ids=None, txt_mask=None,
                itc_w: float = 0.0, itm_w: float = 0.0):
        # --- PDQ: the prompt's own token embeddings are the Q-Former queries ---
        qe = self.q_embeddings(input_ids=query_ids)
        img_att = torch.ones(image_embeds.shape[:-1], dtype=torch.long,
                             device=image_embeds.device)
        qout = self.qformer(query_embeds=qe, query_length=qe.size(1),
                            encoder_hidden_states=image_embeds,
                            encoder_attention_mask=img_att).last_hidden_state
        prefix = self.drop(self.proj(qout[:, :self.n_query]))

        # --- prepend the visual-grounded prefix to the text encoder ---
        we = self.text.get_input_embeddings()(ie_ids)
        inp = torch.cat([prefix, we], dim=1)
        att = torch.cat([torch.ones(prefix.shape[:-1], dtype=ie_mask.dtype,
                                    device=ie_mask.device), ie_mask], dim=1)
        h = self.text(inputs_embeds=inp, attention_mask=att).last_hidden_state

        span_logits = self.epe(h.float(), att.float())
        span_logits = span_logits * prompt_mask - (1 - prompt_mask) * 1e12

        loss = None
        if span_labels is not None:
            n_valid = prompt_mask.sum().clamp(min=1.0)
            loss = self.bce(span_logits.reshape(span_logits.size(0), -1),
                            span_labels.reshape(span_labels.size(0), -1)) / n_valid
            if (itc_w > 0 or itm_w > 0) and txt_ids is not None:
                l_itc, l_itm = self.itc_itm(image_embeds, query_ids, txt_ids, txt_mask, qout)
                loss = loss + itc_w * l_itc + itm_w * l_itm
        return span_logits, loss


# --------------------------------------------------------------------------- #
# data
# --------------------------------------------------------------------------- #
class VitCache:
    def __init__(self, dataset: str):
        d = DATA / "vit_cache" / dataset
        meta = json.load(open(d / "index.json"))
        self.index: Dict[str, int] = meta["index"]
        self.shape = tuple(meta["shape"])
        self.mm = np.memmap(d / "feats.f16", dtype=np.float16, mode="r", shape=self.shape)

    def get(self, image_id: str) -> np.ndarray:
        i = self.index.get(image_id)
        if i is None:
            return np.zeros(self.shape[1:], dtype=np.float16)
        return np.asarray(self.mm[i])


class MascDS(Dataset):
    """One item per aspect. Prompt carries the aspect AND the option list."""

    def __init__(self, examples, tok, cache: VitCache, n_query: int = 32,
                 max_len: int = 128, qtok=None):
        self.ex = examples
        self.tok = tok
        # DQPSA keeps TWO tokenizers (`IE_tokenizer`, `PQ_former_tokenizer`). The query
        # ids index the Q-Former's own BERT embedding table, so they must come from the
        # BLIP-2 tokenizer -- feeding e.g. bertweet ids (64k vocab) into a 30k table is an
        # out-of-bounds gather, not a silent mismatch.
        self.qtok = qtok if qtok is not None else tok
        self.cache = cache
        self.n_query = n_query
        self.max_len = max_len
        # option words must be SINGLE tokens so the answer is one diagonal cell.
        # BPE tokenizers (roberta/bertweet) need the leading space form (" positive").
        self.opt_ids = []
        for o in OPTIONS:
            cand = tok(" " + o, add_special_tokens=False)["input_ids"]
            if len(cand) != 1:
                cand = tok(o, add_special_tokens=False)["input_ids"]
            assert len(cand) == 1, f"option {o!r} is not a single token for this tokenizer"
            self.opt_ids.append(cand[0])

    def __len__(self):
        return len(self.ex)

    def __getitem__(self, i):
        e = self.ex[i]
        prompt = f"sentiment of {e.term} [ positive , neutral , negative ]"
        query = self.qtok(prompt, add_special_tokens=False, truncation=True,
                          max_length=self.n_query, padding="max_length")["input_ids"]
        enc = self.tok(prompt, e.marked_text(), truncation="only_second",
                       max_length=self.max_len)
        ids = enc["input_ids"]
        # locate the three option tokens inside the prompt segment
        opt_pos = {}
        for p, t in enumerate(ids):
            for o, oid in zip(OPTIONS, self.opt_ids):
                if t == oid and o not in opt_pos:
                    opt_pos[o] = p
        blip_txt = self.qtok(e.marked_text(), add_special_tokens=True, truncation=True,
                             max_length=48)["input_ids"]
        return {"ids": ids, "query": query, "opt_pos": opt_pos, "blip_txt": blip_txt,
                "gold": e.polarity, "image_id": e.image_id, "key": e.key}


def make_collate(cache: VitCache, pad_id: int, n_query: int):
    def collate(batch):
        B = len(batch)
        L = max(len(b["ids"]) for b in batch)
        T = n_query + L
        LT = max(len(b["blip_txt"]) for b in batch)
        txt = torch.zeros((B, LT), dtype=torch.long)
        tmask = torch.zeros((B, LT), dtype=torch.long)
        ids = torch.full((B, L), pad_id, dtype=torch.long)
        mask = torch.zeros((B, L), dtype=torch.long)
        query = torch.zeros((B, n_query), dtype=torch.long)
        pmask = torch.zeros((B, T, T), dtype=torch.float)
        labels = torch.zeros((B, T, T), dtype=torch.float)
        feats = np.zeros((B, cache.shape[1], cache.shape[2]), dtype=np.float16)
        opt_cells = []
        for i, b in enumerate(batch):
            n = len(b["ids"])
            ids[i, :n] = torch.tensor(b["ids"])
            mask[i, :n] = 1
            query[i] = torch.tensor(b["query"])
            nt = len(b["blip_txt"])
            txt[i, :nt] = torch.tensor(b["blip_txt"]); tmask[i, :nt] = 1
            feats[i] = cache.get(b["image_id"])
            cells = {}
            for o, p in b["opt_pos"].items():
                c = n_query + p                    # shift past the visual prefix
                pmask[i, c, c] = 1.0
                cells[o] = c
            opt_cells.append(cells)
            g = {v: k for k, v in OPT2POL.items()}[b["gold"]]
            if g in cells:
                labels[i, cells[g], cells[g]] = 1.0
        return {"ids": ids, "mask": mask, "query": query, "txt": txt, "tmask": tmask,
                "feats": torch.from_numpy(feats), "pmask": pmask, "labels": labels,
                "opt_cells": opt_cells, "gold": [b["gold"] for b in batch],
                "key": [b["key"] for b in batch]}
    return collate


# --------------------------------------------------------------------------- #
# train / eval
# --------------------------------------------------------------------------- #
@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    probs, golds, keys = [], [], []
    for batch in loader:
        with torch.autocast("cuda", dtype=torch.float16, enabled=device.type == "cuda"):
            logits, _ = model(batch["feats"].to(device, torch.float32),
                              batch["query"].to(device), batch["ids"].to(device),
                              batch["mask"].to(device), batch["pmask"].to(device))
        logits = logits.float()
        for i, cells in enumerate(batch["opt_cells"]):
            # emit columns in POLARITIES order (NEG, NEU, POS) so every member's
            # probs array is directly combinable in assemble.py
            row = []
            for pol in POLARITIES:
                o = {v: k for k, v in OPT2POL.items()}[pol]
                row.append(logits[i, cells[o], cells[o]].item() if o in cells else -1e12)
            probs.append(torch.softmax(torch.tensor(row), dim=0).numpy())
            golds.append(batch["gold"][i])
            keys.append(batch["key"][i])
    P = np.stack(probs)
    pred = [POLARITIES[i] for i in P.argmax(1)]
    acc = 100.0 * float(np.mean([p == g for p, g in zip(pred, golds)]))
    return acc, P, pred, golds, keys


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="twitter2015")
    ap.add_argument("--text-model", default="bert-base-uncased")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--epochs", type=int, default=8)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--accum", type=int, default=2)
    ap.add_argument("--lr", type=float, default=2e-5)
    ap.add_argument("--head-lr", type=float, default=1e-4)
    ap.add_argument("--patience", type=int, default=3)
    ap.add_argument("--itc", type=float, default=0.0, help="ITC loss weight (DQPSA: 1.0)")
    ap.add_argument("--itm", type=float, default=0.0, help="ITM loss weight (DQPSA: 1.0)")
    ap.add_argument("--n-query", type=int, default=32)
    ap.add_argument("--spans", default=None,
                    help="dir with spans_{dev,test}.json from stage-1; also score those "
                         "candidate anchors (written to probs_span_*.npz)")
    ap.add_argument("--score-only", action="store_true",
                    help="skip training, load <out>/best.pt and just (re-)score")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)

    from transformers import AutoTokenizer, get_linear_schedule_with_warmup
    tok = AutoTokenizer.from_pretrained(args.text_model)
    qtok = AutoTokenizer.from_pretrained(BLIP_ID)   # Q-Former side (BERT vocab)
    cache = VitCache(args.dataset)
    splits = {s: load(args.dataset, s) for s in ("train", "dev", "test")}
    ex = {s: masc_examples(v) for s, v in splits.items()}
    ds = {s: MascDS(v, tok, cache, args.n_query, qtok=qtok) for s, v in ex.items()}
    coll = make_collate(cache, tok.pad_token_id, args.n_query)
    dl = {s: DataLoader(v, batch_size=args.batch, shuffle=(s == "train"), collate_fn=coll)
          for s, v in ds.items()}
    print({s: len(v) for s, v in ex.items()}, flush=True)

    model = PDQ(args.text_model, args.n_query).to(device)
    head = [p for n, p in model.named_parameters()
            if n.startswith(("proj", "epe"))]
    body = [p for n, p in model.named_parameters()
            if not n.startswith(("proj", "epe"))]
    opt = torch.optim.AdamW([{"params": body, "lr": args.lr},
                             {"params": head, "lr": args.head_lr}], weight_decay=0.01)
    steps = (len(dl["train"]) // args.accum + 1) * args.epochs
    sched = get_linear_schedule_with_warmup(opt, int(0.1 * steps), steps)
    scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda")

    best, best_ep, bad = -1.0, -1, 0
    t0 = time.time()
    n_epochs = 0 if args.score_only else args.epochs
    for ep in range(1, n_epochs + 1):
        model.train()
        tot = 0.0
        opt.zero_grad(set_to_none=True)
        for i, batch in enumerate(dl["train"]):
            with torch.autocast("cuda", dtype=torch.float16, enabled=device.type == "cuda"):
                _, loss = model(batch["feats"].to(device, torch.float32),
                                batch["query"].to(device), batch["ids"].to(device),
                                batch["mask"].to(device), batch["pmask"].to(device),
                                batch["labels"].to(device),
                                txt_ids=batch["txt"].to(device),
                                txt_mask=batch["tmask"].to(device),
                                itc_w=args.itc, itm_w=args.itm)
            scaler.scale(loss / args.accum).backward()
            tot += loss.item()
            if (i + 1) % args.accum == 0:
                scaler.unscale_(opt)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                scaler.step(opt); scaler.update()
                opt.zero_grad(set_to_none=True); sched.step()
        dev_acc, *_ = evaluate(model, dl["dev"], device)
        print(f"ep{ep} loss {tot/len(dl['train']):.4f} | dev MASC acc {dev_acc:.2f} "
              f"| {time.time()-t0:.0f}s", flush=True)
        if dev_acc > best:
            best, best_ep, bad = dev_acc, ep, 0
            torch.save(model.state_dict(), out / "best.pt")
        else:
            bad += 1
            if bad >= args.patience:
                print("early stop"); break

    # The pre-C2c members (pdq_*_s4x) were trained before the ITC/ITM heads existed, so
    # their checkpoints lack temp/vision_projection/text_projection/itm_head while the class
    # now always builds them. Those heads are pretraining objectives only -- polarity comes
    # from `proj`/`epe` -- so a non-strict load re-scores them faithfully. Anything missing
    # BEYOND that set is a real mismatch and still raises.
    sd = torch.load(out / "best.pt", map_location=device)
    missing, unexpected = model.load_state_dict(sd, strict=False)
    ITCITM = {"temp", "vision_projection.weight", "vision_projection.bias",
              "text_projection.weight", "text_projection.bias",
              "itm_head.weight", "itm_head.bias"}
    extra = set(missing) - ITCITM
    if extra or unexpected:
        raise RuntimeError(f"state_dict mismatch beyond the ITC/ITM heads: "
                           f"missing={sorted(extra)} unexpected={sorted(unexpected)}")
    if missing:
        print(f"[score-only] pre-ITC/ITM checkpoint: {len(missing)} pretraining-head "
              f"tensors left at init (unused for polarity)", flush=True)
    res = {"text_model": args.text_model, "seed": args.seed,
           "best_dev_acc": best, "best_epoch": best_ep}
    for s in ("dev", "test"):
        acc, P, pred, golds, keys = evaluate(model, dl[s], device)
        res[f"{s}_acc"] = acc
        np.savez_compressed(out / f"probs_{s}.npz", probs=P,
                            keys=np.array(keys, dtype=np.int64),
                            gold=np.array(golds), pred=np.array(pred))
        print(f"[{s}] gold-span MASC acc {acc:.2f} (n={len(keys)})", flush=True)
        if args.spans:
            sp = json.load(open(Path(args.spans) / f"spans_{s}.json"))
            e2 = masc_examples_for_spans(splits[s], [[tuple(x) for x in i] for i in sp])
            dl2 = DataLoader(MascDS(e2, tok, cache, args.n_query, qtok=qtok),
                             batch_size=args.batch, collate_fn=coll)
            _, P2, _, _, K2 = evaluate(model, dl2, device)
            np.savez_compressed(out / f"probs_span_{s}.npz", probs=P2,
                                keys=np.array(K2, dtype=np.int64))
            print(f"[{s}] scored {len(e2)} predicted spans", flush=True)
    json.dump(res, open(out / "metrics.json", "w"), indent=2)
    print(json.dumps(res, indent=2))


if __name__ == "__main__":
    main()
