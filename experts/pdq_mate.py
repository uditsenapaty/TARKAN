"""C2b — PDQ as an ARCHITECTURE-DIVERSE MATE member (span-level, not BIO).

Chapter B's MATE ensemble plateaued at 88.0 and the diagnosis was explicit: MATE "needs
ARCH-DIVERSE strong members, not more DeBERTa seeds". Every member it had was the same
animal -- a BIO/CRF token tagger on a text encoder.

This member is a different animal in the way that matters:
  * it scores (start, end) CELLS directly with the EPE RoPE bi-affine head, so there is no
    tag sequence and no BIO decode step at all -- the failure modes of token tagging
    (fragmentation, O-domination, transition errors) simply cannot occur;
  * it sees the image through the same VL-pretrained Q-Former bridge as `pdq.py`.

DQPSA reports MATE 87.7 on t2015 with this head, i.e. competitive with our best BIO
ensemble, which is exactly what a useful ensemble partner needs to be.

Candidate cells are restricted to (word-start subtoken, word-end subtoken) pairs: aspects
are word-aligned by construction and we score at word level, so scoring sub-word cells
would only add noise.

    python experts/pdq_mate.py --seed 42 --out runs/pdqmate_s42
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import List, Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from experts.common import gold_spans, load, score_spans, set_seed  # noqa: E402
from experts.pdq import BLIP_ID, PDQ, VitCache  # noqa: E402

PROMPT = "find the aspect terms in the text"


class MateDS(Dataset):
    def __init__(self, insts, tok, cache: VitCache, n_query=32, max_len=160, qtok=None):
        self.insts, self.tok, self.cache = insts, tok, cache
        # query ids index the Q-Former's BERT table -> must use the BLIP-2 tokenizer
        self.qtok = qtok if qtok is not None else tok
        self.n_query, self.max_len = n_query, max_len
        self.prompt_ids = tok(PROMPT, add_special_tokens=False)["input_ids"]

    def __len__(self):
        return len(self.insts)

    def __getitem__(self, i):
        inst = self.insts[i]
        enc = self.tok(inst.tokens, is_split_into_words=True, add_special_tokens=False,
                       truncation=True, max_length=self.max_len)
        wids = enc.word_ids()
        first, last = {}, {}
        for pos, w in enumerate(wids):
            if w is None:
                continue
            if w not in first:
                first[w] = pos
            last[w] = pos
        cls, sep = self.tok.cls_token_id, self.tok.sep_token_id
        ids = [cls] + self.prompt_ids + [sep] + enc["input_ids"] + [sep]
        off = 1 + len(self.prompt_ids) + 1
        starts = {w: off + p for w, p in first.items()}
        ends = {w: off + p for w, p in last.items()}
        gold = [(s, e) for (s, e, _) in inst.aspects if s in starts and (e - 1) in ends]
        return {"ids": ids, "starts": starts, "ends": ends, "gold": gold,
                "query": self.qtok(PROMPT, add_special_tokens=False, truncation=True,
                                   max_length=self.n_query,
                                   padding="max_length")["input_ids"],
                "image_id": inst.image_id, "n_words": len(inst.tokens), "idx": i}


def make_collate(cache: VitCache, pad_id: int, n_query: int):
    def collate(batch):
        B = len(batch)
        L = max(len(b["ids"]) for b in batch)
        T = n_query + L
        ids = torch.full((B, L), pad_id, dtype=torch.long)
        mask = torch.zeros((B, L), dtype=torch.long)
        query = torch.zeros((B, n_query), dtype=torch.long)
        pmask = torch.zeros((B, T, T), dtype=torch.float)
        labels = torch.zeros((B, T, T), dtype=torch.float)
        feats = np.zeros((B, cache.shape[1], cache.shape[2]), dtype=np.float16)
        for i, b in enumerate(batch):
            n = len(b["ids"])
            ids[i, :n] = torch.tensor(b["ids"])
            mask[i, :n] = 1
            query[i] = torch.tensor(b["query"])
            feats[i] = cache.get(b["image_id"])
            for ws, sp in b["starts"].items():
                for we, ep in b["ends"].items():
                    if we >= ws:
                        pmask[i, n_query + sp, n_query + ep] = 1.0
            for (s, e) in b["gold"]:
                labels[i, n_query + b["starts"][s], n_query + b["ends"][e - 1]] = 1.0
        return {"ids": ids, "mask": mask, "query": query,
                "feats": torch.from_numpy(feats), "pmask": pmask, "labels": labels,
                "meta": batch}
    return collate


def decode(logits: torch.Tensor, b: dict, n_query: int, thr: float
           ) -> List[Tuple[int, int, float]]:
    """Greedy non-overlapping selection over word-aligned (start,end) cells."""
    inv_s = {v: k for k, v in b["starts"].items()}
    inv_e = {v: k for k, v in b["ends"].items()}
    cand = []
    for sp, ws in inv_s.items():
        for ep, we in inv_e.items():
            if we < ws:
                continue
            p = torch.sigmoid(logits[n_query + sp, n_query + ep]).item()
            if p > thr:
                cand.append((p, ws, we + 1))
    cand.sort(reverse=True)
    out, used = [], set()
    for p, s, e in cand:
        if any(w in used for w in range(s, e)):
            continue
        used.update(range(s, e))
        out.append((s, e, p))
    return sorted(out)


@torch.no_grad()
def predict(model, loader, device, n_query, thr):
    model.eval()
    spans = [None] * len(loader.dataset)
    for batch in loader:
        with torch.autocast("cuda", dtype=torch.float16, enabled=device.type == "cuda"):
            lg, _ = model(batch["feats"].to(device, torch.float32),
                          batch["query"].to(device), batch["ids"].to(device),
                          batch["mask"].to(device), batch["pmask"].to(device))
        lg = lg.float().cpu()
        for i, b in enumerate(batch["meta"]):
            spans[b["idx"]] = decode(lg[i], b, n_query, thr)
    return spans


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="twitter2015")
    ap.add_argument("--text-model", default="bert-base-uncased")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--epochs", type=int, default=10)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--accum", type=int, default=2)
    ap.add_argument("--lr", type=float, default=2e-5)
    ap.add_argument("--head-lr", type=float, default=1e-4)
    ap.add_argument("--patience", type=int, default=3)
    ap.add_argument("--n-query", type=int, default=32)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)

    from transformers import AutoTokenizer, get_linear_schedule_with_warmup
    tok = AutoTokenizer.from_pretrained(args.text_model)
    qtok = AutoTokenizer.from_pretrained(BLIP_ID)
    cache = VitCache(args.dataset)
    splits = {s: load(args.dataset, s) for s in ("train", "dev", "test")}
    coll = make_collate(cache, tok.pad_token_id, args.n_query)
    dl = {s: DataLoader(MateDS(v, tok, cache, args.n_query, qtok=qtok), batch_size=args.batch,
                        shuffle=(s == "train"), collate_fn=coll)
          for s, v in splits.items()}
    gold = {s: gold_spans(v) for s, v in splits.items()}

    model = PDQ(args.text_model, args.n_query).to(device)
    head = [p for n, p in model.named_parameters() if n.startswith(("proj", "epe"))]
    body = [p for n, p in model.named_parameters() if not n.startswith(("proj", "epe"))]
    opt = torch.optim.AdamW([{"params": body, "lr": args.lr},
                             {"params": head, "lr": args.head_lr}], weight_decay=0.01)
    steps = (len(dl["train"]) // args.accum + 1) * args.epochs
    sched = get_linear_schedule_with_warmup(opt, int(0.1 * steps), steps)
    scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda")

    best, best_ep, best_thr, bad, t0 = -1.0, -1, 0.5, 0, time.time()
    for ep in range(1, args.epochs + 1):
        model.train()
        tot = 0.0
        opt.zero_grad(set_to_none=True)
        for i, batch in enumerate(dl["train"]):
            with torch.autocast("cuda", dtype=torch.float16, enabled=device.type == "cuda"):
                _, loss = model(batch["feats"].to(device, torch.float32),
                                batch["query"].to(device), batch["ids"].to(device),
                                batch["mask"].to(device), batch["pmask"].to(device),
                                batch["labels"].to(device))
            scaler.scale(loss / args.accum).backward()
            tot += loss.item()
            if (i + 1) % args.accum == 0:
                scaler.unscale_(opt)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                scaler.step(opt); scaler.update()
                opt.zero_grad(set_to_none=True); sched.step()
        # threshold is part of the decode rule -> tune it on dev, never on test
        bt, bf, bsc = 0.5, -1.0, None
        for thr in (0.2, 0.3, 0.4, 0.5, 0.6):
            sp = predict(model, dl["dev"], device, args.n_query, thr)
            sc = score_spans([[(s, e) for (s, e, _) in x] for x in sp], gold["dev"])
            if sc["F1"] > bf:
                bt, bf, bsc = thr, sc["F1"], sc
        print(f"ep{ep} loss {tot/len(dl['train']):.4f} | dev MATE F1 {bf:.2f} "
              f"(thr {bt}) P {bsc['P']:.2f} R {bsc['R']:.2f} | {time.time()-t0:.0f}s",
              flush=True)
        if bf > best:
            best, best_ep, best_thr, bad = bf, ep, bt, 0
            torch.save(model.state_dict(), out / "best.pt")
        else:
            bad += 1
            if bad >= args.patience:
                print("early stop"); break

    model.load_state_dict(torch.load(out / "best.pt", map_location=device))
    res = {"seed": args.seed, "best_dev_F1": best, "best_epoch": best_ep,
           "thr": best_thr, "text_model": args.text_model}
    for s in ("dev", "test"):
        sp = predict(model, dl[s], device, args.n_query, best_thr)
        sc = score_spans([[(a, b) for (a, b, _) in x] for x in sp], gold[s])
        res[s] = sc
        json.dump([[[a, b] for (a, b, _) in x] for x in sp],
                  open(out / f"spans_{s}.json", "w"))
        # word-level pseudo-marginals so this member can be averaged with BIO members
        marg = {}
        for i, x in enumerate(sp):
            nw = len(splits[s][i].tokens)
            m = np.zeros((nw, 3), dtype=np.float32)
            m[:, 0] = 1.0
            for (a, b, p) in x:
                m[a, 0], m[a, 1] = 1 - p, p
                for w in range(a + 1, b):
                    m[w, 0], m[w, 2] = 1 - p, p
            marg[str(i)] = m
        np.savez_compressed(out / f"marginals_{s}.npz", **marg)
        print(f"[{s}] MATE P {sc['P']:.2f} R {sc['R']:.2f} F1 {sc['F1']:.2f}", flush=True)
    json.dump(res, open(out / "metrics.json", "w"), indent=2)
    print(json.dumps(res, indent=2))


if __name__ == "__main__":
    main()
