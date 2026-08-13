"""C4 — annotation-policy span reranker (attacks MATE PRECISION).

MATE precision is the binding constraint (Chapter B §7d) and Chapter A diagnosed exactly
why it is hard: **110 of 182 dev false positives were reasonable-but-unannotated entities**
("Regions Bank", "# VMworld"). Those are not extraction errors in any linguistic sense —
they are *annotation-policy* misses. A tagger's per-token confidence cannot represent
"this is a real entity that this dataset's annotators would not have marked"; a dedicated
binary judge over whole candidate spans can.

Training data comes from **MATE out-of-fold** predictions, never from the training set
directly: a MATE model scores ~99 F1 on data it memorised, so its train predictions contain
almost no false positives and would teach the reranker nothing.

Output is a recalibrated per-span score consumed by `assemble.py --rerank`, which replaces
the tagger's `mean(1 - P(O))` confidence in the tau decision.

    python experts/span_rerank.py --oof runs/mate_oof_f0 runs/mate_oof_f1 \
        --cand runs/mate_ens5 --out runs/rerank
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
from experts.assemble import decode_spans, load_mate  # noqa: E402
from experts.common import load, set_seed  # noqa: E402


class CandDS(Dataset):
    """One item per CANDIDATE span: sentence with the candidate marked, binary label."""

    def __init__(self, items, tok, max_len=160):
        self.items, self.tok, self.max_len = items, tok, max_len

    def __len__(self):
        return len(self.items)

    def __getitem__(self, i):
        toks, (s, e), y, key, conf = self.items[i]
        marked = " ".join(toks[:s] + ["["] + toks[s:e] + ["]"] + toks[e:])
        term = " ".join(toks[s:e])
        enc = self.tok(term, marked, truncation="only_second", max_length=self.max_len)
        return {"ids": enc["input_ids"], "y": y, "key": key, "conf": conf}


def collate(b, pad_id):
    L = max(len(x["ids"]) for x in b)
    ids = torch.full((len(b), L), pad_id, dtype=torch.long)
    m = torch.zeros((len(b), L), dtype=torch.long)
    for i, x in enumerate(b):
        n = len(x["ids"])
        ids[i, :n] = torch.tensor(x["ids"])
        m[i, :n] = 1
    return {"ids": ids, "mask": m,
            "y": torch.tensor([x["y"] for x in b], dtype=torch.float),
            "conf": torch.tensor([x["conf"] for x in b], dtype=torch.float),
            "key": [x["key"] for x in b]}


class Judge(nn.Module):
    def __init__(self, model_id, dropout=0.1):
        super().__init__()
        from transformers import AutoConfig, AutoModel
        cfg = AutoConfig.from_pretrained(model_id)
        self.enc = AutoModel.from_pretrained(model_id, dtype=torch.float32)
        self.drop = nn.Dropout(dropout)
        self.head = nn.Linear(cfg.hidden_size + 1, 1)   # +1 = tagger confidence

    def forward(self, ids, mask, conf):
        h = self.enc(input_ids=ids, attention_mask=mask).last_hidden_state
        m = mask.unsqueeze(-1).float()
        pooled = (h * m).sum(1) / m.sum(1).clamp(min=1)
        return self.head(self.drop(torch.cat([pooled, conf[:, None]], -1))).squeeze(-1).float()


def build(insts, marg_list, idx_list, cand_thr):
    """(tokens, span, label, key, tagger_conf) for every candidate span."""
    out = []
    for marg, gidx in zip(marg_list, idx_list):
        for local, m in enumerate(marg):
            inst = insts[gidx[local]]
            gold = {(a, b) for (a, b, _) in inst.aspects}
            for (s, e, c) in decode_spans(m, cand_thr):
                out.append((inst.tokens, (s, e), float((s, e) in gold),
                            (gidx[local], s, e), c))
    return out


@torch.no_grad()
def score(model, loader, device):
    model.eval()
    P, K = [], []
    for b in loader:
        with torch.autocast("cuda", dtype=torch.float16, enabled=device.type == "cuda"):
            lg = model(b["ids"].to(device), b["mask"].to(device), b["conf"].to(device))
        P.append(torch.sigmoid(lg.float()).cpu().numpy())
        K.extend(b["key"])
    return np.concatenate(P), K


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="twitter2015")
    ap.add_argument("--model", default="vinai/bertweet-large")
    ap.add_argument("--oof", nargs="+", required=True)
    ap.add_argument("--cand", required=True, help="dir of the full MATE ensemble members")
    ap.add_argument("--mate", nargs="+", required=True)
    ap.add_argument("--cand-thr", type=float, default=0.12)
    ap.add_argument("--seed", type=int, default=100)
    ap.add_argument("--epochs", type=int, default=5)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--lr", type=float, default=1e-5)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    from transformers import AutoTokenizer, get_linear_schedule_with_warmup
    tok = AutoTokenizer.from_pretrained(args.model)

    tr_insts = load(args.dataset, "train")
    margs, idxs = [], []
    for d in args.oof:
        f = list(Path(d).glob("marginals_oof*.npz"))[0]
        z = np.load(f)
        margs.append([z[str(i)] for i in range(len(z.files))])
        idxs.append(json.load(open(Path(d) / f"oofidx{f.stem[-1]}.json")))
    train_items = build(tr_insts, margs, idxs, args.cand_thr)
    pos = sum(x[2] for x in train_items)
    print(f"train candidates {len(train_items)}  positives {int(pos)} "
          f"({100*pos/len(train_items):.1f}%)", flush=True)

    ev = {}
    for split in ("dev", "test"):
        insts = load(args.dataset, split)
        m = load_mate(args.mate, split)
        ev[split] = build(insts, [m], [list(range(len(insts)))], args.cand_thr)
        p = sum(x[2] for x in ev[split])
        print(f"{split} candidates {len(ev[split])} positives {int(p)} "
              f"({100*p/len(ev[split]):.1f}%)", flush=True)

    coll = lambda b: collate(b, tok.pad_token_id)
    dl_tr = DataLoader(CandDS(train_items, tok), batch_size=args.batch, shuffle=True,
                       collate_fn=coll)
    model = Judge(args.model).to(device)
    body = [p for n, p in model.named_parameters() if n.startswith("enc.")]
    head = [p for n, p in model.named_parameters() if not n.startswith("enc.")]
    opt = torch.optim.AdamW([{"params": body, "lr": args.lr},
                             {"params": head, "lr": 1e-3}], weight_decay=0.01)
    steps = len(dl_tr) * args.epochs
    sched = get_linear_schedule_with_warmup(opt, int(0.1 * steps), steps)
    scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda")
    lossf = nn.BCEWithLogitsLoss()

    best, t0 = -1.0, time.time()
    for ep in range(1, args.epochs + 1):
        model.train(); tot = 0.0
        for b in dl_tr:
            opt.zero_grad(set_to_none=True)
            with torch.autocast("cuda", dtype=torch.float16, enabled=device.type == "cuda"):
                lg = model(b["ids"].to(device), b["mask"].to(device), b["conf"].to(device))
            loss = lossf(lg, b["y"].to(device))
            scaler.scale(loss).backward(); scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(opt); scaler.update(); sched.step()
            tot += loss.item()
        p, k = score(model, DataLoader(CandDS(ev["dev"], tok), batch_size=args.batch,
                                       collate_fn=coll), device)
        y = np.array([x[2] for x in ev["dev"]])
        auc = float(((p[:, None] > p[None, :]) * (y[:, None] > y[None, :])).sum() /
                    max((y[:, None] > y[None, :]).sum(), 1))
        print(f"ep{ep} loss {tot/len(dl_tr):.4f} | dev AUC {auc:.4f} | "
              f"{time.time()-t0:.0f}s", flush=True)
        if auc > best:
            best = auc
            torch.save(model.state_dict(), out / "best.pt")

    model.load_state_dict(torch.load(out / "best.pt", map_location=device))
    for split in ("dev", "test"):
        p, k = score(model, DataLoader(CandDS(ev[split], tok), batch_size=args.batch,
                                       collate_fn=coll), device)
        np.savez_compressed(out / f"spanscore_{split}.npz", score=p,
                            keys=np.array(k, dtype=np.int64))
        print(f"[{split}] scored {len(k)} candidate spans", flush=True)
    json.dump({"dev_auc": best, "cand_thr": args.cand_thr, "model": args.model},
              open(out / "metrics.json", "w"), indent=2)


if __name__ == "__main__":
    main()
