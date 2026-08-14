"""D23 — predict P(pair correct) from the text, and use it as the acceptance score.

**The arithmetic that motivates this.** Micro-F1 is `2*TP/(|pred| + |gold|)`, so dropping a
prediction whose probability of being correct is `q` improves F1 exactly when `q < F1/2`.
At our operating point (TP 728, kept 1032, gold 1037, F1 70.6) that cut is **q < 0.353**.
Applying it with oracle knowledge of `q` projects to **F1 74.68**, comfortably past the
72.9 bar. Note this corrects the intuitive "extract iff q > 0.5" rule, which minimises
*error count* rather than F1 and over-drops (74.28).

**Why this is not §D.2's dead selector.** §D.2 fitted eleven *scalar* features (tagger conf,
C4 judge, PDQ-MATE, MASC confidence/margin/entropy, sentence context) and found they cap at
~70.8 even when fitted directly on TEST. This model reads the **raw marked text**. The gap
is measurable rather than assumed: MASC max-probability carries AUC **0.659** for "pair is
correct" (§D.2), while the OOF-estimated `q` carries AUC **0.906** across *held-out model
families* (2 towers estimating 2 different architectures). The information exists in the
input; the question this answers is whether a model can extract it.

**Why `q` is an estimate, not a label.** §D.11 established that residual errors are
*consensus* errors, so "all towers wrong" is a shared blind spot rather than proof of
unknowability. `q` is therefore the mean OOF `P(gold)` over four architecturally distinct
towers -- a smooth competence estimate. Vote-counting (`k/4`) throws away most of it: k=2
spans span q 0.29-0.71, and vote-count transfers at AUC 0.855 against the probability
estimate's 0.906.

Targets, on OOF MATE candidates so the model sees realistic extraction errors:
    gold span      -> q   (probability the polarity is right)
    non-gold span  -> 0   (the pair is wrong whatever polarity is assigned)

    python experts/qpredict.py --out runs/qpred_btwL
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
from torch.utils.data import DataLoader, Dataset

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from experts.assemble import decode_spans, load_mate  # noqa: E402
from experts.common import load, set_seed  # noqa: E402


class DS(Dataset):
    """One item per candidate span: the tweet with that span marked, target in [0,1]."""

    def __init__(self, items, tok, max_len=160):
        self.items, self.tok, self.max_len = items, tok, max_len

    def __len__(self):
        return len(self.items)

    def __getitem__(self, i):
        toks, (s, e), t, key = self.items[i]
        marked = " ".join(toks[:s] + ["["] + toks[s:e] + ["]"] + toks[e:])
        term = " ".join(toks[s:e])
        enc = self.tok(term, marked, truncation="only_second", max_length=self.max_len)
        return {"ids": enc["input_ids"], "t": t, "key": key}


def collate(b, pad_id):
    L = max(len(x["ids"]) for x in b)
    ids = torch.full((len(b), L), pad_id, dtype=torch.long)
    m = torch.zeros((len(b), L), dtype=torch.long)
    for i, x in enumerate(b):
        n = len(x["ids"])
        ids[i, :n] = torch.tensor(x["ids"]); m[i, :n] = 1
    return {"ids": ids, "mask": m,
            "t": torch.tensor([x["t"] for x in b], dtype=torch.float),
            "key": [x["key"] for x in b]}


class QNet(nn.Module):
    def __init__(self, model_id, dropout=0.1):
        super().__init__()
        from transformers import AutoConfig, AutoModel
        cfg = AutoConfig.from_pretrained(model_id)
        self.enc = AutoModel.from_pretrained(model_id, dtype=torch.float32)
        self.drop = nn.Dropout(dropout)
        self.head = nn.Linear(cfg.hidden_size, 1)

    def forward(self, ids, mask):
        h = self.enc(input_ids=ids, attention_mask=mask).last_hidden_state
        m = mask.unsqueeze(-1).float()
        return self.head(self.drop((h * m).sum(1) / m.sum(1).clamp(min=1))).squeeze(-1).float()


def build_train(dataset, oof_dirs, qbuf, cand_thr):
    """OOF MATE candidates, never train predictions: a model scores ~99 F1 on data it
    memorised, so its train candidates contain almost no realistic errors (§C.19)."""
    insts = load(dataset, "train")
    q = {(int(k[0]), int(k[1]), int(k[2])): float(v)
         for k, v in zip(qbuf["keys"], qbuf["q"])}
    items = []
    for d in oof_dirs:
        fold = int(str(d).rstrip("/")[-1])
        z = np.load(Path(d) / f"marginals_oof{fold}.npz")
        idx = json.load(open(Path(d) / f"oofidx{fold}.json"))
        for j, i in enumerate(idx):
            marg = z[str(j)]
            gm = {(a, b): p for (a, b, p) in insts[i].aspects}
            for (a, b, _) in decode_spans(marg, cand_thr):
                t = q.get((i, a, b), 0.0) if (a, b) in gm else 0.0
                items.append((insts[i].tokens, (a, b), t, (i, a, b)))
    return items


def build_eval(dataset, split, mate_dirs, cand_thr):
    insts = load(dataset, split)
    marg = load_mate(mate_dirs, split)
    items = []
    for i, (inst, m) in enumerate(zip(insts, marg)):
        for (a, b, _) in decode_spans(m, cand_thr):
            items.append((inst.tokens, (a, b), 0.0, (i, a, b)))
    return items


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="vinai/bertweet-large")
    ap.add_argument("--dataset", default="twitter2015")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--epochs", type=int, default=4)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--lr", type=float, default=1e-5)
    ap.add_argument("--head-lr", type=float, default=1e-3)
    ap.add_argument("--cand-thr", type=float, default=0.12)
    ap.add_argument("--qbuf", default="data/qdet_train.npz")
    ap.add_argument("--oof", nargs="+", default=["runs/mate_oof_f0", "runs/mate_oof_f1"])
    ap.add_argument("--mate", nargs="+",
                    default=["runs/mate_deb_s42", "runs/mate_deb_s43", "runs/mate_deb_s44",
                             "runs/mate_probeA", "runs/mate_probeB"])
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    from transformers import AutoTokenizer, get_linear_schedule_with_warmup
    tok = AutoTokenizer.from_pretrained(args.model)
    qbuf = np.load(args.qbuf)

    tr_items = build_train(args.dataset, args.oof, qbuf, args.cand_thr)
    tgt = np.array([t for (_, _, t, _) in tr_items])
    print(f"train candidates {len(tr_items)}  |  mean target {tgt.mean():.3f}  |  "
          f"below F1/2=0.353: {int((tgt < 0.353).sum())} ({100*(tgt<0.353).mean():.1f}%)",
          flush=True)

    coll = lambda b: collate(b, tok.pad_token_id)
    dl_tr = DataLoader(DS(tr_items, tok), batch_size=args.batch, shuffle=True,
                       collate_fn=coll)
    model = QNet(args.model).to(device)
    body = [p for n, p in model.named_parameters() if n.startswith("enc.")]
    head = [p for n, p in model.named_parameters() if not n.startswith("enc.")]
    opt = torch.optim.AdamW([{"params": body, "lr": args.lr},
                             {"params": head, "lr": args.head_lr}], weight_decay=0.01)
    steps = len(dl_tr) * args.epochs
    sched = get_linear_schedule_with_warmup(opt, int(0.1 * steps), steps)
    scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda")

    t0 = time.time()
    for ep in range(1, args.epochs + 1):
        model.train(); tot = 0.0
        for b in dl_tr:
            opt.zero_grad(set_to_none=True)
            with torch.autocast("cuda", dtype=torch.float16, enabled=device.type == "cuda"):
                z = model(b["ids"].to(device), b["mask"].to(device))
            # soft-target BCE: the target is a probability, not a class
            loss = nn.functional.binary_cross_entropy_with_logits(z, b["t"].to(device))
            scaler.scale(loss).backward(); scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(opt); scaler.update(); sched.step()
            tot += float(loss)
        print(f"ep{ep} loss {tot/len(dl_tr):.4f} | {time.time()-t0:.0f}s", flush=True)

    model.eval()
    for split in ("dev", "test"):
        it = build_eval(args.dataset, split, args.mate, args.cand_thr)
        dl = DataLoader(DS(it, tok), batch_size=64, collate_fn=coll)
        P, K = [], []
        with torch.no_grad():
            for b in dl:
                with torch.autocast("cuda", dtype=torch.float16,
                                    enabled=device.type == "cuda"):
                    z = model(b["ids"].to(device), b["mask"].to(device))
                P.append(torch.sigmoid(z.float()).cpu().numpy()); K.extend(b["key"])
        P = np.concatenate(P)
        np.savez_compressed(out / f"qhat_{split}.npz", q=P.astype(np.float32),
                            keys=np.array(K, dtype=np.int64))
        print(f"[{split}] scored {len(P)} candidates  mean q_hat {P.mean():.3f}", flush=True)
    json.dump({"model": args.model, "seed": args.seed, "n_train": len(tr_items)},
              open(out / "metrics.json", "w"), indent=2)


if __name__ == "__main__":
    main()
