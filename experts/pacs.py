"""D22 — PACS: Polarity-Aware Candidate State. One encoder, coupled extraction + polarity.

**The diagnostic this exists for (§C.18), which is the sharpest structural finding in the
whole project:**

    MASC accuracy on ALL gold spans        81.20
    MASC accuracy on MATE-EXTRACTED spans  80.52   <- this is `a`
    MASC accuracy on MATE-MISSED spans     86.73

The extractor systematically selects the spans its own classifier is **worst** at. MADSC's
published numbers imply the opposite sign (`a` 84.18 vs 82.34 gold-span), a ~2.5-point swing
in `a` -- almost exactly the joint gap. Chapters C/D built MATE and MASC as *deliberately
decorrelated* families, which is excellent for ensembling and exactly wrong for joint MABSA.

**Why this reframes the target.** The bar was being read as "get `a` to 83.1 on all gold
spans", which is above every published number. But spans at 86.73 accuracy already exist in
this system -- they are simply not selected. Holding MATE@tau at 87.8, moving
**`a_selected`** from 80.4 to ~83.5 clears 72.9. That is a selection problem inside a
population we already have, not a demand for SOTA polarity.

**Why this is not §D.2's frozen selector.** §D.2 measured that a selector over the existing
scores caps at ~70.8 even when fitted directly on TEST -- the information is absent from
those representations. Here the anchor score A(s) and the polarity P(y|s) are differentiable
functions of the *same* word representations, so a joint margin backprops into the encoder
and makes "is this span's polarity determinable" part of what the candidate state encodes.
The scoring rule is incidental; the representation learning is the mechanism. If PACS only
reproduces the post-hoc product it will land at ~70.8 and that is the falsification.

**The gate is `a_selected`, not joint F1.** A move of 80.5 -> 82+ is above the ±1.31
single-pair detection floor (§D.20) and is the thing being claimed. Joint F1 follows or it
does not.

t2015 only. TARKAN's identity is preserved: BIO anchor generation with a linear-chain CRF,
aspect-conditioned polarity, student-only inference.

    python experts/pacs.py --out runs/pacs_s42
"""
from __future__ import annotations

import argparse
import json
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from experts.common import (POL2ID, POLARITIES, gold_pairs, gold_spans,  # noqa: E402
                            load, score_joint, score_spans, set_seed, spans_from_obi)
from experts.mate_expert import CRF, NTAG, obi_tags  # noqa: E402

O_TAG = 0


class PACS(nn.Module):
    """One encoder feeding both heads, which is the whole point.

    The coupling is the JOINT MARGIN, not a feedback wire. `U = A(s) * P(y|s)^lam` is
    differentiable through both heads, which read the *same* word representations `hw`, so
    the margin gradient reaches `hw` through the anchor path and the polarity path at once.
    That is what a post-hoc product over frozen scores (§D.2, capped at ~70.8 even fitted on
    TEST) structurally cannot do.

    An explicit polarity->emissions feedback gate was tried first and removed: with both the
    gate scalar and its projection zero-initialised, d(loss)/d(alpha) = fb = 0 and
    d(loss)/d(fb) = alpha = 0, so BOTH had identically zero gradient forever -- a dead branch
    that would have contributed nothing while appearing principled. It was also applied only
    to gold spans in training and never at inference, i.e. train/test inconsistent. Removing
    it makes this a clean test of the coupling hypothesis rather than of two entangled
    mechanisms.
    """

    def __init__(self, model_id: str, dropout: float = 0.1):
        super().__init__()
        from transformers import AutoConfig, AutoModel
        cfg = AutoConfig.from_pretrained(model_id)
        h = cfg.hidden_size
        # deberta-v3-large ships fp16; force fp32 masters or AMP unscaling dies.
        self.encoder = AutoModel.from_pretrained(model_id, dtype=torch.float32)
        self.drop = nn.Dropout(dropout)
        self.proj = nn.Linear(h, NTAG)
        self.crf = CRF(NTAG)
        self.pol = nn.Sequential(nn.Linear(3 * h, h), nn.GELU(), nn.Linear(h, 3))

    def words(self, input_ids, attention_mask, word_index):
        h = self.encoder(input_ids=input_ids,
                         attention_mask=attention_mask).last_hidden_state
        idx = word_index.unsqueeze(-1).expand(-1, -1, h.size(-1))
        return h.gather(1, idx)                       # [B, Lw, H]

    def emissions(self, hw):
        return self.proj(self.drop(hw)).float()       # CRF always fp32

    def span_repr(self, hw, b, s, e):
        seg = hw[b, s:e]
        return torch.cat([seg.mean(0), seg[0], seg[-1]], -1)

    def polarity(self, hw, spans):
        """spans: list of (batch_idx, start, end) in word-row coordinates."""
        if not spans:
            return hw.new_zeros((0, 3))
        z = torch.stack([self.span_repr(hw, b, s, e) for (b, s, e) in spans])
        return self.pol(self.drop(z)).float()


def span_score(emis_sm, b, s, e):
    """A(s): mean aspect-bearing probability over the span. Differentiable, and the same
    quantity `decode_spans` thresholds at inference (mean of 1 - P(O))."""
    return (1.0 - emis_sm[b, s:e, O_TAG]).mean()


class PacsDS(Dataset):
    def __init__(self, insts, tok, max_len=160, train=False, n_neg=3):
        self.items, self.train, self.n_neg = [], train, n_neg
        for inst in insts:
            enc = tok(inst.tokens, is_split_into_words=True, truncation=True,
                      max_length=max_len)
            first = {}
            for pos, w in enumerate(enc.word_ids()):
                if w is not None and w not in first:
                    first[w] = pos
            keep = [w for w in range(len(inst.tokens)) if w in first]
            row = {w: r for r, w in enumerate(keep)}          # word idx -> word-row
            tags = obi_tags(inst)
            gold, pol = [], []
            for (s, e, p) in inst.aspects:
                if s in row and (e - 1) in row:
                    gold.append((row[s], row[e - 1] + 1)); pol.append(POL2ID[p])
            self.items.append({
                "input_ids": enc["input_ids"],
                "word_index": [first[w] for w in keep],
                "tags": [tags[w] for w in keep],
                "kept": keep, "n_words": len(inst.tokens),
                "gold": gold, "pol": pol, "n_rows": len(keep),
            })

    def __len__(self):
        return len(self.items)

    def __getitem__(self, i):
        it = self.items[i]
        if not self.train:
            return it
        # HARD candidate contrast: boundary errors and nearby spans, i.e. the mistakes the
        # anchor head actually makes -- not random tokens.
        gset = set(it["gold"]); negs = []
        L = it["n_rows"]
        for (s, e) in it["gold"]:
            cand = [(s + 1, e), (s, e - 1), (s - 1, e), (s, e + 1), (s + 1, e + 1)]
            cand = [(a, b) for (a, b) in cand if 0 <= a < b <= L and (a, b) not in gset]
            random.shuffle(cand)
            negs.append(cand[:self.n_neg])      # grouped PER GOLD SPAN
        return {**it, "negs": negs}


def collate(batch, pad_id):
    B = len(batch)
    Ls = max(len(b["input_ids"]) for b in batch)
    Lw = max(len(b["word_index"]) for b in batch)
    ids = torch.full((B, Ls), pad_id, dtype=torch.long)
    attn = torch.zeros((B, Ls), dtype=torch.long)
    widx = torch.zeros((B, Lw), dtype=torch.long)
    tags = torch.zeros((B, Lw), dtype=torch.long)
    wmask = torch.zeros((B, Lw), dtype=torch.bool)
    for i, b in enumerate(batch):
        n = len(b["input_ids"]); ids[i, :n] = torch.tensor(b["input_ids"]); attn[i, :n] = 1
        m = len(b["word_index"]); widx[i, :m] = torch.tensor(b["word_index"])
        tags[i, :m] = torch.tensor(b["tags"]); wmask[i, :m] = True
    return {"input_ids": ids, "attention_mask": attn, "word_index": widx, "tags": tags,
            "word_mask": wmask, "lengths": [len(b["word_index"]) for b in batch],
            "kept": [b["kept"] for b in batch], "n_words": [b["n_words"] for b in batch],
            "gold": [b["gold"] for b in batch], "pol": [b["pol"] for b in batch],
            "negs": [b.get("negs", []) for b in batch]}


@torch.no_grad()
def infer(model, loader, device, cand_thr):
    """Decode candidates from CRF marginals, score U, attach polarity. Returns per-sentence
    lists of (word_start, word_end, A, polarity_probs)."""
    model.eval()
    out = []
    for b in loader:
        with torch.autocast("cuda", dtype=torch.float16, enabled=device.type == "cuda"):
            hw = model.words(b["input_ids"].to(device), b["attention_mask"].to(device),
                             b["word_index"].to(device))
        emis = model.emissions(hw)
        marg = model.crf.marginals(emis, b["lengths"])
        for i, m in enumerate(marg):
            nonO = m[:, 1] + m[:, 2]
            tags = [(1 if m[t, 1] >= m[t, 2] else 2) if nonO[t] > cand_thr else 0
                    for t in range(len(m))] if cand_thr > 0 else m.argmax(-1).tolist()
            rows = spans_from_obi(tags)
            if not rows:
                out.append([]); continue
            lg = model.polarity(hw[i:i + 1], [(0, s, e) for (s, e) in rows])
            p = torch.softmax(lg, -1).cpu().numpy()
            kept = b["kept"][i]
            res = []
            for j, (s, e) in enumerate(rows):
                A = float(np.mean(1.0 - m[s:e, O_TAG]))
                res.append((kept[s], kept[e - 1] + 1, A, p[j]))
            out.append(res)
    return out


def assemble(preds, lam, tau):
    """U = A * conf^lam, thresholded. Geometric so it matches the §C.23 convention."""
    pairs, spans = [], []
    for sent in preds:
        ps, ss = [], []
        for (s, e, A, p) in sent:
            j = int(np.argmax(p))
            U = A * (float(p[j]) ** lam)
            if U > tau:
                ps.append((s, e, POLARITIES[j])); ss.append((s, e))
        pairs.append(ps); spans.append(ss)
    return pairs, spans


def a_selected(pairs, gold):
    """The §C.18 quantity, and this experiment's gate: polarity accuracy on SELECTED spans
    that are gold, versus on gold spans that were NOT selected."""
    sc = st = mc = mt = 0
    for p, g in zip(pairs, gold):
        gm = {(s, e): pol for (s, e, pol) in g}
        sel = {(s, e) for (s, e, _) in p}
        for (s, e, pol) in p:
            if (s, e) in gm:
                st += 1; sc += int(gm[(s, e)] == pol)
        for (s, e) in gm:
            if (s, e) not in sel:
                mt += 1
    return (100.0 * sc / st if st else 0.0), st, mt


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="microsoft/deberta-v3-large")
    ap.add_argument("--dataset", default="twitter2015")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--epochs", type=int, default=16)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--accum", type=int, default=2)
    ap.add_argument("--lr", type=float, default=1e-5)
    ap.add_argument("--head-lr", type=float, default=1e-4)
    ap.add_argument("--dropout", type=float, default=0.3)
    ap.add_argument("--max-len", type=int, default=160)
    ap.add_argument("--lam-asc", type=float, default=1.0, help="polarity CE weight")
    ap.add_argument("--lam-joint", type=float, default=0.5, help="joint margin weight")
    ap.add_argument("--margin", type=float, default=0.2)
    ap.add_argument("--lam-u", type=float, default=1.0,
                    help="exponent on polarity confidence inside U (0 = pure anchor score)")
    ap.add_argument("--cand-thr", type=float, default=0.12)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    set_seed(args.seed)
    random.seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    from transformers import AutoTokenizer, get_linear_schedule_with_warmup
    tok = AutoTokenizer.from_pretrained(args.model)
    insts = {s: load(args.dataset, s) for s in ("train", "dev", "test")}
    gp = {s: gold_pairs(v) for s, v in insts.items()}
    gs = {s: gold_spans(v) for s, v in insts.items()}
    coll = lambda b: collate(b, tok.pad_token_id)
    dl = {s: DataLoader(PacsDS(v, tok, args.max_len, train=(s == "train")),
                        batch_size=args.batch, shuffle=(s == "train"), collate_fn=coll)
          for s, v in insts.items()}

    model = PACS(args.model, args.dropout).to(device)
    body = [p for n, p in model.named_parameters() if n.startswith("encoder.")]
    head = [p for n, p in model.named_parameters() if not n.startswith("encoder.")]
    opt = torch.optim.AdamW([{"params": body, "lr": args.lr},
                             {"params": head, "lr": args.head_lr}], weight_decay=0.01)
    steps = (len(dl["train"]) // args.accum + 1) * args.epochs
    sched = get_linear_schedule_with_warmup(opt, int(0.1 * steps), steps)
    scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda")
    ce = nn.CrossEntropyLoss()

    best, best_ep, t0 = -1.0, -1, time.time()
    for ep in range(1, args.epochs + 1):
        model.train(); tot = tj = 0.0
        for i, b in enumerate(dl["train"]):
            with torch.autocast("cuda", dtype=torch.float16, enabled=device.type == "cuda"):
                hw = model.words(b["input_ids"].to(device), b["attention_mask"].to(device),
                                 b["word_index"].to(device))
            emis = model.emissions(hw)
            gsp = [(bi, s, e) for bi, g in enumerate(b["gold"]) for (s, e) in g]
            gpol = torch.tensor([p for pl in b["pol"] for p in pl], device=device)
            lg_g = model.polarity(hw, gsp)
            loss = model.crf.nll(emis, b["tags"].to(device), b["word_mask"].to(device))
            if len(gsp):
                loss = loss + args.lam_asc * ce(lg_g, gpol)
            # PAIRWISE hard-negative margin: every gold span against ITS OWN boundary errors.
            # The first version compared batch MEANS, relu(m + Un.mean() - Ug.mean()), which
            # is one scalar constraint per batch that ordinary training already satisfies --
            # the control arm logged that term at 0.0013 by epoch 8 and 0.0000 by epoch 12
            # WITHOUT optimising it. Weighting it would have added no gradient and returned a
            # null result caused by the loss rather than by the hypothesis.
            nsp, owner = [], []
            for bi, ns_per_gold in enumerate(b["negs"]):
                base = sum(len(g) for g in b["gold"][:bi])
                for gi, ns in enumerate(ns_per_gold):
                    for (s, e) in ns:
                        nsp.append((bi, s, e)); owner.append(base + gi)
            if len(gsp) and len(nsp):
                sm = torch.softmax(emis, -1)
                lg_n = model.polarity(hw, nsp)
                pg = torch.softmax(lg_g, -1).gather(1, gpol[:, None]).squeeze(1)
                Ug = torch.stack([span_score(sm, bi, s, e) for (bi, s, e) in gsp]) * pg
                Un = (torch.stack([span_score(sm, bi, s, e) for (bi, s, e) in nsp])
                      * torch.softmax(lg_n, -1).max(1).values)
                own = torch.tensor(owner, device=Un.device)
                lj = torch.relu(args.margin + Un - Ug[own]).mean()
                loss = loss + args.lam_joint * lj
                tj += float(lj)
            scaler.scale(loss / args.accum).backward()
            if (i + 1) % args.accum == 0:
                scaler.unscale_(opt)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                scaler.step(opt); scaler.update(); opt.zero_grad(set_to_none=True)
                sched.step()
            tot += float(loss)
        pr = infer(model, dl["dev"], device, args.cand_thr)
        bt, bf = 0.0, -1.0
        for tau in np.arange(0.0, 0.96, 0.01):
            pairs, _ = assemble(pr, args.lam_u, float(tau))
            f = score_joint(pairs, gp["dev"])["F1"]
            if f > bf:
                bf, bt = f, float(tau)
        print(f"ep{ep} loss {tot/len(dl['train']):.4f} (joint {tj/len(dl['train']):.4f}) "
              f"| dev joint {bf:.2f} @tau {bt:.2f} | {time.time()-t0:.0f}s", flush=True)
        if bf > best:
            best, best_ep, best_tau = bf, ep, bt
            torch.save(model.state_dict(), out / "best.pt")

    model.load_state_dict(torch.load(out / "best.pt", map_location=device))
    res = {"model": args.model, "seed": args.seed, "best_dev_joint": best,
           "best_epoch": best_ep, "tau": best_tau, "lam_u": args.lam_u,
           "lam_joint": args.lam_joint, "margin": args.margin}
    print("=" * 72)
    for s in ("dev", "test"):
        pr = infer(model, dl[s], device, args.cand_thr)
        pairs, spans = assemble(pr, args.lam_u, best_tau)
        j = score_joint(pairs, gp[s]); m = score_spans(spans, gs[s])
        a, nsel, nmiss = a_selected(pairs, gp[s])
        res[s] = {"joint": j, "MATE_at_tau": m, "a_selected": a,
                  "n_selected_gold": nsel, "n_missed_gold": nmiss}
        print(f"[{s}] MATE@tau {m['F1']:.2f}  a_selected {a:.2f}  "
              f"JOINT P {j['P']:.2f} R {j['R']:.2f} F1 {j['F1']:.2f}")
    print(f"\n  GATE (§C.18): a_selected on test = {res['test']['a_selected']:.2f}   "
          f"vs the decoupled pipeline's 80.52")
    json.dump(res, open(out / "metrics.json", "w"), indent=2, default=float)


if __name__ == "__main__":
    main()
