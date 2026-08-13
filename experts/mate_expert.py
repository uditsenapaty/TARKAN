"""C1 / B1 — candidate aspect-ANCHOR generator = MATE stage-1 expert.

This is the module the *new* TARKAN manuscript calls the "candidate aspect-anchor
generator" (B-ASP / I-ASP / O, supervised by `L_anc`), and which Chapter B had already
found empirically as patch **B1**:

  * a DEDICATED aspect-only O/B/I head instead of the unified 7-tag BIO head, and
  * a word-level linear-chain CRF over first-subtoken emissions.

Why it matters (measured, Chapter B): the 7-tag head can only emit a span when its
polarity subtags agree, so a correctly-bounded span with uncertain polarity decodes to
nothing -- P(O)=0.35 beats each of P(B-POS)=0.25 / P(B-NEU)=0.22 / P(B-NEG)=0.18 even
though P(B-*)=0.65. Removing polarity from the extraction head removes the leak at the
training level; the gain lands in recall, exactly as predicted.

Emits per-word tag MARGINALS (forward-backward), not just decoded spans, because
`assemble.py` ensembles MATE members by averaging marginals rather than voting on
decoded spans (Chapter B, B3).

    python experts/mate_expert.py --seed 42 --out runs/mate_deb_s42
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import List, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from experts.common import (OBI, OBI2ID, gold_spans, load, obi_tags,  # noqa: E402
                            score_spans, set_seed, spans_from_obi)

NTAG = 3


# --------------------------------------------------------------------------- #
# linear-chain CRF (fp32; vendored to avoid a torchcrf dependency)
# --------------------------------------------------------------------------- #
class CRF(nn.Module):
    def __init__(self, n_tags: int = NTAG):
        super().__init__()
        self.n = n_tags
        self.trans = nn.Parameter(torch.zeros(n_tags, n_tags))  # [from, to]
        self.start = nn.Parameter(torch.zeros(n_tags))
        self.end = nn.Parameter(torch.zeros(n_tags))

    def _log_Z(self, emis: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        B, L, T = emis.shape
        alpha = self.start.unsqueeze(0) + emis[:, 0]
        for t in range(1, L):
            s = alpha.unsqueeze(2) + self.trans.unsqueeze(0) + emis[:, t].unsqueeze(1)
            nxt = torch.logsumexp(s, dim=1)
            alpha = torch.where(mask[:, t].unsqueeze(1), nxt, alpha)
        return torch.logsumexp(alpha + self.end.unsqueeze(0), dim=1)

    def _gold_score(self, emis, tags, mask) -> torch.Tensor:
        B, L, T = emis.shape
        score = self.start[tags[:, 0]] + emis[:, 0].gather(1, tags[:, :1]).squeeze(1)
        for t in range(1, L):
            tr = self.trans[tags[:, t - 1], tags[:, t]]
            em = emis[:, t].gather(1, tags[:, t:t + 1]).squeeze(1)
            score = score + (tr + em) * mask[:, t].float()
        last = mask.sum(1) - 1
        score = score + self.end[tags.gather(1, last.unsqueeze(1)).squeeze(1)]
        return score

    def nll(self, emis, tags, mask) -> torch.Tensor:
        return (self._log_Z(emis, mask) - self._gold_score(emis, tags, mask)).mean()

    @torch.no_grad()
    def viterbi(self, emis: torch.Tensor, lengths: Sequence[int]) -> List[List[int]]:
        out = []
        for b, L in enumerate(lengths):
            e = emis[b, :L]
            delta = self.start + e[0]
            back = []
            for t in range(1, L):
                s = delta.unsqueeze(1) + self.trans + e[t].unsqueeze(0)
                best, idx = s.max(dim=0)
                back.append(idx)
                delta = best
            delta = delta + self.end
            path = [int(delta.argmax())]
            for idx in reversed(back):
                path.append(int(idx[path[-1]]))
            out.append(path[::-1])
        return out

    @torch.no_grad()
    def marginals(self, emis: torch.Tensor, lengths: Sequence[int]) -> List[np.ndarray]:
        """Per-word posterior P(tag_t | x) by forward-backward. One array per sequence."""
        out = []
        for b, L in enumerate(lengths):
            e = emis[b, :L]
            alpha = torch.empty(L, self.n, device=e.device)
            alpha[0] = self.start + e[0]
            for t in range(1, L):
                alpha[t] = torch.logsumexp(alpha[t - 1].unsqueeze(1) + self.trans, dim=0) + e[t]
            beta = torch.empty(L, self.n, device=e.device)
            beta[L - 1] = self.end
            for t in range(L - 2, -1, -1):
                beta[t] = torch.logsumexp(self.trans + (e[t + 1] + beta[t + 1]).unsqueeze(0), dim=1)
            logZ = torch.logsumexp(alpha[L - 1] + self.end, dim=0)
            out.append(torch.softmax(alpha + beta - logZ, dim=-1).float().cpu().numpy())
        return out


# --------------------------------------------------------------------------- #
# model
# --------------------------------------------------------------------------- #
class AnchorTagger(nn.Module):
    def __init__(self, model_id: str, dropout: float = 0.1):
        super().__init__()
        from transformers import AutoConfig, AutoModel
        cfg = AutoConfig.from_pretrained(model_id)
        # transformers>=4.56 honours the checkpoint's stored dtype, and deberta-v3-large
        # ships fp16 weights -> AMP would then try to unscale fp16 grads. Force fp32
        # master weights; autocast still runs the forward in fp16.
        self.encoder = AutoModel.from_pretrained(model_id, dtype=torch.float32)
        self.drop = nn.Dropout(dropout)
        self.proj = nn.Linear(cfg.hidden_size, NTAG)
        self.crf = CRF(NTAG)

    def emissions(self, input_ids, attention_mask, word_index, word_mask):
        h = self.encoder(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state
        # gather the FIRST sub-token of each word
        idx = word_index.unsqueeze(-1).expand(-1, -1, h.size(-1))
        hw = h.gather(1, idx)
        return self.proj(self.drop(hw)).float()  # CRF always in fp32


# --------------------------------------------------------------------------- #
# data
# --------------------------------------------------------------------------- #
class AnchorDS(Dataset):
    def __init__(self, insts, tok, max_len: int = 160):
        self.items = []
        for inst in insts:
            enc = tok(inst.tokens, is_split_into_words=True, truncation=True,
                      max_length=max_len)
            wids = enc.word_ids()
            first = {}
            for pos, w in enumerate(wids):
                if w is not None and w not in first:
                    first[w] = pos
            nw = len(inst.tokens)
            keep = [w for w in range(nw) if w in first]
            self.items.append({
                "input_ids": enc["input_ids"],
                "word_index": [first[w] for w in keep],
                "tags": [obi_tags(inst)[w] for w in keep],
                "n_words": nw,
                "kept": keep,
            })

    def __len__(self):
        return len(self.items)

    def __getitem__(self, i):
        return self.items[i]


def collate(batch, pad_id: int):
    B = len(batch)
    Ls = max(len(b["input_ids"]) for b in batch)
    Lw = max(len(b["word_index"]) for b in batch)
    input_ids = torch.full((B, Ls), pad_id, dtype=torch.long)
    attn = torch.zeros((B, Ls), dtype=torch.long)
    widx = torch.zeros((B, Lw), dtype=torch.long)
    tags = torch.zeros((B, Lw), dtype=torch.long)
    wmask = torch.zeros((B, Lw), dtype=torch.bool)
    for i, b in enumerate(batch):
        n = len(b["input_ids"])
        input_ids[i, :n] = torch.tensor(b["input_ids"])
        attn[i, :n] = 1
        m = len(b["word_index"])
        widx[i, :m] = torch.tensor(b["word_index"])
        tags[i, :m] = torch.tensor(b["tags"])
        wmask[i, :m] = True
    return {"input_ids": input_ids, "attention_mask": attn, "word_index": widx,
            "tags": tags, "word_mask": wmask,
            "lengths": [len(b["word_index"]) for b in batch],
            "kept": [b["kept"] for b in batch],
            "n_words": [b["n_words"] for b in batch]}


# --------------------------------------------------------------------------- #
# eval
# --------------------------------------------------------------------------- #
@torch.no_grad()
def predict(model, loader, device, want_marginals: bool = True):
    model.eval()
    all_spans, all_marg = [], []
    for batch in loader:
        ids = batch["input_ids"].to(device)
        attn = batch["attention_mask"].to(device)
        widx = batch["word_index"].to(device)
        wmask = batch["word_mask"].to(device)
        with torch.autocast("cuda", dtype=torch.float16, enabled=device.type == "cuda"):
            emis = model.emissions(ids, attn, widx, wmask)
        emis = emis.float()
        paths = model.crf.viterbi(emis, batch["lengths"])
        margs = model.crf.marginals(emis, batch["lengths"]) if want_marginals else None
        for i, path in enumerate(paths):
            nw, kept = batch["n_words"][i], batch["kept"][i]
            full = [OBI2ID["O"]] * nw
            for k, w in enumerate(kept):
                full[w] = path[k]
            all_spans.append(spans_from_obi(full))
            if want_marginals:
                m = np.zeros((nw, NTAG), dtype=np.float32)
                m[:, OBI2ID["O"]] = 1.0
                for k, w in enumerate(kept):
                    m[w] = margs[i][k]
                all_marg.append(m)
    return all_spans, all_marg


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="microsoft/deberta-v3-large")
    ap.add_argument("--dataset", default="twitter2015")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--epochs", type=int, default=10)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--accum", type=int, default=2)
    ap.add_argument("--lr", type=float, default=2e-5)
    ap.add_argument("--head-lr", type=float, default=1e-3)
    ap.add_argument("--warmup", type=float, default=0.1)
    ap.add_argument("--max-len", type=int, default=160)
    ap.add_argument("--patience", type=int, default=4)
    ap.add_argument("--dropout", type=float, default=0.1)
    ap.add_argument("--fold", type=int, default=None,
                    help="out-of-fold mode: train on sentences with idx %% nfolds != fold "
                         "and emit marginals for the held-out fold. Two runs give honest "
                         "TRAIN candidate spans (with a realistic false-positive rate) for "
                         "the span reranker -- predictions on the training set itself are "
                         "memorised and contain almost no FPs, so they are useless for it.")
    ap.add_argument("--nfolds", type=int, default=2)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    from transformers import AutoTokenizer, get_linear_schedule_with_warmup
    tok = AutoTokenizer.from_pretrained(args.model)
    splits = {s: load(args.dataset, s) for s in ("train", "dev", "test")}
    oof_insts = None
    if args.fold is not None:
        oof_insts = [x for i, x in enumerate(splits["train"]) if i % args.nfolds == args.fold]
        oof_idx = [i for i in range(len(splits["train"])) if i % args.nfolds == args.fold]
        splits["train"] = [x for i, x in enumerate(splits["train"])
                           if i % args.nfolds != args.fold]
        print(f"fold {args.fold}/{args.nfolds}: train {len(splits['train'])} "
              f"held-out {len(oof_insts)}", flush=True)
    ds = {s: AnchorDS(v, tok, args.max_len) for s, v in splits.items()}
    pad = tok.pad_token_id
    dl = {s: DataLoader(v, batch_size=args.batch, shuffle=(s == "train"),
                        collate_fn=lambda b: collate(b, pad))
          for s, v in ds.items()}

    model = AnchorTagger(args.model, dropout=args.dropout).to(device)
    head = [p for n, p in model.named_parameters() if not n.startswith("encoder.")]
    enc = [p for n, p in model.named_parameters() if n.startswith("encoder.")]
    opt = torch.optim.AdamW([{"params": enc, "lr": args.lr},
                             {"params": head, "lr": args.head_lr}], weight_decay=0.01)
    steps = (len(dl["train"]) // args.accum + 1) * args.epochs
    sched = get_linear_schedule_with_warmup(opt, int(args.warmup * steps), steps)
    scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda")

    gold = {s: gold_spans(v) for s, v in splits.items()}
    best_dev, best_ep, bad = -1.0, -1, 0
    t0 = time.time()
    for ep in range(1, args.epochs + 1):
        model.train()
        tot = 0.0
        opt.zero_grad(set_to_none=True)
        for i, batch in enumerate(dl["train"]):
            ids = batch["input_ids"].to(device)
            attn = batch["attention_mask"].to(device)
            widx = batch["word_index"].to(device)
            wmask = batch["word_mask"].to(device)
            tags = batch["tags"].to(device)
            with torch.autocast("cuda", dtype=torch.float16, enabled=device.type == "cuda"):
                emis = model.emissions(ids, attn, widx, wmask)
            loss = model.crf.nll(emis.float(), tags, wmask) / args.accum
            scaler.scale(loss).backward()
            tot += loss.item() * args.accum
            if (i + 1) % args.accum == 0:
                scaler.unscale_(opt)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                scaler.step(opt)
                scaler.update()
                opt.zero_grad(set_to_none=True)
                sched.step()

        dev_spans, _ = predict(model, dl["dev"], device, want_marginals=False)
        d = score_spans(dev_spans, gold["dev"])
        print(f"ep{ep} loss {tot/len(dl['train']):.4f} | dev MATE "
              f"P {d['P']:.2f} R {d['R']:.2f} F1 {d['F1']:.2f} | {time.time()-t0:.0f}s", flush=True)
        if d["F1"] > best_dev:
            best_dev, best_ep, bad = d["F1"], ep, 0
            torch.save(model.state_dict(), out / "best.pt")
        else:
            bad += 1
            if bad >= args.patience:
                print(f"early stop (patience {args.patience})")
                break

    model.load_state_dict(torch.load(out / "best.pt", map_location=device))
    res = {"model": args.model, "seed": args.seed, "best_dev_F1": best_dev, "best_epoch": best_ep}
    if oof_insts is not None:
        oof_dl = DataLoader(AnchorDS(oof_insts, tok, args.max_len),
                            batch_size=args.batch, collate_fn=lambda b: collate(b, pad))
        _, margs = predict(model, oof_dl, device, want_marginals=True)
        np.savez_compressed(out / f"marginals_oof{args.fold}.npz",
                            **{str(i): m for i, m in enumerate(margs)})
        json.dump(oof_idx, open(out / f"oofidx{args.fold}.json", "w"))
        sc = score_spans([spans_from_obi(m.argmax(-1).tolist()) for m in margs],
                         gold_spans(oof_insts))
        res[f"oof{args.fold}"] = sc
        print(f"[oof{args.fold}] MATE P {sc['P']:.2f} R {sc['R']:.2f} F1 {sc['F1']:.2f}",
              flush=True)
    for s in ("dev", "test"):
        spans, margs = predict(model, dl[s], device, want_marginals=True)
        sc = score_spans(spans, gold[s])
        res[s] = sc
        np.savez_compressed(out / f"marginals_{s}.npz",
                            **{str(i): m for i, m in enumerate(margs)})
        json.dump([[list(x) for x in sp] for sp in spans], open(out / f"spans_{s}.json", "w"))
        print(f"[{s}] MATE P {sc['P']:.2f} R {sc['R']:.2f} F1 {sc['F1']:.2f}", flush=True)
    json.dump(res, open(out / "metrics.json", "w"), indent=2)
    print(json.dumps(res, indent=2))


if __name__ == "__main__":
    main()
