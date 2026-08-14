"""D27 — ASOE: Aspect-Signed Evidence Ownership.

Every visual experiment in this project asked **"is the image useful?"** and measured ~0
(Chapter B §7c, §C.7 gate +0.10, §D.9/D.12 teachers −0.2 to −0.9, §D.26 CET −0.20). ASOE asks
a different question:

    **which visual evidence is OWNED by this aspect, and does the decision actually need it?**

That was previously unaskable here. Every prior member compressed the image into ONE pooled
vector per aspect, so there was no set to route over. This uses the cached EVA-ViT-g patch
grid (`data/vit_cache`, 3502 images x 257 tokens x 1408 dims, full train coverage) and gives
each aspect an explicit ownership distribution over the 256 patches.

    o(a, i) = softmax_i( (W_q t_a) . (W_k v_i) / sqrt(d) )        E_a = sum_i o(a,i) W_v v_i

Three losses, each targeting a property no previous experiment supervised:

* **Sufficiency** (`--lam-suf`). The gold margin must be HIGHER with the routed evidence than
  without it: `M(lg_base + delta) > M(lg_base) + m`. This says *do not select evidence that
  merely looks relevant, select evidence the classifier actually needs.* Applied only where
  the §C.25 teacher says a shift exists (weighted by `1 - dirp[NONE]`), because forcing help
  on genuinely uninformative images would be wrong. **Nothing in Chapters C-D supervised
  this.**
* **Ownership** (`--lam-own`). For two aspects of the SAME tweet the image is identical, so a
  sibling's ownership distribution can be applied to the same patches for free -- no second
  encoder pass, no extra features. Evidence routed by the sibling must NOT move this aspect's
  decision. §D.26's CET tested this with a FIXED pooled vector; here the routing itself is
  learned, which is the part that was missing.
* **Separation** (`--lam-sep`). For sibling pairs with DIFFERENT gold polarity, the two
  ownership distributions must diverge. t2015 has 444 such pairs and they have never been
  used to supervise attention -- §C13 acted on final logits and §C10 on the input text.

The base is §C.27's proven formulation, unchanged: `lg_base` sees no visual input and is
supervised directly, `delta` is zero-initialised, `lg_full = lg_base + alpha * delta`. So
training starts exactly at a strong text classifier and evidence may only earn its way in.

Student-only inference: no teacher, no Qwen, nothing extra at test time.

    python experts/asoe.py --model vinai/bertweet-large --spans runs/mate_ens5_hr \
        --lam-suf 0.3 --lam-own 0.3 --lam-sep 0.1 --out runs/asoe_btwL
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
from experts.common import (DATA, POL2ID, POLARITIES, load, masc_examples,  # noqa: E402
                            masc_examples_for_spans, set_seed)

NEG, NEU, POS = 0, 1, 2


class PatchCache:
    """Memmapped EVA-ViT-g patch grid. 0.72 MB per image in fp16, so batches are cheap."""

    def __init__(self, dataset):
        d = DATA / "vit_cache" / dataset
        meta = json.load(open(d / "index.json"))
        self.N, self.T, self.D = meta["shape"]
        self.index = meta["index"]
        self.mm = np.memmap(d / "feats.f16", dtype=np.float16, mode="r",
                            shape=(self.N, self.T, self.D))

    def get(self, image_id):
        r = self.index.get(image_id)
        if r is None:
            return np.zeros((self.T - 1, self.D), dtype=np.float16)
        return self.mm[r, 1:]          # drop CLS; the patches are the evidence set


class DS(Dataset):
    def __init__(self, ex, tok, cache, dirp, max_len=160):
        self.ex, self.tok, self.cache, self.dirp = ex, tok, cache, dirp
        self.max_len = max_len

    def __len__(self):
        return len(self.ex)

    def __getitem__(self, i):
        e = self.ex[i]
        enc = self.tok(e.term, e.marked_text(), truncation="only_second",
                       max_length=self.max_len)
        tl = len(self.tok(e.term, add_special_tokens=False)["input_ids"])
        return {"ids": enc["input_ids"], "term_len": tl, "y": POL2ID[e.polarity],
                "key": e.key, "inst": e.inst_idx, "img": e.image_id,
                "dir": self.dirp.get(e.key, np.array([0., 0., 1.], dtype=np.float32))}


def make_collate(pad_id, cache):
    def collate(b):
        L = max(len(x["ids"]) for x in b)
        ids = torch.full((len(b), L), pad_id, dtype=torch.long)
        m = torch.zeros((len(b), L), dtype=torch.long)
        tm = torch.zeros((len(b), L), dtype=torch.float)
        for i, x in enumerate(b):
            n = len(x["ids"]); ids[i, :n] = torch.tensor(x["ids"]); m[i, :n] = 1
            tm[i, 1:1 + min(x["term_len"], L - 1)] = 1.0
        V = np.stack([cache.get(x["img"]) for x in b]).astype(np.float32)
        return {"ids": ids, "mask": m, "term_mask": tm,
                "patches": torch.from_numpy(V),
                "y": torch.tensor([x["y"] for x in b]),
                "inst": torch.tensor([x["inst"] for x in b]),
                "dir": torch.tensor(np.stack([x["dir"] for x in b])),
                "key": [x["key"] for x in b]}
    return collate


class InstanceBatchSampler(torch.utils.data.Sampler):
    """Keep all aspects of one tweet together so the ownership and separation losses have
    sibling pairs to work with; random sampling would co-batch them only by accident."""

    def __init__(self, ex, batch_size, shuffle=True, seed=0):
        g = {}
        for i, e in enumerate(ex):
            g.setdefault(e.inst_idx, []).append(i)
        self.groups = list(g.values())
        self.bs, self.shuffle, self.seed, self.epoch = batch_size, shuffle, seed, 0

    def __iter__(self):
        order = list(range(len(self.groups)))
        if self.shuffle:
            random.Random(self.seed + self.epoch).shuffle(order); self.epoch += 1
        batch = []
        for gi in order:
            grp = self.groups[gi]
            if batch and len(batch) + len(grp) > self.bs:
                yield batch; batch = []
            batch += grp
        if batch:
            yield batch

    def __len__(self):
        return max(1, sum(len(g) for g in self.groups) // self.bs)


class ASOE(nn.Module):
    def __init__(self, model_id, patch_dim=1408, dropout=0.1):
        super().__init__()
        from transformers import AutoConfig, AutoModel
        cfg = AutoConfig.from_pretrained(model_id)
        self.enc = AutoModel.from_pretrained(model_id, dtype=torch.float32)
        h = cfg.hidden_size
        self.drop = nn.Dropout(dropout)
        self.k_proj = nn.Linear(patch_dim, h)      # patch -> key
        self.v_proj = nn.Linear(patch_dim, h)      # patch -> value
        self.q_proj = nn.Linear(h, h)              # aspect -> query
        self.scale = h ** -0.5
        self.head_base = nn.Linear(2 * h, 3)
        self.head_delta = nn.Sequential(nn.Linear(2 * h, h), nn.GELU(), nn.Linear(h, 3))
        nn.init.zeros_(self.head_delta[-1].weight)     # §C.27: start at the text classifier
        nn.init.zeros_(self.head_delta[-1].bias)
        self.alpha = nn.Parameter(torch.tensor(0.1))
        # TTP: aspect-conditioned contrastive projections. §D.27 measured that evidence
        # mechanisms have NEGATIVE marginal value above a baseline of ~77.93, and our towers
        # sit above it -- so the intervention has to move the BASE representation instead of
        # adding another mechanism on top of it. Stage 1 aligns the aspect text with its
        # OWN routed image evidence E_a, using no polarity labels. Ordinary tweet-image
        # alignment cannot express this: two aspects of one tweet share an image and would be
        # positives for each other. Because E_a is aspect-routed and instance batching puts
        # siblings in the same batch, they become HARD NEGATIVES and the routing is forced to
        # be aspect-discriminative.
        self.ct = nn.Linear(h, 256)
        self.ci = nn.Linear(h, 256)
        self.logit_scale = nn.Parameter(torch.tensor(np.log(1 / 0.07), dtype=torch.float32))

    def text(self, ids, mask, term_mask):
        hs = self.enc(input_ids=ids, attention_mask=mask).last_hidden_state
        m = mask.unsqueeze(-1).float(); tmk = term_mask.unsqueeze(-1)
        t_a = (hs * tmk).sum(1) / tmk.sum(1).clamp(min=1)
        ctx = (hs * m).sum(1) / m.sum(1).clamp(min=1)
        return t_a, ctx

    def ownership(self, t_a, patches):
        """o(a, i) over the 256 patches, plus the routed evidence E_a."""
        k = self.k_proj(patches)                                     # [B, P, h]
        o = (k @ self.q_proj(t_a).unsqueeze(-1)).squeeze(-1) * self.scale
        o = o.softmax(-1)
        E = (o.unsqueeze(-1) * self.v_proj(patches)).sum(1)
        return o, E

    def decide(self, t_a, ctx, E):
        """Returns (lg_base, gated delta, RAW delta).

        The raw delta matters: the ownership loss must constrain the ROUTING, not the gate.
        Penalising the gated `alpha * delta_sib` is satisfiable by driving alpha -> 0, i.e.
        the model obeys "a sibling's evidence must not move my decision" by ignoring ALL
        evidence -- and a 1-epoch smoke run did exactly that, alpha 0.100 -> 0.003. That
        degenerate fixed point also fights the sufficiency term, which pushes alpha up.
        """
        lg_base = self.head_base(self.drop(torch.cat([t_a, ctx], -1))).float()
        delta = self.head_delta(self.drop(torch.cat([E, t_a], -1))).float()
        return lg_base, self.alpha * delta, delta

    def forward(self, ids, mask, term_mask, patches):
        t_a, ctx = self.text(ids, mask, term_mask)
        o, E = self.ownership(t_a, patches)
        lg_base, d, _ = self.decide(t_a, ctx, E)
        return lg_base + d, lg_base, d, o, t_a, ctx


def ttp_loss(model, t_a, E):
    """Symmetric InfoNCE between the aspect representation and its routed image evidence."""
    a = nn.functional.normalize(model.ct(t_a).float(), dim=-1)
    b = nn.functional.normalize(model.ci(E).float(), dim=-1)
    logits = model.logit_scale.exp().clamp(max=100.0) * a @ b.t()
    tgt = torch.arange(len(a), device=a.device)
    return 0.5 * (nn.functional.cross_entropy(logits, tgt)
                  + nn.functional.cross_entropy(logits.t(), tgt))


def gold_margin(lg, y):
    """z_gold - max_{other} z. Positive means the gold class is winning."""
    g = lg.gather(1, y[:, None]).squeeze(1)
    other = lg.masked_fill(nn.functional.one_hot(y, 3).bool(), float("-inf")).max(1).values
    return g - other


@torch.no_grad()
def run(model, loader, device):
    model.eval(); P, Y, K = [], [], []
    for b in loader:
        with torch.autocast("cuda", dtype=torch.float16, enabled=device.type == "cuda"):
            mo = model(b["ids"].to(device), b["mask"].to(device),
                       b["term_mask"].to(device), b["patches"].to(device))
        P.append(torch.softmax(mo[0].float(), -1).cpu().numpy())
        Y.extend(b["y"].tolist()); K.extend(b["key"])
    P = np.concatenate(P)
    return 100.0 * float((P.argmax(1) == np.array(Y)).mean()) if Y else 0.0, P, K


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="vinai/bertweet-large")
    ap.add_argument("--dataset", default="twitter2015")
    ap.add_argument("--pds", default="data/pds/twitter2015")
    ap.add_argument("--spans", default=None)
    ap.add_argument("--seed", type=int, default=70)
    ap.add_argument("--epochs", type=int, default=8)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--lr", type=float, default=1e-5)
    ap.add_argument("--head-lr", type=float, default=1e-3)
    ap.add_argument("--patience", type=int, default=3)
    ap.add_argument("--lam-suf", type=float, default=0.3)
    ap.add_argument("--lam-own", type=float, default=0.3)
    ap.add_argument("--lam-sep", type=float, default=0.1)
    ap.add_argument("--margin", type=float, default=0.2)
    ap.add_argument("--ttp-epochs", type=int, default=0,
                    help="TTP stage 1: aspect-conditioned contrastive pretraining epochs "
                         "BEFORE any polarity label is used")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    set_seed(args.seed); random.seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    from transformers import AutoTokenizer, get_linear_schedule_with_warmup
    tok = AutoTokenizer.from_pretrained(args.model)
    cache = PatchCache(args.dataset)
    insts = {s: load(args.dataset, s) for s in ("train", "dev", "test")}
    ex = {s: masc_examples(v) for s, v in insts.items()}
    z = np.load(Path(args.pds) / "direction_train.npz")
    dirp = {(int(k[0]), int(k[1]), int(k[2])): p.astype(np.float32)
            for p, k in zip(z["probs"], z["keys"])}
    coll = make_collate(tok.pad_token_id, cache)
    dl = {}
    for s in insts:
        ds = DS(ex[s], tok, cache, dirp if s == "train" else {})
        dl[s] = (DataLoader(ds, batch_sampler=InstanceBatchSampler(ex[s], args.batch,
                                                                   seed=args.seed),
                            collate_fn=coll) if s == "train"
                 else DataLoader(ds, batch_size=args.batch, collate_fn=coll))

    model = ASOE(args.model, cache.D).to(device)
    body = [p for n, p in model.named_parameters() if n.startswith("enc.")]
    head = [p for n, p in model.named_parameters() if not n.startswith("enc.")]
    opt = torch.optim.AdamW([{"params": body, "lr": args.lr},
                             {"params": head, "lr": args.head_lr}], weight_decay=0.01)
    steps = len(dl["train"]) * args.epochs
    sched = get_linear_schedule_with_warmup(opt, int(0.1 * steps), steps)
    scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda")
    ce = nn.CrossEntropyLoss()

    t0 = time.time()
    if args.ttp_epochs > 0:
        # STAGE 1 -- no polarity labels are touched here.
        popt = torch.optim.AdamW(
            [{"params": body, "lr": args.lr}, {"params": head, "lr": args.head_lr}],
            weight_decay=0.01)
        for ep in range(1, args.ttp_epochs + 1):
            model.train(); tot = 0.0
            for b in dl["train"]:
                popt.zero_grad(set_to_none=True)
                with torch.autocast("cuda", dtype=torch.float16,
                                    enabled=device.type == "cuda"):
                    t_a, _ = model.text(b["ids"].to(device), b["mask"].to(device),
                                        b["term_mask"].to(device))
                _, E = model.ownership(t_a, b["patches"].to(device))
                lo = ttp_loss(model, t_a, E)
                scaler.scale(lo).backward(); scaler.unscale_(popt)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                scaler.step(popt); scaler.update()
                tot += float(lo)
            print(f"[TTP] ep{ep} contrastive {tot/len(dl['train']):.4f} | "
                  f"{time.time()-t0:.0f}s", flush=True)
        # the polarity heads must start fresh -- stage 1 never saw a label
        nn.init.zeros_(model.head_delta[-1].weight); nn.init.zeros_(model.head_delta[-1].bias)

    best, best_ep, bad = -1.0, -1, 0
    for ep in range(1, args.epochs + 1):
        model.train(); tot = ts = to = tp = 0.0
        for b in dl["train"]:
            opt.zero_grad(set_to_none=True)
            ids, mask, tmk = (b["ids"].to(device), b["mask"].to(device),
                              b["term_mask"].to(device))
            V = b["patches"].to(device); y = b["y"].to(device)
            d_teacher = b["dir"].to(device)
            with torch.autocast("cuda", dtype=torch.float16, enabled=device.type == "cuda"):
                t_a, ctx = model.text(ids, mask, tmk)
            o, E = model.ownership(t_a, V)
            lg_base, d, _ = model.decide(t_a, ctx, E)
            lg_full = lg_base + d
            loss = ce(lg_full, y) + 0.5 * ce(lg_base, y)

            # SUFFICIENCY: the routed evidence must RAISE the gold margin, but only where the
            # teacher says a shift exists -- forcing help on an uninformative image is wrong.
            if args.lam_suf > 0:
                rel = (1.0 - d_teacher[:, 2])
                gap = gold_margin(lg_full, y) - gold_margin(lg_base, y)
                ls = (rel * torch.relu(args.margin - gap)).sum() / rel.sum().clamp(min=1e-6)
                loss = loss + args.lam_suf * ls; ts += float(ls)

            # OWNERSHIP / SEPARATION over sibling aspects. Siblings share the SAME image, so
            # a sibling's ownership distribution applies to the same patches for free.
            inst = b["inst"]
            pairs = [(i, j) for i in range(len(inst)) for j in range(len(inst))
                     if i != j and inst[i] == inst[j]]
            if pairs and (args.lam_own > 0 or args.lam_sep > 0):
                I = torch.tensor([p[0] for p in pairs], device=device)
                J = torch.tensor([p[1] for p in pairs], device=device)
                if args.lam_own > 0:
                    # evidence routed by the SIBLING must not move this aspect's decision
                    E_sib = (o[J].unsqueeze(-1) * model.v_proj(V[I])).sum(1)
                    # RAW delta, not alpha*delta -- see decide().
                    _, _, d_sib = model.decide(t_a[I], ctx[I], E_sib)
                    off = ((d_sib[:, POS] - d_sib[:, NEU]).abs()
                           + (d_sib[:, NEG] - d_sib[:, NEU]).abs()).mean()
                    loss = loss + args.lam_own * off; to += float(off)
                if args.lam_sep > 0:
                    diff = y[I] != y[J]        # the 444 different-polarity sibling pairs
                    if diff.any():
                        oi, oj = o[I][diff], o[J][diff]
                        mid = 0.5 * (oi + oj)
                        js = 0.5 * ((oi * (oi.clamp(min=1e-9) / mid.clamp(min=1e-9)).log()).sum(-1)
                                    + (oj * (oj.clamp(min=1e-9) / mid.clamp(min=1e-9)).log()).sum(-1))
                        lp = torch.relu(args.margin - js).mean()
                        loss = loss + args.lam_sep * lp; tp += float(lp)

            scaler.scale(loss).backward(); scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(opt); scaler.update(); sched.step()
            tot += float(loss)
        n = len(dl["train"])
        acc, *_ = run(model, dl["dev"], device)
        print(f"ep{ep} loss {tot/n:.4f} (suf {ts/n:.4f} own {to/n:.4f} sep {tp/n:.4f}) "
              f"| dev acc {acc:.2f} | alpha {float(model.alpha):+.3f} | "
              f"{time.time()-t0:.0f}s", flush=True)
        if acc > best:
            best, best_ep, bad = acc, ep, 0
            torch.save(model.state_dict(), out / "best.pt")
        else:
            bad += 1
            if bad >= args.patience:
                print("early stop"); break

    model.load_state_dict(torch.load(out / "best.pt", map_location=device))
    res = {"model": args.model, "seed": args.seed, "best_dev_acc": best,
           "ttp_epochs": args.ttp_epochs,
           "best_epoch": best_ep, "lam_suf": args.lam_suf, "lam_own": args.lam_own,
           "lam_sep": args.lam_sep}
    for s in ("dev", "test"):
        acc, P, K = run(model, dl[s], device)
        res[f"{s}_acc_goldspan"] = acc
        np.savez_compressed(out / f"probs_{s}.npz", probs=P,
                            keys=np.array(K, dtype=np.int64))
        print(f"[{s}] gold-span MASC acc {acc:.2f}", flush=True)
    if args.spans:
        for s in ("dev", "test"):
            sp = json.load(open(Path(args.spans) / f"spans_{s}.json"))
            e2 = masc_examples_for_spans(insts[s], [[tuple(x) for x in i] for i in sp])
            dl2 = DataLoader(DS(e2, tok, cache, {}), batch_size=args.batch, collate_fn=coll)
            _, P2, K2 = run(model, dl2, device)
            np.savez_compressed(out / f"probs_span_{s}.npz", probs=P2,
                                keys=np.array(K2, dtype=np.int64))
            print(f"[{s}] scored {len(e2)} predicted spans", flush=True)
    json.dump(res, open(out / "metrics.json", "w"), indent=2)


if __name__ == "__main__":
    main()
