"""C7 — MADSC's MASC path: AADG description + confidence-calibrated modality gate.

MADSC's ablation is unambiguous about which of its parts carries the polarity gain on
t2015: removing the modality gate costs **-5.74 MABSA Mac-F1** (78.38 -> 72.64) and removing
confidence calibration costs -2.13, while downgrading the captioner costs only -1.41. So the
gate, not the description, is the load-bearing piece -- and it is ~4 learnable scalars.

Faithful to MADSC Eqs. 9, 13, 14, 16, 19:

    u_a  = sigma(w_u * sim_final(a, o*) + b_u)                (Eq. 9   calibrator)
    g_a  = sigma(W_g * u_a + b_g)                             (Eq. 13  gate)
    z_a  = g_a * v_a + (1 - g_a) * t_a                        (Eq. 14  convex fusion)
    P(s) = softmax(W * z_a + b)                               (Eq. 16)
    L    = CE(polarity) + lambda_conf * BCE(u_a, y_pseudo)    (Eqs. 18-19)

where `t_a` is the pooled aspect token representation, `v_a` the CLIP embedding of the
region that best matches the aspect, and `y_pseudo` the deterministic alignment label
(exact string match after normalisation, or CLIP text-text cosine >= 0.85).

Note the mechanism this shares with TARKAN: `g_a` is exactly a teacher-free version of
TARKAN's aspect-visual relevance gate r^T. Chapter B measured TARKAN's gate at ~0 on a
from-scratch CLIP+DeBERTa trunk (§7c, 75.41 vs 78.5 for a plain head). The difference here
is that the gate is driven by a CALIBRATED external alignment score rather than learned
from scratch on 3.2k aspects -- which is the part Chapter B never had.

    python experts/masc_gated.py --model vinai/bertweet-large --seed 50 \
        --desc data/aadg/twitter2015 --spans runs/mate_ens5 --out runs/masc_gated_s50
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from experts.common import (POL2ID, POLARITIES, load, masc_examples,  # noqa: E402
                            masc_examples_for_spans, set_seed)

Key = Tuple[int, int, int]


def load_aspect_npz(path: Path):
    z = np.load(path)
    vis: Dict[Key, np.ndarray] = {}
    u: Dict[Key, float] = {}
    y: Dict[Key, float] = {}
    for k, v, uu, yy in zip(z["keys"], z["vis"], z["u"], z["y"]):
        key = (int(k[0]), int(k[1]), int(k[2]))
        vis[key], u[key], y[key] = v, float(uu), float(yy)
    return vis, u, y


class DS(Dataset):
    def __init__(self, examples, tok, desc, vis, u, y, vis_dim, max_len=192):
        self.ex, self.tok, self.desc = examples, tok, desc
        self.vis, self.u, self.y, self.vis_dim = vis, u, y, vis_dim
        self.max_len = max_len

    def __len__(self):
        return len(self.ex)

    def __getitem__(self, i):
        e = self.ex[i]
        second = e.marked_text()
        if self.desc is not None:
            second = second + " " + self.tok.sep_token + " " + self.desc[e.inst_idx]
        enc = self.tok(e.term, second, truncation="only_second", max_length=self.max_len)
        term_len = len(self.tok(e.term, add_special_tokens=False)["input_ids"])
        return {"ids": enc["input_ids"], "term_len": term_len,
                "y": POL2ID[e.polarity], "key": e.key,
                "vis": self.vis.get(e.key, np.zeros(self.vis_dim, dtype=np.float32)),
                "u": self.u.get(e.key, 0.0), "yc": self.y.get(e.key, 0.0)}


def make_collate(pad_id, vis_dim):
    def collate(b):
        L = max(len(x["ids"]) for x in b)
        ids = torch.full((len(b), L), pad_id, dtype=torch.long)
        m = torch.zeros((len(b), L), dtype=torch.long)
        tm = torch.zeros((len(b), L), dtype=torch.float)
        for i, x in enumerate(b):
            n = len(x["ids"])
            ids[i, :n] = torch.tensor(x["ids"])
            m[i, :n] = 1
            tm[i, 1:1 + min(x["term_len"], L - 1)] = 1.0    # aspect tokens = segment A
        return {"ids": ids, "mask": m, "term_mask": tm,
                "vis": torch.tensor(np.stack([x["vis"] for x in b])),
                "u": torch.tensor([x["u"] for x in b], dtype=torch.float),
                "yc": torch.tensor([x["yc"] for x in b], dtype=torch.float),
                "y": torch.tensor([x["y"] for x in b]),
                "key": [x["key"] for x in b]}
    return collate


class GatedMASC(nn.Module):
    """MADSC gate + the paper's KAN fusion head, in three forms.

    `--head mlp`  : the incumbent Linear over the fused vector.
    `--head kan`  : the paper as written — a 2-layer width-256 KAN over the SAME fused
                    vector. Chapter A measured this family flat.
    `--head ikan` : interaction-KAN. Rather than asking a KAN to discover cross-modal
                    structure inside one wide concatenation, project each modality to 192
                    with its own LayerNorm and hand the KAN the interaction variables
                    explicitly:  [t', v', t'*v', |t'-v'|].
                    The text path stays a *residual baseline* (z_text = W_t t'), and the
                    KAN supplies a correction scaled by alpha, initialised **0.1 and not
                    zero** — a zero branch times a zero scale is the identically-dead
                    gradient §D.22 found in PACS.

    The KG stream `g_a` of the paper's `[t_a ; v_a ; g_a]` is absent here: data/kg_index
    and data/kg_evidence do not exist (§D.34). The interaction set is therefore the
    2-modality subset of the intended 3-modality one.
    """

    def __init__(self, model_id, vis_dim=512, dropout=0.1, fuse="convex",
                 head="mlp", dproj=192, kan_width=256):
        super().__init__()
        from transformers import AutoConfig, AutoModel
        cfg = AutoConfig.from_pretrained(model_id)
        self.enc = AutoModel.from_pretrained(model_id, dtype=torch.float32)
        h = cfg.hidden_size
        self.fuse, self.head_kind = fuse, head
        self.vproj = nn.Linear(vis_dim, h)
        self.drop = nn.Dropout(dropout)
        # Eq. 9 / Eq. 13: four learnable scalars, exactly as in the paper
        self.w_u = nn.Parameter(torch.tensor(4.0))
        self.b_u = nn.Parameter(torch.tensor(-2.0))
        self.w_g = nn.Parameter(torch.tensor(2.0))
        self.b_g = nn.Parameter(torch.tensor(-1.0))

        d_in = h if fuse == "convex" else 3 * h
        if head == "mlp":
            self.head = nn.Linear(d_in, 3)
        elif head == "kan":
            from experts.kan import KAN
            self.head = KAN([d_in, kan_width, 3])
        elif head == "ikan":
            from experts.kan import KAN
            self.pt = nn.Linear(h, dproj)
            self.pv = nn.Linear(h, dproj)
            self.nt = nn.LayerNorm(dproj)
            self.nv = nn.LayerNorm(dproj)
            self.head = nn.Linear(dproj, 3)              # residual TEXT baseline
            self.kan = KAN([4 * dproj, kan_width, 3])
            self.alpha = nn.Parameter(torch.tensor(0.1))
        else:
            raise ValueError(head)
        self.last_norms = None
        self.no_vis = False

    def forward(self, ids, mask, term_mask, vis, sim):
        hs = self.enc(input_ids=ids, attention_mask=mask).last_hidden_state
        m = mask.unsqueeze(-1).float()
        tmask = term_mask.unsqueeze(-1)
        t_a = (hs * tmask).sum(1) / tmask.sum(1).clamp(min=1)        # pooled aspect tokens
        ctx = (hs * m).sum(1) / m.sum(1).clamp(min=1)                # sentence context
        v_a = self.vproj(vis)

        u = torch.sigmoid(self.w_u * sim + self.b_u)                 # Eq. 9
        g = torch.sigmoid(self.w_g * u + self.b_g).unsqueeze(-1)     # Eq. 13
        if self.no_vis:      # stream ablation: text only, gate hard-closed
            g = g * 0.0

        if self.head_kind == "ikan":
            t = self.nt(self.pt(t_a))
            v = self.nv(self.pv(v_a * g))          # gate still admits the visual evidence
            # the diagnostic the proposal asks for: comparable scales before fusion, or the
            # KAN spends capacity learning "text is 10x bigger than visual"
            self.last_norms = (float(t.norm(dim=-1).mean()), float(v.norm(dim=-1).mean()))
            x = torch.cat([t, v, t * v, (t - v).abs()], dim=-1)
            lg = self.head(self.drop(t)) + self.alpha * self.kan(x)
            return lg.float(), u, g.squeeze(-1)

        z = g * v_a + (1.0 - g) * t_a                                # Eq. 14
        feat = z if self.fuse == "convex" else torch.cat([z, t_a, ctx], dim=-1)
        return self.head(self.drop(feat)).float(), u, g.squeeze(-1)


@torch.no_grad()
def run(model, loader, device):
    model.eval()
    P, Y, K, G = [], [], [], []
    for b in loader:
        with torch.autocast("cuda", dtype=torch.float16, enabled=device.type == "cuda"):
            lg, _, g = model(b["ids"].to(device), b["mask"].to(device),
                             b["term_mask"].to(device), b["vis"].to(device),
                             b["u"].to(device))
        P.append(torch.softmax(lg.float(), -1).cpu().numpy())
        G.append(g.float().cpu().numpy())
        Y.extend(b["y"].tolist()); K.extend(b["key"])
    P = np.concatenate(P)
    acc = 100.0 * float((P.argmax(1) == np.array(Y)).mean()) if Y else 0.0
    return acc, P, K, float(np.concatenate(G).mean())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="vinai/bertweet-large")
    ap.add_argument("--dataset", default="twitter2015")
    ap.add_argument("--desc", default="data/aadg/twitter2015")
    ap.add_argument("--spans", default=None)
    ap.add_argument("--fuse", choices=["convex", "concat"], default="convex")
    ap.add_argument("--head", choices=["mlp", "kan", "ikan"], default="ikan",
                    help="ikan (DEFAULT) = interaction-KAN over [t,v,t*v,|t-v|] with a "
                         "residual text baseline; measured best of the three (§D.35: test "
                         "77.92 vs concat-KAN 77.43, MLP-fusion 77.24, text-only 77.72). "
                         "kan = the paper literally -- a 2-layer width-256 KAN over the "
                         "same blunt fused vector, kept selectable as the faithful "
                         "comparison. mlp = the incumbent Linear, i.e. the -KAN ablation.")
    ap.add_argument("--dproj", type=int, default=192)
    ap.add_argument("--kan-width", type=int, default=256)
    ap.add_argument("--no-vis", action="store_true",
                    help="stream ablation: hard-close the gate, text only")
    ap.add_argument("--lambda-conf", type=float, default=0.5)
    ap.add_argument("--seed", type=int, default=50)
    ap.add_argument("--epochs", type=int, default=8)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--lr", type=float, default=1e-5)
    ap.add_argument("--head-lr", type=float, default=1e-3)
    ap.add_argument("--patience", type=int, default=3)
    ap.add_argument("--score-only", action="store_true",
                    help="skip training, load <out>/best.pt and just (re-)score")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    D = Path(args.desc)

    from transformers import AutoTokenizer, get_linear_schedule_with_warmup
    tok = AutoTokenizer.from_pretrained(args.model)
    insts = {s: load(args.dataset, s) for s in ("train", "dev", "test")}
    ex = {s: masc_examples(v) for s, v in insts.items()}
    desc = {s: json.load(open(D / f"desc_{s}.json")) for s in ("train", "dev", "test")}
    art = {s: load_aspect_npz(D / f"aspect_{s}.npz") for s in ("train", "dev", "test")}
    vis_dim = next(iter(art["train"][0].values())).shape[0]
    coll = make_collate(tok.pad_token_id, vis_dim)
    dl = {s: DataLoader(DS(ex[s], tok, desc[s], *art[s], vis_dim),
                        batch_size=args.batch, shuffle=(s == "train"), collate_fn=coll)
          for s in ("train", "dev", "test")}

    model = GatedMASC(args.model, vis_dim, fuse=args.fuse, head=args.head,
                      dproj=args.dproj, kan_width=args.kan_width).to(device)
    model.no_vis = args.no_vis
    nh = sum(p.numel() for n, p in model.named_parameters()
             if not n.startswith('enc.'))
    print(f"head={args.head} no_vis={args.no_vis} non-encoder params {nh/1e6:.2f}M",
          flush=True)
    body = [p for n, p in model.named_parameters() if n.startswith("enc.")]
    head = [p for n, p in model.named_parameters() if not n.startswith("enc.")]
    opt = torch.optim.AdamW([{"params": body, "lr": args.lr},
                             {"params": head, "lr": args.head_lr}], weight_decay=0.01)
    steps = len(dl["train"]) * args.epochs
    sched = get_linear_schedule_with_warmup(opt, int(0.1 * steps), steps)
    scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda")
    ce, bce = nn.CrossEntropyLoss(), nn.BCELoss()

    best, best_ep, bad, t0 = -1.0, -1, 0, time.time()
    n_epochs = 0 if args.score_only else args.epochs
    for ep in range(1, n_epochs + 1):
        model.train()
        tot = 0.0
        for b in dl["train"]:
            opt.zero_grad(set_to_none=True)
            with torch.autocast("cuda", dtype=torch.float16, enabled=device.type == "cuda"):
                lg, u, _ = model(b["ids"].to(device), b["mask"].to(device),
                                 b["term_mask"].to(device), b["vis"].to(device),
                                 b["u"].to(device))
            loss = ce(lg, b["y"].to(device)) + args.lambda_conf * bce(
                u.float().clamp(1e-6, 1 - 1e-6), b["yc"].to(device))
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(opt); scaler.update(); sched.step()
            tot += loss.item()
        acc, _, _, gbar = run(model, dl["dev"], device)
        print(f"ep{ep} loss {tot/len(dl['train']):.4f} | dev acc {acc:.2f} | mean gate "
              f"{gbar:.3f} | {time.time()-t0:.0f}s", flush=True)
        if acc > best:
            best, best_ep, bad = acc, ep, 0
            torch.save(model.state_dict(), out / "best.pt")
        else:
            bad += 1
            if bad >= args.patience:
                print("early stop"); break

    model.load_state_dict(torch.load(out / "best.pt", map_location=device))
    res = {"model": args.model, "seed": args.seed, "fuse": args.fuse,
           "best_dev_acc": best, "best_epoch": best_ep}
    for s in ("dev", "test"):
        acc, P, K, gbar = run(model, dl[s], device)
        res[f"{s}_acc_goldspan"] = acc
        res[f"{s}_mean_gate"] = gbar
        np.savez_compressed(out / f"probs_{s}.npz", probs=P,
                            keys=np.array(K, dtype=np.int64))
        print(f"[{s}] gold-span MASC acc {acc:.2f} | mean gate {gbar:.3f}", flush=True)
        if args.spans:
            sp = json.load(open(Path(args.spans) / f"spans_{s}.json"))
            e2 = masc_examples_for_spans(insts[s], [[tuple(x) for x in i] for i in sp])
            dl2 = DataLoader(DS(e2, tok, desc[s], *art[s], vis_dim),
                             batch_size=args.batch, collate_fn=coll)
            _, P2, K2, _ = run(model, dl2, device)
            np.savez_compressed(out / f"probs_span_{s}.npz", probs=P2,
                                keys=np.array(K2, dtype=np.int64))
            print(f"[{s}] scored {len(e2)} predicted spans", flush=True)
    json.dump(res, open(out / "metrics.json", "w"), indent=2)
    print(json.dumps(res, indent=2))


if __name__ == "__main__":
    main()
