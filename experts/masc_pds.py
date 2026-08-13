"""C15b — PDS student: learn the DIRECTION evidence pushes, not just whether to look.

Teacher labels come from `experts/pds_teacher.py` as a soft distribution over
{POS-shift, NEG-shift, no-shift} per training aspect. The student is trained so that the
*difference* between its evidence-conditioned logits and its text-only logits matches that
direction:

    z_full = g * v_a + (1 - g) * t_a      (evidence-conditioned, MADSC-style gate)
    z_text = t_a                          (same encoder, evidence removed)
    delta  = head(z_full) - head(z_text)

    teacher POS-shift  ->  hinge:  delta[POS] must beat the other two by a margin
    teacher NEG-shift  ->  hinge:  delta[NEG] must beat the other two by a margin
    teacher no-shift   ->  penalty: ||delta||^2 small
    ... each weighted by the teacher's own probability, so its uncertainty stays uncertainty.

Why this is not another gate: the gate answers "how much visual signal to admit"; this
answers "which way is it allowed to move the answer". Chapter B measured relevance-only
supervision at ~0 and Chapter C measured the calibrated gate at +0.10, which is exactly what
a real-but-directionless signal looks like.

Cost note: the visual vector enters only at the fusion, so both branches share ONE encoder
pass and differ only in the head input -- the counterfactual is nearly free.

    python experts/masc_pds.py --model vinai/bertweet-large --spans runs/mate_ens5_hr \
        --out runs/masc_pds_btwL
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
from experts.common import (POL2ID, POLARITIES, load, masc_examples,  # noqa: E402
                            masc_examples_for_spans, set_seed)
from experts.masc_gated import load_aspect_npz  # noqa: E402

NEG, NEU, POS = 0, 1, 2          # index order of POLARITIES


class DS(Dataset):
    def __init__(self, ex, tok, desc, vis, u, y, dirp, vis_dim, max_len=192):
        self.ex, self.tok, self.desc = ex, tok, desc
        self.vis, self.u, self.dirp, self.vis_dim = vis, u, dirp, vis_dim
        self.max_len = max_len

    def __len__(self):
        return len(self.ex)

    def __getitem__(self, i):
        e = self.ex[i]
        second = e.marked_text()
        if self.desc is not None:
            second = second + " " + self.tok.sep_token + " " + self.desc[e.inst_idx]
        enc = self.tok(e.term, second, truncation="only_second", max_length=self.max_len)
        tl = len(self.tok(e.term, add_special_tokens=False)["input_ids"])
        return {"ids": enc["input_ids"], "term_len": tl, "y": POL2ID[e.polarity],
                "key": e.key,
                "vis": self.vis.get(e.key, np.zeros(self.vis_dim, dtype=np.float32)),
                "u": self.u.get(e.key, 0.0),
                "dir": self.dirp.get(e.key, np.array([0., 0., 1.], dtype=np.float32))}


def make_collate(pad_id):
    def collate(b):
        L = max(len(x["ids"]) for x in b)
        ids = torch.full((len(b), L), pad_id, dtype=torch.long)
        m = torch.zeros((len(b), L), dtype=torch.long)
        tm = torch.zeros((len(b), L), dtype=torch.float)
        for i, x in enumerate(b):
            n = len(x["ids"])
            ids[i, :n] = torch.tensor(x["ids"]); m[i, :n] = 1
            tm[i, 1:1 + min(x["term_len"], L - 1)] = 1.0
        return {"ids": ids, "mask": m, "term_mask": tm,
                "vis": torch.tensor(np.stack([x["vis"] for x in b])),
                "u": torch.tensor([x["u"] for x in b], dtype=torch.float),
                "dir": torch.tensor(np.stack([x["dir"] for x in b])),
                "y": torch.tensor([x["y"] for x in b]),
                "key": [x["key"] for x in b]}
    return collate


class PDSResidual(nn.Module):
    """PDS as a zero-initialised CORRECTION on a strong text classifier.

    The first PDS attempt replaced the classifier and lost 3.9 points standalone while
    producing the most diverse members of the campaign (unique-right 6.56%). The signal is
    real; asking a weak model to be the primary classifier was the error. Here:

        lg_base  = head_base([t_a, ctx])            <- an ordinary strong MASC head, NO visual
        delta    = head_delta([v_a, t_a])           <- FINAL LAYER ZERO-INIT => starts at 0
        lg_full  = lg_base + alpha * g * delta      <- alpha learnable, init 0.1

    So at step 0 the model IS the plain classifier, and evidence may only move the decision
    where the teacher's direction label justifies it. `lg_base` is supervised directly too,
    which stops the base degenerating and outsourcing its job to the residual.

    The direction loss acts on the DECISION MARGIN, not the representation:
        d_pos = delta[POS] - delta[NEU]   d_neg = delta[NEG] - delta[NEU]
    L2 on the representation was the original mistake -- evidence is allowed to change the
    representation freely, only the sign of its effect on the decision is constrained.
    """

    def __init__(self, model_id, vis_dim=512, dropout=0.1):
        super().__init__()
        from transformers import AutoConfig, AutoModel
        cfg = AutoConfig.from_pretrained(model_id)
        self.enc = AutoModel.from_pretrained(model_id, dtype=torch.float32)
        h = cfg.hidden_size
        self.vproj = nn.Linear(vis_dim, h)
        self.drop = nn.Dropout(dropout)
        self.w_u = nn.Parameter(torch.tensor(4.0))
        self.b_u = nn.Parameter(torch.tensor(-2.0))
        self.w_g = nn.Parameter(torch.tensor(2.0))
        self.b_g = nn.Parameter(torch.tensor(-1.0))
        self.head_base = nn.Linear(2 * h, 3)
        self.head_delta = nn.Sequential(nn.Linear(2 * h, h), nn.GELU(), nn.Linear(h, 3))
        nn.init.zeros_(self.head_delta[-1].weight)
        nn.init.zeros_(self.head_delta[-1].bias)
        self.alpha = nn.Parameter(torch.tensor(0.1))
        # NEU-escape: amplify the correction exactly where the measured errors are
        # (86 POS and 35 NEG aspects predicted NEU). beta=0 -> identical to plain residual.
        self.beta = nn.Parameter(torch.tensor(0.0))
        self.neu_escape = False

    def forward(self, ids, mask, term_mask, vis, sim):
        hs = self.enc(input_ids=ids, attention_mask=mask).last_hidden_state
        m = mask.unsqueeze(-1).float(); tmk = term_mask.unsqueeze(-1)
        t_a = (hs * tmk).sum(1) / tmk.sum(1).clamp(min=1)
        ctx = (hs * m).sum(1) / m.sum(1).clamp(min=1)
        v_a = self.vproj(vis)
        u = torch.sigmoid(self.w_u * sim + self.b_u)
        g = torch.sigmoid(self.w_g * u + self.b_g).unsqueeze(-1)
        lg_base = self.head_base(self.drop(torch.cat([t_a, ctx], -1))).float()
        delta = self.head_delta(self.drop(torch.cat([v_a, t_a], -1))).float()
        scale = self.alpha
        if self.neu_escape:
            p_neu = torch.softmax(lg_base, -1)[:, NEU:NEU + 1]
            scale = self.alpha * (1.0 + self.beta * p_neu)
        lg_full = lg_base + scale * g * delta
        return lg_full, lg_base, delta, g.squeeze(-1)


def pds_continuous_loss(delta, dirp, scale=1.0, w_pos=0.147, w_neg=1.0):
    """PDS-v2: regress the residual onto the teacher's SIGNED, CONFIDENCE-SCALED direction.

    The 3-way {POS,NEG,NONE} target throws away 75% of the teacher's output, because "no
    shift" mostly means *the teacher could not determine a direction*, not *the effect is
    zero*. The soft distribution already carries that distinction (mean confidence 0.795),
    so use it directly:

        q     = 1 - P(NONE)                  how sure the teacher is that ANY shift exists
        delta = q * (P(POS) - P(NEG))        signed magnitude in [-1, 1]

    +0.92 = strong positive effect · +0.18 = weak · 0.00 = ambiguous · -0.67 = strong negative.
    The student's signed residual (delta[POS] - delta[NEG]) is regressed onto it with
    SmoothL1, so ambiguous cases pull toward 0 *smoothly* rather than being hard-clamped —
    which is what made the original L2-on-representation a suppression term.
    Class weighting is applied per-example by the sign of the teacher target, PDS-internally.
    """
    q = 1.0 - dirp[:, 2]
    tgt = scale * q * (dirp[:, 0] - dirp[:, 1])
    s_stu = delta[:, POS] - delta[:, NEG]
    w = torch.where(tgt >= 0, torch.full_like(tgt, w_pos), torch.full_like(tgt, w_neg))
    return (w * torch.nn.functional.smooth_l1_loss(s_stu, tgt, reduction="none")).mean()


def pds_margin_loss(delta, dirp, gamma=0.5, eps=0.1, w_none=0.05,
                    w_pos=0.147, w_neg=1.0):
    """Signed-margin direction loss with PDS-internal POS/NEG balancing.

    Left unbalanced the auxiliary task would teach "visual evidence usually pushes one
    way" -- a bias, not a mechanism. Inverse-frequency weights are applied HERE ONLY; the
    MASC classifier itself is never class-weighted (Chapter B/C measured that as harmful).

    The defaults (0.147 / 1.0) are the inverse frequencies of the §C.25 caption teacher's
    687 POS : 101 NEG. **They are wrong for any other teacher** -- the §D12 direct image
    teacher is 214 POS : 382 NEG, i.e. NEG-leaning, and reusing these defaults would push
    the same way the teacher already leans instead of balancing it. `main()` therefore
    derives them from whatever labels are actually loaded unless told otherwise.
    """
    d_pos = delta[:, POS] - delta[:, NEU]
    d_neg = delta[:, NEG] - delta[:, NEU]
    l_pos = torch.relu(gamma - d_pos)
    l_neg = torch.relu(gamma - d_neg)
    l_non = torch.relu(d_pos.abs() - eps) + torch.relu(d_neg.abs() - eps)   # dead zone
    return (w_pos * dirp[:, 0] * l_pos + w_neg * dirp[:, 1] * l_neg
            + w_none * dirp[:, 2] * l_non).mean()


class PDSModel(nn.Module):
    def __init__(self, model_id, vis_dim=512, dropout=0.1):
        super().__init__()
        from transformers import AutoConfig, AutoModel
        cfg = AutoConfig.from_pretrained(model_id)
        self.enc = AutoModel.from_pretrained(model_id, dtype=torch.float32)
        h = cfg.hidden_size
        self.vproj = nn.Linear(vis_dim, h)
        self.drop = nn.Dropout(dropout)
        self.w_u = nn.Parameter(torch.tensor(4.0))
        self.b_u = nn.Parameter(torch.tensor(-2.0))
        self.w_g = nn.Parameter(torch.tensor(2.0))
        self.b_g = nn.Parameter(torch.tensor(-1.0))
        self.head = nn.Linear(3 * h, 3)

    def forward(self, ids, mask, term_mask, vis, sim):
        hs = self.enc(input_ids=ids, attention_mask=mask).last_hidden_state
        m = mask.unsqueeze(-1).float(); tmk = term_mask.unsqueeze(-1)
        t_a = (hs * tmk).sum(1) / tmk.sum(1).clamp(min=1)
        ctx = (hs * m).sum(1) / m.sum(1).clamp(min=1)
        v_a = self.vproj(vis)
        u = torch.sigmoid(self.w_u * sim + self.b_u)
        g = torch.sigmoid(self.w_g * u + self.b_g).unsqueeze(-1)
        z_full = g * v_a + (1 - g) * t_a
        # SAME encoder pass; only the fusion input differs -> counterfactual is nearly free
        lg_full = self.head(self.drop(torch.cat([z_full, t_a, ctx], -1))).float()
        lg_text = self.head(self.drop(torch.cat([t_a, t_a, ctx], -1))).float()
        return lg_full, lg_text, g.squeeze(-1)


def pds_loss(lg_full, lg_text, dirp, margin=0.5, w_none=1.0):
    """Soft-weighted counterfactual direction constraint.

    `w_none` down-weights the no-shift term. MEASURED REASON: the teacher labels 2391/3179
    (75.2%) of train aspects "no shift", so at w_none=1 the objective is dominated by an L2
    penalty pulling z_full toward z_text -- i.e. it becomes a visual-SUPPRESSION regulariser
    rather than a direction constraint, which is exactly the drop we measured (78.11 vs
    79.75). Down-weighting it keeps supervision on the 788 aspects that carry a direction.
    """
    d = lg_full - lg_text                                   # [B,3] over (NEG,NEU,POS)
    other_pos = torch.maximum(d[:, NEG], d[:, NEU])
    other_neg = torch.maximum(d[:, POS], d[:, NEU])
    l_pos = torch.relu(margin - (d[:, POS] - other_pos))
    l_neg = torch.relu(margin - (d[:, NEG] - other_neg))
    l_non = (d ** 2).mean(-1)
    return (dirp[:, 0] * l_pos + dirp[:, 1] * l_neg + w_none * dirp[:, 2] * l_non).mean()


@torch.no_grad()
def run(model, loader, device):
    model.eval(); P, Y, K = [], [], []
    for b in loader:
        with torch.autocast("cuda", dtype=torch.float16, enabled=device.type == "cuda"):
            mo = model(b["ids"].to(device), b["mask"].to(device),
                       b["term_mask"].to(device), b["vis"].to(device), b["u"].to(device))
        P.append(torch.softmax(mo[0].float(), -1).cpu().numpy())
        Y.extend(b["y"].tolist()); K.extend(b["key"])
    P = np.concatenate(P)
    return 100.0 * float((P.argmax(1) == np.array(Y)).mean()) if Y else 0.0, P, K


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="vinai/bertweet-large")
    ap.add_argument("--dataset", default="twitter2015")
    ap.add_argument("--desc", default="data/aadg/twitter2015")
    ap.add_argument("--pds", default="data/pds/twitter2015")
    ap.add_argument("--spans", default=None)
    ap.add_argument("--lambda-pds", type=float, default=0.5)
    ap.add_argument("--margin", type=float, default=0.5)
    ap.add_argument("--pds-mode", choices=["margin", "continuous"], default="margin")
    ap.add_argument("--neu-escape", action="store_true",
                    help="amplify the residual when the base leans NEU (targets the measured "
                         "86 POS->NEU / 35 NEG->NEU failures)")
    ap.add_argument("--residual", action="store_true",
                    help="zero-init residual formulation + signed-margin direction loss")
    ap.add_argument("--gamma", type=float, default=0.5)
    ap.add_argument("--eps", type=float, default=0.1)
    ap.add_argument("--w-pos", type=float, default=None,
                    help="override the auto inverse-frequency POS direction weight")
    ap.add_argument("--w-neg", type=float, default=None)
    ap.add_argument("--w-none", type=float, default=1.0,
                    help="weight of the no-shift L2 term (75%% of teacher labels)")
    ap.add_argument("--seed", type=int, default=70)
    ap.add_argument("--epochs", type=int, default=8)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--lr", type=float, default=1e-5)
    ap.add_argument("--head-lr", type=float, default=1e-3)
    ap.add_argument("--patience", type=int, default=3)
    ap.add_argument("--score-only", action="store_true",
                    help="skip training, load <out>/best.pt and just (re-)score, so the "
                         "candidate anchor set can change without retraining the member")
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
    desc = {s: json.load(open(D / f"desc_{s}.json")) for s in insts}
    art = {s: load_aspect_npz(D / f"aspect_{s}.npz") for s in insts}
    vis_dim = next(iter(art["train"][0].values())).shape[0]

    z = np.load(Path(args.pds) / "direction_train.npz")
    dirp = {(int(k[0]), int(k[1]), int(k[2])): p.astype(np.float32)
            for p, k in zip(z["probs"], z["keys"])}
    hard = np.array([v.argmax() for v in dirp.values()])
    n_pos, n_neg = int((hard == 0).sum()), int((hard == 1).sum())
    print(f"PDS labels {len(dirp)} | POS-shift {n_pos} "
          f"NEG-shift {n_neg} no-shift {int((hard==2).sum())}", flush=True)
    # Inverse-frequency POS/NEG weights derived from THIS teacher, not hardcoded for the
    # §C.25 one. Reproduces 0.147 / 1.0 exactly on the caption teacher's 687:101.
    W_POS = args.w_pos if args.w_pos is not None else min(1.0, n_neg / max(n_pos, 1))
    W_NEG = args.w_neg if args.w_neg is not None else min(1.0, n_pos / max(n_neg, 1))
    print(f"PDS direction weights: w_pos {W_POS:.3f}  w_neg {W_NEG:.3f}", flush=True)

    coll = make_collate(tok.pad_token_id)
    dl = {s: DataLoader(DS(ex[s], tok, desc[s], *art[s],
                           dirp if s == "train" else {}, vis_dim),
                        batch_size=args.batch, shuffle=(s == "train"), collate_fn=coll)
          for s in insts}

    model = (PDSResidual if args.residual else PDSModel)(args.model, vis_dim).to(device)
    if args.residual and args.neu_escape:
        model.neu_escape = True
    body = [p for n, p in model.named_parameters() if n.startswith("enc.")]
    head = [p for n, p in model.named_parameters() if not n.startswith("enc.")]
    opt = torch.optim.AdamW([{"params": body, "lr": args.lr},
                             {"params": head, "lr": args.head_lr}], weight_decay=0.01)
    steps = len(dl["train"]) * args.epochs
    sched = get_linear_schedule_with_warmup(opt, int(0.1 * steps), steps)
    scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda")
    ce = nn.CrossEntropyLoss()

    best, best_ep, bad, t0 = -1.0, -1, 0, time.time()
    for ep in range(1, (0 if args.score_only else args.epochs) + 1):
        model.train(); tot = tpds = 0.0
        for b in dl["train"]:
            opt.zero_grad(set_to_none=True)
            with torch.autocast("cuda", dtype=torch.float16, enabled=device.type == "cuda"):
                mo = model(b["ids"].to(device), b["mask"].to(device),
                           b["term_mask"].to(device), b["vis"].to(device),
                           b["u"].to(device))
            y = b["y"].to(device); dirp = b["dir"].to(device)
            if args.residual:
                lgf, lgb, delta, _ = mo
                lp = (pds_continuous_loss(delta, dirp) if args.pds_mode == "continuous"
                      else pds_margin_loss(delta, dirp, args.gamma, args.eps, args.w_none,
                                           W_POS, W_NEG))
                loss = ce(lgf, y) + 0.5 * ce(lgb, y) + args.lambda_pds * lp
            else:
                lgf, lgt, _ = mo
                lp = pds_loss(lgf, lgt, dirp, args.margin, args.w_none)
                loss = ce(lgf, y) + args.lambda_pds * lp
            scaler.scale(loss).backward(); scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(opt); scaler.update(); sched.step()
            tot += loss.item(); tpds += float(lp)
        acc, *_ = run(model, dl["dev"], device)
        print(f"ep{ep} loss {tot/len(dl['train']):.4f} (pds {tpds/len(dl['train']):.4f}) "
              f"| dev acc {acc:.2f} | {time.time()-t0:.0f}s", flush=True)
        if acc > best:
            best, best_ep, bad = acc, ep, 0
            torch.save(model.state_dict(), out / "best.pt")
        else:
            bad += 1
            if bad >= args.patience:
                print("early stop"); break

    # The pds_res_* members were trained before §C.27 added the NEU-escape `beta`, so their
    # checkpoints lack it. `beta` is read ONLY when self.neu_escape is on (line ~145), which
    # is off unless --neu-escape is passed, so leaving it at its 0.0 init re-scores them
    # exactly. Narrow on purpose: anything else missing still raises.
    sd = torch.load(out / "best.pt", map_location=device)
    missing, unexpected = model.load_state_dict(sd, strict=False)
    extra = (set(missing) - {"beta"}) | set(unexpected)
    if extra:
        raise RuntimeError(f"state_dict mismatch beyond the NEU-escape beta: {sorted(extra)}")
    if missing and args.neu_escape:
        raise RuntimeError("checkpoint has no `beta` but --neu-escape was requested")
    res = {"model": args.model, "seed": args.seed, "best_dev_acc": best,
           "best_epoch": best_ep, "lambda_pds": args.lambda_pds}
    for s in ("dev", "test"):
        acc, P, K = run(model, dl[s], device)
        res[f"{s}_acc_goldspan"] = acc
        np.savez_compressed(out / f"probs_{s}.npz", probs=P,
                            keys=np.array(K, dtype=np.int64))
        print(f"[{s}] gold-span MASC acc {acc:.2f}", flush=True)
        if args.spans:
            sp = json.load(open(Path(args.spans) / f"spans_{s}.json"))
            e2 = masc_examples_for_spans(insts[s], [[tuple(x) for x in i] for i in sp])
            dl2 = DataLoader(DS(e2, tok, desc[s], *art[s], {}, vis_dim),
                             batch_size=args.batch, collate_fn=coll)
            _, P2, K2 = run(model, dl2, device)
            np.savez_compressed(out / f"probs_span_{s}.npz", probs=P2,
                                keys=np.array(K2, dtype=np.int64))
            print(f"[{s}] scored {len(e2)} predicted spans", flush=True)
    json.dump(res, open(out / "metrics.json", "w"), indent=2)
    print(json.dumps(res, indent=2))


if __name__ == "__main__":
    main()
