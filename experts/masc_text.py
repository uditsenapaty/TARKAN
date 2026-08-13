"""Cheap, architecture-diverse TEXT-only MASC members.

Chapter B's 71.26 ensemble was 4 Qwen2.5-VL members **plus 4 diverse text members**
(twitter-roberta x3, bertweet), and dev-selection consistently kept the text members:
diversity, not individual strength, is what they contribute. They cost minutes on a T4,
so they are the right thing to have before spending GPU on the expensive members.

Aspect is marked IN PLACE (Chapter B, B2) -- required to disambiguate a tweet where the
same surface form occurs twice with different gold polarity.

Both this and `pdq.py` accept `--spans`, so stage-2 can score the *predicted* candidate
anchors from stage-1 rather than gold spans.

    python experts/masc_text.py --model cardiffnlp/twitter-roberta-base-sentiment-latest \
        --seed 42 --out runs/masc_twrob_s42
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
from experts.common import (POL2ID, POLARITIES, load, masc_examples,  # noqa: E402
                            masc_examples_for_spans, set_seed)


class DS(Dataset):
    def __init__(self, examples, tok, max_len=128, desc=None, mark_siblings=False,
                 opinion_dropout=0.0, lexicon=None, keep_window=4, train=False):
        self.ex, self.tok = examples, tok
        self.mark_siblings = mark_siblings
        # opinion dropout: hide sentiment words that are FAR from the target, so the model
        # cannot answer "what is this tweet's mood" instead of "what does it say about
        # THIS aspect". Only active during training.
        self.opinion_dropout = opinion_dropout
        self.lexicon = lexicon or {}
        self.keep_window = keep_window
        self.train = train
        # AADG (MADSC): the aspect-aware description is appended as extra context.
        # It is per-INSTANCE, so every aspect of a tweet shares it.
        self.desc = desc
        self.max_len = max_len if desc is None else max(max_len, 192)

    def _drop_distant_opinions(self, e):
        import copy
        s0, e0 = e.span
        lo, hi = s0 - self.keep_window, e0 + self.keep_window
        toks = list(e.tokens)
        changed = False
        for i, w in enumerate(toks):
            if lo <= i < hi:
                continue
            if w.lower() in self.lexicon and random.random() < self.opinion_dropout:
                toks[i] = self.tok.mask_token or "[MASK]"
                changed = True
        if not changed:
            return e
        e2 = copy.copy(e)
        e2.tokens = toks
        return e2

    def __len__(self):
        return len(self.ex)

    def __getitem__(self, i):
        e = self.ex[i]
        if self.train and self.opinion_dropout > 0 and self.lexicon:
            e = self._drop_distant_opinions(e)
        second = e.marked_text(mark_siblings=self.mark_siblings)
        if self.desc is not None:
            second = second + " " + self.tok.sep_token + " " + self.desc[e.inst_idx]
        enc = self.tok(e.term, second, truncation="only_second",
                       max_length=self.max_len)
        return {"ids": enc["input_ids"], "y": POL2ID[e.polarity], "key": e.key,
                "inst": e.inst_idx}


def make_collate(pad_id):
    def collate(b):
        L = max(len(x["ids"]) for x in b)
        ids = torch.full((len(b), L), pad_id, dtype=torch.long)
        m = torch.zeros((len(b), L), dtype=torch.long)
        for i, x in enumerate(b):
            n = len(x["ids"])
            ids[i, :n] = torch.tensor(x["ids"])
            m[i, :n] = 1
        return {"ids": ids, "mask": m,
                "y": torch.tensor([x["y"] for x in b]),
                "inst": torch.tensor([x["inst"] for x in b]),
                "key": [x["key"] for x in b]}
    return collate


class InstanceBatchSampler(torch.utils.data.Sampler):
    """Keep all aspects of one tweet in the same batch.

    Required by the sibling-logit loss: the pairwise term only exists when two aspects of
    the SAME tweet with DIFFERENT gold polarity are present together. Random sampling over
    5.4k aspects would co-batch them only by accident.
    """

    def __init__(self, examples, batch_size, shuffle=True, seed=0):
        groups = {}
        for i, e in enumerate(examples):
            groups.setdefault(e.inst_idx, []).append(i)
        self.groups = list(groups.values())
        self.batch_size, self.shuffle, self.seed, self.epoch = batch_size, shuffle, seed, 0

    def __iter__(self):
        order = list(range(len(self.groups)))
        if self.shuffle:
            rng = random.Random(self.seed + self.epoch)
            rng.shuffle(order)
            self.epoch += 1
        batch = []
        for gi in order:
            g = self.groups[gi]
            if batch and len(batch) + len(g) > self.batch_size:
                yield batch
                batch = []
            batch += g
        if batch:
            yield batch

    def __len__(self):
        return max(1, sum(len(g) for g in self.groups) // self.batch_size)


SENTIC = Path(__file__).resolve().parent.parent / \
    "referred_clones/CORSA/CORSA/src/senticnet_word.txt"


def load_lexicon(min_abs: float = 0.3):
    """CORSA ships 39,891 word->polarity entries that its own code never reads. Dead weight
    for them, a free opinion-word detector for us.

    SenticNet scores plenty of function words (`the` = 0.935), so filter to open-class
    words: drop spaCy stopwords, require alphabetic and length >= 3, and keep only
    ADJ/ADV/VERB/NOUN-ish forms by excluding the stoplist. Masking `the` would be noise,
    not opinion dropout.
    """
    try:
        from spacy.lang.en.stop_words import STOP_WORDS
        stop = {w.lower() for w in STOP_WORDS}
    except Exception:
        stop = set()
    extra = {"rt", "http", "https", "amp", "via", "one", "two", "new", "get", "go", "make",
             "take", "day", "time", "today", "people", "man", "woman", "year", "week"}
    stop |= extra
    lex = {}
    try:
        for line in SENTIC.read_text(encoding="utf-8", errors="ignore").splitlines():
            parts = line.split("\t")
            if len(parts) != 2:
                continue
            w = parts[0].strip().lower()
            try:
                v = float(parts[1])
            except ValueError:
                continue
            if abs(v) < min_abs or w in stop or len(w) < 3 or not w.isalpha():
                continue
            lex[w] = v
    except OSError:
        pass
    return lex


def minority_margin_loss(logits, y, margin: float, neu_id: int = 1):
    """Require POS/NEG examples to beat NEU by a margin; do nothing on NEU examples.

    Global class weighting (Chapter B A1) rescales the whole distribution. The measured
    error here is one-directional -- 86 POS and 35 NEG collapse INTO NEU, while NEU recall
    is already 89.5 -- so only the minority-vs-NEU decision boundary needs pushing.
    """
    minor = y != neu_id
    if not minor.any():
        return logits.sum() * 0.0
    own = logits.gather(1, y[:, None]).squeeze(1)
    neu = logits[:, neu_id]
    return torch.relu(margin - (own - neu))[minor].mean()


def sibling_logit_loss(logits, y, inst, margin=1.0):
    """For two aspects of one tweet with different gold polarity, require BOTH to prefer
    their own label over the sibling's by a joint margin:

        relu( m - [s_A(pA) - s_A(pB)] - [s_B(pB) - s_B(pA)] )

    Chapter B's B9 attacked this with SupCon on representations and produced weak members.
    This constrains the DECISION directly, which is where the measured error lives
    (86 POS and 35 NEG predicted NEU -- the tweet's tone overriding the aspect's).
    """
    B = logits.size(0)
    if B < 2:
        return logits.sum() * 0.0
    same = inst[:, None] == inst[None, :]
    diff = y[:, None] != y[None, :]
    upper = torch.triu(torch.ones(B, B, dtype=torch.bool, device=logits.device), 1)
    mask = same & diff & upper
    if not mask.any():
        return logits.sum() * 0.0
    own = logits.gather(1, y[:, None])          # [B,1]  s_i(y_i)
    cross = logits[:, y]                        # [B,B]  cross[i,j] = s_i(y_j)
    term = margin - (own - cross) - (own - cross).T
    return torch.relu(term)[mask].mean()


class TextMASC(nn.Module):
    def __init__(self, model_id, dropout=0.1):
        super().__init__()
        from transformers import AutoConfig, AutoModel
        cfg = AutoConfig.from_pretrained(model_id)
        self.enc = AutoModel.from_pretrained(model_id, dtype=torch.float32)
        h = cfg.hidden_size
        self.drop = nn.Dropout(dropout)
        self.head = nn.Linear(2 * h, 3)

    def forward(self, ids, mask):
        h = self.enc(input_ids=ids, attention_mask=mask).last_hidden_state
        m = mask.unsqueeze(-1).float()
        mean = (h * m).sum(1) / m.sum(1).clamp(min=1)
        mx = h.masked_fill(m == 0, -1e4).max(1).values
        return self.head(self.drop(torch.cat([mean, mx], dim=-1))).float()


@torch.no_grad()
def run(model, loader, device):
    model.eval()
    P, Y, K = [], [], []
    for b in loader:
        with torch.autocast("cuda", dtype=torch.float16, enabled=device.type == "cuda"):
            lg = model(b["ids"].to(device), b["mask"].to(device))
        P.append(torch.softmax(lg.float(), -1).cpu().numpy())
        Y.extend(b["y"].tolist())
        K.extend(b["key"])
    P = np.concatenate(P)
    acc = 100.0 * float((P.argmax(1) == np.array(Y)).mean()) if Y else 0.0
    return acc, P, K


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="cardiffnlp/twitter-roberta-base-sentiment-latest")
    ap.add_argument("--dataset", default="twitter2015")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--epochs", type=int, default=6)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--lr", type=float, default=2e-5)
    ap.add_argument("--head-lr", type=float, default=1e-3)
    ap.add_argument("--patience", type=int, default=3)
    ap.add_argument("--spans", default=None,
                    help="dir with spans_{dev,test}.json from stage-1; also score those "
                         "candidate anchors (written to probs_span_*.npz)")
    ap.add_argument("--desc", default=None,
                    help="dir with desc_{train,dev,test}.json from experts/aadg.py "
                         "(MADSC aspect-aware descriptions)")
    ap.add_argument("--minority-margin", type=float, default=0.0,
                    help="margin by which POS/NEG must beat NEU (0 = off)")
    ap.add_argument("--opinion-dropout", type=float, default=0.0,
                    help="prob. of masking a lexicon opinion word outside the target window")
    ap.add_argument("--sibling-loss", type=float, default=0.0,
                    help="weight of the pairwise sibling-logit margin loss (needs the "
                         "instance-grouped sampler, enabled automatically)")
    ap.add_argument("--sibling-margin", type=float, default=1.0)
    ap.add_argument("--mark-siblings", action="store_true",
                    help="also bracket the other candidate aspects of the tweet with < >")
    ap.add_argument("--class-weight", choices=["none", "balanced", "sqrt"], default="none",
                    help="Chapter B measured class weighting as harmful, but that was on the "
                         "JOINT 7-tag BIO head where re-weighting collapsed extraction. On a "
                         "dedicated 3-way MASC head it cannot touch extraction, and the "
                         "measured failure mode here is the opposite one: NEU (58.5%% of test) "
                         "swallows POS (recall 72.2) and NEG (recall 62.0). A balanced member "
                         "is added to the ensemble, not used as a replacement.")
    ap.add_argument("--fold", type=int, default=None,
                    help="out-of-fold mode: train on instances with idx %% nfolds != fold, "
                         "then emit predictions for the held-out fold. Two runs give OOF "
                         "coverage of all 3179 train aspects -- 2.8x more combiner-fitting "
                         "data than dev (n=1122), whose binomial sigma ~1.2 is why every "
                         "Chapter-B dev-fit combiner failed to transfer.")
    ap.add_argument("--nfolds", type=int, default=2)
    ap.add_argument("--score-only", action="store_true",
                    help="skip training, load <out>/best.pt and just (re-)score. Lets the "
                         "candidate anchor set change without retraining any member.")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    LEX = load_lexicon() if args.opinion_dropout > 0 else {}
    if LEX:
        print(f"opinion lexicon: {len(LEX)} words |polarity|>=0.3", flush=True)

    from transformers import AutoTokenizer, get_linear_schedule_with_warmup
    tok = AutoTokenizer.from_pretrained(args.model)
    insts = {s: load(args.dataset, s) for s in ("train", "dev", "test")}
    ex = {s: masc_examples(v) for s, v in insts.items()}
    desc = None
    if args.desc:
        desc = {s: json.load(open(Path(args.desc) / f"desc_{s}.json"))
                for s in ("train", "dev", "test")}
    coll = make_collate(tok.pad_token_id)
    oof_ex = None
    if args.fold is not None:
        # split by INSTANCE so sibling aspects of one tweet never straddle the split
        oof_ex = [e for e in ex["train"] if e.inst_idx % args.nfolds == args.fold]
        ex["train"] = [e for e in ex["train"] if e.inst_idx % args.nfolds != args.fold]
        print(f"fold {args.fold}/{args.nfolds}: train {len(ex['train'])} "
              f"held-out {len(oof_ex)}", flush=True)
    dl = {}
    for s_, v in ex.items():
        ds_ = DS(v, tok, desc=(desc[s_] if desc else None),
                 mark_siblings=args.mark_siblings,
                 opinion_dropout=args.opinion_dropout, lexicon=LEX,
                 train=(s_ == "train"))
        if s_ == "train" and args.sibling_loss > 0:
            dl[s_] = DataLoader(ds_, collate_fn=coll,
                                batch_sampler=InstanceBatchSampler(v, args.batch,
                                                                   seed=args.seed))
        else:
            dl[s_] = DataLoader(ds_, batch_size=args.batch, shuffle=(s_ == "train"),
                                collate_fn=coll)

    model = TextMASC(args.model).to(device)
    head = [p for n, p in model.named_parameters() if not n.startswith("enc.")]
    body = [p for n, p in model.named_parameters() if n.startswith("enc.")]
    opt = torch.optim.AdamW([{"params": body, "lr": args.lr},
                             {"params": head, "lr": args.head_lr}], weight_decay=0.01)
    steps = len(dl["train"]) * args.epochs
    sched = get_linear_schedule_with_warmup(opt, int(0.1 * steps), steps)
    scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda")
    if args.class_weight == "none":
        lossf = nn.CrossEntropyLoss()
    else:
        cnt = np.bincount([POL2ID[e.polarity] for e in ex["train"]], minlength=3).astype(float)
        w = cnt.sum() / (3.0 * np.clip(cnt, 1, None))
        if args.class_weight == "sqrt":
            w = np.sqrt(w)
        w = w / w.mean()
        print(f"class counts {cnt.tolist()} -> weights {np.round(w, 3).tolist()}", flush=True)
        lossf = nn.CrossEntropyLoss(
            weight=torch.tensor(w, dtype=torch.float, device=device))

    best, best_ep, bad, t0 = -1.0, -1, 0, time.time()
    n_epochs = 0 if args.score_only else args.epochs
    for ep in range(1, n_epochs + 1):
        model.train()
        tot = 0.0
        for b in dl["train"]:
            opt.zero_grad(set_to_none=True)
            with torch.autocast("cuda", dtype=torch.float16, enabled=device.type == "cuda"):
                lg = model(b["ids"].to(device), b["mask"].to(device))
            loss = lossf(lg, b["y"].to(device))
            if args.minority_margin > 0:
                loss = loss + minority_margin_loss(lg, b["y"].to(device),
                                                   args.minority_margin)
            if args.sibling_loss > 0:
                loss = loss + args.sibling_loss * sibling_logit_loss(
                    lg, b["y"].to(device), b["inst"].to(device), args.sibling_margin)
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(opt); scaler.update(); sched.step()
            tot += loss.item()
        acc, *_ = run(model, dl["dev"], device)
        print(f"ep{ep} loss {tot/len(dl['train']):.4f} | dev acc {acc:.2f} "
              f"| {time.time()-t0:.0f}s", flush=True)
        if acc > best:
            best, best_ep, bad = acc, ep, 0
            torch.save(model.state_dict(), out / "best.pt")
        else:
            bad += 1
            if bad >= args.patience:
                print("early stop"); break

    model.load_state_dict(torch.load(out / "best.pt", map_location=device))
    res = {"model": args.model, "seed": args.seed, "best_dev_acc": best,
           "best_epoch": best_ep}
    if oof_ex is not None:
        acc, P, K = run(model, DataLoader(DS(oof_ex, tok, desc=(desc["train"] if desc else None),
                                             mark_siblings=args.mark_siblings),
                                          batch_size=args.batch, collate_fn=coll), device)
        np.savez_compressed(out / f"probs_oof{args.fold}.npz", probs=P,
                            keys=np.array(K, dtype=np.int64))
        res[f"oof{args.fold}_acc"] = acc
        print(f"[oof{args.fold}] n={len(oof_ex)} acc {acc:.2f}", flush=True)

    for s in ("dev", "test"):
        # (a) gold spans -> the member's honest MASC accuracy
        acc, P, K = run(model, DataLoader(DS(ex[s], tok, desc=(desc[s] if desc else None), mark_siblings=args.mark_siblings),
                                          batch_size=args.batch, collate_fn=coll), device)
        res[f"{s}_acc_goldspan"] = acc
        np.savez_compressed(out / f"probs_{s}.npz", probs=P,
                            keys=np.array(K, dtype=np.int64))
        print(f"[{s}] gold-span MASC acc {acc:.2f} (n={len(ex[s])})", flush=True)
        # (b) stage-1 candidate anchors -> what assemble.py actually consumes
        if args.spans:
            sp = json.load(open(Path(args.spans) / f"spans_{s}.json"))
            e2 = masc_examples_for_spans(insts[s], [[tuple(x) for x in inst] for inst in sp])
            _, P2, K2 = run(model, DataLoader(DS(e2, tok, desc=(desc[s] if desc else None), mark_siblings=args.mark_siblings),
                                              batch_size=args.batch, collate_fn=coll), device)
            np.savez_compressed(out / f"probs_span_{s}.npz", probs=P2,
                                keys=np.array(K2, dtype=np.int64))
            print(f"[{s}] scored {len(e2)} predicted spans", flush=True)
    json.dump(res, open(out / "metrics.json", "w"), indent=2)
    print(json.dumps(res, indent=2))


if __name__ == "__main__":
    main()
