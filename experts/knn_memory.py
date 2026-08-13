"""C11 — non-parametric aspect/context memory member.

Every member built so far is a fine-tuned transformer with a softmax head, which is why the
correlation matrix never drops below ~1.6 and the 95.18 oracle stays unreachable. This
member is not that: it does **no gradient training at all**. It embeds each TRAIN aspect in
its local context with a FROZEN off-the-shelf sentence encoder, stores the gold polarity,
and at inference returns a similarity-weighted vote over the nearest training contexts.

Why it should decorrelate: its errors are governed by whether a similar labelled context
exists in 3179 training aspects, not by what a fine-tuned encoder generalises. Twitter data
makes this unusually promising — recurring entities and stock phrasings ("... is everything",
"RIP ...", "congrats to ...") carry strong, memorisable polarity priors.

It is also the cheapest thing in the whole pipeline: one frozen forward pass over ~5.4k
short strings, no backward pass, seconds on CPU.

Chapter B listed "kNN memory" as mechanism-decorrelated but dismissed it as "likely too weak
to cross 1.2pt". That judged it as a *replacement*. As an ensemble *member* it does not need
to be strong, only decorrelated and non-trivial -- exactly the lesson C9 just re-taught
(the balanced member is 0.87 WORSE standalone yet uniquely right on 4.63%).

    python experts/knn_memory.py --spans runs/mate_ens5 --out runs/masc_knn
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from experts.common import (POL2ID, POLARITIES, load, masc_examples,  # noqa: E402
                            masc_examples_for_spans)

ENCODER = "sentence-transformers/all-MiniLM-L6-v2"


def context_string(e, window: int = 5) -> str:
    """aspect term + a +-window token neighbourhood, with the target made explicit."""
    s, t = e.span
    lo, hi = max(0, s - window), min(len(e.tokens), t + window)
    left = " ".join(e.tokens[lo:s])
    right = " ".join(e.tokens[t:hi])
    return f"{e.term} : {left} [ {e.term} ] {right}"


@torch.no_grad()
def embed(texts, tok, model, device, batch=128):
    out = []
    for i in range(0, len(texts), batch):
        enc = tok(texts[i:i + batch], padding=True, truncation=True, max_length=64,
                  return_tensors="pt").to(device)
        h = model(**enc).last_hidden_state
        m = enc["attention_mask"].unsqueeze(-1).float()
        v = (h * m).sum(1) / m.sum(1).clamp(min=1)              # mean pooling
        out.append(torch.nn.functional.normalize(v, dim=-1).cpu().numpy())
    return np.concatenate(out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="twitter2015")
    ap.add_argument("--k", type=int, default=24)
    ap.add_argument("--temp", type=float, default=0.05)
    ap.add_argument("--window", type=int, default=5)
    ap.add_argument("--prior", type=float, default=0.15,
                    help="mix weight toward the train class prior (smoothing)")
    ap.add_argument("--spans", default=None)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    from transformers import AutoModel, AutoTokenizer
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tok = AutoTokenizer.from_pretrained(ENCODER)
    model = AutoModel.from_pretrained(ENCODER).to(device).eval()

    insts = {s: load(args.dataset, s) for s in ("train", "dev", "test")}
    tr = masc_examples(insts["train"])
    K = embed([context_string(e, args.window) for e in tr], tok, model, device)
    V = np.array([POL2ID[e.polarity] for e in tr])
    prior = np.bincount(V, minlength=3).astype(np.float64)
    prior /= prior.sum()
    print(f"memory: {len(tr)} train contexts, prior {np.round(prior,3).tolist()}", flush=True)

    def vote(ex):
        Q = embed([context_string(e, args.window) for e in ex], tok, model, device)
        sim = Q @ K.T                                            # [n, n_train]
        idx = np.argpartition(-sim, args.k, axis=1)[:, :args.k]
        P = np.zeros((len(ex), 3))
        for i in range(len(ex)):
            nn = idx[i]
            w = np.exp((sim[i, nn] - sim[i, nn].max()) / args.temp)
            for j, wj in zip(nn, w):
                P[i, V[j]] += wj
        P /= P.sum(1, keepdims=True)
        return (1 - args.prior) * P + args.prior * prior[None, :]

    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    res = {"encoder": ENCODER, "k": args.k, "temp": args.temp, "window": args.window}
    for split in ("dev", "test"):
        ex = masc_examples(insts[split])
        P = vote(ex)
        y = np.array([POL2ID[e.polarity] for e in ex])
        acc = 100.0 * float((P.argmax(1) == y).mean())
        res[f"{split}_acc_goldspan"] = acc
        np.savez_compressed(out / f"probs_{split}.npz", probs=P,
                            keys=np.array([e.key for e in ex], dtype=np.int64))
        print(f"[{split}] gold-span MASC acc {acc:.2f} (n={len(ex)})", flush=True)
        if args.spans:
            sp = json.load(open(Path(args.spans) / f"spans_{split}.json"))
            e2 = masc_examples_for_spans(insts[split], [[tuple(x) for x in i] for i in sp])
            P2 = vote(e2)
            np.savez_compressed(out / f"probs_span_{split}.npz", probs=P2,
                                keys=np.array([e.key for e in e2], dtype=np.int64))
            print(f"[{split}] scored {len(e2)} predicted spans", flush=True)
    json.dump(res, open(out / "metrics.json", "w"), indent=2)
    print(json.dumps(res, indent=2))


if __name__ == "__main__":
    main()
