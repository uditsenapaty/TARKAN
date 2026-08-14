"""External aspect-level sentiment supervision — the one input class never tried here.

**Why this and not another mechanism.** §D.27's law says every *mechanism* added to a tower
above 77.9 arrives with the sign flipped, and §D.30–D.32 closed the evidence route from both
ends. §B.8's own conclusion names the only remaining escape: *"Only NEW INFORMATION or a
mechanism-decorrelated member can move it."* Every MASC tower in this project is trained on
**3179 aspects** — t2015's train split and nothing else. §D.1 measured polarity losing twice
what extraction loses. So the untried lever is not another loss; it is more supervision for
the task that is losing.

Four public aspect-sentiment corpora, same 3-class label set, same "aspect marked in the
sentence" input format:

| corpus | domain | train aspects |
|---|---|---|
| Dong et al. 2014 (`acl-14-short-data`) | **Twitter** | 6248 |
| SemEval-2014 Task 4 Restaurants | reviews | 3608 |
| SemEval-2014 Task 4 Laptops | reviews | 2328 |
| MAMS-ATSA | reviews, **multi-aspect multi-polarity** | 11186 |
| **total** | | **23370 = 7.35× t2015** |

The Twitter one matters most for domain: same @-mention/hashtag noise, same
entity-as-aspect convention. MAMS matters most for the failure mode (see SOURCES).
Class balance is the other half of the argument — t2015 train has **368 NEG aspects**;
these carry **6001**, and NEG recall is this campaign's worst number.

This is intermediate-task training, not benchmark adaptation: t2015 dev and test are never
touched, no t2017 is involved, and `overlap_report()` is run as a gate before any training
so a leak cannot be discovered after the fact.

Source files come from `songyouwei/ABSA-PyTorch`, which ships both corpora in one 3-line
format (text with `$T$`, aspect term, polarity in {-1,0,1}).

    python experts/absa_extra.py --check          # overlap gate + census
"""
from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path
from typing import Dict, List, Sequence

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from experts.common import DATA, MascExample, load  # noqa: E402

ROOT = DATA / "absa_extra"
POL_OF = {"-1": "NEG", "0": "NEU", "1": "POS"}

SOURCES = {
    "twitter": "acl14_train.raw",        # Dong et al. 2014 — Twitter
    "rest":    "semeval14_rest_train.seg",
    "laptop":  "semeval14_laptop_train.seg",
    # MAMS-ATSA is built so that EVERY sentence carries at least two aspects with
    # DIFFERENT sentiment — which is precisely the failure mode this project measured:
    # t2015 has 444 within-tweet pairs whose gold polarities differ, and the recorded
    # failure is the model falling back on the tweet's overall tone (86 POS aspects
    # predicted NEU). Chapter B attacked that with SupCon and C10/C13 at the input and
    # logit level; this supplies 11186 aspects where the shortcut is guaranteed wrong.
    "mams":    "mams_atsa_train.seg",
}


def _read_seg(path: Path):
    """The shared 3-line format: text with `$T$`, the aspect term, the polarity."""
    lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
    out = []
    for i in range(0, len(lines) - 2, 3):
        text, term, pol = lines[i].strip(), lines[i + 1].strip(), lines[i + 2].strip()
        if "$T$" not in text or pol not in POL_OF:
            continue
        out.append((text, term, POL_OF[pol]))
    return out


def _to_examples(rows, inst_base: int) -> List[MascExample]:
    """One MascExample per aspect, with siblings recovered by grouping on the FILLED text.

    Grouping matters: `marked_text(mark_siblings=True)` and `InstanceBatchSampler` both
    assume aspects of one sentence share an `inst_idx`. These corpora store one row per
    aspect with no sentence id, so the filled sentence is the only available key.
    """
    by_sent: Dict[str, list] = {}
    for text, term, pol in rows:
        filled = text.replace("$T$", term)
        toks = filled.split()
        # locate the aspect by re-splitting the two halves: the left context's token count
        # is the start index, so tokenisation stays identical to the rest of the pipeline
        left = text.split("$T$")[0].split()
        s = len(left)
        e = s + len(term.split())
        if toks[s:e] != term.split():          # placeholder inside a token, e.g. "$T$'s"
            continue
        by_sent.setdefault(filled, []).append((toks, (s, e), pol))
    out = []
    for k, (filled, items) in enumerate(by_sent.items()):
        sib = [sp for (_, sp, _) in items]
        for (toks, sp, pol) in items:
            out.append(MascExample(inst_base + k, toks, sp, pol, "", sib))
    return out


def load_extra_insts(names: Sequence[str]) -> List["Instance"]:
    """The same corpora as sentence-level `Instance`s, for the EXTRACTION side.

    Grouping by sentence recovers every annotated aspect term, so `obi_tags` produces a
    proper BIO sequence. The annotation is complete for SemEval-14 (all aspect terms are
    labelled) and complete-per-sentence for Dong-2014 once its per-target rows are
    regrouped, which is what `_to_examples` already does.

    Worth stating plainly: `joint = MATE@tau x a`, and at 87.80 x 80.44 the bar needs
    +2.83 on one factor or **+1.4 on each**. The polarity-only version of this lever
    cannot reach the bar alone, so the extraction side gets it too.
    """
    from data import Instance
    ex = load_extra(names)
    by_inst: Dict[int, list] = {}
    toks: Dict[int, list] = {}
    for e in ex:
        by_inst.setdefault(e.inst_idx, []).append((e.span[0], e.span[1], e.polarity))
        toks[e.inst_idx] = e.tokens
    out = []
    for k in sorted(by_inst):
        asp = sorted(set(by_inst[k]))
        out.append(Instance(id=f"extra{k}", tokens=toks[k], image_id="", aspects=asp))
    return out


def load_extra(names: Sequence[str]) -> List[MascExample]:
    out, base = [], 1_000_000            # far above any t2015 inst_idx; keys stay disjoint
    for n in names:
        p = ROOT / SOURCES[n]
        if not p.exists():
            raise FileNotFoundError(f"{p} missing — see the module docstring for the source")
        ex = _to_examples(_read_seg(p), base)
        base += 200_000
        print(f"  {n:8s} {len(ex):5d} aspects", flush=True)
        out += ex
    return out


# --------------------------------------------------------------------------- #
# encoder transfer — one intermediate run, reused by every member of that backbone
# --------------------------------------------------------------------------- #
# The PDQ and PDS members read image features (VitCache, aspect_*.npz) that the external
# corpora do not have, so they cannot run stage 1 themselves. What transfers is the TEXT
# ENCODER. Saving it once per backbone and loading it into every member of that family
# also removes the redundant stage-1 cost: 4 backbones instead of 19 members.
def save_encoder(model, path: Path, attr: str = "enc"):
    sd = getattr(model, attr).state_dict()
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(sd, path)
    print(f"  saved intermediate encoder ({len(sd)} tensors) -> {path}", flush=True)


def load_encoder(model, path: Path, attr: str = "enc"):
    """Strict load. A silent partial load here would look exactly like 'the lever does
    nothing', which is the failure mode this chapter has hit five times."""
    sd = torch.load(path, map_location="cpu")
    missing, unexpected = getattr(model, attr).load_state_dict(sd, strict=False)
    if missing or unexpected:
        raise SystemExit(f"encoder mismatch loading {path}: "
                         f"{len(missing)} missing, {len(unexpected)} unexpected "
                         f"(first: {(missing or unexpected)[:3]})")
    print(f"  loaded intermediate encoder ({len(sd)} tensors) <- {path}", flush=True)


# --------------------------------------------------------------------------- #
# the leak gate
# --------------------------------------------------------------------------- #
def _norm(s: str) -> str:
    return re.sub(r"[^a-z0-9]+", " ", s.lower()).strip()


def _shingles(s: str, n: int = 5):
    w = _norm(s).split()
    return {" ".join(w[i:i + n]) for i in range(max(len(w) - n + 1, 1))}


def overlap_report(names: Sequence[str], dataset: str = "twitter2015"):
    """Hard gate. Exact match on normalised text, plus 5-gram Jaccard for near-duplicates.

    Run before training, not after: a leak found afterwards invalidates the number, and
    §D.18's whole point was that a result you cannot independently validate is worthless.
    """
    ex = load_extra(names)
    ext = {}
    for e in ex:
        ext.setdefault(_norm(" ".join(e.tokens)), []).append(e)
    print(f"  {len(ext)} distinct external sentences", flush=True)

    worst = 0.0
    for split in ("train", "dev", "test"):
        insts = load(dataset, split)
        n_exact = sum(1 for i in insts if _norm(" ".join(i.tokens)) in ext)
        # near-dup: only sentences sharing a rare 5-gram are compared, so this stays linear
        index: Dict[str, list] = {}
        for k in ext:
            for g in _shingles(k):
                index.setdefault(g, []).append(k)
        n_near, ex_near = 0, []
        for i in insts:
            t = _norm(" ".join(i.tokens))
            cand = {c for g in _shingles(t) for c in index.get(g, ())}
            for c in cand:
                a, b = _shingles(t), _shingles(c)
                j = len(a & b) / max(len(a | b), 1)
                if j >= 0.5:
                    n_near += 1
                    worst = max(worst, j)
                    if len(ex_near) < 3:
                        ex_near.append((round(j, 3), t[:70], c[:70]))
                    break
        flag = "LEAK" if (n_exact or n_near) else "clean"
        print(f"  t2015 {split:5s}: {len(insts):5d} sentences | exact {n_exact} | "
              f"near-dup(J>=0.5) {n_near}   -> {flag}", flush=True)
        for row in ex_near:
            print(f"        {row}", flush=True)
    print(f"  max Jaccard seen: {worst:.3f}", flush=True)
    return worst


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sources", default="twitter,rest,laptop")
    ap.add_argument("--check", action="store_true")
    args = ap.parse_args()
    names = args.sources.split(",")
    ex = load_extra(names)
    import collections
    c = collections.Counter(e.polarity for e in ex)
    n = sum(c.values())
    print(f"total {n} aspects  " + "  ".join(f"{k} {c[k]} ({100*c[k]/n:.1f}%)"
                                             for k in ("NEG", "NEU", "POS")))
    tr = load("twitter2015", "train")
    ct = collections.Counter(p for i in tr for (_, _, p) in i.aspects)
    nt = sum(ct.values())
    print(f"t2015 {nt} aspects  " + "  ".join(f"{k} {ct[k]} ({100*ct[k]/nt:.1f}%)"
                                              for k in ("NEG", "NEU", "POS")))
    print(f"external / t2015 supervision ratio: {n/nt:.2f}x")
    print("  example:", ex[0].marked_text(), "->", ex[0].polarity)
    if args.check:
        overlap_report(names)


if __name__ == "__main__":
    main()
