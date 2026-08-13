"""Shared canonical data layer for the two-stage experts (Chapter C rebuild).

Rebuilt from the Chapter-B registry in possible-patches.md (`experts/common.py` was lost).
Everything here is deterministic, import-light and CPU-safe.

Conventions (must match `metrics.py` and the AoM dumps — verified: our reconstruction is
bijective with AoM's canonical split, 674 sents / 1037 aspects on t2015 test):
  * a span is (word_start, word_end_EXCLUSIVE)
  * a pair  is (word_start, word_end_EXCLUSIVE, polarity in {POS,NEU,NEG})
  * anchor tags are the paper's aspect-only B-ASP/I-ASP/O, encoded here as O/B/I.
"""
from __future__ import annotations

import random
import sys
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from data import Instance, load_split  # noqa: E402
from metrics import joint_prf, masc_acc_f1, mate_prf  # noqa: E402

Span = Tuple[int, int]
Pair = Tuple[int, int, str]

OBI = ["O", "B", "I"]
OBI2ID = {t: i for i, t in enumerate(OBI)}
POLARITIES = ["NEG", "NEU", "POS"]
POL2ID = {p: i for i, p in enumerate(POLARITIES)}

DATA = ROOT / "data"


# --------------------------------------------------------------------------- #
# loading
# --------------------------------------------------------------------------- #
def load(dataset: str = "twitter2015", split: str = "test") -> List[Instance]:
    return load_split(DATA / dataset, split)


def image_path(dataset: str, image_id: str) -> Path:
    return DATA / "images" / dataset / image_id


# --------------------------------------------------------------------------- #
# anchor (aspect-only) tags  <->  spans
# --------------------------------------------------------------------------- #
def obi_tags(inst: Instance) -> List[int]:
    """Aspect-only B/I/O anchor labels (paper: B-ASP / I-ASP / O)."""
    tags = [0] * len(inst.tokens)
    for s, e, _ in inst.aspects:
        if e <= s:
            continue
        tags[s] = OBI2ID["B"]
        for j in range(s + 1, min(e, len(tags))):
            tags[j] = OBI2ID["I"]
    return tags


def spans_from_obi(tags: Sequence[int]) -> List[Span]:
    """Decode an O/B/I id sequence into (start, end_exclusive) spans.

    A stray `I` with no preceding `B` opens a span (recall-preserving, and the CRF
    almost never emits one because the O->I transition is learned as improbable).
    """
    spans: List[Span] = []
    start = None
    for i, t in enumerate(tags):
        if t == OBI2ID["B"]:
            if start is not None:
                spans.append((start, i))
            start = i
        elif t == OBI2ID["I"]:
            if start is None:
                start = i
        else:  # O
            if start is not None:
                spans.append((start, i))
                start = None
    if start is not None:
        spans.append((start, len(tags)))
    return spans


# --------------------------------------------------------------------------- #
# scoring
# --------------------------------------------------------------------------- #
def score_spans(pred: Sequence[Sequence[Span]], gold: Sequence[Sequence[Span]]) -> Dict[str, float]:
    """MATE micro P/R/F1 over spans (polarity ignored)."""
    p = [[(s, e, "X") for (s, e) in inst] for inst in pred]
    g = [[(s, e, "X") for (s, e) in inst] for inst in gold]
    return mate_prf(p, g)


def gold_spans(insts: Sequence[Instance]) -> List[List[Span]]:
    return [[(s, e) for (s, e, _) in i.aspects] for i in insts]


def gold_pairs(insts: Sequence[Instance]) -> List[List[Pair]]:
    return [list(i.aspects) for i in insts]


def score_joint(pred: Sequence[Sequence[Pair]], gold: Sequence[Sequence[Pair]]) -> Dict[str, float]:
    return joint_prf(pred, gold)


def polarity_on_extracted(pred_pairs: Sequence[Sequence[Pair]],
                          gold: Sequence[Sequence[Pair]]) -> float:
    """`a` = polarity accuracy restricted to correctly-extracted spans.

    This is the quantity in the identity `joint = MATE_F1 x a` (Chapter B, §7f).
    """
    correct = total = 0
    for p, g in zip(pred_pairs, gold):
        gmap = {(s, e): pol for (s, e, pol) in g}
        for (s, e, pol) in p:
            if (s, e) in gmap:
                total += 1
                correct += int(gmap[(s, e)] == pol)
    return 100.0 * correct / total if total else 0.0


# --------------------------------------------------------------------------- #
# MASC examples  (one per aspect span)
# --------------------------------------------------------------------------- #
class MascExample:
    __slots__ = ("inst_idx", "tokens", "span", "polarity", "image_id", "key", "siblings")

    def __init__(self, inst_idx: int, tokens: List[str], span: Span,
                 polarity: str, image_id: str, siblings: Sequence[Span] = ()):
        self.inst_idx = inst_idx
        self.tokens = tokens
        self.span = span
        self.polarity = polarity
        self.image_id = image_id
        # every other candidate aspect in the SAME tweet (see marked_text)
        self.siblings = [s for s in siblings if tuple(s) != tuple(span)]
        self.key = (inst_idx, span[0], span[1])

    @property
    def term(self) -> str:
        return " ".join(self.tokens[self.span[0]:self.span[1]])

    def marked_text(self, left: str = "[", right: str = "]",
                    mark_siblings: bool = False) -> str:
        """Aspect marked IN PLACE (Chapter B, B2: the only way to disambiguate a tweet
        where the same surface form occurs twice with different gold polarity).

        `mark_siblings` additionally brackets the OTHER candidate aspects of the same
        tweet with < >. t2015 averages 1.51 aspects/sentence and contains 444 within-tweet
        pairs whose gold polarities DIFFER, yet each aspect is otherwise scored with no
        knowledge that the others exist -- which is consistent with the measured failure
        mode (86 POS aspects predicted NEU, i.e. the model falling back on the tweet's
        overall tone instead of the target). Chapter B attacked this at the loss level with
        SupCon and got weak members; this is the input-level version.
        """
        w = list(self.tokens)
        marks = [(self.span[0], self.span[1], left, right)]
        if mark_siblings:
            marks += [(a, b, "<", ">") for (a, b) in self.siblings]
        for (a, b, lo, ro) in sorted(marks, key=lambda m: -m[0]):
            w = w[:a] + [lo] + w[a:b] + [ro] + w[b:]
        return " ".join(w)


def masc_examples(insts: Sequence[Instance]) -> List[MascExample]:
    """Gold-span MASC examples (training / gold-span evaluation)."""
    out = []
    for i, inst in enumerate(insts):
        sib = [(a, b) for (a, b, _) in inst.aspects]
        for (s, e, pol) in inst.aspects:
            out.append(MascExample(i, inst.tokens, (s, e), pol, inst.image_id, sib))
    return out


def masc_examples_for_spans(insts: Sequence[Instance],
                            spans: Sequence[Sequence[Span]]) -> List[MascExample]:
    """MASC examples on *predicted* spans (stage-2 of the two-stage assembly).

    Polarity is the gold one when the predicted span matches a gold span, else 'NEU'
    as a placeholder — it is never used as a training target here, only for bookkeeping.
    """
    out = []
    for i, (inst, sp) in enumerate(zip(insts, spans)):
        gmap = {(a, b): p for (a, b, p) in inst.aspects}
        for (s, e) in sp:
            out.append(MascExample(i, inst.tokens, (s, e), gmap.get((s, e), "NEU"),
                                   inst.image_id, list(sp)))
    return out


# --------------------------------------------------------------------------- #
# misc
# --------------------------------------------------------------------------- #
def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    except ImportError:
        pass
