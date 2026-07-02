"""Neurosymbolic inference-time layer (A12/A13/A14 — DISOBEYING, opt-in, zero retraining).

A12 `ns_bio_rules`   — hard BIO-transition logic in the CRF Viterbi: I-X only after B-X/I-X,
                       no I-* sequence starts. Guarantees structurally valid tag sequences.
A13 `ns_lexicon_alpha` — probabilistic-logic polarity prior (product of experts):
                       final_logits = asc_logits + alpha * log(prior). The prior comes from
                       SenticNet polarity_value of words within +/-window of the aspect span,
                       flipped by nearby negators. alpha=0 disables (strictly safe knob).
A14 `ns_aspect_consistency` — identical aspect strings within one tweet take the majority
                       polarity across their predicted mentions.

All knobs live on the config and act ONLY at inference, so they are tunable on dev against
frozen checkpoints (minutes per configuration, no GPU training).
"""
from __future__ import annotations

import math
from contextlib import contextmanager
from typing import Dict, List, Optional, Sequence, Tuple

from config import CONFIG, BIO_TAGS, POLARITIES

NEGATORS = {"not", "no", "never", "n't", "nt", "hardly", "barely", "cannot", "cant", "won't",
            "dont", "don't", "didnt", "didn't", "isnt", "isn't", "wasnt", "wasn't", "aint", "ain't"}
_NEG_SCOPE = 3  # a negator flips sentiment words up to this many words after it


# --------------------------------------------------------------------------- #
# A12: BIO transition logic for CRF Viterbi
# --------------------------------------------------------------------------- #
@contextmanager
def constrained_crf(crf, enabled: bool = True):
    """Temporarily clamp illegal BIO transitions to -1e4 in a torchcrf.CRF (decode-time)."""
    if not enabled or crf is None:
        yield
        return
    import torch

    NEG = -1e4
    i_tags = [i for i, t in enumerate(BIO_TAGS) if t.startswith("I-")]
    with torch.no_grad():
        st = crf.start_transitions.clone()
        tr = crf.transitions.clone()
        for i in i_tags:
            crf.start_transitions[i] = NEG          # sequences cannot start inside a span
            for prev in range(len(BIO_TAGS)):
                if prev not in (i - 1, i):          # I-X only after B-X (=i-1) or I-X (=i)
                    crf.transitions[prev, i] = NEG
    try:
        yield
    finally:
        with torch.no_grad():
            crf.start_transitions.copy_(st)
            crf.transitions.copy_(tr)


# --------------------------------------------------------------------------- #
# A13: SenticNet polarity prior (product of experts)
# --------------------------------------------------------------------------- #
class LexiconPrior:
    _shared = None

    def __init__(self, parquet_path=None):
        import pandas as pd

        path = parquet_path or (CONFIG.paths.senticnet / "senticnet_en.parquet")
        df = pd.read_parquet(path, columns=["concept", "polarity_value"])
        self.pol = dict(zip(df["concept"].astype(str), df["polarity_value"].astype(float)))

    @classmethod
    def shared(cls) -> "LexiconPrior":
        if cls._shared is None:
            cls._shared = cls()
        return cls._shared

    def _word_pol(self, w: str) -> Optional[float]:
        w = w.lower().strip("#@.,!?:;'\"()[]")
        if not w:
            return None
        return self.pol.get(w) or self.pol.get(w.replace("-", "_"))

    def aspect_prior(self, tokens: Sequence[str], span: Tuple[int, int], window: int = 5) -> List[float]:
        """log-prior over [POS, NEU, NEG] from lexicon polarity around the aspect span."""
        s, e = span
        lo, hi = max(0, s - window), min(len(tokens), e + window)
        vals = []
        i = lo
        toks = [t.lower() for t in tokens]
        while i < hi:
            if s <= i < e:  # skip the aspect words themselves
                i += 1
                continue
            p = self._word_pol(tokens[i])
            if p is not None and abs(p) > 0.1:
                neg = any(toks[j] in NEGATORS for j in range(max(lo, i - _NEG_SCOPE), i))
                vals.append(-p if neg else p)
            i += 1
        m = sum(vals) / len(vals) if vals else 0.0
        tau = max(1e-3, float(getattr(CONFIG, "ns_lexicon_tau", 0.5)))
        logits = [m / tau, 0.0, -m / tau]  # POS, NEU, NEG
        z = math.log(sum(math.exp(x) for x in logits))
        return [x - z for x in logits]  # log-probs


def blend_asc_logits(asc_logits, tokens_per_aspect: List[Optional[Tuple[Sequence[str], Tuple[int, int]]]],
                     alpha: float, window: int = 5):
    """asc_logits [K,3] + alpha * lexicon log-prior per aspect (None entries left unchanged)."""
    if alpha <= 0 or asc_logits is None or asc_logits.numel() == 0:
        return asc_logits
    import torch

    prior = LexiconPrior.shared()
    out = asc_logits.clone()
    for k, info in enumerate(tokens_per_aspect):
        if info is None or k >= out.size(0):
            continue
        tokens, span = info
        lp = prior.aspect_prior(tokens, span, window)
        out[k] = out[k] + alpha * torch.tensor(lp, device=out.device, dtype=out.dtype)
    return out


# --------------------------------------------------------------------------- #
# A14: aspect-consistency (majority polarity for identical aspect strings per tweet)
# --------------------------------------------------------------------------- #
def enforce_aspect_consistency(pred_spans: List[Tuple[int, int, str]], tokens: Sequence[str]):
    if len(pred_spans) < 2:
        return pred_spans
    by_text: Dict[str, List[int]] = {}
    for i, (s, e, _) in enumerate(pred_spans):
        key = " ".join(t.lower() for t in tokens[s:e])
        by_text.setdefault(key, []).append(i)
    out = list(pred_spans)
    for key, idxs in by_text.items():
        if len(idxs) < 2:
            continue
        pols = [pred_spans[i][2] for i in idxs]
        maj = max(set(pols), key=pols.count)
        for i in idxs:
            s, e, _ = out[i]
            out[i] = (s, e, maj)
    return out
