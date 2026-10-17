"""Candidate decoding and one-to-one IoU matching (paper Section 3.3)."""
from config import ANCHOR_TAGS


def decode_anchor_spans(logits, word_ids, n_words):
    tags = ["O"] * n_words
    seen = set()
    for i, wid in enumerate(word_ids):
        if 0 <= wid < n_words and wid not in seen:
            seen.add(wid)
            tags[wid] = ANCHOR_TAGS[logits[i]]
    spans, start = [], None
    for i, tag in enumerate(tags + ["O"]):
        if tag == "B-ASP" or tag == "O":
            if start is not None:
                spans.append((start, i))
                start = None
        if tag == "B-ASP" or (tag == "I-ASP" and start is None):
            start = i
    return spans


def match_anchors(predicted, gold, threshold=0.5):
    """Greedy descending token IoU; each anchor and gold span is used once."""
    eligible = []
    for p, (ps, pe, *_) in enumerate(predicted):
        for g, (gs, ge, *_) in enumerate(gold):
            intersection = max(0, min(pe, ge) - max(ps, gs))
            union = (pe - ps) + (ge - gs) - intersection
            iou = intersection / union if union else 0.0
            if intersection and iou >= threshold:
                eligible.append((-iou, p, g))
    matches, used = [-1] * len(predicted), set()
    for _, p, g in sorted(eligible):
        if matches[p] == -1 and g not in used:
            matches[p] = g
            used.add(g)
    return matches


def subtoken_span(word_ids, span):
    indices = [i for i, wid in enumerate(word_ids) if span[0] <= wid < span[1]]
    if not indices:
        raise ValueError(f"Anchor {span} has no visible subtokens")
    return indices[0], indices[-1] + 1
