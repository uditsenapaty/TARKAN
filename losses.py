"""Six-component objective, TARKAN.pdf Eq. 24.

Losses sum valid supervised terms, as written in the manuscript. Missing labels
and unmatched anchors are masked, never interpreted as negative evidence.
"""
from __future__ import annotations
import torch
from torch.nn import functional as F
from config import CONFIG


def _ce(logits, labels):
    valid = labels != -100
    if not valid.any():
        return logits.sum() * 0
    return F.cross_entropy(logits[valid].float(), labels[valid], reduction="sum")


def tag_loss(tag_logits, bio_labels, cfg=CONFIG):
    return _ce(tag_logits, bio_labels)


def anchor_loss(anchor_logits, anchor_labels):
    return _ce(anchor_logits, anchor_labels)


def relevance_loss(scores, targets, mask=None):
    if mask is None:
        mask = torch.ones_like(targets, dtype=torch.bool)
    if not mask.any():
        return scores.sum() * 0
    return F.binary_cross_entropy(scores[mask].float().clamp(1e-7, 1-1e-7),
                                  targets[mask].float(), reduction="sum")


def kg_loss(scores, targets, masks=None):
    if not scores:
        return torch.tensor(0.0)
    result = scores[0].sum() * 0
    for i, s in enumerate(scores):
        if i >= len(targets) or s.shape != targets[i].shape:
            raise ValueError("KG predictions and teacher labels must align.")
        result = result + relevance_loss(s, targets[i], masks[i] if masks is not None else None)
    return result


def aceq_loss(outputs, targets, cfg=CONFIG):
    zero = outputs["tag_logits"].sum() * 0
    visual, kg = zero, zero
    r = outputs["relevance"]
    if cfg.use_visual_stream and cfg.use_relevance and r.numel():
        rt = targets.get("teacher_relevance", torch.zeros_like(r))
        valid = targets.get("teacher_relevance_mask", torch.zeros_like(r, dtype=torch.bool))
        valid = valid & (rt == 1) & outputs["cf_relevance_mask"]
        if valid.any():
            visual = F.relu(cfg.aceq_mv - r[valid].float()
                            + outputs["cf_relevance"][valid].float()).sum()
    if cfg.use_kg_stream and cfg.use_kg_filter:
        for k, scores in enumerate(outputs["kg_scores"]):
            if k >= len(targets.get("teacher_kg", [])):
                continue
            labels = targets["teacher_kg"][k]
            valid = targets["teacher_kg_mask"][k]
            positive = valid & (labels == 1)
            negative = valid & (labels == 0)
            if not positive.any():
                continue
            if negative.any():
                neg = scores[negative].float().max()
            elif outputs["cf_kg_scores"][k].numel():
                neg = outputs["cf_kg_scores"][k].float().max()
            else:
                continue
            kg = kg + F.relu(cfg.aceq_mg - scores[positive].float() + neg).sum()
    return visual + cfg.aceq_eta * kg


def compute_losses(outputs, targets, cfg=CONFIG, model=None):
    l_tag = tag_loss(outputs["tag_logits"], targets["bio_labels"])
    zero = l_tag * 0
    l_anc = anchor_loss(outputs["anchor_logits"], targets["anchor_labels"])
    l_asc = _ce(outputs["asc_paper_logits"], targets["aspect_polarity"])
    l_rel = l_kg = l_cf = zero
    if cfg.use_teacher:
        if cfg.use_visual_stream and cfg.use_relevance and "teacher_relevance" in targets:
            l_rel = relevance_loss(outputs["relevance"], targets["teacher_relevance"],
                                   targets["teacher_relevance_mask"])
        if cfg.use_kg_stream and cfg.use_kg_filter and "teacher_kg" in targets and outputs["kg_scores"]:
            l_kg = kg_loss(outputs["kg_scores"], targets["teacher_kg"], targets["teacher_kg_mask"])
        if cfg.lambda_cf > 0:
            l_cf = aceq_loss(outputs, targets, cfg)
    total = (l_tag + cfg.lambda_anc*l_anc + cfg.lambda_asc*l_asc
             + cfg.lambda1*l_rel + cfg.lambda2*l_kg + cfg.lambda_cf*l_cf)
    return dict(l_tag=l_tag, l_anc=l_anc, l_asc=l_asc, l_rel=l_rel,
                l_kg=l_kg, l_cf=l_cf, total=total)

