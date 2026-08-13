"""TARKAN training objective (updated paper §3.7, Eqs. 11, 16, 22).

    L_tag = - sum_i log p(b*_i | T, I)                                    (Eq. 22)
    L_rel = - sum_k [ r^T_k log r_k + (1-r^T_k) log(1-r_k) ]             (Eq. 11)
    L_kg  = - sum_k sum_q [ s^T_kq log s_kq + (1-s^T_kq) log(1-s_kq) ]   (Eq. 16)

    L = L_tag + λ1 L_rel + λ2 L_kg                                       (Eq. for L)

The updated methodology unifies aspect extraction + sentiment classification in the
single BIO head (Eq. 21, run on the KAN-fused token representation), so there is no
separate auxiliary span-ASC loss (the old L_asc / λ3 are removed). cfg.use_teacher =
False zeroes L_rel and L_kg ("w/o LLM teacher guidance").
"""
from __future__ import annotations

from typing import Dict, List, Optional

import torch
import torch.nn.functional as F

from config import CONFIG


def tag_loss(tag_logits: torch.Tensor, bio_labels: torch.Tensor, cfg=CONFIG) -> torch.Tensor:
    """Eq. 22. Token CE over 7 BIO classes; -100 positions ignored (subtoken continuations).

    Paper-faithful default is plain CE. Two reproduction-aid patches activate only when the
    corresponding cfg flags are set (both OFF by default):
      A1 tag_class_weight  -> inverse-frequency class weights (up-weight rare NEG/B-I vs O),
                              precomputed once over the train split (cfg._tag_weight_vec).
      A5 tag_label_smoothing -> label smoothing epsilon.
    """
    B, n, C = tag_logits.shape
    weight = getattr(cfg, "_tag_weight_vec", None) if getattr(cfg, "tag_class_weight", False) else None
    if weight is not None:
        weight = weight.to(tag_logits.device, tag_logits.dtype)
    ls = float(getattr(cfg, "tag_label_smoothing", 0.0) or 0.0)
    return F.cross_entropy(
        tag_logits.reshape(B * n, C), bio_labels.reshape(B * n), ignore_index=-100,
        reduction="mean", weight=weight, label_smoothing=ls,
    )


def word_level_emissions(tag_logits: torch.Tensor, word_ids, n_words, bio_labels: Optional[torch.Tensor] = None):
    """Gather word-level BIO emissions/labels from subtoken logits (first subtoken per word).

    Returns (emissions [B, W, C], labels [B, W] or None, mask [B, W]) where W = max(n_words).
    Word 0..n_words[b]-1 all have a first subtoken, so each row's mask is contiguous from 0
    (a torchcrf requirement). Rows are guaranteed >=1 word by construction (tweets non-empty).
    """
    B, _, C = tag_logits.shape
    W = max(1, max((int(w) for w in n_words), default=1))
    emis = tag_logits.new_zeros((B, W, C))
    labs = torch.zeros((B, W), dtype=torch.long, device=tag_logits.device) if bio_labels is not None else None
    mask = torch.zeros((B, W), dtype=torch.bool, device=tag_logits.device)
    for b in range(B):
        seen = set()
        nb = int(n_words[b])
        for i, wid in enumerate(word_ids[b]):
            if isinstance(wid, int) and 0 <= wid < nb and wid not in seen:
                seen.add(wid)
                emis[b, wid] = tag_logits[b, i]
                mask[b, wid] = True
                if labs is not None:
                    lab = int(bio_labels[b, i])
                    labs[b, wid] = lab if lab >= 0 else 0
    return emis, labs, mask


def relevance_loss(r: torch.Tensor, teacher_r: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
    """Eq. 11. BCE over aspects that have a teacher relevance label."""
    if r.numel() == 0:
        return r.new_zeros(())
    if mask is not None:
        if mask.sum() == 0:
            return r.new_zeros(())
        r, teacher_r = r[mask], teacher_r[mask]
    r = r.clamp(1e-7, 1 - 1e-7)
    return F.binary_cross_entropy(r, teacher_r.float(), reduction="mean")


def kg_loss(
    kg_scores: List[torch.Tensor],
    teacher_kg: List[torch.Tensor],
    teacher_kg_mask: Optional[List[torch.Tensor]] = None,
) -> torch.Tensor:
    """Eq. 16. BCE over retrieved triples that have a teacher usefulness label."""
    preds, tgts = [], []
    for i, s in enumerate(kg_scores):
        if s.numel() == 0:
            continue
        t = teacher_kg[i]
        if t.numel() != s.numel():
            continue
        if teacher_kg_mask is not None:
            m = teacher_kg_mask[i]
            if m.sum() == 0:
                continue
            s, t = s[m], t[m]
        preds.append(s)
        tgts.append(t.float())
    if not preds:
        # keep graph connected to wg params if everything is unlabeled
        device = kg_scores[0].device if kg_scores else torch.device("cpu")
        return torch.zeros((), device=device)
    p = torch.cat(preds).clamp(1e-7, 1 - 1e-7)
    return F.binary_cross_entropy(p, torch.cat(tgts), reduction="mean")


def compute_losses(outputs: Dict, targets: Dict, cfg=CONFIG, model=None) -> Dict[str, torch.Tensor]:
    """Returns dict with l_tag, l_rel, l_kg, total (updated §3.7: L = L_tag + λ1 L_rel + λ2 L_kg).

    A4 (opt-in): when cfg.use_crf and `model` (owning model.crf) is passed, L_tag becomes the
    CRF negative log-likelihood over word-level emissions instead of token CE.
    """
    if (
        getattr(cfg, "use_crf", False)
        and model is not None
        and getattr(model, "crf", None) is not None
        and targets.get("word_ids") is not None
        and targets.get("n_words") is not None
    ):
        emis, labs, mask = word_level_emissions(
            outputs["tag_logits"], targets["word_ids"], targets["n_words"], targets["bio_labels"]
        )
        l_tag = -model.crf(emis, labs, mask=mask, reduction="mean")
    else:
        l_tag = tag_loss(outputs["tag_logits"], targets["bio_labels"], cfg)

    if cfg.use_teacher and cfg.use_relevance and cfg.use_visual_stream and "teacher_relevance" in targets:
        l_rel = relevance_loss(
            outputs["relevance"], targets["teacher_relevance"], targets.get("teacher_relevance_mask")
        )
    else:
        l_rel = l_tag.new_zeros(())

    if cfg.use_teacher and cfg.use_kg_stream and cfg.use_kg_filter and "teacher_kg" in targets:
        l_kg = kg_loss(outputs["kg_scores"], targets["teacher_kg"], targets.get("teacher_kg_mask"))
    else:
        l_kg = l_tag.new_zeros(())

    total = l_tag + cfg.lambda1 * l_rel + cfg.lambda2 * l_kg
    out = {"l_tag": l_tag, "l_rel": l_rel, "l_kg": l_kg, "total": total}

    # A7 (opt-in): dedicated ASC polarity CE on the pooled aspect reps (gold spans in training).
    # Only added when explicitly enabled, so the faithful baseline's loss dict is unchanged.
    if getattr(cfg, "aux_asc_head", False) and outputs.get("asc_logits") is not None and "aspect_polarity" in targets:
        al = outputs["asc_logits"]
        tgt = targets["aspect_polarity"]
        l_asc = F.cross_entropy(al, tgt) if (al.numel() and al.shape[0] == tgt.shape[0]) else l_tag.new_zeros(())
        out["l_asc"] = l_asc
        out["total"] = total + getattr(cfg, "lambda_asc", 1.0) * l_asc
    return out
